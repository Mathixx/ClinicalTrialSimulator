"""
1- and 2-compartment pharmacokinetic models (pure functions).
Uses real biological conventions: Vd (L), ke (1/h), CL (L/h), dose (mg), concentration (mg/L).
References: Standard PK texts (e.g. Shargel, Applied Biopharmaceutics).
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp


def _parse_frequency_hours(frequency: str | dict) -> float | None:
    """Parse frequency string like 'every 24 hours' or dict {value: 24, unit: 'hour'} -> interval in hours."""
    if isinstance(frequency, dict):
        v = float(frequency.get("value", 24))
        u = (frequency.get("unit") or "hour").lower()
        if u in ("h", "hour", "hours"):
            return v
        if u in ("min", "minute", "minutes"):
            return v / 60.0
        return v
    if isinstance(frequency, str):
        # "every 24 hours", "every 12 h"
        m = re.search(r"every\s+([\d.]+)\s*(h|hour|hours|min|minute|minutes)?", frequency.lower())
        if m:
            v = float(m.group(1))
            u = (m.group(2) or "hour").lower()
            if u.startswith("min"):
                return v / 60.0
            return v
    return None


def solve_one_compartment(
    dose_mg: float,
    Vd: float,
    ke: float,
    t_end_hours: float,
    interval_hours: float | None = None,
    num_doses: int | None = None,
    ka: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-compartment model: C(t) = (Dose/Vd) * exp(-ke*t) for IV bolus.
    If interval_hours and num_doses are set, uses superposition of doses at t=0, interval, 2*interval, ...
    If ka is set, oral absorption is used: C(t) = (Dose*ka/(Vd*(ka-ke)))*(exp(-ke*t)-exp(-ka*t)) for single dose.
    Returns (t_hours, concentration_mg_L).
    """
    t = np.linspace(0, t_end_hours, max(2, int(t_end_hours * 4)))  # ~4 points per hour

    if ka is not None and ka != ke:
        # Oral single-dose: C = (Dose*ka/(Vd*(ka-ke)))*(exp(-ke*t)-exp(-ka*t))
        C = (dose_mg * ka / (Vd * (ka - ke))) * (np.exp(-ke * t) - np.exp(-ka * t))
        C = np.maximum(C, 0.0)
    elif interval_hours is not None and num_doses is not None and num_doses > 1:
        # IV bolus, repeated doses by superposition
        C = np.zeros_like(t)
        for n in range(num_doses):
            t_dose = n * interval_hours
            mask = t >= t_dose
            C[mask] += (dose_mg / Vd) * np.exp(-ke * (t[mask] - t_dose))
    else:
        # Single IV bolus
        C = (dose_mg / Vd) * np.exp(-ke * t)
        C = np.maximum(C, 0.0)

    return t, C


def solve_two_compartment(
    dose_mg: float,
    Vc: float,
    Vp: float,
    CL: float,
    Q: float,
    t_end_hours: float,
    interval_hours: float | None = None,
    num_doses: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-compartment model (central concentration returned).
    dCentral/dt = -(CL/Vc)*Central - (Q/Vc)*Central + (Q/Vp)*Peripheral
    dPeripheral/dt = (Q/Vc)*Central - (Q/Vp)*Peripheral
    IV bolus: initial Central = Dose/Vc, Peripheral = 0.
    Repeated doses: simulated by running segments and re-injecting at each dose time.
    Returns (t_hours, concentration_mg_L) of central compartment.
    """
    def ode(_t: float, y: np.ndarray) -> np.ndarray:
        central, peripheral = y[0], y[1]
        d_central = -(CL / Vc) * central - (Q / Vc) * central + (Q / Vp) * peripheral
        d_peripheral = (Q / Vc) * central - (Q / Vp) * peripheral
        return np.array([d_central, d_peripheral])

    if interval_hours is not None and num_doses is not None and num_doses > 1:
        # Multiple doses: run segments
        t_list: list[np.ndarray] = []
        c_list: list[np.ndarray] = []
        central, peripheral = dose_mg / Vc, 0.0
        t_cur = 0.0
        for _ in range(num_doses):
            t_span = (t_cur, min(t_cur + interval_hours, t_end_hours))
            if t_span[1] <= t_span[0]:
                break
            sol = solve_ivp(
                ode,
                t_span,
                np.array([central, peripheral]),
                t_eval=np.linspace(t_span[0], t_span[1], max(2, int((t_span[1] - t_span[0]) * 4))),
                method="LSODA",
            )
            t_list.append(sol.t)
            c_list.append(sol.y[0])
            central, peripheral = sol.y[0][-1], sol.y[1][-1]
            t_cur += interval_hours
            if t_cur >= t_end_hours:
                break
            # Next dose: add to central
            central += dose_mg / Vc
        t_out = np.concatenate(t_list)
        c_out = np.concatenate(c_list)
        return t_out, np.maximum(c_out, 0.0)

    # Single IV bolus
    sol = solve_ivp(
        ode,
        (0, t_end_hours),
        np.array([dose_mg / Vc, 0.0]),
        t_eval=np.linspace(0, t_end_hours, max(2, int(t_end_hours * 4))),
        method="LSODA",
    )
    return sol.t, np.maximum(sol.y[0], 0.0)


def run_simulation(
    dose_mg: float,
    frequency: str | dict,
    patient_params: dict,
    t_end_hours: float = 168.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Dispatch to 1- or 2-compartment model based on patient_params.
    patient_params must include model_type: '1_compartment' | '2_compartment',
    and the corresponding parameters (Vd, ke for 1-comp; Vc, Vp, CL, Q for 2-comp).
    Optional: ka for oral 1-compartment.
    Returns (t_hours, concentration_mg_L, reasoning_trace).
    """
    model_type = (patient_params.get("model_type") or "1_compartment").strip().lower()
    interval_hours = _parse_frequency_hours(frequency)
    num_doses = None
    if interval_hours is not None and interval_hours > 0 and t_end_hours > 0:
        num_doses = max(1, int(t_end_hours / interval_hours))

    if model_type == "2_compartment":
        Vc = float(patient_params["Vc"])
        Vp = float(patient_params["Vp"])
        CL = float(patient_params["CL"])
        Q = float(patient_params.get("Q", 0.5 * CL))
        t, C = solve_two_compartment(
            dose_mg, Vc, Vp, CL, Q, t_end_hours,
            interval_hours=interval_hours, num_doses=num_doses,
        )
        trace = (
            f"Used 2-compartment model: Vc={Vc} L, Vp={Vp} L, CL={CL} L/h, Q={Q} L/h. "
            f"Dose={dose_mg} mg, frequency interval={interval_hours}h, num_doses={num_doses}."
        )
        return t, C, trace

    # 1-compartment
    Vd = float(patient_params["Vd"])
    ke = float(patient_params["ke"])
    ka = patient_params.get("ka")
    if ka is not None:
        ka = float(ka)
    t, C = solve_one_compartment(
        dose_mg, Vd, ke, t_end_hours,
        interval_hours=interval_hours, num_doses=num_doses, ka=ka,
    )
    trace = (
        f"Used 1-compartment model: Vd={Vd} L, ke={ke} 1/h. "
        f"Dose={dose_mg} mg, frequency interval={interval_hours}h, num_doses={num_doses}."
    )
    if ka is not None:
        trace += f" Oral absorption ka={ka} 1/h."
    return t, C, trace
