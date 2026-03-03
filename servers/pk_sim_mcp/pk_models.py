"""
1- and 2-compartment pharmacokinetic models (pure functions).
Uses real biological conventions: Vd (L), ke (1/h), CL (L/h), dose (mg), concentration (mg/L).
References: Standard PK texts (e.g. Shargel, Applied Biopharmaceutics).
"""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
from scipy.integrate import solve_ivp

from clinical_trial_simulator.phenotype_params import (
    build_cohort,
    parse_cohort_distribution,
)


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


def noncompartmental_analysis(
    t_hours: list[float] | np.ndarray,
    concentrations_mg_L: list[float] | np.ndarray,
) -> dict[str, Any]:
    """
    Simple non-compartmental analysis (NCA) from time–concentration data.

    Returns Cmax, Tmax, AUC (trapezoidal rule), and an estimated terminal half-life
    using a log-linear fit to the last few non-zero points (if available).
    """
    t = np.asarray(t_hours, dtype=float)
    c = np.asarray(concentrations_mg_L, dtype=float)
    if t.ndim != 1 or c.ndim != 1 or t.size != c.size or t.size == 0:
        raise ValueError("t_hours and concentrations must be 1D arrays of the same non-zero length.")

    warning: str | None = None
    if t.size < 3:
        warning = (
            "Only %d time point(s) provided. Use a full concentration-time profile "
            "(e.g. representative_t_hours and representative_concentration_mg_L from simulate_pop_pk, "
            "or t_hours/concentration_mg_L from simulate_pk) for meaningful AUC and half-life."
        ) % int(t.size)

    # Sort by time in case input is unsorted
    order = np.argsort(t)
    t = t[order]
    c = np.maximum(c[order], 0.0)

    cmax = float(np.max(c))
    tmax = float(t[int(np.argmax(c))])
    auc = float(np.trapezoid(c, t))

    # Estimate terminal half-life from last 3–5 positive points
    positive_mask = c > 0
    t_pos = t[positive_mask]
    c_pos = c[positive_mask]
    half_life_h: float | None = None
    if t_pos.size >= 3:
        # Use last min(5, n) points
        n_tail = min(5, t_pos.size)
        t_tail = t_pos[-n_tail:]
        c_tail = c_pos[-n_tail:]
        # Guard against non-positive after filtering
        if np.all(c_tail > 0):
            ln_c = np.log(c_tail)
            A = np.vstack([t_tail, np.ones_like(t_tail)]).T
            slope, _intercept = np.linalg.lstsq(A, ln_c, rcond=None)[0]
            if slope < 0:
                ke = -slope
                half_life_h = float(np.log(2.0) / ke)

    out: dict[str, Any] = {
        "Cmax_mg_L": cmax,
        "Tmax_h": tmax,
        "AUC_mg_h_L": auc,
        "half_life_h": half_life_h,
    }
    if warning:
        out["warning"] = warning
    return out


def run_pbpk_simulation(
    dose_mg: float,
    frequency: str | dict,
    patient_params: dict[str, Any],
    organ_params: dict[str, Any],
    t_end_hours: float = 168.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Coarse physiologically-based PK (PBPK) model.

    We model gut → liver → central → kidney with simple mass-balance ODEs.
    Disease states are expressed via organ_params (e.g. reduced hepatic blood
    flow or renal clearance).
    """
    interval_hours = _parse_frequency_hours(frequency)
    if interval_hours is None or interval_hours <= 0:
        interval_hours = t_end_hours  # single dose fallback
    num_doses = max(1, int(t_end_hours / interval_hours))

    # Volumes (L)
    V_gut = float(organ_params.get("V_gut", 3.0))
    V_liver = float(organ_params.get("V_liver", 1.5))
    V_central = float(organ_params.get("V_central", patient_params.get("Vd", 10.0)))
    V_kidney = float(organ_params.get("V_kidney", 0.5))

    # Blood flows (L/h)
    Q_hepatic = float(organ_params.get("Q_hepatic", 60.0))
    Q_renal = float(organ_params.get("Q_renal", 20.0))

    # Clearances (L/h), scaled for disease states
    CL_hepatic = float(organ_params.get("CL_hepatic", 0.8))
    CL_renal = float(organ_params.get("CL_renal", 0.2))

    hepatic_function = float(organ_params.get("hepatic_function", 1.0))
    renal_function = float(organ_params.get("renal_function", 1.0))
    CL_hepatic *= hepatic_function
    CL_renal *= renal_function

    ka = float(patient_params.get("ka", 1.0))  # gut absorption rate (1/h)

    def ode(_t: float, y: np.ndarray) -> np.ndarray:
        A_gut, C_liver, C_central, C_kidney = y

        # Gut: first-order absorption to portal vein / liver
        dA_gut = -ka * A_gut

        # Convert amounts to liver concentration surrogate (mg/L)
        # Assume A_gut transfer appears in liver compartment as rate ka * A_gut / V_liver
        absorption_rate = ka * A_gut / V_liver

        dC_liver = (
            absorption_rate
            - (Q_hepatic / V_liver) * C_liver
            + (Q_hepatic / V_central) * C_central
            - (CL_hepatic / V_liver) * C_liver
        )

        dC_central = (
            (Q_hepatic / V_liver) * C_liver
            - (Q_hepatic / V_central) * C_central
            - (Q_renal / V_central) * C_central
            + (Q_renal / V_kidney) * C_kidney
        )

        dC_kidney = (
            (Q_renal / V_central) * C_central
            - (Q_renal / V_kidney) * C_kidney
            - (CL_renal / V_kidney) * C_kidney
        )

        return np.array([dA_gut, dC_liver, dC_central, dC_kidney])

    t_list: list[np.ndarray] = []
    c_list: list[np.ndarray] = []

    # Initial conditions: entire dose in gut
    A_gut0 = dose_mg
    y0 = np.array([A_gut0, 0.0, 0.0, 0.0])
    t_cur = 0.0
    for _ in range(num_doses):
        t_span = (t_cur, min(t_cur + interval_hours, t_end_hours))
        if t_span[1] <= t_span[0]:
            break
        sol = solve_ivp(
            ode,
            t_span,
            y0,
            t_eval=np.linspace(t_span[0], t_span[1], max(2, int((t_span[1] - t_span[0]) * 4))),
            method="LSODA",
        )
        t_list.append(sol.t)
        c_list.append(sol.y[2])  # central compartment concentration surrogate

        # Advance state to end of interval and add next dose to gut
        A_gut_end, C_liver_end, C_central_end, C_kidney_end = sol.y[:, -1]
        A_gut_next = A_gut_end + dose_mg
        y0 = np.array([A_gut_next, C_liver_end, C_central_end, C_kidney_end])
        t_cur += interval_hours
        if t_cur >= t_end_hours:
            break

    t_out = np.concatenate(t_list) if t_list else np.linspace(0, t_end_hours, max(2, int(t_end_hours * 4)))
    c_out = np.concatenate(c_list) if c_list else np.zeros_like(t_out)
    c_out = np.maximum(c_out, 0.0)

    trace = (
        "PBPK model with gut→liver→central→kidney compartments. "
        f"Dose={dose_mg} mg, interval={interval_hours} h, num_doses={num_doses}, "
        f"hepatic_function={hepatic_function}, renal_function={renal_function}."
    )
    return t_out, c_out, trace


def run_population_simulation(
    dose_mg: float,
    frequency: str | dict,
    drug: str,
    cohort_distribution: str | None,
    n_patients: int,
    t_end_hours: float = 168.0,
) -> dict[str, Any]:
    """
    Population PK: build a virtual cohort and run run_simulation for each patient.

    Returns aggregate statistics (mean/median Cmax and AUC) and an outlier list,
    plus per-patient summary metrics.
    """
    if n_patients <= 0:
        raise ValueError("n_patients must be positive.")

    phenotype_dist = None
    if cohort_distribution:
        phenotype_dist = parse_cohort_distribution(cohort_distribution)

    cohort = build_cohort(
        cohort_size=n_patients,
        drug=drug,
        phenotype_distribution=phenotype_dist,
    )

    cmax_list: list[float] = []
    auc_list: list[float] = []
    tmax_list: list[float] = []
    per_patient: list[dict[str, Any]] = []
    representative_t: list[float] = []
    representative_c: list[float] = []

    for idx, (patient_params, phenotype_trace) in enumerate(cohort):
        t, C, sim_trace = run_simulation(
            dose_mg=dose_mg,
            frequency=frequency,
            patient_params=patient_params,
            t_end_hours=t_end_hours,
        )
        C_arr = np.asarray(C, dtype=float)
        t_arr = np.asarray(t, dtype=float)
        cmax = float(np.max(C_arr))
        tmax = float(t_arr[int(np.argmax(C_arr))])
        auc = float(np.trapezoid(C_arr, t_arr))

        if idx == 0:
            representative_t = [round(float(x), 4) for x in t_arr]
            representative_c = [round(float(x), 6) for x in C_arr]

        cmax_list.append(cmax)
        tmax_list.append(tmax)
        auc_list.append(auc)
        per_patient.append(
            {
                "patient_index": idx,
                "Cmax_mg_L": cmax,
                "Tmax_h": tmax,
                "AUC_mg_h_L": auc,
                "patient_params": patient_params,
                "phenotype_trace": phenotype_trace,
                "simulation_trace": sim_trace,
            }
        )

    cmax_arr = np.asarray(cmax_list)
    auc_arr = np.asarray(auc_list)

    def _summary(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=0)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    cmax_summary = _summary(cmax_arr)
    auc_summary = _summary(auc_arr)

    # Flag outliers as > 2× median Cmax or AUC
    cmax_med = cmax_summary["median"]
    auc_med = auc_summary["median"]
    outliers: list[dict[str, Any]] = []
    for p in per_patient:
        reasons: list[str] = []
        if p["Cmax_mg_L"] > 2.0 * cmax_med:
            reasons.append("Cmax > 2x cohort median")
        if p["AUC_mg_h_L"] > 2.0 * auc_med:
            reasons.append("AUC > 2x cohort median")
        if reasons:
            outliers.append(
                {
                    "patient_index": p["patient_index"],
                    "Cmax_mg_L": p["Cmax_mg_L"],
                    "AUC_mg_h_L": p["AUC_mg_h_L"],
                    "reasons": reasons,
                    "patient_params": p["patient_params"],
                }
            )

    return {
        "drug": drug,
        "n_patients": n_patients,
        "cohort_distribution": cohort_distribution,
        "cmax_summary": cmax_summary,
        "auc_summary": auc_summary,
        "outliers": outliers,
        "patients": per_patient,
        "representative_t_hours": representative_t,
        "representative_concentration_mg_L": representative_c,
    }
