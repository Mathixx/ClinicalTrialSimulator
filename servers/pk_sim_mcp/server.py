"""
pk-sim-mcp: FastMCP server exposing simulate_pk for 1- and 2-compartment PK models.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from .pk_models import (
    noncompartmental_analysis,
    run_pbpk_simulation,
    run_population_simulation,
    run_simulation,
)

mcp = FastMCP(name="pk-sim-mcp")


@mcp.tool(
    description=(
        "Run a PK simulation for a given dose, dosing frequency, and patient parameters. "
        "Returns time-series of plasma concentration (mg/L) and a reasoning trace. "
        "patient_params: model_type ('1_compartment' or '2_compartment'), "
        "then for 1-compartment: Vd (L), ke (1/h), optional ka (1/h for oral); "
        "for 2-compartment: Vc, Vp (L), CL, Q (L/h)."
    ),
)
def simulate_pk(
    dose: float,
    frequency: str | dict[str, Any],
    patient_params: dict[str, Any],
    t_end_hours: float = 168.0,
) -> dict[str, Any]:
    """
    Simulate drug concentration over time with 1- or 2-compartment PK.
    """
    t, C, reasoning_trace = run_simulation(
        dose_mg=dose,
        frequency=frequency,
        patient_params=patient_params,
        t_end_hours=t_end_hours,
    )
    t_list = [round(float(x), 4) for x in t]
    c_list = [round(float(x), 6) for x in C]
    return {
        "t_hours": t_list,
        "concentration_mg_L": c_list,
        "reasoning_trace": reasoning_trace,
    }


@mcp.tool(
    description=(
        "Perform non-compartmental analysis (NCA) on a concentration-time profile. "
        "Input: t_hours and concentrations (mg/L) — use a full profile with many time points, "
        "e.g. representative_t_hours and representative_concentration_mg_L from simulate_pop_pk, "
        "or t_hours and concentration_mg_L from simulate_pk. At least 3 time points recommended. "
        "Output: Cmax, Tmax, AUC, half_life (if determinable), and a warning if input is insufficient."
    ),
)
def simulate_nca(
    t_hours: list[float],
    concentrations: list[float],
) -> dict[str, Any]:
    """
    Non-compartmental analysis wrapper for pre-computed PK profiles.
    """
    return noncompartmental_analysis(t_hours=t_hours, concentrations_mg_L=concentrations)


@mcp.tool(
    description=(
        "Run a simple physiologically-based PK (PBPK) simulation using gut→liver→central→kidney compartments. "
        "patient_params can include ka (1/h) for gut absorption; "
        "organ_params controls organ volumes, blood flows, and organ function modifiers "
        "(e.g. hepatic_function, renal_function). Returns central-compartment concentrations over time."
    ),
)
def simulate_pbpk(
    dose: float,
    frequency: str | dict[str, Any],
    patient_params: dict[str, Any],
    organ_params: dict[str, Any],
    t_end_hours: float = 168.0,
) -> dict[str, Any]:
    """
    PBPK simulation wrapper.
    """
    t, C, trace = run_pbpk_simulation(
        dose_mg=dose,
        frequency=frequency,
        patient_params=patient_params,
        organ_params=organ_params,
        t_end_hours=t_end_hours,
    )
    t_list = [round(float(x), 4) for x in t]
    c_list = [round(float(x), 6) for x in C]
    return {
        "t_hours": t_list,
        "concentration_mg_L": c_list,
        "reasoning_trace": trace,
    }


@mcp.tool(
    description=(
        "Population PK simulation. Builds a virtual cohort using phenotype-driven PK parameters, "
        "runs simulate_pk for each patient, and returns aggregate Cmax/AUC statistics, an outlier list, "
        "and a representative concentration-time curve (representative_t_hours, representative_concentration_mg_L) "
        "from the first patient — use these for simulate_nca or viz_mcp plot_pk_curve. "
        "cohort_distribution: human-readable phenotype mix, e.g. '20% CYP2C9*3/*3, 70% CYP2C9*1/*1'."
    ),
)
def simulate_pop_pk(
    dose: float,
    frequency: str | dict[str, Any],
    drug: str,
    cohort_distribution: str | None,
    n_patients: int = 100,
    t_end_hours: float = 168.0,
) -> dict[str, Any]:
    """
    Population PK wrapper around run_population_simulation.
    """
    return run_population_simulation(
        dose_mg=dose,
        frequency=frequency,
        drug=drug,
        cohort_distribution=cohort_distribution,
        n_patients=n_patients,
        t_end_hours=t_end_hours,
    )


if __name__ == "__main__":
    mcp.run(show_banner=False)
