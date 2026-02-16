"""
pk-sim-mcp: FastMCP server exposing simulate_pk for 1- and 2-compartment PK models.
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP

from .pk_models import run_simulation

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
    Simulate drug concentration over time.
    dose: dose in mg.
    frequency: e.g. 'every 24 hours' or {"value": 12, "unit": "hour"}.
    patient_params: dict with model_type, Vd/ke (1-comp) or Vc/Vp/CL/Q (2-comp).
    t_end_hours: simulation end time (default 168 = 1 week).
    """
    t, C, reasoning_trace = run_simulation(
        dose_mg=dose,
        frequency=frequency,
        patient_params=patient_params,
        t_end_hours=t_end_hours,
    )
    # Convert to JSON-serializable lists (float for MCP)
    t_list = [round(float(x), 4) for x in t]
    c_list = [round(float(x), 6) for x in C]
    return {
        "t_hours": t_list,
        "concentration_mg_L": c_list,
        "reasoning_trace": reasoning_trace,
    }


if __name__ == "__main__":
    mcp.run(show_banner=False)
