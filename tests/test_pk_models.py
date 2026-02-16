"""Tests for 1- and 2-compartment PK models (pk_sim_mcp.pk_models)."""

import sys
from pathlib import Path

# Allow importing from servers/pk_sim_mcp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "servers"))

import numpy as np
import pytest
from pk_sim_mcp.pk_models import (
    solve_one_compartment,
    solve_two_compartment,
    run_simulation,
    _parse_frequency_hours,
)


def test_parse_frequency_hours():
    assert _parse_frequency_hours("every 24 hours") == 24.0
    assert _parse_frequency_hours("every 12 h") == 12.0
    assert _parse_frequency_hours({"value": 8, "unit": "hour"}) == 8.0
    assert _parse_frequency_hours({"value": 60, "unit": "minutes"}) == 1.0


def test_one_compartment_single_dose():
    t, C = solve_one_compartment(dose_mg=100, Vd=10.0, ke=0.02, t_end_hours=100)
    assert t[0] == 0
    assert C[0] == pytest.approx(10.0, rel=1e-5)  # Dose/Vd = 10 mg/L
    assert C[-1] < C[0]
    # Half-life ~ ln(2)/ke = 34.66 h; at t=35, C should be ~5
    idx = np.searchsorted(t, 35)
    if idx < len(C):
        assert C[idx] == pytest.approx(5.0, rel=0.15)


def test_one_compartment_repeated_doses():
    t, C = solve_one_compartment(
        dose_mg=50, Vd=10.0, ke=0.1, t_end_hours=24,
        interval_hours=12, num_doses=3,
    )
    assert len(t) > 0 and len(C) == len(t)
    # After multiple doses, peak should be higher than first dose peak (5 mg/L)
    assert C.max() >= 5.0


def test_two_compartment_single_dose():
    t, C = solve_two_compartment(
        dose_mg=100, Vc=10.0, Vp=20.0, CL=1.0, Q=2.0, t_end_hours=48,
    )
    assert t[0] == 0
    assert C[0] == pytest.approx(10.0, rel=1e-5)  # Dose/Vc
    assert C[-1] < C[0]
    assert np.all(C >= 0)


def test_run_simulation_1_comp():
    patient_params = {"model_type": "1_compartment", "Vd": 10.0, "ke": 0.02}
    t, C, trace = run_simulation(50, "every 24 hours", patient_params, t_end_hours=72)
    assert "1-compartment" in trace
    assert "Vd=10" in trace
    assert len(t) == len(C)
    assert np.all(C >= 0)


def test_run_simulation_2_comp():
    patient_params = {"model_type": "2_compartment", "Vc": 10.0, "Vp": 15.0, "CL": 0.5, "Q": 1.0}
    t, C, trace = run_simulation(50, {"value": 12, "unit": "hour"}, patient_params, t_end_hours=24)
    assert "2-compartment" in trace
    assert "Vc=10" in trace
    assert len(t) == len(C)
    assert C[0] == pytest.approx(5.0, rel=1e-5)
