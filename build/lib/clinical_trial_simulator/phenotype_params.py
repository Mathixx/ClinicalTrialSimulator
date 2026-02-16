"""
Phenotype → PK parameter mapping.
Maps high-level phenotypes (e.g. CKD, CYP2C9 genotype) to patient_params for pk-sim-mcp.
Uses real, cited constants (e.g. Warfarin: Vd ~10 L/70kg, CL ~0.2 L/h; refs: StatPearls, clinical PK literature).
"""

from __future__ import annotations

from typing import Any

# Baseline drug constants (literature values). Warfarin: Vd ~10 L, CL ~0.2 L/h (StatPearls, PMC).
DRUG_BASELINES: dict[str, dict[str, float]] = {
    "warfarin": {"Vd": 10.0, "CL": 0.2, "renal_fraction": 0.0},
    "default": {"Vd": 10.0, "CL": 0.2, "renal_fraction": 0.0},
}

# CYP2C9 phenotype → hepatic CL multiplier (S-warfarin clearance; *1/*1 = 1.0).
# *1/*3 and *3/*3: reduced activity (~60% and ~10–20% of wild-type; see PMC9488981, genotype-dependent DDI).
CYP2C9_MULTIPLIERS: dict[str, float] = {
    "CYP2C9*1/*1": 1.0,
    "CYP2C9*1/*2": 0.7,
    "CYP2C9*1/*3": 0.6,
    "CYP2C9*2/*2": 0.5,
    "CYP2C9*2/*3": 0.4,
    "CYP2C9*3/*3": 0.15,
}

# CKD stage → GFR fraction (simplified; used to scale renal clearance).
# KDIGO stages; renal fraction of CL is scaled by this.
CKD_GFR_FRACTION: dict[str, float] = {
    "CKD_stage_1": 1.0,
    "CKD_stage_2": 0.8,
    "CKD_stage_3a": 0.6,
    "CKD_stage_3b": 0.4,
    "CKD_stage_4": 0.25,
    "CKD_stage_5": 0.1,
    "Chronic_Kidney_Disease": 0.5,  # generic
}


def get_params_for_phenotypes(
    base_drug: str,
    phenotypes: list[str],
    baseline_vd: float | None = None,
    baseline_cl: float | None = None,
    model_type: str = "1_compartment",
) -> tuple[dict[str, Any], str]:
    """
    Map phenotypes to patient_params for pk-sim-mcp.
    Returns (patient_params, reasoning_trace).
    Uses baseline Vd/CL from drug table if not provided.
    """
    base = DRUG_BASELINES.get(base_drug.lower(), DRUG_BASELINES["default"])
    Vd = baseline_vd if baseline_vd is not None else base["Vd"]
    CL = baseline_cl if baseline_cl is not None else base["CL"]
    renal_frac = base.get("renal_fraction", 0.0)
    traces: list[str] = []

    # CYP2C9: scale hepatic clearance (hepatic = (1 - renal_frac) * CL)
    hepatic_mult = 1.0
    for p in phenotypes:
        pn = p.strip()
        if pn in CYP2C9_MULTIPLIERS:
            hepatic_mult *= CYP2C9_MULTIPLIERS[pn]
            traces.append(f"Applied {pn}: hepatic CL × {CYP2C9_MULTIPLIERS[pn]}.")

    # CKD: scale renal clearance
    renal_mult = 1.0
    for p in phenotypes:
        pn = p.strip()
        if pn in CKD_GFR_FRACTION:
            renal_mult = CKD_GFR_FRACTION[pn]
            traces.append(f"Applied {pn}: renal CL × {renal_mult}.")

    # Combined CL: renal_frac * CL * renal_mult + (1 - renal_frac) * CL * hepatic_mult
    CL_adj = (renal_frac * renal_mult + (1 - renal_frac) * hepatic_mult) * CL
    ke = CL_adj / Vd if Vd > 0 else 0.02

    if not traces:
        traces.append("No phenotype modifiers in table; used baseline Vd and CL.")

    reasoning_trace = " ".join(traces) + f" Final Vd={Vd} L, CL={CL_adj:.4f} L/h, ke={ke:.4f} 1/h."

    patient_params: dict[str, Any] = {
        "model_type": model_type,
        "Vd": Vd,
        "ke": ke,
    }
    return patient_params, reasoning_trace


def get_params_for_phenotypes_2comp(
    base_drug: str,
    phenotypes: list[str],
    baseline_Vc: float | None = None,
    baseline_Vp: float | None = None,
    baseline_CL: float | None = None,
    baseline_Q: float | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Same as get_params_for_phenotypes but returns 2-compartment patient_params.
    Defaults: Vc=10, Vp=15, CL=0.5, Q=0.5*CL if not provided.
    """
    base = DRUG_BASELINES.get(base_drug.lower(), DRUG_BASELINES["default"])
    CL = baseline_CL if baseline_CL is not None else base["CL"]
    Vc = baseline_Vc if baseline_Vc is not None else 10.0
    Vp = baseline_Vp if baseline_Vp is not None else 15.0
    Q = baseline_Q if baseline_Q is not None else 0.5 * CL

    # Apply same phenotype logic to CL (simplified: scale CL)
    hepatic_mult = 1.0
    renal_mult = 1.0
    renal_frac = base.get("renal_fraction", 0.0)
    traces: list[str] = []
    for p in phenotypes:
        pn = p.strip()
        if pn in CYP2C9_MULTIPLIERS:
            hepatic_mult *= CYP2C9_MULTIPLIERS[pn]
            traces.append(f"Applied {pn}: hepatic CL × {CYP2C9_MULTIPLIERS[pn]}.")
        if pn in CKD_GFR_FRACTION:
            renal_mult = CKD_GFR_FRACTION[pn]
            traces.append(f"Applied {pn}: renal CL × {renal_mult}.")
    CL_adj = (renal_frac * renal_mult + (1 - renal_frac) * hepatic_mult) * CL
    if not traces:
        traces.append("No phenotype modifiers; used baseline.")
    reasoning_trace = " ".join(traces) + f" 2-comp: Vc={Vc}, Vp={Vp}, CL={CL_adj:.4f}, Q={Q}."

    patient_params: dict[str, Any] = {
        "model_type": "2_compartment",
        "Vc": Vc,
        "Vp": Vp,
        "CL": CL_adj,
        "Q": Q,
    }
    return patient_params, reasoning_trace
