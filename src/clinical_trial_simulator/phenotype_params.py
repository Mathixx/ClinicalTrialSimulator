"""
Phenotype → PK parameter mapping and cohort construction.
Maps high-level phenotypes (e.g. CKD, CYP2C9 genotype) to patient_params for pk-sim-mcp.
Parses Designer LLM cohort_distribution strings into phenotype_distribution for building cohorts.
Uses real, cited constants (e.g. Warfarin: Vd ~10 L/70kg, CL ~0.2 L/h; refs: StatPearls, clinical PK literature).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

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

# Valid phenotype keys for cohort_distribution parsing
VALID_PHENOTYPE_KEYS = frozenset(CYP2C9_MULTIPLIERS) | frozenset(CKD_GFR_FRACTION)

# Aliases for LLM-generated cohort_distribution (human-readable -> phenotype list)
COHORT_PHENOTYPE_ALIASES: dict[str, list[str]] = {
    "poor metabolizers": ["CYP2C9*3/*3"],
    "poor metabolizer": ["CYP2C9*3/*3"],
    "cyp2c9*3/*3": ["CYP2C9*3/*3"],
    "cyp2c9*2/*3": ["CYP2C9*2/*3"],
    "cyp2c9*1/*3": ["CYP2C9*1/*3"],
    "cyp2c9*1/*2": ["CYP2C9*1/*2"],
    "cyp2c9*1/*1": ["CYP2C9*1/*1"],
    "cyp2c9*2/*2": ["CYP2C9*2/*2"],
    "ckd stage 4": ["CKD_stage_4"],
    "ckd_stage_4": ["CKD_stage_4"],
    "ckd stage 5": ["CKD_stage_5"],
    "ckd_stage_5": ["CKD_stage_5"],
    "ckd stage 3b": ["CKD_stage_3b"],
    "ckd_stage_3b": ["CKD_stage_3b"],
    "ckd stage 3a": ["CKD_stage_3a"],
    "ckd_stage_3a": ["CKD_stage_3a"],
    "ckd stage 2": ["CKD_stage_2"],
    "ckd_stage_2": ["CKD_stage_2"],
    "ckd stage 1": ["CKD_stage_1"],
    "ckd_stage_1": ["CKD_stage_1"],
    "chronic kidney disease": ["Chronic_Kidney_Disease"],
}


def parse_cohort_distribution(
    distribution_str: str,
) -> list[tuple[list[str], float]] | None:
    """
    Parse a cohort_distribution string from the Designer LLM into phenotype_distribution.
    Format: "20% CYP2C9*3/*3, 10% CKD_stage_4, 70% CYP2C9*1/*1" (percentages sum to 100).
    Returns list of (phenotypes_list, fraction) or None if parsing fails.
    """
    if not distribution_str or not distribution_str.strip():
        return None
    parts = [p.strip() for p in distribution_str.split(",") if p.strip()]
    if not parts:
        return None
    result: list[tuple[list[str], float]] = []
    for part in parts:
        match = re.match(r"^(\d+(?:\.\d+)?)\s*%\s*(.+)$", part, re.IGNORECASE)
        if not match:
            continue
        pct = float(match.group(1)) / 100.0
        label = match.group(2).strip()
        if pct <= 0:
            continue
        label_lower = label.lower()
        if label_lower in COHORT_PHENOTYPE_ALIASES:
            phenotypes = COHORT_PHENOTYPE_ALIASES[label_lower]
        elif label in VALID_PHENOTYPE_KEYS:
            phenotypes = [label]
        else:
            for key in VALID_PHENOTYPE_KEYS:
                if key.lower() == label_lower:
                    phenotypes = [key]
                    break
            else:
                logger.warning("Unknown phenotype label in cohort_distribution: %r, skipping", label)
                continue
        result.append((phenotypes, pct))
    if not result:
        return None
    total = sum(f for _, f in result)
    if total <= 0:
        return None
    return [(p, f / total) for p, f in result]


def build_cohort(
    cohort_size: int,
    drug: str,
    phenotype_distribution: list[tuple[list[str], float]] | None = None,
) -> list[tuple[dict[str, Any], str]]:
    """
    Build synthetic cohort: list of (patient_params, reasoning_trace) for pk-sim-mcp.
    phenotype_distribution: list of (phenotypes_list, fraction). E.g. [(["CYP2C9*1/*1"], 0.8), (["CYP2C9*1/*3"], 0.2)].
    """
    if not phenotype_distribution:
        phenotype_distribution = [(["CYP2C9*1/*1"], 0.7), (["CYP2C9*1/*3"], 0.2), (["CYP2C9*3/*3"], 0.1)]
    cohort: list[tuple[dict[str, Any], str]] = []
    n_remaining = cohort_size
    for phenotypes, frac in phenotype_distribution[:-1]:
        n = max(0, int(cohort_size * frac))
        n_remaining -= n
        for _ in range(n):
            params, trace = get_params_for_phenotypes(drug, phenotypes)
            cohort.append((params, trace))
    phenotypes_last = phenotype_distribution[-1][0]
    for _ in range(n_remaining):
        params, trace = get_params_for_phenotypes(drug, phenotypes_last)
        cohort.append((params, trace))
    return cohort


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
