"""
LLM agents: (1) goal → orchestrator params, (2) trial result → narrative summary.
Uses OpenAI-compatible API. Env vars (OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL) are loaded from .env when present.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Load .env from repo root or cwd so OPENAI_* and LLM_MODEL are available
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Try repo root (parent of src/)
    _this_file = Path(__file__).resolve()
    _repo_root = _this_file.parent.parent.parent
    for d in (_repo_root, Path.cwd()):
        env_file = d / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
    else:
        load_dotenv()  # default: cwd and parents

_load_dotenv()

# Defaults when LLM is unavailable or parsing fails
DEFAULT_PARAMS = {
    "drug": "warfarin",
    "dose_mg": 50.0,
    "cohort_size": 100,
    "frequency": "every 24 hours",
    "c_max_threshold_mg_L": 3.0,
    "cohort_distribution": None,
}

GOAL_TO_PARAMS_SYSTEM = """You are a Senior Clinical Pharmacologist designing a simulated trial (Research Designer phase).

Given the user's goal, output a single JSON object with exactly these keys (no other keys, no markdown):
- drug (string): drug name, e.g. "warfarin"
- dose_mg (number): dose in mg
- cohort_size (integer): number of simulated patients (typically 30-500)
- frequency (string): dosing interval, e.g. "every 24 hours" or "every 12 hours"
- c_max_threshold_mg_L (number): plasma C_max threshold in mg/L above which a patient is flagged for potential toxicity
- cohort_distribution (string): a single line describing the STRESS-TEST cohort so interesting results appear. Use ONLY these phenotype labels and percentages separated by commas. Format: "X% phenotype1, Y% phenotype2, Z% phenotype3" (fractions must sum to 100). Valid phenotypes: CYP2C9*1/*1, CYP2C9*1/*2, CYP2C9*1/*3, CYP2C9*2/*2, CYP2C9*2/*3, CYP2C9*3/*3 (Poor Metabolizers), CKD_stage_4, CKD_stage_5, CKD_stage_3b. To find interesting results you MUST include at least 20% Poor Metabolizers (use CYP2C9*3/*3 or CYP2C9*2/*3) and at least 10% CKD_stage_4 patients to test clearance limits. Example: "20% CYP2C9*3/*3, 10% CKD_stage_4, 70% CYP2C9*1/*1"

Your task:
1. Threshold discovery: Use your knowledge to set c_max_threshold_mg_L (e.g. warfarin narrow therapeutic window ~2-5 mg/L; if unknown, estimate from typical therapeutic index).
2. Population diversity: Do NOT create a generic cohort. Ensure the cohort_distribution string includes at least 20% Poor Metabolizers (CYP2C9*3/*3 or CYP2C9*2/*3) and 10% CKD_stage_4 so the simulation can stress-test clearance limits.

Output only the JSON object, no explanation."""

SUMMARIZE_SYSTEM = """You are a Medical Science Liaison analyzing a simulated trial (Medical Writer phase).

Given the trial result JSON, produce a report that includes the following. Use Markdown for structure. Do not invent numbers; use only what appears in the result.

1) **Outlier summary**: Use safety_auditor_summary (concentration excess + phenotypes) to describe who was flagged. Then for a few representative outliers, explain the biological "why" using param_trace and pk_trace (e.g. "Patient 45 had ~70% reduction in clearance due to CYP2C9*3/*3 genotype, leading to Cmax 8.2 mg/L"). You do not need to list every outlier.

2) **Safety verdict**: Use the SINGLE shared PubMed search result: safety_auditor (query, reasoning_trace, articles). Compare simulation results against these articles. One short paragraph citing the literature (titles/PMIDs) and your known knowledge.

3) **Recommendation**: One short paragraph: Should the dose be adjusted for the high-risk demographic (Poor Metabolizers, CKD) based on this simulation and PubMed evidence? End with a clear yes/no and one sentence rationale.

Be concise. Output valid Markdown only."""


def _get_client():
    """Return OpenAI-compatible client (openai package)."""
    try:
        from openai import OpenAI
    except ImportError:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _get_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4o-mini")


def goal_to_params(goal: str) -> tuple[dict[str, Any], str]:
    """
    Use an LLM to convert a natural-language goal into orchestrator parameters.
    Returns (params_dict, reasoning_trace).
    Raises RuntimeError if the LLM is unreachable or response parsing fails (process stops).
    """
    client = _get_client()
    if not client:
        raise RuntimeError(
            "LLM not configured. Set OPENAI_API_KEY in .env (and install openai). "
            "Cannot run trial from goal without the Designer LLM."
        )
    try:
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": GOAL_TO_PARAMS_SYSTEM},
                {"role": "user", "content": goal.strip() or "Run a standard warfarin trial."},
            ],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown code block if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()
        data = json.loads(text)
        params = {
            "drug": str(data.get("drug", DEFAULT_PARAMS["drug"])),
            "dose_mg": float(data.get("dose_mg", DEFAULT_PARAMS["dose_mg"])),
            "cohort_size": int(data.get("cohort_size", DEFAULT_PARAMS["cohort_size"])),
            "frequency": str(data.get("frequency", DEFAULT_PARAMS["frequency"])),
            "c_max_threshold_mg_L": float(data.get("c_max_threshold_mg_L", DEFAULT_PARAMS["c_max_threshold_mg_L"])),
            "cohort_distribution": data.get("cohort_distribution") or None,
        }
        if isinstance(params["cohort_distribution"], str):
            params["cohort_distribution"] = params["cohort_distribution"].strip() or None
        # Clamp cohort_size
        params["cohort_size"] = max(1, min(500, params["cohort_size"]))
        params["dose_mg"] = max(0.1, params["dose_mg"])
        params["c_max_threshold_mg_L"] = max(0.0, params["c_max_threshold_mg_L"])
        return params, f"LLM parsed goal into params: {json.dumps({k: v for k, v in params.items() if v is not None})}"
    except Exception as e:
        err_parts = [f"{type(e).__name__}: {str(e).strip()}"]
        if getattr(e, "__cause__", None) is not None:
            c = e.__cause__
            err_parts.append(f" (underlying: {type(c).__name__}: {c})")
        err = "".join(err_parts)
        raise RuntimeError(
            f"LLM unreachable or invalid response — {err}. "
            "Check network connectivity, valid OPENAI_API_KEY. Trial stopped."
        ) from e


def summarize_trial_result(result: dict[str, Any]) -> str:
    """
    Use an LLM to produce a short narrative summary of the trial result (reasoning, outcomes, Safety Auditor).
    Returns summary string. If LLM is unavailable or returns empty, returns a plain fallback summary.
    """
    fallback = _fallback_summary(result)
    client = _get_client()
    if not client:
        logger.info("Summarize: no LLM client, using fallback summary")
        return fallback
    try:
        # Truncate very large payloads for context
        payload = {k: v for k, v in result.items() if k not in ("per_patient_summary",)}
        if "per_patient_summary" in result:
            payload["per_patient_summary_sample"] = result["per_patient_summary"][:5]
            payload["per_patient_summary_total"] = len(result["per_patient_summary"])
        json_str = json.dumps(payload, indent=2)[:12000]
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": f"Summarize this trial result:\n\n{json_str}"},
            ],
            temperature=0.3,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            logger.warning("Summarize: LLM returned empty content, using fallback summary")
            return fallback
        logger.info("Summarize: LLM summary length=%d", len(content))
        return content
    except Exception as e:
        logger.warning("Summarize: LLM failed (%s), using fallback summary", e)
        return fallback


def _fallback_summary(result: dict[str, Any]) -> str:
    """Plain summary when LLM is not used."""
    parts = [
        f"Trial: {result.get('drug', '?')} {result.get('dose_mg', '?')} mg, {result.get('frequency', '?')}, cohort n={result.get('cohort_size', 0)}.",
        f"Outliers flagged: {result.get('num_outliers', 0)} (C_max >= {result.get('c_max_threshold_mg_L', '?')} mg/L).",
    ]
    per = result.get("per_patient_summary", [])
    if per:
        c_maxs = [p.get("c_max_mg_L") for p in per if isinstance(p.get("c_max_mg_L"), (int, float))]
        if c_maxs:
            parts.append(f"Mean C_max: {sum(c_maxs) / len(c_maxs):.4f} mg/L; max: {max(c_maxs):.4f} mg/L.")
    outliers = result.get("outliers", [])
    if outliers:
        parts.append("Safety Auditor PubMed queries were run for each outlier; see 'safety_auditor_trace' and 'pubmed_articles' per outlier.")
    return " ".join(parts)
