"""
LLM agents: (1) goal → orchestrator params, (2) trial result → narrative summary.
Uses OpenAI-compatible API. Env vars (OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL) are loaded from .env when present.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

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
}

GOAL_TO_PARAMS_SYSTEM = """You are an expert in clinical trial design and pharmacokinetics.
Given a natural-language goal for a simulated clinical trial, output a single JSON object with exactly these keys (no other keys, no markdown):
- drug (string): drug name, e.g. "warfarin"
- dose_mg (number): dose in mg
- cohort_size (integer): number of simulated patients (typically 30-500)
- frequency (string): dosing interval, e.g. "every 24 hours" or "every 12 hours"
- c_max_threshold_mg_L (number): plasma concentration threshold in mg/L above which a patient is flagged as potential toxicity (e.g. 2.0 to 5.0 for warfarin)

Infer reasonable values from the goal. If the goal mentions "elderly" or "CYP2C9" consider warfarin and a conservative threshold. Output only the JSON object, no explanation."""

SUMMARIZE_SYSTEM = """You are a clinical pharmacologist summarizing a simulated trial for a clinician.
Given the trial result JSON, write a short narrative (2-4 paragraphs) that covers:
1) What was simulated (drug, dose, cohort size, dosing schedule).
2) Overall PK findings (mean/max C_max, number of patients, any errors).
3) Safety flags: how many outliers were flagged and why (C_max above threshold), and a one-sentence summary of the Safety Auditor's PubMed findings for those outliers.
4) Key reasoning from the simulation (e.g. phenotype-based parameter choices) if relevant.

Be concise and use plain language. Do not invent numbers; use only what appears in the result."""


def _get_client():
    """Return OpenAI-compatible client (openai package)."""
    try:
        from openai import OpenAI
    except ImportError:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL")  # e.g. Azure or local proxy
    return OpenAI(api_key=api_key, base_url=base_url if base_url else None)


def _get_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4o-mini")


def goal_to_params(goal: str) -> tuple[dict[str, Any], str]:
    """
    Use an LLM to convert a natural-language goal into orchestrator parameters.
    Returns (params_dict, reasoning_trace). If LLM is unavailable or parsing fails, returns (DEFAULT_PARAMS, trace).
    """
    client = _get_client()
    if not client:
        return (
            {**DEFAULT_PARAMS},
            "LLM not configured (OPENAI_API_KEY unset or openai not installed). Used default params.",
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
        }
        # Clamp cohort_size
        params["cohort_size"] = max(1, min(500, params["cohort_size"]))
        params["dose_mg"] = max(0.1, params["dose_mg"])
        params["c_max_threshold_mg_L"] = max(0.0, params["c_max_threshold_mg_L"])
        return params, f"LLM parsed goal into params: {json.dumps(params)}"
    except Exception as e:
        return (
            {**DEFAULT_PARAMS},
            f"LLM or JSON parse failed ({e!s}). Used default params.",
        )


def summarize_trial_result(result: dict[str, Any]) -> str:
    """
    Use an LLM to produce a short narrative summary of the trial result (reasoning, outcomes, Safety Auditor).
    Returns summary string. If LLM is unavailable, returns a plain fallback summary.
    """
    client = _get_client()
    if not client:
        return _fallback_summary(result)
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
        return (resp.choices[0].message.content or "").strip() or _fallback_summary(result)
    except Exception:
        return _fallback_summary(result)


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
