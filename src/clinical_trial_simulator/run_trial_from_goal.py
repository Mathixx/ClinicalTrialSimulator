"""
Run a trial from a natural-language goal: LLM (goal→params) → orchestrator → LLM (result→summary).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

# Repo root and path setup
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "servers"))

from clinical_trial_simulator.llm_agents import goal_to_params, summarize_trial_result
from clinical_trial_simulator.run_trial import run_trial


def run_trial_from_goal(goal: str, progress_callback: Callable[[str], None] | None = None) -> dict:
    """
    Run the full pipeline: goal → LLM params → orchestrator → result → LLM summary.
    Returns dict with keys: goal, params_used, params_reasoning, result, summary.
    Optional progress_callback(step: str) for UI progress.
    """
    report = progress_callback or (lambda _: None)
    report("Interpreting goal with LLM (extracting drug, dose, cohort size, threshold)…")
    params, params_reasoning = goal_to_params(goal)
    report("Running orchestrator (MCP: pk-sim, cohort, Safety Auditor)…")
    result = run_trial(
        drug=params["drug"],
        dose_mg=params["dose_mg"],
        cohort_size=params["cohort_size"],
        frequency=params["frequency"],
        c_max_threshold_mg_L=params["c_max_threshold_mg_L"],
        progress_callback=progress_callback,
        cohort_distribution=params.get("cohort_distribution"),
    )
    report("Summarizing results with LLM…")
    summary = summarize_trial_result(result)
    report("Done.")
    return {
        "goal": goal,
        "params_used": params,
        "params_reasoning": params_reasoning,
        "result": result,
        "summary": summary,
    }


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run trial from a natural-language goal (LLM → orchestrator → LLM summary).")
    p.add_argument("goal", nargs="?", default="", help="e.g. 'Test 50mg Warfarin on 100 elderly with varying CYP2C9'")
    p.add_argument("--output", type=str, default=None, help="Write JSON to file")
    args = p.parse_args()
    goal = args.goal.strip() or "Run a standard warfarin trial: 50 mg every 24 hours, 50 patients, flag C_max above 3 mg/L."
    out = run_trial_from_goal(goal)
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    else:
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
