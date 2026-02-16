"""
Orchestrator script: run one trial (cohort → PK simulation via pk-sim-mcp → outlier detection → PubMed via pubmed-mcp).
Uses MCP client sessions over stdio; no hardcoded PK logic.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure we can import mcp and local packages
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "servers"))

from clinical_trial_simulator.phenotype_params import get_params_for_phenotypes


def _load_mcp_config() -> dict:
    config_path = _REPO_ROOT / "config" / "mcp_servers.json"
    if not config_path.exists():
        return {
            "pk_sim_mcp": {"command": "python", "args": ["-m", "pk_sim_mcp"], "cwd": str(_REPO_ROOT)},
            "pubmed_mcp": {"command": "python", "args": ["-m", "pubmed_mcp"], "cwd": str(_REPO_ROOT)},
        }
    raw = json.loads(config_path.read_text())
    out = {}
    for k, v in raw.items():
        cwd = v.get("cwd", "${workspace_root}")
        if isinstance(cwd, str) and "${workspace_root}" in cwd:
            cwd = cwd.replace("${workspace_root}", str(_REPO_ROOT))
        out[k] = {"command": v.get("command", "python"), "args": v.get("args", []), "cwd": cwd or str(_REPO_ROOT)}
    return out


def _build_cohort(
    cohort_size: int,
    drug: str,
    phenotype_distribution: list[tuple[list[str], float]] | None = None,
) -> list[tuple[dict, str]]:
    """
    Build synthetic cohort: list of (patient_params, reasoning_trace) for pk-sim-mcp.
    phenotype_distribution: list of (phenotypes_list, fraction). E.g. [(["CYP2C9*1/*1"], 0.8), (["CYP2C9*1/*3"], 0.2)].
    """
    if not phenotype_distribution:
        phenotype_distribution = [(["CYP2C9*1/*1"], 0.7), (["CYP2C9*1/*3"], 0.2), (["CYP2C9*3/*3"], 0.1)]
    cohort: list[tuple[dict, str]] = []
    n_remaining = cohort_size
    for phenotypes, frac in phenotype_distribution[:-1]:
        n = max(0, int(cohort_size * frac))
        n_remaining -= n
        for _ in range(n):
            params, trace = get_params_for_phenotypes(drug, phenotypes)
            cohort.append((params, trace))
    # last bucket gets remainder
    phenotypes_last = phenotype_distribution[-1][0]
    for _ in range(n_remaining):
        params, trace = get_params_for_phenotypes(drug, phenotypes_last)
        cohort.append((params, trace))
    return cohort


def _extract_tool_result(result: object) -> dict | list:
    """Extract JSON from MCP CallToolResult (content blocks or structured_content)."""
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    if hasattr(result, "content") and result.content:
        for block in result.content:
            if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    return {"raw": block.text}
    return {}


async def _run_trial_async(
    dose_mg: float,
    frequency: str,
    cohort_size: int,
    c_max_threshold_mg_L: float,
    drug: str = "warfarin",
) -> dict:
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        return {
            "error": "MCP SDK not installed. pip install mcp",
            "cohort_size": cohort_size,
            "outliers": [],
            "per_patient_summary": [],
        }

    config = _load_mcp_config()
    pk_cfg = config["pk_sim_mcp"]
    pubmed_cfg = config["pubmed_mcp"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT / "servers"), str(_REPO_ROOT / "src"), env.get("PYTHONPATH", "")])

    pk_params = StdioServerParameters(
        command=pk_cfg["command"],
        args=list(pk_cfg["args"]),
        cwd=pk_cfg.get("cwd", str(_REPO_ROOT)),
        env=env,
    )
    pubmed_params = StdioServerParameters(
        command=pubmed_cfg["command"],
        args=list(pubmed_cfg["args"]),
        cwd=pubmed_cfg.get("cwd", str(_REPO_ROOT)),
        env=env,
    )

    per_patient_summary: list[dict] = []
    outliers: list[dict] = []

    async with stdio_client(pk_params) as (r1, w1):
        async with stdio_client(pubmed_params) as (r2, w2):
            pk_session = ClientSession(r1, w1)
            pubmed_session = ClientSession(r2, w2)
            await pk_session.initialize()
            await pubmed_session.initialize()

            cohort = _build_cohort(cohort_size, drug)

            for i, (patient_params, param_trace) in enumerate(cohort):
                res = await pk_session.call_tool(
                    "simulate_pk",
                    arguments={
                        "dose": dose_mg,
                        "frequency": frequency,
                        "patient_params": patient_params,
                        "t_end_hours": 168.0,
                    },
                )
                raw = _extract_tool_result(res)
                if getattr(res, "is_error", False):
                    per_patient_summary.append({"patient_id": i, "error": str(raw)})
                    continue
                t_hours = raw.get("t_hours", [])
                conc = raw.get("concentration_mg_L", [])
                trace = raw.get("reasoning_trace", "")
                c_max = max(conc) if conc else 0.0
                auc = 0.0
                if len(t_hours) > 1 and len(conc) == len(t_hours):
                    for j in range(1, len(t_hours)):
                        auc += 0.5 * (conc[j] + conc[j - 1]) * (t_hours[j] - t_hours[j - 1])
                per_patient_summary.append({
                    "patient_id": i,
                    "c_max_mg_L": round(c_max, 6),
                    "auc_mg_h_L": round(auc, 4),
                    "pk_reasoning_trace": trace,
                    "param_trace": param_trace,
                })
                if c_max >= c_max_threshold_mg_L:
                    outliers.append({
                        "patient_id": i,
                        "c_max_mg_L": c_max,
                        "param_trace": param_trace,
                        "pk_trace": trace,
                    })

            for out in outliers:
                query = f"{drug} toxicity CYP2C9 genotype safety"
                res = await pubmed_session.call_tool("search_pubmed", arguments={"query": query, "max_results": 5})
                pubmed_raw = _extract_tool_result(res)
                out["safety_auditor_trace"] = pubmed_raw.get("reasoning_trace", "")
                out["pubmed_articles"] = pubmed_raw.get("articles", [])[:3]

    return {
        "cohort_size": cohort_size,
        "dose_mg": dose_mg,
        "frequency": frequency,
        "drug": drug,
        "c_max_threshold_mg_L": c_max_threshold_mg_L,
        "per_patient_summary": per_patient_summary,
        "outliers": outliers,
        "num_outliers": len(outliers),
    }


def run_trial(
    dose_mg: float = 50.0,
    frequency: str = "every 24 hours",
    cohort_size: int = 100,
    c_max_threshold_mg_L: float = 3.0,
    drug: str = "warfarin",
) -> dict:
    """Synchronous entrypoint: run trial and return structured result."""
    try:
        import anyio
        return anyio.run(
            _run_trial_async(dose_mg, frequency, cohort_size, c_max_threshold_mg_L, drug),
            backend="asyncio",
        )
    except Exception:
        return asyncio.run(_run_trial_async(dose_mg, frequency, cohort_size, c_max_threshold_mg_L, drug))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run one clinical trial simulation (orchestrator).")
    p.add_argument("--dose", type=float, default=50, help="Dose in mg")
    p.add_argument("--frequency", type=str, default="every 24 hours", help="Dosing frequency")
    p.add_argument("--cohort-size", type=int, default=30, help="Cohort size (use smaller for quick test)")
    p.add_argument("--threshold", type=float, default=3.0, help="C_max threshold (mg/L) for toxicity flag")
    p.add_argument("--drug", type=str, default="warfarin")
    p.add_argument("--output", type=str, default=None, help="Write JSON result to file")
    args = p.parse_args()
    result = run_trial(
        dose_mg=args.dose,
        frequency=args.frequency,
        cohort_size=args.cohort_size,
        c_max_threshold_mg_L=args.threshold,
        drug=args.drug,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
