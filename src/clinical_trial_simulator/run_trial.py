"""
Orchestrator script: run one trial (cohort → PK simulation via pk-sim-mcp → outlier detection → PubMed via pubmed-mcp).
Uses MCP client sessions over stdio; no hardcoded PK logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

# Detailed terminal logs for debugging MCP server startup and tool calls
logging.basicConfig(
    level=logging.INFO,
    format="[orchestrator] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("clinical_trial_simulator.run_trial")

# Default timeout for full trial (avoid indefinite hang if MCP blocks)
DEFAULT_TRIAL_TIMEOUT_SECONDS = 120
# Timeout for MCP session initialize(); if server never responds we fail fast with a clear error
MCP_INIT_TIMEOUT_SECONDS = 20

# Ensure we can import mcp and local packages
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "servers"))

from clinical_trial_simulator.phenotype_params import build_cohort, parse_cohort_distribution


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


def _noop_progress(_: str) -> None:
    pass


# Phenotype keywords to build a single PubMed query from outlier param_traces
_PHENOTYPE_QUERY_TERMS = (
    "CYP2C9",
    "CKD",
    "genotype",
    "phenotype",
    "metabolizer",
    "clearance",
    "warfarin",
    "safety",
    "toxicity",
)


def _build_outlier_summary_and_query(
    drug: str, outliers: list[dict], threshold_mg_L: float
) -> tuple[str, str]:
    """Build (1) a short text summary of all outliers and (2) one PubMed query string.
    Outlier summary: concentration excess + phenotypes. Query: drug + phenotype terms from traces.
    """
    if not outliers:
        return "", f"{drug} toxicity safety"
    c_maxs = [o.get("c_max_mg_L") for o in outliers if isinstance(o.get("c_max_mg_L"), (int, float))]
    c_min = min(c_maxs) if c_maxs else 0
    c_max = max(c_maxs) if c_maxs else 0
    n = len(outliers)
    summary_parts = [
        f"{n} outlier(s); Cmax range {c_min:.2f}–{c_max:.2f} mg/L (threshold {threshold_mg_L} mg/L)."
    ]
    # Collect phenotype-related terms from param_trace
    seen: set[str] = set()
    for o in outliers:
        trace = (o.get("param_trace") or "")[:500]
        for term in _PHENOTYPE_QUERY_TERMS:
            if term.lower() in trace.lower() and term not in seen:
                seen.add(term)
    query_terms = [drug, "toxicity"] + sorted(seen)[:5] + ["safety"]
    query = " ".join(query_terms)
    # One-line phenotype mention for summary
    if seen:
        summary_parts.append(f" Phenotypes/pathways: {', '.join(sorted(seen)[:8])}.")
    summary_text = " ".join(summary_parts)
    return summary_text, query


def _log_mcp_startup_failure(exc: BaseException) -> None:
    """Log a short hint when the failure is likely the MCP server process exiting (e.g. wrong Python)."""
    cause = exc
    if isinstance(exc, BaseExceptionGroup):
        for e in exc.exceptions:
            cause = e
            break
    if "BrokenResourceError" in type(cause).__name__ or (getattr(cause, "__module__", "").endswith("anyio")):
        logger.error(
            "MCP server subprocess exited or closed the connection. "
            "Ensure the server is started with the same Python as this process (orchestrator now uses sys.executable)."
        )


async def _run_trial_async(
    dose_mg: float,
    frequency: str,
    cohort_size: int,
    c_max_threshold_mg_L: float,
    drug: str = "warfarin",
    progress_callback: Callable[[str], None] | None = None,
    phenotype_distribution: list[tuple[list[str], float]] | None = None,
) -> dict:
    report = progress_callback or _noop_progress
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError as e:
        logger.error("MCP SDK not installed: %s. pip install mcp", e)
        return {
            "error": "MCP SDK not installed. pip install mcp",
            "cohort_size": cohort_size,
            "outliers": [],
            "per_patient_summary": [],
            "safety_auditor_summary": "",
            "safety_auditor": {"query": "", "reasoning_trace": "", "articles": []},
            "visual_summary": {"normal_n": 0, "outlier_n": 0},
            "visual_chart_data": {},
        }

    report("Loading MCP config…")
    config = _load_mcp_config()
    pk_cfg = config["pk_sim_mcp"]
    pubmed_cfg = config["pubmed_mcp"]
    logger.info("MCP config loaded: pk_sim_mcp cwd=%s, pubmed_mcp cwd=%s", pk_cfg.get("cwd"), pubmed_cfg.get("cwd"))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT / "servers"), str(_REPO_ROOT / "src"), env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"

    # Use the same Python that's running this process so the subprocess has the venv (mcp, fastmcp, pk_sim_mcp).
    # Otherwise "python" can resolve to system Python, the server exits with ImportError, and we get BrokenResourceError.
    pk_command = sys.executable if pk_cfg["command"] == "python" else pk_cfg["command"]
    pubmed_command = sys.executable if pubmed_cfg["command"] == "python" else pubmed_cfg["command"]

    pk_args = list(pk_cfg["args"])
    if pk_cfg["command"] == "python" and "-u" not in pk_args:
        pk_args = ["-u"] + pk_args
    pubmed_args = list(pubmed_cfg["args"])
    if pubmed_cfg["command"] == "python" and "-u" not in pubmed_args:
        pubmed_args = ["-u"] + pubmed_args

    logger.info("Spawning pk-sim-mcp: command=%s args=%s cwd=%s", pk_command, pk_args, pk_cfg.get("cwd"))
    pk_params = StdioServerParameters(
        command=pk_command,
        args=pk_args,
        cwd=pk_cfg.get("cwd", str(_REPO_ROOT)),
        env=env,
    )
    logger.info("Spawning pubmed-mcp: command=%s args=%s cwd=%s", pubmed_command, pubmed_args, pubmed_cfg.get("cwd"))
    pubmed_params = StdioServerParameters(
        command=pubmed_command,
        args=pubmed_args,
        cwd=pubmed_cfg.get("cwd", str(_REPO_ROOT)),
        env=env,
    )

    per_patient_summary: list[dict] = []
    outliers: list[dict] = []

    report("Spawning pk-sim-mcp and pubmed-mcp…")
    try:
        async with stdio_client(pk_params) as (r1, w1):
            logger.info("pk-sim-mcp subprocess started, initializing session…")
            # ClientSession must be used as async context manager: __aenter__ starts the receive loop
            # that reads server responses; without it initialize() would block forever (MCP SDK design).
            async with ClientSession(r1, w1) as pk_session:
                async with stdio_client(pubmed_params) as (r2, w2):
                    logger.info("pubmed-mcp subprocess started, initializing sessions…")
                    async with ClientSession(r2, w2) as pubmed_session:
                        report("Initializing MCP sessions…")
                        try:
                            await asyncio.wait_for(
                                pk_session.initialize(),
                                timeout=MCP_INIT_TIMEOUT_SECONDS,
                            )
                            logger.info("pk-sim-mcp session initialized OK")
                        except asyncio.TimeoutError:
                            logger.error(
                                "pk-sim-mcp did not respond to initialize within %ss. "
                                "Check that the server runs with show_banner=False and uses stderr for logs.",
                                MCP_INIT_TIMEOUT_SECONDS,
                            )
                            raise RuntimeError(
                                "pk-sim-mcp did not respond to initialize. "
                                "Server may be writing to stdout (breaks MCP stdio) or blocking at startup."
                            ) from None
                        except Exception as e:
                            logger.exception("pk-sim-mcp initialize failed: %s", e)
                            raise
                        try:
                            await asyncio.wait_for(
                                pubmed_session.initialize(),
                                timeout=MCP_INIT_TIMEOUT_SECONDS,
                            )
                            logger.info("pubmed-mcp session initialized OK")
                        except asyncio.TimeoutError:
                            logger.error("pubmed-mcp did not respond to initialize within %ss", MCP_INIT_TIMEOUT_SECONDS)
                            raise RuntimeError(
                                "pubmed-mcp did not respond to initialize."
                            ) from None
                        except Exception as e:
                            logger.exception("pubmed-mcp initialize failed: %s", e)
                            raise

                        report("Building cohort…")
                        cohort = build_cohort(cohort_size, drug, phenotype_distribution=phenotype_distribution)
                        logger.info("Cohort built: %d patients", len(cohort))

                        cohort_sum_conc: list[float] | None = None
                        cohort_t_hours: list[float] = []
                        worst_outlier_t_hours: list[float] = []
                        worst_outlier_conc: list[float] = []
                        worst_outlier_id: int | None = None
                        worst_outlier_cmax: float = -1.0

                        for i, (patient_params, param_trace) in enumerate(cohort):
                            if i == 0:
                                logger.info("First call_tool simulate_pk (patient 0)…")
                            report(f"Running PK simulation (patient {i + 1}/{len(cohort)})…")
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
                                logger.warning("simulate_pk error for patient %s: %s", i, raw)
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
                            if cohort_sum_conc is None and len(t_hours) == len(conc):
                                cohort_sum_conc = [float(x) for x in conc]
                                cohort_t_hours = list(t_hours)
                            elif cohort_sum_conc is not None and len(conc) == len(cohort_sum_conc):
                                for k, v in enumerate(conc):
                                    cohort_sum_conc[k] += float(v)
                            if c_max >= c_max_threshold_mg_L and c_max > worst_outlier_cmax and len(t_hours) == len(conc):
                                worst_outlier_cmax = c_max
                                worst_outlier_id = i
                                worst_outlier_t_hours = list(t_hours)
                                worst_outlier_conc = [float(x) for x in conc]
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

                        report("Detecting outliers…")
                        safety_auditor_summary = ""
                        safety_auditor = {"query": "", "reasoning_trace": "", "articles": []}
                        if not outliers:
                            logger.info("Outliers: 0; no PubMed Safety Auditor query (no high-Cmax patients).")
                        else:
                            summary_text, query = _build_outlier_summary_and_query(
                                drug, outliers, c_max_threshold_mg_L
                            )
                            safety_auditor_summary = summary_text
                            report("Safety Auditor: one PubMed query for all outliers…")
                            res = await pubmed_session.call_tool(
                                "search_pubmed", arguments={"query": query, "max_results": 10}
                            )
                            pubmed_raw = _extract_tool_result(res)
                            if not isinstance(pubmed_raw, dict):
                                pubmed_raw = {}
                            articles = pubmed_raw.get("articles", [])[:10]
                            safety_auditor = {
                                "query": query,
                                "reasoning_trace": pubmed_raw.get("reasoning_trace", ""),
                                "articles": articles,
                            }
                            titles = [a.get("title") or a.get("pmid") or "?" for a in articles]
                            logger.info(
                                "PubMed (single query for all %s outliers): query=%r → %s articles: %s",
                                len(outliers), query, len(articles), titles[:3],
                            )
                        for out in outliers:
                            out["safety_auditor_trace"] = "See shared Safety Auditor review (outlier summary + PubMed)."
                            out["pubmed_articles"] = safety_auditor.get("articles", [])
    except Exception as e:
        _log_mcp_startup_failure(e)
        logger.exception("Orchestrator error: %s", e)
        raise

    report("Trial complete.")
    logger.info("Trial complete: %d patients, %d outliers", len(per_patient_summary), len(outliers))

    outlier_ids = {o["patient_id"] for o in outliers}
    normal_cmaxs = [
        p["c_max_mg_L"] for p in per_patient_summary
        if isinstance(p.get("c_max_mg_L"), (int, float)) and p["patient_id"] not in outlier_ids
    ]
    outlier_cmaxs = [o["c_max_mg_L"] for o in outliers if isinstance(o.get("c_max_mg_L"), (int, float))]
    n_normal = len(normal_cmaxs)
    n_out = len(outlier_cmaxs)
    visual_summary = {
        "normal_n": n_normal,
        "normal_mean_cmax_mg_L": round(sum(normal_cmaxs) / n_normal, 4) if n_normal else None,
        "normal_max_cmax_mg_L": round(max(normal_cmaxs), 4) if n_normal else None,
        "outlier_n": n_out,
        "outlier_mean_cmax_mg_L": round(sum(outlier_cmaxs) / n_out, 4) if n_out else None,
        "outlier_max_cmax_mg_L": round(max(outlier_cmaxs), 4) if n_out else None,
    }

    visual_chart_data: dict = {}
    if cohort_sum_conc is not None and len(per_patient_summary) > 0:
        n_pts = len(per_patient_summary)
        cohort_avg_conc = [round(c / n_pts, 6) for c in cohort_sum_conc]
        visual_chart_data = {
            "cohort_avg_t_hours": cohort_t_hours,
            "cohort_avg_concentration_mg_L": cohort_avg_conc,
        }
        if worst_outlier_id is not None and worst_outlier_t_hours and worst_outlier_conc:
            visual_chart_data["outlier_patient_id"] = worst_outlier_id
            visual_chart_data["outlier_t_hours"] = worst_outlier_t_hours
            visual_chart_data["outlier_concentration_mg_L"] = worst_outlier_conc

    return {
        "cohort_size": cohort_size,
        "dose_mg": dose_mg,
        "frequency": frequency,
        "drug": drug,
        "c_max_threshold_mg_L": c_max_threshold_mg_L,
        "per_patient_summary": per_patient_summary,
        "outliers": outliers,
        "num_outliers": len(outliers),
        "safety_auditor_summary": safety_auditor_summary,
        "safety_auditor": safety_auditor,
        "visual_summary": visual_summary,
        "visual_chart_data": visual_chart_data,
    }


def run_trial(
    dose_mg: float = 50.0,
    frequency: str = "every 24 hours",
    cohort_size: int = 100,
    c_max_threshold_mg_L: float = 3.0,
    drug: str = "warfarin",
    progress_callback: Callable[[str], None] | None = None,
    timeout_seconds: float | None = DEFAULT_TRIAL_TIMEOUT_SECONDS,
    cohort_distribution: str | None = None,
) -> dict:
    """Synchronous entrypoint: run trial and return structured result. Optional progress_callback(step: str). cohort_distribution: Designer LLM output, e.g. '20% CYP2C9*3/*3, 10% CKD_stage_4, 70% CYP2C9*1/*1'."""
    phenotype_distribution = parse_cohort_distribution(cohort_distribution) if cohort_distribution else None
    coro = _run_trial_async(
        dose_mg, frequency, cohort_size, c_max_threshold_mg_L, drug,
        progress_callback=progress_callback,
        phenotype_distribution=phenotype_distribution,
    )
    try:
        if timeout_seconds is not None:
            return asyncio.run(asyncio.wait_for(coro, timeout=timeout_seconds))
        return asyncio.run(coro)
    except asyncio.TimeoutError:
        logger.warning("Trial timed out after %s seconds", timeout_seconds)
        if progress_callback:
            progress_callback("Trial timed out (MCP or simulation took too long).")
        return {
            "error": "Trial timed out (MCP connection or simulation took too long). Try a smaller cohort or check MCP servers.",
            "cohort_size": cohort_size,
            "outliers": [],
            "per_patient_summary": [],
            "safety_auditor_summary": "",
            "safety_auditor": {"query": "", "reasoning_trace": "", "articles": []},
            "visual_summary": {"normal_n": 0, "outlier_n": 0},
            "visual_chart_data": {},
        }


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
