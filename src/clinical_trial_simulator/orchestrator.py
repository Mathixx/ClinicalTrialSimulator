from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .agent import execute, plan
from .mcp_manager import MCPManager
from .schemas import TrialProtocol, TrialState


ProgressCallback = Callable[[Union[str, dict]], None]


async def run_trial_async(
    goal: str,
    *,
    dry_run: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> TrialState:
    """Full autonomous pipeline: plan -> ReAct execute -> TrialState."""
    report: ProgressCallback = progress_callback or (lambda _msg: None)

    report("Planning trial protocol from goal…")
    print("[Orchestrator] Planning trial protocol…", flush=True)
    async with MCPManager() as mcp:
        tool_manifest = mcp.tool_manifest()
        protocol: TrialProtocol = await plan(goal, tool_manifest)
        print("[Orchestrator] Plan done. Starting ReAct execution…", flush=True)
        report("Executing protocol with ReAct loop…")
        state = TrialState(protocol=protocol)
        state = await execute(state, mcp, dry_run=dry_run, progress_callback=report)

    print("[Orchestrator] Trial complete.", flush=True)
    report("Trial complete.")
    return state


def run_trial(
    goal: str,
    *,
    dry_run: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    timeout_seconds: Optional[float] = 300.0,
) -> Dict[str, Any]:
    """Synchronous wrapper for running a trial from a goal string.

    If timeout_seconds is set, it is an *idle* timeout: the timer resets after each
    successful tool use (phase "observation"). So the run is only stopped if no tool
    completes for that many seconds in a row (e.g. LLM or tool hung).
    Returns a plain dict for easy JSON serialization.
    """
    report = progress_callback or (lambda _msg: None)
    last_activity: list[float] = [time.monotonic()]

    def wrap_progress(evt: Any) -> None:
        if isinstance(evt, dict) and evt.get("phase") == "observation":
            last_activity[0] = time.monotonic()
        report(evt)

    async def run_with_idle_timeout() -> TrialState:
        task = asyncio.create_task(
            run_trial_async(goal, dry_run=dry_run, progress_callback=wrap_progress)
        )
        check_interval = 5.0
        while not task.done():
            await asyncio.sleep(check_interval)
            if task.done():
                break
            if timeout_seconds is not None:
                if time.monotonic() - last_activity[0] > timeout_seconds:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise asyncio.TimeoutError(
                        f"No tool completed within {timeout_seconds}s (idle timeout)."
                    )
        if task.cancelled():
            raise asyncio.TimeoutError("Trial was cancelled (idle timeout).")
        return task.result()

    try:
        state = asyncio.run(run_with_idle_timeout())
    except asyncio.TimeoutError:
        report("Trial timed out.")
        return {
            "goal": goal,
            "error": "Trial timed out. Try again with a simpler goal or check MCP servers.",
            "state": None,
        }

    return json.loads(state.model_dump_json())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run an autonomous clinical trial simulation from a natural-language goal."
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default="Run a standard warfarin trial: 50 mg every 24 hours, 50 patients, flag high Cmax.",
        help="Trial goal in natural language.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run: think and plan, but do not actually call MCP tools.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON result.",
    )
    args = parser.parse_args()

    result = run_trial(args.goal, dry_run=args.dry_run)
    out_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(out_json)
    else:
        print(out_json)


if __name__ == "__main__":
    main()

