from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_REACT_LOG_PREFIX = "[ReAct]"


def _react_log(msg: str, banner: bool = False) -> None:
    """Print a visible log line for ReAct loop progress (flushed so it appears immediately)."""
    if banner:
        width = 60
        sys.stdout.write(f"\n{'=' * width}\n{_REACT_LOG_PREFIX} {msg}\n{'=' * width}\n")
    else:
        sys.stdout.write(f"{_REACT_LOG_PREFIX} {msg}\n")
    sys.stdout.flush()

from .mcp_manager import MCPManager
from .schemas import TrialProtocol, TrialState


PLANNER_SYSTEM_PROMPT = """You are a Lead Clinical Scientist designing a simulated trial.

Given the user's goal and a list of available tools, produce a short trial protocol.

Return a JSON object with:
- goal: the original goal string
- planned_steps: an ordered list of high-level steps, each a short sentence in plain English

Examples of steps:
- "Run population PK simulation using simulate_pop_pk."
- "If any high Cmax values are observed, verify dose limits using safety_mcp.verify_dose_limits."
- "Search PubMed for safety data on the outlier phenotype using safety_mcp.search_pubmed."
- "Generate PK curve visualization using viz_mcp.plot_pk_curve."

Do not call tools yourself here; only think about which tools should be called later.
Output JSON only, with keys 'goal' and 'planned_steps'."""


EXECUTOR_SYSTEM_PROMPT = """You are executing a clinical trial protocol as a ReAct-style agent.

You are given:
- The user's original goal.
- An ordered list of planned steps.
- A set of tools you can call (MCP tools exposed as functions).

Tool-call contract (strict):
- When you call a tool, the tool's `arguments` MUST be a single, strictly valid JSON object.
- Use double quotes for all keys and string values. Do not use trailing commas.
- Do not put JSON inside Markdown fences in the tool call.
- If you realize your tool arguments are invalid JSON, immediately re-issue the tool call with corrected JSON arguments.

Loop until the trial is complete:
- Thought: explain briefly what you will do next (1–2 sentences).
- Action: if needed, call exactly one tool with arguments.
- Observation: read the tool result, update your mental model, and decide what to do next.
- Pivot: if observations show toxicity, unexpected values, or feasibility issues, you may deviate from the original plan (e.g., add a safety check).
- If a tool returns an "error" field (e.g. validation failure), acknowledge it, fix parameters if needed or conclude with a summary; do not ignore it.
- If a visualization tool returns "visualization_failed" (e.g. timeout or server error), treat it as non-fatal: acknowledge there is nothing to show and continue with the next step or final summary.

When you are done:
- Stop calling tools.
- Return a final natural-language summary of the trial (outliers, safety verdict, recommendation).
"""


def _get_llm_client():
    """Return OpenAI-compatible client (openai package), or None if not configured."""
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError:  # pragma: no cover - environment issue
        return None
    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _get_model() -> str:
    import os

    return os.environ.get("LLM_MODEL", "gpt-4o-mini")


async def plan(goal: str, tool_manifest: List[Dict[str, Any]]) -> TrialProtocol:
    """Planner phase: goal + tools -> TrialProtocol."""
    client = _get_llm_client()
    if not client:
        # Simple fallback: single-step protocol.
        return TrialProtocol(goal=goal, planned_steps=["Run a standard PK simulation trial."])

    tools_text = json.dumps(
        [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
            }
            for t in tool_manifest
        ],
        indent=2,
    )
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"User goal:\n{goal.strip() or 'Run a standard warfarin trial.'}\n\nAvailable tools:\n{tools_text}",
        },
    ]
    resp = client.chat.completions.create(
        model=_get_model(),
        messages=messages,
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        # Allow for optional Markdown fences.
        if "```" in text:
            import re

            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if m:
                text = m.group(1).strip()
        data = json.loads(text)
        goal_out = str(data.get("goal", goal))
        steps = [str(s) for s in data.get("planned_steps", []) if isinstance(s, str)]
        if not steps:
            steps = ["Run a standard PK simulation trial."]
        return TrialProtocol(goal=goal_out, planned_steps=steps)
    except Exception:
        # Robust fallback on any parsing or LLM error.
        return TrialProtocol(goal=goal, planned_steps=["Run a standard PK simulation trial."])


def _initial_messages(state: TrialState) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Goal:\n{state.protocol.goal}\n\n"
                f"Planned steps:\n"
                + "\n".join(f"- {s}" for s in state.protocol.planned_steps)
            ),
        },
    ]


ProgressEventCallback = Callable[[dict], None]


async def execute(
    state: TrialState,
    mcp: MCPManager,
    *,
    dry_run: bool = False,
    progress_callback: Optional[ProgressEventCallback] = None,
) -> TrialState:
    """ReAct executor: iteratively call tools and update TrialState."""
    _react_log("ReAct execute starting.", banner=True)
    client = _get_llm_client()
    if not client:
        # No LLM: mark complete with a minimal message.
        state.complete("LLM client not configured; no autonomous execution performed.")
        return state

    tools = mcp.tool_manifest()
    messages: List[Dict[str, Any]] = _initial_messages(state)
    summary_nudge_sent = False
    narrated_tool_nudge_count = 0
    bad_args_nudge_count = 0
    cycle = 0

    while not state.is_complete:
        cycle += 1
        _react_log(f"CYCLE {cycle} (steps so far: {len(state.steps)})", banner=True)
        _react_log(f"Calling LLM (model={_get_model()})...")
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )
        elapsed = time.perf_counter() - t0
        msg = resp.choices[0].message

        tool_calls = getattr(msg, "tool_calls", None) or []
        content = (msg.content or "").strip() if msg.content is not None else ""

        _react_log(f"LLM returned in {elapsed:.1f}s: tool_calls={len(tool_calls)}, content_len={len(content)}")

        if tool_calls:
            tc = tool_calls[0]
            fn = tc.function
            full_name = fn.name
            server_name = mcp.resolve_tool_server(full_name)
            try:
                args: Dict[str, Any] = json.loads(fn.arguments or "{}")
            except json.JSONDecodeError:
                bad_args_nudge_count += 1
                _react_log(
                    f"Invalid JSON tool arguments for {full_name}; nudging model to re-issue tool call."
                )
                state.add_cycle(
                    cycle_index=cycle,
                    thought=content or "(Invalid tool arguments)",
                    action="nudge",
                )
                messages.append({"role": "assistant", "content": content or ""})
                if bad_args_nudge_count <= 2:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your tool call to {full_name} had invalid JSON in `arguments`. "
                                "Re-issue the tool call NOW using the tool API with strictly valid JSON arguments "
                                "(double quotes, no trailing commas). Do not reply with only text."
                            ),
                        }
                    )
                    continue
                # Defensive fallback after repeated failures: call with empty args.
                args = {}

            _react_log(f"Calling tool: {full_name} (dry_run={dry_run})")
            # Execute the tool with the exact parameters the agent chose (from the API).
            if progress_callback:
                progress_callback(
                    {
                        "phase": "action",
                        "thought": content,
                        "tool": full_name,
                        "server": server_name,
                        "step_index": len(state.steps),
                    }
                )

            if dry_run:
                _react_log(f"Tool {full_name} skipped (dry_run).")
                observation_obj: Any = {
                    "dry_run": True,
                    "tool": full_name,
                    "arguments": args,
                }
            else:
                t_tool = time.perf_counter()
                try:
                    observation_obj = await mcp.call_tool(full_name, args)
                    elapsed_t = time.perf_counter() - t_tool
                    if isinstance(observation_obj, dict) and observation_obj.get("error"):
                        _react_log(f"Tool {full_name} returned in {elapsed_t:.1f}s (error: {observation_obj.get('error', '')})")
                    else:
                        _react_log(f"Tool {full_name} returned in {elapsed_t:.1f}s (ok)")
                except Exception as e:
                    _react_log(f"Tool {full_name} failed after {time.perf_counter() - t_tool:.1f}s: {e}")
                    observation_obj = {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "tool": full_name,
                    }

            observation_text = json.dumps(observation_obj)
            step_thought = (content or "").strip() or f"Calling {full_name} with the chosen parameters."
            state.add_step(
                thought=step_thought,
                tool_name=full_name,
                tool_args=args,
                observation=observation_text,
            )

            if progress_callback:
                progress_callback(
                    {
                        "phase": "observation",
                        "thought": content,
                        "tool": full_name,
                        "server": server_name,
                        "step_index": len(state.steps) - 1,
                    }
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": full_name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation_text,
                }
            )
            state.add_cycle(
                cycle_index=cycle,
                thought=step_thought,
                action="tool_call",
                tool_name=full_name,
                tool_args=args,
                observation=observation_text,
            )
            _react_log("Appended assistant + tool result; continuing to next cycle.")
            continue

        # No tool calls: treat as final summary only after at least one step.
        # Do not complete if the content looks like the model is describing a tool call (narrated action).
        if len(state.steps) > 0 and content:
            content_lower = content.lower()
            narrated = (
                ("action:" in content_lower and ("call" in content_lower or "invoke" in content_lower))
                or "i will call" in content_lower
                or ("let's proceed with" in content_lower and "call" in content_lower)
            )
            if narrated and narrated_tool_nudge_count < 2:
                narrated_tool_nudge_count += 1
                _react_log("Content looks like a narrated tool call; nudging to use the tool API.", banner=True)
                state.add_cycle(cycle_index=cycle, thought=content, action="nudge")
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": "You described calling a tool. Please use the function/tool API to call it now — do not reply with only text.",
                    }
                )
                continue
            _react_log(f"Completing with summary ({len(content)} chars)", banner=True)
            state.add_cycle(cycle_index=cycle, thought=content, action="complete")
            state.complete(content)
            if progress_callback:
                progress_callback(
                    {
                        "phase": "complete",
                        "thought": content,
                        "tool": None,
                        "server": None,
                        "step_index": len(state.steps),
                    }
                )
            break

        if len(state.steps) == 0:
            # Model replied with text only; may have described a tool call instead of using the API.
            _react_log("No steps yet and no tool_calls: nudging to use tool API.")
            state.add_cycle(cycle_index=cycle, thought=content or "(No response)", action="nudge")
            messages.append({"role": "assistant", "content": content or "(No response)"})
            messages.append(
                {
                    "role": "user",
                    "content": "If you need to run a simulation or use a tool, call it via the function/tool API so it runs with your chosen parameters. If you are done without using tools, give a brief final summary.",
                }
            )
            continue

        # We have steps but no tool_calls and no (or empty) content — model may have returned empty.
        # Nudge once: ask explicitly for the final summary so the user gets a proper result.
        if len(state.steps) > 0 and not content and not summary_nudge_sent:
            _react_log("Empty content after tool run: nudging for final summary.", banner=True)
            summary_nudge_sent = True
            state.add_cycle(cycle_index=cycle, thought="", action="summary_nudge")
            messages.append({"role": "assistant", "content": ""})
            messages.append(
                {
                    "role": "user",
                    "content": "The tool(s) have run. Please provide a brief final summary of the trial for the user (what was done, key results, and any recommendation).",
                }
            )
            continue

        # Defensive: empty response after some steps (and we already nudged for summary).
        _react_log("Defensive exit: empty response after steps (summary nudge already sent).", banner=True)
        state.add_cycle(
            cycle_index=cycle,
            thought=content or "Execution stopped due to empty LLM response.",
            action="defensive_exit",
        )
        state.complete(content or "Execution stopped due to empty LLM response.")
        break

    _react_log(f"ReAct loop finished. Total cycles={cycle}, steps={len(state.steps)}", banner=True)
    return state

