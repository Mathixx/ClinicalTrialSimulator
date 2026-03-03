from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from clinical_trial_simulator.agent import execute, plan
from clinical_trial_simulator.schemas import TrialProtocol, TrialState


_KNOWN_SERVERS = ("pk_sim_mcp", "safety_mcp", "recruitment_mcp", "viz_mcp")


class MockMCPManager:
    """Minimal stand-in for MCPManager used in agent tests."""

    def __init__(self, tool_responses: Dict[str, Any]) -> None:
        self._tool_responses = tool_responses
        self.called: List[str] = []

    def tool_manifest(self) -> List[Dict[str, Any]]:
        # Expose tools that match keys in tool_responses (OpenAI-safe: server_tool).
        out: List[Dict[str, Any]] = []
        for full_name in self._tool_responses:
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": full_name,
                        "description": f"Mock tool {full_name}",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": True,
                        },
                    },
                }
            )
        return out

    def resolve_tool_server(self, full_name: str) -> str:
        for s in sorted(_KNOWN_SERVERS, key=len, reverse=True):
            if full_name.startswith(s + "_"):
                return s
        return ""

    async def __aenter__(self) -> "MockMCPManager":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None

    async def call_tool(self, full_name: str, arguments: Dict[str, Any]) -> Any:
        self.called.append(full_name)
        return self._tool_responses.get(full_name, {})


@pytest.mark.anyio
async def test_execute_dry_run_records_steps_without_calling_tools(monkeypatch):
    protocol = TrialProtocol(
        goal="Test dry-run behaviour.",
        planned_steps=["Call a mock PK tool."],
    )
    state = TrialState(protocol=protocol)
    mcp = MockMCPManager({"pk_sim_mcp_simulate_pop_pk": {"ok": True}})

    # Monkeypatch LLM client to return a tool call followed by a final summary.
    class Msg:
        def __init__(self, content: str, tool_calls: Any = None) -> None:
            self.content = content
            self.tool_calls = tool_calls or []

    class Choice:
        def __init__(self, message: Msg) -> None:
            self.message = message

    class Resp:
        def __init__(self, msg: Msg) -> None:
            self.choices = [Choice(msg)]

    calls: List[int] = []

    def fake_create(*_args, **_kwargs):
        # First call returns a tool invocation, second call returns final text.
        if not calls:
            calls.append(1)
            tool_call = type(
                "ToolCall",
                (),
                {
                    "id": "1",
                    "function": type(
                        "Fn",
                        (),
                        {"name": "pk_sim_mcp_simulate_pop_pk", "arguments": json.dumps({})},
                    )(),
                },
            )()
            return Resp(Msg("Run population PK.", [tool_call]))
        return Resp(Msg("Final summary."))

    class DummyClient:
        class chat:
            class completions:
                create = staticmethod(fake_create)

    from clinical_trial_simulator import agent as agent_mod

    monkeypatch.setattr(agent_mod, "_get_llm_client", lambda: DummyClient())

    out_state = await execute(state, mcp, dry_run=True)

    assert out_state.is_complete is True
    assert out_state.summary == "Final summary."
    # One step recorded with a dry-run observation.
    assert len(out_state.steps) == 1
    obs = json.loads(out_state.steps[0].observation)
    assert obs.get("dry_run") is True
    # Under dry_run, underlying MCP manager should still be called by execute(),
    # but responses are tagged as dry_run in the observation.
    assert mcp.called == []  # execute() bypasses MockMCPManager in dry_run


