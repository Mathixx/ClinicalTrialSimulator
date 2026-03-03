from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Step(BaseModel):
    """One tool-call step in the ReAct loop."""

    thought: str = Field("", description="Why the agent chose this action.")
    tool_name: Optional[str] = Field(
        default=None, description="Name of the tool that was called (if any)."
    )
    tool_args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool."
    )
    observation: str = Field(
        "", description="Serialized observation returned from the tool."
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this step was created."
    )


class ReActCycle(BaseModel):
    """One full ReAct cycle (every LLM turn + what we did)."""

    cycle_index: int = Field(..., description="1-based cycle number.")
    thought: str = Field("", description="LLM content for this turn.")
    action: str = Field(
        ...,
        description="One of: tool_call, nudge, summary_nudge, complete, defensive_exit.",
    )
    tool_name: Optional[str] = Field(default=None, description="Tool called (if action=tool_call).")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments (if action=tool_call).")
    observation: str = Field("", description="Tool result (if action=tool_call).")


class TrialProtocol(BaseModel):
    """Planner output: high-level protocol for the trial."""

    goal: str = Field(..., description="Original natural-language goal from the user.")
    planned_steps: List[str] = Field(
        default_factory=list,
        description="High-level steps the planner thinks should be executed in order.",
    )


class TrialState(BaseModel):
    """Evolving state for a single trial run."""

    protocol: TrialProtocol
    steps: List[Step] = Field(default_factory=list)
    cycles: List[ReActCycle] = Field(
        default_factory=list,
        description="Every ReAct cycle (nudges, tool calls, completions) for the Developers view.",
    )
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured results accumulated during execution.",
    )
    is_complete: bool = Field(
        False, description="Whether the agent considers the trial finished."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Final narrative summary from the agent once the trial is complete.",
    )

    def add_step(
        self,
        thought: str,
        tool_name: Optional[str],
        tool_args: Dict[str, Any],
        observation: str,
    ) -> Step:
        step = Step(
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args,
            observation=observation,
        )
        self.steps.append(step)
        return step

    def add_cycle(
        self,
        cycle_index: int,
        thought: str,
        action: str,
        *,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        observation: str = "",
    ) -> ReActCycle:
        cycle = ReActCycle(
            cycle_index=cycle_index,
            thought=thought,
            action=action,
            tool_name=tool_name,
            tool_args=tool_args or {},
            observation=observation,
        )
        self.cycles.append(cycle)
        return cycle

    def complete(self, summary: str) -> None:
        self.is_complete = True
        self.summary = summary


class ToolDef(BaseModel):
    """Tool metadata exposed to the planner / executor."""

    name: str
    server: str
    description: str
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-schema-like parameters definition for LLM tool calling.",
    )

