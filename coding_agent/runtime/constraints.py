"""Shared runtime budget counters (session-wide, including subagents)."""
from __future__ import annotations

from dataclasses import dataclass


class ConstraintError(Exception):
    pass


@dataclass
class RunBudget:
    max_tool_calls: int
    max_subagents: int
    tool_calls_used: int = 0
    subagents_used: int = 0

    def reserve_tool_call(self) -> None:
        if self.tool_calls_used >= self.max_tool_calls:
            raise ConstraintError(
                f"Maximum tool calls reached ({self.max_tool_calls}). "
                "Return a final answer explaining what remains unfinished."
            )
        self.tool_calls_used += 1

    def reserve_subagent(self) -> None:
        if self.subagents_used >= self.max_subagents:
            raise ConstraintError(
                f"Maximum subagents reached ({self.max_subagents}). "
                "Do the remaining work directly in the main agent."
            )
        self.subagents_used += 1
