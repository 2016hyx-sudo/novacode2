"""The memory object produced by NovaCode's structured-context construction.

A ``NovaCodeMemory`` mirrors what the live harness keeps in its prompt: Task
State (findings, decisions, completed work), Tool State (reusable experience)
and the never-folded Recent Trajectory groups.  Retrieval reads from all three
via :mod:`ama_bench.retrieve`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coding_agent.structured_context.models import TaskState, ToolState, Trajectory


@dataclass
class NovaCodeMemory:
    """Immutable-by-convention memory: the builder does not mutate it later."""

    task_state: TaskState
    tool_state: ToolState
    trajectory: Trajectory
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_state": self.task_state.to_dict(),
            "tool_state": self.tool_state.to_dict(),
            "trajectory": self.trajectory.to_dict(),
            "stats": dict(self.stats),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NovaCodeMemory:
        return cls(
            task_state=TaskState.from_dict(data.get("task_state") or {}),
            tool_state=ToolState.from_dict(data.get("tool_state") or {}),
            trajectory=Trajectory.from_dict(data.get("trajectory") or {}),
            stats=dict(data.get("stats") or {}),
        )
