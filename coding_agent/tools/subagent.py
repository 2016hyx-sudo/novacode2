"""Subagent tool.

The tool does not import AgentLoop. The composition root injects a callback
that creates and runs a child AgentLoop with an independent context, a smaller
max_steps budget, and (by default) no nested subagent tool.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, ClassVar

from ..runtime.constraints import ConstraintError, RunBudget
from ..runtime.trace import TraceWriter
from .base import AgentRunOutcome, ToolErrorKind, ToolResult

SubagentRunner = Callable[[str, int], AgentRunOutcome]


class SubagentTool:
    name = "subagent"
    description = (
        "Delegate a small, clearly bounded coding task (analysis, code search, "
        "bug investigation, or a local change proposal) to a child coding agent. "
        "The child has its own conversation context and returns one final summary. "
        "Use it for isolated subproblems; do not use it for trivial single-tool actions."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Exact task for the subagent. Must be self-contained.",
            },
            "max_steps": {
                "type": "integer",
                "description": "Optional max LLM steps for the subagent (clamped by harness).",
            },
        },
        "required": ["task"],
    }

    def __init__(
        self,
        *,
        run_subagent: SubagentRunner,
        budget: RunBudget,
        trace: TraceWriter | None = None,
        default_max_steps: int = 5,
        max_steps_limit: int = 20,
        current_depth: int = 0,
    ) -> None:
        self._run_subagent = run_subagent
        self._budget = budget
        self._trace = trace
        self.default_max_steps = max(1, default_max_steps)
        self.max_steps_limit = max(1, max_steps_limit)
        self.current_depth = current_depth

    def execute(self, task: str, max_steps: int | None = None) -> ToolResult:
        if not task or not task.strip():
            return ToolResult.fail(
                "subagent task must not be empty",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )
        if max_steps is not None:
            try:
                steps = int(max_steps)
            except (TypeError, ValueError):
                steps = self.default_max_steps
        else:
            steps = self.default_max_steps
        steps = max(1, min(steps, self.max_steps_limit))

        try:
            self._budget.reserve_subagent()
        except ConstraintError as exc:
            return ToolResult.fail(
                str(exc),
                metadata={"kind": ToolErrorKind.WORKSPACE_VIOLATION.value, "hint": "Do not create more subagents."},
            )

        self._emit("subagent_start", task=task, max_steps=steps, depth=self.current_depth + 1)
        started = time.monotonic()
        try:
            outcome = self._run_subagent(task.strip(), steps)
        except Exception as exc:
            self._emit("subagent_end", ok=False, error=str(exc), depth=self.current_depth + 1)
            return ToolResult.fail(
                f"Subagent crashed: {exc}",
                metadata={"kind": ToolErrorKind.UNEXPECTED.value},
            )

        duration_ms = int((time.monotonic() - started) * 1000)
        text = getattr(outcome, "text", "") or ""
        status = getattr(outcome, "status", "failed")
        success = status == "completed"
        self._emit(
            "subagent_end",
            ok=success,
            status=status,
            steps_used=getattr(outcome, "steps_used", 0),
            tool_calls_used=getattr(outcome, "tool_calls_used", 0),
            duration_ms=duration_ms,
            depth=self.current_depth + 1,
        )
        metadata = {
            "subagent_status": status,
            "subagent_steps": getattr(outcome, "steps_used", 0),
            "subagent_tool_calls": getattr(outcome, "tool_calls_used", 0),
            "subagent_depth": self.current_depth + 1,
        }
        report = getattr(outcome, "structured_report", None)
        if isinstance(report, dict):
            metadata["structured_report"] = report
        return ToolResult(
            success=success,
            output=text or "(subagent returned no output)",
            error=None if success else f"Subagent ended with status {status!r}",
            metadata=metadata,
        )

    def _emit(self, event_type: str, **data: Any) -> None:
        if self._trace is not None:
            self._trace.emit(event_type, **data)
