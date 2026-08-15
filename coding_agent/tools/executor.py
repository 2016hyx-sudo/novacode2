"""ToolExecutor: validation, timeout, retry and structured error feedback."""
from __future__ import annotations

import threading
import time
from typing import Any

from ..llm.base import ToolCall
from ..runtime.constraints import ConstraintError, RunBudget
from ..runtime.trace import TraceWriter
from .base import (
    RETRYABLE_KINDS,
    Tool,
    ToolCallResult,
    ToolError,
    ToolErrorKind,
    ToolResult,
)
from .registry import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_retries: int = 1,
        tool_timeout: float = 120.0,
        max_output_chars: int = 20_000,
        trace: TraceWriter | None = None,
        budget: RunBudget | None = None,
    ) -> None:
        self.registry = registry
        self.budget = budget
        self.max_retries = max(0, max_retries)
        self.tool_timeout = max(1.0, tool_timeout)
        self.max_output_chars = max(1_000, max_output_chars)
        self.trace = trace

    def execute_all(self, calls: list[ToolCall]) -> list[ToolCallResult]:
        return [self.execute(call) for call in calls]

    def execute(self, call: ToolCall) -> ToolCallResult:
        started = time.monotonic()
        self._emit("tool_call", call_id=call.id, name=call.name, arguments=call.arguments)

        if self.budget is not None:
            try:
                self.budget.reserve_tool_call()
            except ConstraintError as exc:
                result = ToolResult.fail(
                    str(exc),
                    metadata={
                        "kind": ToolErrorKind.WORKSPACE_VIOLATION.value,
                        "hint": "Return a final answer explaining what is done and what remains.",
                    },
                )
                return self._finish(call, result, started, retries_used=0)

        tool = self.registry.get(call.name)
        if tool is None:
            result = ToolResult.fail(
                f"Unknown tool: {call.name}. Available tools: {', '.join(self.registry.names()) or '(none)'}",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )
            return self._finish(call, result, started, retries_used=0)

        if not isinstance(call.arguments, dict):
            result = ToolResult.fail(
                f"Tool arguments must be a JSON object, got {type(call.arguments).__name__}",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )
            return self._finish(call, result, started, retries_used=0)

        retries_used = 0
        while True:
            try:
                result = self._call_with_timeout(tool, call.arguments)
            except ToolError as exc:
                result = ToolResult.fail(
                    exc.message,
                    metadata={"kind": exc.kind.value, "hint": exc.hint or ""},
                )
            except TypeError as exc:
                result = ToolResult.fail(
                    f"Invalid arguments for tool {call.name!r}: {exc}",
                    metadata={
                        "kind": ToolErrorKind.INVALID_ARGUMENTS.value,
                        "hint": "Check the tool schema and provide the required arguments.",
                    },
                )
            except Exception as exc:
                result = ToolResult.fail(
                    f"Unexpected error in tool {call.name!r}: {exc}",
                    metadata={"kind": ToolErrorKind.UNEXPECTED.value},
                )

            retryable = (
                not result.success
                and ToolErrorKind(result.metadata.get("kind", ToolErrorKind.UNEXPECTED.value))
                in RETRYABLE_KINDS
            )
            if retryable and retries_used < self.max_retries:
                retries_used += 1
                self._emit(
                    "tool_retry",
                    call_id=call.id,
                    name=call.name,
                    attempt=retries_used,
                    error=result.error,
                )
                time.sleep(0.5 * retries_used)
                continue

            return self._finish(call, result, started, retries_used=retries_used)

    def _call_with_timeout(self, tool: Tool, arguments: dict[str, Any]) -> ToolResult:
        holder: dict[str, Any] = {}

        def target() -> None:
            try:
                holder["result"] = tool.execute(**arguments)
            except BaseException as exc:  # stored and re-raised on the caller thread
                holder["exc"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(self.tool_timeout)
        if thread.is_alive():
            raise ToolError(
                f"Tool {tool.name!r} timed out after {self.tool_timeout:.1f}s",
                kind=ToolErrorKind.TIMEOUT,
                retryable=True,
            )
        if "exc" in holder:
            raise holder["exc"]
        return holder.get("result", ToolResult.fail("Tool returned no result"))

    def _finish(
        self,
        call: ToolCall,
        result: ToolResult,
        started: float,
        *,
        retries_used: int,
    ) -> ToolCallResult:
        duration_ms = int((time.monotonic() - started) * 1000)
        if result.output:
            result.output = self._truncate(result.output)
        final = ToolCallResult(
            call_id=call.id,
            name=call.name,
            result=result,
            retries_used=retries_used,
            duration_ms=duration_ms,
        )
        self._emit(
            "tool_result",
            call_id=call.id,
            name=call.name,
            success=result.success,
            output_preview=result.output[:500],
            error=result.error,
            metadata=result.metadata,
            retries_used=retries_used,
            duration_ms=duration_ms,
        )
        return final

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        omitted = len(text) - self.max_output_chars
        return text[: self.max_output_chars] + f"\n... [truncated, {omitted} chars omitted]"

    def _emit(self, event_type: str, **data: Any) -> None:
        if self.trace is not None:
            self.trace.emit(event_type, **data)
