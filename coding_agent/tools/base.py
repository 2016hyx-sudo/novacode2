"""Tool protocol and shared tool result types."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ToolErrorKind(str, Enum):
    INVALID_ARGUMENTS = "invalid_arguments"
    NOT_FOUND = "not_found"
    WORKSPACE_VIOLATION = "workspace_violation"
    SHELL_TIMEOUT = "shell_timeout"
    SHELL_FAILED = "shell_failed"
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    UNEXPECTED = "unexpected"


# Error kinds worth retrying with identical arguments. Invalid arguments and
# workspace violations are deliberately excluded.
RETRYABLE_KINDS = {ToolErrorKind.SHELL_TIMEOUT, ToolErrorKind.TIMEOUT, ToolErrorKind.TRANSIENT}


class ToolError(Exception):
    """A structured tool error.

    Tools should raise this for expected failures so ToolExecutor can format
    consistent feedback and decide whether retrying is useful.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: ToolErrorKind = ToolErrorKind.UNEXPECTED,
        retryable: bool | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.retryable = kind in RETRYABLE_KINDS if retryable is None else retryable
        self.hint = hint


@dataclass
class ToolResult:
    """Structured result returned to the AgentLoop and then to the LLM."""

    success: bool
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: str = "", **metadata: Any) -> ToolResult:
        return cls(success=True, output=output, metadata=dict(metadata))

    @classmethod
    def fail(
        cls,
        error: str,
        *,
        output: str = "",
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> ToolResult:
        merged = dict(metadata or {})
        merged.update(extra)
        return cls(success=False, output=output, error=error, metadata=merged)


@dataclass
class ToolCallResult:
    """The result of executing one model-requested tool call."""

    call_id: str
    name: str
    result: ToolResult
    retries_used: int = 0
    duration_ms: int = 0


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and never raise for expected tool-level failures."""
        ...


@runtime_checkable
class AgentRunOutcome(Protocol):
    """Structural type returned by a subagent run."""

    text: str
    status: str
    steps_used: int
    tool_calls_used: int


class FunctionTool:
    """Small adapter that turns a plain function into a Tool."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., ToolResult],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._func = func

    def execute(self, **kwargs: Any) -> ToolResult:
        return self._func(**kwargs)


def format_tool_result_for_llm(name: str, result: ToolResult) -> str:
    """Render a ToolResult as the content of a tool-result message.

    This is the structured feedback the model sees after every tool call.
    """
    if result.success:
        return result.output or "(tool completed successfully with no output)"

    lines = [f"Tool {name!r} failed: {result.error or 'unknown error'}"]
    if result.output:
        lines.append(f"Output captured before failure:\n{result.output}")
    hint = result.metadata.get("hint")
    if hint:
        lines.append(f"Hint: {hint}")
    kind = result.metadata.get("kind")
    if kind:
        lines.append(f"Error kind: {kind}")
    return "\n".join(lines)
