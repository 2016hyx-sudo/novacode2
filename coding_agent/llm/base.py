"""Provider-neutral data structures shared by the whole harness."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            arguments=dict(data.get("arguments") or {}),
        )


@dataclass
class ToolSchema:
    """JSON Schema description of a tool exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """Unified conversation message.

    ``tool_calls`` is used for assistant tool-call messages.
    ``tool_call_id`` is used for tool-result messages.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        if self.is_error:
            data["is_error"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=data.get("role", "user"),  # type: ignore[arg-type]
            content=data.get("content"),
            tool_calls=[ToolCall.from_dict(item) for item in data.get("tool_calls", [])],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            is_error=bool(data.get("is_error", False)),
        )


@dataclass
class LLMResponse:
    """Provider-neutral model response."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """Raised by providers for API/transport failures.

    ``retryable`` marks transient failures (timeout, 429, 5xx, connection) that
    the AgentLoop may retry a limited number of times.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@runtime_checkable
class LLMProvider(Protocol):
    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
    ) -> LLMResponse:
        """Send messages and optional tool schemas, return a unified response."""
        ...
