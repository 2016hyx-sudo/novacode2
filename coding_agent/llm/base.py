"""Provider-neutral data structures shared by the whole harness."""
from __future__ import annotations

from collections.abc import Callable, Sequence
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
    cache_control: bool = False
    # Verbatim assistant content blocks as returned by the provider (thinking,
    # text, tool_use with any signature/opaque data). Replayed unchanged so
    # reasoning survives multi-turn tool loops. Provider-neutral: each adapter
    # decides its own block shape.
    raw_content: list[dict[str, Any]] | None = None

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
        if self.cache_control:
            data["cache_control"] = True
        if self.raw_content is not None:
            data["raw_content"] = self.raw_content
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
            cache_control=bool(data.get("cache_control", False)),
            raw_content=data.get("raw_content"),
        )


def raw_content_reasoning_blocks(message: Message) -> list[Any]:
    """Extra reasoning/thinking parts carried only by ``Message.raw_content``.

    ``raw_content`` is provider-shaped: Anthropic adapters store a list of
    content blocks (thinking/text/tool_use); OpenAI adapters store a single
    assistant dict (content + reasoning_content + tool_calls). The text and
    tool_use parts duplicate ``Message.content`` / ``Message.tool_calls`` and
    are already counted elsewhere, so only the thinking/reasoning parts should
    be added to token estimates.
    """
    raw = message.raw_content
    if not raw:
        return []
    if isinstance(raw, list):
        return [block for block in raw if isinstance(block, dict) and block.get("type") == "thinking"]
    if isinstance(raw, dict):
        reasoning = raw.get("reasoning_content")
        if reasoning is None:
            return []
        return [reasoning]
    return []


@dataclass
class LLMResponse:
    """Provider-neutral model response."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    # Joined reasoning/thinking text, when the provider exposes it separately
    # from the answer (e.g. Anthropic thinking blocks, OpenAI reasoning_content).
    thinking: str | None = None
    # Verbatim assistant content blocks from the response, replayed unchanged
    # on the next turn (see Message.raw_content).
    raw_content: list[dict[str, Any]] | None = None
    # Populated by AgentLoop/FoldEngine measurement wrappers, not providers.
    request_id: str | None = None
    normalized_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A streaming chunk emitted during LLM response generation."""

    delta_text: str | None = None
    delta_thinking: str | None = None
    tool_call_chunks: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None


StreamCallback = Callable[[StreamChunk], None]


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
        *,
        reasoning_effort: str | None = None,
        on_chunk: StreamCallback | None = None,
    ) -> LLMResponse:
        """Send messages and optional tool schemas, return a unified response.

        ``reasoning_effort`` overrides the provider's configured effort for this
        call ("none" | "low" | "high" | "max"); None defers to the config.
        ``on_chunk`` receives real-time StreamChunk events when streaming is active.
        """
        ...
