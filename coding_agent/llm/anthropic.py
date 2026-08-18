"""Anthropic provider adapter built on the official anthropic SDK."""
from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from config import LLMConfig

from .base import LLMError, LLMResponse, Message, ToolCall, ToolSchema


class AnthropicProvider:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = anthropic.Anthropic(
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=config.base_url or None,
            timeout=config.timeout,
        )

    def chat(
        self,
        messages: list[Message] | tuple[Message, ...],
        tools: list[ToolSchema] | tuple[ToolSchema, ...] | None = None,
        *,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        system_parts: list[str] = []
        api_messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []
        cache_marker = {"type": "ephemeral"}

        def with_cache_control(content_blocks: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
            if enabled and content_blocks:
                content_blocks[-1]["cache_control"] = cache_marker
            return content_blocks

        def flush_tool_results() -> None:
            if pending_tool_results:
                api_messages.append({"role": "user", "content": pending_tool_results[:]})
                pending_tool_results.clear()

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue
            if message.role == "tool":
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": message.content or "(empty result)",
                        "is_error": bool(message.is_error),
                    }
                )
                continue

            # A tool result must directly follow an assistant tool_use turn. If a
            # non-tool message appears while results are pending, close the group
            # first (should not normally happen with well-formed context).
            flush_tool_results()

            if message.role == "assistant":
                if message.raw_content:
                    # Replay the provider's blocks verbatim (thinking with its
                    # signature, text, tool_use) so reasoning survives the loop.
                    content_blocks = [dict(block) for block in message.raw_content]
                else:
                    content_blocks = []
                    if message.content:
                        content_blocks.append({"type": "text", "text": message.content})
                    for call in message.tool_calls:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.name,
                                "input": call.arguments or {},
                            }
                        )
                content_blocks = with_cache_control(content_blocks, message.cache_control)
                api_messages.append({"role": "assistant", "content": content_blocks or [{"type": "text", "text": ""}]})
            else:
                content: Any = message.content or ""
                if message.cache_control:
                    content = with_cache_control([{"type": "text", "text": content}], True)
                api_messages.append({"role": "user", "content": content})

        flush_tool_results()

        system_blocks: list[dict[str, Any]] = []
        for part in system_parts:
            if part.strip():
                system_blocks.append({"type": "text", "text": part.strip()})
        if system_blocks:
            system_blocks[-1]["cache_control"] = cache_marker
        effort = reasoning_effort if reasoning_effort is not None else self.config.reasoning_effort
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": api_messages,
        }
        if effort:
            if effort == "none":
                # Disable thinking entirely; the budget then all goes to text.
                kwargs["thinking"] = {"type": "disabled"}
            else:
                # DeepSeek Anthropic-format knob: effort also controls thinking
                # intensity (low/medium map to high on DeepSeek models).
                kwargs["output_config"] = {"effort": effort}
        if system_blocks:
            kwargs["system"] = system_blocks
        if tools:
            tool_params: list[dict[str, Any]] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ]
            if tool_params:
                tool_params[-1]["cache_control"] = cache_marker
            kwargs["tools"] = tool_params

        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise self._translate_error(exc) from exc

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_blocks: list[dict[str, Any]] = []
        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "thinking":
                # Keep the reasoning text for display and the block (with its
                # signature) verbatim so it can be replayed unchanged.
                thinking_parts.append(getattr(block, "thinking", "") or "")
                raw_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": getattr(block, "thinking", "") or "",
                        "signature": getattr(block, "signature", "") or "",
                    }
                )
            elif block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
                raw_blocks.append({"type": "text", "text": getattr(block, "text", "") or ""})
            elif block_type == "tool_use":
                raw_input = getattr(block, "input", {}) or {}
                if isinstance(raw_input, str):
                    try:
                        raw_input = json.loads(raw_input)
                    except json.JSONDecodeError:
                        raw_input = {}
                arguments = raw_input if isinstance(raw_input, dict) else {}
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", "") or "",
                        name=getattr(block, "name", "") or "",
                        arguments=dict(arguments),
                    )
                )
                raw_blocks.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", "") or "",
                        "name": getattr(block, "name", "") or "",
                        "input": dict(arguments),
                    }
                )

        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", None),
            }

        return LLMResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            stop_reason=getattr(response, "stop_reason", None),
            usage=usage,
            thinking="\n".join(thinking_parts) or None,
            raw_content=raw_blocks or None,
        )

    @staticmethod
    def _translate_error(exc: anthropic.APIError) -> LLMError:
        retryable = isinstance(
            exc,
            (
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.InternalServerError,
            ),
        )
        if isinstance(exc, anthropic.APIStatusError):
            retryable = exc.status_code in (408, 409, 429, 500, 502, 503, 504)
        return LLMError(f"Anthropic API error: {exc}", retryable=retryable)
