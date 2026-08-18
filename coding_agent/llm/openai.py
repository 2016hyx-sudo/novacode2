"""OpenAI provider adapter built on the official openai SDK."""
from __future__ import annotations

import json
import os
from typing import Any

import openai

from config import LLMConfig

from .base import LLMError, LLMResponse, Message, ToolCall, ToolSchema


class OpenAIProvider:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
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
        payload_messages = [self._to_openai_message(msg) for msg in messages]
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": payload_messages,
            "max_tokens": self.config.max_tokens,
        }
        effort = reasoning_effort if reasoning_effort is not None else self.config.reasoning_effort
        if effort:
            if effort == "none":
                # DeepSeek: thinking on/off lives in extra_body for Chat Completions.
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            else:
                kwargs["reasoning_effort"] = effort
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        try:
            response = self.client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            raise self._translate_error(exc) from exc

        choice = response.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []
        for item in message.tool_calls or []:
            try:
                arguments = json.loads(item.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(
                ToolCall(id=item.id or "", name=item.function.name or "", arguments=arguments)
            )

        # Keep the assistant message verbatim (content, tool_calls,
        # reasoning_content) so reasoning is replayed in multi-turn tool loops.
        raw_content: dict[str, Any] | None = None
        if message.tool_calls or getattr(message, "reasoning_content", None):
            raw_content = {"role": "assistant"}
            if message.content is not None:
                raw_content["content"] = message.content
            if message.tool_calls:
                raw_content["tool_calls"] = [item.model_dump() for item in message.tool_calls]
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                raw_content["reasoning_content"] = reasoning

        usage = {}
        if response.usage is not None:
            details = getattr(response.usage, "prompt_tokens_details", None)
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
                "cached_tokens": getattr(details, "cached_tokens", None),
            }

        return LLMResponse(
            text=message.content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            usage=usage,
            thinking=getattr(message, "reasoning_content", None),
            raw_content=raw_content,
        )

    @staticmethod
    def _to_openai_message(message: Message) -> dict[str, Any]:
        if message.role == "assistant" and message.raw_content:
            # Replay the provider's assistant message verbatim, keeping
            # reasoning_content so tool-loop reasoning survives.
            return dict(message.raw_content)
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id or "",
                "content": message.content or "",
            }
        if message.role == "assistant" and message.tool_calls:
            return {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        return {"role": message.role, "content": message.content or ""}

    @staticmethod
    def _translate_error(exc: openai.OpenAIError) -> LLMError:
        retryable = isinstance(
            exc,
            (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
            ),
        )
        if isinstance(exc, openai.APIStatusError):
            retryable = exc.status_code in (408, 409, 429, 500, 502, 503, 504)
        return LLMError(f"OpenAI API error: {exc}", retryable=retryable)
