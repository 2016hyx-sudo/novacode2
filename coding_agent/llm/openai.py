"""OpenAI provider adapter built on the official openai SDK."""
from __future__ import annotations

import json
import os
from typing import Any

import openai

from config import LLMConfig

from .base import LLMError, LLMResponse, Message, StreamCallback, StreamChunk, ToolCall, ToolSchema


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
        on_chunk: StreamCallback | None = None,
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

        if on_chunk is not None and getattr(self.config, "stream", True):
            return self._chat_stream(kwargs, on_chunk)
        return self._chat_sync(kwargs)

    def _chat_sync(self, kwargs: dict[str, Any]) -> LLMResponse:
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
        reasoning = getattr(message, "reasoning_content", None)
        raw_content: dict[str, Any] | None = None
        if message.tool_calls or reasoning is not None:
            raw_content = {"role": "assistant"}
            if message.content is not None:
                raw_content["content"] = message.content
            if message.tool_calls:
                raw_content["tool_calls"] = [item.model_dump() for item in message.tool_calls]
            # DeepSeek thinking mode requires reasoning_content echoed back
            # verbatim — even an empty string — or the next request returns a
            # 400. Presence check, not truthiness, so "" is preserved.
            if reasoning is not None:
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

    def _chat_stream(self, kwargs: dict[str, Any], on_chunk: StreamCallback) -> LLMResponse:
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream_kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = self.client.chat.completions.create(**stream_kwargs)
        except openai.OpenAIError as exc:
            raise self._translate_error(exc) from exc

        if not hasattr(stream, "__iter__"):
            # Fallback if mock returns non-iterable object
            return self._chat_sync(kwargs)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls_map: dict[int, dict[str, Any]] = {}
        stop_reason: str | None = None
        usage: dict[str, Any] = {}

        try:
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    u = chunk.usage
                    details = getattr(u, "prompt_tokens_details", None)
                    usage = {
                        "prompt_tokens": getattr(u, "prompt_tokens", None),
                        "completion_tokens": getattr(u, "completion_tokens", None),
                        "total_tokens": getattr(u, "total_tokens", None),
                        "cached_tokens": getattr(details, "cached_tokens", None),
                    }

                choices = getattr(chunk, "choices", [])
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None) is not None:
                    stop_reason = choice.finish_reason

                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                delta_text = getattr(delta, "content", None)
                delta_thinking = getattr(delta, "reasoning_content", None)

                if delta_text:
                    text_parts.append(delta_text)
                if delta_thinking:
                    thinking_parts.append(delta_thinking)

                delta_tc = getattr(delta, "tool_calls", None)
                if delta_tc:
                    for tc in delta_tc:
                        idx = getattr(tc, "index", 0)
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": getattr(tc, "id", "") or "",
                                "name": "",
                                "arguments": "",
                            }
                        if getattr(tc, "id", None):
                            tool_calls_map[idx]["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                tool_calls_map[idx]["name"] += fn.name
                            if getattr(fn, "arguments", None):
                                tool_calls_map[idx]["arguments"] += fn.arguments

                if delta_text or delta_thinking:
                    on_chunk(
                        StreamChunk(
                            delta_text=delta_text,
                            delta_thinking=delta_thinking,
                            stop_reason=stop_reason,
                        )
                    )
        except openai.OpenAIError as exc:
            raise self._translate_error(exc) from exc

        full_text = "".join(text_parts) if text_parts else None
        full_thinking = "".join(thinking_parts) if thinking_parts else None

        tool_calls: list[ToolCall] = []
        raw_tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_calls_map.keys()):
            tc_data = tool_calls_map[idx]
            call_id = tc_data["id"] or f"call_{idx}"
            call_name = tc_data["name"]
            raw_args = tc_data["arguments"]
            try:
                parsed_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed_args = {}
            if not isinstance(parsed_args, dict):
                parsed_args = {}
            tool_calls.append(ToolCall(id=call_id, name=call_name, arguments=parsed_args))
            raw_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call_name,
                        "arguments": raw_args or "{}",
                    },
                }
            )

        raw_content: dict[str, Any] | None = None
        if tool_calls or full_thinking is not None:
            raw_content = {"role": "assistant"}
            if full_text is not None:
                raw_content["content"] = full_text
            if raw_tool_calls:
                raw_content["tool_calls"] = raw_tool_calls
            if full_thinking is not None:
                raw_content["reasoning_content"] = full_thinking

        return LLMResponse(
            text=full_text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            thinking=full_thinking,
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
