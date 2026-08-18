"""Conversation context management with a simple token estimate and trimming."""
from __future__ import annotations

import json
from typing import Any

from ..llm.base import Message, ToolCall
from ..tools.base import ToolResult, format_tool_result_for_llm


class ContextManager:
    def __init__(
        self,
        system_prompt: str,
        *,
        messages: list[Message] | None = None,
        max_context_tokens: int = 100_000,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_context_tokens = max(1_000, max_context_tokens)
        self._truncated = False
        if messages:
            self._messages = list(messages)
            if not any(msg.role == "system" for msg in self._messages):
                self._messages.insert(0, Message(role="system", content=system_prompt))
        else:
            self._messages = [Message(role="system", content=system_prompt)]
        self._trim()

    @property
    def messages(self) -> list[Message]:
        return self._messages

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    def add_user(self, content: str) -> None:
        self.add(Message(role="user", content=content))

    def add_system(self, content: str) -> None:
        self.add(Message(role="system", content=content))

    def add_assistant(
        self,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
        *,
        raw_content: list[dict[str, Any]] | None = None,
    ) -> None:
        self.add(
            Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls or [],
                raw_content=raw_content,
            )
        )

    def add_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        self.add(
            Message(
                role="tool",
                content=format_tool_result_for_llm(call.name, result),
                tool_call_id=call.id,
                name=call.name,
                is_error=not result.success,
            )
        )

    PLAN_PREFIX = "Task plan (keep this in mind, follow it, update when needed):"

    def add_plan(self, plan: object) -> None:
        self.set_plan(plan)

    def set_plan(self, plan: object) -> None:
        render = getattr(plan, "render", None)
        text = render() if callable(render) else str(plan)
        message = Message(role="system", content=f"{self.PLAN_PREFIX}\n{text}")
        # Replace an existing plan message instead of accumulating plan copies.
        self._messages = [msg for msg in self._messages if not (msg.role == "system" and (msg.content or "").startswith(self.PLAN_PREFIX))]
        insert_at = 1
        while insert_at < len(self._messages) and self._messages[insert_at].role == "system":
            insert_at += 1
        self._messages.insert(min(insert_at, len(self._messages)), message)

    @staticmethod
    def estimate_tokens(messages: list[Message]) -> int:
        """Rough char-based token estimate. Good enough for trimming, not billing."""
        total = 0
        for message in messages:
            text = message.content or ""
            total += len(text) // 4 + 1
            for call in message.tool_calls:
                try:
                    total += len(json.dumps(call.arguments, ensure_ascii=False)) // 4 + 1
                except TypeError:
                    total += 4
            # Replayed raw blocks (thinking, tool_use) are also input tokens.
            if message.raw_content:
                for block in message.raw_content:
                    try:
                        total += len(json.dumps(block, ensure_ascii=False)) // 4 + 1
                    except TypeError:
                        total += 4
        return total

    def _trim(self) -> None:
        if self.estimate_tokens(self._messages) <= self.max_context_tokens:
            return

        notice = Message(
            role="system",
            content="[Context truncated] Earlier conversation was removed because the context budget was exceeded.",
        )
        insert_at = 1
        if len(self._messages) > insert_at and self._messages[insert_at].content != notice.content:
            self._messages.insert(insert_at, notice)
            self._truncated = True

        # Drop the oldest non-system messages, preserving assistant tool-call
        # groups together with their tool results so providers stay well-formed.
        while self.estimate_tokens(self._messages) > self.max_context_tokens and len(self._messages) > 3:
            message = self._messages[2]
            del self._messages[2]
            if message.role == "assistant" and message.tool_calls:
                call_ids = {call.id for call in message.tool_calls if call.id}
                while (
                    len(self._messages) > 2
                    and self._messages[2].role == "tool"
                    and (not call_ids or self._messages[2].tool_call_id in call_ids)
                ):
                    del self._messages[2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "messages": [message.to_dict() for message in self._messages],
        }
