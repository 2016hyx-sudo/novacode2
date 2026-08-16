"""Local token estimation with provider-usage calibration."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import Message, ToolSchema


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Calibration:
    coefficient: float = 1.0
    window: list[tuple[float, float]] = field(default_factory=list)  # (estimate, provider_prompt_tokens)
    max_window: int = 8

    def record(self, estimate: float, usage: TokenUsage) -> None:
        if estimate <= 0 or usage.prompt_tokens <= 0:
            return
        ratio = usage.prompt_tokens / estimate
        self.window.append((estimate, ratio))
        if len(self.window) > self.max_window:
            self.window.pop(0)
        self.coefficient = sum(ratio for _, ratio in self.window) / len(self.window)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficient": self.coefficient,
            "samples": len(self.window),
            "max_window": self.max_window,
            "window": [[float(estimate), float(ratio)] for estimate, ratio in self.window],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibration:
        data = data or {}
        calibration = cls(
            coefficient=float(data.get("coefficient", 1.0)),
            max_window=int(data.get("max_window", 8)),
        )
        for pair in data.get("window") or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                calibration.window.append((float(pair[0]), float(pair[1])))
        return calibration


class TokenCounter:
    def __init__(self, *, calibration: Calibration | None = None) -> None:
        self.calibration = calibration or Calibration()

    @staticmethod
    def estimate_text(text: str | None) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4 + 1)

    @classmethod
    def estimate_message(cls, message: Message) -> int:
        total = cls.estimate_text(message.content)
        for call in message.tool_calls:
            try:
                total += cls.estimate_text(json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")))
            except TypeError:
                total += 4
        return total

    @classmethod
    def estimate_messages(cls, messages: list[Message]) -> int:
        return sum(cls.estimate_message(message) for message in messages)

    @classmethod
    def estimate_tools(cls, tools: list[ToolSchema] | tuple[ToolSchema, ...] | None) -> int:
        total = 0
        for tool in tools or []:
            text = json.dumps(
                {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            total += cls.estimate_text(text)
        return total

    def estimate_prompt(
        self,
        *,
        system_text: str,
        tools: list[ToolSchema] | tuple[ToolSchema, ...] | None,
        messages: list[Message],
    ) -> int:
        raw = self.estimate_text(system_text) + self.estimate_tools(tools) + self.estimate_messages(messages)
        return max(1, int(raw * self.calibration.coefficient))

    def to_dict(self) -> dict[str, Any]:
        return {"calibration": self.calibration.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TokenCounter:
        data = data or {}
        return cls(calibration=Calibration.from_dict(data.get("calibration") or data))

    def record_usage(self, estimate: int, usage: TokenUsage | dict[str, Any]) -> None:
        if isinstance(usage, dict):
            usage = TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
                cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            )
        self.calibration.record(estimate, usage)

    def usage_ratio(self, estimate: int, window_limit: int = 256_000) -> float:
        if window_limit <= 0:
            return 0.0
        return estimate / window_limit
