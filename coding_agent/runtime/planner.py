"""Lightweight optional planner.

This is not a Planner Agent. It makes one planning LLM call with tools
disabled, parses a numbered list, and supports deterministic correction steps
during execution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.base import LLMProvider, Message
from .trace import TraceWriter

_PLAN_STEP = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$")


@dataclass
class Plan:
    steps: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.steps:
            return "(no plan)"
        return "\n".join(f"{i}. {step}" for i, step in enumerate(self.steps, 1))

    def add_correction(self, summary: str) -> None:
        note = summary.strip().replace("\n", " ")
        if not note:
            return
        # Avoid appending the exact same correction repeatedly.
        marker = f"[修正] {note}"
        if marker not in self.steps:
            self.steps.append(marker)


class Planner:
    def __init__(self, llm: LLMProvider, *, trace: TraceWriter | None = None) -> None:
        self.llm = llm
        self.trace = trace
        self.current_plan: Plan | None = None

    def generate(self, task: str) -> Plan:
        messages = [
            Message(
                role="system",
                content=(
                    "You are a lightweight task planner for a coding agent. "
                    "Break the user task into 3-8 short, concrete, ordered steps. "
                    "Each step must be a single line. Reply with only the numbered list, "
                    "no introduction and no markdown formatting."
                ),
            ),
            Message(role="user", content=task),
        ]
        response = self.llm.chat(messages, tools=[])
        steps = self._parse(response.text or "")
        if not steps:
            steps = ["Inspect the relevant part of the workspace", "Implement the requested change", "Verify the result"]
        self.current_plan = Plan(steps)
        self._emit("planner_result", steps=steps)
        return self.current_plan

    def add_correction(self, summary: str) -> Plan:
        if self.current_plan is None:
            self.current_plan = Plan()
        self.current_plan.add_correction(summary)
        self._emit("planner_update", steps=self.current_plan.steps, correction=summary)
        return self.current_plan

    @staticmethod
    def _parse(text: str) -> list[str]:
        steps: list[str] = []
        inside_fence = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("```"):
                inside_fence = not inside_fence
                continue
            if inside_fence:
                continue
            match = _PLAN_STEP.match(line)
            if match and match.group(1):
                steps.append(match.group(1))
        return steps[:10]

    def _emit(self, event_type: str, **data: object) -> None:
        if self.trace is not None:
            self.trace.emit(event_type, **data)
