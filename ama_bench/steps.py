"""Parsing of AMA-Bench trajectory text into uniform trajectory steps.

The official benchmark feeds every memory method the same rendered trajectory
text (``Step N:`` / ``Action: ...`` / ``Observation: ...`` blocks).  This
module is the narrow boundary that turns that text into steps NovaCode's
structured-context builders can consume.  It deliberately never looks at the
episode JSON, so the same parser works for any dataset rendered with the same
text format.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_STEP_RE = re.compile(r"^\s*(?:Step|Turn)\s+(\d+)\s*[:.]?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Step:
    """One trajectory step: an action and the observation it produced."""

    turn_idx: int
    action: str
    observation: str

    def to_dict(self) -> dict[str, Any]:
        return {"turn_idx": self.turn_idx, "action": self.action, "observation": self.observation}


def parse_trajectory_text(text: str) -> list[Step]:
    """Parse rendered trajectory text into :class:`Step` records.

    Handles the benchmark's ``Step N:`` / ``Turn N`` markers (either may be
    followed by ``Action:`` / ``Observation:`` lines, which may span multiple
    lines).  A step without an explicit action is emitted as an empty action so
    the caller can distinguish it from a missing observation.
    """
    steps: list[Step] = []
    current_index: int | None = None
    action_parts: list[str] = []
    observation_parts: list[str] = []
    in_observation = False

    def flush() -> None:
        nonlocal current_index, action_parts, observation_parts, in_observation
        if current_index is not None:
            steps.append(
                Step(
                    turn_idx=current_index,
                    action="\n".join(part for part in action_parts if part.strip()).strip(),
                    observation="\n".join(part for part in observation_parts if part.strip()).strip(),
                )
            )
        current_index = None
        action_parts = []
        observation_parts = []
        in_observation = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        step_match = _STEP_RE.match(line)
        if step_match:
            flush()
            current_index = int(step_match.group(1))
            in_observation = False
            continue
        if line.lower().startswith("action:"):
            action_parts.append(line[len("action:") :].strip())
            in_observation = False
            continue
        if line.lower().startswith("observation:"):
            observation_parts.append(line[len("observation:") :].strip())
            in_observation = True
            continue
        if in_observation:
            observation_parts.append(line)
        else:
            action_parts.append(line)
    flush()
    return steps


def steps_to_text(steps: list[Step]) -> str:
    """Render steps back to the benchmark's canonical text format."""
    blocks: list[str] = []
    for step in steps:
        lines = [f"Step {step.turn_idx}:", f"Action: {step.action}", f"Observation: {step.observation}"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
