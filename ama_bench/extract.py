"""Answer parsing compatible with AMA-Bench's evaluation format.

AMA-Bench expects ``(A)(B)(C)(D)``-style letter answers for MCQ subsets and
free-form answers for open-ended subsets.  NovaCode's generated responses
(and the evaluation harness itself) use ``Answer[i]:`` markers followed by a
``##Answer:`` block, mirroring the benchmark's own reference implementation.
"""
from __future__ import annotations

import re

_ANSWER_MARKER_RE = re.compile(r"Answer\s*\[\s*(\d+)\s*\]\s*:", re.IGNORECASE)
_ANSWER_BLOCK_RE = re.compile(r"(?:^|\n)\s*##\s*Answer\s*:", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Letters that appear inside an MCQ option token like "(A)" or "(C)(D)".
_OPTION_TOKEN_RE = re.compile(r"\(([A-D])\)")


def extract_final_answer(response: str, mcq_mode: bool = False) -> str:
    """Extract the final answer from a model response (AMA-Bench compatible)."""
    if not response:
        return ""
    cleaned = _THINK_RE.sub("", response)
    if _ANSWER_BLOCK_RE.search(cleaned):
        parts = _ANSWER_BLOCK_RE.split(cleaned, maxsplit=1)
        if len(parts) > 1:
            cleaned = parts[1].strip()
    elif mcq_mode and _OPTION_TOKEN_RE.search(cleaned):
        # A bare "(A)(B)" answer with no marker.
        return "".join(f"({letter})" for letter in _OPTION_TOKEN_RE.findall(cleaned))
    if mcq_mode:
        return cleaned.splitlines()[0].strip() if cleaned.strip() else ""
    return cleaned.strip()


def parse_answer_blocks(response: str, question_count: int, mcq_mode: bool = False) -> list[str]:
    """Parse an ``Answer[1]:``-blocked response into one answer per question."""
    matches = list(_ANSWER_MARKER_RE.finditer(response))
    if not matches:
        return [extract_final_answer(response, mcq_mode=mcq_mode)] * question_count if question_count == 1 else [
            "" for _ in range(question_count)
        ]
    answers: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
        block = response[match.end() : end]
        answers.append(extract_final_answer(block, mcq_mode=mcq_mode))
    return answers


def format_answer_block(question_index: int, answer: str) -> str:
    """Render one answer line in the benchmark's ``Answer[i]:`` format."""
    return f"Answer[{question_index + 1}]: {answer}"


__all__ = [
    "extract_final_answer",
    "format_answer_block",
    "parse_answer_blocks",
]
