"""Evidence retrieval and rendering for NovaCode memory.

Retrieval scores every candidate (trajectory group, task finding, tool
experience) against the question with a light lexical scorer and returns the
top candidates rendered as a compact evidence block.  The rendered block is
what AMA-Bench's answer stage (or the batch prompt builder) feeds the LLM.
"""
from __future__ import annotations

import re
from typing import Any

from .memory import NovaCodeMemory
from .steps import Step

_TERM_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}")

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "will", "would", "can",
        "could", "should", "may", "might", "must", "shall", "not", "no", "nor",
        "and", "or", "but", "if", "then", "else", "when", "where", "which",
        "what", "who", "whom", "whose", "why", "how", "this", "that", "these",
        "those", "it", "its", "of", "in", "on", "at", "to", "for", "from",
        "with", "by", "about", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "out", "off", "over", "under",
        "again", "further", "once", "here", "there", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "only", "own",
        "same", "so", "than", "too", "very", "just", "because", "until", "while",
        "step", "action", "observation", "turn", "many", "much",
        "please", "explain", "describe", "tell", "list", "doing", "agent", "environment", "task", "trajectory",
    }
)


def _tokens(text: str) -> list[str]:
    return [match.group(0) for match in _TERM_RE.finditer(text.lower())]


def _terms(text: str) -> list[str]:
    return [term for term in _tokens(text) if term not in _STOPWORDS]


def _score_text(text: str, question_terms: list[str]) -> int:
    haystack = set(_terms(text))
    return sum(1 for term in question_terms if term in haystack)


def _score_step(step: Step, question_terms: list[str]) -> int:
    return _score_text(step.action, question_terms) + _score_text(step.observation, question_terms)


def score_candidates(memory: NovaCodeMemory, question: str, *, top_k: int = 8) -> list[dict[str, Any]]:
    """Score all retrievable units and return the best ``top_k`` as records."""
    question_terms = _terms(question)
    scored: list[dict[str, Any]] = []

    for group in memory.trajectory.groups:
        rendered = _render_group(group, max_chars=1_500)
        score = 0
        for message in group.messages:
            if message.role in {"user", "assistant"} and message.content or message.role == "tool" and message.content:
                score += _score_text(message.content, question_terms)
        scored.append(
            {
                "type": "group",
                "id": group.group_id,
                "score": score,
                "text": rendered,
                "meta": {"step_range": _group_step_range(group)},
            }
        )

    for finding in memory.task_state.key_findings:
        score = _score_text(finding.fact, question_terms) * 2
        for evidence in finding.evidence:
            if isinstance(evidence, dict):
                score += _score_text(str(evidence.get("path", "")) + " " + str(evidence.get("source", "")), question_terms)
        scored.append({"type": "finding", "id": finding.id, "score": score, "text": finding.fact, "meta": {"status": finding.status}})

    for decision in memory.task_state.decisions:
        text = f"{decision.decision} {decision.reason}"
        scored.append(
            {"type": "decision", "id": decision.id, "score": _score_text(text, question_terms), "text": text, "meta": {"status": decision.status}}
        )

    for profile in memory.tool_state.profiles.values():
        for slot, entries in profile.items():
            for entry in entries:
                value = entry.value if isinstance(entry.value, dict) else {}
                text = str(value.get("text") or value.get("path") or value.get("command") or "")
                if not text:
                    text = str(entry.value)
                scored.append(
                    {"type": "tool", "id": entry.id, "score": _score_text(text, question_terms), "text": text, "meta": {"kind": slot}}
                )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def render_evidence(candidates: list[dict[str, Any]], *, max_total_chars: int = 12_000) -> str:
    """Render scored candidates into a compact evidence block for the LLM."""
    sections: list[str] = []
    budget = max_total_chars
    for candidate in candidates:
        if budget <= 0:
            break
        text = str(candidate.get("text", "")).strip()
        if not text:
            continue
        if len(text) > budget:
            text = text[:budget]
        header = _candidate_header(candidate)
        sections.append(f"<evidence id=\"{candidate.get('id', '')}\" type=\"{candidate.get('type', '')}\">\n{header}{text}\n</evidence>")
        budget -= len(text) + 120
    if not sections:
        return "<evidence>no relevant evidence found in trajectory</evidence>"
    return "\n".join(sections)


def _candidate_header(candidate: dict[str, Any]) -> str:
    meta = candidate.get("meta") or {}
    parts = []
    for key in ("status", "kind", "step_range"):
        value = meta.get(key)
        if value:
            parts.append(f"{key}={value}")
    return f"<!-- {' '.join(parts)} -->\n" if parts else ""


def _render_group(group: Any, *, max_chars: int = 1_500) -> str:
    lines: list[str] = []
    total = 0
    for message in group.messages:
        if message.role == "tool":
            text = str(message.content or "")[:800]
        elif message.role == "assistant":
            text = str(message.content or "")[:200]
            calls = " ".join(f"{call.name}({_brief_args(call.arguments)})" for call in message.tool_calls)
            if calls:
                text = (text + " " + calls).strip()
        else:
            text = str(message.content or "")[:400]
        if not text:
            continue
        if total + len(text) > max_chars:
            lines.append("... [group truncated] ...")
            break
        lines.append(text)
        total += len(text)
    return "\n".join(lines)


def _brief_args(arguments: dict[str, Any]) -> str:
    path = arguments.get("path") or arguments.get("command") or arguments.get("raw") or ""
    return str(path)[:120]


def _group_step_range(group: Any) -> str:
    first = last = ""
    for message in group.messages:
        if message.role == "tool" and message.content:
            match = re.search(r"Step\s+(\d+)", str(message.content))
            if match:
                first = first or match.group(1)
                last = match.group(1)
    return f"{first}-{last}" if first else ""
