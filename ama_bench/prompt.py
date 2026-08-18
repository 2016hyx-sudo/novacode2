"""Batch prompt builder for answering questions from NovaCode memory.

Mirrors AMA-Bench's ``longcontext`` batch format (``Answer[i]:`` slots) while
using per-question evidence retrieval instead of one giant trajectory dump.
The final answer line is extracted by :func:`ama_bench.extract.extract_final_answer`.
"""
from __future__ import annotations

from .memory import NovaCodeMemory
from .retrieve import render_evidence, score_candidates

_MAX_PROMPT_CHARS = 24_000


def build_batch_prompt(
    memory: NovaCodeMemory,
    questions: list[str],
    *,
    mcq_mode: bool = False,
    max_total_chars: int = _MAX_PROMPT_CHARS,
) -> str:
    """Build one prompt that answers every question in a single LLM call.

    Per-question evidence shrinks when many questions share one call, keeping
    the total prompt well under provider timeouts (a 12-question episode stays
    ~15 KB rather than ~26 KB).
    """
    evidence_budget = 3_000 if len(questions) <= 1 else 2_000
    objective = (memory.task_state.objective or "").strip()
    lines: list[str] = []
    lines.append("You are answering questions about an agent trajectory.")
    if objective:
        lines.append(f"Episode task: {objective}")
    budget = max_total_chars
    for index, question in enumerate(questions, start=1):
        candidates = score_candidates(memory, question)
        evidence = render_evidence(candidates, max_total_chars=evidence_budget)
        if len(evidence) > budget:
            evidence = evidence[:budget]
        lines.append("")
        lines.append(f"Question {index}: {question}")
        lines.append("Relevant trajectory evidence:")
        lines.append(evidence)
        if mcq_mode:
            lines.append(f"Answer[{index}]: [select all correct options, e.g. (A) or (A)(C)]")
        else:
            lines.append(f"Answer[{index}]: [your answer here]")
        budget -= len(evidence) + 200
        if budget <= 0:
            lines.append("(context budget exhausted; remaining questions omitted)")
            break
    return "\n".join(lines)
