"""AMA-Bench adapter: exposes NovaCode structured context as a memory method.

This package implements the AMA-Bench two-stage interface
(``memory_construction`` / ``memory_retrieve``) on top of NovaCode's
structured-context components (Interaction Groups, deterministic/LLM
Trajectory Fold, Task/Tool State).  It is importable without AMA-Bench
installed; only :mod:`ama_bench.method` needs ``BaseMethod`` (with a local
fallback when the benchmark is absent) and the optional provider adapter
needs AMA's ``ModelClient``.
"""

from .extract import extract_final_answer
from .fold import NovaCodeMemoryBuilder
from .memory import NovaCodeMemory
from .prompt import build_batch_prompt
from .retrieve import render_evidence, score_candidates
from .steps import Step, parse_trajectory_text

__version__ = "0.1.0"

__all__ = [
    "NovaCodeMemory",
    "NovaCodeMemoryBuilder",
    "Step",
    "__version__",
    "build_batch_prompt",
    "extract_final_answer",
    "parse_trajectory_text",
    "render_evidence",
    "score_candidates",
]
