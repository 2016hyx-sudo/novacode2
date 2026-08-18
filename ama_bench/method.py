"""AMA-Bench memory method exposing NovaCode's structured context.

Two-stage interface:

* ``memory_construction(traj_text, task="")`` — parse the rendered trajectory,
  group steps into Interaction Groups, then distill them through NovaCode's
  deterministic or LLM-assisted Trajectory Fold into Task/Tool State.
* ``memory_retrieve(memory, question)`` — score every group/finding/tool entry
  against the question and render the top evidence block for the answer stage.

The class is defined against a local ``BaseMethod`` fallback so this package
imports standalone (NovaCode tests, offline use).  In the benchmark repository
run :mod:`ama_bench.register_ama` to bind it to AMA's real ``BaseMethod`` and
register it as ``novacode``.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:  # importable outside the benchmark repo
    from src.method.base_method import BaseMethod  # type: ignore
except Exception:  # pragma: no cover - fallback for standalone use
    class BaseMethod:
        """Local stand-in matching AMA-Bench's BaseMethod surface."""

        @staticmethod
        def _load_config(config_path: str | None) -> dict[str, Any]:
            if not config_path:
                return {}
            path = Path(config_path)
            if path.suffix in {".yaml", ".yml"}:
                import yaml

                return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if path.suffix == ".json":
                return json.loads(path.read_text(encoding="utf-8"))
            return {}

        def memory_construction(self, traj_text: str, task: str = "") -> Any:
            raise NotImplementedError

        def memory_retrieve(self, memory: Any, question: str) -> str:
            raise NotImplementedError


from coding_agent.llm.base import LLMProvider, LLMResponse, Message

from .extract import extract_final_answer
from .fold import BuilderConfig, NovaCodeMemoryBuilder
from .memory import NovaCodeMemory
from .prompt import build_batch_prompt
from .retrieve import render_evidence, score_candidates
from .steps import parse_trajectory_text


class AMAClientFoldProvider:
    """Adapts AMA-Bench's ModelClient to NovaCode's LLMProvider protocol.

    The fold stage receives ``[system, user]`` messages and returns the LLM's
    JSON delta as plain text.  Used only when the benchmark's ``client`` is
    injected, enabling LLM-assisted fold with the same model as the run.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] | None = None,
    ) -> LLMResponse:
        system = "\n\n".join(str(m.content) for m in messages if m.role == "system" and m.content)
        prompt = "\n\n".join(str(m.content) for m in messages if m.role == "user" and m.content)
        try:
            max_tokens = int(getattr(self._client, "config", {}).get("max_tokens", 4096))
        except (TypeError, ValueError):
            max_tokens = 4096
        try:
            text = self._client.query(
                prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                system=system or None,
            )
        except Exception as exc:
            from coding_agent.llm.base import LLMError

            raise LLMError(f"AMA fold provider query failed: {exc}", retryable=False) from exc
        return LLMResponse(text=str(text or "").strip(), stop_reason="end_turn", usage={})


class NovaCodeMemoryMethod(BaseMethod):
    """AMA-Bench method: NovaCode structured context as agent memory."""

    def __init__(
        self,
        config_path: str | None = None,
        embedding_engine: Any = None,
        client: Any = None,
        fold_provider: LLMProvider | None = None,
        fold_max_attempts: int = 2,
        keep_work_dir: bool = False,
        max_context_tokens: int = 96_000,
    ) -> None:
        config: dict[str, Any] = self._load_config(config_path) if config_path else {}
        self.embedding_engine = embedding_engine
        if fold_provider is None and client is not None:
            fold_provider = AMAClientFoldProvider(client)
        self._fold_provider = fold_provider
        self._builder_config = BuilderConfig(
            max_context_tokens=int(config.get("max_context_tokens", max_context_tokens)),
            fold_max_attempts=int(config.get("fold_max_attempts", fold_max_attempts)),
            keep_work_dir=bool(config.get("keep_work_dir", keep_work_dir)),
        )

    # ------------------------------------------------------------- two-stage interface

    def memory_construction(self, traj_text: str, task: str = "") -> NovaCodeMemory:
        steps = parse_trajectory_text(traj_text)
        builder = NovaCodeMemoryBuilder(
            fold_provider=self._fold_provider,
            config=self._builder_config,
        )
        return builder.build(steps, task=task)

    def memory_retrieve(self, memory: NovaCodeMemory, question: Any, mcq_mode: bool = False) -> str:
        if isinstance(question, (list, tuple)):
            questions = [str(item) for item in question]
            if not questions:
                return ""
            return build_batch_prompt(memory, questions, mcq_mode=mcq_mode)
        candidates = score_candidates(memory, str(question))
        return render_evidence(candidates)

    def build_prompt(self, memory: NovaCodeMemory, questions: list[str], mcq_mode: bool = False) -> str:
        """Convenience wrapper around the batch prompt builder."""
        return build_batch_prompt(memory, list(questions), mcq_mode=mcq_mode)

    def extract_final_answer(self, response: str, mcq_mode: bool = False) -> str:
        return extract_final_answer(response, mcq_mode=mcq_mode)

    @property
    def requires_embedding(self) -> bool:
        return False


__all__ = [
    "AMAClientFoldProvider",
    "NovaCodeMemoryMethod",
]
