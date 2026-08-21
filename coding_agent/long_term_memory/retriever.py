"""Two-stage out-of-band memory retrieval pipeline with lexical fallback."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .models import MemoryEntry, MemoryHeader, MemoryType, QuotaConfig
from .store import MemoryStore

if TYPE_CHECKING:
    from ..llm.base import LLMProvider


SIDE_QUERY_SYSTEM_PROMPT = """You are a memory selection system. You are given a user's query and a list of available memories with their types, names, and one-line descriptions.
Select ONLY the memories that are directly and clearly relevant to helping answer the query.

Constraints:
- Return a JSON object: {"selected_memories": ["name1", "name2"]}
- Select at most 5 memories.
- If unsure whether a memory is helpful, DO NOT include it.
- If no memories are relevant, return {"selected_memories": []}.
"""


class PrefetchGate:
    """Evaluates whether to trigger out-of-band memory retrieval for a turn."""

    IGNORE_COMMANDS = frozenset({"exit", "quit", "clear", "help", "status", "q", ":q"})

    @classmethod
    def can_prefetch(
        cls,
        query: str,
        *,
        store: MemoryStore,
        session_injected_bytes: int = 0,
        is_subagent: bool = False,
    ) -> bool:
        if is_subagent:
            return False

        stripped = query.strip()
        if len(stripped) < 3:
            return False

        if stripped.lower() in cls.IGNORE_COMMANDS:
            return False

        if session_injected_bytes >= store.quota.max_session_injected_bytes:
            return False

        headers = store.list_headers(limit=1)
        if not headers:
            return False

        return True


class HeaderScanner:
    """Fast (<5ms) metadata scanner producing a compact candidate manifest."""

    @classmethod
    def scan_manifest(
        cls,
        store: MemoryStore,
        already_surfaced_names: set[str],
        limit: int = 200,
    ) -> tuple[list[MemoryHeader], str]:
        all_headers = store.list_headers(limit=limit)
        candidates = [h for h in all_headers if h.name not in already_surfaced_names]

        manifest_lines: list[str] = []
        for h in candidates:
            manifest_lines.append(f"- [{h.type.value}] {h.name} ({h.updated_at[:10]}): {h.description}")

        manifest_text = "\n".join(manifest_lines)
        return candidates, manifest_text


class LexicalScorer:
    """Deterministic token overlap & keyword matching fallback engine."""

    STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
        "of", "with", "by", "and", "or", "not", "this", "that", "it", "my", "we", "you",
        "把", "在", "的", "了", "和", "是", "我", "你", "这", "那", "用", "做", "请"
    })

    @classmethod
    def score_candidates(
        cls,
        query: str,
        candidates: list[MemoryHeader],
        top_k: int = 5,
    ) -> list[str]:
        query_tokens = cls._tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, str]] = []
        for h in candidates:
            score = 0.0
            header_tokens = cls._tokenize(f"{h.name} {h.description} {h.type.value}")
            if not header_tokens:
                continue

            # Exact name or token overlap
            overlap = query_tokens.intersection(header_tokens)
            if overlap:
                score += len(overlap) * 2.0

            # Substring match in description or name
            clean_name = h.name.lower().replace("_", " ")
            if clean_name in query.lower():
                score += 5.0
            for token in query_tokens:
                if len(token) >= 3 and (token in h.description.lower() or token in h.name.lower()):
                    score += 1.5

            if score > 1.0:
                scored.append((score, h.name))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in scored[:top_k]]

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        text = text.lower()
        # 1. English / alphanumeric tokens
        en_words = re.findall(r"[a-z0-9_\-]+", text)
        # 2. Chinese characters and 2-grams
        cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
        cn_bigrams = [cn_chars[i] + cn_chars[i + 1] for i in range(len(cn_chars) - 1)]

        all_tokens = set(en_words + cn_bigrams + cn_chars)
        return {w for w in all_tokens if w not in cls.STOP_WORDS and len(w) > 0}


class SideQueryEngine:
    """Invokes a lightweight secondary LLM call (max_tokens=256) to arbitrate recall."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    def select(
        self,
        query: str,
        manifest: str,
        candidates: list[MemoryHeader],
        top_k: int = 5,
    ) -> list[str]:
        if not manifest.strip() or not candidates:
            return []

        if self.provider is not None:
            try:
                prompt_text = (
                    f"User Query:\n{query}\n\n"
                    f"Available Memories:\n{manifest}\n\n"
                    "Select relevant memories JSON:"
                )
                from ..llm.base import Message
                messages = [
                    Message.system(SIDE_QUERY_SYSTEM_PROMPT),
                    Message.user(prompt_text),
                ]
                resp = self.provider.chat(
                    messages=messages,
                    max_tokens=256,
                    temperature=0.0,
                )
                raw_text = resp.text.strip()
                # Parse JSON block
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    selected = data.get("selected_memories", [])
                    if isinstance(selected, list):
                        valid_names = {c.name for c in candidates}
                        filtered = [str(n) for n in selected if str(n) in valid_names]
                        return filtered[:top_k]
            except Exception:
                pass

        # Fallback to deterministic lexical scorer
        return LexicalScorer.score_candidates(query, candidates, top_k=top_k)


class MemoryRetriever:
    """Out-of-band two-stage retrieval pipeline."""

    def __init__(
        self,
        store: MemoryStore,
        side_query_provider: LLMProvider | None = None,
    ) -> None:
        self.store = store
        self.side_engine = SideQueryEngine(provider=side_query_provider)

    def prefetch(
        self,
        query: str,
        *,
        already_surfaced: set[str],
        session_injected_bytes: int = 0,
        is_subagent: bool = False,
    ) -> list[MemoryEntry]:
        """Execute the two-stage prefetch pipeline and return relevant memory entries."""
        if not PrefetchGate.can_prefetch(
            query,
            store=self.store,
            session_injected_bytes=session_injected_bytes,
            is_subagent=is_subagent,
        ):
            return []

        candidates, manifest = HeaderScanner.scan_manifest(
            self.store,
            already_surfaced_names=already_surfaced,
            limit=self.store.quota.max_scan_pool_size,
        )
        if not candidates:
            return []

        selected_names = self.side_engine.select(
            query=query,
            manifest=manifest,
            candidates=candidates,
            top_k=self.store.quota.max_recalled_items,
        )
        if not selected_names:
            return []

        entries: list[MemoryEntry] = []
        for name in selected_names:
            entry = self.store.load_entry(name)
            if entry is not None:
                entries.append(entry)

        return entries
