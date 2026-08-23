"""Negative suppression engine enforcing the 7 hard-stop governance rules."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING

from .models import MemoryEntry, MemoryHeader, MemoryType

if TYPE_CHECKING:
    from .store import MemoryStore


@dataclass
class SuppressionVerdict:
    """Outcome of negative suppression check."""
    passed: bool
    reason: str = ""
    rule_id: int | None = None
    suggested_action: str = ""  # e.g., "use_update_memory"


# Regex patterns for sensitive credentials (Rule 6)
SECRET_PATTERNS = [
    (re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE), "OpenAI / generic API key"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36}", re.IGNORECASE), "GitHub Personal Access Token"),
    (re.compile(r"gho_[a-zA-Z0-9]{36}", re.IGNORECASE), "GitHub OAuth Token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS Access Key ID"),
    (re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA|PGP)?\s*PRIVATE KEY-----"), "Private Key"),
    (re.compile(r"(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*['\"][a-zA-Z0-9_\-\.]{8,}['\"]", re.IGNORECASE), "Hardcoded Secret/Password"),
]

# Patterns for codebase facts / directory trees (Rule 2)
TREE_PATTERNS = [
    re.compile(r"(?:[├└]──\s+)+", re.MULTILINE),
    re.compile(r"(?:^|\n)(?:├──|└──|│\s+├──)", re.MULTILINE),
    re.compile(r"^def\s+[a-zA-Z0-9_]+\s*\([^)]*\)\s*->\s*[^:]+:\s*$", re.MULTILINE),
]

# Patterns for single-task ephemeral states (Rule 3)
EPHEMERAL_PATTERNS = [
    re.compile(r"\b(?:TODO|FIXME|WIP)\s*:\s*(?:finish|fix|test)\s+line\s+\d+", re.IGNORECASE),
    re.compile(r"\bline\s+\d+\s+in\s+file\s+['\"][^'\"]+['\"]\s+is\s+temporarily", re.IGNORECASE),
    re.compile(r"/tmp/[a-zA-Z0-9_\-]+", re.IGNORECASE),
    re.compile(r"(?:workspace|directory|repo|folder)\s+is\s+(?:currently\s+)?empty", re.IGNORECASE),
    re.compile(r"currently\s+empty\s+of\s+(?:source\s+)?files", re.IGNORECASE),
    re.compile(r"currently\s+contains?\s+only\s+(?:\.agent|\.git)", re.IGNORECASE),
    re.compile(r"(?:当前目录为空|工作区为空|当前没有文件|目前只有\.agent)", re.IGNORECASE),
]

# Patterns for one-off temporary instructions (Rule 4)
ONEOFF_PATTERNS = [
    re.compile(r"(?:just\s+for\s+now|only\s+this\s+time|this\s+single\s+run|for\s+testing\s+purposes\s+only|这次先|临时打印|单次测试)", re.IGNORECASE),
]


class SuppressionEngine:
    """Evaluates memory writes against the 7 negative boundaries."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store

    def validate_write(
        self,
        entry: MemoryEntry,
        *,
        is_subagent: bool = False,
        is_update: bool = False,
    ) -> SuppressionVerdict:
        """Run all 7 negative suppression checks. Returns pass/fail verdict."""

        # Rule 7: Subagent write prohibition
        if is_subagent:
            return SuppressionVerdict(
                passed=False,
                rule_id=7,
                reason="Subagents are prohibited from writing or modifying persistent memories.",
            )

        # Rule 6: Secret Hard-Stop
        combined_text = f"{entry.header.name} {entry.header.description} {entry.content}"
        for pattern, desc in SECRET_PATTERNS:
            if pattern.search(combined_text):
                return SuppressionVerdict(
                    passed=False,
                    rule_id=6,
                    reason=f"Found sensitive credential ({desc}). Secret storage is strictly forbidden.",
                )

        # Rule 1: No unverified hypotheses (for feedback type)
        if entry.header.type == MemoryType.FEEDBACK:
            # Must have explanation of why/how and not be a guess
            lower_body = entry.content.lower()
            if any(guess_word in lower_body for guess_word in ["maybe it works", "untested guess", "not sure if", "未经验证", "猜测"]):
                return SuppressionVerdict(
                    passed=False,
                    rule_id=1,
                    reason="Feedback memory contains unverified hypotheses. Verified fix and root cause required.",
                )
            if len(entry.content.strip()) < 15:
                return SuppressionVerdict(
                    passed=False,
                    rule_id=1,
                    reason="Feedback memory too short. Must explain the root cause (Why) and verified solution (How).",
                )

        # Rule 2: No codebase facts (file trees, raw signatures without rationale)
        for pattern in TREE_PATTERNS:
            if pattern.search(entry.content):
                return SuppressionVerdict(
                    passed=False,
                    rule_id=2,
                    reason="Codebase facts (directory trees / raw signatures) should be queried via tools, not stored in memory.",
                )

        # Rule 3: No single-task ephemeral states
        for pattern in EPHEMERAL_PATTERNS:
            if pattern.search(entry.content):
                return SuppressionVerdict(
                    passed=False,
                    rule_id=3,
                    reason="Contains single-task ephemeral state (temporary file/line TODO) belonging to Working Memory.",
                )

        # Rule 4: No generalizing one-off instructions
        for pattern in ONEOFF_PATTERNS:
            if pattern.search(entry.header.description) or pattern.search(entry.content):
                return SuppressionVerdict(
                    passed=False,
                    rule_id=4,
                    reason="One-off temporary instruction cannot be generalized into persistent memory.",
                )

        # Rule 5: No duplicate redundant entries (if creating a new entry)
        if not is_update and self.store is not None:
            duplicate_verdict = self._check_duplicate(entry)
            if not duplicate_verdict.passed:
                return duplicate_verdict

        return SuppressionVerdict(passed=True)

    def _check_duplicate(self, entry: MemoryEntry) -> SuppressionVerdict:
        """Check for high semantic overlap with existing entries (>75% description similarity)."""
        existing_headers = self.store.list_headers(limit=200) if self.store else []
        new_desc_tokens = self._tokenize(entry.header.description)
        if not new_desc_tokens:
            return SuppressionVerdict(passed=True)

        for h in existing_headers:
            if h.name.lower() == entry.header.name.lower():
                return SuppressionVerdict(
                    passed=False,
                    rule_id=5,
                    reason=f"Memory with name '{h.name}' already exists. Use update_memory to modify.",
                    suggested_action="use_update_memory",
                )

            existing_tokens = self._tokenize(h.description)
            if not existing_tokens:
                continue

            # Jaccard similarity over description tokens
            intersection = new_desc_tokens.intersection(existing_tokens)
            union = new_desc_tokens.union(existing_tokens)
            jaccard = len(intersection) / len(union) if union else 0.0

            if jaccard >= 0.75:
                return SuppressionVerdict(
                    passed=False,
                    rule_id=5,
                    reason=f"High semantic overlap ({jaccard:.0%}) with existing memory '{h.name}'. Please merge with update_memory.",
                    suggested_action="use_update_memory",
                )

        return SuppressionVerdict(passed=True)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        text = text.lower()
        en_words = re.findall(r"[a-z0-9_\-]+", text)
        cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
        cn_bigrams = [cn_chars[i] + cn_chars[i + 1] for i in range(len(cn_chars) - 1)]
        return set(en_words + cn_bigrams + cn_chars)
