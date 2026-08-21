"""Dynamic slot injection and Freshness Guard formatting for recalled memories."""
from __future__ import annotations

import datetime
from typing import Sequence

from .models import MemoryEntry


class FreshnessGuard:
    """Computes relative age and injects safety warnings for older memories."""

    @classmethod
    def evaluate_freshness(cls, timestamp_str: str) -> tuple[str, str | None]:
        """Returns (relative_age_label, warning_message_or_None)."""
        if not timestamp_str:
            return "unknown", None

        try:
            # Handle ISO formats including trailing Z
            clean_ts = timestamp_str.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(clean_ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc)
            delta = now - dt
            days = delta.days
            seconds = delta.seconds

            if days <= 0 and seconds < 86400:
                return "saved today", None
            elif days == 1:
                return "saved yesterday", None
            else:
                warning = (
                    f"⚠️ This memory is {days} days old. Memories are point-in-time observations, "
                    "not live state — claims about code behavior may be outdated. "
                    "Verify against current code before asserting as fact."
                )
                return f"saved {days} days ago", warning
        except Exception:
            return "unknown", None


class MemoryInjector:
    """Formats recalled memories into structured <system-reminder> blocks for current turn."""

    @classmethod
    def format_entry(cls, entry: MemoryEntry) -> str:
        """Format a single memory entry with metadata header and freshness warning."""
        ts = entry.header.updated_at or entry.header.created_at
        age_label, warning = FreshnessGuard.evaluate_freshness(ts)

        lines: list[str] = ["<system-reminder>"]
        if warning:
            lines.append(warning)
        lines.append(f"Memory ID: {entry.name} (Category: {entry.type.value}, Saved: {age_label})")
        lines.append(entry.content.strip())
        lines.append("</system-reminder>")
        return "\n".join(lines)

    @classmethod
    def wrap_user_message(cls, query: str, memories: Sequence[MemoryEntry]) -> str:
        """Prepend recalled memory XML blocks to the current turn's user message."""
        if not memories:
            return query

        blocks = [cls.format_entry(m) for m in memories]
        merged_blocks = "\n\n".join(blocks)
        return f"{merged_blocks}\n\n{query}"
