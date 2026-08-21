"""Lightweight JSONL execution trace with optional in-process listeners."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EventListener = Callable[["TraceEvent"], None]


@dataclass
class TraceEvent:
    type: str
    session_id: str
    data: dict[str, Any]
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "type": self.type, "session_id": self.session_id, "data": self.data}


class TraceWriter:
    """Append-only JSONL trace writer.

    Also calls a small list of listeners (the TUI) synchronously. This is a
    plain callback, not an event bus.
    """

    def __init__(
        self,
        directory: Path,
        *,
        session_id: str = "",
        listeners: list[EventListener] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.session_id = session_id
        self.listeners = listeners or []
        self._lock = threading.Lock()

    def bind(self, session_id: str) -> None:
        self.session_id = session_id

    def _path(self) -> Path | None:
        if not self.session_id:
            return None
        return self.directory / f"{self.session_id}.jsonl"

    def emit(self, event_type: str, **data: Any) -> TraceEvent:
        event = TraceEvent(type=event_type, session_id=self.session_id, data=dict(data))
        with self._lock:
            # Skip writing ephemeral token stream chunks to disk to avoid bloating JSONL traces
            if event_type != "llm_chunk":
                path = self._path()
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")
            for listener in self.listeners:
                listener(event)
        return event
