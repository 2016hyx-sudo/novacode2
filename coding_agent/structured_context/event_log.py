"""Append-only event log with sequence numbers and a SHA-256 hash chain."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import canonical_json, sha256_text, utcnow


class EventLogCorruption(Exception):
    pass


class EventLog:
    def __init__(self, path: Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self._last_seq = 0
        self._last_hash = ""
        self._last_offset = 0
        self._scan()

    def _scan(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            offset = 0
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    offset += len(raw_line.encode("utf-8"))
                    continue
                try:
                    event = json.loads(line)
                    seq = int(event["seq"])
                    payload_hash = str(event["payload_hash"])
                    prev_hash = str(event["prev_hash"])
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    raise EventLogCorruption(f"Malformed event at offset {offset}: {exc}") from exc
                computed = sha256_text(canonical_json(event["payload"]))
                if computed != payload_hash:
                    raise EventLogCorruption(f"Payload hash mismatch at seq {seq}")
                if self._last_hash and prev_hash != self._last_hash:
                    raise EventLogCorruption(f"Hash chain broken before seq {seq}")
                self._last_seq = max(self._last_seq, seq)
                self._last_hash = sha256_text(
                    canonical_json({"seq": seq, "payload_hash": payload_hash, "prev_hash": prev_hash})
                )
                offset += len(raw_line.encode("utf-8"))
        self._last_offset = offset

    @property
    def last_seq(self) -> int:
        return self._last_seq

    @property
    def last_hash(self) -> str:
        return self._last_hash

    @property
    def last_offset(self) -> int:
        return self._last_offset

    def anchor(self) -> dict[str, Any]:
        return {
            "last_event_seq": self._last_seq,
            "last_event_offset": self._last_offset,
            "last_event_hash": self._last_hash,
            "log_file": self.path.name,
        }

    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        seq = self._last_seq + 1
        event = {
            "seq": seq,
            "ts": utcnow(),
            "type": event_type,
            "prev_hash": self._last_hash,
            "payload": payload,
            "payload_hash": sha256_text(canonical_json(payload)),
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            if self.fsync:
                handle.flush()
                os.fsync(handle.fileno())
        self._last_seq = seq
        self._last_hash = sha256_text(
            canonical_json({"seq": seq, "payload_hash": event["payload_hash"], "prev_hash": event["prev_hash"]})
        )
        self._last_offset += len(line.encode("utf-8"))
        return event

    def read_since(self, after_seq: int) -> list[dict[str, Any]]:
        if after_seq < 0:
            after_seq = 0
        events: list[dict[str, Any]] = []
        if not self.path.exists():
            return events
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if int(event["seq"]) > after_seq:
                    events.append(event)
        return events

    def validate_anchor(self, anchor: dict[str, Any]) -> bool:
        seq = int(anchor.get("last_event_seq", -1))
        if seq < 0 or seq > self._last_seq:
            return False
        expected = anchor.get("last_event_hash")
        if seq == 0:
            return True
        if not expected:
            return False
        # Replay the chain up to the anchored sequence. Events after the anchor
        # are a valid crash-recovery suffix and do not invalidate the checkpoint.
        current_hash = ""
        if not self.path.exists():
            return False
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if int(event["seq"]) > seq:
                    break
                current_hash = sha256_text(
                    canonical_json(
                        {
                            "seq": int(event["seq"]),
                            "payload_hash": str(event["payload_hash"]),
                            "prev_hash": str(event["prev_hash"]),
                        }
                    )
                )
        return current_hash == expected
