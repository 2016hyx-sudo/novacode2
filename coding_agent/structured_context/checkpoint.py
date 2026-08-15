"""Checkpoint persistence with immutable snapshots and atomic manifest commit."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .event_log import EventLog
from .models import (
    CheckpointManifest,
    StructuredSession,
    TaskState,
    ToolState,
    Trajectory,
    WorkspaceExpected,
    canonical_json,
    sha256_text,
    utcnow,
)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return sha256_text(text)


class CheckpointError(RuntimeError):
    pass


class CheckpointManager:
    def __init__(self, session_dir: Path, *, fsync: bool = True) -> None:
        self.session_dir = Path(session_dir)
        self.checkpoint_dir = self.session_dir / "checkpoints"
        self.current_path = self.checkpoint_dir / "CURRENT"
        self.fsync = fsync

    def save(
        self,
        *,
        task_state: TaskState,
        tool_state: ToolState,
        trajectory: Trajectory,
        session: StructuredSession,
        event_log: EventLog,
        workspace_expected: WorkspaceExpected,
        artifact_index_hash: str,
        checkpoint_kind: str,
        recovery: dict[str, Any] | None = None,
    ) -> CheckpointManifest:
        session.updated_at = utcnow()
        state_path = self.session_dir / "task-state.json"
        tool_path = self.session_dir / "tool-state.json"
        traj_path = self.session_dir / "trajectory.json"
        session_path = self.session_dir / "session.json"

        task_hash = _write_json_atomic(state_path, task_state.to_dict())
        tool_hash = _write_json_atomic(tool_path, tool_state.to_dict())
        traj_hash = _write_json_atomic(traj_path, trajectory.to_dict())

        previous = self._latest_seq()
        seq = previous + 1
        state_refs = {
            "task_state": {"sha256": task_hash, "size": state_path.stat().st_size},
            "tool_state": {"sha256": tool_hash, "size": tool_path.stat().st_size},
            "trajectory": {"sha256": traj_hash, "size": traj_path.stat().st_size},
        }
        manifest = CheckpointManifest(
            checkpoint_seq=seq,
            parent_checkpoint_seq=previous if previous >= 0 else -1,
            checkpoint_kind=checkpoint_kind,
            created_at=utcnow(),
            epoch_id=trajectory.epoch_id,
            log_anchor=event_log.anchor(),
            state_refs=state_refs,
            runtime_cursor=session.runtime_cursor,
            workspace_expected=workspace_expected.to_dict(),
            config_fingerprint=session.config_fingerprint,
            artifact_anchor={"artifact_index_hash": artifact_index_hash},
            recovery=dict(recovery or {}),
        )
        manifest.checkpoint_id = sha256_text(canonical_json(manifest.to_dict()))
        session.last_checkpoint = {
            "seq": seq,
            "checkpoint_id": manifest.checkpoint_id,
            "created_at": manifest.created_at,
        }
        _write_json_atomic(session_path, session.to_dict())
        manifest_path = self.checkpoint_dir / f"ckpt-{seq:06d}.json"
        _write_json_atomic(manifest_path, manifest.to_dict())

        current_tmp = self.checkpoint_dir / ".CURRENT.tmp"
        current_tmp.write_text(f"{manifest_path.name}\n", encoding="utf-8")
        with current_tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(current_tmp, self.current_path)
        return manifest

    def _latest_seq(self) -> int:
        if self.current_path.exists():
            name = self.current_path.read_text(encoding="utf-8").strip()
            path = self.checkpoint_dir / name
            if path.exists():
                try:
                    return int(json.loads(path.read_text(encoding="utf-8"))["checkpoint_seq"])
                except (KeyError, ValueError, json.JSONDecodeError):
                    raise CheckpointError(f"Invalid CURRENT checkpoint {name}")
        manifests = sorted(self.checkpoint_dir.glob("ckpt-*.json"))
        if manifests:
            return len(manifests) - 1
        return -1

    def load_latest(self) -> tuple[CheckpointManifest, TaskState, ToolState, Trajectory, StructuredSession]:
        if not self.current_path.exists():
            raise CheckpointError("No checkpoint available")
        name = self.current_path.read_text(encoding="utf-8").strip()
        manifest_path = self.checkpoint_dir / name
        if not manifest_path.exists():
            raise CheckpointError(f"CURRENT points to missing checkpoint {name}")
        manifest = CheckpointManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        self._validate_refs(manifest)
        task_state = TaskState.from_dict(json.loads((self.session_dir / "task-state.json").read_text(encoding="utf-8")))
        tool_state = ToolState.from_dict(json.loads((self.session_dir / "tool-state.json").read_text(encoding="utf-8")))
        trajectory = Trajectory.from_dict(json.loads((self.session_dir / "trajectory.json").read_text(encoding="utf-8")))
        session = StructuredSession.from_dict(json.loads((self.session_dir / "session.json").read_text(encoding="utf-8")))
        return manifest, task_state, tool_state, trajectory, session

    def _validate_refs(self, manifest: CheckpointManifest) -> None:
        for key, filename in {
            "task_state": "task-state.json",
            "tool_state": "tool-state.json",
            "trajectory": "trajectory.json",
        }.items():
            ref = manifest.state_refs.get(key) or {}
            path = self.session_dir / filename
            if not path.exists():
                raise CheckpointError(f"Missing state file {path}")
            expected = ref.get("sha256")
            if expected and sha256_text(path.read_text(encoding="utf-8").rstrip("\n")) != expected:
                raise CheckpointError(f"State file hash mismatch: {filename}")
