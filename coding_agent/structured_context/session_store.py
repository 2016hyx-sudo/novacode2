"""Structured session storage: create, load, checkpoint and list sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact_store import ArtifactStore
from .checkpoint import CheckpointManager
from .event_log import EventLog
from .event_replay import EventReplayer
from .models import (
    StructuredSession,
    TaskState,
    ToolState,
    Trajectory,
    WorkspaceExpected,
    sha256_text,
    utcnow,
)
from .structured_context import StructuredContext, StructuredContextConfig
from .token_counter import TokenCounter
from .trajectory_archive import TrajectoryArchive
from .workspace import WorkspaceFingerprint


class StructuredSessionStore:
    def __init__(self, root: Path, *, fsync: bool = True) -> None:
        self.root = Path(root)
        self.fsync = fsync

    def session_dir(self, session_id: str) -> Path:
        if Path(session_id).name != session_id:
            raise ValueError(f"Invalid session id: {session_id!r}")
        return self.root / session_id

    def create_context(
        self,
        *,
        session_id: str,
        user_task: str,
        provider: str,
        model: str,
        workspace_root: Path,
        system_prompt: str,
        prefix_hash: str,
        tools_hash: str,
        context_config: StructuredContextConfig | None = None,
        context_window_limit: int = 256_000,
        model_hard_limit: int | None = None,
    ) -> StructuredContext:
        directory = self.session_dir(session_id)
        directory.mkdir(parents=True, exist_ok=False)
        event_log = EventLog(directory / "events.jsonl", fsync=self.fsync)
        event_log.append("session_start", {"user_task": user_task, "provider": provider, "model": model})
        artifact_store = ArtifactStore(directory / "artifacts", fsync=self.fsync)
        fingerprint = WorkspaceFingerprint(workspace_root)
        expected = fingerprint.expected_from_actual()
        session = StructuredSession(
            session_id=session_id,
            created_at=utcnow(),
            updated_at=utcnow(),
            provider=provider,
            model=model,
            workspace_root=str(Path(workspace_root).resolve()),
            user_task=user_task,
            runtime_cursor={
                "position": {"user_turn": 0, "step": 0, "phase": "exploration", "status": "running"},
                "verification": {"required": False, "status": "not_required"},
                "limits": {"steps_used": 0, "tool_calls_used": 0},
                "recovery": {"blocked": False, "resumable": True, "resume_action": "continue_next_action"},
            },
            config_fingerprint={
                "prefix_hash": prefix_hash,
                "tools_hash": tools_hash,
                "context_window_limit": context_window_limit,
                "model_hard_limit": model_hard_limit,
            },
        )
        task_state = TaskState.new(task_id=session_id, objective=user_task)
        tool_state = ToolState.new()
        trajectory = Trajectory(epoch_id=0)
        context = StructuredContext(
            system_prompt=system_prompt,
            session=session,
            task_state=task_state,
            tool_state=tool_state,
            trajectory=trajectory,
            event_log=event_log,
            artifact_store=artifact_store,
            workspace_fingerprint=fingerprint,
            token_counter=TokenCounter(),
            trajectory_archive=TrajectoryArchive(directory / "trajectory-archive.jsonl", fsync=self.fsync),
            config=context_config or StructuredContextConfig(max_context_tokens=context_window_limit),
            prefix_hash=prefix_hash,
            tools_hash=tools_hash,
            workspace_expected=expected,
        )
        manager = CheckpointManager(directory, fsync=self.fsync)
        manager.save(
            task_state=task_state,
            tool_state=tool_state,
            trajectory=trajectory,
            session=session,
            event_log=event_log,
            workspace_expected=expected,
            artifact_index_hash=self._artifact_index_hash(directory),
            checkpoint_kind="baseline",
        )
        return context

    def load_context(
        self,
        session_id: str,
        *,
        system_prompt: str,
        context_config: StructuredContextConfig | None = None,
    ) -> StructuredContext:
        directory = self.session_dir(session_id)
        manager = CheckpointManager(directory, fsync=self.fsync)
        manifest, task_state, tool_state, trajectory, session = manager.load_latest()
        event_log = EventLog(directory / "events.jsonl", fsync=self.fsync)
        if not event_log.validate_anchor(manifest.log_anchor):
            raise RuntimeError(f"Event log anchor mismatch for session {session_id}")
        workspace_expected = WorkspaceExpected.from_dict(manifest.workspace_expected)
        artifact_store = ArtifactStore(directory / "artifacts", fsync=self.fsync)
        fingerprint = WorkspaceFingerprint(Path(session.workspace_root or manifest.workspace_expected.get("workspace_root", ".")))
        context_config = context_config or StructuredContextConfig(
            max_context_tokens=int(manifest.config_fingerprint.get("context_window_limit", 256_000))
        )
        context = StructuredContext(
            system_prompt=system_prompt,
            session=session,
            task_state=task_state,
            tool_state=tool_state,
            trajectory=trajectory,
            event_log=event_log,
            artifact_store=artifact_store,
            workspace_fingerprint=fingerprint,
            token_counter=TokenCounter.from_dict(session.metrics.get("token_calibration")),
            trajectory_archive=TrajectoryArchive(directory / "trajectory-archive.jsonl", fsync=self.fsync),
            config=context_config,
            prefix_hash=str(session.config_fingerprint.get("prefix_hash", "")),
            tools_hash=str(session.config_fingerprint.get("tools_hash", "")),
            workspace_expected=workspace_expected,
        )
        suffix = event_log.read_since(int(manifest.log_anchor.get("last_event_seq", 0)))
        if suffix:
            replay_result = EventReplayer().replay(context, suffix)
            context.replay_result = replay_result  # type: ignore[attr-defined]
            context.event_log  # suffix is already present in the authoritative log
        return context

    def save_context(
        self,
        context: StructuredContext,
        *,
        checkpoint_kind: str = "periodic",
    ) -> dict[str, Any]:
        context.finalize()
        directory = self.session_dir(context.session.session_id)
        # Capacity control is a safe-cut-point invariant: run it before every
        # snapshot, not only after a fold.
        compact_result = context._state_compactor.compact(
            context.task_state,
            context.tool_state,
            artifact_store=context.artifact_store,
        )
        if compact_result.evicted_task_items or compact_result.evicted_tool_items:
            context.event_log.append("state_compact", compact_result.to_dict())
        expected = context.workspace_expected()
        # Persist token calibration alongside the session snapshot.
        context.session.metrics["token_calibration"] = context.token_counter.calibration.to_dict()
        drift = context.workspace_fingerprint.diff(expected)
        recovery = {
            "resumable": True,
            "resume_action": "continue_next_action",
            "blocked_reason": None,
        }
        if drift.severity in {"HIGH", "STRUCTURAL"}:
            context.event_log.append("drift_detected", drift.to_dict())
            recovery = {
                "resumable": drift.severity != "STRUCTURAL",
                "resume_action": "replan_required" if drift.severity == "HIGH" else "await_user_decision",
                "blocked_reason": drift.summary,
            }
            if checkpoint_kind not in {"blocked", "terminal"}:
                checkpoint_kind = "blocked" if drift.severity == "STRUCTURAL" else "recovery_transition"
        manager = CheckpointManager(directory, fsync=self.fsync)
        manifest = manager.save(
            task_state=context.snapshot_task_state(),
            tool_state=context.snapshot_tool_state(),
            trajectory=context.snapshot_trajectory(),
            session=context.session,
            event_log=context.event_log,
            workspace_expected=expected,
            artifact_index_hash=self._artifact_index_hash(directory),
            checkpoint_kind=checkpoint_kind,
            recovery=recovery,
        )
        context.event_log.append(
            "checkpoint_created",
            {"checkpoint_seq": manifest.checkpoint_seq, "checkpoint_id": manifest.checkpoint_id, "kind": checkpoint_kind},
        )
        # The checkpoint event lands after the manifest anchor; that is intentional
        # and the next checkpoint will cover it.
        return {"manifest": manifest.to_dict(), "drift": drift.to_dict()}

    def list_sessions(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(
            [path for path in self.root.iterdir() if path.is_dir() and (path / "session.json").exists()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    @staticmethod
    def _artifact_index_hash(directory: Path) -> str:
        path = directory / "artifact-index.jsonl"
        if not path.exists():
            return ""
        return sha256_text(path.read_text(encoding="utf-8"))
