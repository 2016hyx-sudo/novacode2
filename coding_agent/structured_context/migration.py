"""One-time migration from legacy ``.sessions/<id>.json`` sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..context.session import SessionStore
from ..llm.base import Message
from .event_log import EventLog
from .models import (
    InteractionGroup,
    PlanStep,
    StructuredSession,
    TaskState,
    ToolState,
    Trajectory,
    WorkspaceExpected,
    utcnow,
)
from .session_store import StructuredSessionStore


def migrate_legacy_session(
    legacy_dir: Path,
    session_id: str,
    structured_store: StructuredSessionStore,
    *,
    workspace_root: Path,
    system_prompt: str,
    prefix_hash: str,
    tools_hash: str,
) -> Any:
    """Convert one legacy session into a structured session directory.

    Returns the newly created :class:`StructuredContext` so the caller can save
    a terminal/recovery checkpoint if desired.  The legacy file is left in
    place untouched.
    """
    legacy_store = SessionStore(Path(legacy_dir))
    legacy = legacy_store.load(session_id)
    now = utcnow()
    directory = structured_store.session_dir(session_id)
    directory.mkdir(parents=True, exist_ok=False)
    event_log = EventLog(directory / "events.jsonl", fsync=structured_store.fsync)
    event_log.append("migration_start", {"legacy_session_file": str(legacy_store.directory / f"{session_id}.json")})

    from .artifact_store import ArtifactStore
    from .structured_context import StructuredContext, StructuredContextConfig
    from .token_counter import TokenCounter
    from .trajectory_archive import TrajectoryArchive
    from .workspace import WorkspaceFingerprint

    fingerprint = WorkspaceFingerprint(workspace_root)
    expected: WorkspaceExpected = fingerprint.expected_from_actual()
    session = StructuredSession(
        session_id=legacy.id,
        status=legacy.status,
        created_at=legacy.created_at or now,
        updated_at=now,
        provider=legacy.provider,
        model=legacy.model,
        workspace_root=str(Path(workspace_root).resolve()),
        user_task=legacy.user_task,
        runtime_cursor={
            "position": {"user_turn": 0, "step": 0, "phase": "exploration", "status": legacy.status},
            "verification": {"required": False, "status": "not_required"},
            "limits": {
                "steps_used": int(legacy.metadata.get("last_steps_used", 0) or 0),
                "tool_calls_used": int(legacy.metadata.get("last_tool_calls_used", 0) or 0),
            },
            "recovery": {"blocked": False, "resumable": True, "resume_action": "continue_next_action"},
        },
        config_fingerprint={
            "prefix_hash": prefix_hash,
            "tools_hash": tools_hash,
            "context_window_limit": 256_000,
        },
        metrics={"migrated_from_legacy": True, "legacy_metadata": legacy.metadata},
        plan=list(legacy.plan or []),
    )
    task_state = TaskState.new(task_id=legacy.id, objective=legacy.user_task)
    if legacy.plan:
        task_state.remaining = [
            PlanStep(id=f"plan-{i + 1}", text=text, status="active" if i == 0 else "pending")
            for i, text in enumerate(legacy.plan)
        ]
        task_state.current = legacy.plan[0]
    tool_state = ToolState.new()
    trajectory = Trajectory(epoch_id=0)
    _migrate_messages(legacy.messages, trajectory)

    context = StructuredContext(
        system_prompt=system_prompt,
        session=session,
        task_state=task_state,
        tool_state=tool_state,
        trajectory=trajectory,
        event_log=event_log,
        artifact_store=ArtifactStore(directory / "artifacts", fsync=structured_store.fsync),
        workspace_fingerprint=fingerprint,
        token_counter=TokenCounter(),
        trajectory_archive=TrajectoryArchive(directory / "trajectory-archive.jsonl", fsync=structured_store.fsync),
        config=StructuredContextConfig(),
        prefix_hash=prefix_hash,
        tools_hash=tools_hash,
        workspace_expected=expected,
    )
    event_log.append("migration_complete", {"message_count": len(legacy.messages)})
    from .checkpoint import CheckpointManager

    manager = CheckpointManager(directory, fsync=structured_store.fsync)
    manager.save(
        task_state=task_state,
        tool_state=tool_state,
        trajectory=trajectory,
        session=session,
        event_log=event_log,
        workspace_expected=expected,
        artifact_index_hash=structured_store._artifact_index_hash(directory),
        checkpoint_kind="baseline",
    )
    return context


def _migrate_messages(messages: list[Message], trajectory: Trajectory) -> None:
    from .token_counter import TokenCounter

    counter = TokenCounter()
    current: InteractionGroup | None = None
    pending_assistant: Message | None = None

    def close_group() -> None:
        nonlocal current, pending_assistant
        if current is not None:
            current.status = "complete"
        current = None
        pending_assistant = None

    for index, message in enumerate(messages):
        if message.role == "system":
            continue
        if message.role == "user":
            close_group()
            current = InteractionGroup(
                group_id=f"g-{index + 1:06d}",
                epoch_id=trajectory.epoch_id,
                created_step=0,
                protected=False,
            )
            trajectory.groups.append(current)
            current.messages.append(message)
            current.token_count += counter.estimate_message(message)
        elif message.role == "assistant":
            if current is None:
                current = InteractionGroup(group_id=f"g-{index + 1:06d}", epoch_id=trajectory.epoch_id, created_step=0)
                trajectory.groups.append(current)
            current.messages.append(message)
            current.token_count += counter.estimate_message(message)
            pending_assistant = message
        elif message.role == "tool":
            if current is None:
                current = InteractionGroup(group_id=f"g-{index + 1:06d}", epoch_id=trajectory.epoch_id, created_step=0)
                trajectory.groups.append(current)
            current.messages.append(message)
            current.token_count += counter.estimate_message(message)
            if pending_assistant is not None:
                call_ids = {call.id for call in pending_assistant.tool_calls if call.id}
                if not call_ids or message.tool_call_id in call_ids:
                    pending_assistant = None
    close_group()
