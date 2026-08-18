"""Data models for the structured context / checkpoint-resume subsystem."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..llm.base import Message


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- task state


@dataclass
class PlanStep:
    id: str
    text: str
    status: str = "pending"
    depends_on: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "depends_on": list(self.depends_on),
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanStep:
        return cls(
            id=str(data["id"]),
            text=str(data.get("text", "")),
            status=str(data.get("status", "pending")),
            depends_on=list(data.get("depends_on") or []),
            evidence_refs=list(data.get("evidence_refs") or []),
        )


@dataclass
class ProgressItem:
    id: str
    text: str
    completed_step: int
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "completed_step": self.completed_step,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressItem:
        return cls(
            id=str(data["id"]),
            text=str(data.get("text", "")),
            completed_step=int(data.get("completed_step", 0)),
            evidence_refs=list(data.get("evidence_refs") or []),
        )


@dataclass
class Finding:
    id: str
    fact: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    status: str = "valid"
    updated_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fact": self.fact,
            "evidence": list(self.evidence),
            "status": self.status,
            "updated_step": self.updated_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        return cls(
            id=str(data["id"]),
            fact=str(data.get("fact", "")),
            evidence=list(data.get("evidence") or []),
            status=str(data.get("status", "valid")),
            updated_step=int(data.get("updated_step", 0)),
        )


@dataclass
class Decision:
    id: str
    decision: str
    reason: str
    evidence: list[str] = field(default_factory=list)
    status: str = "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "decision": self.decision,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decision:
        return cls(
            id=str(data["id"]),
            decision=str(data.get("decision", "")),
            reason=str(data.get("reason", "")),
            evidence=list(data.get("evidence") or []),
            status=str(data.get("status", "valid")),
        )


@dataclass
class UnresolvedItem:
    id: str
    text: str
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "blocked": self.blocked}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnresolvedItem:
        return cls(
            id=str(data["id"]),
            text=str(data.get("text", "")),
            blocked=bool(data.get("blocked", False)),
        )


@dataclass
class KeySequence:
    """A significant action sequence / exploratory pattern worth keeping.

    Unlike a Finding (a durable fact), a sequence records *how* the agent moved
    — inverse action pairs, repeated maneuvers, blocked attempts — plus the
    inferred intent, which later questions about strategy can only answer from
    the pattern itself.
    """

    id: str
    pattern: str  # what happened: the actions, with step numbers
    intent: str  # why / what the agent was testing or achieving
    step_range: str = ""
    status: str = "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "intent": self.intent,
            "step_range": self.step_range,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeySequence:
        return cls(
            id=str(data["id"]),
            pattern=str(data.get("pattern", "")),
            intent=str(data.get("intent", "")),
            step_range=str(data.get("step_range", "")),
            status=str(data.get("status", "valid")),
        )


@dataclass
class TaskState:
    schema_version: str = "1.0"
    task_id: str = ""
    objective: str = ""
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    completed: list[ProgressItem] = field(default_factory=list)
    current: str = ""
    remaining: list[PlanStep] = field(default_factory=list)
    key_findings: list[Finding] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    unresolved: list[UnresolvedItem] = field(default_factory=list)
    key_sequences: list[KeySequence] = field(default_factory=list)
    extensions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, *, task_id: str, objective: str) -> TaskState:
        return cls(schema_version="1.0", task_id=task_id, objective=objective)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "objective": self.objective,
            "constraints": list(self.constraints),
            "success_criteria": list(self.success_criteria),
            "progress": {
                "completed": [item.to_dict() for item in self.completed],
                "current": self.current,
                "remaining": [item.to_dict() for item in self.remaining],
            },
            "key_findings": [item.to_dict() for item in self.key_findings],
            "decisions": [item.to_dict() for item in self.decisions],
            "unresolved": [item.to_dict() for item in self.unresolved],
            "key_sequences": [item.to_dict() for item in self.key_sequences],
            "extensions": dict(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskState:
        progress = data.get("progress") or {}
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            task_id=str(data.get("task_id", "")),
            objective=str(data.get("objective", "")),
            constraints=[str(x) for x in data.get("constraints") or []],
            success_criteria=[str(x) for x in data.get("success_criteria") or []],
            completed=[ProgressItem.from_dict(x) for x in progress.get("completed") or []],
            current=str(progress.get("current", "")),
            remaining=[PlanStep.from_dict(x) for x in progress.get("remaining") or []],
            key_findings=[Finding.from_dict(x) for x in data.get("key_findings") or []],
            decisions=[Decision.from_dict(x) for x in data.get("decisions") or []],
            unresolved=[UnresolvedItem.from_dict(x) for x in data.get("unresolved") or []],
            key_sequences=[KeySequence.from_dict(x) for x in data.get("key_sequences") or []],
            extensions=dict(data.get("extensions") or {}),
        )


@dataclass
class ToolExperience:
    id: str
    kind: str
    value: dict[str, Any] = field(default_factory=dict)
    status: str = "valid"
    updated_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "value": dict(self.value),
            "status": self.status,
            "updated_step": self.updated_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolExperience:
        return cls(
            id=str(data["id"]),
            kind=str(data.get("kind", "")),
            value=dict(data.get("value") or {}),
            status=str(data.get("status", "valid")),
            updated_step=int(data.get("updated_step", 0)),
        )


@dataclass
class ToolState:
    schema_version: str = "1.0"
    profiles: dict[str, dict[str, list[ToolExperience]]] = field(default_factory=dict)
    evidence_index: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def new(cls) -> ToolState:
        return cls(
            profiles={
                "grep": {"useful_scopes": [], "effective_queries": [], "known_error_patterns": []},
                "read": {"useful_files": [], "effective_ranges": [], "known_symbols": []},
                "shell": {"effective_commands": [], "known_failures": []},
                "test": {"effective_commands": [], "known_failures": []},
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profiles": {
                tool: {
                    key: [item.to_dict() for item in entries]
                    for key, entries in profile.items()
                }
                for tool, profile in self.profiles.items()
            },
            "evidence_index": [dict(x) for x in self.evidence_index],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolState:
        profiles: dict[str, dict[str, list[ToolExperience]]] = {}
        for tool, profile in (data.get("profiles") or {}).items():
            profiles[str(tool)] = {
                str(key): [ToolExperience.from_dict(x) for x in entries or []]
                for key, entries in profile.items()
            }
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            profiles=profiles,
            evidence_index=[dict(x) for x in data.get("evidence_index") or []],
        )


# --------------------------------------------------------------------------- trajectory


@dataclass
class InteractionGroup:
    group_id: str
    epoch_id: int
    created_step: int
    messages: list[Message] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    workspace_changes: list[dict[str, Any]] = field(default_factory=list)
    protected: bool = False
    protected_reasons: list[str] = field(default_factory=list)
    token_count: int = 0
    source_events: dict[str, int] = field(default_factory=dict)
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "epoch_id": self.epoch_id,
            "created_step": self.created_step,
            "status": self.status,
            "messages": [message.to_dict() for message in self.messages],
            "corrections": list(self.corrections),
            "workspace_changes": [dict(x) for x in self.workspace_changes],
            "protected": self.protected,
            "protected_reasons": list(self.protected_reasons),
            "token_count": self.token_count,
            "source_events": dict(self.source_events),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionGroup:
        return cls(
            group_id=str(data["group_id"]),
            epoch_id=int(data.get("epoch_id", 0)),
            created_step=int(data.get("created_step", 0)),
            messages=[Message.from_dict(x) for x in data.get("messages") or []],
            corrections=[str(x) for x in data.get("corrections") or []],
            workspace_changes=[dict(x) for x in data.get("workspace_changes") or []],
            protected=bool(data.get("protected", False)),
            protected_reasons=list(data.get("protected_reasons") or []),
            token_count=int(data.get("token_count", 0)),
            source_events=dict(data.get("source_events") or {}),
            status=str(data.get("status", "open")),
        )


@dataclass
class Trajectory:
    schema_version: str = "1.0"
    epoch_id: int = 0
    groups: list[InteractionGroup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epoch_id": self.epoch_id,
            "group_count": len(self.groups),
            "groups": [group.to_dict() for group in self.groups],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            epoch_id=int(data.get("epoch_id", 0)),
            groups=[InteractionGroup.from_dict(x) for x in data.get("groups") or []],
        )


# --------------------------------------------------------------------------- workspace / drift


@dataclass
class WorkspaceExpected:
    workspace_root: str = ""
    branch: str = ""
    head: str = ""
    status_hash: str = ""
    expected_dirty: list[dict[str, Any]] = field(default_factory=list)
    expected_untracked: list[dict[str, Any]] = field(default_factory=list)
    postconditions: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""

    def recompute_fingerprint(self) -> str:
        payload = {
            "workspace_root": self.workspace_root,
            "branch": self.branch,
            "head": self.head,
            "status_hash": self.status_hash,
            "expected_dirty": self.expected_dirty,
            "expected_untracked": self.expected_untracked,
            "postconditions": self.postconditions,
        }
        return sha256_text(canonical_json(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "workspace_root": self.workspace_root,
            "git": {
                "present": bool(self.head or self.branch),
                "branch": self.branch,
                "head": self.head,
                "status_hash": self.status_hash,
            },
            "expected_dirty": [dict(x) for x in self.expected_dirty],
            "expected_untracked": [dict(x) for x in self.expected_untracked],
            "postconditions": [dict(x) for x in self.postconditions],
            "workspace_fingerprint": self.fingerprint or self.recompute_fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceExpected:
        git = data.get("git") or {}
        return cls(
            workspace_root=str(data.get("workspace_root", "")),
            branch=str(git.get("branch", "")),
            head=str(git.get("head", "")),
            status_hash=str(git.get("status_hash", "")),
            expected_dirty=[dict(x) for x in data.get("expected_dirty") or []],
            expected_untracked=[dict(x) for x in data.get("expected_untracked") or []],
            postconditions=[dict(x) for x in data.get("postconditions") or []],
            fingerprint=str(data.get("workspace_fingerprint", "")),
        )


@dataclass
class DriftReport:
    severity: str = "NONE"  # NONE | IGNORED | LOW | HIGH | STRUCTURAL
    unexpected_changes: list[dict[str, Any]] = field(default_factory=list)
    missing_expected_changes: list[dict[str, Any]] = field(default_factory=list)
    hash_mismatches: list[dict[str, Any]] = field(default_factory=list)
    git_divergence: dict[str, Any] | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "unexpected_changes": [dict(x) for x in self.unexpected_changes],
            "missing_expected_changes": [dict(x) for x in self.missing_expected_changes],
            "hash_mismatches": [dict(x) for x in self.hash_mismatches],
            "git_divergence": dict(self.git_divergence) if self.git_divergence else None,
            "summary": self.summary,
        }


# --------------------------------------------------------------------------- checkpoint / session


@dataclass
class CheckpointManifest:
    schema_version: str = "1.0"
    checkpoint_seq: int = 0
    checkpoint_id: str = ""
    parent_checkpoint_seq: int = -1
    checkpoint_kind: str = "periodic"
    created_at: str = ""
    epoch_id: int = 0
    log_anchor: dict[str, Any] = field(default_factory=dict)
    state_refs: dict[str, Any] = field(default_factory=dict)
    runtime_cursor: dict[str, Any] = field(default_factory=dict)
    workspace_expected: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: dict[str, Any] = field(default_factory=dict)
    artifact_anchor: dict[str, Any] = field(default_factory=dict)
    recovery: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_seq": self.checkpoint_seq,
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_seq": self.parent_checkpoint_seq,
            "checkpoint_kind": self.checkpoint_kind,
            "created_at": self.created_at,
            "epoch_id": self.epoch_id,
            "log_anchor": dict(self.log_anchor),
            "state_refs": dict(self.state_refs),
            "runtime_cursor": dict(self.runtime_cursor),
            "workspace_expected": dict(self.workspace_expected),
            "config_fingerprint": dict(self.config_fingerprint),
            "artifact_anchor": dict(self.artifact_anchor),
            "recovery": dict(self.recovery),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointManifest:
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            checkpoint_seq=int(data.get("checkpoint_seq", 0)),
            checkpoint_id=str(data.get("checkpoint_id", "")),
            parent_checkpoint_seq=int(data.get("parent_checkpoint_seq", -1)),
            checkpoint_kind=str(data.get("checkpoint_kind", "periodic")),
            created_at=str(data.get("created_at", "")),
            epoch_id=int(data.get("epoch_id", 0)),
            log_anchor=dict(data.get("log_anchor") or {}),
            state_refs=dict(data.get("state_refs") or {}),
            runtime_cursor=dict(data.get("runtime_cursor") or {}),
            workspace_expected=dict(data.get("workspace_expected") or {}),
            config_fingerprint=dict(data.get("config_fingerprint") or {}),
            artifact_anchor=dict(data.get("artifact_anchor") or {}),
            recovery=dict(data.get("recovery") or {}),
        )


@dataclass
class StructuredSession:
    schema_version: str = "1.0"
    session_id: str = ""
    status: str = "running"
    created_at: str = ""
    updated_at: str = ""
    provider: str = ""
    model: str = ""
    workspace_root: str = ""
    user_task: str = ""
    epoch_id: int = 0
    runtime_cursor: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: dict[str, Any] = field(default_factory=dict)
    last_checkpoint: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.session_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provider": self.provider,
            "model": self.model,
            "workspace_root": self.workspace_root,
            "user_task": self.user_task,
            "epoch_id": self.epoch_id,
            "runtime_cursor": dict(self.runtime_cursor),
            "config_fingerprint": dict(self.config_fingerprint),
            "last_checkpoint": dict(self.last_checkpoint),
            "metrics": dict(self.metrics),
            "plan": list(self.plan),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredSession:
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            session_id=str(data.get("session_id", "")),
            status=str(data.get("status", "running")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            workspace_root=str(data.get("workspace_root", "")),
            user_task=str(data.get("user_task", "")),
            epoch_id=int(data.get("epoch_id", 0)),
            runtime_cursor=dict(data.get("runtime_cursor") or {}),
            config_fingerprint=dict(data.get("config_fingerprint") or {}),
            last_checkpoint=dict(data.get("last_checkpoint") or {}),
            metrics=dict(data.get("metrics") or {}),
            plan=[str(x) for x in data.get("plan") or []],
        )
