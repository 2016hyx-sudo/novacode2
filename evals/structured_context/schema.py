"""Serializable data models shared by structured-context evaluation modes.

The evaluator deliberately keeps its interchange format small and based on
plain JSON.  These models are therefore intentionally tolerant when reading
older result files: unknown keys are preserved in ``metadata`` and the common
token field spellings used by early prototypes are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping


EVAL_SCHEMA_VERSION = "1.0"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _str(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


@dataclass(frozen=True)
class Scenario:
    """A serializable full-run scenario declaration.

    ``offline_cases.json`` intentionally uses a smaller recipe schema, but
    using this model for full-run records keeps Phase 1 and Phase 2 output
    compatible.
    """

    id: str
    bucket: str
    task: str = ""
    title: str = ""
    fixture: dict[str, Any] = field(default_factory=dict)
    setup_commands: tuple[str, ...] = ()
    verification_commands: tuple[str, ...] = ()
    expected: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    pressure: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: ClassVar[str] = EVAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "bucket": self.bucket,
            "title": self.title,
            "task": self.task,
            "fixture": dict(self.fixture),
            "setup_commands": list(self.setup_commands),
            "verification_commands": list(self.verification_commands),
            "expected": dict(self.expected),
            "limits": dict(self.limits),
            "tags": list(self.tags),
            "pressure": dict(self.pressure),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Scenario":
        known = {
            "schema_version",
            "id",
            "bucket",
            "title",
            "task",
            "fixture",
            "setup_commands",
            "verification_commands",
            "expected",
            "limits",
            "tags",
            "pressure",
            "metadata",
        }
        metadata = _as_dict(value.get("metadata"))
        metadata.update({key: item for key, item in value.items() if key not in known})
        return cls(
            id=_str(value.get("id")),
            bucket=_str(value.get("bucket")),
            title=_str(value.get("title")),
            task=_str(value.get("task")),
            fixture=_as_dict(value.get("fixture")),
            setup_commands=tuple(_str(item) for item in _as_list(value.get("setup_commands"))),
            verification_commands=tuple(
                _str(item) for item in _as_list(value.get("verification_commands"))
            ),
            expected=_as_dict(value.get("expected")),
            limits=_as_dict(value.get("limits")),
            tags=tuple(_str(item) for item in _as_list(value.get("tags"))),
            pressure=_as_dict(value.get("pressure")),
            metadata=metadata,
        )


@dataclass(frozen=True)
class PromptVariant:
    """One replayed prompt representation and its deterministic fingerprint."""

    name: str
    messages: tuple[dict[str, Any], ...]
    tokens: int
    payload_hash: str
    layers: dict[str, int] = field(default_factory=dict)

    def to_dict(self, *, include_messages: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "tokens": int(self.tokens),
            "payload_hash": self.payload_hash,
            "layers": {key: int(value) for key, value in sorted(self.layers.items())},
        }
        if include_messages:
            result["messages"] = [dict(message) for message in self.messages]
        return result


@dataclass(frozen=True)
class RequestMetric:
    """One provider-attempt-sized measurement record.

    The three replay token fields are available on every offline record.  A
    live-run record may only have ``logical_input_tokens``; keeping both forms
    makes report aggregation provider-neutral while preserving raw usage.
    """

    task_case_id: str
    request_id: str
    raw_full_tokens: int = 0
    observation_full_tokens: int = 0
    structured_tokens: int = 0
    bucket: str = ""
    run_id: str = ""
    repetition: int = 0
    variant: str = "structured"
    session_id: str = ""
    parent_request_id: str | None = None
    agent_role: str = "main"
    step: int = 0
    attempt: int = 1
    epoch_id: int = 0
    provider: str = "offline"
    model: str = "deterministic"
    cache_mode: str = "unknown"
    status: str = "success"
    logical_input_tokens: int = 0
    fresh_processed_input_tokens: int = 0
    cache_hit_tokens: int = 0
    fold_input_tokens: int = 0
    token_source: str = "estimated"
    payload_hashes: dict[str, str] = field(default_factory=dict)
    layers: dict[str, int] = field(default_factory=dict)
    replay: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: ClassVar[str] = EVAL_SCHEMA_VERSION

    @property
    def raw_full_input_tokens(self) -> int:
        return self.raw_full_tokens

    @property
    def observation_full_input_tokens(self) -> int:
        return self.observation_full_tokens

    @property
    def structured_input_tokens(self) -> int:
        return self.structured_tokens or self.logical_input_tokens

    @property
    def effective_logical_input_tokens(self) -> int:
        """Use provider usage when present, otherwise structured replay tokens."""
        return self.logical_input_tokens or self.structured_tokens

    def identity(self) -> tuple[str, int, int, int, str, str]:
        """Stable identity for deterministic replay/baseline joins."""
        return (
            self.task_case_id,
            int(self.repetition),
            int(self.step),
            int(self.attempt),
            self.agent_role,
            self.request_id,
        )

    def sort_key(self) -> tuple[str, int, int, int, str, str, str]:
        return (*self.identity(), self.variant)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_case_id": self.task_case_id,
            "bucket": self.bucket,
            "repetition": int(self.repetition),
            "variant": self.variant,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "parent_request_id": self.parent_request_id,
            "agent_role": self.agent_role,
            "step": int(self.step),
            "attempt": int(self.attempt),
            "epoch_id": int(self.epoch_id),
            "provider": self.provider,
            "model": self.model,
            "cache_mode": self.cache_mode,
            "status": self.status,
            "token_source": self.token_source,
            "raw_full_tokens": int(self.raw_full_tokens),
            "observation_full_tokens": int(self.observation_full_tokens),
            "structured_tokens": int(self.structured_tokens),
            "logical_input_tokens": int(self.logical_input_tokens),
            "fresh_processed_input_tokens": int(self.fresh_processed_input_tokens),
            "cache_hit_tokens": int(self.cache_hit_tokens),
            "fold_input_tokens": int(self.fold_input_tokens),
            "payload_hashes": dict(sorted(self.payload_hashes.items())),
            "layers": {key: int(item) for key, item in sorted(self.layers.items())},
            "replay": dict(self.replay),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RequestMetric":
        # Accept the explicit input-token spellings from the evaluation design
        # as well as the concise spellings written by this implementation.
        tokens = _as_dict(value.get("tokens"))
        raw = value.get("raw_full_tokens", value.get("raw_full_input_tokens", tokens.get("raw_full", 0)))
        observation = value.get(
            "observation_full_tokens",
            value.get("observation_full_input_tokens", tokens.get("observation_full", 0)),
        )
        structured = value.get(
            "structured_tokens",
            value.get("structured_input_tokens", tokens.get("structured", 0)),
        )
        known = {
            "schema_version", "run_id", "task_case_id", "case_id", "bucket", "repetition", "variant",
            "session_id", "request_id", "parent_request_id", "agent_role", "step", "attempt", "epoch_id",
            "provider", "model", "cache_mode", "status", "token_source", "raw_full_tokens",
            "raw_full_input_tokens", "observation_full_tokens", "observation_full_input_tokens",
            "structured_tokens", "structured_input_tokens", "logical_input_tokens",
            "fresh_processed_input_tokens", "cache_hit_tokens", "fold_input_tokens", "payload_hashes",
            "layers", "replay", "metadata", "tokens",
        }
        metadata = _as_dict(value.get("metadata"))
        metadata.update({key: item for key, item in value.items() if key not in known})
        return cls(
            task_case_id=_str(value.get("task_case_id", value.get("case_id", ""))),
            request_id=_str(value.get("request_id")),
            raw_full_tokens=max(0, _int(raw)),
            observation_full_tokens=max(0, _int(observation)),
            structured_tokens=max(0, _int(structured)),
            bucket=_str(value.get("bucket")),
            run_id=_str(value.get("run_id")),
            repetition=_int(value.get("repetition")),
            variant=_str(value.get("variant"), "structured"),
            session_id=_str(value.get("session_id")),
            parent_request_id=(
                _str(value.get("parent_request_id")) if value.get("parent_request_id") is not None else None
            ),
            agent_role=_str(value.get("agent_role"), "main"),
            step=_int(value.get("step")),
            attempt=max(1, _int(value.get("attempt"), 1)),
            epoch_id=_int(value.get("epoch_id")),
            provider=_str(value.get("provider"), "offline"),
            model=_str(value.get("model"), "deterministic"),
            cache_mode=_str(value.get("cache_mode"), "unknown"),
            status=_str(value.get("status"), "success"),
            logical_input_tokens=max(0, _int(value.get("logical_input_tokens"))),
            fresh_processed_input_tokens=max(0, _int(value.get("fresh_processed_input_tokens"))),
            cache_hit_tokens=max(0, _int(value.get("cache_hit_tokens"))),
            fold_input_tokens=max(0, _int(value.get("fold_input_tokens"))),
            token_source=_str(value.get("token_source"), "estimated"),
            payload_hashes={str(key): _str(item) for key, item in _as_dict(value.get("payload_hashes")).items()},
            layers={str(key): _int(item) for key, item in _as_dict(value.get("layers")).items()},
            replay=_as_dict(value.get("replay")),
            metadata=metadata,
        )


@dataclass(frozen=True)
class RunResult:
    """The serializable result of one offline or full-run evaluation."""

    run_id: str
    mode: str
    requests: tuple[RequestMetric, ...] = ()
    tasks: tuple[dict[str, Any], ...] = ()
    manifest: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version: ClassVar[str] = EVAL_SCHEMA_VERSION

    @property
    def request_metrics(self) -> tuple[RequestMetric, ...]:
        return self.requests

    def sorted_requests(self) -> tuple[RequestMetric, ...]:
        return tuple(sorted(self.requests, key=lambda item: item.sort_key()))

    def to_dict(self, *, include_requests: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "mode": self.mode,
            "tasks": [dict(item) for item in self.tasks],
            "manifest": dict(self.manifest),
            "metadata": dict(self.metadata),
        }
        if include_requests:
            result["requests"] = [item.to_dict() for item in self.sorted_requests()]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunResult":
        return cls(
            run_id=_str(value.get("run_id")),
            mode=_str(value.get("mode")),
            requests=tuple(
                RequestMetric.from_dict(item)
                for item in _as_list(value.get("requests", value.get("request_metrics", [])))
                if isinstance(item, Mapping)
            ),
            tasks=tuple(_as_dict(item) for item in _as_list(value.get("tasks"))),
            manifest=_as_dict(value.get("manifest")),
            metadata=_as_dict(value.get("metadata")),
        )


@dataclass(frozen=True)
class GateResult:
    """A pure baseline-comparison outcome suitable for CI serialization."""

    passed: bool
    failures: tuple[dict[str, Any], ...] = ()
    checks: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "failures": [dict(item) for item in self.failures],
            "checks": [dict(item) for item in self.checks],
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

