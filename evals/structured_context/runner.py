"""Guarded, deterministic full-run evaluation runner.

The runner never constructs a real provider.  A caller must inject a provider
or harness factory explicitly; the default provider fails closed before any
network request.  Each run materializes a source fixture, copies it into a
second workspace, and keeps session/trace state outside both fixture trees.

Generated fixtures are trusted test code. Workspace separation, command
allowlists, and source hashes are regression guards, not an OS security
sandbox for hostile code.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .fixtures import (
    FixtureMaterializer,
    MaterializedFixture,
    copy_fixture,
    materialize_fixture,
    workspace_file_hashes,
    workspace_hash,
)
from .schema import RequestMetric, RunResult
from .scenarios import OracleCheck, Scenario as RecipeScenario


EvaluationVariant = Literal["raw_full", "observation_full", "structured"]
SUPPORTED_VARIANTS: tuple[EvaluationVariant, ...] = (
    "raw_full",
    "observation_full",
    "structured",
)


class RunnerSafetyError(RuntimeError):
    """Raised when a caller attempts an unsafe evaluation operation."""


@dataclass(frozen=True)
class VariantConfig:
    """Explicit variant intent recorded beside every run."""

    name: EvaluationVariant
    structured_context_enabled: bool
    raw_tool_results: bool
    observation_compression: bool
    trajectory_fold: bool


VARIANT_CONFIGS: dict[str, VariantConfig] = {
    "raw_full": VariantConfig(
        name="raw_full",
        structured_context_enabled=False,
        raw_tool_results=True,
        observation_compression=False,
        trajectory_fold=False,
    ),
    "observation_full": VariantConfig(
        name="observation_full",
        structured_context_enabled=False,
        raw_tool_results=False,
        observation_compression=True,
        trajectory_fold=False,
    ),
    "structured": VariantConfig(
        name="structured",
        structured_context_enabled=True,
        raw_tool_results=False,
        observation_compression=True,
        trajectory_fold=True,
    ),
}


@dataclass(frozen=True)
class RunnerConfig:
    variant: EvaluationVariant = "structured"
    work_root: Path | None = None
    keep_workspaces: bool = False
    oracle_timeout_seconds: float = 30.0
    max_steps: int | None = None
    max_tool_calls: int | None = None

    def variant_config(self, variant: str | None = None) -> VariantConfig:
        selected = variant or self.variant
        try:
            return VARIANT_CONFIGS[selected]
        except KeyError as exc:
            raise ValueError(f"unsupported evaluation variant: {selected!r}") from exc


@dataclass(frozen=True)
class OracleCheckResult:
    type: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class OracleResult:
    passed: bool
    checks: tuple[OracleCheckResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [check.to_dict() for check in self.checks]}


@dataclass(frozen=True)
class EvaluationRunResult:
    run_id: str
    scenario_id: str
    bucket: str
    variant: EvaluationVariant
    status: str
    passed: bool
    fixture_hash: str
    workspace_hash_before: str
    workspace_hash_after: str
    source_hash_after: str
    session_id: str | None
    result_text: str
    steps_used: int
    tool_calls_used: int
    oracle: OracleResult
    request_metrics: tuple[RequestMetric, ...] = ()
    provider_trace: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    workspace_path: str | None = None
    session_path: str | None = None
    trace_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def task_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "bucket": self.bucket,
            "variant": self.variant,
            "status": self.status,
            "passed": self.passed,
            "fixture_hash": self.fixture_hash,
            "workspace_hash_before": self.workspace_hash_before,
            "workspace_hash_after": self.workspace_hash_after,
            "source_hash_after": self.source_hash_after,
            "session_id": self.session_id,
            "steps_used": self.steps_used,
            "tool_calls_used": self.tool_calls_used,
            "oracle": self.oracle.to_dict(),
            "error": self.error,
            "workspace_path": self.workspace_path,
            "session_path": self.session_path,
            "trace_path": self.trace_path,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.task_record()
        result["result_text"] = self.result_text
        result["request_metrics"] = [item.to_dict() for item in self.request_metrics]
        result["provider_trace"] = [dict(item) for item in self.provider_trace]
        return result

    def to_run_result(self) -> RunResult:
        return RunResult(
            run_id=self.run_id,
            mode="full-run",
            requests=self.request_metrics,
            tasks=(self.task_record(),),
            manifest={"fixture_hash": self.fixture_hash, "variant": self.variant},
            metadata={"runner": "structured-context", "passed": self.passed},
        )


class DisabledNetworkProvider:
    """Default provider: it is intentionally incapable of real API traffic."""

    def chat(self, messages: Sequence[Any], tools: Sequence[Any] | None = None, *, reasoning_effort: str | None = None) -> Any:
        from coding_agent.llm.base import LLMError

        raise LLMError(
            "Full-run evaluation has no provider. Inject provider_factory or harness_factory; "
            "the default never calls a real API.",
            retryable=False,
        )


class ScriptedProvider:
    """Deterministic LLM stand-in suitable for end-to-end runner smoke tests."""

    def __init__(self, responses: Sequence[Any] | None = None) -> None:
        self.provider_name = "scripted"
        self.model = "deterministic-eval"
        self._responses = list(responses or ())
        self.requests: list[dict[str, Any]] = []
        self._index = 0

    def chat(self, messages: Sequence[Any], tools: Sequence[Any] | None = None, *, reasoning_effort: str | None = None) -> Any:
        from coding_agent.llm.base import LLMResponse

        self.requests.append(
            {
                "index": self._index + 1,
                "message_count": len(messages),
                "tool_count": len(tools or ()),
            }
        )
        if self._index >= len(self._responses):
            return LLMResponse(
                text="Scripted provider completed the task.",
                stop_reason="end_turn",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )
        response = self._responses[self._index]
        self._index += 1
        if callable(response):
            response = response(messages, tools)
        response = copy.deepcopy(response)
        if not getattr(response, "usage", None):
            response.usage = {
                "prompt_tokens": 32 + len(messages),
                "completion_tokens": 8,
                "cached_tokens": 0,
            }
        return response

    @classmethod
    def inclusive_total_smoke(cls) -> "ScriptedProvider":
        """A complete read/write/verify/final conversation for ``short-01``."""

        from coding_agent.llm.base import LLMResponse, ToolCall

        fixed_source = (
            "def inclusive_total(start: int, end: int) -> int:\n"
            "    return sum(range(start, end + 1))\n"
        )
        return cls(
            [
                LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(id="script-read", name="read_file", arguments={"path": "totals.py"})],
                    stop_reason="tool_calls",
                ),
                LLMResponse(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            id="script-write",
                            name="write_file",
                            arguments={"path": "totals.py", "content": fixed_source},
                        ),
                        ToolCall(
                            id="script-test",
                            name="run_shell",
                            arguments={"command": "python -m unittest discover -s tests -q"},
                        ),
                    ],
                    stop_reason="tool_calls",
                ),
                LLMResponse(text="Fixed inclusive_total and ran the fixture tests.", stop_reason="end_turn"),
            ]
        )


# Exact command matching makes the oracle's command surface auditable.  It
# deliberately accepts no shell syntax, paths outside the workspace, package
# installation, or arbitrary setup command.
SAFE_ORACLE_COMMANDS: dict[str, tuple[str, ...]] = {
    "python -m unittest discover -s tests -q": (sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"),
    "python -m unittest discover -s packages -p 'test_*.py' -q": (
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "packages",
        "-p",
        "test_*.py",
        "-q",
    ),
    "python tests/repeat_scheduler.py --runs 50": (
        sys.executable,
        "tests/repeat_scheduler.py",
        "--runs",
        "50",
    ),
    "node tests/check.js": ("node", "tests/check.js"),
}


def is_safe_oracle_command(command: object) -> bool:
    return isinstance(command, str) and command in SAFE_ORACLE_COMMANDS


def run_safe_oracle_command(
    command: str,
    workspace: str | Path,
    *,
    timeout_seconds: float = 30.0,
) -> tuple[bool, str]:
    """Run one exact allowlisted verification command without a shell."""

    if command not in SAFE_ORACLE_COMMANDS:
        raise RunnerSafetyError(f"oracle command is not allowlisted: {command!r}")
    root = Path(workspace).resolve()
    environment = _verification_environment(root)
    environment.update(
        {
            "NO_PROXY": "",
            "no_proxy": "",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "http_proxy": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    try:
        completed = subprocess.run(
            SAFE_ORACLE_COMMANDS[command],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout_seconds)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"command could not run: {exc}"
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if len(output) > 2_000:
        output = output[-2_000:]
    return completed.returncode == 0, output or f"exit code {completed.returncode}"


def _verification_environment(root: Path) -> dict[str, str]:
    """Keep toolchain essentials while withholding host credentials."""

    allowed = ("PATH", "PATHEXT", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL")
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    home = root / ".eval-run" / "home"
    home.mkdir(parents=True, exist_ok=True)
    environment["HOME"] = str(home)
    return environment


def evaluate_oracle(
    scenario: object,
    workspace: str | Path,
    *,
    baseline_files: Mapping[str, str] | None = None,
    baseline_public_api: Mapping[str, tuple[str, ...]] | None = None,
    baseline_assertions: int | None = None,
    trace_events: Sequence[Mapping[str, Any]] = (),
    run_status: str = "",
    timeout_seconds: float = 30.0,
) -> OracleResult:
    """Evaluate declarative checks only; never execute setup commands."""

    root = Path(workspace).resolve()
    checks = _scenario_checks(scenario)
    results: list[OracleCheckResult] = []
    for check in checks:
        check_type = str(check.get("type", ""))
        try:
            passed, detail = _evaluate_check(
                check_type,
                check,
                root,
                baseline_files=dict(baseline_files or {}),
                baseline_public_api=dict(baseline_public_api or {}),
                baseline_assertions=baseline_assertions,
                trace_events=trace_events,
                run_status=run_status,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # fail closed while preserving useful diagnostics
            passed, detail = False, f"oracle check raised {type(exc).__name__}: {exc}"
        results.append(OracleCheckResult(type=check_type or "unknown", passed=passed, detail=detail))
    return OracleResult(passed=bool(results) and all(item.passed for item in results), checks=tuple(results))


class ContextEvaluationRunner:
    """Isolated full-run executor with injectable, offline-friendly wiring."""

    def __init__(
        self,
        *,
        config: RunnerConfig | None = None,
        materializer: FixtureMaterializer | None = None,
        provider_factory: Callable[..., Any] | None = None,
        harness_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or RunnerConfig()
        self.materializer = materializer or FixtureMaterializer()
        self.provider_factory = provider_factory
        self.harness_factory = harness_factory

    def run_scenario(
        self,
        scenario: object,
        *,
        variant: EvaluationVariant | str | None = None,
        provider_factory: Callable[..., Any] | None = None,
        harness_factory: Callable[..., Any] | None = None,
    ) -> EvaluationRunResult:
        """Run one case from a copied fixture, then evaluate only its oracle."""

        selected = str(variant or self.config.variant)
        variant_config = self.config.variant_config(selected)
        scenario_id = _scenario_value(scenario, "id", "unnamed")
        bucket = _scenario_value(scenario, "bucket", "")
        run_id = f"eval-{uuid.uuid4().hex[:12]}"
        root, cleanup = self._create_run_root(run_id)
        source_root = root / "fixture-source"
        workspace = root / "workspace"
        session_root = root / "session"
        trace_root = root / "trace"
        source_fixture: MaterializedFixture | None = None
        trace_events: list[dict[str, Any]] = []
        result_obj: Any | None = None
        session: Any | None = None
        error: str | None = None
        status = "failed"
        try:
            if _scenario_setup_commands(scenario):
                raise RunnerSafetyError("scenario setup_commands are forbidden; fixtures must be declarative")
            source_fixture = materialize_fixture(scenario, source_root, materializer=self.materializer)
            copy_fixture(source_root, workspace)
            workspace_before = workspace_hash(workspace)
            if workspace_before != source_fixture.fixture_hash:
                raise RunnerSafetyError("copied fixture hash does not match source fixture hash")
            baseline_files = workspace_file_hashes(workspace)
            baseline_api = _public_api_snapshot(workspace)
            baseline_assertions = _assertion_count(workspace)
            provider = _invoke_factory(
                provider_factory or self.provider_factory,
                scenario=scenario,
                variant=selected,
                workspace=workspace,
            ) if (provider_factory or self.provider_factory) else DisabledNetworkProvider()
            provider_name, provider_model = _provider_identity(provider)
            agent_config, trace = _build_agent_config(
                scenario,
                workspace=workspace,
                session_root=session_root,
                trace_root=trace_root,
                variant=variant_config,
                runner_config=self.config,
                provider_name=provider_name,
                provider_model=provider_model,
            )
            harness = _make_harness(
                harness_factory or self.harness_factory,
                config=agent_config,
                provider=provider,
                trace=trace,
                variant=variant_config,
            )
            session = harness.new_session(_scenario_value(scenario, "task", ""))
            if bool(_scenario_mapping(scenario, "pressure").get("resume_boundary")):
                _prepare_resume_boundary(harness, session, scenario, workspace)
                # Oracle baselines describe what the resumed agent received,
                # not the pre-drift checkpoint state.
                baseline_files = workspace_file_hashes(workspace)
                baseline_api = _public_api_snapshot(workspace)
                baseline_assertions = _assertion_count(workspace)
                session = harness.load_session(
                    str(getattr(session, "session_id", getattr(session, "id", "")))
                )
            result_obj = harness.run_task(session)
            status = str(getattr(result_obj, "status", "failed"))
            trace_events = _read_trace_events(trace_root)
            oracle = evaluate_oracle(
                scenario,
                workspace,
                baseline_files=baseline_files,
                baseline_public_api=baseline_api,
                baseline_assertions=baseline_assertions,
                trace_events=trace_events,
                run_status=status,
                timeout_seconds=self.config.oracle_timeout_seconds,
            )
            workspace_after = workspace_hash(workspace)
            source_after = workspace_hash(source_root)
            metrics = _request_metrics_from_trace(
                trace_events,
                run_id=run_id,
                scenario_id=scenario_id,
                bucket=bucket,
                variant=selected,
                session_id=str(getattr(session, "session_id", getattr(session, "id", ""))),
                provider_name=provider_name,
                model=provider_model,
            )
            return self._finish_result(
                run_id=run_id,
                scenario_id=scenario_id,
                bucket=bucket,
                variant=selected,
                status=status,
                source_fixture=source_fixture,
                workspace_before=workspace_before,
                workspace_after=workspace_after,
                source_after=source_after,
                session=session,
                result_obj=result_obj,
                oracle=oracle,
                metrics=metrics,
                trace_events=trace_events,
                error=None,
                root=root,
                session_root=session_root,
                trace_root=trace_root,
                variant_config=variant_config,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            trace_events = _read_trace_events(trace_root)
            baseline_files = workspace_file_hashes(workspace) if workspace.exists() else {}
            oracle = OracleResult(
                passed=False,
                checks=(OracleCheckResult(type="runner", passed=False, detail=error),),
            )
            return self._finish_result(
                run_id=run_id,
                scenario_id=scenario_id,
                bucket=bucket,
                variant=selected,
                status=status,
                source_fixture=source_fixture,
                workspace_before=workspace_hash(workspace) if workspace.exists() else "",
                workspace_after=workspace_hash(workspace) if workspace.exists() else "",
                source_after=workspace_hash(source_root) if source_root.exists() else "",
                session=session,
                result_obj=result_obj,
                oracle=oracle,
                metrics=(),
                trace_events=trace_events,
                error=error,
                root=root,
                session_root=session_root,
                trace_root=trace_root,
                variant_config=variant_config,
            )
        finally:
            if cleanup:
                cleanup.cleanup()

    def run_many(
        self,
        scenarios: Iterable[object],
        *,
        variants: Sequence[EvaluationVariant | str] | None = None,
    ) -> RunResult:
        """Run a deterministic suite sequentially and return the shared schema model."""

        selected_variants = tuple(variants or (self.config.variant,))
        outcomes = [
            self.run_scenario(scenario, variant=variant)
            for scenario in scenarios
            for variant in selected_variants
        ]
        run_id = f"suite-{uuid.uuid4().hex[:12]}"
        return RunResult(
            run_id=run_id,
            mode="full-run",
            requests=tuple(metric for outcome in outcomes for metric in outcome.request_metrics),
            tasks=tuple(outcome.task_record() for outcome in outcomes),
            manifest={"variants": list(selected_variants), "task_count": len(outcomes)},
            metadata={"passed": all(outcome.passed for outcome in outcomes)},
        )

    def _create_run_root(self, run_id: str) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
        parent = Path(self.config.work_root) if self.config.work_root is not None else None
        if self.config.keep_workspaces:
            root = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=parent))
            return root, None
        temporary = tempfile.TemporaryDirectory(prefix=f"{run_id}-", dir=parent)
        return Path(temporary.name), temporary

    def _finish_result(
        self,
        *,
        run_id: str,
        scenario_id: str,
        bucket: str,
        variant: str,
        status: str,
        source_fixture: MaterializedFixture | None,
        workspace_before: str,
        workspace_after: str,
        source_after: str,
        session: Any | None,
        result_obj: Any | None,
        oracle: OracleResult,
        metrics: Sequence[RequestMetric],
        trace_events: Sequence[Mapping[str, Any]],
        error: str | None,
        root: Path,
        session_root: Path,
        trace_root: Path,
        variant_config: VariantConfig,
    ) -> EvaluationRunResult:
        keep = self.config.keep_workspaces
        fixture_hash = source_fixture.fixture_hash if source_fixture is not None else ""
        source_isolated = bool(fixture_hash) and fixture_hash == source_after
        blocked_expected = _scenario_expected_recovery_action_from_metadata(
            source_fixture.requested_parameters if source_fixture is not None else {}
        ) == "BLOCKED"
        terminal_ok = status == "completed" or (blocked_expected and status == "failed")
        return EvaluationRunResult(
            run_id=run_id,
            scenario_id=scenario_id,
            bucket=bucket,
            variant=variant,  # type: ignore[arg-type]
            status=status,
            passed=terminal_ok and oracle.passed and error is None and source_isolated,
            fixture_hash=fixture_hash,
            workspace_hash_before=workspace_before,
            workspace_hash_after=workspace_after,
            source_hash_after=source_after,
            session_id=(
                str(getattr(session, "session_id", getattr(session, "id", ""))) if session is not None else None
            ),
            result_text=str(getattr(result_obj, "text", "")) if result_obj is not None else "",
            steps_used=int(getattr(result_obj, "steps_used", 0)) if result_obj is not None else 0,
            tool_calls_used=int(getattr(result_obj, "tool_calls_used", 0)) if result_obj is not None else 0,
            oracle=oracle,
            request_metrics=tuple(metrics),
            provider_trace=tuple(dict(event) for event in trace_events if _event_type(event).startswith("llm_")),
            error=error,
            workspace_path=str(root / "workspace") if keep else None,
            session_path=str(session_root) if keep else None,
            trace_path=str(trace_root) if keep else None,
            metadata={"variant_config": variant_config.__dict__, "source_isolated": source_isolated},
        )


def _build_agent_config(
    scenario: object,
    *,
    workspace: Path,
    session_root: Path,
    trace_root: Path,
    variant: VariantConfig,
    runner_config: RunnerConfig,
    provider_name: str,
    provider_model: str,
) -> tuple[Any, Any]:
    from config import AgentConfig, Constraints, LLMConfig
    from coding_agent.runtime.trace import TraceWriter

    limits = _scenario_mapping(scenario, "limits")
    max_steps = runner_config.max_steps or _positive_int(limits.get("max_steps"), 30)
    max_tool_calls = runner_config.max_tool_calls or _positive_int(limits.get("max_tool_calls"), 80)
    max_subagents = _positive_int(limits.get("max_subagents"), 0, allow_zero=True)
    window = _positive_int(limits.get("context_window_tokens"), 256_000)
    parameters = _fixture_parameters(scenario)
    retry_count = 1 if isinstance(parameters, Mapping) and parameters.get("provider_fault") else 0
    config = AgentConfig(
        llm=LLMConfig(provider=provider_name, model=provider_model, api_key=""),
        workspace=workspace,
        planner_enabled=False,
        constraints=Constraints(
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            max_subagents=max_subagents,
            max_subagent_depth=1,
            shell_timeout=30.0,
            tool_timeout=30.0,
            max_llm_retries=retry_count,
        ),
        session_dir=session_root,
        trace_dir=trace_root,
        session_dir_explicit=True,
        trace_dir_explicit=True,
        agent_dir=session_root.parent / ".agent",
        structured_context_enabled=variant.structured_context_enabled,
        # raw_full is intentionally untrimmed; the injected provider remains
        # responsible for enforcing its actual model window.
        max_context_tokens=1_000_000_000 if variant.name == "raw_full" else window,
        structured_context_window_limit=window,
    )
    return config, TraceWriter(trace_root)


def _make_harness(
    factory: Callable[..., Any] | None,
    *,
    config: Any,
    provider: Any,
    trace: Any,
    variant: VariantConfig,
) -> Any:
    if factory is not None:
        return _invoke_factory(factory, config=config, provider=provider, trace=trace, variant=variant)
    if variant.name == "observation_full":
        raise RunnerSafetyError(
            "observation_full live execution requires an injected harness_factory; "
            "use offline replay for exact three-way reconstruction"
        )
    from coding_agent import Harness
    from coding_agent.context.session import SessionStore
    from coding_agent.runtime.validator import Validator
    from coding_agent.structured_context.structured_harness import StructuredHarness

    if variant.structured_context_enabled:
        return StructuredHarness(config, provider=provider, trace=trace)
    return Harness(
        config,
        provider=provider,
        trace=trace,
        session_store=SessionStore(config.session_dir),
        planner=None,
        validator=Validator(require_verification_after_edit=True),
    )


def _invoke_factory(factory: Callable[..., Any], **available: Any) -> Any:
    """Support practical zero-arg, keyword, and positional test factories."""

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()
    parameters = list(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return factory(**available)
    kwargs = {
        name: value
        for name, value in available.items()
        if name in signature.parameters
        and signature.parameters[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    required = [
        parameter
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if all(parameter.name in kwargs for parameter in required):
        return factory(**kwargs)
    positional = [available[name] for name in ("scenario", "variant", "workspace", "config", "provider", "trace") if name in available]
    return factory(*positional[: len(required)])


def _scenario_value(scenario: object, key: str, default: str) -> str:
    if isinstance(scenario, Mapping):
        value = scenario.get(key, default)
    else:
        value = getattr(scenario, key, default)
    return str(value) if value is not None else default


def _scenario_mapping(scenario: object, key: str) -> dict[str, Any]:
    value = scenario.get(key, {}) if isinstance(scenario, Mapping) else getattr(scenario, key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _fixture_parameters(scenario: object) -> dict[str, Any]:
    fixture = scenario.get("fixture", {}) if isinstance(scenario, Mapping) else getattr(scenario, "fixture", {})
    if isinstance(fixture, Mapping):
        parameters = fixture.get("parameters", {})
    else:
        parameters = getattr(fixture, "parameters", {})
    return dict(parameters) if isinstance(parameters, Mapping) else {}


def _scenario_setup_commands(scenario: object) -> tuple[str, ...]:
    value = scenario.get("setup_commands", ()) if isinstance(scenario, Mapping) else getattr(scenario, "setup_commands", ())
    return tuple(str(item) for item in value) if isinstance(value, (list, tuple)) else ()


def _provider_identity(provider: object) -> tuple[str, str]:
    provider_config = getattr(provider, "config", None)
    name = getattr(provider, "provider_name", None) or getattr(provider_config, "provider", None)
    model = getattr(provider, "model", None) or getattr(provider_config, "model", None)
    if not name:
        class_name = type(provider).__name__.removesuffix("Provider")
        name = class_name.lower() or "injected"
    return str(name), str(model or "unknown")


def _scenario_expected_recovery_action_from_metadata(parameters: Mapping[str, Any]) -> str:
    return str(parameters.get("expected_action", ""))


def _prepare_resume_boundary(
    harness: object,
    session: object,
    scenario: object,
    workspace: Path,
) -> None:
    """Persist a checkpoint, apply the declared external drift, then resume."""

    session_id = str(getattr(session, "session_id", getattr(session, "id", "")))
    contexts = getattr(harness, "_contexts", {})
    store = getattr(harness, "structured_store", None)
    context = contexts.get(session_id) if isinstance(contexts, Mapping) else None
    if not session_id or context is None or store is None:
        raise RunnerSafetyError("resume scenarios require the built-in StructuredHarness")
    parameters = _fixture_parameters(scenario)
    drift = str(parameters.get("drift", ""))
    paths = [str(item) for item in parameters.get("impacted_paths", ())]
    if drift == "LOW":
        paths = paths or ["unrelated.txt"]
    if drift == "HIGH" and paths:
        from coding_agent.structured_context.models import Finding

        context.task_state.key_findings.extend(
            Finding(
                id=f"eval-impact-{index}",
                fact=f"Declared resume impact path: {path}",
                evidence=[{"path": path, "source": "evaluation-recipe"}],
            )
            for index, path in enumerate(paths, start=1)
        )
    store.save_context(context, checkpoint_kind="evaluation-resume-boundary")
    for relative in paths:
        target = _safe_path(workspace, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        target.write_text(previous + "# external evaluation drift\n", encoding="utf-8")
    if parameters.get("branch_change"):
        try:
            subprocess.run(
                ["git", "checkout", "-q", "-b", "evaluation-structural-drift"],
                cwd=workspace,
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RunnerSafetyError(f"could not apply structural git drift: {exc}") from exc


def _scenario_checks(scenario: object) -> list[dict[str, Any]]:
    if isinstance(scenario, RecipeScenario):
        return [item.to_dict() for item in scenario.oracle]
    oracle = scenario.get("oracle") if isinstance(scenario, Mapping) else getattr(scenario, "oracle", None)
    if isinstance(oracle, Mapping):
        items = oracle.get("checks", ())
    elif isinstance(oracle, (list, tuple)):
        items = oracle
    else:
        expected = scenario.get("expected", {}) if isinstance(scenario, Mapping) else getattr(scenario, "expected", {})
        items = expected.get("checks", ()) if isinstance(expected, Mapping) else ()
    output: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, OracleCheck):
            output.append(item.to_dict())
        elif isinstance(item, Mapping):
            output.append(dict(item))
    return output


def _safe_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RunnerSafetyError("oracle path must be a non-empty workspace-relative path")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise RunnerSafetyError("oracle path escapes workspace") from exc
    return target


def _evaluate_check(
    check_type: str,
    check: Mapping[str, Any],
    root: Path,
    *,
    baseline_files: dict[str, str],
    baseline_public_api: dict[str, tuple[str, ...]],
    baseline_assertions: int | None,
    trace_events: Sequence[Mapping[str, Any]],
    run_status: str,
    timeout_seconds: float,
) -> tuple[bool, str]:
    if check_type == "command_exit_zero":
        command = check.get("command")
        if not isinstance(command, str):
            return False, "command is missing"
        try:
            passed, output = run_safe_oracle_command(command, root, timeout_seconds=timeout_seconds)
        except RunnerSafetyError as exc:
            return False, str(exc)
        return passed, output
    if check_type in {"file_contains", "file_not_contains"}:
        target = _safe_path(root, check.get("path"))
        text = str(check.get("text", ""))
        exists = target.is_file()
        contains = exists and text in target.read_text(encoding="utf-8", errors="replace")
        passed = contains if check_type == "file_contains" else exists and not contains
        return passed, f"{target.relative_to(root)} {'contains' if contains else 'does not contain'} requested text"
    if check_type == "file_exists":
        target = _safe_path(root, check.get("path"))
        return target.exists(), f"{target.relative_to(root)} {'exists' if target.exists() else 'is missing'}"
    if check_type in {"file_unchanged", "response_golden_unchanged"}:
        target = _safe_path(root, check.get("path"))
        relative = target.relative_to(root).as_posix()
        current = workspace_file_hashes(root).get(relative)
        baseline = baseline_files.get(relative)
        return bool(baseline and current == baseline), f"baseline={baseline}, current={current}"
    if check_type == "fixture_marker_preserved":
        marker = str(check.get("marker", ""))
        found = any(marker in path.read_text(encoding="utf-8", errors="replace") for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)
        return found, f"marker {marker!r} {'found' if found else 'not found'}"
    if check_type == "python_expression":
        return _safe_python_expression(str(check.get("expression", "")), root, timeout_seconds)
    if check_type == "status_equals":
        expected = str(check.get("value", ""))
        return run_status == expected, f"status={run_status!r}, expected={expected!r}"
    if check_type == "changed_files_match":
        patterns = tuple(str(item) for item in check.get("patterns", ()))
        current = workspace_file_hashes(root)
        changed = sorted(path for path in set(current) | set(baseline_files) if current.get(path) != baseline_files.get(path))
        def matches(path: str) -> bool:
            return any(_path_matches(path, pattern) for pattern in patterns)
        passed = bool(changed) and all(matches(path) for path in changed) and all(any(_path_matches(path, pattern) for path in changed) for pattern in patterns)
        return passed, f"changed files: {changed}"
    if check_type == "json_schema_version":
        target = _safe_path(root, check.get("path"))
        try:
            value = json.loads(target.read_text(encoding="utf-8")).get("schema_version")
        except (OSError, json.JSONDecodeError):
            value = None
        return value == check.get("value"), f"schema_version={value!r}"
    if check_type == "unknown_fields_preserved":
        target = _safe_path(root, check.get("path"))
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        return any(str(key).startswith("x-") for key in payload), "extension field present" if payload else "invalid JSON"
    if check_type == "public_api_unchanged":
        module = str(check.get("module", ""))
        before = baseline_public_api.get(module, ())
        after = _public_api_snapshot(root).get(module, ())
        return bool(before) and before == after, f"before={before}, after={after}"
    if check_type == "forbidden_test_skip_count":
        maximum = _positive_int(check.get("maximum"), 0, allow_zero=True)
        count = sum(_count_skips(path) for path in (root / "tests").rglob("*.py")) if (root / "tests").exists() else 0
        return count <= maximum, f"skip count={count}, maximum={maximum}"
    if check_type == "decimal_precision_equals":
        target = root / "decimal_precision.py"
        value = str(check.get("value", ""))
        return target.is_file() and value in target.read_text(encoding="utf-8", errors="replace"), f"expected {value!r} in decimal_precision.py"
    if check_type == "symbol_reference_count":
        symbol = str(check.get("symbol", ""))
        count = sum(path.read_text(encoding="utf-8", errors="replace").count(symbol) for path in root.rglob("*.py") if path.is_file() and ".git" not in path.parts)
        return count <= 1, f"symbol reference count={count}"
    if check_type == "event_loss_count":
        target = root / "events.py"
        match = re.search(r"EVENT_LOSS_COUNT\s*=\s*(\d+)", target.read_text(encoding="utf-8", errors="replace")) if target.exists() else None
        value = int(match.group(1)) if match else None
        return value == check.get("value"), f"EVENT_LOSS_COUNT={value!r}"
    if check_type == "forbidden_assertion_deletion":
        now = _assertion_count(root)
        return baseline_assertions is not None and now >= baseline_assertions, f"assertions before={baseline_assertions}, after={now}"
    if check_type == "normalization_form":
        import unicodedata

        expected = str(check.get("value", "NFC"))
        text = "".join(path.read_text(encoding="utf-8", errors="replace") for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)
        return text == unicodedata.normalize(expected, text), f"content is {expected}"
    if check_type == "display_name_roundtrip":
        target = root / "unicode_names.py"
        return target.is_file() and "café" in target.read_text(encoding="utf-8", errors="replace"), "unicode display name preserved"
    if check_type == "unicode_error_message_preserved":
        needle = str(check.get("text", ""))
        corpus = _trace_text(trace_events) + "\n" + "\n".join(
            path.read_text(encoding="utf-8", errors="replace") for path in root.rglob("*") if path.is_file()
        )
        return needle in corpus, f"unicode text {needle!r} {'present' if needle in corpus else 'absent'}"
    if check_type == "validation_passed_after_last_edit":
        return _has_successful_verification_after_edit(trace_events), "checked tool trace ordering"
    if check_type == "tool_call_results_complete":
        calls = {str(_event_data(event).get("call_id")) for event in trace_events if _event_type(event) == "tool_call"}
        results = {str(_event_data(event).get("call_id")) for event in trace_events if _event_type(event) == "tool_result"}
        return calls <= results, f"unpaired calls={sorted(calls - results)}"
    if check_type == "protocol_errors_equal":
        expected = _positive_int(check.get("value"), 0, allow_zero=True)
        actual = sum(1 for event in trace_events if _event_type(event) in {"protocol_error", "error"})
        return actual == expected, f"protocol errors={actual}"
    if check_type == "observed_tool_failure_count":
        expected = _positive_int(check.get("minimum"), 0, allow_zero=True)
        actual = sum(1 for event in trace_events if _event_type(event) == "tool_result" and not _event_data(event).get("success"))
        return actual >= expected, f"tool failures={actual}"
    if check_type == "provider_retry_count":
        expected = _positive_int(check.get("value"), 0, allow_zero=True)
        actual = sum(1 for event in trace_events if _event_type(event) == "llm_retry")
        return actual == expected, f"provider retries={actual}"
    if check_type == "model_request_count":
        expected = _positive_int(check.get("value"), 0, allow_zero=True)
        actual = sum(
            1
            for event in trace_events
            if _event_type(event) in {"llm_request", "llm_request_prepared"}
        )
        return actual == expected, f"model requests={actual}"
    if check_type == "workspace_change_count":
        expected = _positive_int(check.get("value"), 0, allow_zero=True)
        current = workspace_file_hashes(root)
        actual = sum(1 for path in set(current) | set(baseline_files) if current.get(path) != baseline_files.get(path))
        return actual == expected, f"workspace changes={actual}"
    if check_type == "latest_verification_is_passing":
        shells = [event for event in trace_events if _event_type(event) == "tool_result" and _event_data(event).get("name") == "run_shell"]
        passed = bool(shells and _event_data(shells[-1]).get("success"))
        return passed == bool(check.get("value")), f"latest verification passed={passed}"
    if check_type in {"artifact_expected", "stale_finding_revalidated", "impacted_paths_reread", "recovery_action_equals", "fold_fallback_count", "subagent_report_is_structured", "subagent_evidence_preserved", "main_agent_verified", "duplicate_interaction_groups"}:
        return _trace_or_metadata_check(check_type, check, trace_events)
    return False, f"unsupported oracle type {check_type!r}"


def _safe_python_expression(expression: str, root: Path, timeout_seconds: float) -> tuple[bool, str]:
    """Only accept the one published expression, never ``eval`` source data."""

    allowed = "parse_retry_count('0') == 0 and parse_retry_count('-1') == 3"
    if expression != allowed:
        return False, "python_expression is not in the safe expression allowlist"
    script = "from config import parse_retry_count; raise SystemExit(0 if parse_retry_count('0') == 0 and parse_retry_count('-1') == 3 else 1)"
    environment = _verification_environment(root)
    environment.update(
        {
            "NO_PROXY": "",
            "no_proxy": "",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    try:
        result = subprocess.run([sys.executable, "-c", script], cwd=root, env=environment, timeout=max(0.1, timeout_seconds), capture_output=True, text=True, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return result.returncode == 0, "allowlisted parse_retry_count expression"


def _trace_or_metadata_check(check_type: str, check: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    if check_type == "artifact_expected":
        required = _positive_int(check.get("minimum_count"), 1)
        artifacts = {
            str(_event_data(event).get("artifact_id"))
            for event in events
            if _event_type(event) == "artifact_stored"
            and _event_data(event).get("artifact_id")
        }
        return len(artifacts) >= required, f"stored artifacts={len(artifacts)}"
    if check_type == "stale_finding_revalidated":
        required = _positive_int(check.get("minimum_count"), 1)
        actual = sum(1 for event in events if _event_type(event) == "finding_revalidated")
        return actual >= required, f"revalidations={actual}"
    if check_type == "impacted_paths_reread":
        paths = [str(item) for item in check.get("paths", ())]
        reads = {
            str((_event_data(event).get("arguments") or {}).get("path"))
            for event in events
            if _event_type(event) == "tool_call"
            and _event_data(event).get("name") in {"read_file", "grep_search"}
            and isinstance(_event_data(event).get("arguments"), Mapping)
        }
        return all(path in reads for path in paths), f"reread paths={sorted(reads)}"
    if check_type == "recovery_action_equals":
        expected = str(check.get("value", ""))
        actions = [
            str(_event_data(event).get("recovery_action"))
            for event in events
            if _event_type(event) == "session_resume"
        ]
        return expected in actions, f"recovery actions={actions}"
    if check_type == "fold_fallback_count":
        required = _positive_int(check.get("minimum"), 1)
        actual = sum(1 for event in events if _event_type(event) == "fold_fallback")
        return actual >= required, f"fold fallbacks={actual}"
    if check_type == "subagent_report_is_structured":
        reports = [
            _event_data(event).get("structured_report")
            for event in events
            if _event_type(event) == "subagent_end"
        ]
        return any(isinstance(report, Mapping) for report in reports), "structured report checked"
    if check_type == "subagent_evidence_preserved":
        reports = [
            _event_data(event).get("structured_report")
            for event in events
            if _event_type(event) == "subagent_end"
        ]
        present = any(isinstance(report, Mapping) and report.get("evidence") for report in reports)
        return present, "structured subagent evidence checked"
    if check_type == "main_agent_verified":
        return _has_successful_verification_after_edit(events), "main verification trace checked"
    if check_type == "duplicate_interaction_groups":
        expected = _positive_int(check.get("value"), 0, allow_zero=True)
        groups = [
            str(_event_data(event).get("group_id"))
            for event in events
            if _event_type(event) == "interaction_group_closed"
            and _event_data(event).get("group_id")
        ]
        duplicates = len(groups) - len(set(groups))
        return duplicates == expected, f"duplicate groups={duplicates}"
    return False, "unknown trace check"


def _path_matches(path: str, pattern: str) -> bool:
    return path == pattern or Path(path).match(pattern)


def _public_api_snapshot(root: Path) -> dict[str, tuple[str, ...]]:
    snapshot: dict[str, tuple[str, ...]] = {}
    for path in root.glob("*.py"):
        functions = tuple(re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", path.read_text(encoding="utf-8", errors="replace"), flags=re.MULTILINE))
        snapshot[path.stem] = functions
    return snapshot


def _assertion_count(root: Path) -> int:
    tests = root / "tests"
    if not tests.exists():
        return 0
    return sum(path.read_text(encoding="utf-8", errors="replace").count("assert") for path in tests.rglob("*.py"))


def _count_skips(path: Path) -> int:
    return len(re.findall(r"(?:@unittest\.skip|pytest\.skip|\.skipTest\()", path.read_text(encoding="utf-8", errors="replace")))


def _has_successful_verification_after_edit(events: Sequence[Mapping[str, Any]]) -> bool:
    last_edit = -1
    last_verification = -1
    verification_ok = False
    for index, event in enumerate(events):
        if _event_type(event) != "tool_result":
            continue
        data = _event_data(event)
        if data.get("name") in {"write_file", "edit_file"} and data.get("success"):
            last_edit = index
        if data.get("name") == "run_shell":
            last_verification = index
            verification_ok = bool(data.get("success"))
    return last_edit >= 0 and last_verification > last_edit and verification_ok


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("type", ""))


def _event_data(event: Mapping[str, Any]) -> dict[str, Any]:
    value = event.get("data", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _trace_text(events: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(json.dumps(dict(event), ensure_ascii=False, default=str) for event in events)


def _read_trace_events(root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not root.exists():
        return events
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                events.append(dict(value))
    return events


def _request_metrics_from_trace(
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    scenario_id: str,
    bucket: str,
    variant: str,
    session_id: str,
    provider_name: str,
    model: str,
) -> tuple[RequestMetric, ...]:
    prepared: dict[str, Mapping[str, Any]] = {}
    output: list[RequestMetric] = []
    for event in events:
        event_type = _event_type(event)
        data = _event_data(event)
        request_id = str(data.get("request_id", ""))
        if event_type == "llm_request_prepared" and request_id:
            prepared[request_id] = data
        if event_type != "llm_request_finished" or not request_id:
            continue
        normalized = data.get("normalized_usage", {})
        normalized = dict(normalized) if isinstance(normalized, Mapping) else {}
        snapshot = prepared.get(request_id, {})
        layers = snapshot.get("layers_estimated", {}) if isinstance(snapshot, Mapping) else {}
        layers = dict(layers) if isinstance(layers, Mapping) else {}
        logical = _positive_int(normalized.get("logical_input_tokens"), 0, allow_zero=True)
        estimated = _positive_int(layers.get("total"), 0, allow_zero=True)
        measured = logical or estimated
        payload_hash = str(snapshot.get("payload_hash", ""))
        role = str(snapshot.get("agent_role", data.get("agent_role", "main")))
        output.append(
            RequestMetric(
                task_case_id=scenario_id,
                request_id=request_id,
                bucket=bucket,
                run_id=run_id,
                variant=variant,
                session_id=session_id,
                parent_request_id=(
                    str(snapshot.get("parent_request_id"))
                    if snapshot.get("parent_request_id") is not None
                    else None
                ),
                agent_role=role,
                step=_positive_int(snapshot.get("step", data.get("step")), 0, allow_zero=True),
                attempt=_positive_int(snapshot.get("attempt", data.get("attempt")), 1),
                epoch_id=_positive_int(snapshot.get("epoch_id"), 0, allow_zero=True),
                provider=provider_name,
                model=model,
                status=str(data.get("status", "success")),
                logical_input_tokens=logical,
                fresh_processed_input_tokens=_positive_int(normalized.get("fresh_processed_input_tokens"), 0, allow_zero=True),
                cache_hit_tokens=_positive_int(normalized.get("cache_hit_tokens"), 0, allow_zero=True),
                raw_full_tokens=measured if variant == "raw_full" and role == "main" else 0,
                observation_full_tokens=(
                    measured if variant == "observation_full" and role == "main" else 0
                ),
                structured_tokens=measured if variant == "structured" and role == "main" else 0,
                token_source=str(normalized.get("token_source", "missing")),
                payload_hashes={variant: payload_hash} if payload_hash else {},
                layers={str(key): _positive_int(value, 0, allow_zero=True) for key, value in layers.items()},
                metadata={
                    "raw_usage": data.get("raw_usage", {}),
                    "request_group_id": snapshot.get("request_group_id"),
                    "payload_hash": payload_hash,
                },
            )
        )
    return tuple(output)


def _positive_int(value: object, default: int, *, allow_zero: bool = False) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number < 0 or (number == 0 and not allow_zero):
        return default
    return number


__all__ = [
    "ContextEvaluationRunner",
    "DisabledNetworkProvider",
    "EvaluationRunResult",
    "EvaluationVariant",
    "OracleCheckResult",
    "OracleResult",
    "RunnerConfig",
    "RunnerSafetyError",
    "SAFE_ORACLE_COMMANDS",
    "SUPPORTED_VARIANTS",
    "ScriptedProvider",
    "VariantConfig",
    "VARIANT_CONFIGS",
    "evaluate_oracle",
    "is_safe_oracle_command",
    "run_safe_oracle_command",
]
