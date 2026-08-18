"""Scenario loading and validation for the structured-context evaluation suite.

The checked-in JSON is intentionally data-only.  This module is the narrow,
deterministic boundary between that data and code which materializes or runs a
case.  Keeping validation here prevents a malformed recipe from accidentally
turning into a shell/setup instruction in the runner.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DATA_DIRECTORY = Path(__file__).with_name("data")
DEFAULT_SCENARIOS_PATH = DATA_DIRECTORY / "scenarios.json"
DEFAULT_MANIFEST_PATH = DATA_DIRECTORY / "manifest.json"


class ScenarioValidationError(ValueError):
    """Raised when a checked-in scenario recipe is incomplete or unsafe."""


SUPPORTED_FIXTURE_TEMPLATES = frozenset(
    {
        "python-micro-package",
        "python-log-pipeline",
        "python-multimodule-service",
        "python-generated-tree",
        "javascript-browser-app",
        "mixed-config-repo",
        "git-recovery-repo",
        "python-monorepo",
    }
)

# These are data-level check names, not executable commands.  The runner maps
# them to bounded local checks and rejects anything outside this set.
SUPPORTED_ORACLE_CHECK_TYPES = frozenset(
    {
        "artifact_expected",
        "changed_files_match",
        "command_exit_zero",
        "decimal_precision_equals",
        "display_name_roundtrip",
        "event_loss_count",
        "file_contains",
        "file_exists",
        "file_not_contains",
        "file_unchanged",
        "fixture_marker_preserved",
        "fold_fallback_count",
        "forbidden_assertion_deletion",
        "forbidden_test_skip_count",
        "impacted_paths_reread",
        "json_schema_version",
        "latest_verification_is_passing",
        "main_agent_verified",
        "model_request_count",
        "normalization_form",
        "observed_tool_failure_count",
        "protocol_errors_equal",
        "provider_retry_count",
        "public_api_unchanged",
        "python_expression",
        "recovery_action_equals",
        "response_golden_unchanged",
        "stale_finding_revalidated",
        "status_equals",
        "subagent_evidence_preserved",
        "subagent_report_is_structured",
        "symbol_reference_count",
        "tool_call_results_complete",
        "unicode_error_message_preserved",
        "unknown_fields_preserved",
        "validation_passed_after_last_edit",
        "workspace_change_count",
        "duplicate_interaction_groups",
    }
)


def _plain_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScenarioValidationError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class FixtureRecipe:
    """A deterministic fixture recipe, separate from its scenario prose."""

    template: str
    seed: int
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FixtureRecipe":
        template = value.get("template")
        seed = value.get("seed")
        parameters = value.get("parameters", {})
        if not isinstance(template, str) or not template:
            raise ScenarioValidationError("fixture.template must be a non-empty string")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ScenarioValidationError("fixture.seed must be an integer")
        return cls(
            template=template,
            seed=seed,
            parameters=_plain_mapping(parameters, label="fixture.parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class OracleCheck:
    """One declarative correctness check from a scenario recipe."""

    type: str
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OracleCheck":
        raw = _plain_mapping(value, label="oracle check")
        check_type = raw.pop("type", None)
        if not isinstance(check_type, str) or not check_type:
            raise ScenarioValidationError("oracle check type must be a non-empty string")
        return cls(type=check_type, values=raw)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.values}


@dataclass(frozen=True)
class Scenario:
    """One full-run evaluation task loaded from ``scenarios.json``."""

    id: str
    bucket: str
    title: str
    task: str
    fixture: FixtureRecipe
    pressure: dict[str, Any]
    limits: dict[str, Any]
    oracle: tuple[OracleCheck, ...]
    tags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Scenario":
        raw = _plain_mapping(value, label="scenario")
        fixture = FixtureRecipe.from_mapping(_plain_mapping(raw.get("fixture"), label="fixture"))
        oracle_raw = raw.get("oracle")
        oracle_obj = _plain_mapping(oracle_raw, label="oracle")
        checks = oracle_obj.get("checks")
        if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
            raise ScenarioValidationError("oracle.checks must be a list")
        parsed_checks = tuple(
            OracleCheck.from_mapping(_plain_mapping(item, label="oracle check")) for item in checks
        )
        tags = raw.get("tags", [])
        if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
            raise ScenarioValidationError("scenario.tags must be a list")
        return cls(
            id=_required_text(raw, "id"),
            bucket=_required_text(raw, "bucket"),
            title=_required_text(raw, "title"),
            task=_required_text(raw, "task"),
            fixture=fixture,
            pressure=_plain_mapping(raw.get("pressure"), label="pressure"),
            limits=_plain_mapping(raw.get("limits"), label="limits"),
            oracle=parsed_checks,
            tags=tuple(str(tag) for tag in tags),
        )

    @property
    def fixture_template(self) -> str:
        return self.fixture.template

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bucket": self.bucket,
            "title": self.title,
            "task": self.task,
            "fixture": self.fixture.to_dict(),
            "pressure": dict(self.pressure),
            "limits": dict(self.limits),
            "oracle": {"checks": [check.to_dict() for check in self.oracle]},
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ScenarioSuite:
    """Validated core suite metadata plus its scenario definitions."""

    schema_version: str
    suite: str
    scenarios: tuple[Scenario, ...]
    manifest: dict[str, Any]

    def by_id(self, scenario_id: str) -> Scenario:
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        raise KeyError(f"unknown scenario id: {scenario_id}")

    def filter(
        self,
        *,
        buckets: Iterable[str] | None = None,
        ids: Iterable[str] | None = None,
    ) -> tuple[Scenario, ...]:
        bucket_set = set(buckets or ())
        id_set = set(ids or ())
        return tuple(
            scenario
            for scenario in self.scenarios
            if (not bucket_set or scenario.bucket in bucket_set)
            and (not id_set or scenario.id in id_set)
        )


def _required_text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ScenarioValidationError(f"scenario.{key} must be a non-empty string")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScenarioValidationError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ScenarioValidationError(f"{label} contains invalid JSON: {exc}") from exc
    return _plain_mapping(parsed, label=label)


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load the immutable dataset manifest without materializing a fixture."""

    return _load_json(Path(path) if path is not None else DEFAULT_MANIFEST_PATH, label="manifest")


def validate_scenarios(
    scenarios: Sequence[Scenario],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed on malformed core-suite recipes.

    Validation checks the published 30-task/bucket contract, every fixture's
    required recipe fields, declared pressure/budgets, and every declarative
    correctness oracle.  It intentionally does not execute a command.
    """

    manifest_data = dict(manifest or load_manifest())
    expected_count = manifest_data.get("scenario_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ScenarioValidationError("manifest.scenario_count must be an integer")
    if len(scenarios) != expected_count:
        raise ScenarioValidationError(
            f"expected {expected_count} scenarios, found {len(scenarios)}"
        )

    ids = [scenario.id for scenario in scenarios]
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ScenarioValidationError(f"duplicate scenario ids: {', '.join(duplicate_ids)}")

    expected_bucket_counts = _plain_mapping(
        manifest_data.get("bucket_counts"), label="manifest.bucket_counts"
    )
    actual_bucket_counts = Counter(scenario.bucket for scenario in scenarios)
    normalized_expected = {
        str(bucket): int(count) for bucket, count in expected_bucket_counts.items()
    }
    if dict(sorted(actual_bucket_counts.items())) != dict(sorted(normalized_expected.items())):
        raise ScenarioValidationError(
            "bucket counts do not match manifest: "
            f"expected {dict(sorted(normalized_expected.items()))}, "
            f"got {dict(sorted(actual_bucket_counts.items()))}"
        )

    manifest_templates = _plain_mapping(
        manifest_data.get("fixture_templates"), label="manifest.fixture_templates"
    )
    allowed_templates = set(manifest_templates) & set(SUPPORTED_FIXTURE_TEMPLATES)
    if allowed_templates != set(SUPPORTED_FIXTURE_TEMPLATES):
        missing = sorted(SUPPORTED_FIXTURE_TEMPLATES - allowed_templates)
        raise ScenarioValidationError(
            f"manifest does not declare all supported fixture templates: {missing}"
        )

    for scenario in scenarios:
        if scenario.bucket not in normalized_expected:
            raise ScenarioValidationError(f"{scenario.id}: unknown bucket {scenario.bucket!r}")
        if scenario.fixture.template not in allowed_templates:
            raise ScenarioValidationError(
                f"{scenario.id}: unsupported fixture template {scenario.fixture.template!r}"
            )
        if scenario.fixture.seed < 0:
            raise ScenarioValidationError(f"{scenario.id}: fixture.seed must be non-negative")

        template_spec = _plain_mapping(
            manifest_templates[scenario.fixture.template],
            label=f"manifest fixture template {scenario.fixture.template}",
        )
        required_parameters = template_spec.get("parameters", [])
        if not isinstance(required_parameters, Sequence) or isinstance(required_parameters, (str, bytes)):
            raise ScenarioValidationError(
                f"manifest fixture template {scenario.fixture.template}.parameters must be a list"
            )
        missing_parameters = [
            str(parameter)
            for parameter in required_parameters
            if str(parameter) not in scenario.fixture.parameters
        ]
        if missing_parameters:
            raise ScenarioValidationError(
                f"{scenario.id}: fixture is missing parameters {missing_parameters}"
            )

        expected_steps = scenario.pressure.get("expected_steps")
        if (
            not isinstance(expected_steps, Sequence)
            or isinstance(expected_steps, (str, bytes))
            or len(expected_steps) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in expected_steps)
            or expected_steps[0] < 0
            or expected_steps[0] > expected_steps[1]
        ):
            raise ScenarioValidationError(
                f"{scenario.id}: pressure.expected_steps must be an ascending two-integer range"
            )
        raw_output = scenario.pressure.get("raw_tool_output_chars")
        if (
            not isinstance(raw_output, Sequence)
            or isinstance(raw_output, (str, bytes))
            or len(raw_output) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in raw_output)
        ):
            raise ScenarioValidationError(
                f"{scenario.id}: pressure.raw_tool_output_chars must be a non-negative range"
            )

        for key in ("max_steps", "max_tool_calls", "context_window_tokens"):
            value = scenario.limits.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ScenarioValidationError(f"{scenario.id}: limits.{key} must be a positive integer")

        if not scenario.oracle:
            raise ScenarioValidationError(f"{scenario.id}: oracle.checks must not be empty")
        for check in scenario.oracle:
            if check.type not in SUPPORTED_ORACLE_CHECK_TYPES:
                raise ScenarioValidationError(
                    f"{scenario.id}: unsupported oracle type {check.type!r}"
                )
            if check.type == "command_exit_zero" and not isinstance(check.values.get("command"), str):
                raise ScenarioValidationError(
                    f"{scenario.id}: command_exit_zero requires a command string"
                )


def load_scenario_suite(
    path: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
    validate: bool = True,
) -> ScenarioSuite:
    """Load the core-30 suite and, by default, validate its published contract."""

    source = Path(path) if path is not None else DEFAULT_SCENARIOS_PATH
    raw = _load_json(source, label="scenarios")
    raw_scenarios = raw.get("scenarios")
    if not isinstance(raw_scenarios, Sequence) or isinstance(raw_scenarios, (str, bytes)):
        raise ScenarioValidationError("scenarios.scenarios must be a list")
    scenarios = tuple(
        Scenario.from_mapping(_plain_mapping(item, label="scenario")) for item in raw_scenarios
    )
    schema_version = raw.get("schema_version")
    suite = raw.get("suite")
    if not isinstance(schema_version, str) or not schema_version:
        raise ScenarioValidationError("scenarios.schema_version must be a non-empty string")
    if not isinstance(suite, str) or not suite:
        raise ScenarioValidationError("scenarios.suite must be a non-empty string")
    manifest = load_manifest(manifest_path)
    if validate:
        validate_scenarios(scenarios, manifest=manifest)
    return ScenarioSuite(
        schema_version=schema_version,
        suite=suite,
        scenarios=scenarios,
        manifest=manifest,
    )


def load_scenarios(
    path: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
    validate: bool = True,
) -> tuple[Scenario, ...]:
    """Convenience entry point returning the validated 30 scenario objects."""

    return load_scenario_suite(path, manifest_path=manifest_path, validate=validate).scenarios


def get_scenario(scenario_id: str, *, path: str | Path | None = None) -> Scenario:
    """Return one validated scenario by its stable dataset identifier."""

    return load_scenario_suite(path).by_id(scenario_id)


__all__ = [
    "DATA_DIRECTORY",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_SCENARIOS_PATH",
    "FixtureRecipe",
    "OracleCheck",
    "Scenario",
    "ScenarioSuite",
    "ScenarioValidationError",
    "SUPPORTED_FIXTURE_TEMPLATES",
    "SUPPORTED_ORACLE_CHECK_TYPES",
    "get_scenario",
    "load_manifest",
    "load_scenario_suite",
    "load_scenarios",
    "validate_scenarios",
]
