from __future__ import annotations

from pathlib import Path

from evals.structured_context.fixtures import (
    copy_fixture,
    materialize_fixture,
    workspace_hash,
)
from evals.structured_context.runner import (
    ContextEvaluationRunner,
    RunnerConfig,
    ScriptedProvider,
    evaluate_oracle,
)
from evals.structured_context.scenarios import (
    SUPPORTED_FIXTURE_TEMPLATES,
    get_scenario,
    load_scenario_suite,
)


def test_core_30_scenarios_load_with_manifest_contract() -> None:
    suite = load_scenario_suite()

    assert suite.suite == "core-30"
    assert len(suite.scenarios) == 30
    assert {scenario.fixture.template for scenario in suite.scenarios} == SUPPORTED_FIXTURE_TEMPLATES
    assert sum(scenario.bucket == "tool-output-heavy" for scenario in suite.scenarios) == 6
    assert all(scenario.oracle for scenario in suite.scenarios)


def test_fixture_hash_is_repeatable_and_every_template_materializes(tmp_path: Path) -> None:
    suite = load_scenario_suite()
    by_template = {}
    for scenario in suite.scenarios:
        by_template.setdefault(scenario.fixture.template, scenario)

    first = materialize_fixture(by_template["python-micro-package"], tmp_path / "first")
    second = materialize_fixture(by_template["python-micro-package"], tmp_path / "second")
    assert first.fixture_hash == second.fixture_hash
    assert first.file_hashes == second.file_hashes
    assert first.pressure == second.pressure

    for template, scenario in by_template.items():
        fixture = materialize_fixture(scenario, tmp_path / f"template-{template}")
        assert fixture.fixture_hash == workspace_hash(fixture.root)
        assert fixture.metadata_path.is_file()


def test_copied_workspace_isolated_from_source_fixture(tmp_path: Path) -> None:
    source = materialize_fixture(get_scenario("short-01"), tmp_path / "source")
    workspace = copy_fixture(source.root, tmp_path / "workspace")
    before_source = workspace_hash(source.root)

    (workspace / "totals.py").write_text("changed\n", encoding="utf-8")

    assert workspace_hash(source.root) == before_source == source.fixture_hash
    assert workspace_hash(workspace) != source.fixture_hash


def test_scripted_provider_runs_complete_dialogue_and_collects_usage(tmp_path: Path) -> None:
    runner = ContextEvaluationRunner(
        config=RunnerConfig(work_root=tmp_path, keep_workspaces=True),
        provider_factory=lambda: ScriptedProvider.inclusive_total_smoke(),
    )

    result = runner.run_scenario(get_scenario("short-01"), variant="structured")

    assert result.status == "completed"
    assert result.passed
    assert result.oracle.passed
    assert result.steps_used == 3
    assert result.tool_calls_used == 3
    assert len(result.request_metrics) == 3
    assert all(metric.logical_input_tokens > 0 for metric in result.request_metrics)
    assert all(metric.structured_tokens == metric.logical_input_tokens for metric in result.request_metrics)
    assert all(metric.provider == "scripted" and metric.model == "deterministic-eval" for metric in result.request_metrics)
    assert all(metric.payload_hashes.get("structured") for metric in result.request_metrics)
    assert result.provider_trace
    assert result.workspace_path is not None
    assert (Path(result.workspace_path) / "totals.py").read_text(encoding="utf-8").find("end + 1") >= 0
    assert result.to_run_result().requests == result.request_metrics


def test_builtin_runner_rejects_inexact_observation_full_variant(tmp_path: Path) -> None:
    runner = ContextEvaluationRunner(
        config=RunnerConfig(work_root=tmp_path),
        provider_factory=ScriptedProvider.inclusive_total_smoke,
    )

    result = runner.run_scenario(get_scenario("short-01"), variant="observation_full")

    assert not result.passed
    assert "requires an injected harness_factory" in (result.error or "")


def test_structural_resume_scenario_blocks_before_model_request(tmp_path: Path) -> None:
    runner = ContextEvaluationRunner(
        config=RunnerConfig(work_root=tmp_path),
        provider_factory=ScriptedProvider,
    )

    result = runner.run_scenario(get_scenario("resume-03"), variant="structured")

    assert result.status == "failed"
    assert result.passed
    assert result.oracle.passed
    assert not result.request_metrics


def test_source_fixture_mutation_fails_isolation_guard(tmp_path: Path) -> None:
    def provider_factory(*, workspace: Path, **_kwargs):
        source_readme = workspace.parent / "fixture-source" / "README.md"
        source_readme.write_text("tampered\n", encoding="utf-8")
        return ScriptedProvider.inclusive_total_smoke()

    runner = ContextEvaluationRunner(
        config=RunnerConfig(work_root=tmp_path),
        provider_factory=provider_factory,
    )

    result = runner.run_scenario(get_scenario("short-01"), variant="structured")

    assert result.oracle.passed
    assert not result.passed
    assert result.metadata["source_isolated"] is False


def test_failed_oracle_and_unallowlisted_command_are_reported_without_execution(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "safe.txt").write_text("expected", encoding="utf-8")
    scenario = {
        "oracle": {
            "checks": [
                {"type": "file_contains", "path": "safe.txt", "text": "missing"},
                {"type": "command_exit_zero", "command": "touch should-not-exist"},
            ]
        }
    }

    oracle = evaluate_oracle(scenario, workspace)

    assert not oracle.passed
    assert not oracle.checks[0].passed
    assert not oracle.checks[1].passed
    assert not (workspace / "should-not-exist").exists()
