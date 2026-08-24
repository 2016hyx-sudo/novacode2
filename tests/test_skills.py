from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from coding_agent.runtime.episode import TaskEpisode
from coding_agent.skills.bank import SkillBank, SkillBankError
from coding_agent.skills.candidates import (
    PendingCandidateStore,
    SkillCandidate,
    SkillRule,
)
from coding_agent.skills.hook import SkillLifecycleHook
from coding_agent.skills.maintainer import SkillMaintainer
from coding_agent.skills.tool import InvokeSkillTool
from coding_agent.skills.verifier import SurrogateVerifier
from coding_agent.structured_context.artifact_store import ArtifactStore


class _ForkOutcome:
    text = "fallback"
    status = "completed"
    steps_used = 1
    tool_calls_used = 0
    structured_report: ClassVar[dict] = {"summary": "isolated", "findings": []}


def _skill(root: Path, name: str, *, body: str = "# Goal\nDo ${ARGUMENTS} in ${SKILL_DIR}", context: str = "inline") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"title: {name}\n"
        "granularity: event-driven\n"
        "version: 0.1.0\n"
        f"context: {context}\n"
        "when_to_apply: during tests\n"
        "user_invocable: true\n"
        "allowed_tools:\n"
        "  - read_file\n"
        "tags: [\"test\"]\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return directory


def test_skill_bank_project_precedence_and_manifest_validation(tmp_path: Path) -> None:
    user = tmp_path / "user"
    project = tmp_path / "project"
    _skill(user, "test_skill", body="user")
    _skill(project, "test_skill", body="project")
    bank = SkillBank(project_dir=project, user_dir=user)
    entry = bank.get("test_skill")
    assert entry is not None
    assert entry.source == "project"
    assert entry.body == "project"
    assert entry.manifest.allowed_tools == ("read_file",)

    bad = project / "bad_name"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: other_name\n---\nbody", encoding="utf-8")
    with pytest.raises(SkillBankError, match="invalid skill bank"):
        bank.scan()


def test_invoke_skill_inline_interpolation_and_artifact_spill(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    directory = _skill(root, "test_skill")
    artifacts = ArtifactStore(tmp_path / "session" / "artifacts")
    bank = SkillBank(project_dir=root)
    small = InvokeSkillTool(bank, artifact_store=lambda: artifacts, inline_token_limit=2000)
    result = small.execute("test_skill", {"value": 3})
    assert result.success
    assert str(directory.resolve()) in result.output
    assert '"value": 3' in result.output

    (directory / "SKILL.md").write_text(
        (directory / "SKILL.md").read_text(encoding="utf-8") + ("x" * 10_000),
        encoding="utf-8",
    )
    large = InvokeSkillTool(bank, artifact_store=lambda: artifacts, inline_token_limit=100)
    result = large.execute("test_skill")
    assert result.success
    assert result.metadata["compressed"] is True
    assert artifacts.find(result.metadata["artifact_id"]) is not None


def test_static_tool_schema_does_not_depend_on_bank_contents(tmp_path: Path) -> None:
    bank = SkillBank(project_dir=tmp_path / "skills")
    tool = InvokeSkillTool(bank)
    before = tool.parameters.copy()
    _skill(bank.project_dir, "late_skill")
    assert tool.parameters == before
    assert tool.execute("late_skill").success


def test_fork_skill_returns_only_structured_report(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _skill(root, "fork_skill", context="fork")
    captured = []

    def run(entry, rendered):
        captured.append((entry.name, rendered))
        return _ForkOutcome()

    result = InvokeSkillTool(SkillBank(project_dir=root), fork_runner=run).execute("fork_skill")
    assert result.success
    assert '"summary": "isolated"' in result.output
    assert captured[0][0] == "fork_skill"


def _candidate(*, succeeded: bool = True) -> SkillCandidate:
    return SkillCandidate.new(
        episode_id="ep-1",
        outcome_version=1,
        name="verified_flow",
        title="Verified flow",
        granularity="task-level",
        when_to_apply="when verification is needed",
        constraints_and_style=["verify"],
        workflow_rules=[SkillRule("Run the matching check", ["artifact-1"])],
        evidence_refs=["artifact-1"],
        source="episode_succeeded",
        episode_succeeded=succeeded,
    )


def test_maintainer_promotion_version_merge_and_stale_gate(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    store = PendingCandidateStore(root)
    bank = SkillBank(project_dir=root)
    maintainer = SkillMaintainer(bank, store)
    candidate = _candidate()
    decision = maintainer.promote(candidate, current_outcome_version=1)
    assert decision.action == "add"
    assert bank.get("verified_flow").manifest.version == "0.1.0"  # type: ignore[union-attr]

    second = _candidate()
    second.workflow_rules.append(SkillRule("Check an edge case", ["artifact-2"]))
    decision = maintainer.promote(second, current_outcome_version=1)
    assert decision.action == "merge"
    assert bank.get("verified_flow").manifest.version == "0.1.1"  # type: ignore[union-attr]
    assert (store.history_dir / "verified_flow" / "0.1.0.SKILL.md").is_file()

    stale = _candidate()
    stale.outcome_version = 1
    assert maintainer.promote(stale, current_outcome_version=2).action == "drop"


def test_surrogate_script_failure_blocks_promotion(tmp_path: Path) -> None:
    package = tmp_path / "package"
    scripts = package / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "bad.py").write_text("raise SystemExit(2)\n", encoding="utf-8")
    result = SurrogateVerifier(timeout=2).verify(_candidate(), package)
    assert result.passed is False
    assert result.diagnostics


def test_episode_success_promotes_previously_pending_event_candidate(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    store = PendingCandidateStore(root)
    bank = SkillBank(project_dir=root)
    candidate = SkillCandidate.new(
        episode_id="ep-1",
        outcome_version=1,
        name="event_recovery",
        title="Event recovery",
        granularity="event-driven",
        when_to_apply="when an event fails",
        constraints_and_style=["verify"],
        workflow_rules=[SkillRule("Repeat the check", ["event-1"])],
        evidence_refs=["event-1"],
        source="verified_event",
    )
    store.save(candidate)
    episode = TaskEpisode.new(session_id="s", objective="o")
    episode.episode_id = "ep-1"
    episode.status = "succeeded"
    episode.last_summary = "done"
    hook = SkillLifecycleHook(store, maintainer=SkillMaintainer(bank, store))
    hook.on_episode_succeeded(episode)
    assert bank.get("event_recovery") is not None
    assert store.get(candidate.candidate_id).status == "promoted"  # type: ignore[union-attr]
