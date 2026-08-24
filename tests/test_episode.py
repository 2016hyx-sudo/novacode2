from __future__ import annotations

from pathlib import Path

from coding_agent.llm.base import LLMResponse, ToolCall
from coding_agent.runtime.episode import (
    CriterionEvidence,
    EpisodeController,
    TaskEpisode,
    TurnDisposition,
)
from coding_agent.structured_context.episode_store import EpisodeStore
from coding_agent.tools.task_outcome import ReportTaskOutcomeTool
from config import AgentConfig, LLMConfig


class _OutcomeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None, *, reasoning_effort=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="outcome-1",
                        name="report_task_outcome",
                        arguments={
                            "disposition": "completion_proposed",
                            "summary": "read-only task complete",
                            "open_questions": [],
                            "criteria_evidence": [],
                        },
                    )
                ],
                stop_reason="tool_calls",
            )
        return LLMResponse(text="Done", stop_reason="end_turn")


def _episode() -> TaskEpisode:
    return TaskEpisode.new(
        session_id="session-1",
        objective="implement and verify",
        start_event_seq=10,
        success_criteria=["tests pass"],
    )


def test_report_task_outcome_validation_and_callback() -> None:
    reports = []
    tool = ReportTaskOutcomeTool(reports.append)
    assert not tool.execute("done", "x", [], []).success
    result = tool.execute(
        "completion_proposed",
        "implemented",
        [],
        [{"criterion_id": "criterion-1", "evidence_refs": ["artifact-1"]}],
    )
    assert result.success
    assert reports[0].disposition == "completion_proposed"


def test_completion_gate_requires_current_evidence_and_post_edit_verification() -> None:
    controller = EpisodeController(_episode())
    controller.submit(
        TurnDisposition(
            "completion_proposed",
            "done",
            criteria_evidence=[CriterionEvidence("criterion-1", ["artifact-1"])],
        )
    )
    evidence = [
        {"seq": 11, "type": "file_change", "episode_id": controller.episode.episode_id},
        {
            "seq": 12,
            "type": "tool_result",
            "name": "run_shell",
            "success": True,
            "artifact_id": "artifact-1",
            "episode_id": controller.episode.episode_id,
        },
    ]
    verdict = controller.apply_turn(evidence=evidence)
    assert verdict.passed
    assert controller.episode.status == "succeeded"

    other = EpisodeController(_episode())
    other.submit(
        TurnDisposition(
            "completion_proposed",
            "done",
            criteria_evidence=[CriterionEvidence("criterion-1", ["other-artifact"])],
        )
    )
    verdict = other.apply_turn(
        evidence=[
            {
                "seq": 12,
                "type": "tool_result",
                "artifact_id": "other-artifact",
                "episode_id": "a-different-episode",
            }
        ]
    )
    assert not verdict.passed
    assert other.episode.status == "active"


def test_waiting_reopen_and_store(tmp_path: Path) -> None:
    controller = EpisodeController(_episode())
    controller.submit(TurnDisposition("waiting_user", "need input", ["choose version"]))
    assert not controller.apply_turn(evidence=[]).passed
    assert controller.episode.status == "waiting_user"
    assert controller.begin_turn("use version two") == "continue"
    controller.episode.status = "succeeded"
    version = controller.episode.outcome_version
    assert controller.begin_turn("still fails, this is wrong") == "reopened"
    assert controller.episode.outcome_version == version + 1

    store = EpisodeStore(tmp_path / "session")
    store.append("episode_reopened", controller.episode)
    loaded = store.latest()
    assert loaded is not None
    assert loaded.episode_id == controller.episode.episode_id
    assert loaded.outcome_version == version + 1


def test_structured_harness_persists_episode_success_only_after_report(tmp_path: Path) -> None:
    from coding_agent.structured_context.structured_harness import StructuredHarness

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = AgentConfig(
        llm=LLMConfig(api_key="test"),
        workspace=workspace,
        agent_dir=workspace / ".agent",
        session_dir=workspace / ".agent" / "sessions",
        trace_dir=workspace / ".agent" / "traces",
        session_dir_explicit=True,
        trace_dir_explicit=True,
        structured_context_enabled=True,
        long_term_memory_enabled=False,
    )
    harness = StructuredHarness(config, provider=_OutcomeProvider())
    session = harness.new_session("explain the repository")
    result = harness.run_task(session, "explain the repository")
    assert result.status == "completed"
    episode = harness._episode_store(session.session_id).latest()
    assert episode is not None
    assert episode.status == "succeeded"
    events = harness._contexts[session.session_id].event_log.read_since(0)
    assert any(event["type"] == "episode_succeeded" for event in events)
