from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from coding_agent.llm.base import LLMResponse, Message, ToolCall
from coding_agent.tools.base import ToolResult
from config import AgentConfig, LLMConfig

from coding_agent.structured_context.artifact_store import ArtifactStore
from coding_agent.structured_context.fold_engine import FoldEngine
from coding_agent.structured_context.migration import migrate_legacy_session
from coding_agent.structured_context.session_lock import SessionLock
from coding_agent.structured_context.session_store import StructuredSessionStore
from coding_agent.structured_context.state_compact import StateCompactConfig, StateCompactor
from coding_agent.structured_context.structured_context import StructuredContextConfig
from coding_agent.structured_context.subagent_report import parse_subagent_report
from coding_agent.structured_context.token_counter import TokenCounter, TokenUsage
from coding_agent.structured_context.models import Finding, PlanStep, TaskState, ToolState


class FoldLLM:
    def __init__(self, text: str | None = None, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0

    def chat(self, messages, tools=None, *, reasoning_effort=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("fold model unavailable")
        return LLMResponse(text=self.text, stop_reason="end_turn")


def _make_fold_context(tmp_path: Path, *, llm: FoldLLM | None) -> object:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = StructuredSessionStore(tmp_path / ".agent" / "sessions")
    config = StructuredContextConfig(max_context_tokens=1000)
    context = store.create_context(
        session_id="fold-session",
        user_task="fold task",
        provider="test",
        model="test",
        workspace_root=workspace,
        system_prompt="sys",
        prefix_hash="p",
        tools_hash="t",
        context_config=config,
        context_window_limit=1000,
    )
    if llm is not None:
        context.set_fold_engine(FoldEngine(llm, token_counter=context.token_counter))
    context.add_user("fold task")
    context.add_assistant(None, [ToolCall(id="c1", name="grep_search", arguments={"query": "x" * 3500})])
    context.add_tool_result(
        ToolCall(id="c1", name="grep_search", arguments={"query": "x" * 3500}),
        ToolResult.ok("ok"),
    )
    context.finalize()
    context.add_assistant(None, [ToolCall(id="c2", name="grep_search", arguments={"query": "y" * 3500})])
    context.add_tool_result(
        ToolCall(id="c2", name="grep_search", arguments={"query": "y" * 3500}),
        ToolResult.ok("ok"),
    )
    context.finalize()
    return context


def test_llm_fold_uses_model_and_valid_delta(tmp_path: Path) -> None:
    delta = {
        "task_delta": {
            "set": {"progress.current": "merged"},
            "append": [
                {
                    "target": "progress.completed",
                    "dedupe_key": "id",
                    "item": {"id": "p-1", "text": "merged", "completed_step": 1},
                }
            ],
        },
        "tool_delta": {},
    }
    llm = FoldLLM(json.dumps(delta))
    context = _make_fold_context(tmp_path, llm=llm)
    _ = context.messages
    fold_events = [e for e in context.event_log.read_since(0) if e["type"] == "fold_event"]
    assert fold_events
    payload = fold_events[-1]["payload"]
    assert payload["model"]["used"] is True
    assert payload["model"]["calls"] == 1
    assert payload["model"]["fallback_used"] is False
    assert context.task_state.completed[0].id == "p-1"
    assert context.trajectory.epoch_id == 1


def test_llm_fold_falls_back_to_deterministic(tmp_path: Path) -> None:
    context = _make_fold_context(tmp_path, llm=FoldLLM(fail=True))
    _ = context.messages
    fold_events = [e for e in context.event_log.read_since(0) if e["type"] == "fold_event"]
    assert fold_events
    payload = fold_events[-1]["payload"]
    assert payload["model"]["used"] is False
    assert payload["model"]["fallback_used"] is True
    assert payload["model"]["calls"] == 2
    assert payload["model"]["retries"] == 1
    assert context.trajectory.epoch_id == 1


def test_state_compactor_budget_stale_first_and_overflow(tmp_path: Path) -> None:
    task = TaskState.new(task_id="t", objective="o")
    task.remaining = [PlanStep(id="plan-1", text="active", status="active", evidence_refs=["f-keep"])]
    task.key_findings.append(
        Finding(id="f-keep", fact="keep" * 200, evidence=[{"path": "a.py"}], status="valid", updated_step=999)
    )
    task.key_findings.append(Finding(id="f-stale", fact="x" * 200, status="stale", updated_step=1))
    for index in range(100):
        task.key_findings.append(Finding(id=f"f-{index}", fact="x" * 120, status="valid", updated_step=index))
    tool = ToolState.new()
    compactor = StateCompactor(
        TokenCounter(),
        config=StateCompactConfig(task_budget_tokens=800, tool_budget_tokens=800),
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")
    result = compactor.compact(task, tool, artifact_store=artifacts)

    assert result.task_tokens_after <= 800
    assert result.overflow_artifact_id
    assert artifacts.find(result.overflow_artifact_id) is not None
    kept_ids = {finding.id for finding in task.key_findings}
    assert "f-stale" not in kept_ids
    assert "f-keep" in kept_ids


def test_event_replayer_closes_interrupted_tool_batch(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = StructuredSessionStore(tmp_path / ".agent" / "sessions")
    context = store.create_context(
        session_id="crash-session",
        user_task="crash task",
        provider="test",
        model="test",
        workspace_root=workspace,
        system_prompt="sys",
        prefix_hash="p",
        tools_hash="t",
    )
    # Simulate a crash suffix after the baseline checkpoint.
    context.event_log.append("user_message", {"group_id": "g-2", "content": "do it"})
    context.event_log.append(
        "assistant_message",
        {
            "group_id": "g-2",
            "content": None,
            "tool_calls": [{"id": "c1", "name": "read_file", "arguments": {"path": "a.py"}}],
        },
    )
    context.event_log.append(
        "tool_batch_intent",
        {"step": 1, "calls": [{"id": "c1", "name": "read_file", "arguments": {"path": "a.py"}}]},
    )

    loaded = store.load_context("crash-session", system_prompt="sys")
    replay = loaded.replay_result
    assert replay.interrupted_batch is True
    assert replay.missing_tool_call_ids == ["c1"]
    assert loaded.session.status == "recovery_pending"
    messages = [message for group in loaded.trajectory.groups for message in group.messages]
    assert messages[-1].role == "tool"
    assert messages[-1].is_error is True


def test_structured_harness_low_drift_resumes_and_structural_blocks(tmp_path: Path) -> None:
    from coding_agent.structured_context.structured_harness import StructuredHarness

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.txt").write_text("one", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    config = AgentConfig(
        llm=LLMConfig(provider="openai", model="scripted", api_key="demo"),
        workspace=repo,
        agent_dir=tmp_path / ".agent",
    )
    harness = StructuredHarness(config)

    class NoopProvider:
        def chat(self, messages, tools=None, *, reasoning_effort=None):
            return LLMResponse(text="done", stop_reason="end_turn")

    harness.provider = NoopProvider()
    if harness.planner is not None:
        harness.planner.llm = harness.provider
    session = harness.new_session("do nothing")
    # External, non-impacting drift: append an untracked file unrelated to the task.
    (repo / "unrelated.txt").write_text("external", encoding="utf-8")
    loaded = harness.load_session(session.session_id)
    assert loaded.status == "running"
    assert harness._recovery_decisions[session.session_id].action == "RESUME"

    # Structural drift: switch branches after baseline.
    subprocess.run(["git", "checkout", "-b", "other"], cwd=repo, check=True, capture_output=True)
    loaded = harness.load_session(session.session_id)
    assert loaded.status == "blocked"
    result = harness.run_task(loaded)
    assert result.status == "failed"
    assert "blocked" in result.text


def test_subagent_report_parser_degraded_mode() -> None:
    report = parse_subagent_report(
        'intro text\n```json\n{"summary": "s", "findings": [{"fact": "f"}], '
        '"evidence": [], "blockers": [], "next_action": "n"}\n```'
    )
    assert report.summary == "s"
    assert report.findings[0]["fact"] == "f"
    degraded = parse_subagent_report("no json at all")
    assert degraded.summary == "no json at all"
    assert degraded.findings == []


def test_token_calibration_is_persisted(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = StructuredSessionStore(tmp_path / ".agent" / "sessions")
    context = store.create_context(
        session_id="cal-session",
        user_task="task",
        provider="test",
        model="test",
        workspace_root=workspace,
        system_prompt="sys",
        prefix_hash="p",
        tools_hash="t",
    )
    context.token_counter.calibration.record(1000, TokenUsage(prompt_tokens=1500))
    context.update_runtime_cursor(step=1, tool_calls_used=0, status="completed")
    store.save_context(context, checkpoint_kind="terminal")
    loaded = store.load_context("cal-session", system_prompt="sys")
    assert loaded.token_counter.calibration.coefficient == pytest.approx(1.5)
    assert len(loaded.token_counter.calibration.window) == 1


def test_artifact_store_uses_in_memory_index(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    first = store.save("read_file", "one")
    second = store.save("read_file", "two")
    assert store.last_index_seq() == 2
    assert store.find(first["artifact_id"]) == first
    assert store.find(second["artifact_id"]) == second
    assert [entry["seq"] for entry in store.entries()] == [1, 2]


def test_session_lock_is_exclusive(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    with SessionLock(root, "s-1"):
        with pytest.raises(TimeoutError):
            with SessionLock(root, "s-1", timeout=0.05):
                pass


def test_migrate_legacy_session(tmp_path: Path) -> None:
    from coding_agent.context.session import Session, SessionStore as LegacyStore

    workspace = tmp_path / "ws"
    workspace.mkdir()
    legacy_dir = tmp_path / "legacy"
    legacy = Session.new(user_task="add health check", provider="openai", model="m")
    legacy.plan = ["inspect", "implement", "verify"]
    legacy.messages = [
        Message(role="user", content="add health check"),
        Message(role="assistant", content=None, tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "app.py"})]),
        Message(role="tool", content="ok", tool_call_id="c1", name="read_file"),
    ]
    LegacyStore(legacy_dir).save(legacy)

    structured_store = StructuredSessionStore(tmp_path / ".agent" / "sessions")
    context = migrate_legacy_session(
        legacy_dir,
        legacy.id,
        structured_store,
        workspace_root=workspace,
        system_prompt="sys",
        prefix_hash="p",
        tools_hash="t",
    )
    assert context.task_state.remaining[0].text == "inspect"
    assert sum(len(group.messages) for group in context.trajectory.groups) == 3
    assert context.session.metrics.get("migrated_from_legacy") is True


def test_structured_harness_replan_after_high_drift(tmp_path: Path) -> None:
    import hashlib

    from coding_agent.structured_context.structured_harness import StructuredHarness

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.txt").write_text("one", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    config = AgentConfig(
        llm=LLMConfig(provider="openai", model="scripted", api_key="demo"),
        workspace=repo,
        planner_enabled=True,
        agent_dir=tmp_path / ".agent",
    )
    harness = StructuredHarness(config)

    class ReplanProvider:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, tools=None, *, reasoning_effort=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(text="1. replan inspect\n2. replan verify", stop_reason="end_turn")
            return LLMResponse(text="done after replan", stop_reason="end_turn")

    provider = ReplanProvider()
    harness.provider = provider
    harness.planner.llm = provider
    session = harness.new_session("task")
    context = harness._contexts[session.session_id]
    expected = context.workspace_expected()
    expected.expected_dirty = [{"path": "a.txt", "status": "M", "sha256": hashlib.sha256(b"one").hexdigest()}]
    expected.postconditions = [dict(expected.expected_dirty[0])]
    expected.fingerprint = expected.recompute_fingerprint()
    harness.structured_store.save_context(context, checkpoint_kind="baseline")

    (repo / "a.txt").write_text("two", encoding="utf-8")
    loaded = harness.load_session(session.session_id)
    decision = harness._recovery_decisions[session.session_id]
    assert decision.action == "REPLAN"
    assert loaded.status == "recovery_pending"

    result = harness.run_task(loaded)
    new_context = harness._contexts[session.session_id]
    assert result.status == "completed"
    assert new_context.trajectory.epoch_id == 1
    assert new_context.session.plan == ["replan inspect", "replan verify"]
    assert new_context.session.runtime_cursor["recovery"]["replan_attempts"] == 1


def test_structured_harness_subagent_report_and_file_propagation(tmp_path: Path) -> None:
    from coding_agent.structured_context.structured_harness import StructuredHarness

    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = AgentConfig(
        llm=LLMConfig(provider="openai", model="scripted", api_key="demo"),
        workspace=workspace,
        agent_dir=tmp_path / ".agent",
    )
    harness = StructuredHarness(config)

    class SubagentProvider:
        def __init__(self) -> None:
            self.turns = 0

        def chat(self, messages, tools=None, *, reasoning_effort=None):
            self.turns += 1
            if self.turns == 1:
                return LLMResponse(
                    text=None,
                    tool_calls=[ToolCall(id="s1", name="write_file", arguments={"path": "sub.txt", "content": "x"})],
                    stop_reason="tool_calls",
                )
            return LLMResponse(
                text='{"summary": "did", "findings": [], "evidence": [], "blockers": [], "next_action": "ok"}',
                stop_reason="end_turn",
            )

    harness.provider = SubagentProvider()
    session = harness.new_session("task")
    result = harness._run_subagent("do sub", 3, parent_depth=0)
    assert result.status == "completed"
    assert result.structured_report is not None
    assert result.structured_report["summary"] == "did"
    context = harness._contexts[session.session_id]
    paths = [item["path"] for item in context.workspace_expected().postconditions]
    assert "sub.txt" in paths


def test_structured_harness_resolves_explicit_dirs(tmp_path: Path) -> None:
    from coding_agent.structured_context.structured_harness import StructuredHarness

    config = AgentConfig(
        llm=LLMConfig(provider="openai", model="scripted", api_key="demo"),
        workspace=tmp_path / "ws",
        agent_dir=tmp_path / ".agent",
        session_dir=tmp_path / "custom-sessions",
        trace_dir=tmp_path / "custom-traces",
    )
    session_dir, trace_dir = StructuredHarness._resolve_dirs(config)
    assert session_dir == tmp_path / "custom-sessions"
    assert trace_dir == tmp_path / "custom-traces"

    default_config = AgentConfig(
        llm=LLMConfig(provider="openai", model="scripted", api_key="demo"),
        workspace=tmp_path / "ws",
        agent_dir=tmp_path / ".agent",
    )
    session_dir, trace_dir = StructuredHarness._resolve_dirs(default_config)
    assert session_dir == tmp_path / ".agent" / "sessions"
    assert trace_dir == tmp_path / ".agent" / "traces"
