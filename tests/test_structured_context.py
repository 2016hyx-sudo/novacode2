from __future__ import annotations

from pathlib import Path

from coding_agent.structured_context.artifact_store import ArtifactStore
from coding_agent.structured_context.event_log import EventLog
from coding_agent.structured_context.models import TaskState
from coding_agent.structured_context.session_store import StructuredSessionStore
from coding_agent.structured_context.state_merge import merge_task_delta
from coding_agent.structured_context.workspace import WorkspaceFingerprint


def test_event_log_append_and_replay(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.append("session_start", {"task": "hello"})
    log.append("user_message", {"content": "hi"})
    assert log.last_seq == 2
    events = log.read_since(0)
    assert [e["seq"] for e in events] == [1, 2]
    assert log.validate_anchor(log.anchor())


def test_artifact_store_roundtrip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    entry = store.save("read_file", "line1\nline2", tool_call_id="call_1", arguments={"path": "a.py"})
    assert store.read(entry["artifact_id"]) == b"line1\nline2"
    assert store.find(entry["artifact_id"]) is not None


def test_state_delta_merge() -> None:
    state = TaskState.new(task_id="t", objective="o").to_dict()
    state["key_findings"].append({"id": "f-1", "fact": "x", "evidence": [], "status": "valid"})
    notes = merge_task_delta(
        state,
        {
            "set": {"progress.current": "implement"},
            "append": [
                {
                    "target": "progress.completed",
                    "dedupe_key": "id",
                    "item": {"id": "p-1", "text": "done", "completed_step": 1},
                }
            ],
            "mark_stale": [{"target": "key_findings", "id": "f-1", "reason": "changed"}],
        },
    )
    assert state["progress"]["current"] == "implement"
    assert state["progress"]["completed"][0]["id"] == "p-1"
    assert state["key_findings"][0]["status"] == "stale"
    assert notes


def test_key_sequences_merge_and_roundtrip() -> None:
    state = TaskState.new(task_id="t", objective="o").to_dict()
    merge_task_delta(
        state,
        {
            "append": [
                {
                    "target": "key_sequences",
                    "dedupe_key": "id",
                    "item": {
                        "id": "k-1",
                        "pattern": "down (step 7) then up (step 8)",
                        "intent": "push a block, then revert the rule change",
                        "step_range": "7-8",
                        "status": "valid",
                    },
                }
            ]
        },
    )
    assert state["key_sequences"][0]["id"] == "k-1"
    restored = TaskState.from_dict(state)
    assert restored.key_sequences[0].pattern == "down (step 7) then up (step 8)"
    assert restored.key_sequences[0].intent == "push a block, then revert the rule change"
    assert restored.key_sequences[0].step_range == "7-8"


def test_fold_prompt_instructs_key_sequences() -> None:
    from coding_agent.llm.base import Message
    from coding_agent.structured_context.fold_engine import FoldEngine
    from coding_agent.structured_context.models import InteractionGroup, ToolState

    group = InteractionGroup(group_id="g-1", epoch_id=0, created_step=0)
    group.messages.append(Message(role="user", content="hello"))
    engine = FoldEngine(None)
    messages = engine._build_request_messages(
        TaskState.new(task_id="t", objective="o"), ToolState.new(), [group]
    )
    system_prompt = messages[0].content
    assert "key_sequences" in system_prompt
    assert "intent" in system_prompt
    assert "exploratory" in system_prompt


def test_workspace_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    import subprocess

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.txt").write_text("one", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    fp = WorkspaceFingerprint(repo)
    expected = fp.expected_from_actual()
    assert fp.diff(expected).severity == "NONE"
    (repo / "a.txt").write_text("two", encoding="utf-8")
    report = fp.diff(expected)
    assert report.severity in {"HIGH", "LOW", "STRUCTURAL"}
    assert report.unexpected_changes or report.hash_mismatches


def test_fold_event_logs_token_statistics(tmp_path: Path) -> None:
    from coding_agent.llm.base import ToolCall
    from coding_agent.structured_context.session_store import StructuredSessionStore
    from coding_agent.structured_context.structured_context import StructuredContextConfig
    from coding_agent.tools.base import ToolResult

    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = StructuredSessionStore(tmp_path / ".agent" / "sessions")
    config = StructuredContextConfig(max_context_tokens=1000)
    context = store.create_context(
        session_id="fold-stats",
        user_task="task",
        provider="test",
        model="test",
        workspace_root=workspace,
        system_prompt="sys",
        prefix_hash="p",
        tools_hash="t",
        context_config=config,
        context_window_limit=1000,
    )
    context.add_user("task")
    context.add_assistant(None, [ToolCall(id="call_1", name="grep_search", arguments={"query": "needle"})])
    context.add_tool_result(
        ToolCall(id="call_1", name="grep_search", arguments={"query": "needle"}),
        ToolResult.ok("result"),
    )
    context.finalize()
    # A second, unprotected tool group that is large enough to force a fold.
    context.add_assistant(None, [ToolCall(id="call_2", name="grep_search", arguments={"query": "x" * 4000})])
    context.add_tool_result(
        ToolCall(id="call_2", name="grep_search", arguments={"query": "x" * 4000}),
        ToolResult.ok("ok"),
    )
    context.finalize()
    _ = context.messages
    fold_events = [event for event in context.event_log.read_since(0) if event["type"] == "fold_event"]
    assert fold_events
    payload = fold_events[0]["payload"]
    assert payload["trigger"]["threshold_tokens"] == int(1000 * 0.70)
    assert payload["before"]["total"] > payload["trigger"]["threshold_tokens"]
    assert set(payload["after"]) >= {"stable_prefix", "task_tool_state", "recent_trajectory", "agent_state", "total"}
    assert payload["folded"]["group_count"] == 1
    assert "compression_ratio" in payload["folded"]
    assert payload["folded"]["compression_ratio"] == payload["folded"]["residual_ratio"]
    assert abs(
        payload["folded"]["fold_reduction_ratio"]
        - (1 - payload["folded"]["residual_ratio"])
    ) < 1e-6
    assert context.session.metrics["last_fold"]["fold_id"] == payload["fold_id"]

    archive = context.trajectory_archive.read_groups()
    archived_ids = {group["group_id"] for group in archive}
    assert payload["folded"]["group_ids"][0] in archived_ids
    prompt_group_ids = {group.group_id for group in context.trajectory.groups}
    assert payload["folded"]["group_ids"][0] not in prompt_group_ids


def test_workspace_fingerprint_inside_git_subdirectory(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    workspace = repo / "sub" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "a.txt").write_text("one", encoding="utf-8")
    fp = WorkspaceFingerprint(workspace)
    actual = fp.actual()
    assert [x["path"] for x in actual["untracked"]] == ["a.txt"]
    expected = fp.expected_from_actual()
    assert fp.diff(expected).severity == "NONE"


def test_session_create_save_load(tmp_path: Path) -> None:
    root = tmp_path / ".agent" / "sessions"
    store = StructuredSessionStore(root)
    context = store.create_context(
        session_id="sess-1",
        user_task="add health check",
        provider="test",
        model="test-model",
        workspace_root=tmp_path / "ws",
        system_prompt="you are a coding agent",
        prefix_hash="p",
        tools_hash="t",
    )
    (tmp_path / "ws").mkdir(exist_ok=True)
    context.add_user("add health check")
    context.add_assistant("ok", [])
    context.update_runtime_cursor(step=1, tool_calls_used=0, status="completed")
    result = store.save_context(context, checkpoint_kind="terminal")
    assert result["drift"]["severity"] in {"NONE", "HIGH", "LOW"}

    loaded = store.load_context("sess-1", system_prompt="you are a coding agent")
    assert loaded.task_state.objective == "add health check"
    assert len(loaded.trajectory.groups) >= 1
    assert loaded.session.last_checkpoint["seq"] >= 1


def test_structured_harness_run_and_resume(tmp_path: Path) -> None:
    from config import AgentConfig, LLMConfig
    from coding_agent.structured_context.structured_harness import StructuredHarness
    from demo import ScriptedDemoProvider

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "app.py").write_text("def greet(name): return name", encoding="utf-8")
    config = AgentConfig(
        llm=LLMConfig(provider="openai", model="scripted", api_key="demo"),
        workspace=workspace,
        planner_enabled=True,
        agent_dir=tmp_path / ".agent",
    )
    harness = StructuredHarness(config)
    harness.provider = ScriptedDemoProvider()
    harness.planner.llm = harness.provider

    session = harness.new_session("add health check")
    result = harness.run_task(session)
    assert result.status == "completed"
    assert harness.structured_store.list_sessions()

    class FinalProvider:
        def chat(self, messages, tools=None, *, reasoning_effort=None):
            from coding_agent.llm.base import LLMResponse

            return LLMResponse(text="done", stop_reason="end_turn")

    harness.provider = FinalProvider()
    harness.planner.llm = harness.provider
    loaded = harness.load_session(session.session_id)
    resumed = harness.run_task(loaded, "add one more function")
    assert resumed.status == "completed"
