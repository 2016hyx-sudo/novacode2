from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.agent import AgentLoop
from coding_agent.llm.base import LLMError, LLMResponse, Message, ToolCall
from coding_agent.llm.usage import normalize_usage
from coding_agent.runtime.trace import TraceEvent, TraceWriter
from coding_agent.structured_context.fold_engine import FoldEngine
from coding_agent.structured_context.models import InteractionGroup, TaskState, ToolState
from coding_agent.structured_context.session_store import StructuredSessionStore
from coding_agent.structured_context.structured_context import StructuredContextConfig
from coding_agent.structured_context.token_counter import TokenCounter
from coding_agent.tools.base import ToolResult
from coding_agent.tools.executor import ToolExecutor
from coding_agent.tools.registry import ToolRegistry


def test_usage_normalization_and_anthropic_calibration() -> None:
    openai = normalize_usage(
        {"prompt_tokens": 1000, "completion_tokens": 80, "cached_tokens": 600}
    )
    assert openai == {
        "provider_format": "openai",
        "token_source": "provider_reported",
        "logical_input_tokens": 1000,
        "cache_hit_tokens": 600,
        "cache_creation_input_tokens": 0,
        "fresh_processed_input_tokens": 400,
        "output_tokens": 80,
    }

    anthropic_raw = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 200,
    }
    anthropic = normalize_usage(anthropic_raw)
    assert anthropic["logical_input_tokens"] == 1300
    assert anthropic["fresh_processed_input_tokens"] == 300
    assert anthropic["cache_hit_tokens"] == 1000

    counter = TokenCounter()
    counter.record_usage(650, anthropic_raw)
    assert counter.calibration.coefficient == pytest.approx(2.0)


def test_agent_retry_uses_one_snapshot_and_unique_attempt_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[TraceEvent] = []
    trace = TraceWriter(tmp_path / "traces", session_id="s1", listeners=[events.append])

    class CountingContext:
        def __init__(self) -> None:
            self.reads = 0

        @property
        def messages(self):
            self.reads += 1
            return [Message(role="system", content="stable"), Message(role="user", content="task")]

    class RetryProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.message_objects: list[object] = []

        def chat(self, messages, tools=None, *, reasoning_effort=None):
            self.calls += 1
            self.message_objects.append(messages)
            if self.calls == 1:
                raise LLMError("temporary", retryable=True)
            return LLMResponse(
                text="done",
                usage={
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 900,
                    "cache_creation_input_tokens": 0,
                },
            )

    context = CountingContext()
    provider = RetryProvider()
    tools = ToolRegistry()
    agent = AgentLoop(
        llm=provider,
        tools=tools,
        executor=ToolExecutor(tools),
        context=context,  # type: ignore[arg-type]
        max_llm_retries=1,
        trace=trace,
    )
    monkeypatch.setattr("coding_agent.agent.time.sleep", lambda _: None)

    response = agent._chat_with_retry(step=7)

    assert context.reads == 1
    assert provider.message_objects[0] is not provider.message_objects[1]
    assert provider.message_objects[0] == provider.message_objects[1]
    assert response.normalized_usage["logical_input_tokens"] == 1000

    prepared = [event.data for event in events if event.type == "llm_request_prepared"]
    finished = [event.data for event in events if event.type == "llm_request_finished"]
    assert len(prepared) == len(finished) == 2
    assert prepared[0]["request_id"] != prepared[1]["request_id"]
    assert prepared[0]["request_group_id"] == prepared[1]["request_group_id"]
    assert prepared[0]["payload_hash"] == prepared[1]["payload_hash"]
    assert [item["attempt"] for item in prepared] == [1, 2]
    assert all(item["step"] == 7 and item["agent_role"] == "main" for item in prepared)
    assert all("event_seq_anchor" in item and "epoch_id" in item for item in prepared)
    assert [item["status"] for item in finished] == ["provider_error", "success"]


def test_structured_snapshot_emits_one_estimate_with_anchor(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = StructuredSessionStore(tmp_path / ".agent" / "sessions")
    context = store.create_context(
        session_id="snapshot-session",
        user_task="task",
        provider="test",
        model="test",
        workspace_root=workspace,
        system_prompt="sys",
        prefix_hash="p",
        tools_hash="t",
    )
    events: list[TraceEvent] = []
    trace = TraceWriter(tmp_path / "traces", session_id="snapshot-session", listeners=[events.append])
    context.set_trace(trace)
    context.add_user("task")

    snapshot = context.snapshot_for_request(request_id="req-main", step=3)
    _ = context.messages
    _ = context.messages

    estimates = [event for event in events if event.type == "context_estimate"]
    assert len(estimates) == 1
    assert estimates[0].data["request_id"] == "req-main"
    assert estimates[0].data["step"] == 3
    assert snapshot["metadata"]["event_seq_anchor"] == context.event_log.last_seq
    assert snapshot["metadata"]["epoch_id"] == context.trajectory.epoch_id
    assert tuple(snapshot["messages"])


def test_fold_requests_report_usage_and_parent_request(tmp_path: Path) -> None:
    events: list[TraceEvent] = []
    trace = TraceWriter(tmp_path / "traces", session_id="fold-session", listeners=[events.append])

    class FoldProvider:
        def chat(self, messages, tools=None, *, reasoning_effort=None):
            return LLMResponse(
                text='{"task_delta": {}, "tool_delta": {}}',
                usage={
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 200,
                },
            )

    group = InteractionGroup(
        group_id="g-1",
        epoch_id=0,
        created_step=1,
        messages=[Message(role="user", content="inspect")],
        status="complete",
    )
    engine = FoldEngine(
        FoldProvider(),
        trace=trace,
        provider_name="anthropic",
        model="test-model",
    )
    result = engine.fold(
        task_state=TaskState.new(task_id="t", objective="o"),
        tool_state=ToolState.new(),
        groups=[group],
        epoch_id=1,
        parent_request_id="req-main",
        step=4,
        event_seq_anchor=17,
    )

    assert result.model_used is True
    assert result.logical_input_tokens == 1300
    assert result.output_tokens == 25
    assert len(result.request_ids) == 1
    prepared = [event.data for event in events if event.type == "llm_request_prepared"]
    finished = [event.data for event in events if event.type == "llm_request_finished"]
    assert len(prepared) == len(finished) == 1
    assert prepared[0]["agent_role"] == "fold"
    assert prepared[0]["parent_request_id"] == "req-main"
    assert prepared[0]["event_seq_anchor"] == 17
    assert finished[0]["normalized_usage"]["logical_input_tokens"] == 1300


def test_fold_fallback_traces_every_failed_attempt(tmp_path: Path) -> None:
    events: list[TraceEvent] = []
    trace = TraceWriter(tmp_path / "traces", session_id="fold-failure", listeners=[events.append])

    class FailingProvider:
        def chat(self, messages, tools=None, *, reasoning_effort=None):
            raise RuntimeError("fold unavailable")

    group = InteractionGroup(
        group_id="g-1",
        epoch_id=0,
        created_step=1,
        messages=[Message(role="user", content="inspect")],
        status="complete",
    )
    result = FoldEngine(FailingProvider(), trace=trace).fold(
        task_state=TaskState.new(task_id="t", objective="o"),
        tool_state=ToolState.new(),
        groups=[group],
        epoch_id=1,
        parent_request_id="req-main",
        step=5,
    )

    assert result.fallback_used is True
    assert result.calls == 2
    assert result.retries == 1
    assert len(result.request_ids) == 2
    prepared = [event.data for event in events if event.type == "llm_request_prepared"]
    finished = [event.data for event in events if event.type == "llm_request_finished"]
    assert len(prepared) == len(finished) == 2
    assert len({item["request_id"] for item in prepared}) == 2
    assert all(item["parent_request_id"] == "req-main" for item in prepared)
    assert all(item["status"] == "provider_error" for item in finished)


def test_structured_fold_and_main_retry_share_request_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = StructuredSessionStore(tmp_path / ".agent" / "sessions")
    context = store.create_context(
        session_id="lineage-session",
        user_task="task",
        provider="test",
        model="test",
        workspace_root=workspace,
        system_prompt="sys",
        prefix_hash="p",
        tools_hash="t",
        context_config=StructuredContextConfig(max_context_tokens=1000),
        context_window_limit=1000,
    )
    events: list[TraceEvent] = []
    trace = TraceWriter(tmp_path / "traces", session_id="lineage-session", listeners=[events.append])

    class FoldProvider:
        def chat(self, messages, tools=None, *, reasoning_effort=None):
            return LLMResponse(text='{"task_delta": {}, "tool_delta": {}}')

    class MainProvider:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, tools=None, *, reasoning_effort=None):
            self.calls += 1
            if self.calls == 1:
                raise LLMError("retry main", retryable=True)
            return LLMResponse(text="done", usage={"prompt_tokens": 900})

    context.set_trace(trace)
    context.set_fold_engine(FoldEngine(FoldProvider(), trace=trace))
    context.add_user("task")
    first = ToolCall(id="c1", name="grep_search", arguments={"query": "x" * 3500})
    context.add_assistant(None, [first])
    context.add_tool_result(first, ToolResult.ok("ok"))
    context.finalize()
    second = ToolCall(id="c2", name="grep_search", arguments={"query": "y" * 3500})
    context.add_assistant(None, [second])
    context.add_tool_result(second, ToolResult.ok("ok"))
    context.finalize()

    tools = ToolRegistry()
    agent = AgentLoop(
        llm=MainProvider(),
        tools=tools,
        executor=ToolExecutor(tools),
        context=context,  # type: ignore[arg-type]
        max_llm_retries=1,
        trace=trace,
    )
    monkeypatch.setattr("coding_agent.agent.time.sleep", lambda _: None)
    response = agent._chat_with_retry(step=2)
    assert response.text == "done"

    prepared = [event.data for event in events if event.type == "llm_request_prepared"]
    fold = [item for item in prepared if item["agent_role"] == "fold"]
    main = [item for item in prepared if item["agent_role"] == "main"]
    assert len(fold) == 1
    assert len(main) == 2
    assert fold[0]["parent_request_id"] == main[0]["request_group_id"]
    assert main[0]["request_group_id"] == main[1]["request_group_id"]
    assert len([event for event in events if event.type == "context_estimate"]) == 1
