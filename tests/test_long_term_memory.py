"""Comprehensive test suite for the NovaCode long-term memory system."""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from config import AgentConfig, Constraints
from coding_agent import Harness
from coding_agent.agent import AgentLoop
from coding_agent.context.manager import ContextManager
from coding_agent.llm.base import LLMResponse, ToolCall
from coding_agent.long_term_memory import (
    DeleteMemoryTool,
    HeaderScanner,
    LexicalScorer,
    MemoryEntry,
    MemoryHeader,
    MemoryInjector,
    MemoryLifecycleHook,
    MemoryRetriever,
    MemoryStore,
    MemoryType,
    PrefetchGate,
    QuotaConfig,
    SaveMemoryTool,
    SideQueryEngine,
    SuppressionEngine,
    SuppressionVerdict,
    UpdateMemoryTool,
    create_memory_tools,
)
from coding_agent.long_term_memory.injector import FreshnessGuard
from coding_agent.structured_context.models import Decision, TaskState, ToolExperience, ToolState
from coding_agent.tools.registry import ToolRegistry


# ==============================================================================
# Phase 0 (M0): Models & Store Tests
# ==============================================================================

def test_memory_entry_markdown_roundtrip(tmp_path: Path) -> None:
    header = MemoryHeader(
        name="test_rule",
        type=MemoryType.FEEDBACK,
        description="Always check build flags on Windows",
        created_at="2026-08-20T10:00:00+00:00",
        updated_at="2026-08-20T12:00:00+00:00",
    )
    body = "When compiling under Windows, `--no-daemon` must be passed to avoid locking."
    entry = MemoryEntry(header=header, content=body)

    md_text = entry.to_markdown()
    assert "name: test_rule" in md_text
    assert "type: feedback" in md_text
    assert "Always check build flags on Windows" in md_text
    assert body in md_text

    parsed = MemoryEntry.from_markdown(md_text, file_path="/some/path/test_rule.md")
    assert parsed.name == "test_rule"
    assert parsed.type == MemoryType.FEEDBACK
    assert parsed.description == "Always check build flags on Windows"
    assert parsed.content == body
    assert parsed.file_path == "/some/path/test_rule.md"


def test_memory_store_save_load_delete_and_scope(tmp_path: Path) -> None:
    project_dir = tmp_path / "project_mem"
    global_dir = tmp_path / "global_mem"
    store = MemoryStore(project_dir=project_dir, global_dir=global_dir)

    # 1. Save project-level memory
    p_header = MemoryHeader(name="arch_auth", type=MemoryType.PROJECT, description="Auth architecture decision")
    p_entry = MemoryEntry(header=p_header, content="Use AuthMiddleware.")
    store.save_entry(p_entry, scope="project")
    assert (project_dir / "arch_auth.md").is_file()

    # 2. Save global user memory (auto routed to global)
    u_header = MemoryHeader(name="user_pref", type=MemoryType.USER, description="User python preference")
    u_entry = MemoryEntry(header=u_header, content="Strict type annotations.")
    store.save_entry(u_entry, scope="auto")
    assert (global_dir / "user_pref.md").is_file()

    # 3. Load entry
    loaded_p = store.load_entry("arch_auth")
    assert loaded_p is not None
    assert loaded_p.name == "arch_auth"
    assert "AuthMiddleware" in loaded_p.content

    loaded_u = store.load_entry("user_pref")
    assert loaded_u is not None
    assert loaded_u.name == "user_pref"

    # 4. List headers (<5ms scan)
    headers = store.list_headers()
    assert len(headers) == 2
    names = {h.name for h in headers}
    assert "arch_auth" in names
    assert "user_pref" in names

    # 5. Delete entry
    assert store.delete_entry("arch_auth") is True
    assert not (project_dir / "arch_auth.md").is_file()
    assert store.load_entry("arch_auth") is None
    assert len(store.list_headers()) == 1


def test_memory_store_content_truncation_limit(tmp_path: Path) -> None:
    store = MemoryStore(project_dir=tmp_path / "proj", global_dir=tmp_path / "glob")
    huge_content = "X" * 6000  # 6KB > 4KB quota
    header = MemoryHeader(name="large_mem", type=MemoryType.REFERENCE, description="Large test")
    entry = MemoryEntry(header=header, content=huge_content)

    saved = store.save_entry(entry)
    assert "[... truncated, memory file too large ...]" in saved.content
    assert len(saved.content.encode("utf-8")) <= 4 * 1024 + 100


# ==============================================================================
# Phase 1 (M1): Negative Suppression & Memory Tools Tests
# ==============================================================================

def test_suppression_engine_7_rules(tmp_path: Path) -> None:
    store = MemoryStore(project_dir=tmp_path / "proj", global_dir=tmp_path / "glob")
    engine = SuppressionEngine(store=store)

    # Rule 7: Subagent write prohibition
    h_sub = MemoryHeader(name="sub_entry", type=MemoryType.USER, description="Subagent trying to save")
    v7 = engine.validate_write(MemoryEntry(h_sub, "content"), is_subagent=True)
    assert v7.passed is False
    assert v7.rule_id == 7

    # Rule 6: Secret Hard-Stop (API Key / Token / Secret)
    h_sec = MemoryHeader(name="secret_entry", type=MemoryType.REFERENCE, description="API Key storage")
    v6_openai = engine.validate_write(MemoryEntry(h_sec, "My key is sk-abcdef1234567890abcdef1234567890"))
    assert v6_openai.passed is False
    assert v6_openai.rule_id == 6

    v6_gh = engine.validate_write(MemoryEntry(h_sec, "ghp_111122223333444455556666777788889999"))
    assert v6_gh.passed is False
    assert v6_gh.rule_id == 6

    # Rule 1: Unverified hypotheses in feedback
    h_fb = MemoryHeader(name="fb_unverified", type=MemoryType.FEEDBACK, description="Bug fix hypothesis")
    v1 = engine.validate_write(MemoryEntry(h_fb, "Maybe it works if we change line 10, untested guess."))
    assert v1.passed is False
    assert v1.rule_id == 1

    # Rule 2: Codebase facts (directory tree)
    h_tree = MemoryHeader(name="tree_entry", type=MemoryType.PROJECT, description="Dir layout")
    v2 = engine.validate_write(MemoryEntry(h_tree, "├── src\n│   ├── main.py\n└── tests"))
    assert v2.passed is False
    assert v2.rule_id == 2

    # Rule 3: Single-task ephemeral state
    h_eph = MemoryHeader(name="eph_entry", type=MemoryType.PROJECT, description="Current todo")
    v3 = engine.validate_write(MemoryEntry(h_eph, "TODO: finish line 42 in file.py"))
    assert v3.passed is False
    assert v3.rule_id == 3

    # Rule 4: Generalizing one-off instructions
    h_oneoff = MemoryHeader(name="oneoff_entry", type=MemoryType.USER, description="Only this single run print logs")
    v4 = engine.validate_write(MemoryEntry(h_oneoff, "Always print debug logs just for now"))
    assert v4.passed is False
    assert v4.rule_id == 4

    # Rule 5: Duplicate redundant entries
    h_init = MemoryHeader(name="tailwind_pref", type=MemoryType.USER, description="Use TailwindCSS for all frontend UI components")
    store.save_entry(MemoryEntry(h_init, "Always use TailwindCSS."))

    h_dup = MemoryHeader(name="tailwind_pref2", type=MemoryType.USER, description="Use TailwindCSS for all frontend UI components")
    v5 = engine.validate_write(MemoryEntry(h_dup, "Duplicate tailwind entry."))
    assert v5.passed is False
    assert v5.rule_id == 5
    assert v5.suggested_action == "use_update_memory"


def test_memory_tools_execution(tmp_path: Path) -> None:
    store = MemoryStore(project_dir=tmp_path / "proj", global_dir=tmp_path / "glob")
    save_tool = SaveMemoryTool(store=store)
    update_tool = UpdateMemoryTool(store=store)
    del_tool = DeleteMemoryTool(store=store)

    # 1. save_memory success
    res_save = save_tool.execute(
        name="pytest_no_daemon",
        type="feedback",
        description="Run pytest without daemon on Windows",
        content="Root cause: Daemon locks file descriptors. Fix: Pass --no-daemon flag in pytest commands.",
    )
    assert res_save.success is True
    assert "pytest_no_daemon" in res_save.output

    # 2. save_memory failure due to suppression (e.g. secret)
    res_secret = save_tool.execute(
        name="my_api_key",
        type="reference",
        description="OpenAI Key",
        content="sk-12345678901234567890123456789012",
    )
    assert res_secret.success is False
    assert "suppressed" in res_secret.error

    # 3. update_memory append
    res_update = update_tool.execute(
        name="pytest_no_daemon",
        patch_mode="append",
        content="Applies also to Django test runner.",
    )
    assert res_update.success is True
    loaded = store.load_entry("pytest_no_daemon")
    assert loaded is not None
    assert "Applies also to Django test runner." in loaded.content

    # 4. delete_memory
    res_del = del_tool.execute(name="pytest_no_daemon", reason="No longer needed")
    assert res_del.success is True
    assert store.load_entry("pytest_no_daemon") is None


def test_subagent_tool_isolation(tmp_path: Path) -> None:
    store = MemoryStore(project_dir=tmp_path / "proj", global_dir=tmp_path / "glob")
    subagent_tools = create_memory_tools(store, is_subagent=True)
    assert len(subagent_tools) == 0

    main_tools = create_memory_tools(store, is_subagent=False)
    assert len(main_tools) == 3


# ==============================================================================
# Phase 2 (M2): Retrieval & Injection Tests
# ==============================================================================

def test_prefetch_gate_and_header_scan(tmp_path: Path) -> None:
    store = MemoryStore(project_dir=tmp_path / "proj", global_dir=tmp_path / "glob")

    # Gate on empty store
    assert PrefetchGate.can_prefetch("How to build?", store=store) is False

    # Save a memory
    store.save_entry(MemoryEntry(
        header=MemoryHeader(name="build_script", type=MemoryType.REFERENCE, description="Build commands using make target"),
        content="Use make build-fast",
    ))

    # Gate checks
    assert PrefetchGate.can_prefetch("q", store=store) is False  # command
    assert PrefetchGate.can_prefetch("exit", store=store) is False  # command
    assert PrefetchGate.can_prefetch("How do I build the project?", store=store) is True
    assert PrefetchGate.can_prefetch("How do I build the project?", store=store, is_subagent=True) is False
    assert PrefetchGate.can_prefetch("How do I build?", store=store, session_injected_bytes=70000) is False  # > 60KB

    # Header scan & deduplication
    candidates, manifest = HeaderScanner.scan_manifest(store, already_surfaced_names=set())
    assert len(candidates) == 1
    assert "build_script" in manifest

    candidates_surfaced, _ = HeaderScanner.scan_manifest(store, already_surfaced_names={"build_script"})
    assert len(candidates_surfaced) == 0


def test_lexical_scorer_and_retriever(tmp_path: Path) -> None:
    store = MemoryStore(project_dir=tmp_path / "proj", global_dir=tmp_path / "glob")
    store.save_entry(MemoryEntry(
        header=MemoryHeader(name="tailwind_rule", type=MemoryType.USER, description="Frontend styling preference TailwindCSS"),
        content="Always write utility classes with TailwindCSS.",
    ))
    store.save_entry(MemoryEntry(
        header=MemoryHeader(name="db_migration", type=MemoryType.PROJECT, description="Database schema migration guidelines"),
        content="Use Alembic revision --autogenerate.",
    ))

    retriever = MemoryRetriever(store=store, side_query_provider=None)

    # Query related to Tailwind
    results = retriever.prefetch(
        query="Please help me write a new frontend button with TailwindCSS styles",
        already_surfaced=set(),
        session_injected_bytes=0,
    )
    assert len(results) == 1
    assert results[0].name == "tailwind_rule"

    # Query related to DB
    results_db = retriever.prefetch(
        query="I need to update the database schema with a new migration column",
        already_surfaced=set(),
        session_injected_bytes=0,
    )
    assert len(results_db) == 1
    assert results_db[0].name == "db_migration"


def test_freshness_guard_and_dynamic_slot_injection() -> None:
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    old_iso = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)).isoformat()

    # 1. Fresh memory (saved today)
    fresh_entry = MemoryEntry(
        header=MemoryHeader(name="fresh_mem", type=MemoryType.USER, description="Fresh preference", updated_at=now_iso),
        content="Prefer pytest.",
    )
    formatted_fresh = MemoryInjector.format_entry(fresh_entry)
    assert "<system-reminder>" in formatted_fresh
    assert "saved today" in formatted_fresh
    assert "⚠️" not in formatted_fresh

    # 2. Old memory (> 1 day -> Freshness Warning)
    old_entry = MemoryEntry(
        header=MemoryHeader(name="old_mem", type=MemoryType.FEEDBACK, description="Old feedback", updated_at=old_iso),
        content="Historical lesson.",
    )
    formatted_old = MemoryInjector.format_entry(old_entry)
    assert "<system-reminder>" in formatted_old
    assert "⚠️ This memory is 5 days old" in formatted_old

    # 3. Dynamic wrap of user message
    user_query = "What test framework do we use?"
    wrapped = MemoryInjector.wrap_user_message(user_query, [fresh_entry])
    assert wrapped.startswith("<system-reminder>")
    assert wrapped.endswith(user_query)


# ==============================================================================
# Phase 3 (M3): Lifecycle Hook Tests
# ==============================================================================

def test_lifecycle_hook_extraction(tmp_path: Path) -> None:
    store = MemoryStore(project_dir=tmp_path / "proj", global_dir=tmp_path / "glob")
    hook = MemoryLifecycleHook(store=store)

    # 1. TaskState decisions extraction
    task_state = TaskState.new(task_id="t1", objective="Refactor auth")
    task_state.decisions.append(
        Decision(
            id="d1",
            decision="All auth endpoints must require JWT token verification",
            reason="Security audit requirement for stateless auth",
            evidence=["auth_test.py:45"],
        )
    )

    # 2. ToolState known error pattern extraction
    tool_state = ToolState.new()
    tool_state.profiles["shell"]["known_failures"].append(
        ToolExperience(
            id="exp1",
            kind="tool_experience",
            value={
                "error": "port 8080 already in use",
                "fix": "Kill existing process with fuser -k 8080/tcp or use PORT=8081.",
            },
        )
    )

    harvested = hook.harvest_and_save(task_state=task_state, tool_state=tool_state)
    assert len(harvested) == 2

    # Check store contents
    headers = store.list_headers()
    assert len(headers) == 2
    types = {h.type for h in headers}
    assert MemoryType.PROJECT in types
    assert MemoryType.FEEDBACK in types


# ==============================================================================
# Phase 4 (M4): End-to-End Integration Tests
# ==============================================================================

class MockMemoryProvider:
    """Mock LLM Provider for deterministic E2E test."""

    def __init__(self) -> None:
        self.call_count = 0

    def chat(self, messages: list, tools: Any = None, **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        last_msg = (getattr(messages[-1], "content", None) or "") if messages else ""

        # Step 1: User says remember my preference -> Model calls save_memory
        if "remember my coding preference" in last_msg.lower():
            return LLMResponse(
                text="Saving preference...",
                tool_calls=[
                    ToolCall(
                        id="call_save_1",
                        name="save_memory",
                        arguments={
                            "name": "python_type_annotations",
                            "type": "user",
                            "description": "Always use strict type annotations in Python",
                            "content": "All Python functions must have explicit type annotations and pass mypy.",
                        },
                    )
                ],
            )
        # Step 2: Tool result received -> Model completes
        elif "saved memory" in last_msg.lower() or "successfully" in last_msg.lower():
            return LLMResponse(text="I have remembered your Python typing preference.")
        # Step 3: Next session question -> Prompt has <system-reminder> -> Model answers
        elif "<system-reminder>" in last_msg:
            return LLMResponse(text="According to your saved preference, you require strict type annotations.")
        else:
            return LLMResponse(text="Done.")


def test_e2e_memory_lifecycle_cross_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = AgentConfig(
        workspace=workspace,
        long_term_memory_enabled=True,
        memory_project_dir=workspace / ".agent" / "memories",
    )
    from coding_agent.context.session import SessionStore
    from coding_agent.runtime.trace import TraceWriter
    from coding_agent.runtime.validator import Validator

    provider = MockMemoryProvider()
    trace = TraceWriter(workspace / ".traces")
    session_store = SessionStore(workspace / ".sessions")
    validator = Validator()
    harness = Harness(
        config,
        provider=provider,
        trace=trace,
        session_store=session_store,
        planner=None,
        validator=validator,
    )

    # Session 1: User requests to save memory
    session1 = harness.new_session("Remember my coding preference for Python")
    context1 = ContextManager(config.system_prompt)
    agent1 = harness.create_agent(context1)

    result1 = agent1.run("Remember my coding preference for Python: strict type annotations")
    assert result1.status == "completed", result1.text

    # Verify memory is persisted on disk
    saved_entry = harness.memory_store.load_entry("python_type_annotations")
    assert saved_entry is not None
    assert saved_entry.type == MemoryType.USER

    # Session 2: User asks a coding question; memory is prefetched and injected in turn
    context2 = ContextManager(config.system_prompt)
    agent2 = harness.create_agent(context2)
    result2 = agent2.run("How should I write new Python functions for this codebase?")

    # Verify the User message in context2 received the <system-reminder>
    user_msgs = [m for m in context2.messages if m.role == "user"]
    assert len(user_msgs) == 1
    assert "<system-reminder>" in user_msgs[0].content
    assert "python_type_annotations" in user_msgs[0].content
    assert "How should I write new Python functions" in user_msgs[0].content

    # System prompt remains 100% static (Prefix Invariance)
    assert context2.system_prompt == config.system_prompt
