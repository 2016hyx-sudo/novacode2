"""Structured Harness: wires AgentLoop to structured context and checkpointing."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from config import AgentConfig

from .. import Harness
from ..context.session import new_session_id
from ..llm.base import LLMProvider
from ..runtime.trace import TraceEvent, TraceWriter
from ..tools.executor import ToolExecutor
from .models import DriftReport, StructuredSession, canonical_json, sha256_text
from .read_artifact_tool import ReadArtifactTool
from .recovery import RecoveryEngine, RecoveryPolicy
from .session_store import StructuredSessionStore
from .structured_context import StructuredContext, StructuredContextConfig


class StructuredHarness(Harness):
    """Harness variant that persists structured state under ``agent_dir``."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        provider: LLMProvider | None = None,
        trace: TraceWriter | None = None,
        listeners: list[Callable[[TraceEvent], None]] | None = None,
    ) -> None:
        if trace is None:
            trace = TraceWriter(config.agent_dir / "traces", listeners=listeners)
        if provider is None:
            from ..llm import create_provider

            provider = create_provider(config.llm)
        from ..context.session import SessionStore
        from ..runtime.planner import Planner
        from ..runtime.validator import Validator

        # Base Harness builds tools and budgets; its legacy session store is unused.
        super().__init__(
            config,
            provider=provider,
            trace=trace,
            session_store=SessionStore(config.agent_dir / "sessions"),
            planner=Planner(provider, trace=trace) if config.planner_enabled else None,
            validator=Validator(
                require_verification_after_edit=config.constraints.require_verification_after_edit
            ),
        )
        self.structured_store = StructuredSessionStore(config.agent_dir / "sessions")
        self._contexts: dict[str, StructuredContext] = {}
        self._active_session_id: str | None = None
        self._structured_config = StructuredContextConfig(max_context_tokens=config.structured_context_window_limit)

        # Rebuild tools with the runtime state directory explicitly protected.
        protected: list[str] = []
        try:
            rel = config.agent_dir.expanduser().resolve().relative_to(self.workspace)
            protected.append(rel.parts[0])
        except ValueError:
            pass
        from ..tools import build_tool_registry

        self.tools = build_tool_registry(
            self.workspace,
            config.constraints,
            subagent_tool=self._make_subagent_tool(0),
            protected_rel=protected,
        )
        self.tools.register(ReadArtifactTool(self._active_artifact_store))
        self.executor = ToolExecutor(
            self.tools,
            max_retries=config.constraints.tool_max_retries,
            tool_timeout=config.constraints.tool_timeout,
            max_output_chars=config.constraints.max_output_chars,
            trace=trace,
            budget=self.budget,
        )
        self._tools_hash = sha256_text(
            canonical_json([schema.__dict__ for schema in self.tools.schemas()])
        )
        self._prefix_hash = sha256_text(config.system_prompt)
        self.recovery = RecoveryEngine(RecoveryPolicy())
        self._drift_reports: dict[str, DriftReport] = {}

    def _new_context(self, session: StructuredSession, workspace_root: Path) -> StructuredContext:
        context = self.structured_store.create_context(
            session_id=session.session_id,
            user_task=session.user_task,
            provider=session.provider,
            model=session.model,
            workspace_root=workspace_root,
            system_prompt=self.config.system_prompt,
            prefix_hash=self._prefix_hash,
            tools_hash=self._tools_hash,
            context_config=self._structured_config,
            context_window_limit=self.config.structured_context_window_limit,
        )
        context.set_checkpoint_callback(lambda step: self._periodic_checkpoint(session.session_id, step))
        context.set_trace(self.trace)
        self._contexts[session.session_id] = context
        return context

    def _active_artifact_store(self) -> Any:
        context = self._contexts.get(self._active_session_id or "")
        return context.artifact_store if context is not None else None

    def _periodic_checkpoint(self, session_id: str, step: int) -> None:
        context = self._contexts.get(session_id)
        if context is None:
            return
        context.update_runtime_cursor(
            step=step,
            tool_calls_used=int((context.session.runtime_cursor.get("limits") or {}).get("tool_calls_used", 0)),
            status="running",
        )
        self.structured_store.save_context(context, checkpoint_kind="periodic")

    def new_session(self, user_task: str) -> StructuredSession:
        session_id = new_session_id()
        self._active_session_id = session_id
        session = StructuredSession(
            session_id=session_id,
            created_at="",
            updated_at="",
            provider=self.config.llm.provider,
            model=self.config.llm.model,
            workspace_root=str(self.workspace),
            user_task=user_task,
            plan=[],
        )
        self._new_context(session, self.workspace)
        self.trace.bind(session_id)
        self.trace.emit(
            "session_start",
            session_id=session_id,
            provider=session.provider,
            model=session.model,
            task=user_task,
            workspace=str(self.workspace),
        )
        return session

    def load_session(self, session_id: str) -> StructuredSession:
        self._active_session_id = session_id
        context = self.structured_store.load_context(
            session_id,
            system_prompt=self.config.system_prompt,
            context_config=self._structured_config,
        )
        context.set_checkpoint_callback(lambda step: self._periodic_checkpoint(session_id, step))
        context.set_trace(self.trace)
        session = context.session
        session.plan = [step.text for step in context.task_state.remaining]
        self._contexts[session_id] = context
        drift = context.workspace_fingerprint.diff(context.workspace_expected())
        if drift.severity != "NONE":
            context.event_log.append("drift_detected", drift.to_dict())
            self._drift_reports[session_id] = drift
            if drift.severity == "STRUCTURAL":
                session.status = "blocked"
                context.session.status = "blocked"
            else:
                session.status = "recovery_pending"
                context.session.status = "recovery_pending"
        self.trace.bind(session_id)
        self.trace.emit(
            "session_resume",
            session_id=session_id,
            task=session.user_task,
            message_count=sum(len(group.messages) for group in context.trajectory.groups),
        )
        return session

    def run_task(
        self,
        session: StructuredSession,
        task: str | None = None,
        *,
        append_user: bool = True,
    ) -> Any:
        self._active_session_id = session.session_id
        context = self._contexts.get(session.session_id)
        if context is None:
            context = self.structured_store.load_context(
                session.session_id,
                system_prompt=self.config.system_prompt,
                context_config=self._structured_config,
            )
            self._contexts[session.session_id] = context

        self.trace.bind(session.session_id)
        drift = self._drift_reports.get(session.session_id)
        if session.status == "blocked":
            from ..agent import AgentRunResult

            self.trace.emit("error", error="session is blocked by structural drift", step=0)
            return AgentRunResult(
                text="Session is blocked because workspace/git drift is structural. Resolve the workspace state before resuming.",
                status="failed",
            )
        if session.status == "recovery_pending" and drift is not None:
            context.add_user(f"[Workspace drift detected]\n{drift.summary}\nRe-plan if needed before continuing.")
            context.event_log.append("recovery_decision", {"action": "REPLAN", "reason": drift.summary})
            session.status = "running"
            context.session.status = "running"

        effective_task = task or (session.user_task if not context.trajectory.groups else None)
        if append_user and effective_task:
            context.add_user(effective_task)
            session.user_task = effective_task

        if self.planner is not None:
            if session.plan:
                from ..runtime.planner import Plan

                self.planner.current_plan = Plan(list(session.plan))
            else:
                self.planner.generate(effective_task or session.user_task)
                session.plan = list(self.planner.current_plan.steps) if self.planner.current_plan else []
            if self.planner.current_plan is not None:
                context.set_plan(self.planner.current_plan)
                session.plan = [step.text for step in context.task_state.remaining]

        agent = self.create_agent(context, agent_name="main")
        result = agent.run()
        context.update_runtime_cursor(
            step=result.steps_used,
            tool_calls_used=result.tool_calls_used,
            status=result.status,
        )
        if self.planner is not None and self.planner.current_plan is not None:
            session.plan = list(self.planner.current_plan.steps)
        checkpoint_kind = "terminal" if result.status in {"completed", "failed", "stopped"} else "periodic"
        checkpoint = self.structured_store.save_context(context, checkpoint_kind=checkpoint_kind)

        self.trace.emit(
            "run_finished",
            status=result.status,
            steps_used=result.steps_used,
            tool_calls_used=result.tool_calls_used,
            checkpoint_seq=checkpoint["manifest"].get("checkpoint_seq"),
            drift=checkpoint["drift"].get("severity"),
        )
        return result
