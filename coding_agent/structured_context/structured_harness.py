"""Structured Harness: wires AgentLoop to structured context and checkpointing."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from config import AgentConfig

from .. import Harness
from ..agent import AgentRunResult
from ..context.session import SessionStore, new_session_id
from ..llm.base import LLMProvider
from ..runtime.episode import EpisodeController, TaskEpisode, TurnDisposition
from ..runtime.trace import TraceEvent, TraceWriter
from ..tools.executor import ToolExecutor
from ..tools.shell import ShellRunner
from .episode_store import EpisodeStore
from .fold_engine import FoldEngine
from .migration import migrate_legacy_session
from .models import DriftReport, StructuredSession, canonical_json, sha256_text
from .read_artifact_tool import ReadArtifactTool
from .recovery import RecoveryDecision, RecoveryEngine, RecoveryPolicy
from .session_lock import SessionLock
from .session_store import StructuredSessionStore
from .structured_context import StructuredContext, StructuredContextConfig
from .subagent_report import parse_subagent_report
from .usage_stats import UsageEventAggregator, UsageStats


class StructuredHarness(Harness):
    """Harness variant that persists structured state under ``agent_dir``."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        provider: LLMProvider | None = None,
        trace: TraceWriter | None = None,
        listeners: list[Callable[[TraceEvent], None]] | None = None,
        shell_runner: ShellRunner | None = None,
    ) -> None:
        session_dir, trace_dir = self._resolve_dirs(config)
        self.usage_aggregator = UsageEventAggregator()
        if trace is None:
            trace = TraceWriter(trace_dir, listeners=(listeners or []) + [self.usage_aggregator])
        if provider is None:
            from ..llm import create_provider

            provider = create_provider(config.llm)
        from ..runtime.planner import Planner
        from ..runtime.validator import Validator

        # Base Harness builds tools and budgets; its legacy session store is unused
        # for the structured path but is pointed at the resolved session directory.
        super().__init__(
            config,
            provider=provider,
            trace=trace,
            session_store=SessionStore(session_dir),
            planner=Planner(provider, trace=trace) if config.planner_enabled else None,
            validator=Validator(
                require_verification_after_edit=config.constraints.require_verification_after_edit
            ),
            shell_runner=shell_runner,
        )
        self.session_root = session_dir
        self.trace_root = trace_dir
        self.structured_store = StructuredSessionStore(session_dir)
        self._contexts: dict[str, StructuredContext] = {}
        self._active_session_id: str | None = None
        self._episode_controllers: dict[str, EpisodeController] = {}
        self._structured_config = StructuredContextConfig(max_context_tokens=config.structured_context_window_limit)

        # Rebuild tools with every runtime state directory explicitly protected.
        protected: list[str] = []
        for runtime_dir in (config.agent_dir, session_dir, trace_dir):
            try:
                rel = Path(runtime_dir).expanduser().resolve().relative_to(self.workspace)
                protected.append(str(rel))
            except (ValueError, OSError):
                pass
        skill_project_dir = config.skill_project_dir or (self.workspace / ".agent" / "skills")
        try:
            protected.append(str(Path(skill_project_dir).expanduser().resolve().relative_to(self.workspace)))
        except (ValueError, OSError):
            pass
        from ..tools import build_tool_registry

        self.tools = build_tool_registry(
            self.workspace,
            config.constraints,
            subagent_tool=self._make_subagent_tool(0),
            protected_rel=protected,
            shell_runner=self.shell_runner,
        )
        self.tools.register(ReadArtifactTool(self._active_artifact_store))
        if config.long_term_memory_enabled and self.memory_store is not None:
            from ..long_term_memory.tools import create_memory_tools

            for m_tool in create_memory_tools(self.memory_store, is_subagent=False):
                self.tools.register(m_tool)
        if config.skills_enabled:
            from ..skills.bank import SkillBank
            from ..skills.tool import InvokeSkillTool

            self.skill_bank = SkillBank(
                project_dir=skill_project_dir,
                user_dir=config.skill_user_dir or (Path.home() / ".novacode" / "skills"),
            )
            self.tools.register(
                InvokeSkillTool(
                    self.skill_bank,
                    artifact_store=self._active_artifact_store,
                    fork_runner=self._run_skill_subagent,
                    inline_token_limit=config.skill_inline_token_limit,
                )
            )
        from ..tools.task_outcome import ReportTaskOutcomeTool

        self.tools.register(ReportTaskOutcomeTool(self._submit_task_outcome))
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
        self._recovery_decisions: dict[str, RecoveryDecision] = {}

    # ------------------------------------------------------------------ wiring

    @staticmethod
    def _resolve_dirs(config: AgentConfig) -> tuple[Path, Path]:
        """CLI/tests may pass explicit directories; otherwise derive from agent_dir.

        The legacy defaults ``.sessions`` / ``.traces`` are treated as absent so
        AgentConfig objects constructed directly for tests still get the
        structured layout beneath ``agent_dir``.
        """
        agent_dir = Path(config.agent_dir)
        session_dir = config.session_dir
        trace_dir = config.trace_dir
        if not config.session_dir_explicit and Path(session_dir) == Path(".sessions"):
            session_dir = agent_dir / "sessions"
        if not config.trace_dir_explicit and Path(trace_dir) == Path(".traces"):
            trace_dir = agent_dir / "traces"
        return Path(session_dir), Path(trace_dir)

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
        self._wire_context(context, session.session_id)
        return context

    def _wire_context(self, context: StructuredContext, session_id: str) -> None:
        context.set_checkpoint_callback(lambda step: self._periodic_checkpoint(session_id, step))
        context.set_post_fold_callback(lambda step: self._post_fold_checkpoint(session_id, step))
        context.set_trace(self.trace)
        context.set_fold_engine(self._fold_engine(context))
        context.set_tool_schemas(self.tools.schemas())
        context.set_planner_enabled(self.planner is not None)
        self._contexts[session_id] = context

    def _active_artifact_store(self) -> Any:
        context = self._contexts.get(self._active_session_id or "")
        return context.artifact_store if context is not None else None

    def _episode_store(self, session_id: str) -> EpisodeStore:
        return EpisodeStore(self.structured_store.session_dir(session_id))

    def _submit_task_outcome(self, report: TurnDisposition) -> None:
        session_id = self._active_session_id or ""
        controller = self._episode_controllers.get(session_id)
        if controller is not None:
            controller.submit(report)
        else:
            super()._submit_task_outcome(report)

    @staticmethod
    def _episode_evidence(context: StructuredContext, episode: TaskEpisode) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for event in context.event_log.read_since(max(0, episode.start_event_seq - 1)):
            payload = dict(event.get("payload") or {})
            evidence.append(
                {
                    "seq": int(event.get("seq", 0)),
                    "ref": f"event-{int(event.get('seq', 0))}",
                    "type": event.get("type"),
                    "episode_id": episode.episode_id,
                    **payload,
                }
            )
        return evidence

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

    def _post_fold_checkpoint(self, session_id: str, step: int) -> None:
        context = self._contexts.get(session_id)
        if context is None:
            return
        context.update_runtime_cursor(
            step=step,
            tool_calls_used=int((context.session.runtime_cursor.get("limits") or {}).get("tool_calls_used", 0)),
            status="running",
        )
        self.structured_store.save_context(context, checkpoint_kind="post_fold")
        if self.config.long_term_memory_enabled and self.memory_store is not None:
            from ..long_term_memory.hook import MemoryLifecycleHook

            hook = MemoryLifecycleHook(self.memory_store)
            hook.on_fold(context.task_state, context.tool_state)
        controller = self._episode_controllers.get(session_id)
        if (
            controller is not None
            and self.config.skill_evolution_enabled
            and self.config.skills_enabled
            and self.skill_bank is not None
        ):
            from ..skills.candidates import PendingCandidateStore
            from ..skills.hook import SkillLifecycleHook

            SkillLifecycleHook(PendingCandidateStore(self.skill_bank.project_dir)).on_fold(
                episode=controller.episode,
                evidence=self._episode_evidence(context, controller.episode),
            )

    def _lock(self, session_id: str) -> SessionLock:
        return SessionLock(self.session_root, session_id)

    # ------------------------------------------------------------ session flow

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
        context = self._contexts[session_id]
        episode = TaskEpisode.new(
            session_id=session_id,
            objective=user_task,
            start_event_seq=context.event_log.last_seq + 1,
            success_criteria=list(context.task_state.success_criteria),
        )
        self._episode_controllers[session_id] = EpisodeController(episode)
        self._episode_store(session_id).append("episode_started", episode)
        self.trace.bind(session_id)
        self.usage_aggregator.stats.reset()
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
        with self._lock(session_id):
            return self._load_session_locked(session_id)

    def _load_session_locked(self, session_id: str) -> StructuredSession:
        self._active_session_id = session_id
        context = self.structured_store.load_context(
            session_id,
            system_prompt=self.config.system_prompt,
            context_config=self._structured_config,
        )
        self._wire_context(context, session_id)
        session = context.session
        session.plan = [step.text for step in context.task_state.remaining]
        self._contexts[session_id] = context
        episode = self._episode_store(session_id).latest()
        if episode is None:
            episode = TaskEpisode.new(
                session_id=session_id,
                objective=context.task_state.objective or session.user_task,
                start_event_seq=context.event_log.last_seq + 1,
                success_criteria=list(context.task_state.success_criteria),
            )
            self._episode_store(session_id).append("episode_started", episode)
        self._episode_controllers[session_id] = EpisodeController(episode)

        drift = context.workspace_fingerprint.diff(context.workspace_expected())
        impact_paths = self._impact_paths(context)
        severity = self.recovery.classify(drift, impact_paths) if drift.severity != "NONE" else "NONE"
        replan_attempts = int(
            (context.session.runtime_cursor.get("recovery") or {}).get("replan_attempts", 0)
        )
        decision = self.recovery.decide(
            checkpoint_valid=True,
            log_valid=True,
            drift=drift,
            severity=severity,
            replan_attempts=replan_attempts,
            affected_paths=sorted(impact_paths),
        )
        if drift.severity != "NONE":
            context.event_log.append("drift_detected", drift.to_dict())
        context.event_log.append("recovery_decision", decision.to_dict())
        self._drift_reports[session_id] = drift
        self._recovery_decisions[session_id] = decision

        if decision.action == "BLOCKED":
            session.status = "blocked"
            context.session.status = "blocked"
            self.structured_store.save_context(context, checkpoint_kind="blocked")
        elif decision.action == "REPLAN":
            session.status = "recovery_pending"
            context.session.status = "recovery_pending"
        else:
            # RESUME for NONE / IGNORED / LOW.  LOW paths are marked stale now.
            if severity in {"LOW", "IGNORED"} and impact_paths:
                context.mark_findings_stale(set(impact_paths))
            session.status = "running"
            context.session.status = "running"

        replay = getattr(context, "replay_result", None)
        if replay is not None and replay.interrupted_batch and session.status != "blocked":
            session.status = "recovery_pending"
            context.session.status = "recovery_pending"

        self.trace.bind(session_id)
        self.usage_aggregator.stats = UsageStats.from_dict(session.metrics.get("usage") or {})
        self.trace.emit(
            "session_resume",
            session_id=session_id,
            task=session.user_task,
            message_count=sum(len(group.messages) for group in context.trajectory.groups),
            recovery_action=decision.action,
        )
        return session

    def run_task(
        self,
        session: StructuredSession,
        task: str | None = None,
        *,
        append_user: bool = True,
    ) -> AgentRunResult:
        with self._lock(session.session_id):
            return self._run_task_locked(session, task, append_user=append_user)

    def _run_task_locked(
        self,
        session: StructuredSession,
        task: str | None,
        *,
        append_user: bool,
    ) -> AgentRunResult:
        self._active_session_id = session.session_id
        context = self._contexts.get(session.session_id)
        if context is not None:
            # Provider may have been swapped after new_session (demo/test pattern).
            context.set_fold_engine(self._fold_engine(context))
        if context is None:
            context = self.structured_store.load_context(
                session.session_id,
                system_prompt=self.config.system_prompt,
                context_config=self._structured_config,
            )
            self._wire_context(context, session.session_id)

        self.trace.bind(session.session_id)
        decision = self._recovery_decisions.get(session.session_id)
        drift = self._drift_reports.get(session.session_id)

        if session.status == "blocked" or (decision is not None and decision.action == "BLOCKED"):
            context.session.status = "blocked"
            self.structured_store.save_context(context, checkpoint_kind="blocked")
            self.trace.emit("error", error="session is blocked by structural drift", step=0)
            return AgentRunResult(
                text=(
                    "Session is blocked because workspace/git drift is structural. "
                    "Resolve the workspace state or restore the checkpoint before resuming."
                ),
                status="failed",
            )

        if decision is not None and decision.action == "REPLAN":
            self._replan_after_drift(context, drift, decision)

        effective_task = task or (session.user_task if not context.trajectory.groups else None)
        controller = self._episode_controllers[session.session_id]
        if append_user and effective_task and context.trajectory.groups:
            transition = controller.begin_turn(effective_task)
            if transition == "reopened":
                self._episode_store(session.session_id).append("episode_reopened", controller.episode)
                if self.config.skill_evolution_enabled and self.skill_bank is not None:
                    from ..skills.candidates import PendingCandidateStore

                    PendingCandidateStore(self.skill_bank.project_dir).mark_episode_stale(
                        controller.episode.episode_id,
                        older_than_version=controller.episode.outcome_version,
                    )
                context.event_log.append("episode_reopened", controller.episode.to_dict())
            elif transition == "new_episode":
                controller.supersede()
                self._episode_store(session.session_id).append("episode_superseded", controller.episode)
                episode = TaskEpisode.new(
                    session_id=session.session_id,
                    objective=effective_task,
                    started_turn=controller.turn,
                    start_event_seq=context.event_log.last_seq + 1,
                    success_criteria=list(context.task_state.success_criteria),
                )
                controller = EpisodeController(episode)
                self._episode_controllers[session.session_id] = controller
                context.task_state.objective = effective_task
                self._episode_store(session.session_id).append("episode_started", episode)
                context.event_log.append("episode_started", episode.to_dict())
        else:
            controller.pending_report = None
        if append_user and effective_task:
            context.add_user(effective_task)
            session.user_task = effective_task

        if self.planner is not None:
            from ..runtime.planner import Plan

            if session.plan:
                self.planner.current_plan = Plan(list(session.plan))
            else:
                self.planner.generate(effective_task or session.user_task)
                session.plan = list(self.planner.current_plan.steps) if self.planner.current_plan else []
            if self.planner.current_plan is not None:
                context.set_plan(self.planner.current_plan)
                session.plan = [step.text for step in context.task_state.remaining]

        agent = self.create_agent(context, agent_name="main")
        result = agent.run()
        unresolved = [
            item.text
            for item in context.task_state.unresolved
            if item.text
        ]
        if result.status != "completed":
            unresolved.append(f"turn ended with status {result.status}")
        evidence = self._episode_evidence(context, controller.episode)
        gate = controller.apply_turn(
            evidence=evidence,
            unresolved=unresolved,
            # Existing Planner steps are advisory and have no authoritative
            # completion mutation API, so they are not marked required here.
            plan_steps=[{**step.to_dict(), "required": False} for step in context.task_state.remaining],
            current_objective=context.task_state.objective,
        )
        event_name = "episode_succeeded" if gate.passed else f"episode_{controller.episode.status}"
        self._episode_store(session.session_id).append(
            event_name,
            controller.episode,
            gate_reasons=list(gate.reasons),
        )
        context.event_log.append(
            event_name,
            {
                "episode_id": controller.episode.episode_id,
                "outcome_version": controller.episode.outcome_version,
                "gate_reasons": list(gate.reasons),
                "evidence_refs": list(gate.evidence_refs),
            },
        )
        result.structured_report = {
            "episode_id": controller.episode.episode_id,
            "episode_status": controller.episode.status,
            "outcome_version": controller.episode.outcome_version,
            "completion_gate_passed": gate.passed,
            "gate_reasons": list(gate.reasons),
        }
        self.trace.emit("episode_outcome", **result.structured_report)
        if self.config.skill_evolution_enabled and self.skill_bank is not None:
            from ..skills.candidates import PendingCandidateStore
            from ..skills.hook import SkillLifecycleHook
            from ..skills.maintainer import SkillMaintainer

            candidate_store = PendingCandidateStore(self.skill_bank.project_dir)
            lifecycle = SkillLifecycleHook(
                candidate_store,
                maintainer=SkillMaintainer(self.skill_bank, candidate_store),
            )
            self._consume_verified_events(lifecycle, controller.episode, evidence)
            if gate.passed:
                lifecycle.on_episode_succeeded(controller.episode)
        context.update_runtime_cursor(
            step=result.steps_used,
            tool_calls_used=result.tool_calls_used,
            status=result.status,
        )
        if self.planner is not None and self.planner.current_plan is not None:
            session.plan = list(self.planner.current_plan.steps)

        metrics = context.session.metrics
        metrics["total_steps"] = int(metrics.get("total_steps", 0)) + result.steps_used
        metrics["total_tool_calls"] = int(metrics.get("total_tool_calls", 0)) + result.tool_calls_used
        metrics["token_calibration"] = context.token_counter.calibration.to_dict()
        # Session-level usage aggregation (requests, tokens, cache hit rate) —
        # written to both the trace (session_metrics event) and the checkpointed
        # session metrics dict, so it survives resume and feeds offline replay.
        metrics["usage"] = self.usage_aggregator.stats.to_dict()
        if self.trace is not None:
            self.trace.emit("session_metrics", **metrics)

        checkpoint_kind = "terminal" if result.status in {"completed", "failed", "stopped"} else "periodic"
        checkpoint = self.structured_store.save_context(context, checkpoint_kind=checkpoint_kind)
        if checkpoint_kind == "terminal" and self.config.long_term_memory_enabled and self.memory_store is not None:
            from ..long_term_memory.hook import MemoryLifecycleHook

            hook = MemoryLifecycleHook(self.memory_store)
            hook.on_session_end(context.task_state, context.tool_state)

        self.trace.emit(
            "run_finished",
            status=result.status,
            steps_used=result.steps_used,
            tool_calls_used=result.tool_calls_used,
            checkpoint_seq=checkpoint["manifest"].get("checkpoint_seq"),
            drift=checkpoint["drift"].get("severity"),
        )
        return result

    def _transition_episode(self, session_id: str, status: str) -> None:
        controller = self._episode_controllers.get(session_id)
        if controller is None:
            return
        if status == "cancelled":
            controller.cancel()
        elif status == "superseded":
            controller.supersede()
        elif status == "dormant":
            controller.dormant()
        else:
            raise ValueError(f"invalid episode transition: {status}")
        self._episode_store(session_id).append(f"episode_{status}", controller.episode)
        context = self._contexts.get(session_id)
        if context is not None:
            context.event_log.append(
                f"episode_{status}",
                {
                    "episode_id": controller.episode.episode_id,
                    "outcome_version": controller.episode.outcome_version,
                },
            )

    def cancel_episode(self, session: StructuredSession) -> None:
        self._transition_episode(session.session_id, "cancelled")

    def supersede_episode(self, session: StructuredSession) -> None:
        self._transition_episode(session.session_id, "superseded")

    def dormancy_episode(self, session: StructuredSession) -> None:
        self._transition_episode(session.session_id, "dormant")

    def _fold_engine(self, context: StructuredContext) -> FoldEngine:
        return FoldEngine(
            self.provider,
            token_counter=context.token_counter,
            trace=self.trace,
            provider_name=self.config.llm.provider,
            model=self.config.llm.model,
            reasoning_effort=self.config.llm.secondary_reasoning_effort,
        )

    @staticmethod
    def _consume_verified_events(lifecycle: Any, episode: TaskEpisode, evidence: list[dict[str, Any]]) -> None:
        failures = [item for item in evidence if item.get("type") == "tool_result" and not item.get("success")]
        verifications = [
            item
            for item in evidence
            if item.get("type") == "tool_result"
            and item.get("name") == "run_shell"
            and item.get("success")
        ]
        if not failures or not verifications:
            return
        failure = failures[-1]
        verification = next(
            (item for item in verifications if int(item.get("seq", 0)) > int(failure.get("seq", 0))),
            None,
        )
        if verification is None:
            return
        fixes = [
            item
            for item in evidence
            if int(failure.get("seq", 0)) < int(item.get("seq", 0)) < int(verification.get("seq", 0))
            and item.get("type") in {"file_change", "tool_result"}
            and item.get("success", True)
        ]
        if fixes:
            lifecycle.on_verified_event(
                episode=episode,
                failure=failure,
                fix=fixes[-1],
                verification=verification,
            )

    def _replan_after_drift(
        self,
        context: StructuredContext,
        drift: DriftReport | None,
        decision: RecoveryDecision | None,
    ) -> None:
        recovery = context.session.runtime_cursor.setdefault("recovery", {})
        attempts = int(recovery.get("replan_attempts", 0)) + 1
        recovery["replan_attempts"] = attempts
        recovery["last_decision"] = decision.to_dict() if decision else None
        context.start_new_epoch(reason="replan")
        if decision is not None:
            context.mark_findings_stale(set(decision.affected_paths))
        context.mark_verification_stale()

        objective = context.task_state.objective or context.session.user_task
        prompt = objective
        if drift is not None and drift.summary:
            prompt = f"{objective}\n\n[Workspace drift]\n{drift.summary}"
        context.add_user(f"[Recovery replan]\n{prompt}")

        if self.planner is not None:
            from ..runtime.planner import Plan

            self.planner.generate(prompt)
            self.planner.current_plan = self.planner.current_plan or Plan([objective])
            context.set_plan(self.planner.current_plan)
            context.session.plan = list(self.planner.current_plan.steps)
        # Accept the observed workspace as the new recovery baseline.
        context.set_workspace_expected(context.workspace_fingerprint.expected_from_actual())
        context.session.status = "running"
        self.structured_store.save_context(context, checkpoint_kind="recovery_transition")

    @staticmethod
    def _impact_paths(context: StructuredContext) -> set[str]:
        paths: set[str] = set()
        expected = context.workspace_expected()
        paths.update(str(item.get("path", "")) for item in expected.postconditions if item.get("path"))
        for finding in context.task_state.key_findings:
            if finding.status != "valid":
                continue
            for evidence in finding.evidence:
                if isinstance(evidence, dict) and evidence.get("path"):
                    paths.add(str(evidence["path"]))
        for group in context.trajectory.groups:
            for message in group.messages:
                if message.role == "assistant":
                    for call in message.tool_calls:
                        path = (call.arguments or {}).get("path")
                        if path:
                            paths.add(str(path))
        return {path for path in paths if path}

    # ------------------------------------------------------------ subagents

    def _run_subagent(
        self,
        task: str,
        max_steps: int,
        *,
        parent_depth: int,
        allowed_tools: set[str] | None = None,
    ) -> AgentRunResult:
        child_depth = parent_depth + 1
        captured_paths: list[str] = []

        def capture(event: TraceEvent) -> None:
            data = event.data or {}
            if event.type != "tool_result" or not data.get("success"):
                return
            if data.get("name") not in {"write_file", "edit_file"}:
                return
            metadata = data.get("metadata") or {}
            path = metadata.get("path")
            if path:
                captured_paths.append(str(path))

        self.trace.listeners.append(capture)
        try:
            result = super()._run_subagent(
                task,
                max_steps,
                parent_depth=parent_depth,
                allowed_tools=allowed_tools,
            )
        finally:
            if capture in self.trace.listeners:
                self.trace.listeners.remove(capture)

        parent_context = self._contexts.get(self._active_session_id or "")
        for path in dict.fromkeys(captured_paths):
            if parent_context is not None:
                parent_context.apply_external_file_change(
                    path=path,
                    operation="subagent",
                    depth=child_depth,
                )
        report = parse_subagent_report(result.text)
        result.structured_report = report.to_dict()
        return result

    # ------------------------------------------------------------ migration

    def migrate_legacy_session(self, session_id: str, *, legacy_dir: str | Path = ".sessions") -> StructuredSession:
        context = migrate_legacy_session(
            legacy_dir,
            session_id,
            self.structured_store,
            workspace_root=self.workspace,
            system_prompt=self.config.system_prompt,
            prefix_hash=self._prefix_hash,
            tools_hash=self._tools_hash,
        )
        self._wire_context(context, session_id)
        return context.session
