"""NovaCode composition root.

This is the only place that knows every concrete component and wires them
together. AgentLoop and tools only depend on protocols.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from config import AgentConfig

from .agent import AgentLoop, AgentRunResult
from .context.manager import ContextManager
from .context.session import Session, SessionStore
from .llm import create_provider
from .llm.base import LLMProvider
from .runtime.constraints import RunBudget
from .runtime.planner import Plan, Planner
from .runtime.trace import TraceEvent, TraceWriter
from .runtime.validator import Validator
from .tools import SubagentTool, ToolRegistry, build_tool_registry
from .tools.executor import ToolExecutor

SUBAGENT_SYSTEM_PROMPT = """You are a NovaCode subagent. Complete the single task given by the
main coding agent and return one concise final report with your findings or changes.

Rules:
- Work only inside the workspace.
- Prefer small tool calls and verify any change you make.
- Do not ask questions; make reasonable assumptions and report them.
- Your final message is returned to the main agent, so make it self-contained."""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Harness:
    """Assembled runtime: provider + tools + budgets + session/trace plumbing."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        provider: LLMProvider,
        trace: TraceWriter,
        session_store: SessionStore,
        planner: Planner | None,
        validator: Validator,
    ) -> None:
        self.config = config
        self.provider = provider
        self.trace = trace
        self.session_store = session_store
        self.planner = planner
        self.validator = validator
        self.workspace = Path(config.workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.budget = RunBudget(
            max_tool_calls=config.constraints.max_tool_calls,
            max_subagents=config.constraints.max_subagents,
        )
        self.tools = self._build_tools(depth=0)
        self.executor = ToolExecutor(
            self.tools,
            max_retries=config.constraints.tool_max_retries,
            tool_timeout=config.constraints.tool_timeout,
            max_output_chars=config.constraints.max_output_chars,
            trace=trace,
            budget=self.budget,
        )

    # ------------------------------------------------------------------ wiring

    def create_agent(
        self,
        context: ContextManager,
        *,
        tools: ToolRegistry | None = None,
        max_steps: int | None = None,
        agent_name: str = "main",
    ) -> AgentLoop:
        selected_tools = self.tools if tools is None else tools
        executor = self.executor if selected_tools is self.tools else self._make_executor(selected_tools)
        is_main = agent_name == "main"
        return AgentLoop(
            llm=self.provider,
            tools=selected_tools,
            executor=executor,
            context=context,
            max_steps=self.config.constraints.max_steps if max_steps is None else max_steps,
            max_correction_attempts=self.config.constraints.max_correction_attempts,
            max_llm_retries=self.config.constraints.max_llm_retries,
            planner=self.planner if is_main else None,
            validator=self.validator if is_main else None,
            trace=self.trace,
            agent_name=agent_name,
        )

    def _make_executor(self, tools: ToolRegistry) -> ToolExecutor:
        return ToolExecutor(
            tools,
            max_retries=self.config.constraints.tool_max_retries,
            tool_timeout=self.config.constraints.tool_timeout,
            max_output_chars=self.config.constraints.max_output_chars,
            trace=self.trace,
            budget=self.budget,
        )

    def _build_tools(self, *, depth: int) -> ToolRegistry:
        can_nest = depth < self.config.constraints.max_subagent_depth
        subagent_tool = self._make_subagent_tool(depth) if can_nest else None
        return build_tool_registry(
            self.workspace,
            self.config.constraints,
            subagent_tool=subagent_tool,
        )

    def _make_subagent_tool(self, depth: int) -> SubagentTool:
        return SubagentTool(
            run_subagent=lambda task, steps: self._run_subagent(task, steps, parent_depth=depth),
            budget=self.budget,
            trace=self.trace,
            default_max_steps=min(5, self.config.constraints.max_steps),
            max_steps_limit=self.config.constraints.max_steps,
            current_depth=depth,
        )

    def _run_subagent(self, task: str, max_steps: int, *, parent_depth: int) -> AgentRunResult:
        child_depth = parent_depth + 1
        tools = self._build_tools(depth=child_depth)
        context = ContextManager(
            SUBAGENT_SYSTEM_PROMPT,
            max_context_tokens=self.config.max_context_tokens,
        )
        agent = self.create_agent(
            context,
            tools=tools,
            max_steps=max_steps,
            agent_name=f"subagent-{child_depth}",
        )
        return agent.run(task)

    # ------------------------------------------------------------ session flow

    def new_session(self, user_task: str) -> Session:
        session = Session.new(
            user_task=user_task,
            provider=self.config.llm.provider,
            model=self.config.llm.model,
        )
        self.trace.bind(session.id)
        self.trace.emit(
            "session_start",
            session_id=session.id,
            provider=session.provider,
            model=session.model,
            task=user_task,
            workspace=str(self.workspace),
        )
        return session

    def load_session(self, session_id: str) -> Session:
        session = self.session_store.load(session_id)
        self.trace.bind(session.id)
        self.trace.emit(
            "session_resume",
            session_id=session.id,
            task=session.user_task,
            message_count=len(session.messages),
        )
        return session

    def run_task(
        self,
        session: Session,
        task: str | None = None,
        *,
        append_user: bool = True,
    ) -> AgentRunResult:
        self.trace.bind(session.id)
        session.provider = session.provider or self.config.llm.provider
        session.model = session.model or self.config.llm.model

        context = ContextManager(
            self.config.system_prompt,
            messages=list(session.messages),
            max_context_tokens=self.config.max_context_tokens,
        )

        effective_task = task or (session.user_task if not session.messages else None)
        if append_user and effective_task:
            context.add_user(effective_task)
            session.user_task = effective_task

        if self.planner is not None:
            if session.plan:
                self.planner.current_plan = Plan(list(session.plan))
            else:
                self.planner.generate(effective_task or session.user_task)
                session.plan = list(self.planner.current_plan.steps) if self.planner.current_plan else None
            if self.planner.current_plan is not None:
                context.add_plan(self.planner.current_plan)

        agent = self.create_agent(context, agent_name="main")
        result = agent.run()

        session.messages = list(context.messages)
        session.updated_at = _utcnow()
        session.status = result.status
        if self.planner is not None and self.planner.current_plan is not None:
            session.plan = list(self.planner.current_plan.steps)
        session.metadata.setdefault("last_steps_used", result.steps_used)
        session.metadata["last_tool_calls_used"] = result.tool_calls_used
        path = self.session_store.save(session)

        self.trace.emit(
            "run_finished",
            status=result.status,
            steps_used=result.steps_used,
            tool_calls_used=result.tool_calls_used,
            session_file=str(path),
        )
        return result


def create_harness(
    config: AgentConfig,
    *,
    listeners: list[Callable[[TraceEvent], None]] | None = None,
) -> Harness:
    provider = create_provider(config.llm)
    trace = TraceWriter(config.trace_dir, listeners=listeners)
    session_store = SessionStore(config.session_dir)
    validator = Validator(
        require_verification_after_edit=config.constraints.require_verification_after_edit
    )
    planner = Planner(provider, trace=trace) if config.planner_enabled else None
    return Harness(
        config,
        provider=provider,
        trace=trace,
        session_store=session_store,
        planner=planner,
        validator=validator,
    )


__all__ = [
    "SUBAGENT_SYSTEM_PROMPT",
    "AgentLoop",
    "AgentRunResult",
    "Harness",
    "create_harness",
]
