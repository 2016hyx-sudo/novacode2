"""The core AgentLoop.

The loop only orchestrates: ask the LLM, execute requested tools, feed results
back, validate completion, and retry/correct within configured budgets.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from .context.manager import ContextManager
from .llm.base import LLMError, LLMProvider, LLMResponse, ToolCall
from .runtime.planner import Planner
from .runtime.trace import TraceWriter
from .runtime.validator import ToolUseRecord, ValidationResult, Validator
from .tools.executor import ToolExecutor
from .tools.registry import ToolRegistry


@dataclass
class AgentRunResult:
    text: str
    status: Literal["completed", "stopped", "failed"] = "completed"
    steps_used: int = 0
    tool_calls_used: int = 0


class AgentLoop:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        tools: ToolRegistry,
        executor: ToolExecutor,
        context: ContextManager,
        max_steps: int = 30,
        max_correction_attempts: int = 2,
        max_llm_retries: int = 1,
        planner: Planner | None = None,
        validator: Validator | None = None,
        trace: TraceWriter | None = None,
        agent_name: str = "main",
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.executor = executor
        self.context = context
        self.max_steps = max(1, max_steps)
        self.max_correction_attempts = max(0, max_correction_attempts)
        self.max_llm_retries = max(0, max_llm_retries)
        self.planner = planner
        self.validator = validator
        self.trace = trace
        self.agent_name = agent_name

    def run(self, task: str | None = None) -> AgentRunResult:
        if task is not None:
            self.context.add_user(task)

        history: list[ToolUseRecord] = []
        steps_used = 0
        tool_calls_used = 0
        correction_attempts = 0
        consecutive_failures = 0

        try:
            while steps_used < self.max_steps:
                step = steps_used + 1
                self._emit("step_start", step=step)

                try:
                    response = self._chat_with_retry()
                except LLMError as exc:
                    self._emit("error", error=str(exc), retryable=exc.retryable, step=step)
                    return AgentRunResult(
                        text=f"LLM request failed after retries: {exc}",
                        status="failed",
                        steps_used=steps_used,
                        tool_calls_used=tool_calls_used,
                    )

                steps_used += 1
                self.context.add_assistant(response.text, response.tool_calls)
                self._emit(
                    "llm_response",
                    step=step,
                    text=response.text,
                    tool_calls=[call.name for call in response.tool_calls],
                    stop_reason=response.stop_reason,
                    usage=response.usage,
                )
                record_usage = getattr(self.context, "record_usage", None)
                if record_usage is not None:
                    record_usage(response.usage)

                if not response.tool_calls:
                    verdict = self._validate_final(response.text, history)
                    if not verdict.passed and correction_attempts < self.max_correction_attempts:
                        correction_attempts += 1
                        feedback = verdict.feedback or "Validation failed."
                        self.context.add_user(
                            "[Harness validation feedback]\n"
                            f"{feedback}\n"
                            "Do not give a final answer yet. Perform the missing verification or "
                            "correction, then continue."
                        )
                        self._emit(
                            "validation_failed",
                            step=step,
                            feedback=feedback,
                            correction_attempt=correction_attempts,
                        )
                        if self.planner is not None:
                            plan = self.planner.add_correction(feedback)
                            self.context.set_plan(plan)
                        self._emit("step_end", step=step)
                        continue

                    if not verdict.passed:
                        self._emit(
                            "validation_failed",
                            step=step,
                            feedback=verdict.feedback,
                            final=True,
                            correction_attempts_exhausted=True,
                        )
                        warning = (
                            f"\n\n[Harness warning] Final validation is still failing: "
                            f"{verdict.feedback or 'unknown validation issue'}"
                        )
                        self._emit("step_end", step=step)
                        return AgentRunResult(
                            text=(response.text or "") + warning,
                            status="stopped",
                            steps_used=steps_used,
                            tool_calls_used=tool_calls_used,
                        )
                    self._emit(
                        "validation_passed",
                        step=step,
                        checks=verdict.checks,
                    )
                    self._emit("step_end", step=step)
                    return AgentRunResult(
                        text=response.text or "",
                        status="completed",
                        steps_used=steps_used,
                        tool_calls_used=tool_calls_used,
                    )

                # Execute every requested tool call and feed structured results back.
                calls = self._ensure_call_ids(response.tool_calls, step)
                on_batch_start = getattr(self.context, "on_tool_batch_start", None)
                if on_batch_start is not None:
                    on_batch_start(calls, step)
                results = self.executor.execute_all(calls)
                tool_calls_used += len(results)
                for call, call_result in zip(calls, results):
                    result = call_result.result
                    self.context.add_tool_result(call, result)
                    history.append(
                        ToolUseRecord(
                            step=step,
                            name=call.name,
                            success=result.success,
                            error=result.error,
                            output_preview=result.output[:300],
                        )
                    )
                    if result.success:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                if consecutive_failures >= 2 and self.planner is not None:
                    summary = (
                        f"连续 {consecutive_failures} 次工具调用失败；先诊断错误原因，"
                        "改用更可靠的方式，并在修改后运行验证。"
                    )
                    plan = self.planner.add_correction(summary)
                    self.context.set_plan(plan)
                    consecutive_failures = 0

                on_batch_complete = getattr(self.context, "on_tool_batch_complete", None)
                if on_batch_complete is not None:
                    on_batch_complete(step)

                self._emit("step_end", step=step)

            # max_steps exhausted
            message = (
                f"Stopped after reaching max_steps ({self.max_steps}). "
                "The task may be incomplete; review the trace and continue the session if needed."
            )
            self._emit("error", error=message, step=steps_used)
            return AgentRunResult(
                text=message,
                status="stopped",
                steps_used=steps_used,
                tool_calls_used=tool_calls_used,
            )
        except Exception as exc:
            self._emit("error", error=f"{type(exc).__name__}: {exc}", step=steps_used + 1)
            return AgentRunResult(
                text=f"Agent run failed with an unexpected error: {exc}",
                status="failed",
                steps_used=steps_used,
                tool_calls_used=tool_calls_used,
            )

    def _chat_with_retry(self) -> LLMResponse:
        attempts = self.max_llm_retries + 1
        for attempt in range(attempts):
            self._emit(
                "llm_request",
                attempt=attempt + 1,
                message_count=len(self.context.messages),
                tools=self.tools.names(),
            )
            try:
                return self.llm.chat(self.context.messages, self.tools.schemas())
            except LLMError as exc:
                if not exc.retryable or attempt + 1 >= attempts:
                    raise
                delay = min(2.0 * (attempt + 1), 8.0)
                self._emit("llm_retry", attempt=attempt + 1, error=str(exc), delay=delay)
                time.sleep(delay)
        raise LLMError("unreachable")

    def _validate_final(
        self,
        text: str | None,
        history: list[ToolUseRecord],
    ) -> ValidationResult:
        if self.validator is None:
            return ValidationResult.ok("validator disabled")
        return self.validator.validate_completion(text, history)

    @staticmethod
    def _ensure_call_ids(calls: list[ToolCall], step: int) -> list[ToolCall]:
        for index, call in enumerate(calls):
            if not call.id:
                call.id = f"call_{step}_{index + 1}"
        return calls

    def _emit(self, event_type: str, **data: object) -> None:
        if self.trace is not None:
            self.trace.emit(event_type, agent=self.agent_name, **data)
