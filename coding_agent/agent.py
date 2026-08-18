"""The core AgentLoop.

The loop only orchestrates: ask the LLM, execute requested tools, feed results
back, validate completion, and retry/correct within configured budgets.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Literal

from .context.manager import ContextManager
from .llm.base import LLMError, LLMProvider, LLMResponse, ToolCall
from .llm.usage import (
    MEASUREMENT_SCHEMA_VERSION,
    new_request_id,
    normalize_usage,
    request_payload_hash,
)
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
    # Optional structured report populated by structured-context subagents.
    structured_report: dict | None = None


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
                    response = self._chat_with_retry(step=step)
                except LLMError as exc:
                    self._emit("error", error=str(exc), retryable=exc.retryable, step=step)
                    return AgentRunResult(
                        text=f"LLM request failed after retries: {exc}",
                        status="failed",
                        steps_used=steps_used,
                        tool_calls_used=tool_calls_used,
                    )

                steps_used += 1
                self.context.add_assistant(
                    response.text, response.tool_calls, raw_content=response.raw_content
                )
                self._emit(
                    "llm_response",
                    step=step,
                    text=response.text,
                    thinking=response.thinking,
                    tool_calls=[call.name for call in response.tool_calls],
                    stop_reason=response.stop_reason,
                    usage=response.usage,
                    normalized_usage=response.normalized_usage,
                    request_id=response.request_id,
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

    def _chat_with_retry(self, *, step: int) -> LLMResponse:
        # Allocate the first main request ID before prompt preparation so any
        # Fold calls triggered during preparation can point back to it.
        request_group_id = new_request_id()
        snapshot_builder = getattr(self.context, "snapshot_for_request", None)
        snapshot_metadata: dict[str, object] = {}
        if callable(snapshot_builder):
            snapshot = snapshot_builder(request_id=request_group_id, step=step)
            raw_messages = snapshot.get("messages", ())
            snapshot_metadata = dict(snapshot.get("metadata") or {})
        else:
            # Legacy contexts have no folding preparation step.
            prepare_for_chat = getattr(self.context, "prepare_for_chat", None)
            if callable(prepare_for_chat):
                prepare_for_chat()
            raw_messages = self.context.messages

        # The provider receives this one deep-copied snapshot on every retry;
        # later context mutation or property access cannot change the payload.
        messages = tuple(copy.deepcopy(list(raw_messages)))
        tool_schemas = tuple(copy.deepcopy(self.tools.schemas()))
        payload_hash = request_payload_hash(messages, tool_schemas)
        tool_names = [tool.name for tool in tool_schemas]
        provider_config = getattr(self.llm, "config", None)
        provider_name = str(getattr(provider_config, "provider", "") or "")
        model = str(getattr(provider_config, "model", "") or "")
        attempts = self.max_llm_retries + 1
        for attempt in range(attempts):
            request_id = request_group_id if attempt == 0 else new_request_id()
            prepared = {
                "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
                "request_id": request_id,
                "request_group_id": request_group_id,
                "parent_request_id": None,
                "agent_role": self._agent_role(),
                "step": step,
                "attempt": attempt + 1,
                "message_count": len(messages),
                "tool_count": len(tool_schemas),
                "tools": tool_names,
                "payload_hash": payload_hash,
                "provider": provider_name,
                "model": model,
                "event_seq_anchor": None,
                "epoch_id": None,
                "layers_estimated": None,
                **snapshot_metadata,
            }
            self._emit("llm_request_prepared", **prepared)
            self._emit(
                "llm_request",
                request_id=request_id,
                request_group_id=request_group_id,
                agent_role=self._agent_role(),
                step=step,
                attempt=attempt + 1,
                message_count=len(messages),
                tools=tool_names,
                payload_hash=payload_hash,
            )
            attempt_messages = tuple(copy.deepcopy(list(messages)))
            attempt_tools = tuple(copy.deepcopy(list(tool_schemas)))
            # Subagents and other secondary roles run at a cheaper effort; the
            # main agent uses the configured reasoning effort (None = provider
            # default). getattr keeps this duck-typed for any provider.
            llm_config = getattr(self.llm, "config", None)
            if self._agent_role() == "subagent":
                effort = getattr(llm_config, "secondary_reasoning_effort", None) or "none"
            else:
                effort = getattr(llm_config, "reasoning_effort", None)
            started = time.monotonic()
            try:
                response = self.llm.chat(
                    attempt_messages, attempt_tools, reasoning_effort=effort
                )
            except LLMError as exc:
                self._emit(
                    "llm_request_finished",
                    measurement_schema_version=MEASUREMENT_SCHEMA_VERSION,
                    request_id=request_id,
                    request_group_id=request_group_id,
                    parent_request_id=None,
                    agent_role=self._agent_role(),
                    step=step,
                    attempt=attempt + 1,
                    status="provider_error",
                    latency_ms=int((time.monotonic() - started) * 1000),
                    raw_usage={},
                    normalized_usage=normalize_usage(None),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    retryable=exc.retryable,
                )
                if not exc.retryable or attempt + 1 >= attempts:
                    raise
                delay = min(2.0 * (attempt + 1), 8.0)
                self._emit(
                    "llm_retry",
                    request_id=request_id,
                    request_group_id=request_group_id,
                    step=step,
                    attempt=attempt + 1,
                    error=str(exc),
                    delay=delay,
                )
                time.sleep(delay)
                continue
            except Exception as exc:
                self._emit(
                    "llm_request_finished",
                    measurement_schema_version=MEASUREMENT_SCHEMA_VERSION,
                    request_id=request_id,
                    request_group_id=request_group_id,
                    parent_request_id=None,
                    agent_role=self._agent_role(),
                    step=step,
                    attempt=attempt + 1,
                    status="unexpected_error",
                    latency_ms=int((time.monotonic() - started) * 1000),
                    raw_usage={},
                    normalized_usage=normalize_usage(None),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    retryable=False,
                )
                raise

            normalized = normalize_usage(response.usage)
            response.request_id = request_id
            response.normalized_usage = normalized
            self._emit(
                "llm_request_finished",
                measurement_schema_version=MEASUREMENT_SCHEMA_VERSION,
                request_id=request_id,
                request_group_id=request_group_id,
                parent_request_id=None,
                agent_role=self._agent_role(),
                step=step,
                attempt=attempt + 1,
                status="success",
                latency_ms=int((time.monotonic() - started) * 1000),
                raw_usage=dict(response.usage or {}),
                normalized_usage=normalized,
                stop_reason=response.stop_reason,
                error_type=None,
                error=None,
                retryable=False,
            )
            return response
        raise LLMError("unreachable")

    def _agent_role(self) -> str:
        if self.agent_name == "main":
            return "main"
        if self.agent_name.startswith("subagent"):
            return "subagent"
        return self.agent_name

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
