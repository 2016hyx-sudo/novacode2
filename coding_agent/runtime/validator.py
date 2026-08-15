"""Simple completion validation and correction feedback."""
from __future__ import annotations

from dataclasses import dataclass, field

MODIFY_TOOLS = {"write_file", "edit_file"}
VERIFY_TOOLS = {"run_shell"}


@dataclass
class ToolUseRecord:
    step: int
    name: str
    success: bool
    error: str | None = None
    output_preview: str = ""


@dataclass
class ValidationResult:
    passed: bool
    checks: list[str] = field(default_factory=list)
    feedback: str | None = None

    @classmethod
    def ok(cls, *checks: str) -> ValidationResult:
        return cls(passed=True, checks=list(checks))

    @classmethod
    def fail(cls, feedback: str, *checks: str) -> ValidationResult:
        return cls(passed=False, checks=list(checks), feedback=feedback)


class Validator:
    """Validates that the model has actually earned its final answer."""

    def __init__(self, *, require_verification_after_edit: bool = True) -> None:
        self.require_verification_after_edit = require_verification_after_edit

    def validate_completion(
        self,
        final_text: str | None,
        history: list[ToolUseRecord],
    ) -> ValidationResult:
        if not final_text or not final_text.strip():
            return ValidationResult.fail(
                "The model returned an empty final answer. Provide a concrete summary of the completed work."
            )

        checks = ["final answer is non-empty"]

        if not self.require_verification_after_edit:
            return ValidationResult.ok(*checks)

        modifications = [i for i, item in enumerate(history) if item.name in MODIFY_TOOLS and item.success]
        if not modifications:
            return ValidationResult.ok(*checks, "no file modification requires verification")

        last_modification = max(modifications)
        later_shell = [
            item for i, item in enumerate(history) if i > last_modification and item.name in VERIFY_TOOLS
        ]
        if not later_shell:
            return ValidationResult.fail(
                "Files were modified but no verification command (run_shell) was executed afterwards. "
                "Run tests or a syntax/build check for the changed files before giving the final answer.",
                *checks,
                "file modification detected",
            )

        last_shell = max(i for i, item in enumerate(history) if item.name in VERIFY_TOOLS)
        last_verification = history[last_shell]
        if not last_verification.success:
            return ValidationResult.fail(
                f"The verification command after file modification failed: {last_verification.error or 'unknown error'}. "
                "Fix the problem and run verification again before giving the final answer.",
                *checks,
                "verification command failed",
            )

        return ValidationResult.ok(
            *checks,
            "file modification detected",
            "successful verification executed after last modification",
        )
