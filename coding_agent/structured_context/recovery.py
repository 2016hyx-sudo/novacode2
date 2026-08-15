"""Drift severity classification and RESUME / REPLAN / BLOCKED decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import DriftReport


@dataclass
class RecoveryPolicy:
    max_replan_drift_files: int = 20
    autonomous_replan_enabled: bool = True
    git_head_change_policy: str = "blocked"  # blocked | ask | replan
    max_replan_attempts: int = 2
    require_user_confirmation_for_structural: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RecoveryPolicy:
        data = data or {}
        return cls(
            max_replan_drift_files=int(data.get("max_replan_drift_files", 20)),
            autonomous_replan_enabled=bool(data.get("autonomous_replan_enabled", True)),
            git_head_change_policy=str(data.get("git_head_change_policy", "blocked")),
            max_replan_attempts=int(data.get("max_replan_attempts", 2)),
            require_user_confirmation_for_structural=bool(
                data.get("require_user_confirmation_for_structural", True)
            ),
        )


@dataclass
class RecoveryDecision:
    action: str  # RESUME | REPLAN | BLOCKED
    reason: str
    drift: DriftReport | None = None
    affected_paths: list[str] = field(default_factory=list)
    user_choices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "drift": self.drift.to_dict() if self.drift else None,
            "affected_paths": list(self.affected_paths),
            "user_choices": list(self.user_choices),
        }


class RecoveryEngine:
    def __init__(self, policy: RecoveryPolicy | None = None) -> None:
        self.policy = policy or RecoveryPolicy()

    def classify(self, report: DriftReport, impact_paths: list[str]) -> str:
        if report.severity == "NONE":
            return "NONE"
        if report.git_divergence:
            return "STRUCTURAL"
        if report.severity == "STRUCTURAL":
            return "STRUCTURAL"

        impact_set = {path.replace("\\", "/") for path in impact_paths}
        affected = [
            item.get("path", "")
            for group in (report.hash_mismatches, report.missing_expected_changes, report.unexpected_changes)
            for item in group
        ]
        affected = [path for path in affected if path in impact_set]
        if affected:
            return "HIGH"
        if report.hash_mismatches or report.missing_expected_changes:
            return "LOW"
        if report.unexpected_changes:
            return "LOW"
        return "IGNORED"

    def decide(
        self,
        *,
        checkpoint_valid: bool,
        log_valid: bool,
        drift: DriftReport,
        severity: str,
        replan_attempts: int,
        affected_paths: list[str],
    ) -> RecoveryDecision:
        if not checkpoint_valid and not log_valid:
            return RecoveryDecision(
                "BLOCKED",
                "checkpoint and event log are both unavailable",
                drift,
                affected_paths,
                ["recreate session", "inspect logs manually"],
            )
        if not checkpoint_valid:
            return RecoveryDecision(
                "BLOCKED",
                "checkpoint invalid and event log rebuild is not yet available",
                drift,
                affected_paths,
                ["rebuild from event log", "start new session"],
            )
        if severity == "STRUCTURAL":
            return RecoveryDecision(
                "BLOCKED",
                f"structural drift: {drift.summary}",
                drift,
                affected_paths,
                ["accept current workspace and replan", "restore checkpoint workspace", "switch git branch/head"],
            )
        if severity == "HIGH":
            if not self.policy.autonomous_replan_enabled:
                return RecoveryDecision(
                    "BLOCKED",
                    "autonomous replan is disabled",
                    drift,
                    affected_paths,
                    ["enable replan", "manual recovery"],
                )
            drift_file_count = (
                len(drift.hash_mismatches) + len(drift.missing_expected_changes) + len(drift.unexpected_changes)
            )
            if drift_file_count > self.policy.max_replan_drift_files:
                return RecoveryDecision(
                    "BLOCKED",
                    f"drift file count {drift_file_count} exceeds policy limit",
                    drift,
                    affected_paths,
                    ["accept and replan", "restore checkpoint"],
                )
            if replan_attempts >= self.policy.max_replan_attempts:
                return RecoveryDecision(
                    "BLOCKED",
                    f"replan attempts exhausted ({replan_attempts})",
                    drift,
                    affected_paths,
                    ["manual recovery", "start new session"],
                )
            return RecoveryDecision("REPLAN", f"high-impact drift: {drift.summary}", drift, affected_paths)
        if severity in {"LOW", "IGNORED"}:
            return RecoveryDecision(
                "RESUME",
                f"low-impact drift accepted: {drift.summary}",
                drift,
                affected_paths,
            )
        return RecoveryDecision("RESUME", "no drift detected", drift, affected_paths)
