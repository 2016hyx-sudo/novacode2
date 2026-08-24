"""Information-isolated assertions and bounded local script checks."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidates import SkillCandidate


@dataclass
class VerificationResult:
    passed: bool
    assertions: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "assertions": list(self.assertions),
            "diagnostics": list(self.diagnostics),
            "runs": list(self.runs),
        }


class SurrogateVerifier:
    """Verifier receives public skill data only, never the main-agent trajectory."""

    def __init__(
        self,
        assertion_generator: Callable[[dict[str, Any]], list[str]] | None = None,
        refiner: Callable[[Path, list[str]], None] | None = None,
        *,
        timeout: float = 10.0,
    ) -> None:
        self.assertion_generator = assertion_generator
        self.refiner = refiner
        self.timeout = max(1.0, timeout)

    def verify(self, candidate: SkillCandidate, package_dir: Path | None = None) -> VerificationResult:
        public = {
            "name": candidate.name,
            "title": candidate.title,
            "granularity": candidate.granularity,
            "when_to_apply": candidate.when_to_apply,
            "rules": [rule.rule for rule in candidate.workflow_rules],
        }
        assertions = (
            self.assertion_generator(public)
            if self.assertion_generator is not None
            else ["candidate has grounded rules", "all rules cite evidence"]
        )
        diagnostics: list[str] = []
        if not candidate.workflow_rules:
            diagnostics.append("candidate has no workflow rules")
        if any(not rule.evidence_refs for rule in candidate.workflow_rules):
            diagnostics.append("one or more workflow rules lack evidence refs")
        runs: list[dict[str, Any]] = []
        if package_dir is not None and (Path(package_dir) / "scripts").is_dir():
            runs, script_errors = self._run_scripts(Path(package_dir) / "scripts")
            diagnostics.extend(script_errors)
        return VerificationResult(not diagnostics, assertions, diagnostics, runs)

    def verify_with_refinement(
        self,
        candidate: SkillCandidate,
        package_dir: Path | None = None,
        *,
        max_attempts: int = 2,
    ) -> VerificationResult:
        result = self.verify(candidate, package_dir)
        attempts = 1
        while (
            not result.passed
            and self.refiner is not None
            and package_dir is not None
            and attempts < max(1, max_attempts)
        ):
            # The refiner receives only public diagnostics and the isolated
            # candidate package, never main-agent reasoning or trajectory.
            self.refiner(Path(package_dir), list(result.diagnostics))
            result = self.verify(candidate, package_dir)
            attempts += 1
        return result

    def _run_scripts(self, scripts_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
        runs: list[dict[str, Any]] = []
        errors: list[str] = []
        with tempfile.TemporaryDirectory(prefix="novacode-skill-verify-") as temp:
            target = Path(temp) / "scripts"
            shutil.copytree(scripts_dir, target)
            for path in sorted(item for item in target.rglob("*") if item.is_file()):
                if path.suffix == ".py":
                    command = [os.environ.get("PYTHON", "python3"), str(path)]
                elif path.suffix == ".sh":
                    command = ["sh", str(path)]
                else:
                    continue
                try:
                    completed = subprocess.run(
                        command,
                        cwd=target,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                        env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
                        check=False,
                    )
                    run = {
                        "script": str(path.relative_to(target)),
                        "returncode": completed.returncode,
                        "stdout": completed.stdout[-2000:],
                        "stderr": completed.stderr[-2000:],
                    }
                    runs.append(run)
                    if completed.returncode != 0:
                        errors.append(
                            f"{run['script']} failed with exit {completed.returncode}: {completed.stderr[-500:]}"
                        )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(f"{path.name} could not be verified: {exc}")
        return runs, errors
