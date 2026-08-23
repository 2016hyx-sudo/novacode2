"""Single-instance SWE-bench inference runner."""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from coding_agent import create_harness
from config import AgentConfig, Constraints, LLMConfig

from .environment import SWEBenchDockerEnvironment
from .patch import export_patch

DATASET_ALIASES = {
    "full": "SWE-bench/SWE-bench",
    "lite": "SWE-bench/SWE-bench_Lite",
    "verified": "SWE-bench/SWE-bench_Verified",
    "multilingual": "SWE-bench/SWE-bench_Multilingual",
}

SWE_SYSTEM_PROMPT = """You are NovaCode running a SWE-bench software-engineering task.
Work only in the provided repository workspace. Inspect the issue and code, identify
the root cause, implement a general fix, and run focused verification commands.

Rules:
- Do not access benchmark answers, gold patches, test patches, or evaluator files.
- Do not modify tests merely to make them pass and do not commit changes.
- Keep changes scoped to the issue and avoid generated or environment files.
- The evaluator collects the final answer from the workspace as a Git patch.
- Return a concise summary only after you have implemented and verified the fix."""


@dataclass(frozen=True)
class SingleRunConfig:
    dataset: str = "verified"
    split: str = "test"
    instance_id: str = ""
    output_dir: Path = Path(".eval-results/swe-bench-single")
    work_root: Path | None = None
    keep_workspace: bool = False
    image: str | None = None
    pull_image: bool = True
    docker_command: str = "docker"
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_steps: int = 50
    max_tool_calls: int = 120
    shell_timeout: float = 120.0
    planner: bool = False
    structured_context: bool = False


@dataclass(frozen=True)
class SingleRunResult:
    instance_id: str
    model_name_or_path: str
    model_patch: str
    status: str
    result_text: str
    steps_used: int
    tool_calls_used: int
    elapsed_seconds: float
    base_commit: str
    image: str
    session_id: str
    output_dir: str
    workspace: str | None

    def prediction(self) -> dict[str, str]:
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_instance(dataset: str, split: str, instance_id: str) -> tuple[dict[str, Any], str]:
    """Load exactly one official instance and derive its image via SWE-bench."""

    if not instance_id:
        raise ValueError("--instance is required")
    dataset_name = DATASET_ALIASES.get(dataset, dataset)
    try:
        from swebench.harness.utils import load_swebench_dataset, make_test_spec
    except ImportError as exc:
        raise RuntimeError(
            "SWE-bench dependencies are missing; install with: pip install -e '.[swebench]'"
        ) from exc
    instances = load_swebench_dataset(dataset_name, split, [instance_id])
    matches = [dict(item) for item in instances if item.get("instance_id") == instance_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one instance {instance_id!r} in {dataset_name}/{split}, got {len(matches)}"
        )
    instance = matches[0]
    return instance, make_test_spec(instance).image


def build_task(problem_statement: str) -> str:
    return (
        "Resolve the repository issue below.\n\n"
        "<issue>\n"
        f"{problem_statement.strip()}\n"
        "</issue>"
    )


def run_single(config: SingleRunConfig) -> SingleRunResult:
    instance, official_image = load_instance(config.dataset, config.split, config.instance_id)
    image = config.image or official_image
    base_commit = str(instance["base_commit"])
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.keep_workspace:
        workspace = output_dir / "workspace"
        if workspace.exists() and any(workspace.iterdir()):
            raise ValueError(f"kept workspace already exists and is non-empty: {workspace}")
        workspace.mkdir(parents=True, exist_ok=True)
        temporary = None
    else:
        root = config.work_root.expanduser().resolve() if config.work_root else None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="novacode-swe-", dir=root)
        workspace = Path(temporary.name) / "workspace"

    started = time.monotonic()
    environment = SWEBenchDockerEnvironment(
        image=image,
        workspace=workspace,
        base_commit=base_commit,
        docker_command=config.docker_command,
        pull_image=config.pull_image,
    )
    try:
        with environment as shell_runner:
            llm = LLMConfig.from_env()
            if config.provider:
                llm.provider = config.provider  # type: ignore[assignment]
            if config.model:
                llm.model = config.model
            if config.api_key:
                llm.api_key = config.api_key
            if config.base_url:
                llm.base_url = config.base_url
            constraints = Constraints(
                max_steps=config.max_steps,
                max_tool_calls=config.max_tool_calls,
                shell_timeout=config.shell_timeout,
                tool_timeout=max(config.shell_timeout + 30.0, 120.0),
            )
            agent_config = AgentConfig(
                llm=llm,
                workspace=workspace,
                planner_enabled=config.planner,
                constraints=constraints,
                system_prompt=SWE_SYSTEM_PROMPT,
                session_dir=output_dir / "sessions",
                trace_dir=output_dir / "traces",
                session_dir_explicit=True,
                trace_dir_explicit=True,
                structured_context_enabled=config.structured_context,
                agent_dir=output_dir / "agent",
            )
            harness = create_harness(agent_config, shell_runner=shell_runner)
            task = build_task(str(instance["problem_statement"]))
            session = harness.new_session(task)
            agent_result = harness.run_task(session)
            patch = export_patch(workspace, base_commit, output_dir / "patch.diff")

        session_id = str(getattr(session, "id", None) or getattr(session, "session_id", ""))
        result = SingleRunResult(
            instance_id=config.instance_id,
            model_name_or_path=f"novacode/{llm.provider}/{llm.model}",
            model_patch=patch,
            status=agent_result.status,
            result_text=agent_result.text,
            steps_used=agent_result.steps_used,
            tool_calls_used=agent_result.tool_calls_used,
            elapsed_seconds=round(time.monotonic() - started, 3),
            base_commit=base_commit,
            image=image,
            session_id=session_id,
            output_dir=str(output_dir),
            workspace=str(workspace) if config.keep_workspace else None,
        )
        _write_outputs(output_dir, result)
        return result
    finally:
        environment.cleanup()
        if temporary is not None:
            temporary.cleanup()


def _write_outputs(output_dir: Path, result: SingleRunResult) -> None:
    (output_dir / "result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "predictions.jsonl").write_text(
        json.dumps(result.prediction(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
