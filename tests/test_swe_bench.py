from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from coding_agent.tools.shell import ShellExecutionResult, build_shell_tool
from evals.swe_bench.environment import DockerCommandRunner
from evals.swe_bench.patch import export_patch
from evals.swe_bench.runner import build_task


class RecordingRunner:
    def __init__(self, result: ShellExecutionResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    def run(self, command: str, *, workspace: Path, timeout: float) -> ShellExecutionResult:
        self.calls.append({"command": command, "workspace": workspace, "timeout": timeout})
        return self.result


def test_shell_tool_uses_injected_runner(tmp_path: Path) -> None:
    runner = RecordingRunner(
        ShellExecutionResult(returncode=0, stdout="container output\n", cwd="/testbed")
    )
    tool = build_shell_tool(tmp_path, shell_timeout=20, runner=runner)

    result = tool.execute(command="pytest -q", timeout=7)

    assert result.success is True
    assert result.output == "container output\n"
    assert result.metadata["cwd"] == "/testbed"
    assert runner.calls == [
        {"command": "pytest -q", "workspace": tmp_path.resolve(), "timeout": 7.0}
    ]


def test_docker_runner_builds_exec_without_forwarding_host_environment(
    tmp_path: Path, monkeypatch
) -> None:
    captured: list[tuple[list[str], dict]] = []

    def fake_run(args, **kwargs):
        captured.append((list(args), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("evals.swe_bench.environment.subprocess.run", fake_run)
    runner = DockerCommandRunner(container_id="container-123")

    result = runner.run("python -m pytest -q", workspace=tmp_path, timeout=33)

    assert result == ShellExecutionResult(
        returncode=0, stdout="ok\n", stderr="", cwd="/testbed"
    )
    args, kwargs = captured[0]
    assert args[:4] == ["docker", "exec", "-w", "/testbed"]
    assert args[-4:] == ["container-123", "bash", "-lc", "python -m pytest -q"]
    assert kwargs["timeout"] == 33
    assert not any(item.startswith("OPENAI_API_KEY=") for item in args)
    assert not any(item.startswith("ANTHROPIC_API_KEY=") for item in args)
    assert "PYTHONDONTWRITEBYTECODE=1" in args


def test_environment_initializes_git_safe_directory(tmp_path: Path, monkeypatch) -> None:
    from evals.swe_bench.environment import SWEBenchDockerEnvironment

    calls: list[tuple[str, ...]] = []

    def fake_docker(self, *args, **kwargs):
        calls.append(tuple(args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(SWEBenchDockerEnvironment, "_docker", fake_docker)
    environment = SWEBenchDockerEnvironment(
        image="example/image",
        workspace=tmp_path,
        base_commit="base",
    )
    environment.container_id = "container-123"

    environment._initialize_inference_container()

    assert (
        "exec",
        "container-123",
        "git",
        "config",
        "--global",
        "--add",
        "safe.directory",
        "/testbed",
    ) in calls
    assert (
        "exec",
        "-w",
        "/testbed",
        "container-123",
        "git",
        "reset",
        "--hard",
        "base",
    ) in calls
    assert (
        "exec",
        "-w",
        "/testbed",
        "container-123",
        "git",
        "clean",
        "-fd",
    ) in calls


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_patch_export_includes_tracked_and_untracked_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(
        repo,
        "-c",
        "user.name=NovaCode Test",
        "-c",
        "user.email=novacode@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "new_module.py").write_text("created = True\n", encoding="utf-8")

    patch_path = tmp_path / "prediction.diff"
    patch = export_patch(repo, base, patch_path)

    assert "-value = 1" in patch
    assert "+value = 2" in patch
    assert "new_module.py" in patch
    assert "+created = True" in patch
    assert patch_path.read_text(encoding="utf-8") == patch


def test_task_builder_contains_only_supplied_issue() -> None:
    task = build_task("Public issue text")

    assert "Public issue text" in task
    assert "gold" not in task.lower()
    assert "test_patch" not in task
