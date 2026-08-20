"""Docker execution environment for one SWE-bench instance.

NovaCode itself stays on the host.  The repository is copied out of the
official instance image and bind-mounted back at ``/testbed`` so filesystem
tools can edit it locally while shell commands use the image's dependencies.
"""
from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from coding_agent.tools.shell import ShellExecutionResult, ShellRunner


class DockerEnvironmentError(RuntimeError):
    """Raised when an inference container cannot be prepared or inspected."""


@dataclass
class DockerCommandRunner(ShellRunner):
    """Execute NovaCode shell calls in an already-running Docker container."""

    container_id: str
    docker_command: str = "docker"
    container_cwd: str = "/testbed"
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "BASH_ENV": "/root/.bashrc",
            "PAGER": "cat",
            "MANPAGER": "cat",
            "PIP_PROGRESS_BAR": "off",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TQDM_DISABLE": "1",
        }
    )

    def run(
        self,
        command: str,
        *,
        workspace: Path,
        timeout: float,
    ) -> ShellExecutionResult:
        del workspace  # The bind mount maps it to container_cwd.
        args = [self.docker_command, "exec", "-w", self.container_cwd]
        for key, value in sorted(self.environment.items()):
            args.extend(["-e", f"{key}={value}"])
        args.extend([self.container_id, "bash", "-lc", command])
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return ShellExecutionResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            cwd=self.container_cwd,
        )


@dataclass
class SWEBenchDockerEnvironment:
    """Lifecycle manager for one official SWE-bench instance image."""

    image: str
    workspace: Path
    base_commit: str
    docker_command: str = "docker"
    pull_image: bool = True
    network: str = "none"
    container_cwd: str = "/testbed"
    container_lifetime: str = "2h"
    container_id: str | None = field(default=None, init=False)
    _seed_container_id: str | None = field(default=None, init=False)

    def prepare(self) -> DockerCommandRunner:
        self.workspace = self.workspace.expanduser().resolve()
        self._check_workspace_target()
        self._ensure_image()
        self._copy_pristine_workspace()
        self._start_inference_container()
        self._initialize_inference_container()
        runner = DockerCommandRunner(
            container_id=self.container_id or "",
            docker_command=self.docker_command,
            container_cwd=self.container_cwd,
        )
        probe = runner.run(
            "git rev-parse HEAD",
            workspace=self.workspace,
            timeout=30,
        )
        if probe.returncode != 0:
            self.cleanup()
            raise DockerEnvironmentError(
                f"instance image does not contain a usable Git repository at "
                f"{self.container_cwd}: {(probe.stdout + probe.stderr).strip()}"
            )
        actual = probe.stdout.strip()
        if self.base_commit and actual != self.base_commit:
            self.cleanup()
            raise DockerEnvironmentError(
                f"failed to initialize instance at base commit: expected "
                f"{self.base_commit}, got {actual}"
            )
        return runner

    def cleanup(self) -> None:
        if self.container_id:
            # Shell commands run as the image's default user (commonly root) and
            # may create caches/build products on the host bind mount. Restore
            # ownership so patch export and temporary-directory cleanup remain
            # reliable on the host.
            try:
                subprocess.run(
                    [
                        self.docker_command,
                        "exec",
                        self.container_id,
                        "chown",
                        "-R",
                        f"{os.getuid()}:{os.getgid()}",
                        self.container_cwd,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                # Container removal must still be attempted when ownership
                # restoration is unavailable.
                pass
        for identifier in (self.container_id, self._seed_container_id):
            if identifier:
                try:
                    subprocess.run(
                        [self.docker_command, "rm", "-f", identifier],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
        self.container_id = None
        self._seed_container_id = None

    def __enter__(self) -> DockerCommandRunner:
        try:
            return self.prepare()
        except Exception:
            self.cleanup()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.cleanup()

    def _check_workspace_target(self) -> None:
        if self.workspace.exists() and any(self.workspace.iterdir()):
            raise DockerEnvironmentError(
                f"refusing to overwrite non-empty workspace: {self.workspace}"
            )
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _ensure_image(self) -> None:
        inspected = self._docker("image", "inspect", self.image, check=False, timeout=60)
        if inspected.returncode == 0:
            return
        if not self.pull_image:
            raise DockerEnvironmentError(
                f"Docker image is not available locally and pulling is disabled: {self.image}"
            )
        self._docker("pull", self.image, timeout=1800)

    def _copy_pristine_workspace(self) -> None:
        seed_name = f"novacode-swe-seed-{uuid.uuid4().hex[:10]}"
        created = self._docker(
            "create",
            "--name",
            seed_name,
            self.image,
            "sleep",
            "10m",
            timeout=120,
        )
        self._seed_container_id = created.stdout.strip() or seed_name
        try:
            self._docker(
                "cp",
                f"{self._seed_container_id}:{self.container_cwd}/.",
                os.fspath(self.workspace),
                timeout=600,
            )
        finally:
            if self._seed_container_id:
                self._docker("rm", "-f", self._seed_container_id, check=False, timeout=60)
                self._seed_container_id = None

    def _start_inference_container(self) -> None:
        name = f"novacode-swe-{uuid.uuid4().hex[:10]}"
        completed = self._docker(
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "--network",
            self.network,
            "--workdir",
            self.container_cwd,
            "--volume",
            f"{self.workspace}:{self.container_cwd}",
            self.image,
            "sleep",
            self.container_lifetime,
            timeout=180,
        )
        self.container_id = completed.stdout.strip() or name

    def _initialize_inference_container(self) -> None:
        """Make the host-owned bind mount usable by the image's container user."""

        identifier = self.container_id or ""
        self._docker(
            "exec",
            identifier,
            "git",
            "config",
            "--global",
            "--add",
            "safe.directory",
            self.container_cwd,
            timeout=30,
        )
        if self.base_commit:
            self._docker(
                "exec",
                "-w",
                self.container_cwd,
                identifier,
                "git",
                "reset",
                "--hard",
                self.base_commit,
                timeout=120,
            )
            self._docker(
                "exec",
                "-w",
                self.container_cwd,
                identifier,
                "git",
                "clean",
                "-fd",
                timeout=120,
            )
        writable = self._docker(
            "exec",
            "-w",
            self.container_cwd,
            identifier,
            "bash",
            "-c",
            "test -w .",
            check=False,
            timeout=30,
        )
        if writable.returncode != 0:
            raise DockerEnvironmentError(
                f"bind-mounted workspace is not writable in the inference container: "
                f"{self.container_cwd}"
            )

    def _docker(
        self,
        *args: str,
        check: bool = True,
        timeout: float = 120,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                [self.docker_command, *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DockerEnvironmentError(
                f"Docker command failed to start or timed out: {self.docker_command} {' '.join(args)}: {exc}"
            ) from exc
        if check and completed.returncode != 0:
            detail = (completed.stdout + completed.stderr).strip()
            raise DockerEnvironmentError(
                f"Docker command exited with {completed.returncode}: "
                f"{self.docker_command} {' '.join(args)}\n{detail}"
            )
        return completed
