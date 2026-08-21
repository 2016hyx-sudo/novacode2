"""Tests for storage_location and global_memory_location configuration."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from main import build_config, parse_args


def test_default_storage_location_is_project(tmp_path: Path) -> None:
    args = parse_args(["--workspace", str(tmp_path), "--structured-context"])
    config = build_config(args)

    assert config.storage_location == "project"
    assert config.agent_dir == tmp_path / ".agent"
    assert config.session_dir == tmp_path / ".agent" / "sessions"
    assert config.trace_dir == tmp_path / ".agent" / "traces"
    # Project memory is ALWAYS under project .agent/memories
    assert config.memory_project_dir == tmp_path / ".agent" / "memories"
    # Default global memory is in user home ~/.novacode/memories/global
    assert config.memory_global_dir == Path.home() / ".novacode" / "memories" / "global"


def test_global_memory_location_in_project(tmp_path: Path) -> None:
    args = parse_args([
        "--workspace", str(tmp_path),
        "--global-memory-location", "project",
    ])
    config = build_config(args)

    assert config.global_memory_location == "project"
    assert config.memory_project_dir == tmp_path / ".agent" / "memories"
    assert config.memory_global_dir == tmp_path / ".agent" / "memories" / "global"


def test_global_memory_location_env_var(tmp_path: Path) -> None:
    with patch.dict(os.environ, {"NOVACODE_GLOBAL_MEMORY_LOCATION": "project"}):
        args = parse_args(["--workspace", str(tmp_path)])
        config = build_config(args)

        assert config.global_memory_location == "project"
        assert config.memory_project_dir == tmp_path / ".agent" / "memories"
        assert config.memory_global_dir == tmp_path / ".agent" / "memories" / "global"


def test_user_storage_location_keeps_project_memory_in_workspace(tmp_path: Path) -> None:
    args = parse_args([
        "--workspace", str(tmp_path),
        "--structured-context",
        "--storage-location", "user",
    ])
    config = build_config(args)

    assert config.storage_location == "user"
    user_home = Path.home()
    # Sessions and traces live in user directory
    assert str(config.agent_dir).startswith(str(user_home / ".novacode" / "projects"))
    assert str(config.session_dir).startswith(str(config.agent_dir / "sessions"))
    assert str(config.trace_dir).startswith(str(config.agent_dir / "traces"))
    # Project memory remains in workspace .agent/memories
    assert config.memory_project_dir == tmp_path / ".agent" / "memories"
    assert config.memory_global_dir == user_home / ".novacode" / "memories" / "global"
