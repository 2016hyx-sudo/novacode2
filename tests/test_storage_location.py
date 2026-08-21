"""Tests for storage_location configuration (project vs user directory)."""
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
    assert config.memory_project_dir == tmp_path / ".agent" / "memories"


def test_user_storage_location_cli_flag(tmp_path: Path) -> None:
    args = parse_args([
        "--workspace", str(tmp_path),
        "--structured-context",
        "--storage-location", "user",
    ])
    config = build_config(args)

    assert config.storage_location == "user"
    user_home = Path.home()
    assert str(config.agent_dir).startswith(str(user_home / ".novacode" / "projects"))
    assert str(config.session_dir).startswith(str(config.agent_dir / "sessions"))
    assert str(config.trace_dir).startswith(str(config.agent_dir / "traces"))
    assert str(config.memory_project_dir).startswith(str(config.agent_dir / "memories"))
    assert config.memory_global_dir == user_home / ".novacode" / "memories" / "global"


def test_user_storage_location_env_var(tmp_path: Path) -> None:
    with patch.dict(os.environ, {"NOVACODE_STORAGE_LOCATION": "user", "NOVACODE_STRUCTURED_CONTEXT": "1"}):
        args = parse_args(["--workspace", str(tmp_path)])
        config = build_config(args)

        assert config.storage_location == "user"
        user_home = Path.home()
        assert str(config.agent_dir).startswith(str(user_home / ".novacode" / "projects"))
        assert str(config.session_dir).startswith(str(config.agent_dir / "sessions"))
        assert str(config.memory_project_dir).startswith(str(config.agent_dir / "memories"))
