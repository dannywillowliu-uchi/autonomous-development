"""Tests for config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_control.config import MissionConfig, load_config


@pytest.fixture()
def full_config(tmp_path: Path) -> Path:
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "my-project"
path = "~/personal_projects/my-project"
branch = "main"
objective = "Build a production-ready API server"

[target.verification]
command = "uv run pytest -q && uv run ruff check src/"
timeout = 300

[scheduler]
session_timeout = 2700
cooldown = 60
max_sessions_per_run = 10
model = "sonnet"

[scheduler.git]
strategy = "branch-per-session"
auto_merge = false

[scheduler.budget]
max_per_session_usd = 5.0
max_per_run_usd = 50.0
""")
	return toml


@pytest.fixture()
def minimal_config(tmp_path: Path) -> Path:
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "tiny"
path = "/tmp/tiny"
""")
	return toml


def test_load_full_config(full_config: Path) -> None:
	cfg = load_config(full_config)
	assert cfg.target.name == "my-project"
	assert cfg.target.branch == "main"
	assert cfg.target.objective == "Build a production-ready API server"
	assert cfg.target.verification.command == "uv run pytest -q && uv run ruff check src/"
	assert cfg.target.verification.timeout == 300
	assert cfg.scheduler.session_timeout == 2700
	assert cfg.scheduler.cooldown == 60
	assert cfg.scheduler.max_sessions_per_run == 10
	assert cfg.scheduler.model == "sonnet"
	assert cfg.scheduler.git.strategy == "branch-per-session"
	assert cfg.scheduler.git.auto_merge is False
	assert cfg.scheduler.budget.max_per_session_usd == 5.0
	assert cfg.scheduler.budget.max_per_run_usd == 50.0


def test_load_minimal_config(minimal_config: Path) -> None:
	cfg = load_config(minimal_config)
	assert cfg.target.name == "tiny"
	assert cfg.target.path == "/tmp/tiny"
	# Defaults
	assert cfg.target.branch == "main"
	assert cfg.scheduler.session_timeout == 2700
	assert cfg.scheduler.model == "sonnet"
	assert cfg.scheduler.git.strategy == "branch-per-session"
	assert cfg.scheduler.budget.max_per_session_usd == 5.0


def test_config_defaults() -> None:
	cfg = MissionConfig()
	assert cfg.target.name == ""
	assert cfg.target.branch == "main"
	assert cfg.scheduler.session_timeout == 2700
	assert cfg.scheduler.cooldown == 60


def test_config_file_not_found() -> None:
	with pytest.raises(FileNotFoundError):
		load_config("/nonexistent/path.toml")


def test_resolved_path(minimal_config: Path) -> None:
	cfg = load_config(minimal_config)
	assert cfg.target.resolved_path == Path("/tmp/tiny")


def test_resolved_path_tilde(full_config: Path) -> None:
	cfg = load_config(full_config)
	resolved = cfg.target.resolved_path
	assert "~" not in str(resolved)
	assert str(resolved).endswith("personal_projects/my-project")
