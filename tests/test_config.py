"""Tests for config loading."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mission_control.config import ContinuousConfig, MissionConfig, ModelsConfig, load_config, validate_config


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
	assert cfg.scheduler.model == "opus"
	assert cfg.scheduler.git.strategy == "branch-per-session"
	assert cfg.scheduler.budget.max_per_session_usd == 5.0


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


def test_planner_max_depth_capped_at_absolute(tmp_path: Path) -> None:
	"""max_depth is capped at absolute_max_depth."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[planner]
max_depth = 10
absolute_max_depth = 6
""")
	cfg = load_config(toml)
	assert cfg.planner.max_depth == 6
	assert cfg.planner.absolute_max_depth == 6


def _make_git_repo(path: Path) -> None:
	"""Initialize a bare git repo at the given path."""
	path.mkdir(parents=True, exist_ok=True)
	subprocess.run(["git", "init", str(path)], capture_output=True, check=True)


def _valid_config(tmp_path: Path) -> MissionConfig:
	"""Build a MissionConfig that passes all validation checks."""
	repo = tmp_path / "repo"
	_make_git_repo(repo)
	cfg = MissionConfig()
	cfg.target.name = "test"
	cfg.target.path = str(repo)
	cfg.target.verification.command = "git status"
	# Disable telegram notifications so token isn't checked
	cfg.notifications.telegram.on_heartbeat = False
	cfg.notifications.telegram.on_merge_fail = False
	cfg.notifications.telegram.on_mission_end = False
	return cfg


def test_validate_valid_config(tmp_path: Path) -> None:
	"""A well-formed config produces no errors or warnings."""
	cfg = _valid_config(tmp_path)
	issues = validate_config(cfg)
	assert issues == []


def test_validate_missing_target_path(tmp_path: Path) -> None:
	"""Missing target.path generates an error."""
	cfg = _valid_config(tmp_path)
	cfg.target.path = str(tmp_path / "nonexistent")
	issues = validate_config(cfg)
	errors = [msg for lvl, msg in issues if lvl == "error"]
	assert any("does not exist" in e for e in errors)


def test_validate_non_git_path(tmp_path: Path) -> None:
	"""A directory without .git generates an error."""
	plain_dir = tmp_path / "plain"
	plain_dir.mkdir()
	cfg = _valid_config(tmp_path)
	cfg.target.path = str(plain_dir)
	issues = validate_config(cfg)
	errors = [msg for lvl, msg in issues if lvl == "error"]
	assert any("not a git repository" in e for e in errors)


def test_validate_bad_bot_token_format(tmp_path: Path) -> None:
	"""Invalid Telegram bot_token format generates an error."""
	cfg = _valid_config(tmp_path)
	cfg.notifications.telegram.on_heartbeat = True
	cfg.notifications.telegram.bot_token = "not-a-valid-token"
	issues = validate_config(cfg)
	errors = [msg for lvl, msg in issues if lvl == "error"]
	assert any("bot_token format invalid" in e for e in errors)


def test_validate_good_bot_token_format(tmp_path: Path) -> None:
	"""Valid Telegram bot_token format does not generate an error."""
	cfg = _valid_config(tmp_path)
	cfg.notifications.telegram.on_heartbeat = True
	cfg.notifications.telegram.bot_token = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz_012345"
	issues = validate_config(cfg)
	errors = [msg for lvl, msg in issues if lvl == "error"]
	assert not any("bot_token" in e for e in errors)


def test_validate_suspicious_session_timeout(tmp_path: Path) -> None:
	"""session_timeout < 60 generates a warning."""
	cfg = _valid_config(tmp_path)
	cfg.scheduler.session_timeout = 30
	issues = validate_config(cfg)
	warnings = [msg for lvl, msg in issues if lvl == "warning"]
	assert any("session_timeout" in w for w in warnings)


def test_validate_suspicious_num_workers(tmp_path: Path) -> None:
	"""num_workers > 8 generates a warning."""
	cfg = _valid_config(tmp_path)
	cfg.scheduler.parallel.num_workers = 16
	issues = validate_config(cfg)
	warnings = [msg for lvl, msg in issues if lvl == "warning"]
	assert any("num_workers" in w for w in warnings)


def test_validate_zero_workers(tmp_path: Path) -> None:
	"""num_workers == 0 generates a warning."""
	cfg = _valid_config(tmp_path)
	cfg.scheduler.parallel.num_workers = 0
	issues = validate_config(cfg)
	warnings = [msg for lvl, msg in issues if lvl == "warning"]
	assert any("num_workers is zero" in w for w in warnings)


def test_validate_pool_dir_writable(tmp_path: Path) -> None:
	"""Non-existent pool_dir generates an error."""
	cfg = _valid_config(tmp_path)
	cfg.scheduler.parallel.pool_dir = str(tmp_path / "nonexistent_pool")
	issues = validate_config(cfg)
	errors = [msg for lvl, msg in issues if lvl == "error"]
	assert any("pool_dir" in e for e in errors)


def test_validate_pool_dir_exists_writable(tmp_path: Path) -> None:
	"""Existing writable pool_dir generates no error."""
	pool = tmp_path / "pool"
	pool.mkdir()
	cfg = _valid_config(tmp_path)
	cfg.scheduler.parallel.pool_dir = str(pool)
	issues = validate_config(cfg)
	errors = [msg for lvl, msg in issues if lvl == "error"]
	assert not any("pool_dir" in e for e in errors)


def test_validate_verification_command_not_found(tmp_path: Path) -> None:
	"""Non-existent verification command first token generates an error."""
	cfg = _valid_config(tmp_path)
	cfg.target.verification.command = "totally_nonexistent_binary_xyz --check"
	issues = validate_config(cfg)
	errors = [msg for lvl, msg in issues if lvl == "error"]
	assert any("verification command not found" in e for e in errors)


def test_models_defaults(minimal_config: Path) -> None:
	"""Without [models] section, all fields use defaults."""
	cfg = load_config(minimal_config)
	assert cfg.models.planner_model == "opus"
	assert cfg.models.worker_model == "opus"
	assert cfg.models.fixup_model == "opus"
	assert cfg.models.architect_editor_mode is False


def test_models_parsed(tmp_path: Path) -> None:
	"""[models] section values are parsed correctly."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[models]
planner_model = "sonnet"
worker_model = "haiku"
fixup_model = "sonnet"
architect_editor_mode = true
""")
	cfg = load_config(toml)
	assert cfg.models.planner_model == "sonnet"
	assert cfg.models.worker_model == "haiku"
	assert cfg.models.fixup_model == "sonnet"
	assert cfg.models.architect_editor_mode is True


def test_models_partial(tmp_path: Path) -> None:
	"""Partial [models] section fills only specified fields, rest default."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[models]
worker_model = "haiku"
""")
	cfg = load_config(toml)
	assert cfg.models.planner_model == "opus"
	assert cfg.models.worker_model == "haiku"
	assert cfg.models.fixup_model == "opus"
	assert cfg.models.architect_editor_mode is False


def test_models_dataclass_defaults() -> None:
	"""ModelsConfig dataclass defaults are correct."""
	mc = ModelsConfig()
	assert mc.planner_model == "opus"
	assert mc.worker_model == "opus"
	assert mc.fixup_model == "opus"
	assert mc.architect_editor_mode is False


def test_continuous_cleanup_defaults() -> None:
	"""ContinuousConfig has correct defaults for cleanup fields."""
	cc = ContinuousConfig()
	assert cc.cleanup_enabled is True
	assert cc.cleanup_interval == 3


def test_continuous_failure_fields_defaults() -> None:
	"""ContinuousConfig has correct defaults for failure fields."""
	cc = ContinuousConfig()
	assert cc.max_consecutive_failures == 3
	assert cc.failure_backoff_seconds == 60


def test_continuous_failure_fields_parsed(tmp_path: Path) -> None:
	"""[continuous] failure fields are parsed from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[continuous]
max_consecutive_failures = 5
failure_backoff_seconds = 120
""")
	cfg = load_config(toml)
	assert cfg.continuous.max_consecutive_failures == 5
	assert cfg.continuous.failure_backoff_seconds == 120


def test_continuous_failure_fields_defaults_when_omitted(minimal_config: Path) -> None:
	"""Without failure fields in [continuous], defaults are used."""
	cfg = load_config(minimal_config)
	assert cfg.continuous.max_consecutive_failures == 3
	assert cfg.continuous.failure_backoff_seconds == 60
