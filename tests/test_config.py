"""Tests for config loading."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mission_control.config import (
	_ENV_DENYLIST,
	ContainerConfig,
	ContinuousConfig,
	HITLConfig,
	MissionConfig,
	ModelsConfig,
	SecurityConfig,
	ZFCConfig,
	claude_subprocess_env,
	load_config,
	validate_config,
)


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


class TestContinuousConfig:
	def test_defaults(self) -> None:
		cc = ContinuousConfig()
		assert cc.max_wall_time_seconds == 7200
		assert cc.stall_threshold_units == 10
		assert cc.stall_score_epsilon == 0.01
		assert cc.replan_interval_units == 5
		assert cc.verify_before_merge is True
		assert cc.backlog_min_size == 2
		assert cc.cooldown_between_units == 0

	def test_toml_parsing(self, tmp_path: Path) -> None:
		toml = tmp_path / "mission-control.toml"
		toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[continuous]
max_wall_time_seconds = 3600
stall_threshold_units = 5
stall_score_epsilon = 0.02
replan_interval_units = 3
verify_before_merge = false
backlog_min_size = 4
cooldown_between_units = 10
""")
		config = load_config(toml)
		assert config.continuous.max_wall_time_seconds == 3600
		assert config.continuous.stall_threshold_units == 5
		assert config.continuous.stall_score_epsilon == 0.02
		assert config.continuous.replan_interval_units == 3
		assert config.continuous.verify_before_merge is False
		assert config.continuous.backlog_min_size == 4
		assert config.continuous.cooldown_between_units == 10


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


# -- Security config tests --


def test_security_defaults() -> None:
	"""SecurityConfig defaults to empty extra_env_keys."""
	sc = SecurityConfig()
	assert sc.extra_env_keys == []


def test_security_parsed(tmp_path: Path) -> None:
	"""[security] extra_env_keys are parsed from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[security]
extra_env_keys = ["CUSTOM_VAR", "MY_PROJECT_KEY"]
""")
	cfg = load_config(toml)
	assert cfg.security.extra_env_keys == ["CUSTOM_VAR", "MY_PROJECT_KEY"]


def test_claude_subprocess_env_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Only allowlisted env vars pass through to workers."""
	monkeypatch.setenv("HOME", "/home/test")
	monkeypatch.setenv("PATH", "/usr/bin")
	monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "supersecret")
	monkeypatch.setenv("GITHUB_TOKEN", "ghp_abc123")
	monkeypatch.setenv("RANDOM_UNKNOWN_VAR", "should_not_pass")

	env = claude_subprocess_env()
	assert env.get("HOME") == "/home/test"
	assert env.get("PATH") == "/usr/bin"
	assert "AWS_SECRET_ACCESS_KEY" not in env
	assert "GITHUB_TOKEN" not in env
	assert "RANDOM_UNKNOWN_VAR" not in env


def test_claude_subprocess_env_denylist_blocks_extra_keys(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Denylist overrides extra_env_keys -- secrets can never be added."""
	cfg = MissionConfig()
	cfg.security.extra_env_keys = ["AWS_SECRET_ACCESS_KEY", "MY_SAFE_VAR"]
	cfg._resolved_extra_env_keys = set(cfg.security.extra_env_keys) - _ENV_DENYLIST

	monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "supersecret")
	monkeypatch.setenv("MY_SAFE_VAR", "allowed_value")

	env = claude_subprocess_env(cfg)
	assert "AWS_SECRET_ACCESS_KEY" not in env
	assert env.get("MY_SAFE_VAR") == "allowed_value"


def test_claude_subprocess_env_extra_keys_passed(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Extra env keys from config are included in worker env."""
	cfg = MissionConfig()
	cfg.security.extra_env_keys = ["CUSTOM_PROJECT_VAR"]
	cfg._resolved_extra_env_keys = {"CUSTOM_PROJECT_VAR"}

	monkeypatch.setenv("CUSTOM_PROJECT_VAR", "my_value")

	env = claude_subprocess_env(cfg)
	assert env.get("CUSTOM_PROJECT_VAR") == "my_value"


def test_claude_subprocess_env_anthropic_key_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
	"""ANTHROPIC_API_KEY is always blocked."""
	monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

	env = claude_subprocess_env()
	assert "ANTHROPIC_API_KEY" not in env


def test_claude_subprocess_env_no_config_returns_only_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
	"""claude_subprocess_env() with no args returns only allowlisted keys."""
	monkeypatch.setenv("HOME", "/home/test")
	monkeypatch.setenv("CUSTOM_PROJECT_VAR", "should_not_pass")

	env = claude_subprocess_env()
	assert env.get("HOME") == "/home/test"
	assert "CUSTOM_PROJECT_VAR" not in env


def test_two_configs_no_env_leakage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""Loading two configs produces isolated env dicts -- no cross-project leakage."""
	toml_a = tmp_path / "a.toml"
	toml_a.write_text("""
[target]
name = "project-a"
path = "/tmp/a"

[security]
extra_env_keys = ["VAR_A"]
""")
	toml_b = tmp_path / "b.toml"
	toml_b.write_text("""
[target]
name = "project-b"
path = "/tmp/b"

[security]
extra_env_keys = ["VAR_B"]
""")

	cfg_a = load_config(toml_a)
	cfg_b = load_config(toml_b)

	monkeypatch.setenv("VAR_A", "val_a")
	monkeypatch.setenv("VAR_B", "val_b")

	env_a = claude_subprocess_env(cfg_a)
	env_b = claude_subprocess_env(cfg_b)

	assert env_a.get("VAR_A") == "val_a"
	assert "VAR_B" not in env_a
	assert env_b.get("VAR_B") == "val_b"
	assert "VAR_A" not in env_b


def test_env_denylist_coverage() -> None:
	"""Denylist covers common secret env var names."""
	expected_blocked = {
		"ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN",
		"TELEGRAM_BOT_TOKEN", "DATABASE_URL",
	}
	assert expected_blocked.issubset(_ENV_DENYLIST)


# -- Container config tests --


def test_container_defaults() -> None:
	"""ContainerConfig defaults are sensible."""
	cc = ContainerConfig()
	assert cc.image == "mission-control-worker:latest"
	assert cc.docker_executable == "docker"
	assert cc.workspace_mount == "/workspace"
	assert cc.claude_config_dir == ""
	assert cc.extra_volumes == []
	assert cc.cap_drop == ["ALL"]
	assert cc.security_opt == ["no-new-privileges:true"]
	assert cc.network == "bridge"
	assert cc.run_as_user == "10000:10000"
	assert cc.startup_timeout == 60


def test_container_parsed(tmp_path: Path) -> None:
	"""[backend.container] values are parsed correctly from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[backend]
type = "container"

[backend.container]
image = "my-worker:v2"
docker_executable = "podman"
workspace_mount = "/work"
claude_config_dir = "~/.config/claude"
extra_volumes = ["/data:/data:ro"]
cap_drop = ["NET_RAW", "SYS_PTRACE"]
security_opt = ["no-new-privileges:true", "seccomp=default"]
network = "none"
run_as_user = "1000:1000"
startup_timeout = 120
""")
	cfg = load_config(toml)
	assert cfg.backend.type == "container"
	cc = cfg.backend.container
	assert cc.image == "my-worker:v2"
	assert cc.docker_executable == "podman"
	assert cc.workspace_mount == "/work"
	assert cc.claude_config_dir == "~/.config/claude"
	assert cc.extra_volumes == ["/data:/data:ro"]
	assert cc.cap_drop == ["NET_RAW", "SYS_PTRACE"]
	assert cc.security_opt == ["no-new-privileges:true", "seccomp=default"]
	assert cc.network == "none"
	assert cc.run_as_user == "1000:1000"
	assert cc.startup_timeout == 120


def test_container_partial(tmp_path: Path) -> None:
	"""Partial [backend.container] fills only specified fields."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[backend]
type = "container"

[backend.container]
image = "custom:latest"
""")
	cfg = load_config(toml)
	cc = cfg.backend.container
	assert cc.image == "custom:latest"
	assert cc.docker_executable == "docker"
	assert cc.cap_drop == ["ALL"]


def test_validate_docker_missing(tmp_path: Path) -> None:
	"""validate_config reports error when docker executable not found."""
	cfg = _valid_config(tmp_path)
	cfg.backend.type = "container"
	cfg.backend.container.docker_executable = "totally_nonexistent_docker_xyz"
	issues = validate_config(cfg)
	errors = [msg for lvl, msg in issues if lvl == "error"]
	assert any("docker executable not found" in e for e in errors)


def test_validate_container_empty_image(tmp_path: Path) -> None:
	"""validate_config reports error when container image is empty."""
	cfg = _valid_config(tmp_path)
	cfg.backend.type = "container"
	cfg.backend.container.image = ""
	issues = validate_config(cfg)
	errors = [msg for lvl, msg in issues if lvl == "error"]
	assert any("image must not be empty" in e for e in errors)


def test_validate_container_claude_config_dir_missing(tmp_path: Path) -> None:
	"""validate_config reports error when claude_config_dir doesn't exist."""
	cfg = _valid_config(tmp_path)
	cfg.backend.type = "container"
	cfg.backend.container.claude_config_dir = str(tmp_path / "nonexistent_dir")
	issues = validate_config(cfg)
	errors = [msg for lvl, msg in issues if lvl == "error"]
	assert any("claude_config_dir does not exist" in e for e in errors)


# -- HITL config tests --


class TestHITLConfig:
	def test_hitl_defaults(self) -> None:
		"""HITLConfig and HITLGateConfig have correct defaults."""
		hc = HITLConfig()
		assert hc.approvals_dir == ".mc-approvals"
		assert hc.telegram_poll_interval == 5.0
		assert hc.push_gate.enabled is False
		assert hc.push_gate.timeout_seconds == 300
		assert hc.push_gate.timeout_action == "approve"
		assert hc.large_merge_gate.enabled is False
		assert hc.large_merge_gate.large_merge_threshold_lines == 500
		assert hc.large_merge_gate.large_merge_threshold_files == 20

	def test_hitl_on_mission_config(self) -> None:
		"""MissionConfig includes hitl field with defaults."""
		mc = MissionConfig()
		assert isinstance(mc.hitl, HITLConfig)
		assert mc.hitl.push_gate.enabled is False

	def test_hitl_parsed_from_toml(self, tmp_path: Path) -> None:
		"""[hitl] section values are parsed correctly from TOML."""
		toml = tmp_path / "mission-control.toml"
		toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[hitl]
approvals_dir = ".approvals"
telegram_poll_interval = 2.0

[hitl.push_gate]
enabled = true
timeout_seconds = 120
timeout_action = "deny"

[hitl.large_merge_gate]
enabled = true
large_merge_threshold_lines = 200
large_merge_threshold_files = 10
""")
		cfg = load_config(toml)
		assert cfg.hitl.approvals_dir == ".approvals"
		assert cfg.hitl.telegram_poll_interval == 2.0
		assert cfg.hitl.push_gate.enabled is True
		assert cfg.hitl.push_gate.timeout_seconds == 120
		assert cfg.hitl.push_gate.timeout_action == "deny"
		assert cfg.hitl.large_merge_gate.enabled is True
		assert cfg.hitl.large_merge_gate.large_merge_threshold_lines == 200
		assert cfg.hitl.large_merge_gate.large_merge_threshold_files == 10

	def test_hitl_defaults_when_omitted(self, minimal_config: Path) -> None:
		"""Without [hitl] section, defaults are used."""
		cfg = load_config(minimal_config)
		assert cfg.hitl.push_gate.enabled is False
		assert cfg.hitl.large_merge_gate.enabled is False
		assert cfg.hitl.approvals_dir == ".mc-approvals"


# -- ZFC config tests --


class TestZFCConfig:
	def test_zfc_defaults(self) -> None:
		"""ZFCConfig has correct defaults."""
		zc = ZFCConfig()
		assert zc.zfc_ambition_scoring is False
		assert zc.zfc_fixup_prompts is False
		assert zc.zfc_propose_objective is False
		assert zc.llm_timeout == 60
		assert zc.llm_budget_usd == 0.10
		assert zc.model == ""

	def test_zfc_on_mission_config(self) -> None:
		"""MissionConfig includes zfc field with defaults."""
		mc = MissionConfig()
		assert isinstance(mc.zfc, ZFCConfig)
		assert mc.zfc.zfc_ambition_scoring is False

	def test_zfc_parsed_from_toml(self, tmp_path: Path) -> None:
		"""[zfc] section values are parsed correctly from TOML."""
		toml = tmp_path / "mission-control.toml"
		toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[zfc]
zfc_ambition_scoring = true
zfc_fixup_prompts = true
zfc_propose_objective = true
llm_timeout = 30
llm_budget_usd = 0.25
model = "sonnet"
""")
		cfg = load_config(toml)
		assert cfg.zfc.zfc_ambition_scoring is True
		assert cfg.zfc.zfc_fixup_prompts is True
		assert cfg.zfc.zfc_propose_objective is True
		assert cfg.zfc.llm_timeout == 30
		assert cfg.zfc.llm_budget_usd == 0.25
		assert cfg.zfc.model == "sonnet"

	def test_zfc_partial(self, tmp_path: Path) -> None:
		"""Partial [zfc] section fills only specified fields."""
		toml = tmp_path / "mission-control.toml"
		toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[zfc]
zfc_ambition_scoring = true
""")
		cfg = load_config(toml)
		assert cfg.zfc.zfc_ambition_scoring is True
		assert cfg.zfc.zfc_fixup_prompts is False
		assert cfg.zfc.model == ""

	def test_zfc_defaults_when_omitted(self, minimal_config: Path) -> None:
		"""Without [zfc] section, defaults are used."""
		cfg = load_config(minimal_config)
		assert cfg.zfc.zfc_ambition_scoring is False
		assert cfg.zfc.zfc_fixup_prompts is False
		assert cfg.zfc.llm_timeout == 60
