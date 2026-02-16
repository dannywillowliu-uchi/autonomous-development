"""Tests for config loading."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mission_control.config import MissionConfig, load_config, validate_config


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


# -- Parallel config tests --


def test_parallel_config_defaults() -> None:
	cfg = MissionConfig()
	assert cfg.scheduler.parallel.num_workers == 4
	assert cfg.scheduler.parallel.pool_dir == ""
	assert cfg.scheduler.parallel.heartbeat_timeout == 600
	assert cfg.scheduler.parallel.max_rebase_attempts == 3
	assert cfg.scheduler.parallel.warm_clones == 0


def test_parallel_config_from_toml(tmp_path: Path) -> None:
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[scheduler.parallel]
num_workers = 8
pool_dir = "/tmp/pool"
heartbeat_timeout = 300
max_rebase_attempts = 5
warm_clones = 2
""")
	cfg = load_config(toml)
	assert cfg.scheduler.parallel.num_workers == 8
	assert cfg.scheduler.parallel.pool_dir == "/tmp/pool"
	assert cfg.scheduler.parallel.heartbeat_timeout == 300
	assert cfg.scheduler.parallel.max_rebase_attempts == 5
	assert cfg.scheduler.parallel.warm_clones == 2


def test_parallel_config_partial(tmp_path: Path) -> None:
	"""Partial parallel config uses defaults for missing fields."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[scheduler.parallel]
num_workers = 16
""")
	cfg = load_config(toml)
	assert cfg.scheduler.parallel.num_workers == 16
	assert cfg.scheduler.parallel.heartbeat_timeout == 600  # default


def test_new_config_field_defaults() -> None:
	"""New config fields have sensible defaults."""
	cfg = MissionConfig()
	assert cfg.scheduler.budget.evaluator_budget_usd == 0.50
	assert cfg.scheduler.budget.fixup_budget_usd == 2.0
	assert cfg.scheduler.monitor_interval == 5
	assert cfg.scheduler.output_summary_max_chars == 500
	assert cfg.scheduler.polling_interval == 5
	assert cfg.scheduler.raw_output_max_chars == 4000
	assert cfg.scheduler.session_lookback == 5


def test_new_config_fields_from_toml(tmp_path: Path) -> None:
	"""New config fields can be loaded from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[scheduler]
monitor_interval = 10
output_summary_max_chars = 1000
polling_interval = 3
raw_output_max_chars = 8000
session_lookback = 10

[scheduler.budget]
evaluator_budget_usd = 1.0
fixup_budget_usd = 3.0
""")
	cfg = load_config(toml)
	assert cfg.scheduler.monitor_interval == 10
	assert cfg.scheduler.output_summary_max_chars == 1000
	assert cfg.scheduler.polling_interval == 3
	assert cfg.scheduler.raw_output_max_chars == 8000
	assert cfg.scheduler.session_lookback == 10
	assert cfg.scheduler.budget.evaluator_budget_usd == 1.0
	assert cfg.scheduler.budget.fixup_budget_usd == 3.0


def test_rounds_config_defaults() -> None:
	"""RoundsConfig new fields have sensible defaults."""
	cfg = MissionConfig()
	assert cfg.rounds.stall_score_epsilon == 0.01
	assert cfg.rounds.max_discoveries_per_round == 20
	assert cfg.rounds.max_discovery_chars == 4000
	assert cfg.rounds.max_summary_items == 10
	assert cfg.rounds.timeout_multiplier == 1.2


def test_rounds_config_from_toml(tmp_path: Path) -> None:
	"""RoundsConfig new fields can be loaded from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[rounds]
max_rounds = 5
stall_threshold = 2
stall_score_epsilon = 0.05
max_discoveries_per_round = 10
max_discovery_chars = 2000
max_summary_items = 5
timeout_multiplier = 1.5
""")
	cfg = load_config(toml)
	assert cfg.rounds.max_rounds == 5
	assert cfg.rounds.stall_threshold == 2
	assert cfg.rounds.stall_score_epsilon == 0.05
	assert cfg.rounds.max_discoveries_per_round == 10
	assert cfg.rounds.max_discovery_chars == 2000
	assert cfg.rounds.max_summary_items == 5
	assert cfg.rounds.timeout_multiplier == 1.5


def test_planner_config_defaults() -> None:
	"""PlannerConfig new fields have sensible defaults."""
	cfg = MissionConfig()
	assert cfg.planner.max_depth == 3
	assert cfg.planner.absolute_max_depth == 4
	assert cfg.planner.max_file_tree_chars == 2000


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


def test_planner_config_from_toml(tmp_path: Path) -> None:
	"""PlannerConfig new fields can be loaded from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[planner]
max_depth = 2
max_file_tree_chars = 5000
budget_per_call_usd = 0.5
""")
	cfg = load_config(toml)
	assert cfg.planner.max_depth == 2
	assert cfg.planner.max_file_tree_chars == 5000
	assert cfg.planner.budget_per_call_usd == 0.5


def test_verification_setup_command_defaults() -> None:
	"""VerificationConfig setup fields default to empty/120."""
	cfg = MissionConfig()
	assert cfg.target.verification.setup_command == ""
	assert cfg.target.verification.setup_timeout == 120


def test_verification_setup_command_from_toml(tmp_path: Path) -> None:
	"""setup_command and setup_timeout parsed from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[target.verification]
command = "cd app && npx tsc --noEmit"
setup_command = "cd app && npm install --silent"
setup_timeout = 60
""")
	cfg = load_config(toml)
	assert cfg.target.verification.setup_command == "cd app && npm install --silent"
	assert cfg.target.verification.setup_timeout == 60


def test_deploy_config_defaults() -> None:
	"""DeployConfig has sensible defaults."""
	cfg = MissionConfig()
	assert cfg.deploy.enabled is False
	assert cfg.deploy.command == ""
	assert cfg.deploy.health_check_url == ""
	assert cfg.deploy.health_check_timeout == 60
	assert cfg.deploy.timeout == 300
	assert cfg.deploy.on_auto_push is False
	assert cfg.deploy.on_mission_end is True


def test_deploy_config_from_toml(tmp_path: Path) -> None:
	"""DeployConfig fields parsed from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[deploy]
enabled = true
command = "vercel deploy --prod"
health_check_url = "https://example.com"
health_check_timeout = 30
timeout = 600
on_auto_push = true
on_mission_end = false
""")
	cfg = load_config(toml)
	assert cfg.deploy.enabled is True
	assert cfg.deploy.command == "vercel deploy --prod"
	assert cfg.deploy.health_check_url == "https://example.com"
	assert cfg.deploy.health_check_timeout == 30
	assert cfg.deploy.timeout == 600
	assert cfg.deploy.on_auto_push is True
	assert cfg.deploy.on_mission_end is False


def test_green_branch_defaults() -> None:
	"""GreenBranchConfig defaults."""
	cfg = MissionConfig()
	assert cfg.green_branch.reset_on_init is True
	assert cfg.green_branch.working_branch == "mc/working"
	assert cfg.green_branch.green_branch == "mc/green"
	assert cfg.green_branch.fixup_max_attempts == 3


def test_green_branch_reset_on_init_false_from_toml(tmp_path: Path) -> None:
	"""reset_on_init=false in config is respected."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[green_branch]
reset_on_init = false
""")
	cfg = load_config(toml)
	assert cfg.green_branch.reset_on_init is False


def test_green_branch_reset_on_init_missing_defaults_true(tmp_path: Path) -> None:
	"""Missing reset_on_init defaults to True."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[green_branch]
working_branch = "mc/dev"
""")
	cfg = load_config(toml)
	assert cfg.green_branch.reset_on_init is True
	assert cfg.green_branch.working_branch == "mc/dev"


def test_green_branch_full_config_from_toml(tmp_path: Path) -> None:
	"""All GreenBranchConfig fields loaded from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[green_branch]
working_branch = "mc/wip"
green_branch = "mc/stable"
fixup_max_attempts = 5
reset_on_init = false
""")
	cfg = load_config(toml)
	assert cfg.green_branch.working_branch == "mc/wip"
	assert cfg.green_branch.green_branch == "mc/stable"
	assert cfg.green_branch.fixup_max_attempts == 5
	assert cfg.green_branch.reset_on_init is False


def test_green_branch_fixup_candidates_default() -> None:
	"""fixup_candidates defaults to 3."""
	cfg = MissionConfig()
	assert cfg.green_branch.fixup_candidates == 3


def test_green_branch_fixup_candidates_from_toml(tmp_path: Path) -> None:
	"""fixup_candidates loaded from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[green_branch]
fixup_candidates = 5
""")
	cfg = load_config(toml)
	assert cfg.green_branch.fixup_candidates == 5


# -- Discovery config tests --


def test_discovery_config_defaults() -> None:
	"""DiscoveryConfig new fields have sensible defaults."""
	cfg = MissionConfig()
	assert cfg.discovery.research_enabled is True
	assert cfg.discovery.research_model == "sonnet"
	assert cfg.discovery.research_parallel_queries == 3


def test_discovery_config_from_toml(tmp_path: Path) -> None:
	"""DiscoveryConfig new fields can be loaded from TOML."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[discovery]
research_enabled = false
research_model = "haiku"
research_parallel_queries = 5
""")
	cfg = load_config(toml)
	assert cfg.discovery.research_enabled is False
	assert cfg.discovery.research_model == "haiku"
	assert cfg.discovery.research_parallel_queries == 5


def test_discovery_config_partial(tmp_path: Path) -> None:
	"""Partial discovery config uses defaults for missing research fields."""
	toml = tmp_path / "mission-control.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[discovery]
model = "opus"
budget_per_call_usd = 3.0
""")
	cfg = load_config(toml)
	assert cfg.discovery.model == "opus"
	assert cfg.discovery.budget_per_call_usd == 3.0
	# New fields should use defaults
	assert cfg.discovery.research_enabled is True
	assert cfg.discovery.research_model == "sonnet"
	assert cfg.discovery.research_parallel_queries == 3


# -- validate_config tests --


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
