"""TOML configuration loader for mission-control."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VerificationConfig:
	"""Target project verification settings."""

	command: str = "pytest -q"
	timeout: int = 300


@dataclass
class TargetConfig:
	"""Target project configuration."""

	name: str = ""
	path: str = ""
	branch: str = "main"
	objective: str = ""
	verification: VerificationConfig = field(default_factory=VerificationConfig)

	@property
	def resolved_path(self) -> Path:
		return Path(os.path.expanduser(self.path))


@dataclass
class GitConfig:
	"""Git strategy settings."""

	strategy: str = "branch-per-session"
	auto_merge: bool = False


@dataclass
class BudgetConfig:
	"""Cost budget settings."""

	max_per_session_usd: float = 5.0
	max_per_run_usd: float = 50.0


@dataclass
class ParallelConfig:
	"""Parallel execution settings."""

	num_workers: int = 4
	pool_dir: str = ""
	heartbeat_timeout: int = 600
	max_rebase_attempts: int = 3
	warm_clones: int = 0


@dataclass
class SchedulerConfig:
	"""Scheduler settings."""

	session_timeout: int = 2700
	cooldown: int = 60
	max_sessions_per_run: int = 10
	model: str = "sonnet"
	git: GitConfig = field(default_factory=GitConfig)
	budget: BudgetConfig = field(default_factory=BudgetConfig)
	parallel: ParallelConfig = field(default_factory=ParallelConfig)


@dataclass
class RoundsConfig:
	"""Outer loop round settings for mission mode."""

	max_rounds: int = 20
	stall_threshold: int = 3  # rounds with no improvement before stopping
	cooldown_between_rounds: int = 30  # seconds


@dataclass
class PlannerConfig:
	"""Recursive planner settings."""

	max_depth: int = 3  # max recursion depth (capped at 4)
	max_children_per_node: int = 5
	budget_per_call_usd: float = 1.0


@dataclass
class GreenBranchConfig:
	"""Green branch pattern settings."""

	working_branch: str = "mc/working"
	green_branch: str = "mc/green"
	fixup_max_attempts: int = 3


@dataclass
class SSHHostConfig:
	"""A single SSH host for distributed workers."""

	hostname: str = ""
	user: str = ""
	max_workers: int = 4


@dataclass
class BackendConfig:
	"""Worker backend settings."""

	type: str = "local"  # local/ssh/container
	ssh_hosts: list[SSHHostConfig] = field(default_factory=list)


@dataclass
class MissionConfig:
	"""Top-level mission-control configuration."""

	target: TargetConfig = field(default_factory=TargetConfig)
	scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
	rounds: RoundsConfig = field(default_factory=RoundsConfig)
	planner: PlannerConfig = field(default_factory=PlannerConfig)
	green_branch: GreenBranchConfig = field(default_factory=GreenBranchConfig)
	backend: BackendConfig = field(default_factory=BackendConfig)


def _build_verification(data: dict[str, Any]) -> VerificationConfig:
	vc = VerificationConfig()
	if "command" in data:
		vc.command = str(data["command"])
	if "timeout" in data:
		vc.timeout = int(data["timeout"])
	return vc


def _build_target(data: dict[str, Any]) -> TargetConfig:
	tc = TargetConfig()
	for key in ("name", "path", "branch", "objective"):
		if key in data:
			setattr(tc, key, str(data[key]))
	if "verification" in data:
		tc.verification = _build_verification(data["verification"])
	return tc


def _build_git(data: dict[str, Any]) -> GitConfig:
	gc = GitConfig()
	if "strategy" in data:
		gc.strategy = str(data["strategy"])
	if "auto_merge" in data:
		gc.auto_merge = bool(data["auto_merge"])
	return gc


def _build_budget(data: dict[str, Any]) -> BudgetConfig:
	bc = BudgetConfig()
	if "max_per_session_usd" in data:
		bc.max_per_session_usd = float(data["max_per_session_usd"])
	if "max_per_run_usd" in data:
		bc.max_per_run_usd = float(data["max_per_run_usd"])
	return bc


def _build_parallel(data: dict[str, Any]) -> ParallelConfig:
	pc = ParallelConfig()
	for key in ("num_workers", "heartbeat_timeout", "max_rebase_attempts", "warm_clones"):
		if key in data:
			setattr(pc, key, int(data[key]))
	if "pool_dir" in data:
		pc.pool_dir = str(data["pool_dir"])
	return pc


def _build_scheduler(data: dict[str, Any]) -> SchedulerConfig:
	sc = SchedulerConfig()
	for key in ("session_timeout", "cooldown", "max_sessions_per_run"):
		if key in data:
			setattr(sc, key, int(data[key]))
	if "model" in data:
		sc.model = str(data["model"])
	if "git" in data:
		sc.git = _build_git(data["git"])
	if "budget" in data:
		sc.budget = _build_budget(data["budget"])
	if "parallel" in data:
		sc.parallel = _build_parallel(data["parallel"])
	return sc


def _build_rounds(data: dict[str, Any]) -> RoundsConfig:
	rc = RoundsConfig()
	for key in ("max_rounds", "stall_threshold", "cooldown_between_rounds"):
		if key in data:
			setattr(rc, key, int(data[key]))
	return rc


def _build_planner_config(data: dict[str, Any]) -> PlannerConfig:
	pc = PlannerConfig()
	for key in ("max_depth", "max_children_per_node"):
		if key in data:
			setattr(pc, key, int(data[key]))
	if "budget_per_call_usd" in data:
		pc.budget_per_call_usd = float(data["budget_per_call_usd"])
	# Hard cap max_depth at 4 to prevent runaway recursion
	pc.max_depth = min(pc.max_depth, 4)
	return pc


def _build_green_branch(data: dict[str, Any]) -> GreenBranchConfig:
	gc = GreenBranchConfig()
	for key in ("working_branch", "green_branch"):
		if key in data:
			setattr(gc, key, str(data[key]))
	if "fixup_max_attempts" in data:
		gc.fixup_max_attempts = int(data["fixup_max_attempts"])
	return gc


def _build_backend(data: dict[str, Any]) -> BackendConfig:
	bc = BackendConfig()
	if "type" in data:
		bc.type = str(data["type"])
	if "ssh" in data:
		ssh_data = data["ssh"]
		hosts = ssh_data.get("hosts", [])
		for h in hosts:
			host = SSHHostConfig()
			if "hostname" in h:
				host.hostname = str(h["hostname"])
			if "user" in h:
				host.user = str(h["user"])
			if "max_workers" in h:
				host.max_workers = int(h["max_workers"])
			bc.ssh_hosts.append(host)
	return bc


def load_config(path: str | Path) -> MissionConfig:
	"""Load a mission-control.toml config file.

	Args:
		path: Path to the TOML config file.

	Returns:
		Parsed MissionConfig.

	Raises:
		FileNotFoundError: If config file doesn't exist.
		tomllib.TOMLDecodeError: If config is invalid TOML.
	"""
	config_path = Path(path)
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	with open(config_path, "rb") as f:
		data = tomllib.load(f)

	mc = MissionConfig()
	if "target" in data:
		mc.target = _build_target(data["target"])
	if "scheduler" in data:
		mc.scheduler = _build_scheduler(data["scheduler"])
	if "rounds" in data:
		mc.rounds = _build_rounds(data["rounds"])
	if "planner" in data:
		mc.planner = _build_planner_config(data["planner"])
	if "green_branch" in data:
		mc.green_branch = _build_green_branch(data["green_branch"])
	if "backend" in data:
		mc.backend = _build_backend(data["backend"])
	return mc
