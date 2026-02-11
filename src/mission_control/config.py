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
class SchedulerConfig:
	"""Scheduler settings."""

	session_timeout: int = 2700
	cooldown: int = 60
	max_sessions_per_run: int = 10
	model: str = "sonnet"
	git: GitConfig = field(default_factory=GitConfig)
	budget: BudgetConfig = field(default_factory=BudgetConfig)


@dataclass
class MissionConfig:
	"""Top-level mission-control configuration."""

	target: TargetConfig = field(default_factory=TargetConfig)
	scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


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
	return sc


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
	return mc
