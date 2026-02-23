"""Shared pytest fixtures and factory functions for mission-control tests."""

from __future__ import annotations

from typing import Any

import pytest

from mission_control.config import MissionConfig, TargetConfig, VerificationConfig
from mission_control.db import Database


@pytest.fixture()
def db() -> Database:
	"""In-memory Database with schema initialized."""
	return Database(":memory:")


@pytest.fixture()
def config(tmp_path: Any) -> MissionConfig:
	"""Minimal MissionConfig with TargetConfig pointing to a tmp_path git repo."""
	cfg = MissionConfig()
	cfg.target = TargetConfig(
		name="test-proj",
		path=str(tmp_path),
		branch="main",
		verification=VerificationConfig(command="pytest -q"),
	)
	return cfg
