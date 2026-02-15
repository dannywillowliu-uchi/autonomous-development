"""Shared pytest fixtures and factory functions for mission-control tests."""

from __future__ import annotations

from typing import Any

import pytest

from mission_control.config import MissionConfig, TargetConfig, VerificationConfig
from mission_control.db import Database
from mission_control.models import Plan, Session, WorkUnit


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


def make_work_unit(**overrides: Any) -> WorkUnit:
	"""Create a WorkUnit with sensible defaults, overridable via kwargs."""
	defaults: dict[str, Any] = {
		"id": "wu1",
		"plan_id": "p1",
		"title": "Test task",
	}
	defaults.update(overrides)
	return WorkUnit(**defaults)


def make_session(**overrides: Any) -> Session:
	"""Create a Session with sensible defaults, overridable via kwargs."""
	defaults: dict[str, Any] = {
		"id": "s1",
		"target_name": "test-proj",
		"task_description": "Test session",
	}
	defaults.update(overrides)
	return Session(**defaults)


def make_plan(**overrides: Any) -> Plan:
	"""Create a Plan with sensible defaults, overridable via kwargs."""
	defaults: dict[str, Any] = {
		"id": "p1",
		"objective": "Test objective",
	}
	defaults.update(overrides)
	return Plan(**defaults)
