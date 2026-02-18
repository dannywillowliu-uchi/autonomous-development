"""Tests for continuous mode foundation: config, models, DB schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_control.config import ContinuousConfig, load_config
from mission_control.db import Database
from mission_control.models import Epoch, UnitEvent, WorkUnit

# -- Config tests --


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


# -- Database tests --


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


def _insert_mission(db: Database, mission_id: str = "m1") -> None:
	from mission_control.models import Mission
	db.insert_mission(Mission(id=mission_id, objective="test"))


class TestEpochDB:
	def test_insert_and_get(self, db: Database) -> None:
		_insert_mission(db)
		e = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(e)
		result = db.get_epoch("ep1")
		assert result is not None
		assert result.id == "ep1"
		assert result.mission_id == "m1"
		assert result.number == 1

	def test_update(self, db: Database) -> None:
		_insert_mission(db)
		e = Epoch(id="ep2", mission_id="m1", number=1)
		db.insert_epoch(e)
		e.units_completed = 3
		e.score_at_end = 0.75
		e.finished_at = "2025-01-01T00:00:00"
		db.update_epoch(e)
		result = db.get_epoch("ep2")
		assert result is not None
		assert result.units_completed == 3
		assert result.score_at_end == 0.75
		assert result.finished_at == "2025-01-01T00:00:00"

	def test_get_epochs_for_mission(self, db: Database) -> None:
		_insert_mission(db, "m1")
		_insert_mission(db, "m2")
		db.insert_epoch(Epoch(id="ep1", mission_id="m1", number=1))
		db.insert_epoch(Epoch(id="ep2", mission_id="m1", number=2))
		db.insert_epoch(Epoch(id="ep3", mission_id="m2", number=1))
		epochs = db.get_epochs_for_mission("m1")
		assert len(epochs) == 2
		assert epochs[0].number == 1
		assert epochs[1].number == 2


class TestUnitEventDB:
	def _make_deps(self, db: Database) -> tuple[str, str, str]:
		"""Create prerequisite rows for FK constraints."""
		from mission_control.models import Mission, Plan

		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		wu = WorkUnit(id="wu1", plan_id="p1")
		db.insert_work_unit(wu)
		return "m1", "ep1", "wu1"

	def test_insert_and_get(self, db: Database) -> None:
		m_id, ep_id, wu_id = self._make_deps(db)
		ue = UnitEvent(
			id="ue1", mission_id=m_id, epoch_id=ep_id,
			work_unit_id=wu_id, event_type="completed", score_after=0.6,
		)
		db.insert_unit_event(ue)
		events = db.get_unit_events_for_mission(m_id)
		assert len(events) == 1
		assert events[0].id == "ue1"
		assert events[0].event_type == "completed"
		assert events[0].score_after == 0.6

	def test_get_events_for_epoch(self, db: Database) -> None:
		m_id, ep_id, wu_id = self._make_deps(db)
		db.insert_unit_event(UnitEvent(
			id="ue1", mission_id=m_id, epoch_id=ep_id,
			work_unit_id=wu_id, event_type="dispatched",
		))
		db.insert_unit_event(UnitEvent(
			id="ue2", mission_id=m_id, epoch_id=ep_id,
			work_unit_id=wu_id, event_type="completed",
		))
		events = db.get_unit_events_for_epoch(ep_id)
		assert len(events) == 2


class TestMigrationIdempotency:
	def test_double_create_tables(self) -> None:
		"""Calling _create_tables twice should not error (idempotent)."""
		db = Database(":memory:")
		db._create_tables()
		# Should not raise
		db._create_tables()
