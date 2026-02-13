"""Tests for continuous mode feedback and evaluation."""

from __future__ import annotations

import json

from mission_control.db import Database
from mission_control.evaluator import compute_running_score
from mission_control.feedback import get_continuous_planner_context, record_unit_outcome
from mission_control.models import (
	Epoch,
	Handoff,
	Mission,
	Plan,
	Snapshot,
	UnitEvent,
	WorkUnit,
)


def _setup_db() -> Database:
	"""Create an in-memory DB with prerequisite records."""
	db = Database(":memory:")
	db.insert_mission(Mission(id="m1", objective="test"))
	db.insert_epoch(Epoch(id="ep1", mission_id="m1", number=1))
	plan = Plan(id="p1", objective="test")
	db.insert_plan(plan)
	return db


class TestRecordUnitOutcome:
	def test_creates_reflection_reward_experience(self) -> None:
		db = _setup_db()
		unit = WorkUnit(id="wu1", plan_id="p1", title="Fix tests", status="completed")
		db.insert_work_unit(unit)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)

		handoff = Handoff(
			work_unit_id="wu1",
			round_id="",
			status="completed",
			summary="Fixed 3 tests",
			discoveries=json.dumps(["Found config issue"]),
			concerns=json.dumps([]),
			files_changed=json.dumps(["test_foo.py"]),
		)

		snapshot_before = Snapshot(test_passed=10, test_total=12, test_failed=2)
		snapshot_after = Snapshot(test_passed=12, test_total=12, test_failed=0)

		reflection, reward, experience = record_unit_outcome(
			db=db,
			mission_id="m1",
			epoch=epoch,
			unit=unit,
			handoff=handoff,
			snapshot_before=snapshot_before,
			snapshot_after=snapshot_after,
			prev_score=0.5,
			current_score=0.7,
		)

		assert reflection.mission_id == "m1"
		assert reflection.epoch_id == "ep1"
		assert reflection.objective_score == 0.7
		assert reflection.score_delta == pytest.approx(0.2)
		assert reflection.completion_rate == 1.0
		assert reflection.discoveries_count == 1

		assert reward.reward > 0
		assert reward.epoch_id == "ep1"

		assert experience is not None
		assert experience.epoch_id == "ep1"
		assert experience.title == "Fix tests"
		assert experience.status == "completed"

	def test_creates_unit_event(self) -> None:
		db = _setup_db()
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task", status="completed")
		db.insert_work_unit(unit)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)

		record_unit_outcome(
			db=db, mission_id="m1", epoch=epoch, unit=unit,
			handoff=None, snapshot_before=None, snapshot_after=None,
			prev_score=0.0, current_score=0.5,
		)

		events = db.get_unit_events_for_epoch("ep1")
		assert len(events) == 1
		assert events[0].event_type == "completed"
		assert events[0].score_after == 0.5

	def test_failed_unit(self) -> None:
		db = _setup_db()
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task", status="failed")
		db.insert_work_unit(unit)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)

		reflection, reward, experience = record_unit_outcome(
			db=db, mission_id="m1", epoch=epoch, unit=unit,
			handoff=None, snapshot_before=None, snapshot_after=None,
			prev_score=0.5, current_score=0.5,
		)

		assert reflection.completion_rate == 0.0
		assert reflection.units_failed == 1
		assert experience is None

		events = db.get_unit_events_for_epoch("ep1")
		assert events[0].event_type == "failed"

	def test_no_handoff(self) -> None:
		db = _setup_db()
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task", status="completed")
		db.insert_work_unit(unit)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)

		_, _, experience = record_unit_outcome(
			db=db, mission_id="m1", epoch=epoch, unit=unit,
			handoff=None, snapshot_before=None, snapshot_after=None,
			prev_score=0.0, current_score=0.5,
		)

		assert experience is None


class TestGetContinuousPlannerContext:
	def test_empty_events(self) -> None:
		db = Database(":memory:")
		db.insert_mission(Mission(id="m1", objective="test"))
		result = get_continuous_planner_context(db, "m1")
		assert result == ""

	def test_with_events(self) -> None:
		db = _setup_db()
		plan = Plan(id="p2", objective="test")
		db.insert_plan(plan)
		wu = WorkUnit(id="wu1", plan_id="p2")
		db.insert_work_unit(wu)

		db.insert_unit_event(UnitEvent(
			id="ue1", mission_id="m1", epoch_id="ep1",
			work_unit_id="wu1", event_type="completed", score_after=0.6,
		))
		db.insert_unit_event(UnitEvent(
			id="ue2", mission_id="m1", epoch_id="ep1",
			work_unit_id="wu1", event_type="failed", score_after=0.5,
		))

		context = get_continuous_planner_context(db, "m1")
		assert "Recent scores" in context
		assert "Event distribution" in context
		assert "completed=1" in context
		assert "failed=1" in context


class TestComputeRunningScore:
	def test_clean_merge(self) -> None:
		before = Snapshot(test_passed=10, test_total=10, test_failed=0)
		after = Snapshot(test_passed=12, test_total=12, test_failed=0)
		result = compute_running_score(before, after, prev_score=0.5, unit_merged=True)
		assert result.score > 0.5

	def test_no_merge(self) -> None:
		before = Snapshot(test_passed=10, test_total=10, test_failed=0)
		after = Snapshot(test_passed=10, test_total=10, test_failed=0)
		result = compute_running_score(before, after, prev_score=0.5, unit_merged=False)
		# Baseline delta (0.5) * 0.60 + 0 * 0.25 + 0.5 * 0.15 = 0.375
		assert result.score < 0.5

	def test_regression(self) -> None:
		before = Snapshot(test_passed=10, test_total=10, test_failed=0)
		after = Snapshot(test_passed=8, test_total=10, test_failed=2)
		result = compute_running_score(before, after, prev_score=0.5, unit_merged=False)
		# Regression score should be very low
		assert result.score < 0.3

	def test_no_snapshots(self) -> None:
		result = compute_running_score(None, None, prev_score=0.0, unit_merged=True)
		assert result.score > 0  # Should still have merge bonus

	def test_met_threshold(self) -> None:
		before = Snapshot(test_passed=10, test_total=10, test_failed=0)
		after = Snapshot(test_passed=20, test_total=20, test_failed=0)
		result = compute_running_score(before, after, prev_score=0.8, unit_merged=True)
		# High delta + merge + momentum should approach 0.9+
		assert result.score >= 0.8


# Need pytest for approx
import pytest  # noqa: E402
