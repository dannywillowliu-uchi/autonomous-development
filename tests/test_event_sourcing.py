"""Tests for event sourcing: replay_events, derive_unit_status, claim event emission."""

from __future__ import annotations

import pytest

from mission_control.constants import (
	EVENT_TO_STATUS,
	UNIT_EVENT_CLAIMED,
	UNIT_EVENT_COMPLETED,
	UNIT_EVENT_DISPATCHED,
	UNIT_EVENT_MERGE_FAILED,
	UNIT_EVENT_MERGED,
	UNIT_EVENT_REJECTED,
	UNIT_EVENT_RESEARCH_COMPLETED,
	UNIT_EVENT_RETRY_QUEUED,
	UNIT_EVENT_RUNNING,
	UNIT_EVENT_TYPES,
)
from mission_control.db import Database, derive_unit_status
from mission_control.models import Epoch, Mission, Plan, UnitEvent, WorkUnit


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


def _insert_event(db: Database, **kwargs: object) -> UnitEvent:
	defaults = {
		"mission_id": "m1",
		"epoch_id": "ep1",
		"work_unit_id": "wu1",
		"event_type": "dispatched",
	}
	defaults.update(kwargs)
	event = UnitEvent(**defaults)  # type: ignore[arg-type]
	db.insert_unit_event(event)
	return event


class TestReplayEvents:
	def test_chronological_order(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1"))
		db.insert_epoch(Epoch(id="ep1", mission_id="m1"))
		db.insert_plan(Plan(id="p1"))
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="p1"))

		_insert_event(db, event_type="dispatched")
		_insert_event(db, event_type="claimed")
		_insert_event(db, event_type="running")

		events = db.replay_events("unit", "wu1")
		assert len(events) == 3
		assert [e.event_type for e in events] == ["dispatched", "claimed", "running"]

	def test_entity_type_mission(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1"))
		db.insert_epoch(Epoch(id="ep1", mission_id="m1"))
		db.insert_plan(Plan(id="p1"))
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="p1"))
		db.insert_work_unit(WorkUnit(id="wu2", plan_id="p1"))

		_insert_event(db, work_unit_id="wu1", event_type="dispatched")
		_insert_event(db, work_unit_id="wu2", event_type="dispatched")

		events = db.replay_events("mission", "m1")
		assert len(events) == 2

	def test_entity_type_epoch(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1"))
		db.insert_epoch(Epoch(id="ep1", mission_id="m1"))
		db.insert_epoch(Epoch(id="ep2", mission_id="m1"))
		db.insert_plan(Plan(id="p1"))
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="p1"))
		db.insert_work_unit(WorkUnit(id="wu2", plan_id="p1"))

		_insert_event(db, epoch_id="ep1", work_unit_id="wu1", event_type="dispatched")
		_insert_event(db, epoch_id="ep2", work_unit_id="wu2", event_type="dispatched")

		events = db.replay_events("epoch", "ep1")
		assert len(events) == 1
		assert events[0].epoch_id == "ep1"

	def test_empty_results(self, db: Database) -> None:
		events = db.replay_events("unit", "nonexistent")
		assert events == []

	def test_invalid_entity_type(self, db: Database) -> None:
		with pytest.raises(ValueError, match="Invalid entity_type"):
			db.replay_events("invalid", "id1")


class TestDeriveUnitStatus:
	def test_empty_returns_pending(self) -> None:
		assert derive_unit_status([]) == "pending"

	def test_full_lifecycle(self) -> None:
		events = [
			UnitEvent(event_type=UNIT_EVENT_DISPATCHED),
			UnitEvent(event_type=UNIT_EVENT_CLAIMED),
			UnitEvent(event_type=UNIT_EVENT_RUNNING),
			UnitEvent(event_type=UNIT_EVENT_COMPLETED),
			UnitEvent(event_type=UNIT_EVENT_MERGED),
		]
		assert derive_unit_status(events) == "completed"

	def test_retry_cycle(self) -> None:
		events = [
			UnitEvent(event_type=UNIT_EVENT_DISPATCHED),
			UnitEvent(event_type=UNIT_EVENT_CLAIMED),
			UnitEvent(event_type=UNIT_EVENT_RUNNING),
			UnitEvent(event_type=UNIT_EVENT_MERGE_FAILED),
			UnitEvent(event_type=UNIT_EVENT_RETRY_QUEUED),
			UnitEvent(event_type=UNIT_EVENT_DISPATCHED),
			UnitEvent(event_type=UNIT_EVENT_CLAIMED),
			UnitEvent(event_type=UNIT_EVENT_RUNNING),
			UnitEvent(event_type=UNIT_EVENT_COMPLETED),
			UnitEvent(event_type=UNIT_EVENT_MERGED),
		]
		assert derive_unit_status(events) == "completed"

	def test_rejection(self) -> None:
		events = [
			UnitEvent(event_type=UNIT_EVENT_DISPATCHED),
			UnitEvent(event_type=UNIT_EVENT_CLAIMED),
			UnitEvent(event_type=UNIT_EVENT_RUNNING),
			UnitEvent(event_type=UNIT_EVENT_MERGE_FAILED),
			UnitEvent(event_type=UNIT_EVENT_REJECTED),
		]
		assert derive_unit_status(events) == "failed"

	def test_unknown_events_skipped(self) -> None:
		events = [
			UnitEvent(event_type=UNIT_EVENT_DISPATCHED),
			UnitEvent(event_type="some_unknown_event"),
			UnitEvent(event_type=UNIT_EVENT_COMPLETED),
		]
		assert derive_unit_status(events) == "completed"

	def test_research_completed(self) -> None:
		events = [
			UnitEvent(event_type=UNIT_EVENT_DISPATCHED),
			UnitEvent(event_type=UNIT_EVENT_RESEARCH_COMPLETED),
		]
		assert derive_unit_status(events) == "completed"

	def test_all_event_types_mapped(self) -> None:
		"""Every UNIT_EVENT_TYPE should have an EVENT_TO_STATUS mapping."""
		for event_type in UNIT_EVENT_TYPES:
			assert event_type in EVENT_TO_STATUS


class TestClaimWorkUnitEvent:
	def test_claim_emits_event(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1"))
		db.insert_epoch(Epoch(id="ep1", mission_id="m1"))
		db.insert_plan(Plan(id="p1"))
		db.insert_work_unit(WorkUnit(
			id="wu1", plan_id="p1", status="pending", epoch_id="ep1",
		))

		unit = db.claim_work_unit("worker1")
		assert unit is not None
		assert unit.id == "wu1"

		events = db.replay_events("unit", "wu1")
		assert len(events) == 1
		assert events[0].event_type == UNIT_EVENT_CLAIMED
		assert events[0].mission_id == "m1"
		assert events[0].epoch_id == "ep1"

	def test_no_epoch_skips_event(self, db: Database) -> None:
		"""Units without epoch_id should not emit a claimed event."""
		db.insert_plan(Plan(id="p1"))
		db.insert_work_unit(WorkUnit(
			id="wu1", plan_id="p1", status="pending", epoch_id=None,
		))

		unit = db.claim_work_unit("worker1")
		assert unit is not None

		# No events because no epoch_id
		events = db.get_unit_events_for_mission("m1")
		assert len(events) == 0


class TestCrashRecoveryReplay:
	def test_derive_matches_mutable(self, db: Database) -> None:
		"""Derived status from events should match the mutable work_unit status."""
		db.insert_mission(Mission(id="m1"))
		db.insert_epoch(Epoch(id="ep1", mission_id="m1"))
		db.insert_plan(Plan(id="p1"))
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="p1", status="pending", epoch_id="ep1"))

		_insert_event(db, event_type=UNIT_EVENT_DISPATCHED)
		_insert_event(db, event_type=UNIT_EVENT_CLAIMED)
		_insert_event(db, event_type=UNIT_EVENT_RUNNING)
		_insert_event(db, event_type=UNIT_EVENT_COMPLETED)
		_insert_event(db, event_type=UNIT_EVENT_MERGED)

		# Mutable status would be "completed" after merge
		derived = db.derive_unit_status_from_db("wu1")
		assert derived == "completed"

	def test_derive_survives_stale_mutable(self, db: Database) -> None:
		"""Even if mutable status is stale, events tell the true story."""
		db.insert_mission(Mission(id="m1"))
		db.insert_epoch(Epoch(id="ep1", mission_id="m1"))
		db.insert_plan(Plan(id="p1"))
		# Mutable status stuck at "running" (simulating a crash)
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="p1", status="running", epoch_id="ep1"))

		# But events show it actually completed and was merged
		_insert_event(db, event_type=UNIT_EVENT_DISPATCHED)
		_insert_event(db, event_type=UNIT_EVENT_CLAIMED)
		_insert_event(db, event_type=UNIT_EVENT_RUNNING)
		_insert_event(db, event_type=UNIT_EVENT_COMPLETED)
		_insert_event(db, event_type=UNIT_EVENT_MERGED)

		derived = db.derive_unit_status_from_db("wu1")
		assert derived == "completed"

		# Mutable status is stale
		unit = db.get_work_unit("wu1")
		assert unit is not None
		assert unit.status == "running"  # stale
