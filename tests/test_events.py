"""Tests for event sourcing, JSONL event stream, and signal table CRUD."""

from __future__ import annotations

import json

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
from mission_control.event_stream import EventStream
from mission_control.models import Epoch, Mission, Plan, Signal, UnitEvent, WorkUnit, _now_iso


@pytest.fixture
def db():
	d = Database(":memory:")
	yield d
	d.close()


@pytest.fixture
def mission(db):
	m = Mission(objective="test objective", status="running")
	db.insert_mission(m)
	return m


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


class TestEventStream:
	def test_emit_writes_jsonl(self, tmp_path: object) -> None:
		from pathlib import Path
		p = Path(str(tmp_path)) / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("dispatched", mission_id="m1", unit_id="u1")
		stream.close()

		lines = p.read_text().strip().split("\n")
		assert len(lines) == 1
		record = json.loads(lines[0])
		assert record["event_type"] == "dispatched"
		assert record["mission_id"] == "m1"
		assert record["unit_id"] == "u1"
		assert "timestamp" in record

	def test_emit_multiple_events(self, tmp_path: object) -> None:
		from pathlib import Path
		p = Path(str(tmp_path)) / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("mission_started", mission_id="m1")
		stream.emit("dispatched", mission_id="m1", unit_id="u1")
		stream.emit("merged", mission_id="m1", unit_id="u1", cost_usd=0.05)
		stream.close()

		lines = p.read_text().strip().split("\n")
		assert len(lines) == 3
		types = [json.loads(line)["event_type"] for line in lines]
		assert types == ["mission_started", "dispatched", "merged"]

	def test_emit_with_details(self, tmp_path: object) -> None:
		from pathlib import Path
		p = Path(str(tmp_path)) / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit(
			"worker_started",
			mission_id="m1",
			worker_id="w1",
			details={"pid": 1234, "workspace": "/tmp/ws"},
		)
		stream.close()

		record = json.loads(p.read_text().strip())
		assert record["details"]["pid"] == 1234
		assert record["details"]["workspace"] == "/tmp/ws"

	def test_emit_with_token_usage(self, tmp_path: object) -> None:
		from pathlib import Path
		p = Path(str(tmp_path)) / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit(
			"merged",
			mission_id="m1",
			unit_id="u1",
			input_tokens=5000,
			output_tokens=2000,
			cost_usd=0.12,
		)
		stream.close()

		record = json.loads(p.read_text().strip())
		assert record["input_tokens"] == 5000
		assert record["output_tokens"] == 2000
		assert record["cost_usd"] == 0.12

	def test_emit_noop_when_not_opened(self) -> None:
		from pathlib import Path
		p = Path("/tmp/should-not-exist-event-stream-test.jsonl")
		stream = EventStream(p)
		# emit without open() should be a no-op
		stream.emit("dispatched", mission_id="m1")
		assert not p.exists()

	def test_emit_noop_after_close(self, tmp_path: object) -> None:
		from pathlib import Path
		p = Path(str(tmp_path)) / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("dispatched", mission_id="m1")
		stream.close()
		# emit after close should be a no-op
		stream.emit("merged", mission_id="m1")

		lines = p.read_text().strip().split("\n")
		assert len(lines) == 1

	def test_creates_parent_directories(self, tmp_path: object) -> None:
		from pathlib import Path
		p = Path(str(tmp_path)) / "nested" / "dir" / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("mission_started", mission_id="m1")
		stream.close()

		assert p.exists()
		record = json.loads(p.read_text().strip())
		assert record["event_type"] == "mission_started"

	def test_appends_to_existing_file(self, tmp_path: object) -> None:
		from pathlib import Path
		p = Path(str(tmp_path)) / "events.jsonl"
		# Write initial content
		p.write_text('{"event_type":"old"}\n')

		stream = EventStream(p)
		stream.open()
		stream.emit("new_event", mission_id="m1")
		stream.close()

		lines = p.read_text().strip().split("\n")
		assert len(lines) == 2
		assert json.loads(lines[0])["event_type"] == "old"
		assert json.loads(lines[1])["event_type"] == "new_event"

	def test_default_field_values(self, tmp_path: object) -> None:
		from pathlib import Path
		p = Path(str(tmp_path)) / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("test_event")
		stream.close()

		record = json.loads(p.read_text().strip())
		assert record["mission_id"] == ""
		assert record["epoch_id"] == ""
		assert record["unit_id"] == ""
		assert record["worker_id"] == ""
		assert record["details"] == {}
		assert record["input_tokens"] == 0
		assert record["output_tokens"] == 0
		assert record["cost_usd"] == 0.0


class TestSignalCRUD:
	def test_insert_and_get_pending(self, db, mission):
		signal = Signal(
			mission_id=mission.id,
			signal_type="stop",
			created_at=_now_iso(),
		)
		db.insert_signal(signal)

		pending = db.get_pending_signals(mission.id)
		assert len(pending) == 1
		assert pending[0].id == signal.id
		assert pending[0].signal_type == "stop"
		assert pending[0].status == "pending"

	def test_acknowledge_signal(self, db, mission):
		signal = Signal(
			mission_id=mission.id,
			signal_type="stop",
			created_at=_now_iso(),
		)
		db.insert_signal(signal)

		db.acknowledge_signal(signal.id)

		pending = db.get_pending_signals(mission.id)
		assert len(pending) == 0

	def test_multiple_signals(self, db, mission):
		for sig_type in ("stop", "retry_unit", "adjust"):
			signal = Signal(
				mission_id=mission.id,
				signal_type=sig_type,
				created_at=_now_iso(),
			)
			db.insert_signal(signal)

		pending = db.get_pending_signals(mission.id)
		assert len(pending) == 3

	def test_signals_scoped_to_mission(self, db, mission):
		other = Mission(objective="other", status="running")
		db.insert_mission(other)

		db.insert_signal(Signal(
			mission_id=mission.id, signal_type="stop", created_at=_now_iso(),
		))
		db.insert_signal(Signal(
			mission_id=other.id, signal_type="stop", created_at=_now_iso(),
		))

		pending = db.get_pending_signals(mission.id)
		assert len(pending) == 1

	def test_retry_unit_payload(self, db, mission):
		signal = Signal(
			mission_id=mission.id,
			signal_type="retry_unit",
			payload="unit123",
			created_at=_now_iso(),
		)
		db.insert_signal(signal)

		pending = db.get_pending_signals(mission.id)
		assert pending[0].payload == "unit123"

	def test_adjust_payload_json(self, db, mission):
		payload = json.dumps({"max_rounds": 30, "num_workers": 6})
		signal = Signal(
			mission_id=mission.id,
			signal_type="adjust",
			payload=payload,
			created_at=_now_iso(),
		)
		db.insert_signal(signal)

		pending = db.get_pending_signals(mission.id)
		parsed = json.loads(pending[0].payload)
		assert parsed["max_rounds"] == 30
		assert parsed["num_workers"] == 6

	def test_expire_stale_signals(self, db, mission):
		# Insert a signal with old timestamp
		signal = Signal(
			mission_id=mission.id,
			signal_type="stop",
			created_at="2020-01-01T00:00:00+00:00",
		)
		db.insert_signal(signal)

		expired_count = db.expire_stale_signals(timeout_minutes=10)
		assert expired_count == 1

		pending = db.get_pending_signals(mission.id)
		assert len(pending) == 0

	def test_expire_does_not_touch_fresh(self, db, mission):
		signal = Signal(
			mission_id=mission.id,
			signal_type="stop",
			created_at=_now_iso(),
		)
		db.insert_signal(signal)

		expired_count = db.expire_stale_signals(timeout_minutes=10)
		assert expired_count == 0

		pending = db.get_pending_signals(mission.id)
		assert len(pending) == 1
