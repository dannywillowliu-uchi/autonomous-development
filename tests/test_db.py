"""Tests for SQLite database operations."""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

import pytest

from mission_control.db import Database
from mission_control.models import (
	Decision,
	DecompositionGrade,
	Epoch,
	MergeRequest,
	Mission,
	Plan,
	Session,
	Snapshot,
	TrajectoryRating,
	UnitEvent,
	UnitReview,
	Worker,
	WorkUnit,
)


class TestSessions:
	def test_insert_and_get(self, db: Database) -> None:
		s = Session(id="s1", target_name="proj", task_description="Fix bug")
		db.insert_session(s)
		result = db.get_session("s1")
		assert result is not None
		assert result.id == "s1"
		assert result.target_name == "proj"
		assert result.task_description == "Fix bug"
		assert result.status == "pending"

	def test_update_session(self, db: Database) -> None:
		s = Session(id="s2", target_name="proj")
		db.insert_session(s)
		s.status = "completed"
		s.exit_code = 0
		s.output_summary = "All done"
		db.update_session(s)
		result = db.get_session("s2")
		assert result is not None
		assert result.status == "completed"
		assert result.exit_code == 0
		assert result.output_summary == "All done"

	def test_get_nonexistent(self, db: Database) -> None:
		assert db.get_session("nope") is None

	def test_recent_sessions_ordering(self, db: Database) -> None:
		db.insert_session(Session(id="a", target_name="p", started_at="2025-01-01T00:00:00"))
		db.insert_session(Session(id="b", target_name="p", started_at="2025-01-02T00:00:00"))
		db.insert_session(Session(id="c", target_name="p", started_at="2025-01-03T00:00:00"))
		recent = db.get_recent_sessions(limit=2)
		assert len(recent) == 2
		assert recent[0].id == "c"
		assert recent[1].id == "b"


class TestSnapshots:
	def test_insert_and_get_latest(self, db: Database) -> None:
		s1 = Snapshot(id="snap1", taken_at="2025-01-01T00:00:00", test_total=10, test_passed=8, test_failed=2)
		s2 = Snapshot(id="snap2", taken_at="2025-01-02T00:00:00", test_total=10, test_passed=10, test_failed=0)
		db.insert_snapshot(s1)
		db.insert_snapshot(s2)
		latest = db.get_latest_snapshot()
		assert latest is not None
		assert latest.id == "snap2"
		assert latest.test_passed == 10

	def test_empty_snapshots(self, db: Database) -> None:
		assert db.get_latest_snapshot() is None


class TestDecisions:
	def test_insert_and_get_recent(self, db: Database) -> None:
		# Need a session for FK
		db.insert_session(Session(id="s1", target_name="p"))
		d = Decision(id="d1", session_id="s1", decision="Use branch strategy", rationale="Safer")
		db.insert_decision(d)
		recent = db.get_recent_decisions()
		assert len(recent) == 1
		assert recent[0].decision == "Use branch strategy"


class TestBulkPersist:
	def test_persist_session_result(self, db: Database) -> None:
		session = Session(id="bulk1", target_name="proj", task_description="Do stuff")
		before = Snapshot(id="b1", test_total=10, test_passed=8, test_failed=2)
		after = Snapshot(id="a1", test_total=10, test_passed=10, test_failed=0)
		decisions = [Decision(id="d1", session_id="", decision="Chose X", rationale="Better")]

		db.persist_session_result(session, before, after, decisions)

		assert db.get_session("bulk1") is not None
		assert db.get_latest_snapshot() is not None
		assert len(db.get_recent_decisions()) == 1

	def test_persist_session_result_atomic_rollback(self, db: Database) -> None:
		"""Verify persist_session_result rolls back all inserts on failure."""
		session = Session(id="atomic1", target_name="proj", task_description="Test atomicity")
		before = Snapshot(id="snap-b", test_total=5, test_passed=5, test_failed=0)
		# Create a snapshot with the same ID to trigger a unique constraint violation
		after = Snapshot(id="snap-b", test_total=5, test_passed=5, test_failed=0)

		with pytest.raises(sqlite3.IntegrityError):
			db.persist_session_result(session, before, after)

		# Session should NOT have been persisted due to rollback
		assert db.get_session("atomic1") is None


# -- Parallel mode tests --


class TestPlans:
	def test_insert_and_get(self, db: Database) -> None:
		p = Plan(id="p1", objective="Build API", status="active", total_units=3)
		db.insert_plan(p)
		result = db.get_plan("p1")
		assert result is not None
		assert result.objective == "Build API"
		assert result.status == "active"
		assert result.total_units == 3

	def test_update_plan(self, db: Database) -> None:
		p = Plan(id="p2", objective="Fix tests")
		db.insert_plan(p)
		p.status = "completed"
		p.completed_units = 5
		p.finished_at = "2025-01-01T00:00:00"
		db.update_plan(p)
		result = db.get_plan("p2")
		assert result is not None
		assert result.status == "completed"
		assert result.completed_units == 5
		assert result.finished_at == "2025-01-01T00:00:00"

	def test_get_nonexistent(self, db: Database) -> None:
		assert db.get_plan("nope") is None


class TestWorkUnits:
	def _make_plan(self, db: Database, plan_id: str = "plan1") -> None:
		db.insert_plan(Plan(id=plan_id, objective="test"))

	def test_insert_and_get(self, db: Database) -> None:
		self._make_plan(db)
		wu = WorkUnit(id="wu1", plan_id="plan1", title="Fix tests", priority=1)
		db.insert_work_unit(wu)
		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.title == "Fix tests"
		assert result.status == "pending"
		assert result.attempt == 0

	def test_update_work_unit(self, db: Database) -> None:
		self._make_plan(db)
		wu = WorkUnit(id="wu2", plan_id="plan1", title="Lint")
		db.insert_work_unit(wu)
		wu.status = "completed"
		wu.exit_code = 0
		wu.commit_hash = "abc123"
		db.update_work_unit(wu)
		result = db.get_work_unit("wu2")
		assert result is not None
		assert result.status == "completed"
		assert result.commit_hash == "abc123"

	def test_get_units_for_plan(self, db: Database) -> None:
		self._make_plan(db)
		db.insert_work_unit(WorkUnit(id="a", plan_id="plan1", title="A", priority=2))
		db.insert_work_unit(WorkUnit(id="b", plan_id="plan1", title="B", priority=1))
		units = db.get_work_units_for_plan("plan1")
		assert len(units) == 2
		assert units[0].id == "b"  # priority 1 first

	def test_atomic_claim_returns_different_units(self, db: Database) -> None:
		"""Two claims should return different work units."""
		self._make_plan(db)
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="plan1", title="Task 1", priority=1))
		db.insert_work_unit(WorkUnit(id="wu2", plan_id="plan1", title="Task 2", priority=2))

		claimed1 = db.claim_work_unit("worker-A")
		claimed2 = db.claim_work_unit("worker-B")

		assert claimed1 is not None
		assert claimed2 is not None
		assert claimed1.id != claimed2.id
		assert claimed1.status == "claimed"
		assert claimed2.status == "claimed"

	def test_claim_no_available_units(self, db: Database) -> None:
		self._make_plan(db)
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="plan1", status="completed"))
		result = db.claim_work_unit("worker-A")
		assert result is None

	def test_claim_respects_dependencies(self, db: Database) -> None:
		"""Units with incomplete deps should not be claimable."""
		self._make_plan(db)
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="plan1", title="Base", priority=1))
		db.insert_work_unit(WorkUnit(
			id="wu2", plan_id="plan1", title="Dependent", priority=1, depends_on="wu1",
		))

		# First claim should get wu1 (wu2 is blocked by wu1)
		claimed = db.claim_work_unit("worker-A")
		assert claimed is not None
		assert claimed.id == "wu1"

		# Second claim should get nothing (wu2 depends on incomplete wu1)
		claimed2 = db.claim_work_unit("worker-B")
		assert claimed2 is None

	def test_claim_after_dependency_completed(self, db: Database) -> None:
		"""Units become claimable when deps are completed."""
		self._make_plan(db)
		wu1 = WorkUnit(id="wu1", plan_id="plan1", title="Base")
		wu2 = WorkUnit(id="wu2", plan_id="plan1", title="Dependent", depends_on="wu1")
		db.insert_work_unit(wu1)
		db.insert_work_unit(wu2)

		# Complete wu1
		wu1.status = "completed"
		db.update_work_unit(wu1)

		# Now wu2 should be claimable
		claimed = db.claim_work_unit("worker-A")
		assert claimed is not None
		assert claimed.id == "wu2"

	def test_recover_stale_units(self, db: Database) -> None:
		"""Stale units should be released back to pending with incremented attempt."""
		self._make_plan(db)
		wu = WorkUnit(
			id="wu1", plan_id="plan1", title="Stale",
			status="running", worker_id="dead-worker",
			heartbeat_at="2020-01-01T00:00:00",  # very old
			attempt=1, max_attempts=3,
		)
		db.insert_work_unit(wu)

		recovered = db.recover_stale_units(timeout_seconds=60)
		assert len(recovered) == 1
		assert recovered[0].id == "wu1"
		assert recovered[0].status == "pending"
		assert recovered[0].worker_id is None
		assert recovered[0].attempt == 2  # incremented from 1

	def test_recover_skips_max_attempts(self, db: Database) -> None:
		"""Units at max attempts should not be recovered."""
		self._make_plan(db)
		wu = WorkUnit(
			id="wu1", plan_id="plan1", title="Exhausted",
			status="running", worker_id="dead",
			heartbeat_at="2020-01-01T00:00:00",
			attempt=3, max_attempts=3,
		)
		db.insert_work_unit(wu)
		recovered = db.recover_stale_units(timeout_seconds=60)
		assert len(recovered) == 0


class TestWorkers:
	def test_insert_and_get(self, db: Database) -> None:
		w = Worker(id="w1", workspace_path="/tmp/clone1", status="idle")
		db.insert_worker(w)
		result = db.get_worker("w1")
		assert result is not None
		assert result.workspace_path == "/tmp/clone1"
		assert result.status == "idle"

	def test_update_worker(self, db: Database) -> None:
		w = Worker(id="w2", workspace_path="/tmp/clone2")
		db.insert_worker(w)
		w.status = "working"
		w.current_unit_id = "wu1"
		w.units_completed = 3
		db.update_worker(w)
		result = db.get_worker("w2")
		assert result is not None
		assert result.status == "working"
		assert result.units_completed == 3

	def test_get_all_workers(self, db: Database) -> None:
		db.insert_worker(Worker(id="w1", workspace_path="/a", started_at="2025-01-01T00:00:00"))
		db.insert_worker(Worker(id="w2", workspace_path="/b", started_at="2025-01-02T00:00:00"))
		workers = db.get_all_workers()
		assert len(workers) == 2


class TestMergeRequests:
	def _setup(self, db: Database) -> None:
		db.insert_plan(Plan(id="p1", objective="test"))
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="p1"))

	def test_insert_and_get_next(self, db: Database) -> None:
		self._setup(db)
		mr = MergeRequest(
			id="mr1", work_unit_id="wu1", worker_id="w1",
			branch_name="mc/unit-wu1", commit_hash="abc",
			position=1,
		)
		db.insert_merge_request(mr)
		result = db.get_next_merge_request()
		assert result is not None
		assert result.id == "mr1"
		assert result.branch_name == "mc/unit-wu1"

	def test_update_merge_request(self, db: Database) -> None:
		self._setup(db)
		mr = MergeRequest(id="mr1", work_unit_id="wu1", worker_id="w1", position=1)
		db.insert_merge_request(mr)
		mr.status = "merged"
		mr.merged_at = "2025-01-01T00:00:00"
		db.update_merge_request(mr)
		# No pending left
		assert db.get_next_merge_request() is None

	def test_position_ordering(self, db: Database) -> None:
		self._setup(db)
		db.insert_work_unit(WorkUnit(id="wu2", plan_id="p1"))
		db.insert_merge_request(MergeRequest(
			id="mr2", work_unit_id="wu2", worker_id="w1", position=2,
		))
		db.insert_merge_request(MergeRequest(
			id="mr1", work_unit_id="wu1", worker_id="w1", position=1,
		))
		first = db.get_next_merge_request()
		assert first is not None
		assert first.id == "mr1"  # position 1 first

	def test_next_merge_position(self, db: Database) -> None:
		self._setup(db)
		assert db.get_next_merge_position() == 1
		db.insert_merge_request(MergeRequest(
			id="mr1", work_unit_id="wu1", worker_id="w1", position=1,
		))
		assert db.get_next_merge_position() == 2

	def test_get_processed_merge_requests_for_worker(self, db: Database) -> None:
		"""Returns only merged/rejected/conflict MRs for the specified worker."""
		self._setup(db)
		db.insert_work_unit(WorkUnit(id="wu2", plan_id="p1"))
		db.insert_work_unit(WorkUnit(id="wu3", plan_id="p1"))
		db.insert_work_unit(WorkUnit(id="wu4", plan_id="p1"))

		# Pending MR -- should NOT be returned
		db.insert_merge_request(MergeRequest(
			id="mr1", work_unit_id="wu1", worker_id="w1",
			branch_name="mc/unit-wu1", status="pending", position=1,
		))
		# Merged MR -- SHOULD be returned
		db.insert_merge_request(MergeRequest(
			id="mr2", work_unit_id="wu2", worker_id="w1",
			branch_name="mc/unit-wu2", status="merged", position=2,
		))
		# Rejected MR -- SHOULD be returned
		db.insert_merge_request(MergeRequest(
			id="mr3", work_unit_id="wu3", worker_id="w1",
			branch_name="mc/unit-wu3", status="rejected", position=3,
		))
		# Different worker -- should NOT be returned
		db.insert_merge_request(MergeRequest(
			id="mr4", work_unit_id="wu4", worker_id="w2",
			branch_name="mc/unit-wu4", status="merged", position=4,
		))

		results = db.get_processed_merge_requests_for_worker("w1")
		assert len(results) == 2
		assert {r.id for r in results} == {"mr2", "mr3"}

	def test_get_processed_merge_requests_includes_conflict(self, db: Database) -> None:
		"""Conflict status MRs are also returned as processed."""
		self._setup(db)
		db.insert_merge_request(MergeRequest(
			id="mr1", work_unit_id="wu1", worker_id="w1",
			branch_name="mc/unit-wu1", status="conflict", position=1,
		))

		results = db.get_processed_merge_requests_for_worker("w1")
		assert len(results) == 1
		assert results[0].status == "conflict"


class TestAsyncLock:
	def test_db_has_lock(self) -> None:
		"""Database should have an asyncio.Lock attribute."""
		db = Database(":memory:")
		assert hasattr(db, "_lock")
		assert isinstance(db._lock, asyncio.Lock)

	async def test_locked_call(self) -> None:
		"""locked_call should serialize access through the asyncio lock."""
		db = Database(":memory:")
		db.insert_plan(Plan(id="p1", objective="test"))
		wu = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(wu)

		result = await db.locked_call("get_work_unit", "wu1")
		assert result is not None
		assert result.id == "wu1"

	async def test_concurrent_claims_via_locked_call(self) -> None:
		"""Concurrent locked_call claims should return different units."""
		db = Database(":memory:")
		db.insert_plan(Plan(id="p1", objective="test"))
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="p1", title="Task 1", priority=1))
		db.insert_work_unit(WorkUnit(id="wu2", plan_id="p1", title="Task 2", priority=2))

		results = await asyncio.gather(
			db.locked_call("claim_work_unit", "worker-A"),
			db.locked_call("claim_work_unit", "worker-B"),
		)

		# Both should succeed with different units
		assert results[0] is not None
		assert results[1] is not None
		assert results[0].id != results[1].id


class TestMissionSummary:
	def _setup_mission_data(self, db: Database) -> str:
		"""Insert a mission with epochs, units, and events. Returns mission_id."""
		mission = Mission(id="m1", objective="Build API", status="completed", total_cost_usd=12.50)
		db.insert_mission(mission)

		epoch = Epoch(id="ep1", mission_id="m1", number=1, units_planned=3, units_completed=2, units_failed=1)
		db.insert_epoch(epoch)

		units = [
			WorkUnit(
				id="wu1", plan_id="p1", title="Unit 1", status="completed",
				epoch_id="ep1", cost_usd=3.0, input_tokens=1000, output_tokens=500,
			),
			WorkUnit(
				id="wu2", plan_id="p1", title="Unit 2", status="completed",
				epoch_id="ep1", cost_usd=4.0, input_tokens=2000, output_tokens=800,
			),
			WorkUnit(
				id="wu3", plan_id="p1", title="Unit 3", status="failed",
				epoch_id="ep1", cost_usd=1.5, input_tokens=500, output_tokens=200,
			),
		]
		db.insert_plan(Plan(id="p1", objective="Build API"))
		for u in units:
			db.insert_work_unit(u)

		events = [
			UnitEvent(id="ev1", mission_id="m1", epoch_id="ep1", work_unit_id="wu1", event_type="dispatched"),
			UnitEvent(id="ev2", mission_id="m1", epoch_id="ep1", work_unit_id="wu1", event_type="merged"),
			UnitEvent(id="ev3", mission_id="m1", epoch_id="ep1", work_unit_id="wu2", event_type="dispatched"),
			UnitEvent(id="ev4", mission_id="m1", epoch_id="ep1", work_unit_id="wu2", event_type="merged"),
			UnitEvent(id="ev5", mission_id="m1", epoch_id="ep1", work_unit_id="wu3", event_type="dispatched"),
			UnitEvent(id="ev6", mission_id="m1", epoch_id="ep1", work_unit_id="wu3", event_type="failed"),
		]
		for e in events:
			db.insert_unit_event(e)

		return "m1"

	def test_get_all_missions(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="First"))
		db.insert_mission(Mission(id="m2", objective="Second"))
		missions = db.get_all_missions()
		assert len(missions) == 2

	def test_get_mission_summary_unit_counts(self, db: Database) -> None:
		self._setup_mission_data(db)
		summary = db.get_mission_summary("m1")
		assert summary["units_by_status"]["completed"] == 2
		assert summary["units_by_status"]["failed"] == 1

	def test_get_mission_summary_tokens(self, db: Database) -> None:
		self._setup_mission_data(db)
		summary = db.get_mission_summary("m1")
		assert summary["total_input_tokens"] == 3500
		assert summary["total_output_tokens"] == 1500
		assert summary["total_cost_usd"] == 8.5

	def test_get_mission_summary_events(self, db: Database) -> None:
		self._setup_mission_data(db)
		summary = db.get_mission_summary("m1")
		assert summary["events_by_type"]["dispatched"] == 3
		assert summary["events_by_type"]["merged"] == 2
		assert summary["events_by_type"]["failed"] == 1

	def test_get_mission_summary_empty(self, db: Database) -> None:
		db.insert_mission(Mission(id="empty", objective="Empty"))
		summary = db.get_mission_summary("empty")
		assert summary["units_by_status"] == {}
		assert summary["total_input_tokens"] == 0
		assert summary["events_by_type"] == {}
		assert summary["epochs"] == []


class TestEpochDB:
	def _insert_mission(self, db: Database, mission_id: str = "m1") -> None:
		db.insert_mission(Mission(id=mission_id, objective="test"))

	def test_insert_and_get(self, db: Database) -> None:
		self._insert_mission(db)
		e = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(e)
		result = db.get_epoch("ep1")
		assert result is not None
		assert result.id == "ep1"
		assert result.mission_id == "m1"
		assert result.number == 1

	def test_update(self, db: Database) -> None:
		self._insert_mission(db)
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
		self._insert_mission(db, "m1")
		self._insert_mission(db, "m2")
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
		db._create_tables()


class TestBusyTimeout:
	def test_busy_timeout_set_on_file_db(self, tmp_path: Any) -> None:
		"""busy_timeout pragma should be set to 5000 for file-based DBs."""
		db_path = tmp_path / "test.db"
		db = Database(db_path)
		row = db.conn.execute("PRAGMA busy_timeout").fetchone()
		assert row[0] == 5000
		db.close()

class TestMigrationResilience:
	def test_duplicate_column_passes_silently(self, db: Database) -> None:
		"""Re-running migrations with already-existing columns should not raise."""
		db._migrate_epoch_columns()
		db._migrate_token_columns()


class TestContextManager:
	def test_context_manager_opens_and_closes(self) -> None:
		"""Database used as context manager should close on exit."""
		with Database(":memory:") as db:
			db.insert_session(Session(id="cm1", target_name="proj"))
			result = db.get_session("cm1")
			assert result is not None
			assert result.id == "cm1"
		# After exiting, the connection should be closed
		with pytest.raises(sqlite3.ProgrammingError):
			db.conn.execute("SELECT 1")


class TestValidateIdentifier:
	def test_valid_names(self) -> None:
		"""Valid SQL identifiers should pass without error."""
		for name in ["work_units", "epoch_id", "a", "_private", "Table1", "col_2_x"]:
			Database._validate_identifier(name)

	def test_rejects_sql_injection_attempts(self) -> None:
		injection_attempts = [
			"work_units; DROP TABLE work_units;--",
			"col FROM work_units;--",
			"epoch_id TEXT); DROP TABLE work_units;--",
			"name OR 1=1",
			"col\nDROP",
			"table name",
			"col-name",
			"col.name",
		]
		for attempt in injection_attempts:
			with pytest.raises(ValueError, match="Invalid SQL identifier"):
				Database._validate_identifier(attempt)


class TestMissionAmbitionScore:
	def test_insert_mission_with_ambition_score(self, db: Database) -> None:
		"""Mission with ambition_score round-trips through insert/get."""
		m = Mission(id="m1", objective="Build new system", ambition_score=7)
		db.insert_mission(m)
		result = db.get_mission("m1")
		assert result is not None
		assert result.ambition_score == 7

	def test_update_mission_ambition_score(self, db: Database) -> None:
		"""ambition_score can be updated after insert."""
		m = Mission(id="m3", objective="Refactor")
		db.insert_mission(m)
		m.ambition_score = 5
		db.update_mission(m)
		result = db.get_mission("m3")
		assert result is not None
		assert result.ambition_score == 5


class TestUnitReviews:
	def _setup(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="Build API"))
		db.insert_plan(Plan(id="p1", objective="Build API"))
		db.insert_work_unit(WorkUnit(id="wu1", plan_id="p1", title="Unit 1"))
		db.insert_work_unit(WorkUnit(id="wu2", plan_id="p1", title="Unit 2"))

	def test_insert_and_get_for_mission(self, db: Database) -> None:
		self._setup(db)
		review = UnitReview(
			id="r1", work_unit_id="wu1", mission_id="m1", epoch_id="ep1",
			alignment_score=7, approach_score=8, test_score=6,
			avg_score=7.0, rationale="Clean implementation",
			model="sonnet", cost_usd=0.05,
		)
		db.insert_unit_review(review)
		reviews = db.get_unit_reviews_for_mission("m1")
		assert len(reviews) == 1
		assert reviews[0].alignment_score == 7
		assert reviews[0].approach_score == 8
		assert reviews[0].test_score == 6
		assert reviews[0].avg_score == 7.0
		assert reviews[0].rationale == "Clean implementation"

	def test_get_for_unit(self, db: Database) -> None:
		self._setup(db)
		db.insert_unit_review(UnitReview(
			id="r1", work_unit_id="wu1", mission_id="m1",
			alignment_score=5, approach_score=6, test_score=4, avg_score=5.0,
		))
		result = db.get_unit_review_for_unit("wu1")
		assert result is not None
		assert result.alignment_score == 5

	def test_get_for_unit_nonexistent(self, db: Database) -> None:
		self._setup(db)
		assert db.get_unit_review_for_unit("missing") is None

	def test_multiple_reviews_for_mission(self, db: Database) -> None:
		self._setup(db)
		db.insert_unit_review(UnitReview(
			id="r1", work_unit_id="wu1", mission_id="m1",
			timestamp="2025-01-01T00:00:00",
		))
		db.insert_unit_review(UnitReview(
			id="r2", work_unit_id="wu2", mission_id="m1",
			timestamp="2025-01-02T00:00:00",
		))
		reviews = db.get_unit_reviews_for_mission("m1")
		assert len(reviews) == 2
		assert reviews[0].id == "r1"  # ordered by timestamp ASC


class TestTrajectoryRatings:
	def _setup(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="Build API"))

	def test_insert_and_get(self, db: Database) -> None:
		self._setup(db)
		rating = TrajectoryRating(
			id="tr1", mission_id="m1", rating=8,
			feedback="Great mission, ambitious scope",
		)
		db.insert_trajectory_rating(rating)
		ratings = db.get_trajectory_ratings_for_mission("m1")
		assert len(ratings) == 1
		assert ratings[0].rating == 8
		assert ratings[0].feedback == "Great mission, ambitious scope"

	def test_multiple_ratings(self, db: Database) -> None:
		self._setup(db)
		db.insert_trajectory_rating(TrajectoryRating(
			id="tr1", mission_id="m1", rating=7,
			timestamp="2025-01-01T00:00:00",
		))
		db.insert_trajectory_rating(TrajectoryRating(
			id="tr2", mission_id="m1", rating=9,
			timestamp="2025-01-02T00:00:00",
		))
		ratings = db.get_trajectory_ratings_for_mission("m1")
		assert len(ratings) == 2
		assert ratings[0].id == "tr2"  # ordered by timestamp DESC (most recent first)

	def test_empty_ratings(self, db: Database) -> None:
		self._setup(db)
		assert db.get_trajectory_ratings_for_mission("m1") == []


class TestDecompositionGrades:
	def _setup(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="Build API"))

	def test_insert_and_get(self, db: Database) -> None:
		self._setup(db)
		grade = DecompositionGrade(
			id="dg1", plan_id="p1", epoch_id="ep1", mission_id="m1",
			avg_review_score=7.2, retry_rate=0.2, overlap_rate=0.1,
			completion_rate=1.0, composite_score=0.72, unit_count=5,
		)
		db.insert_decomposition_grade(grade)
		grades = db.get_decomposition_grades_for_mission("m1")
		assert len(grades) == 1
		assert grades[0].avg_review_score == 7.2
		assert grades[0].retry_rate == 0.2
		assert grades[0].composite_score == 0.72
		assert grades[0].unit_count == 5

	def test_multiple_grades(self, db: Database) -> None:
		self._setup(db)
		db.insert_decomposition_grade(DecompositionGrade(
			id="dg1", plan_id="p1", epoch_id="ep1", mission_id="m1",
			timestamp="2025-01-01T00:00:00", composite_score=0.6,
		))
		db.insert_decomposition_grade(DecompositionGrade(
			id="dg2", plan_id="p2", epoch_id="ep2", mission_id="m1",
			timestamp="2025-01-02T00:00:00", composite_score=0.8,
		))
		grades = db.get_decomposition_grades_for_mission("m1")
		assert len(grades) == 2
		assert grades[0].id == "dg2"  # ordered by timestamp DESC

	def test_empty_grades(self, db: Database) -> None:
		self._setup(db)
		assert db.get_decomposition_grades_for_mission("m1") == []


class TestUpdateDegradationLevel:
	def test_updates_running_mission(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="Test", status="running"))
		db.update_degradation_level("REDUCED")
		row = db.conn.execute("SELECT degradation_level FROM missions WHERE id='m1'").fetchone()
		assert row["degradation_level"] == "REDUCED"

	def test_skips_non_running_missions(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="Test", status="completed"))
		db.update_degradation_level("REDUCED")
		row = db.conn.execute("SELECT degradation_level FROM missions WHERE id='m1'").fetchone()
		assert row["degradation_level"] == "FULL_CAPACITY"


class TestResetOrphanedUnits:
	def _setup(self, db: Database) -> None:
		db.insert_plan(Plan(id="p1", objective="test"))

	def test_resets_running_and_pending(self, db: Database) -> None:
		self._setup(db)
		db.insert_work_unit(WorkUnit(id="a", plan_id="p1", status="running"))
		db.insert_work_unit(WorkUnit(id="b", plan_id="p1", status="pending"))
		db.insert_work_unit(WorkUnit(id="c", plan_id="p1", status="completed"))
		count = db.reset_orphaned_units()
		assert count == 2
		assert db.get_work_unit("a").status == "failed"
		assert db.get_work_unit("b").status == "failed"
		assert db.get_work_unit("c").status == "completed"

	def test_returns_zero_when_none(self, db: Database) -> None:
		self._setup(db)
		db.insert_work_unit(WorkUnit(id="a", plan_id="p1", status="completed"))
		assert db.reset_orphaned_units() == 0


class TestGetRunningUnits:
	def _setup(self, db: Database) -> None:
		db.insert_plan(Plan(id="p1", objective="test"))

	def test_returns_running_units(self, db: Database) -> None:
		self._setup(db)
		db.insert_work_unit(WorkUnit(id="a", plan_id="p1", status="running", title="A"))
		db.insert_work_unit(WorkUnit(id="b", plan_id="p1", status="running", title="B"))
		db.insert_work_unit(WorkUnit(id="c", plan_id="p1", status="completed", title="C"))
		units = db.get_running_units()
		assert len(units) == 2
		ids = {u.id for u in units}
		assert ids == {"a", "b"}

	def test_excludes_by_id(self, db: Database) -> None:
		self._setup(db)
		db.insert_work_unit(WorkUnit(id="a", plan_id="p1", status="running"))
		db.insert_work_unit(WorkUnit(id="b", plan_id="p1", status="running"))
		units = db.get_running_units(exclude_id="a")
		assert len(units) == 1
		assert units[0].id == "b"

	def test_returns_work_unit_objects(self, db: Database) -> None:
		self._setup(db)
		db.insert_work_unit(WorkUnit(id="a", plan_id="p1", status="running", title="Task A"))
		units = db.get_running_units()
		assert len(units) == 1
		assert units[0].title == "Task A"
		assert isinstance(units[0], WorkUnit)
