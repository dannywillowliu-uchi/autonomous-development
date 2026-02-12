"""Tests for SQLite database operations."""

from __future__ import annotations

import pytest

from mission_control.db import Database
from mission_control.models import (
	Decision,
	MergeRequest,
	Plan,
	Session,
	Snapshot,
	TaskRecord,
	Worker,
	WorkUnit,
)


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


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


class TestTasks:
	def test_insert_and_get_open(self, db: Database) -> None:
		t1 = TaskRecord(id="t1", source="test_failure", description="Fix test_foo", priority=2)
		t2 = TaskRecord(id="t2", source="lint", description="Fix ruff E501", priority=3)
		db.insert_task(t1)
		db.insert_task(t2)
		open_tasks = db.get_open_tasks()
		assert len(open_tasks) == 2
		assert open_tasks[0].priority <= open_tasks[1].priority

	def test_update_task_status(self, db: Database) -> None:
		t = TaskRecord(id="t3", source="todo", description="Implement feature")
		db.insert_task(t)
		t.status = "completed"
		t.resolved_at = "2025-01-01T00:00:00"
		db.update_task(t)
		open_tasks = db.get_open_tasks()
		assert len(open_tasks) == 0

	def test_priority_ordering(self, db: Database) -> None:
		db.insert_task(TaskRecord(id="low", source="todo", priority=5))
		db.insert_task(TaskRecord(id="high", source="test_failure", priority=1))
		db.insert_task(TaskRecord(id="mid", source="lint", priority=3))
		tasks = db.get_open_tasks()
		assert [t.id for t in tasks] == ["high", "mid", "low"]


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
		"""Stale units should be released back to pending."""
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

	def test_per_unit_overrides_roundtrip(self, db: Database) -> None:
		"""timeout and verification_command persist through insert/get."""
		self._make_plan(db)
		wu = WorkUnit(
			id="wu_ov", plan_id="plan1", title="Override test",
			timeout=600, verification_command="make test",
		)
		db.insert_work_unit(wu)
		result = db.get_work_unit("wu_ov")
		assert result is not None
		assert result.timeout == 600
		assert result.verification_command == "make test"

	def test_per_unit_overrides_default_none(self, db: Database) -> None:
		"""Without overrides, timeout and verification_command are None."""
		self._make_plan(db)
		wu = WorkUnit(id="wu_def", plan_id="plan1", title="Default test")
		db.insert_work_unit(wu)
		result = db.get_work_unit("wu_def")
		assert result is not None
		assert result.timeout is None
		assert result.verification_command is None

	def test_per_unit_overrides_update(self, db: Database) -> None:
		"""Overrides can be set via update."""
		self._make_plan(db)
		wu = WorkUnit(id="wu_upd", plan_id="plan1", title="Update test")
		db.insert_work_unit(wu)
		wu.timeout = 120
		wu.verification_command = "pytest tests/specific.py"
		db.update_work_unit(wu)
		result = db.get_work_unit("wu_upd")
		assert result is not None
		assert result.timeout == 120
		assert result.verification_command == "pytest tests/specific.py"


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
