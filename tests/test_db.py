"""Tests for SQLite database operations."""

from __future__ import annotations

import pytest

from mission_control.db import Database
from mission_control.models import Decision, Session, Snapshot, TaskRecord


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
