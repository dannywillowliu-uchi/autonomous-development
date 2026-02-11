"""SQLite database operations for mission-control state."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from mission_control.models import Decision, Session, Snapshot, TaskRecord

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
	id TEXT PRIMARY KEY,
	target_name TEXT NOT NULL,
	task_description TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'pending',
	branch_name TEXT NOT NULL DEFAULT '',
	started_at TEXT NOT NULL,
	finished_at TEXT,
	exit_code INTEGER,
	commit_hash TEXT,
	cost_usd REAL,
	output_summary TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS snapshots (
	id TEXT PRIMARY KEY,
	session_id TEXT,
	taken_at TEXT NOT NULL,
	test_total INTEGER NOT NULL DEFAULT 0,
	test_passed INTEGER NOT NULL DEFAULT 0,
	test_failed INTEGER NOT NULL DEFAULT 0,
	lint_errors INTEGER NOT NULL DEFAULT 0,
	type_errors INTEGER NOT NULL DEFAULT 0,
	security_findings INTEGER NOT NULL DEFAULT 0,
	raw_output TEXT NOT NULL DEFAULT '',
	FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS tasks (
	id TEXT PRIMARY KEY,
	source TEXT NOT NULL DEFAULT '',
	description TEXT NOT NULL DEFAULT '',
	priority INTEGER NOT NULL DEFAULT 7,
	status TEXT NOT NULL DEFAULT 'discovered',
	session_id TEXT,
	created_at TEXT NOT NULL,
	resolved_at TEXT,
	FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS decisions (
	id TEXT PRIMARY KEY,
	session_id TEXT NOT NULL,
	decision TEXT NOT NULL DEFAULT '',
	rationale TEXT NOT NULL DEFAULT '',
	timestamp TEXT NOT NULL,
	FOREIGN KEY (session_id) REFERENCES sessions(id)
);
"""


class Database:
	"""SQLite database for mission-control state."""

	def __init__(self, path: str | Path = ":memory:") -> None:
		db_path = str(path)
		self.conn = sqlite3.connect(db_path)
		self.conn.row_factory = sqlite3.Row
		if db_path != ":memory:":
			self.conn.execute("PRAGMA journal_mode=WAL")
		self.conn.execute("PRAGMA foreign_keys=ON")
		self._create_tables()

	def _create_tables(self) -> None:
		self.conn.executescript(SCHEMA_SQL)

	def close(self) -> None:
		self.conn.close()

	# -- Sessions --

	def insert_session(self, session: Session) -> None:
		self.conn.execute(
			"""INSERT INTO sessions
			(id, target_name, task_description, status, branch_name,
			 started_at, finished_at, exit_code, commit_hash, cost_usd, output_summary)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				session.id, session.target_name, session.task_description,
				session.status, session.branch_name, session.started_at,
				session.finished_at, session.exit_code, session.commit_hash,
				session.cost_usd, session.output_summary,
			),
		)
		self.conn.commit()

	def update_session(self, session: Session) -> None:
		self.conn.execute(
			"""UPDATE sessions SET
			target_name=?, task_description=?, status=?, branch_name=?,
			started_at=?, finished_at=?, exit_code=?, commit_hash=?,
			cost_usd=?, output_summary=?
			WHERE id=?""",
			(
				session.target_name, session.task_description, session.status,
				session.branch_name, session.started_at, session.finished_at,
				session.exit_code, session.commit_hash, session.cost_usd,
				session.output_summary, session.id,
			),
		)
		self.conn.commit()

	def get_session(self, session_id: str) -> Session | None:
		row = self.conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
		if row is None:
			return None
		return self._row_to_session(row)

	def get_recent_sessions(self, limit: int = 10) -> list[Session]:
		rows = self.conn.execute(
			"SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
		).fetchall()
		return [self._row_to_session(r) for r in rows]

	@staticmethod
	def _row_to_session(row: sqlite3.Row) -> Session:
		return Session(
			id=row["id"],
			target_name=row["target_name"],
			task_description=row["task_description"],
			status=row["status"],
			branch_name=row["branch_name"],
			started_at=row["started_at"],
			finished_at=row["finished_at"],
			exit_code=row["exit_code"],
			commit_hash=row["commit_hash"],
			cost_usd=row["cost_usd"],
			output_summary=row["output_summary"],
		)

	# -- Snapshots --

	def insert_snapshot(self, snapshot: Snapshot) -> None:
		self.conn.execute(
			"""INSERT INTO snapshots
			(id, session_id, taken_at, test_total, test_passed, test_failed,
			 lint_errors, type_errors, security_findings, raw_output)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				snapshot.id, snapshot.session_id, snapshot.taken_at,
				snapshot.test_total, snapshot.test_passed, snapshot.test_failed,
				snapshot.lint_errors, snapshot.type_errors, snapshot.security_findings,
				snapshot.raw_output,
			),
		)
		self.conn.commit()

	def get_latest_snapshot(self) -> Snapshot | None:
		row = self.conn.execute(
			"SELECT * FROM snapshots ORDER BY taken_at DESC LIMIT 1"
		).fetchone()
		if row is None:
			return None
		return self._row_to_snapshot(row)

	@staticmethod
	def _row_to_snapshot(row: sqlite3.Row) -> Snapshot:
		return Snapshot(
			id=row["id"],
			session_id=row["session_id"],
			taken_at=row["taken_at"],
			test_total=row["test_total"],
			test_passed=row["test_passed"],
			test_failed=row["test_failed"],
			lint_errors=row["lint_errors"],
			type_errors=row["type_errors"],
			security_findings=row["security_findings"],
			raw_output=row["raw_output"],
		)

	# -- Tasks --

	def insert_task(self, task: TaskRecord) -> None:
		self.conn.execute(
			"""INSERT INTO tasks
			(id, source, description, priority, status, session_id, created_at, resolved_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				task.id, task.source, task.description, task.priority,
				task.status, task.session_id, task.created_at, task.resolved_at,
			),
		)
		self.conn.commit()

	def update_task(self, task: TaskRecord) -> None:
		self.conn.execute(
			"""UPDATE tasks SET
			source=?, description=?, priority=?, status=?,
			session_id=?, created_at=?, resolved_at=?
			WHERE id=?""",
			(
				task.source, task.description, task.priority, task.status,
				task.session_id, task.created_at, task.resolved_at, task.id,
			),
		)
		self.conn.commit()

	def get_open_tasks(self, limit: int = 20) -> list[TaskRecord]:
		rows = self.conn.execute(
			"SELECT * FROM tasks WHERE status IN ('discovered', 'assigned') "
			"ORDER BY priority ASC, created_at ASC LIMIT ?",
			(limit,),
		).fetchall()
		return [self._row_to_task(r) for r in rows]

	@staticmethod
	def _row_to_task(row: sqlite3.Row) -> TaskRecord:
		return TaskRecord(
			id=row["id"],
			source=row["source"],
			description=row["description"],
			priority=row["priority"],
			status=row["status"],
			session_id=row["session_id"],
			created_at=row["created_at"],
			resolved_at=row["resolved_at"],
		)

	# -- Decisions --

	def insert_decision(self, decision: Decision) -> None:
		self.conn.execute(
			"""INSERT INTO decisions (id, session_id, decision, rationale, timestamp)
			VALUES (?, ?, ?, ?, ?)""",
			(decision.id, decision.session_id, decision.decision, decision.rationale, decision.timestamp),
		)
		self.conn.commit()

	def get_recent_decisions(self, limit: int = 5) -> list[Decision]:
		rows = self.conn.execute(
			"SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
		).fetchall()
		return [self._row_to_decision(r) for r in rows]

	@staticmethod
	def _row_to_decision(row: sqlite3.Row) -> Decision:
		return Decision(
			id=row["id"],
			session_id=row["session_id"],
			decision=row["decision"],
			rationale=row["rationale"],
			timestamp=row["timestamp"],
		)

	# -- Bulk persist --

	def persist_session_result(
		self,
		session: Session,
		before: Snapshot,
		after: Snapshot,
		decisions: Sequence[Decision] | None = None,
	) -> None:
		"""Persist a complete session result atomically."""
		self.insert_session(session)
		before.session_id = session.id
		after.session_id = session.id
		self.insert_snapshot(before)
		self.insert_snapshot(after)
		if decisions:
			for d in decisions:
				d.session_id = session.id
				self.insert_decision(d)
