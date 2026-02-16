"""SQLite database operations for mission-control state."""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Sequence

from mission_control.models import (
	BacklogItem,
	Decision,
	DiscoveryItem,
	DiscoveryResult,
	Epoch,
	Experience,
	ExperimentResult,
	Handoff,
	MergeRequest,
	Mission,
	Plan,
	PlanNode,
	Reflection,
	Reward,
	Round,
	Session,
	Signal,
	Snapshot,
	StrategicContext,
	TaskRecord,
	UnitEvent,
	Worker,
	WorkUnit,
)

logger = logging.getLogger(__name__)

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

CREATE TABLE IF NOT EXISTS plans (
	id TEXT PRIMARY KEY,
	objective TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'pending',
	created_at TEXT NOT NULL,
	finished_at TEXT,
	raw_planner_output TEXT NOT NULL DEFAULT '',
	total_units INTEGER NOT NULL DEFAULT 0,
	completed_units INTEGER NOT NULL DEFAULT 0,
	failed_units INTEGER NOT NULL DEFAULT 0,
	round_id TEXT,
	root_node_id TEXT,
	FOREIGN KEY (round_id) REFERENCES rounds(id)
);

CREATE TABLE IF NOT EXISTS work_units (
	id TEXT PRIMARY KEY,
	plan_id TEXT NOT NULL,
	title TEXT NOT NULL DEFAULT '',
	description TEXT NOT NULL DEFAULT '',
	files_hint TEXT NOT NULL DEFAULT '',
	verification_hint TEXT NOT NULL DEFAULT '',
	priority INTEGER NOT NULL DEFAULT 1,
	status TEXT NOT NULL DEFAULT 'pending',
	worker_id TEXT,
	round_id TEXT,
	plan_node_id TEXT,
	handoff_id TEXT,
	depends_on TEXT NOT NULL DEFAULT '',
	branch_name TEXT NOT NULL DEFAULT '',
	claimed_at TEXT,
	heartbeat_at TEXT,
	started_at TEXT,
	finished_at TEXT,
	exit_code INTEGER,
	commit_hash TEXT,
	output_summary TEXT NOT NULL DEFAULT '',
	attempt INTEGER NOT NULL DEFAULT 0,
	max_attempts INTEGER NOT NULL DEFAULT 3,
	timeout INTEGER,
	verification_command TEXT,
	experiment_mode INTEGER NOT NULL DEFAULT 0,
	FOREIGN KEY (plan_id) REFERENCES plans(id)
);

CREATE INDEX IF NOT EXISTS idx_work_units_status ON work_units(status, priority);

CREATE TABLE IF NOT EXISTS workers (
	id TEXT PRIMARY KEY,
	workspace_path TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'idle',
	current_unit_id TEXT,
	pid INTEGER,
	started_at TEXT NOT NULL,
	last_heartbeat TEXT NOT NULL,
	units_completed INTEGER NOT NULL DEFAULT 0,
	units_failed INTEGER NOT NULL DEFAULT 0,
	total_cost_usd REAL NOT NULL DEFAULT 0.0,
	backend_type TEXT NOT NULL DEFAULT 'local',
	backend_metadata TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS merge_requests (
	id TEXT PRIMARY KEY,
	work_unit_id TEXT NOT NULL,
	worker_id TEXT NOT NULL,
	branch_name TEXT NOT NULL DEFAULT '',
	commit_hash TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'pending',
	position INTEGER NOT NULL DEFAULT 0,
	created_at TEXT NOT NULL,
	verified_at TEXT,
	merged_at TEXT,
	rejection_reason TEXT NOT NULL DEFAULT '',
	rebase_attempts INTEGER NOT NULL DEFAULT 0,
	FOREIGN KEY (work_unit_id) REFERENCES work_units(id)
);

CREATE INDEX IF NOT EXISTS idx_merge_requests_status ON merge_requests(status, position);

CREATE TABLE IF NOT EXISTS missions (
	id TEXT PRIMARY KEY,
	objective TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'pending',
	started_at TEXT NOT NULL,
	finished_at TEXT,
	total_rounds INTEGER NOT NULL DEFAULT 0,
	total_cost_usd REAL NOT NULL DEFAULT 0.0,
	final_score REAL NOT NULL DEFAULT 0.0,
	stopped_reason TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS rounds (
	id TEXT PRIMARY KEY,
	mission_id TEXT NOT NULL,
	number INTEGER NOT NULL DEFAULT 0,
	status TEXT NOT NULL DEFAULT 'pending',
	started_at TEXT NOT NULL,
	finished_at TEXT,
	snapshot_hash TEXT NOT NULL DEFAULT '',
	plan_id TEXT,
	objective_score REAL NOT NULL DEFAULT 0.0,
	objective_met INTEGER NOT NULL DEFAULT 0,
	total_units INTEGER NOT NULL DEFAULT 0,
	completed_units INTEGER NOT NULL DEFAULT 0,
	failed_units INTEGER NOT NULL DEFAULT 0,
	cost_usd REAL NOT NULL DEFAULT 0.0,
	discoveries TEXT NOT NULL DEFAULT '',
	FOREIGN KEY (mission_id) REFERENCES missions(id)
);

CREATE INDEX IF NOT EXISTS idx_rounds_mission ON rounds(mission_id, number);

CREATE TABLE IF NOT EXISTS plan_nodes (
	id TEXT PRIMARY KEY,
	plan_id TEXT NOT NULL,
	parent_id TEXT,
	depth INTEGER NOT NULL DEFAULT 0,
	scope TEXT NOT NULL DEFAULT '',
	strategy TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'pending',
	node_type TEXT NOT NULL DEFAULT 'branch',
	work_unit_id TEXT,
	children_ids TEXT NOT NULL DEFAULT '',
	FOREIGN KEY (plan_id) REFERENCES plans(id)
);

CREATE INDEX IF NOT EXISTS idx_plan_nodes_plan ON plan_nodes(plan_id);

CREATE TABLE IF NOT EXISTS handoffs (
	id TEXT PRIMARY KEY,
	work_unit_id TEXT NOT NULL,
	round_id TEXT NOT NULL DEFAULT '',
	epoch_id TEXT,
	status TEXT NOT NULL DEFAULT '',
	commits TEXT NOT NULL DEFAULT '',
	summary TEXT NOT NULL DEFAULT '',
	discoveries TEXT NOT NULL DEFAULT '',
	concerns TEXT NOT NULL DEFAULT '',
	files_changed TEXT NOT NULL DEFAULT '',
	FOREIGN KEY (work_unit_id) REFERENCES work_units(id)
);

CREATE INDEX IF NOT EXISTS idx_handoffs_round ON handoffs(round_id);
CREATE INDEX IF NOT EXISTS idx_handoffs_epoch ON handoffs(epoch_id);

CREATE TABLE IF NOT EXISTS reflections (
	id TEXT PRIMARY KEY,
	mission_id TEXT NOT NULL,
	round_id TEXT NOT NULL,
	round_number INTEGER NOT NULL,
	timestamp TEXT NOT NULL,
	tests_before INTEGER DEFAULT 0,
	tests_after INTEGER DEFAULT 0,
	tests_delta INTEGER DEFAULT 0,
	lint_delta INTEGER DEFAULT 0,
	type_delta INTEGER DEFAULT 0,
	objective_score REAL DEFAULT 0.0,
	score_delta REAL DEFAULT 0.0,
	units_planned INTEGER DEFAULT 0,
	units_completed INTEGER DEFAULT 0,
	units_failed INTEGER DEFAULT 0,
	completion_rate REAL DEFAULT 0.0,
	plan_depth INTEGER DEFAULT 0,
	plan_strategy TEXT DEFAULT '',
	fixup_promoted INTEGER DEFAULT 0,
	fixup_attempts INTEGER DEFAULT 0,
	merge_conflicts INTEGER DEFAULT 0,
	discoveries_count INTEGER DEFAULT 0,
	FOREIGN KEY (mission_id) REFERENCES missions(id),
	FOREIGN KEY (round_id) REFERENCES rounds(id)
);

CREATE INDEX IF NOT EXISTS idx_reflections_mission ON reflections(mission_id, round_number);

CREATE TABLE IF NOT EXISTS rewards (
	id TEXT PRIMARY KEY,
	round_id TEXT NOT NULL,
	mission_id TEXT NOT NULL,
	timestamp TEXT NOT NULL,
	reward REAL DEFAULT 0.0,
	verification_improvement REAL DEFAULT 0.0,
	completion_rate REAL DEFAULT 0.0,
	score_progress REAL DEFAULT 0.0,
	fixup_efficiency REAL DEFAULT 0.0,
	no_regression REAL DEFAULT 0.0,
	FOREIGN KEY (round_id) REFERENCES rounds(id),
	FOREIGN KEY (mission_id) REFERENCES missions(id)
);

CREATE TABLE IF NOT EXISTS experiences (
	id TEXT PRIMARY KEY,
	round_id TEXT NOT NULL,
	work_unit_id TEXT NOT NULL,
	timestamp TEXT NOT NULL,
	title TEXT DEFAULT '',
	scope TEXT DEFAULT '',
	files_hint TEXT DEFAULT '',
	status TEXT DEFAULT '',
	summary TEXT DEFAULT '',
	files_changed TEXT DEFAULT '',
	discoveries TEXT DEFAULT '',
	concerns TEXT DEFAULT '',
	reward REAL DEFAULT 0.0,
	FOREIGN KEY (round_id) REFERENCES rounds(id),
	FOREIGN KEY (work_unit_id) REFERENCES work_units(id)
);

CREATE INDEX IF NOT EXISTS idx_experiences_reward ON experiences(reward DESC);

CREATE TABLE IF NOT EXISTS signals (
	id TEXT PRIMARY KEY,
	mission_id TEXT NOT NULL,
	signal_type TEXT NOT NULL,
	payload TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'pending',
	created_at TEXT NOT NULL,
	acknowledged_at TEXT,
	FOREIGN KEY (mission_id) REFERENCES missions(id)
);

CREATE INDEX IF NOT EXISTS idx_signals_mission ON signals(mission_id, status);

CREATE TABLE IF NOT EXISTS epochs (
	id TEXT PRIMARY KEY,
	mission_id TEXT NOT NULL,
	number INTEGER NOT NULL DEFAULT 0,
	started_at TEXT NOT NULL,
	finished_at TEXT,
	units_planned INTEGER NOT NULL DEFAULT 0,
	units_completed INTEGER NOT NULL DEFAULT 0,
	units_failed INTEGER NOT NULL DEFAULT 0,
	score_at_start REAL NOT NULL DEFAULT 0.0,
	score_at_end REAL NOT NULL DEFAULT 0.0,
	FOREIGN KEY (mission_id) REFERENCES missions(id)
);

CREATE INDEX IF NOT EXISTS idx_epochs_mission ON epochs(mission_id, number);

CREATE TABLE IF NOT EXISTS unit_events (
	id TEXT PRIMARY KEY,
	mission_id TEXT NOT NULL,
	epoch_id TEXT NOT NULL,
	work_unit_id TEXT NOT NULL,
	event_type TEXT NOT NULL DEFAULT '',
	timestamp TEXT NOT NULL,
	score_after REAL NOT NULL DEFAULT 0.0,
	details TEXT NOT NULL DEFAULT '',
	FOREIGN KEY (mission_id) REFERENCES missions(id),
	FOREIGN KEY (epoch_id) REFERENCES epochs(id),
	FOREIGN KEY (work_unit_id) REFERENCES work_units(id)
);

CREATE INDEX IF NOT EXISTS idx_unit_events_mission ON unit_events(mission_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_unit_events_epoch ON unit_events(epoch_id);

CREATE TABLE IF NOT EXISTS discoveries (
	id TEXT PRIMARY KEY,
	target_path TEXT NOT NULL DEFAULT '',
	timestamp TEXT NOT NULL,
	raw_output TEXT NOT NULL DEFAULT '',
	model TEXT NOT NULL DEFAULT '',
	item_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS discovery_items (
	id TEXT PRIMARY KEY,
	discovery_id TEXT NOT NULL,
	track TEXT NOT NULL DEFAULT '',
	title TEXT NOT NULL DEFAULT '',
	description TEXT NOT NULL DEFAULT '',
	rationale TEXT NOT NULL DEFAULT '',
	files_hint TEXT NOT NULL DEFAULT '',
	impact INTEGER NOT NULL DEFAULT 5,
	effort INTEGER NOT NULL DEFAULT 5,
	priority_score REAL NOT NULL DEFAULT 0.0,
	status TEXT NOT NULL DEFAULT 'proposed',
	FOREIGN KEY (discovery_id) REFERENCES discoveries(id)
);

CREATE INDEX IF NOT EXISTS idx_discovery_items_discovery ON discovery_items(discovery_id);
CREATE INDEX IF NOT EXISTS idx_discovery_items_status ON discovery_items(status, priority_score DESC);

CREATE TABLE IF NOT EXISTS backlog_items (
	id TEXT PRIMARY KEY,
	title TEXT NOT NULL DEFAULT '',
	description TEXT NOT NULL DEFAULT '',
	priority_score REAL NOT NULL DEFAULT 0.0,
	impact INTEGER NOT NULL DEFAULT 5,
	effort INTEGER NOT NULL DEFAULT 5,
	track TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'pending',
	source_mission_id TEXT,
	created_at TEXT NOT NULL,
	updated_at TEXT NOT NULL,
	attempt_count INTEGER NOT NULL DEFAULT 0,
	last_failure_reason TEXT,
	pinned_score REAL,
	depends_on TEXT NOT NULL DEFAULT '',
	tags TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_backlog_items_status ON backlog_items(status, priority_score DESC);

CREATE TABLE IF NOT EXISTS strategic_context (
	id TEXT PRIMARY KEY,
	mission_id TEXT NOT NULL,
	timestamp TEXT NOT NULL,
	what_attempted TEXT NOT NULL DEFAULT '',
	what_worked TEXT NOT NULL DEFAULT '',
	what_failed TEXT NOT NULL DEFAULT '',
	recommended_next TEXT NOT NULL DEFAULT '',
	FOREIGN KEY (mission_id) REFERENCES missions(id)
);

CREATE INDEX IF NOT EXISTS idx_strategic_context_timestamp ON strategic_context(timestamp DESC);

CREATE TABLE IF NOT EXISTS experiment_results (
	id TEXT PRIMARY KEY,
	work_unit_id TEXT NOT NULL,
	epoch_id TEXT,
	mission_id TEXT NOT NULL,
	timestamp TEXT NOT NULL,
	approach_count INTEGER NOT NULL DEFAULT 2,
	comparison_report TEXT NOT NULL DEFAULT '',
	recommended_approach TEXT NOT NULL DEFAULT '',
	created_at TEXT NOT NULL,
	FOREIGN KEY (work_unit_id) REFERENCES work_units(id),
	FOREIGN KEY (mission_id) REFERENCES missions(id)
);

CREATE INDEX IF NOT EXISTS idx_experiment_results_mission ON experiment_results(mission_id);
CREATE INDEX IF NOT EXISTS idx_experiment_results_unit ON experiment_results(work_unit_id);
"""


class Database:
	"""SQLite database for mission-control state."""

	def __init__(self, path: str | Path = ":memory:") -> None:
		db_path = str(path)
		self.conn = sqlite3.connect(db_path)
		self.conn.row_factory = sqlite3.Row
		logger.debug("Opened database connection: %s", db_path)
		if db_path != ":memory:":
			self.conn.execute("PRAGMA journal_mode=WAL")
			self.conn.execute("PRAGMA busy_timeout=5000")
			logger.debug("WAL mode activated for %s", db_path)
		self.conn.execute("PRAGMA foreign_keys=ON")
		self._lock = asyncio.Lock()
		self._create_tables()

	@staticmethod
	def _validate_identifier(name: str) -> None:
		"""Validate a SQL identifier to prevent injection in dynamic ALTER TABLE statements."""
		if not name or len(name) > 64 or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
			raise ValueError(f"Invalid SQL identifier: {name!r}")

	def _create_tables(self) -> None:
		self.conn.executescript(SCHEMA_SQL)
		self._migrate_epoch_columns()
		self._migrate_token_columns()
		self._migrate_unit_type_column()
		self._migrate_backlog_table()
		self._migrate_strategic_context()
		self._migrate_mission_strategy_columns()
		self._migrate_experiment_mode_column()

	def _migrate_epoch_columns(self) -> None:
		"""Add epoch_id columns to existing tables (idempotent)."""
		migrations = [
			("work_units", "epoch_id", "TEXT"),
			("handoffs", "epoch_id", "TEXT"),
			("reflections", "epoch_id", "TEXT"),
			("rewards", "epoch_id", "TEXT"),
			("experiences", "epoch_id", "TEXT"),
		]
		for table, column, col_type in migrations:
			self._validate_identifier(table)
			self._validate_identifier(column)
			try:
				self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")  # noqa: S608
				logger.debug("Migration: added column %s.%s", table, column)
			except sqlite3.OperationalError as exc:
				if "duplicate column name" in str(exc):
					pass
				else:
					logger.warning("Migration failed for %s.%s: %s", table, column, exc)
					raise

	def _migrate_token_columns(self) -> None:
		"""Add token tracking columns to existing tables (idempotent)."""
		migrations = [
			("work_units", "input_tokens", "INTEGER DEFAULT 0"),
			("work_units", "output_tokens", "INTEGER DEFAULT 0"),
			("work_units", "cost_usd", "REAL DEFAULT 0.0"),
			("unit_events", "input_tokens", "INTEGER DEFAULT 0"),
			("unit_events", "output_tokens", "INTEGER DEFAULT 0"),
			("work_units", "unit_type", "TEXT DEFAULT 'implementation'"),
		]
		for table, column, col_type in migrations:
			self._validate_identifier(table)
			self._validate_identifier(column)
			try:
				self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")  # noqa: S608
				logger.debug("Migration: added column %s.%s", table, column)
			except sqlite3.OperationalError as exc:
				if "duplicate column name" in str(exc):
					pass
				else:
					logger.warning("Migration failed for %s.%s: %s", table, column, exc)
					raise

	def _migrate_unit_type_column(self) -> None:
		"""Add unit_type column to work_units table (idempotent)."""
		try:
			self.conn.execute("ALTER TABLE work_units ADD COLUMN unit_type TEXT NOT NULL DEFAULT 'implementation'")
			logger.debug("Migration: added column work_units.unit_type")
		except sqlite3.OperationalError as exc:
			if "duplicate column name" in str(exc):
				pass
			else:
				logger.warning("Migration failed for work_units.unit_type: %s", exc)
				raise

	def _migrate_backlog_table(self) -> None:
		"""Ensure backlog_items table exists (forward-compat for existing DBs)."""
		logger.debug("Migration: ensuring backlog_items table exists")

	def _migrate_strategic_context(self) -> None:
		"""Ensure strategic_context table exists (forward-compat for existing DBs)."""
		logger.debug("Migration: ensuring strategic_context table exists")

	def _migrate_mission_strategy_columns(self) -> None:
		"""Add ambition_score, next_objective, proposed_by_strategist to missions (idempotent)."""
		migrations = [
			("missions", "ambition_score", "INTEGER DEFAULT 0"),
			("missions", "next_objective", "TEXT DEFAULT ''"),
			("missions", "proposed_by_strategist", "INTEGER DEFAULT 0"),
		]
		for table, column, col_type in migrations:
			self._validate_identifier(table)
			self._validate_identifier(column)
			try:
				self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")  # noqa: S608
				logger.debug("Migration: added column %s.%s", table, column)
			except sqlite3.OperationalError as exc:
				if "duplicate column name" in str(exc):
					pass
				else:
					logger.warning("Migration failed for %s.%s: %s", table, column, exc)
					raise

	def _migrate_experiment_mode_column(self) -> None:
		"""Add experiment_mode column to work_units table (idempotent)."""
		try:
			self.conn.execute("ALTER TABLE work_units ADD COLUMN experiment_mode INTEGER NOT NULL DEFAULT 0")
			logger.debug("Migration: added column work_units.experiment_mode")
		except sqlite3.OperationalError as exc:
			if "duplicate column name" in str(exc):
				pass
			else:
				logger.warning("Migration failed for work_units.experiment_mode: %s", exc)
				raise

	def close(self) -> None:
		logger.debug("Closing database connection")
		self.conn.close()

	def __enter__(self) -> Database:
		return self

	def __exit__(self, *args: object) -> None:
		self.close()

	@contextmanager
	def transaction(self) -> Generator[sqlite3.Connection, None, None]:
		"""Context manager for explicit transactions.

		Commits on success, rolls back on exception.
		"""
		try:
			yield self.conn
		except Exception:
			self.conn.rollback()
			raise
		else:
			self.conn.commit()

	async def locked_call(self, fn: str, *args: Any, **kwargs: Any) -> Any:
		"""Call a Database method while holding the asyncio lock.

		Use this to serialize concurrent access from multiple asyncio tasks.
		"""
		async with self._lock:
			return getattr(self, fn)(*args, **kwargs)

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
		"""Persist a complete session result atomically.

		Uses a single transaction so all inserts either succeed together
		or roll back together on failure.
		"""
		before.session_id = session.id
		after.session_id = session.id
		try:
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
			for snap in (before, after):
				self.conn.execute(
					"""INSERT INTO snapshots
					(id, session_id, taken_at, test_total, test_passed, test_failed,
					 lint_errors, type_errors, security_findings, raw_output)
					VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
					(
						snap.id, snap.session_id, snap.taken_at,
						snap.test_total, snap.test_passed, snap.test_failed,
						snap.lint_errors, snap.type_errors, snap.security_findings,
						snap.raw_output,
					),
				)
			if decisions:
				for d in decisions:
					d.session_id = session.id
					self.conn.execute(
						"""INSERT INTO decisions (id, session_id, decision, rationale, timestamp)
						VALUES (?, ?, ?, ?, ?)""",
						(d.id, d.session_id, d.decision, d.rationale, d.timestamp),
					)
			self.conn.commit()
		except sqlite3.Error:
			logger.error("persist_session_result failed for session %s, rolling back", session.id, exc_info=True)
			self.conn.rollback()
			raise

	# -- Plans --

	def insert_plan(self, plan: Plan) -> None:
		self.conn.execute(
			"""INSERT INTO plans
			(id, objective, status, created_at, finished_at,
			 raw_planner_output, total_units, completed_units, failed_units,
			 round_id, root_node_id)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				plan.id, plan.objective, plan.status, plan.created_at,
				plan.finished_at, plan.raw_planner_output,
				plan.total_units, plan.completed_units, plan.failed_units,
				plan.round_id, plan.root_node_id,
			),
		)
		self.conn.commit()

	def update_plan(self, plan: Plan) -> None:
		self.conn.execute(
			"""UPDATE plans SET
			objective=?, status=?, finished_at=?,
			raw_planner_output=?, total_units=?,
			completed_units=?, failed_units=?,
			round_id=?, root_node_id=?
			WHERE id=?""",
			(
				plan.objective, plan.status, plan.finished_at,
				plan.raw_planner_output, plan.total_units,
				plan.completed_units, plan.failed_units,
				plan.round_id, plan.root_node_id, plan.id,
			),
		)
		self.conn.commit()

	def get_plan(self, plan_id: str) -> Plan | None:
		row = self.conn.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone()
		if row is None:
			return None
		return self._row_to_plan(row)

	@staticmethod
	def _row_to_plan(row: sqlite3.Row) -> Plan:
		return Plan(
			id=row["id"],
			objective=row["objective"],
			status=row["status"],
			created_at=row["created_at"],
			finished_at=row["finished_at"],
			raw_planner_output=row["raw_planner_output"],
			total_units=row["total_units"],
			completed_units=row["completed_units"],
			failed_units=row["failed_units"],
			round_id=row["round_id"],
			root_node_id=row["root_node_id"],
		)

	# -- Work Units --

	def insert_work_unit(self, unit: WorkUnit) -> None:
		self.conn.execute(
			"""INSERT INTO work_units
			(id, plan_id, title, description, files_hint, verification_hint,
			 priority, status, worker_id, round_id, plan_node_id, handoff_id,
			 depends_on, branch_name,
			 claimed_at, heartbeat_at, started_at, finished_at,
			 exit_code, commit_hash, output_summary, attempt, max_attempts,
			 unit_type, timeout, verification_command,
			 epoch_id, input_tokens, output_tokens, cost_usd, experiment_mode)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				unit.id, unit.plan_id, unit.title, unit.description,
				unit.files_hint, unit.verification_hint, unit.priority,
				unit.status, unit.worker_id, unit.round_id, unit.plan_node_id,
				unit.handoff_id, unit.depends_on, unit.branch_name,
				unit.claimed_at, unit.heartbeat_at, unit.started_at,
				unit.finished_at, unit.exit_code, unit.commit_hash,
				unit.output_summary, unit.attempt, unit.max_attempts,
				unit.unit_type, unit.timeout, unit.verification_command,
				unit.epoch_id, unit.input_tokens, unit.output_tokens, unit.cost_usd,
				int(unit.experiment_mode),
			),
		)
		self.conn.commit()
		logger.info("Inserted work_unit %s (status=%s, type=%s)", unit.id, unit.status, unit.unit_type)

	def update_work_unit(self, unit: WorkUnit) -> None:
		self.conn.execute(
			"""UPDATE work_units SET
			plan_id=?, title=?, description=?, files_hint=?,
			verification_hint=?, priority=?, status=?, worker_id=?,
			round_id=?, plan_node_id=?, handoff_id=?,
			depends_on=?, branch_name=?, claimed_at=?, heartbeat_at=?,
			started_at=?, finished_at=?, exit_code=?, commit_hash=?,
			output_summary=?, attempt=?, max_attempts=?,
			unit_type=?, timeout=?, verification_command=?,
			epoch_id=?, input_tokens=?, output_tokens=?, cost_usd=?,
			experiment_mode=?
			WHERE id=?""",
			(
				unit.plan_id, unit.title, unit.description, unit.files_hint,
				unit.verification_hint, unit.priority, unit.status,
				unit.worker_id, unit.round_id, unit.plan_node_id,
				unit.handoff_id, unit.depends_on, unit.branch_name,
				unit.claimed_at, unit.heartbeat_at, unit.started_at,
				unit.finished_at, unit.exit_code, unit.commit_hash,
				unit.output_summary, unit.attempt, unit.max_attempts,
				unit.unit_type, unit.timeout, unit.verification_command,
				unit.epoch_id, unit.input_tokens, unit.output_tokens, unit.cost_usd,
				int(unit.experiment_mode),
				unit.id,
			),
		)
		self.conn.commit()
		logger.info("Updated work_unit %s -> status=%s", unit.id, unit.status)

	def get_work_unit(self, unit_id: str) -> WorkUnit | None:
		row = self.conn.execute("SELECT * FROM work_units WHERE id=?", (unit_id,)).fetchone()
		if row is None:
			return None
		return self._row_to_work_unit(row)

	def get_work_units_for_plan(self, plan_id: str) -> list[WorkUnit]:
		rows = self.conn.execute(
			"SELECT * FROM work_units WHERE plan_id=? ORDER BY priority ASC",
			(plan_id,),
		).fetchall()
		return [self._row_to_work_unit(r) for r in rows]

	def get_work_units_for_mission(self, mission_id: str) -> list[WorkUnit]:
		"""Get all work units for a mission via epoch_id -> epochs.mission_id."""
		rows = self.conn.execute(
			"""SELECT wu.* FROM work_units wu
			JOIN epochs e ON wu.epoch_id = e.id
			WHERE e.mission_id = ?
			ORDER BY wu.started_at ASC NULLS LAST""",
			(mission_id,),
		).fetchall()
		return [self._row_to_work_unit(r) for r in rows]

	def get_active_mission(self) -> Mission | None:
		"""Get the currently running mission, if any."""
		row = self.conn.execute(
			"SELECT * FROM missions WHERE status='running' ORDER BY started_at DESC LIMIT 1"
		).fetchone()
		if row is None:
			return None
		return self._row_to_mission(row)

	def claim_work_unit(self, worker_id: str, now: str | None = None) -> WorkUnit | None:
		"""Atomically claim the next available work unit.

		Only claims units that are pending and whose dependencies are all completed.
		Uses a single UPDATE to prevent race conditions.
		"""
		if now is None:
			from mission_control.models import _now_iso
			now = _now_iso()

		# Find claimable unit: pending, no incomplete deps, ordered by priority
		row = self.conn.execute(
			"""UPDATE work_units SET
				status='claimed', worker_id=?, claimed_at=?, heartbeat_at=?
			WHERE id = (
				SELECT wu.id FROM work_units wu
				WHERE wu.status = 'pending'
				AND NOT EXISTS (
					SELECT 1 FROM work_units dep
					WHERE dep.id IN (
						SELECT value FROM (
							WITH RECURSIVE split(value, rest) AS (
								SELECT '', wu.depends_on || ','
								UNION ALL
								SELECT substr(rest, 1, instr(rest, ',') - 1),
									   substr(rest, instr(rest, ',') + 1)
								FROM split WHERE rest != ''
							)
							SELECT value FROM split WHERE value != ''
						)
					)
					AND dep.status != 'completed'
				)
				ORDER BY wu.priority ASC, wu.id ASC
				LIMIT 1
			)
			RETURNING *""",
			(worker_id, now, now),
		).fetchone()
		self.conn.commit()
		if row is None:
			logger.debug("No claimable work unit for worker %s", worker_id)
			return None
		unit = self._row_to_work_unit(row)
		logger.info("Worker %s claimed work_unit %s", worker_id, unit.id)
		return unit

	def recover_stale_units(self, timeout_seconds: int) -> list[WorkUnit]:
		"""Release work units where heartbeat is stale (worker likely dead)."""
		from mission_control.models import _now_iso
		now = _now_iso()

		rows = self.conn.execute(
			"""UPDATE work_units SET
				status='pending', worker_id=NULL, claimed_at=NULL, heartbeat_at=NULL,
				attempt = attempt + 1
			WHERE status IN ('claimed', 'running')
			AND heartbeat_at IS NOT NULL
			AND (julianday(?) - julianday(heartbeat_at)) * 86400 > ?
			AND attempt < max_attempts
			RETURNING *""",
			(now, timeout_seconds),
		).fetchall()
		self.conn.commit()
		return [self._row_to_work_unit(r) for r in rows]

	def update_heartbeat(self, worker_id: str) -> None:
		"""Update heartbeat for all units claimed by this worker."""
		from mission_control.models import _now_iso
		now = _now_iso()

		self.conn.execute(
			"UPDATE work_units SET heartbeat_at=? WHERE worker_id=? AND status IN ('claimed', 'running')",
			(now, worker_id),
		)
		self.conn.execute(
			"UPDATE workers SET last_heartbeat=? WHERE id=?",
			(now, worker_id),
		)
		self.conn.commit()

	@staticmethod
	def _row_to_work_unit(row: sqlite3.Row) -> WorkUnit:
		keys = row.keys()
		return WorkUnit(
			id=row["id"],
			plan_id=row["plan_id"],
			title=row["title"],
			description=row["description"],
			files_hint=row["files_hint"],
			verification_hint=row["verification_hint"],
			priority=row["priority"],
			status=row["status"],
			worker_id=row["worker_id"],
			round_id=row["round_id"],
			plan_node_id=row["plan_node_id"],
			handoff_id=row["handoff_id"],
			depends_on=row["depends_on"],
			branch_name=row["branch_name"],
			claimed_at=row["claimed_at"],
			heartbeat_at=row["heartbeat_at"],
			started_at=row["started_at"],
			finished_at=row["finished_at"],
			exit_code=row["exit_code"],
			commit_hash=row["commit_hash"],
			output_summary=row["output_summary"],
			attempt=row["attempt"],
			max_attempts=row["max_attempts"],
			unit_type=row["unit_type"] if "unit_type" in keys else "implementation",
			timeout=row["timeout"],
			verification_command=row["verification_command"],
			epoch_id=row["epoch_id"] if "epoch_id" in keys else None,
			input_tokens=row["input_tokens"] if "input_tokens" in keys else 0,
			output_tokens=row["output_tokens"] if "output_tokens" in keys else 0,
			cost_usd=row["cost_usd"] if "cost_usd" in keys else 0.0,
			experiment_mode=bool(row["experiment_mode"]) if "experiment_mode" in keys else False,
		)

	# -- Workers --

	def insert_worker(self, worker: Worker) -> None:
		self.conn.execute(
			"""INSERT INTO workers
			(id, workspace_path, status, current_unit_id, pid,
			 started_at, last_heartbeat, units_completed, units_failed, total_cost_usd,
			 backend_type, backend_metadata)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				worker.id, worker.workspace_path, worker.status,
				worker.current_unit_id, worker.pid, worker.started_at,
				worker.last_heartbeat, worker.units_completed,
				worker.units_failed, worker.total_cost_usd,
				worker.backend_type, worker.backend_metadata,
			),
		)
		self.conn.commit()

	def update_worker(self, worker: Worker) -> None:
		self.conn.execute(
			"""UPDATE workers SET
			workspace_path=?, status=?, current_unit_id=?, pid=?,
			started_at=?, last_heartbeat=?, units_completed=?,
			units_failed=?, total_cost_usd=?,
			backend_type=?, backend_metadata=?
			WHERE id=?""",
			(
				worker.workspace_path, worker.status, worker.current_unit_id,
				worker.pid, worker.started_at, worker.last_heartbeat,
				worker.units_completed, worker.units_failed,
				worker.total_cost_usd, worker.backend_type,
				worker.backend_metadata, worker.id,
			),
		)
		self.conn.commit()

	def get_worker(self, worker_id: str) -> Worker | None:
		row = self.conn.execute("SELECT * FROM workers WHERE id=?", (worker_id,)).fetchone()
		if row is None:
			return None
		return self._row_to_worker(row)

	def get_all_workers(self) -> list[Worker]:
		rows = self.conn.execute("SELECT * FROM workers ORDER BY started_at").fetchall()
		return [self._row_to_worker(r) for r in rows]

	@staticmethod
	def _row_to_worker(row: sqlite3.Row) -> Worker:
		return Worker(
			id=row["id"],
			workspace_path=row["workspace_path"],
			status=row["status"],
			current_unit_id=row["current_unit_id"],
			pid=row["pid"],
			started_at=row["started_at"],
			last_heartbeat=row["last_heartbeat"],
			units_completed=row["units_completed"],
			units_failed=row["units_failed"],
			total_cost_usd=row["total_cost_usd"],
			backend_type=row["backend_type"],
			backend_metadata=row["backend_metadata"],
		)

	# -- Merge Requests --

	def insert_merge_request(self, mr: MergeRequest) -> None:
		self.conn.execute(
			"""INSERT INTO merge_requests
			(id, work_unit_id, worker_id, branch_name, commit_hash,
			 status, position, created_at, verified_at, merged_at,
			 rejection_reason, rebase_attempts)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				mr.id, mr.work_unit_id, mr.worker_id, mr.branch_name,
				mr.commit_hash, mr.status, mr.position, mr.created_at,
				mr.verified_at, mr.merged_at, mr.rejection_reason,
				mr.rebase_attempts,
			),
		)
		self.conn.commit()

	def update_merge_request(self, mr: MergeRequest) -> None:
		self.conn.execute(
			"""UPDATE merge_requests SET
			work_unit_id=?, worker_id=?, branch_name=?, commit_hash=?,
			status=?, position=?, verified_at=?, merged_at=?,
			rejection_reason=?, rebase_attempts=?
			WHERE id=?""",
			(
				mr.work_unit_id, mr.worker_id, mr.branch_name, mr.commit_hash,
				mr.status, mr.position, mr.verified_at, mr.merged_at,
				mr.rejection_reason, mr.rebase_attempts, mr.id,
			),
		)
		self.conn.commit()
		logger.info("Updated merge_request %s -> status=%s", mr.id, mr.status)

	def get_next_merge_request(self) -> MergeRequest | None:
		"""Get the next pending merge request by position."""
		row = self.conn.execute(
			"SELECT * FROM merge_requests WHERE status='pending' ORDER BY position ASC LIMIT 1"
		).fetchone()
		if row is None:
			return None
		return self._row_to_merge_request(row)

	def get_processed_merge_requests_for_worker(self, worker_id: str) -> list[MergeRequest]:
		"""Get merge requests for a worker that have been processed (merged/rejected/conflict).

		These are MRs whose branches can safely be cleaned up from the worker workspace.
		"""
		rows = self.conn.execute(
			"""SELECT * FROM merge_requests
			WHERE worker_id = ? AND status IN ('merged', 'rejected', 'conflict')
			ORDER BY position ASC""",
			(worker_id,),
		).fetchall()
		return [self._row_to_merge_request(r) for r in rows]

	def get_merge_requests_for_plan(self, plan_id: str) -> list[MergeRequest]:
		"""Get all merge requests for work units in a plan."""
		rows = self.conn.execute(
			"""SELECT mr.* FROM merge_requests mr
			JOIN work_units wu ON mr.work_unit_id = wu.id
			WHERE wu.plan_id = ?
			ORDER BY mr.position ASC""",
			(plan_id,),
		).fetchall()
		return [self._row_to_merge_request(r) for r in rows]

	def get_next_merge_position(self) -> int:
		"""Get the next available merge position."""
		row = self.conn.execute(
			"SELECT COALESCE(MAX(position), 0) + 1 AS next_pos FROM merge_requests"
		).fetchone()
		return int(row["next_pos"]) if row else 1

	def insert_merge_request_atomic(self, mr: MergeRequest) -> MergeRequest:
		"""Assign next position and insert in one transaction (TOCTOU-safe)."""
		row = self.conn.execute(
			"SELECT COALESCE(MAX(position), 0) + 1 AS next_pos FROM merge_requests"
		).fetchone()
		mr.position = int(row["next_pos"]) if row else 1
		self.conn.execute(
			"""INSERT INTO merge_requests
			(id, work_unit_id, worker_id, branch_name, commit_hash,
			 status, position, created_at, verified_at, merged_at,
			 rejection_reason, rebase_attempts)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				mr.id, mr.work_unit_id, mr.worker_id, mr.branch_name,
				mr.commit_hash, mr.status, mr.position, mr.created_at,
				mr.verified_at, mr.merged_at, mr.rejection_reason,
				mr.rebase_attempts,
			),
		)
		self.conn.commit()
		logger.info("Inserted merge_request %s at position %d for unit %s", mr.id, mr.position, mr.work_unit_id)
		return mr

	@staticmethod
	def _row_to_merge_request(row: sqlite3.Row) -> MergeRequest:
		return MergeRequest(
			id=row["id"],
			work_unit_id=row["work_unit_id"],
			worker_id=row["worker_id"],
			branch_name=row["branch_name"],
			commit_hash=row["commit_hash"],
			status=row["status"],
			position=row["position"],
			created_at=row["created_at"],
			verified_at=row["verified_at"],
			merged_at=row["merged_at"],
			rejection_reason=row["rejection_reason"],
			rebase_attempts=row["rebase_attempts"],
		)

	# -- Missions --

	def insert_mission(self, mission: Mission) -> None:
		self.conn.execute(
			"""INSERT INTO missions
			(id, objective, status, started_at, finished_at,
			 total_rounds, total_cost_usd, final_score, stopped_reason,
			 ambition_score, next_objective, proposed_by_strategist)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				mission.id, mission.objective, mission.status,
				mission.started_at, mission.finished_at,
				mission.total_rounds, mission.total_cost_usd,
				mission.final_score, mission.stopped_reason,
				mission.ambition_score, mission.next_objective,
				int(mission.proposed_by_strategist),
			),
		)
		self.conn.commit()
		logger.info("Inserted mission %s (status=%s)", mission.id, mission.status)

	def update_mission(self, mission: Mission) -> None:
		self.conn.execute(
			"""UPDATE missions SET
			objective=?, status=?, started_at=?, finished_at=?,
			total_rounds=?, total_cost_usd=?, final_score=?, stopped_reason=?,
			ambition_score=?, next_objective=?, proposed_by_strategist=?
			WHERE id=?""",
			(
				mission.objective, mission.status, mission.started_at,
				mission.finished_at, mission.total_rounds,
				mission.total_cost_usd, mission.final_score,
				mission.stopped_reason, mission.ambition_score,
				mission.next_objective, int(mission.proposed_by_strategist),
				mission.id,
			),
		)
		self.conn.commit()

	def get_mission(self, mission_id: str) -> Mission | None:
		row = self.conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
		if row is None:
			return None
		return self._row_to_mission(row)

	def get_latest_mission(self) -> Mission | None:
		row = self.conn.execute(
			"SELECT * FROM missions ORDER BY started_at DESC LIMIT 1"
		).fetchone()
		if row is None:
			return None
		return self._row_to_mission(row)

	def get_all_missions(self, limit: int = 20) -> list[Mission]:
		rows = self.conn.execute(
			"SELECT * FROM missions ORDER BY started_at DESC LIMIT ?", (limit,)
		).fetchall()
		return [self._row_to_mission(r) for r in rows]

	def get_mission_summary(self, mission_id: str) -> dict[str, Any]:
		"""Aggregate summary stats for a mission.

		Returns dict with unit counts by status, total cost/tokens, epoch
		breakdown, and event type distribution.
		"""
		# Unit counts by status
		unit_rows = self.conn.execute(
			"""SELECT wu.status, COUNT(*) AS cnt
			FROM work_units wu
			JOIN epochs e ON wu.epoch_id = e.id
			WHERE e.mission_id = ?
			GROUP BY wu.status""",
			(mission_id,),
		).fetchall()
		units_by_status: dict[str, int] = {r["status"]: int(r["cnt"]) for r in unit_rows}

		# Token/cost totals
		totals_row = self.conn.execute(
			"""SELECT
				COALESCE(SUM(wu.input_tokens), 0) AS total_input_tokens,
				COALESCE(SUM(wu.output_tokens), 0) AS total_output_tokens,
				COALESCE(SUM(wu.cost_usd), 0.0) AS total_cost_usd
			FROM work_units wu
			JOIN epochs e ON wu.epoch_id = e.id
			WHERE e.mission_id = ?""",
			(mission_id,),
		).fetchone()

		# Event type distribution
		event_rows = self.conn.execute(
			"""SELECT event_type, COUNT(*) AS cnt
			FROM unit_events
			WHERE mission_id = ?
			GROUP BY event_type""",
			(mission_id,),
		).fetchall()
		events_by_type: dict[str, int] = {r["event_type"]: int(r["cnt"]) for r in event_rows}

		# Epoch breakdown
		epoch_rows = self.conn.execute(
			"""SELECT number, units_planned, units_completed, units_failed
			FROM epochs
			WHERE mission_id = ?
			ORDER BY number ASC""",
			(mission_id,),
		).fetchall()
		epochs = [
			{
				"number": int(r["number"]),
				"units_planned": int(r["units_planned"]),
				"units_completed": int(r["units_completed"]),
				"units_failed": int(r["units_failed"]),
			}
			for r in epoch_rows
		]

		return {
			"units_by_status": units_by_status,
			"total_input_tokens": int(totals_row["total_input_tokens"]) if totals_row else 0,
			"total_output_tokens": int(totals_row["total_output_tokens"]) if totals_row else 0,
			"total_cost_usd": float(totals_row["total_cost_usd"]) if totals_row else 0.0,
			"events_by_type": events_by_type,
			"epochs": epochs,
		}

	@staticmethod
	def _row_to_mission(row: sqlite3.Row) -> Mission:
		keys = row.keys()
		return Mission(
			id=row["id"],
			objective=row["objective"],
			status=row["status"],
			started_at=row["started_at"],
			finished_at=row["finished_at"],
			total_rounds=row["total_rounds"],
			total_cost_usd=row["total_cost_usd"],
			final_score=row["final_score"],
			stopped_reason=row["stopped_reason"],
			ambition_score=row["ambition_score"] if "ambition_score" in keys else 0,
			next_objective=row["next_objective"] if "next_objective" in keys else "",
			proposed_by_strategist=bool(row["proposed_by_strategist"]) if "proposed_by_strategist" in keys else False,
		)

	# -- Rounds --

	def insert_round(self, rnd: Round) -> None:
		self.conn.execute(
			"""INSERT INTO rounds
			(id, mission_id, number, status, started_at, finished_at,
			 snapshot_hash, plan_id, objective_score, objective_met,
			 total_units, completed_units, failed_units, cost_usd, discoveries)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				rnd.id, rnd.mission_id, rnd.number, rnd.status,
				rnd.started_at, rnd.finished_at, rnd.snapshot_hash,
				rnd.plan_id, rnd.objective_score, int(rnd.objective_met),
				rnd.total_units, rnd.completed_units, rnd.failed_units,
				rnd.cost_usd, rnd.discoveries,
			),
		)
		self.conn.commit()

	def update_round(self, rnd: Round) -> None:
		self.conn.execute(
			"""UPDATE rounds SET
			mission_id=?, number=?, status=?, started_at=?, finished_at=?,
			snapshot_hash=?, plan_id=?, objective_score=?, objective_met=?,
			total_units=?, completed_units=?, failed_units=?,
			cost_usd=?, discoveries=?
			WHERE id=?""",
			(
				rnd.mission_id, rnd.number, rnd.status, rnd.started_at,
				rnd.finished_at, rnd.snapshot_hash, rnd.plan_id,
				rnd.objective_score, int(rnd.objective_met),
				rnd.total_units, rnd.completed_units, rnd.failed_units,
				rnd.cost_usd, rnd.discoveries, rnd.id,
			),
		)
		self.conn.commit()

	def get_round(self, round_id: str) -> Round | None:
		row = self.conn.execute("SELECT * FROM rounds WHERE id=?", (round_id,)).fetchone()
		if row is None:
			return None
		return self._row_to_round(row)

	def get_rounds_for_mission(self, mission_id: str) -> list[Round]:
		rows = self.conn.execute(
			"SELECT * FROM rounds WHERE mission_id=? ORDER BY number ASC",
			(mission_id,),
		).fetchall()
		return [self._row_to_round(r) for r in rows]

	@staticmethod
	def _row_to_round(row: sqlite3.Row) -> Round:
		return Round(
			id=row["id"],
			mission_id=row["mission_id"],
			number=row["number"],
			status=row["status"],
			started_at=row["started_at"],
			finished_at=row["finished_at"],
			snapshot_hash=row["snapshot_hash"],
			plan_id=row["plan_id"],
			objective_score=row["objective_score"],
			objective_met=bool(row["objective_met"]),
			total_units=row["total_units"],
			completed_units=row["completed_units"],
			failed_units=row["failed_units"],
			cost_usd=row["cost_usd"],
			discoveries=row["discoveries"],
		)

	# -- Plan Nodes --

	def insert_plan_node(self, node: PlanNode) -> None:
		self.conn.execute(
			"""INSERT INTO plan_nodes
			(id, plan_id, parent_id, depth, scope, strategy,
			 status, node_type, work_unit_id, children_ids)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				node.id, node.plan_id, node.parent_id, node.depth,
				node.scope, node.strategy, node.status, node.node_type,
				node.work_unit_id, node.children_ids,
			),
		)
		self.conn.commit()

	def update_plan_node(self, node: PlanNode) -> None:
		self.conn.execute(
			"""UPDATE plan_nodes SET
			plan_id=?, parent_id=?, depth=?, scope=?, strategy=?,
			status=?, node_type=?, work_unit_id=?, children_ids=?
			WHERE id=?""",
			(
				node.plan_id, node.parent_id, node.depth, node.scope,
				node.strategy, node.status, node.node_type,
				node.work_unit_id, node.children_ids, node.id,
			),
		)
		self.conn.commit()

	def get_plan_node(self, node_id: str) -> PlanNode | None:
		row = self.conn.execute("SELECT * FROM plan_nodes WHERE id=?", (node_id,)).fetchone()
		if row is None:
			return None
		return self._row_to_plan_node(row)

	def get_plan_nodes_for_plan(self, plan_id: str) -> list[PlanNode]:
		rows = self.conn.execute(
			"SELECT * FROM plan_nodes WHERE plan_id=? ORDER BY depth ASC",
			(plan_id,),
		).fetchall()
		return [self._row_to_plan_node(r) for r in rows]

	def get_leaf_nodes_for_plan(self, plan_id: str) -> list[PlanNode]:
		"""Get all leaf nodes (that produce work units) for a plan."""
		rows = self.conn.execute(
			"SELECT * FROM plan_nodes WHERE plan_id=? AND node_type='leaf' ORDER BY id",
			(plan_id,),
		).fetchall()
		return [self._row_to_plan_node(r) for r in rows]

	@staticmethod
	def _row_to_plan_node(row: sqlite3.Row) -> PlanNode:
		return PlanNode(
			id=row["id"],
			plan_id=row["plan_id"],
			parent_id=row["parent_id"],
			depth=row["depth"],
			scope=row["scope"],
			strategy=row["strategy"],
			status=row["status"],
			node_type=row["node_type"],
			work_unit_id=row["work_unit_id"],
			children_ids=row["children_ids"],
		)

	# -- Handoffs --

	def insert_handoff(self, handoff: Handoff) -> None:
		self.conn.execute(
			"""INSERT INTO handoffs
			(id, work_unit_id, round_id, epoch_id, status, commits,
			 summary, discoveries, concerns, files_changed)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				handoff.id, handoff.work_unit_id, handoff.round_id,
				handoff.epoch_id, handoff.status, handoff.commits,
				handoff.summary, handoff.discoveries, handoff.concerns,
				handoff.files_changed,
			),
		)
		self.conn.commit()

	def get_handoff(self, handoff_id: str) -> Handoff | None:
		row = self.conn.execute("SELECT * FROM handoffs WHERE id=?", (handoff_id,)).fetchone()
		if row is None:
			return None
		return self._row_to_handoff(row)

	def get_handoffs_for_round(self, round_id: str) -> list[Handoff]:
		rows = self.conn.execute(
			"SELECT * FROM handoffs WHERE round_id=?",
			(round_id,),
		).fetchall()
		return [self._row_to_handoff(r) for r in rows]

	@staticmethod
	def _row_to_handoff(row: sqlite3.Row) -> Handoff:
		keys = row.keys()
		return Handoff(
			id=row["id"],
			work_unit_id=row["work_unit_id"],
			round_id=row["round_id"],
			status=row["status"],
			commits=row["commits"],
			summary=row["summary"],
			discoveries=row["discoveries"],
			concerns=row["concerns"],
			files_changed=row["files_changed"],
			epoch_id=row["epoch_id"] if "epoch_id" in keys else None,
		)

	def get_recent_handoffs(self, mission_id: str, limit: int = 15) -> list[Handoff]:
		"""Get recent handoffs for a mission via epoch_id -> epochs.mission_id."""
		rows = self.conn.execute(
			"""SELECT h.* FROM handoffs h
			JOIN epochs e ON h.epoch_id = e.id
			WHERE e.mission_id = ?
			ORDER BY h.rowid DESC
			LIMIT ?""",
			(mission_id, limit),
		).fetchall()
		return [self._row_to_handoff(r) for r in rows]

	# -- Reflections --

	def insert_reflection(self, r: Reflection) -> None:
		self.conn.execute(
			"""INSERT INTO reflections
			(id, mission_id, round_id, round_number, timestamp,
			 tests_before, tests_after, tests_delta, lint_delta, type_delta,
			 objective_score, score_delta, units_planned, units_completed,
			 units_failed, completion_rate, plan_depth, plan_strategy,
			 fixup_promoted, fixup_attempts, merge_conflicts, discoveries_count)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				r.id, r.mission_id, r.round_id, r.round_number, r.timestamp,
				r.tests_before, r.tests_after, r.tests_delta, r.lint_delta,
				r.type_delta, r.objective_score, r.score_delta,
				r.units_planned, r.units_completed, r.units_failed,
				r.completion_rate, r.plan_depth, r.plan_strategy,
				int(r.fixup_promoted), r.fixup_attempts, r.merge_conflicts,
				r.discoveries_count,
			),
		)
		self.conn.commit()

	def get_recent_reflections(self, mission_id: str, limit: int = 5) -> list[Reflection]:
		rows = self.conn.execute(
			"SELECT * FROM reflections WHERE mission_id=? ORDER BY round_number DESC LIMIT ?",
			(mission_id, limit),
		).fetchall()
		return [self._row_to_reflection(r) for r in rows]

	@staticmethod
	def _row_to_reflection(row: sqlite3.Row) -> Reflection:
		return Reflection(
			id=row["id"],
			mission_id=row["mission_id"],
			round_id=row["round_id"],
			round_number=row["round_number"],
			timestamp=row["timestamp"],
			tests_before=row["tests_before"],
			tests_after=row["tests_after"],
			tests_delta=row["tests_delta"],
			lint_delta=row["lint_delta"],
			type_delta=row["type_delta"],
			objective_score=row["objective_score"],
			score_delta=row["score_delta"],
			units_planned=row["units_planned"],
			units_completed=row["units_completed"],
			units_failed=row["units_failed"],
			completion_rate=row["completion_rate"],
			plan_depth=row["plan_depth"],
			plan_strategy=row["plan_strategy"],
			fixup_promoted=bool(row["fixup_promoted"]),
			fixup_attempts=row["fixup_attempts"],
			merge_conflicts=row["merge_conflicts"],
			discoveries_count=row["discoveries_count"],
		)

	# -- Rewards --

	def insert_reward(self, r: Reward) -> None:
		self.conn.execute(
			"""INSERT INTO rewards
			(id, round_id, mission_id, timestamp, reward,
			 verification_improvement, completion_rate, score_progress,
			 fixup_efficiency, no_regression)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				r.id, r.round_id, r.mission_id, r.timestamp, r.reward,
				r.verification_improvement, r.completion_rate,
				r.score_progress, r.fixup_efficiency, r.no_regression,
			),
		)
		self.conn.commit()

	@staticmethod
	def _row_to_reward(row: sqlite3.Row) -> Reward:
		return Reward(
			id=row["id"],
			round_id=row["round_id"],
			mission_id=row["mission_id"],
			timestamp=row["timestamp"],
			reward=row["reward"],
			verification_improvement=row["verification_improvement"],
			completion_rate=row["completion_rate"],
			score_progress=row["score_progress"],
			fixup_efficiency=row["fixup_efficiency"],
			no_regression=row["no_regression"],
		)

	# -- Experiences --

	def insert_experience(self, e: Experience) -> None:
		self.conn.execute(
			"""INSERT INTO experiences
			(id, round_id, work_unit_id, timestamp, title, scope,
			 files_hint, status, summary, files_changed,
			 discoveries, concerns, reward)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				e.id, e.round_id, e.work_unit_id, e.timestamp,
				e.title, e.scope, e.files_hint, e.status, e.summary,
				e.files_changed, e.discoveries, e.concerns, e.reward,
			),
		)
		self.conn.commit()

	def get_high_reward_experiences(self, limit: int = 10) -> list[Experience]:
		rows = self.conn.execute(
			"SELECT * FROM experiences ORDER BY reward DESC LIMIT ?",
			(limit,),
		).fetchall()
		return [self._row_to_experience(r) for r in rows]

	def search_experiences(self, keywords: list[str], limit: int = 5) -> list[Experience]:
		"""Search experiences by keyword overlap on title + scope + files_hint."""
		if not keywords:
			return []
		conditions = []
		params: list[str] = []
		for kw in keywords:
			conditions.append(
				"(title LIKE ? OR scope LIKE ? OR files_hint LIKE ?)"
			)
			pattern = f"%{kw}%"
			params.extend([pattern, pattern, pattern])
		where = " OR ".join(conditions)
		rows = self.conn.execute(
			f"SELECT * FROM experiences WHERE {where} ORDER BY reward DESC LIMIT ?",  # noqa: S608
			(*params, limit),
		).fetchall()
		return [self._row_to_experience(r) for r in rows]

	def get_top_experiences(self, limit: int = 10) -> list[Experience]:
		"""Get the top-rewarded experiences, prioritizing those with discoveries."""
		rows = self.conn.execute(
			"""SELECT * FROM experiences
			WHERE status = 'completed' AND discoveries != '' AND discoveries != '[]'
			ORDER BY reward DESC LIMIT ?""",
			(limit,),
		).fetchall()
		return [self._row_to_experience(r) for r in rows]

	@staticmethod
	def _row_to_experience(row: sqlite3.Row) -> Experience:
		return Experience(
			id=row["id"],
			round_id=row["round_id"],
			work_unit_id=row["work_unit_id"],
			timestamp=row["timestamp"],
			title=row["title"],
			scope=row["scope"],
			files_hint=row["files_hint"],
			status=row["status"],
			summary=row["summary"],
			files_changed=row["files_changed"],
			discoveries=row["discoveries"],
			concerns=row["concerns"],
			reward=row["reward"],
		)

	# -- Signals --

	def insert_signal(self, signal: Signal) -> None:
		self.conn.execute(
			"""INSERT INTO signals
			(id, mission_id, signal_type, payload, status, created_at, acknowledged_at)
			VALUES (?, ?, ?, ?, ?, ?, ?)""",
			(
				signal.id, signal.mission_id, signal.signal_type,
				signal.payload, signal.status, signal.created_at,
				signal.acknowledged_at,
			),
		)
		self.conn.commit()

	def get_pending_signals(self, mission_id: str) -> list[Signal]:
		"""Get all pending signals for a mission."""
		rows = self.conn.execute(
			"SELECT * FROM signals WHERE mission_id=? AND status='pending' ORDER BY created_at ASC",
			(mission_id,),
		).fetchall()
		return [self._row_to_signal(r) for r in rows]

	def acknowledge_signal(self, signal_id: str) -> None:
		"""Mark a signal as acknowledged."""
		from mission_control.models import _now_iso
		self.conn.execute(
			"UPDATE signals SET status='acknowledged', acknowledged_at=? WHERE id=?",
			(_now_iso(), signal_id),
		)
		self.conn.commit()

	def expire_stale_signals(self, timeout_minutes: int = 10) -> int:
		"""Expire unacknowledged signals older than timeout. Returns count expired."""
		from mission_control.models import _now_iso
		now = _now_iso()
		cursor = self.conn.execute(
			"""UPDATE signals SET status='expired'
			WHERE status='pending'
			AND (julianday(?) - julianday(created_at)) * 1440 > ?""",
			(now, timeout_minutes),
		)
		self.conn.commit()
		return cursor.rowcount

	@staticmethod
	def _row_to_signal(row: sqlite3.Row) -> Signal:
		return Signal(
			id=row["id"],
			mission_id=row["mission_id"],
			signal_type=row["signal_type"],
			payload=row["payload"],
			status=row["status"],
			created_at=row["created_at"],
			acknowledged_at=row["acknowledged_at"],
		)

	# -- Epochs --

	def insert_epoch(self, epoch: Epoch) -> None:
		self.conn.execute(
			"""INSERT INTO epochs
			(id, mission_id, number, started_at, finished_at,
			 units_planned, units_completed, units_failed,
			 score_at_start, score_at_end)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				epoch.id, epoch.mission_id, epoch.number,
				epoch.started_at, epoch.finished_at,
				epoch.units_planned, epoch.units_completed,
				epoch.units_failed, epoch.score_at_start, epoch.score_at_end,
			),
		)
		self.conn.commit()

	def update_epoch(self, epoch: Epoch) -> None:
		self.conn.execute(
			"""UPDATE epochs SET
			mission_id=?, number=?, started_at=?, finished_at=?,
			units_planned=?, units_completed=?, units_failed=?,
			score_at_start=?, score_at_end=?
			WHERE id=?""",
			(
				epoch.mission_id, epoch.number, epoch.started_at,
				epoch.finished_at, epoch.units_planned, epoch.units_completed,
				epoch.units_failed, epoch.score_at_start, epoch.score_at_end,
				epoch.id,
			),
		)
		self.conn.commit()

	def get_epoch(self, epoch_id: str) -> Epoch | None:
		row = self.conn.execute("SELECT * FROM epochs WHERE id=?", (epoch_id,)).fetchone()
		if row is None:
			return None
		return self._row_to_epoch(row)

	def get_epochs_for_mission(self, mission_id: str) -> list[Epoch]:
		rows = self.conn.execute(
			"SELECT * FROM epochs WHERE mission_id=? ORDER BY number ASC",
			(mission_id,),
		).fetchall()
		return [self._row_to_epoch(r) for r in rows]

	@staticmethod
	def _row_to_epoch(row: sqlite3.Row) -> Epoch:
		return Epoch(
			id=row["id"],
			mission_id=row["mission_id"],
			number=row["number"],
			started_at=row["started_at"],
			finished_at=row["finished_at"],
			units_planned=row["units_planned"],
			units_completed=row["units_completed"],
			units_failed=row["units_failed"],
			score_at_start=row["score_at_start"],
			score_at_end=row["score_at_end"],
		)

	# -- Unit Events --

	def insert_unit_event(self, event: UnitEvent) -> None:
		self.conn.execute(
			"""INSERT INTO unit_events
			(id, mission_id, epoch_id, work_unit_id, event_type,
			 timestamp, score_after, details,
			 input_tokens, output_tokens)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				event.id, event.mission_id, event.epoch_id,
				event.work_unit_id, event.event_type,
				event.timestamp, event.score_after, event.details,
				event.input_tokens, event.output_tokens,
			),
		)
		self.conn.commit()

	def get_unit_events_for_mission(self, mission_id: str, limit: int = 100) -> list[UnitEvent]:
		rows = self.conn.execute(
			"SELECT * FROM unit_events WHERE mission_id=? ORDER BY timestamp ASC LIMIT ?",
			(mission_id, limit),
		).fetchall()
		return [self._row_to_unit_event(r) for r in rows]

	def get_unit_events_for_epoch(self, epoch_id: str) -> list[UnitEvent]:
		rows = self.conn.execute(
			"SELECT * FROM unit_events WHERE epoch_id=? ORDER BY timestamp ASC",
			(epoch_id,),
		).fetchall()
		return [self._row_to_unit_event(r) for r in rows]

	def get_token_usage_by_epoch(self, mission_id: str) -> list[dict[str, object]]:
		"""Aggregate token usage per epoch for charting."""
		rows = self.conn.execute(
			"""SELECT e.number AS epoch_number,
				COALESCE(SUM(wu.input_tokens), 0) AS input_tokens,
				COALESCE(SUM(wu.output_tokens), 0) AS output_tokens
			FROM epochs e
			LEFT JOIN work_units wu ON wu.epoch_id = e.id
			WHERE e.mission_id = ?
			GROUP BY e.id, e.number
			ORDER BY e.number ASC""",
			(mission_id,),
		).fetchall()
		return [
			{
				"epoch": int(r["epoch_number"]),
				"input_tokens": int(r["input_tokens"]),
				"output_tokens": int(r["output_tokens"]),
			}
			for r in rows
		]

	@staticmethod
	def _row_to_unit_event(row: sqlite3.Row) -> UnitEvent:
		keys = row.keys()
		return UnitEvent(
			id=row["id"],
			mission_id=row["mission_id"],
			epoch_id=row["epoch_id"],
			work_unit_id=row["work_unit_id"],
			event_type=row["event_type"],
			timestamp=row["timestamp"],
			score_after=row["score_after"],
			details=row["details"],
			input_tokens=row["input_tokens"] if "input_tokens" in keys else 0,
			output_tokens=row["output_tokens"] if "output_tokens" in keys else 0,
		)

	# -- Discoveries --

	def insert_discovery_result(self, result: DiscoveryResult, items: list[DiscoveryItem]) -> None:
		"""Insert a discovery result and its items atomically."""
		try:
			self.conn.execute(
				"""INSERT INTO discoveries
				(id, target_path, timestamp, raw_output, model, item_count)
				VALUES (?, ?, ?, ?, ?, ?)""",
				(
					result.id, result.target_path, result.timestamp,
					result.raw_output, result.model, result.item_count,
				),
			)
			for item in items:
				item.discovery_id = result.id
				self.conn.execute(
					"""INSERT INTO discovery_items
					(id, discovery_id, track, title, description, rationale,
					 files_hint, impact, effort, priority_score, status)
					VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
					(
						item.id, item.discovery_id, item.track, item.title,
						item.description, item.rationale, item.files_hint,
						item.impact, item.effort, item.priority_score, item.status,
					),
				)
			self.conn.commit()
		except sqlite3.Error:
			logger.error("insert_discovery_result failed for %s, rolling back", result.id, exc_info=True)
			self.conn.rollback()
			raise

	def get_latest_discovery(self) -> tuple[DiscoveryResult | None, list[DiscoveryItem]]:
		"""Get the most recent discovery result and its items."""
		row = self.conn.execute(
			"SELECT * FROM discoveries ORDER BY timestamp DESC LIMIT 1"
		).fetchone()
		if row is None:
			return None, []
		result = self._row_to_discovery_result(row)
		items = self._get_discovery_items(result.id)
		return result, items

	def _get_discovery_items(self, discovery_id: str) -> list[DiscoveryItem]:
		"""Get all items for a discovery, sorted by priority score descending."""
		rows = self.conn.execute(
			"SELECT * FROM discovery_items WHERE discovery_id=? ORDER BY priority_score DESC",
			(discovery_id,),
		).fetchall()
		return [self._row_to_discovery_item(r) for r in rows]

	def get_all_discovery_results(self, limit: int = 10) -> list[DiscoveryResult]:
		"""Get recent discovery results."""
		rows = self.conn.execute(
			"SELECT * FROM discoveries ORDER BY timestamp DESC LIMIT ?",
			(limit,),
		).fetchall()
		return [self._row_to_discovery_result(r) for r in rows]

	def update_discovery_item_status(self, item_id: str, status: str) -> None:
		"""Update the status of a discovery item."""
		self.conn.execute(
			"UPDATE discovery_items SET status=? WHERE id=?",
			(status, item_id),
		)
		self.conn.commit()

	def get_past_discovery_titles(self, limit: int = 50) -> list[str]:
		"""Get titles of past discovery items to avoid repetition."""
		rows = self.conn.execute(
			"""SELECT di.title FROM discovery_items di
			JOIN discoveries d ON di.discovery_id = d.id
			ORDER BY d.timestamp DESC LIMIT ?""",
			(limit,),
		).fetchall()
		return [row["title"] for row in rows]

	@staticmethod
	def _row_to_discovery_result(row: sqlite3.Row) -> DiscoveryResult:
		return DiscoveryResult(
			id=row["id"],
			target_path=row["target_path"],
			timestamp=row["timestamp"],
			raw_output=row["raw_output"],
			model=row["model"],
			item_count=row["item_count"],
		)

	@staticmethod
	def _row_to_discovery_item(row: sqlite3.Row) -> DiscoveryItem:
		return DiscoveryItem(
			id=row["id"],
			discovery_id=row["discovery_id"],
			track=row["track"],
			title=row["title"],
			description=row["description"],
			rationale=row["rationale"],
			files_hint=row["files_hint"],
			impact=row["impact"],
			effort=row["effort"],
			priority_score=row["priority_score"],
			status=row["status"],
		)

	# -- Backlog Items --

	def insert_backlog_item(self, item: BacklogItem) -> None:
		self.conn.execute(
			"""INSERT INTO backlog_items
			(id, title, description, priority_score, impact, effort, track, status,
			 source_mission_id, created_at, updated_at, attempt_count,
			 last_failure_reason, pinned_score, depends_on, tags)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				item.id, item.title, item.description, item.priority_score,
				item.impact, item.effort, item.track, item.status,
				item.source_mission_id, item.created_at, item.updated_at,
				item.attempt_count, item.last_failure_reason, item.pinned_score,
				item.depends_on, item.tags,
			),
		)
		self.conn.commit()

	def update_backlog_item(self, item: BacklogItem) -> None:
		self.conn.execute(
			"""UPDATE backlog_items SET
			title=?, description=?, priority_score=?, impact=?, effort=?,
			track=?, status=?, source_mission_id=?, created_at=?, updated_at=?,
			attempt_count=?, last_failure_reason=?, pinned_score=?,
			depends_on=?, tags=?
			WHERE id=?""",
			(
				item.title, item.description, item.priority_score, item.impact,
				item.effort, item.track, item.status, item.source_mission_id,
				item.created_at, item.updated_at, item.attempt_count,
				item.last_failure_reason, item.pinned_score, item.depends_on,
				item.tags, item.id,
			),
		)
		self.conn.commit()

	def get_backlog_item(self, item_id: str) -> BacklogItem | None:
		row = self.conn.execute("SELECT * FROM backlog_items WHERE id=?", (item_id,)).fetchone()
		if row is None:
			return None
		return self._row_to_backlog_item(row)

	def list_backlog_items(
		self, status: str | None = None, track: str | None = None, limit: int = 50
	) -> list[BacklogItem]:
		conditions: list[str] = []
		params: list[str | int] = []
		if status is not None:
			conditions.append("status = ?")
			params.append(status)
		if track is not None:
			conditions.append("track = ?")
			params.append(track)
		where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
		rows = self.conn.execute(
			f"SELECT * FROM backlog_items {where} ORDER BY priority_score DESC LIMIT ?",  # noqa: S608
			(*params, limit),
		).fetchall()
		return [self._row_to_backlog_item(r) for r in rows]

	def get_pending_backlog(self, limit: int = 20) -> list[BacklogItem]:
		rows = self.conn.execute(
			"""SELECT * FROM backlog_items WHERE status = 'pending'
			ORDER BY
				CASE WHEN pinned_score IS NOT NULL THEN pinned_score ELSE priority_score END DESC
			LIMIT ?""",
			(limit,),
		).fetchall()
		return [self._row_to_backlog_item(r) for r in rows]

	def get_backlog_items_for_mission(self, mission_id: str) -> list[BacklogItem]:
		rows = self.conn.execute(
			"SELECT * FROM backlog_items WHERE source_mission_id = ? ORDER BY priority_score DESC",
			(mission_id,),
		).fetchall()
		return [self._row_to_backlog_item(r) for r in rows]

	def update_attempt_count(self, item_id: str, failure_reason: str | None = None) -> None:
		from mission_control.models import _now_iso
		self.conn.execute(
			"""UPDATE backlog_items SET
			attempt_count = attempt_count + 1,
			last_failure_reason = ?,
			updated_at = ?
			WHERE id = ?""",
			(failure_reason, _now_iso(), item_id),
		)
		self.conn.commit()

	def search_backlog_items(self, keywords: list[str], limit: int = 10) -> list[BacklogItem]:
		if not keywords:
			return []
		conditions = []
		params: list[str] = []
		for kw in keywords:
			conditions.append("(title LIKE ? OR description LIKE ? OR tags LIKE ?)")
			pattern = f"%{kw}%"
			params.extend([pattern, pattern, pattern])
		where = " OR ".join(conditions)
		rows = self.conn.execute(
			f"SELECT * FROM backlog_items WHERE {where} ORDER BY priority_score DESC LIMIT ?",  # noqa: S608
			(*params, limit),
		).fetchall()
		return [self._row_to_backlog_item(r) for r in rows]

	def defer_backlog_item(self, item_id: str) -> None:
		from mission_control.models import _now_iso
		self.conn.execute(
			"UPDATE backlog_items SET status = 'deferred', updated_at = ? WHERE id = ?",
			(_now_iso(), item_id),
		)
		self.conn.commit()

	def pin_backlog_score(self, item_id: str, score: float) -> None:
		from mission_control.models import _now_iso
		self.conn.execute(
			"UPDATE backlog_items SET pinned_score = ?, updated_at = ? WHERE id = ?",
			(score, _now_iso(), item_id),
		)
		self.conn.commit()

	@staticmethod
	def _row_to_backlog_item(row: sqlite3.Row) -> BacklogItem:
		return BacklogItem(
			id=row["id"],
			title=row["title"],
			description=row["description"],
			priority_score=row["priority_score"],
			impact=row["impact"],
			effort=row["effort"],
			track=row["track"],
			status=row["status"],
			source_mission_id=row["source_mission_id"],
			created_at=row["created_at"],
			updated_at=row["updated_at"],
			attempt_count=row["attempt_count"],
			last_failure_reason=row["last_failure_reason"],
			pinned_score=row["pinned_score"],
			depends_on=row["depends_on"],
			tags=row["tags"],
		)

	# -- Strategic Context --

	def insert_strategic_context(self, ctx: StrategicContext) -> None:
		self.conn.execute(
			"""INSERT INTO strategic_context
			(id, mission_id, timestamp, what_attempted, what_worked, what_failed, recommended_next)
			VALUES (?, ?, ?, ?, ?, ?, ?)""",
			(
				ctx.id, ctx.mission_id, ctx.timestamp,
				ctx.what_attempted, ctx.what_worked,
				ctx.what_failed, ctx.recommended_next,
			),
		)
		self.conn.commit()

	def get_strategic_context(self, limit: int = 10) -> list[StrategicContext]:
		rows = self.conn.execute(
			"SELECT * FROM strategic_context ORDER BY timestamp DESC LIMIT ?",
			(limit,),
		).fetchall()
		return [self._row_to_strategic_context(r) for r in rows]

	def append_strategic_context(
		self,
		mission_id: str,
		what_attempted: str,
		what_worked: str,
		what_failed: str,
		recommended_next: str,
	) -> StrategicContext:
		ctx = StrategicContext(
			mission_id=mission_id,
			what_attempted=what_attempted,
			what_worked=what_worked,
			what_failed=what_failed,
			recommended_next=recommended_next,
		)
		self.insert_strategic_context(ctx)
		return ctx

	@staticmethod
	def _row_to_strategic_context(row: sqlite3.Row) -> StrategicContext:
		return StrategicContext(
			id=row["id"],
			mission_id=row["mission_id"],
			timestamp=row["timestamp"],
			what_attempted=row["what_attempted"],
			what_worked=row["what_worked"],
			what_failed=row["what_failed"],
			recommended_next=row["recommended_next"],
		)

	# -- Experiment Results --

	def insert_experiment_result(self, result: ExperimentResult) -> None:
		self.conn.execute(
			"""INSERT INTO experiment_results
			(id, work_unit_id, epoch_id, mission_id, timestamp,
			 approach_count, comparison_report, recommended_approach, created_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				result.id, result.work_unit_id, result.epoch_id,
				result.mission_id, result.timestamp,
				result.approach_count, result.comparison_report,
				result.recommended_approach, result.created_at,
			),
		)
		self.conn.commit()

	def get_experiment_result(self, result_id: str) -> ExperimentResult | None:
		row = self.conn.execute(
			"SELECT * FROM experiment_results WHERE id=?", (result_id,)
		).fetchone()
		if row is None:
			return None
		return self._row_to_experiment_result(row)

	def get_experiment_results_for_mission(self, mission_id: str) -> list[ExperimentResult]:
		rows = self.conn.execute(
			"SELECT * FROM experiment_results WHERE mission_id=? ORDER BY timestamp DESC",
			(mission_id,),
		).fetchall()
		return [self._row_to_experiment_result(r) for r in rows]

	@staticmethod
	def _row_to_experiment_result(row: sqlite3.Row) -> ExperimentResult:
		return ExperimentResult(
			id=row["id"],
			work_unit_id=row["work_unit_id"],
			epoch_id=row["epoch_id"],
			mission_id=row["mission_id"],
			timestamp=row["timestamp"],
			approach_count=row["approach_count"],
			comparison_report=row["comparison_report"],
			recommended_approach=row["recommended_approach"],
			created_at=row["created_at"],
		)
