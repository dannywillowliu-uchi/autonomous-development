"""Shared data provider for TUI and web dashboards.

Read-only layer over the mission-control Database.
Both dashboards consume DashboardSnapshot via DashboardProvider.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from mission_control.db import Database
from mission_control.models import MergeRequest, Mission, Round, Worker, WorkUnit

log = logging.getLogger(__name__)


@dataclass
class DashboardEvent:
	"""A recent event for the activity log."""

	timestamp: str
	message: str
	event_type: str = ""  # completed/merged/claimed/failed


@dataclass
class WorkerInfo:
	"""Worker with resolved current unit title."""

	id: str
	status: str
	current_unit_title: str = ""
	units_completed: int = 0
	units_failed: int = 0
	total_cost_usd: float = 0.0
	backend_type: str = "local"


@dataclass
class DashboardSnapshot:
	"""Complete state for rendering a dashboard frame."""

	timestamp: str = ""
	# Mission
	mission: Mission | None = None
	current_round: Round | None = None
	phase: str = "idle"  # planning/executing/fixup/evaluating/idle
	# Workers
	workers: list[WorkerInfo] = field(default_factory=list)
	workers_active: int = 0
	workers_idle: int = 0
	workers_dead: int = 0
	# Work units
	units_total: int = 0
	units_pending: int = 0
	units_running: int = 0
	units_completed: int = 0
	units_failed: int = 0
	# Merge queue
	merge_queue_depth: int = 0
	# Cost
	round_cost: float = 0.0
	total_cost: float = 0.0
	# Historical (list of tuples for charting)
	score_history: list[tuple[int, float]] = field(default_factory=list)
	test_trend: list[tuple[int, int, int]] = field(default_factory=list)
	completion_rates: list[tuple[int, float]] = field(default_factory=list)
	cost_per_round: list[tuple[int, float]] = field(default_factory=list)
	# Recent events
	recent_events: list[DashboardEvent] = field(default_factory=list)

	@property
	def completion_pct(self) -> float:
		"""Work unit completion percentage."""
		if self.units_total == 0:
			return 0.0
		return (self.units_completed / self.units_total) * 100.0

	@property
	def score_delta(self) -> float:
		"""Score change from previous round."""
		if len(self.score_history) < 2:
			return 0.0
		return self.score_history[-1][1] - self.score_history[-2][1]

	@property
	def previous_score(self) -> float:
		"""Score from previous round."""
		if len(self.score_history) < 2:
			return 0.0
		return self.score_history[-2][1]


class DashboardProvider:
	"""Polls the database and produces DashboardSnapshots.

	Thread-safe. Can serve both TUI (callback subscription) and
	web (request-response via get_snapshot).
	"""

	def __init__(self, db_path: str | Path) -> None:
		self._db_path = str(db_path)
		self._db: Database | None = None
		self._snapshot: DashboardSnapshot = DashboardSnapshot()
		self._lock = threading.Lock()
		self._callbacks: list[Callable[[DashboardSnapshot], None]] = []
		self._poll_thread: threading.Thread | None = None
		self._running = False

	def _get_db(self) -> Database:
		if self._db is None:
			self._db = Database(self._db_path)
		return self._db

	def get_snapshot(self) -> DashboardSnapshot:
		"""Return the latest cached snapshot."""
		with self._lock:
			return self._snapshot

	def refresh(self) -> DashboardSnapshot:
		"""Force a fresh snapshot from the database."""
		snapshot = self._build_snapshot()
		with self._lock:
			self._snapshot = snapshot
		for cb in self._callbacks:
			try:
				cb(snapshot)
			except Exception:
				log.exception("Dashboard callback error")
		return snapshot

	def subscribe(self, callback: Callable[[DashboardSnapshot], None]) -> None:
		"""Register a callback for snapshot updates."""
		self._callbacks.append(callback)

	def start_polling(self, interval: float = 2.0) -> None:
		"""Start background polling thread."""
		if self._running:
			return
		self._running = True
		self._poll_thread = threading.Thread(
			target=self._poll_loop, args=(interval,), daemon=True
		)
		self._poll_thread.start()

	def stop(self) -> None:
		"""Stop polling and close database."""
		self._running = False
		if self._poll_thread is not None:
			self._poll_thread.join(timeout=5.0)
			self._poll_thread = None
		if self._db is not None:
			self._db.close()
			self._db = None

	def _poll_loop(self, interval: float) -> None:
		while self._running:
			try:
				self.refresh()
			except Exception:
				log.exception("Dashboard poll error")
			time.sleep(interval)

	def _build_snapshot(self) -> DashboardSnapshot:
		"""Build a complete snapshot from the database."""
		db = self._get_db()
		snap = DashboardSnapshot(
			timestamp=datetime.now(timezone.utc).isoformat(),
		)

		# Mission
		mission = db.get_latest_mission()
		snap.mission = mission
		if mission is None:
			return snap

		# Rounds
		rounds = db.get_rounds_for_mission(mission.id)
		if rounds:
			snap.current_round = rounds[-1]
			snap.phase = _round_status_to_phase(rounds[-1].status)

		# Score history
		snap.score_history = [
			(r.number, r.objective_score) for r in rounds
		]

		# Completion rates + cost per round
		snap.completion_rates = [
			(r.number, (r.completed_units / r.total_units * 100.0) if r.total_units > 0 else 0.0)
			for r in rounds
		]
		snap.cost_per_round = [
			(r.number, r.cost_usd) for r in rounds
		]

		# Test trend from reflections: (round, passed, failed)
		reflections = db.get_recent_reflections(mission.id, limit=50)
		reflections.sort(key=lambda r: r.round_number)
		snap.test_trend = [
			(
				ref.round_number,
				ref.tests_after,
				max(0, ref.tests_before - ref.tests_after + ref.tests_delta),
			)
			for ref in reflections
		]

		# Workers
		all_workers = db.get_all_workers()
		snap.workers = _build_worker_infos(db, all_workers)
		snap.workers_active = sum(1 for w in all_workers if w.status == "working")
		snap.workers_idle = sum(1 for w in all_workers if w.status == "idle")
		snap.workers_dead = sum(1 for w in all_workers if w.status == "dead")

		# Work units for current round
		current_round = snap.current_round
		if current_round and current_round.plan_id:
			units = db.get_work_units_for_plan(current_round.plan_id)
			snap.units_total = len(units)
			snap.units_pending = sum(1 for u in units if u.status in ("pending", "claimed"))
			snap.units_running = sum(1 for u in units if u.status == "running")
			snap.units_completed = sum(1 for u in units if u.status == "completed")
			snap.units_failed = sum(1 for u in units if u.status == "failed")

			# Merge queue
			merge_requests = db.get_merge_requests_for_plan(current_round.plan_id)
			snap.merge_queue_depth = sum(
				1 for mr in merge_requests if mr.status == "pending"
			)

			# Recent events from work units + merge requests
			snap.recent_events = _build_events(units, merge_requests)

		# Cost
		if current_round:
			snap.round_cost = current_round.cost_usd
		snap.total_cost = mission.total_cost_usd

		return snap


def _round_status_to_phase(status: str) -> str:
	"""Map Round.status to dashboard phase label."""
	mapping = {
		"pending": "idle",
		"planning": "planning",
		"executing": "executing",
		"evaluating": "evaluating",
		"completed": "idle",
		"failed": "idle",
	}
	return mapping.get(status, "idle")


def _build_worker_infos(db: Database, workers: list[Worker]) -> list[WorkerInfo]:
	"""Build WorkerInfo list with resolved unit titles."""
	infos = []
	for w in workers:
		title = ""
		if w.current_unit_id:
			unit = db.get_work_unit(w.current_unit_id)
			if unit:
				title = unit.title
		infos.append(WorkerInfo(
			id=w.id,
			status=w.status,
			current_unit_title=title,
			units_completed=w.units_completed,
			units_failed=w.units_failed,
			total_cost_usd=w.total_cost_usd,
			backend_type=w.backend_type,
		))
	return infos


def _build_events(
	units: list[WorkUnit],
	merge_requests: list[MergeRequest],
) -> list[DashboardEvent]:
	"""Build a time-ordered list of recent events from work units and merge requests."""
	events: list[DashboardEvent] = []

	for u in units:
		if u.status == "completed" and u.finished_at:
			events.append(DashboardEvent(
				timestamp=u.finished_at,
				message=f'Completed "{u.title}"',
				event_type="completed",
			))
		elif u.status == "running" and u.claimed_at:
			events.append(DashboardEvent(
				timestamp=u.claimed_at,
				message=f'Claimed "{u.title}"',
				event_type="claimed",
			))
		elif u.status == "failed" and u.finished_at:
			events.append(DashboardEvent(
				timestamp=u.finished_at,
				message=f'Failed "{u.title}"',
				event_type="failed",
			))

	for mr in merge_requests:
		if mr.status == "merged" and mr.merged_at:
			events.append(DashboardEvent(
				timestamp=mr.merged_at,
				message=f"Merged {mr.branch_name}",
				event_type="merged",
			))
		elif mr.status == "conflict" and mr.verified_at:
			events.append(DashboardEvent(
				timestamp=mr.verified_at,
				message=f"Conflict on {mr.branch_name}",
				event_type="failed",
			))

	# Sort by timestamp descending, take most recent 20
	events.sort(key=lambda e: e.timestamp, reverse=True)
	return events[:20]
