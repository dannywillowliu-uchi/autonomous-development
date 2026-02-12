"""Coordinator -- top-level orchestrator for parallel execution."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from mission_control.backends.local import LocalBackend
from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.merge_queue import MergeQueue
from mission_control.models import Plan, Worker, _new_id, _now_iso
from mission_control.planner import create_plan
from mission_control.state import snapshot_project_health
from mission_control.worker import WorkerAgent

logger = logging.getLogger(__name__)

MONITOR_INTERVAL = 5  # seconds between progress checks


@dataclass
class CoordinatorReport:
	"""Summary of a parallel execution run."""

	plan_id: str = ""
	total_units: int = 0
	units_completed: int = 0
	units_failed: int = 0
	units_merged: int = 0
	units_rejected: int = 0
	workers_spawned: int = 0
	total_cost_usd: float = 0.0
	wall_time_seconds: float = 0.0
	stopped_reason: str = ""


class Coordinator:
	"""Top-level orchestrator for parallel execution mode.

	Flow:
	1. Take initial snapshot
	2. Run planner agent -> Plan with WorkUnits
	3. Initialize workspace pool (warm clones)
	4. Start merge queue (dedicated clone, asyncio task)
	5. Start N workers (each in own clone, asyncio tasks)
	6. Monitor: check plan progress, budget, stale workers
	7. When all units done + merge queue drained -> shutdown
	8. Cleanup pool
	"""

	def __init__(
		self,
		config: MissionConfig,
		db: Database,
		num_workers: int | None = None,
		pool_dir: str | None = None,
	) -> None:
		self.config = config
		self.db = db
		self.num_workers = num_workers or config.scheduler.parallel.num_workers
		self.pool_dir = pool_dir or config.scheduler.parallel.pool_dir
		self.running = True
		self._workers: list[WorkerAgent] = []
		self._worker_tasks: list[asyncio.Task[None]] = []
		self._merge_queue: MergeQueue | None = None
		self._merge_task: asyncio.Task[None] | None = None
		self._backend: LocalBackend | None = None

	async def run(self) -> CoordinatorReport:
		"""Execute the full parallel workflow."""
		report = CoordinatorReport()
		start_time = time.monotonic()

		source_repo = str(self.config.target.resolved_path)
		pool_path = self.pool_dir or str(Path(source_repo).parent / ".mc-pool")

		try:
			# 1. Take initial snapshot
			logger.info("Taking initial snapshot...")
			snapshot = await snapshot_project_health(self.config)

			# 2. Run planner
			logger.info("Running planner agent...")
			plan = await create_plan(self.config, snapshot, self.db)
			report.plan_id = plan.id
			report.total_units = plan.total_units

			if plan.total_units == 0:
				report.stopped_reason = "planner_produced_no_units"
				report.wall_time_seconds = time.monotonic() - start_time
				return report

			plan.status = "active"
			self.db.update_plan(plan)

			# 3. Initialize backend (single pool for workers + merge queue)
			logger.info("Initializing workspace pool with %d clones...", self.num_workers)
			self._backend = LocalBackend(
				source_repo=source_repo,
				pool_dir=pool_path,
				max_clones=self.num_workers + 1,  # +1 for merge queue
				base_branch=self.config.target.branch,
			)
			warm = self.config.scheduler.parallel.warm_clones
			await self._backend.initialize(warm_count=min(warm, self.num_workers + 1))

			# 4. Start merge queue in a dedicated clone
			merge_workspace = await self._backend._pool.acquire()
			if merge_workspace is None:
				report.stopped_reason = "failed_to_acquire_merge_workspace"
				report.wall_time_seconds = time.monotonic() - start_time
				return report

			self._merge_queue = MergeQueue(self.config, self.db, str(merge_workspace))
			self._merge_task = asyncio.create_task(self._merge_queue.run())

			# 5. Start N workers
			for i in range(self.num_workers):
				workspace = await self._backend._pool.acquire()
				if workspace is None:
					logger.warning("Could not acquire workspace for worker %d", i)
					break

				worker = Worker(
					id=_new_id(),
					workspace_path=str(workspace),
					status="idle",
				)
				self.db.insert_worker(worker)

				agent = WorkerAgent(worker, self.db, self.config, self._backend)
				self._workers.append(agent)
				self._worker_tasks.append(asyncio.create_task(agent.run()))
				report.workers_spawned += 1

			logger.info("Started %d workers", report.workers_spawned)

			# 6. Monitor progress
			await self._monitor_progress(plan, report)

			# 7. Finalize plan
			plan = self.db.get_plan(plan.id) or plan
			plan.status = "completed" if plan.failed_units == 0 else "failed"
			plan.finished_at = _now_iso()
			self.db.update_plan(plan)

		except (RuntimeError, OSError) as exc:
			logger.error("Coordinator infrastructure error: %s", exc, exc_info=True)
			report.stopped_reason = "error"
		except asyncio.CancelledError:
			logger.info("Coordinator cancelled")
			report.stopped_reason = "cancelled"
		except (ValueError, KeyError) as exc:
			logger.error("Coordinator data error: %s", exc, exc_info=True)
			report.stopped_reason = "error"
		finally:
			# Shutdown workers and merge queue
			await self._shutdown()

			# Cleanup backend and its pool
			if self._backend:
				await self._backend.cleanup()

			report.wall_time_seconds = time.monotonic() - start_time

		return report

	async def _monitor_progress(self, plan: Plan, report: CoordinatorReport) -> None:
		"""Monitor plan progress until all units are done."""
		heartbeat_timeout = self.config.scheduler.parallel.heartbeat_timeout

		while self.running:
			await asyncio.sleep(MONITOR_INTERVAL)

			# Recover stale units
			recovered = await self.db.locked_call("recover_stale_units", heartbeat_timeout)
			if recovered:
				logger.info("Recovered %d stale units", len(recovered))

			# Count unit statuses
			units = await self.db.locked_call("get_work_units_for_plan", plan.id)
			completed = sum(1 for u in units if u.status == "completed")
			failed = sum(1 for u in units if u.status == "failed" and u.attempt >= u.max_attempts)
			pending_or_running = sum(1 for u in units if u.status in ("pending", "claimed", "running"))

			plan.completed_units = completed
			plan.failed_units = failed
			await self.db.locked_call("update_plan", plan)

			report.units_completed = completed
			report.units_failed = failed

			# Count merge request outcomes
			mrs = await self.db.locked_call("get_merge_requests_for_plan", plan.id)
			report.units_merged = sum(1 for mr in mrs if mr.status == "merged")
			report.units_rejected = sum(1 for mr in mrs if mr.status == "rejected")

			# Check if done
			if pending_or_running == 0:
				# Wait a bit for merge queue to drain
				await asyncio.sleep(2)
				remaining_mrs = await self.db.locked_call("get_next_merge_request")
				if remaining_mrs is None:
					report.stopped_reason = "all_units_done"
					return

			# Check all workers dead
			if all(not agent.running for agent in self._workers):
				report.stopped_reason = "all_workers_stopped"
				return

	async def _shutdown(self) -> None:
		"""Stop all workers and the merge queue."""
		# Stop workers
		for agent in self._workers:
			agent.stop()

		# Wait for worker tasks to finish
		for task in self._worker_tasks:
			task.cancel()
			try:
				await task
			except asyncio.CancelledError:
				pass

		# Stop merge queue
		if self._merge_queue:
			self._merge_queue.stop()
		if self._merge_task:
			self._merge_task.cancel()
			try:
				await self._merge_task
			except asyncio.CancelledError:
				pass

	def stop(self) -> None:
		self.running = False
