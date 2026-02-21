"""Continuous controller -- event-driven development loop without round boundaries."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mission_control.backends import ContainerBackend, LocalBackend, SSHBackend, WorkerBackend
from mission_control.backlog_manager import BacklogManager
from mission_control.circuit_breaker import CircuitBreakerManager
from mission_control.config import ContinuousConfig, MissionConfig, claude_subprocess_env
from mission_control.constants import (
	UNIT_EVENT_DEGRADATION_TRANSITION,
	UNIT_EVENT_DISPATCHED,
	UNIT_EVENT_EXPERIMENT_COMPLETED,
	UNIT_EVENT_MERGE_FAILED,
	UNIT_EVENT_MERGED,
	UNIT_EVENT_REJECTED,
	UNIT_EVENT_RESEARCH_COMPLETED,
	UNIT_EVENT_RETRY_QUEUED,
)
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.degradation import DegradationManager, DegradationTransition
from mission_control.diff_reviewer import DiffReviewer
from mission_control.ema import ExponentialMovingAverage
from mission_control.event_stream import EventStream
from mission_control.feedback import get_worker_context
from mission_control.green_branch import GreenBranchManager
from mission_control.heartbeat import Heartbeat
from mission_control.json_utils import extract_json_from_text
from mission_control.models import (
	BacklogItem,
	Epoch,
	ExperimentResult,
	Handoff,
	Mission,
	Signal,
	UnitEvent,
	UnitReview,
	Worker,
	WorkUnit,
	_now_iso,
)
from mission_control.notifier import TelegramNotifier
from mission_control.overlap import _parse_files_hint, topological_layers
from mission_control.planner_context import build_planner_context, update_mission_state
from mission_control.session import parse_mc_result
from mission_control.strategist import Strategist
from mission_control.token_parser import compute_token_cost, parse_stream_json
from mission_control.tracing import MissionTracer
from mission_control.worker import load_specialist_template, render_mission_worker_prompt

logger = logging.getLogger(__name__)


@dataclass
class WorkerCompletion:
	"""A completed unit ready for verification and merge."""

	unit: WorkUnit
	handoff: Handoff | None
	workspace: str
	epoch: Epoch


@dataclass
class _RoundTracker:
	"""Tracks unit outcomes within a single dispatch round."""

	unit_ids: set[str]
	completed_ids: set[str]
	failed_ids: set[str]

	@property
	def all_resolved(self) -> bool:
		return self.completed_ids | self.failed_ids == self.unit_ids

	@property
	def all_failed(self) -> bool:
		return self.all_resolved and self.failed_ids == self.unit_ids


@dataclass
class ContinuousMissionResult:
	"""Summary of a completed continuous mission."""

	mission_id: str = ""
	objective: str = ""
	objective_met: bool = False
	total_units_dispatched: int = 0
	total_units_merged: int = 0
	total_units_failed: int = 0
	wall_time_seconds: float = 0.0
	stopped_reason: str = ""
	final_verification_passed: bool | None = None
	final_verification_output: str = ""
	backlog_item_ids: list[str] | None = None
	ambition_score: int = 0
	next_objective: str = ""
	proposed_by_strategist: bool = False
	db_errors: int = 0
	units_failed_unrecovered: int = 0
	sync_failures: int = 0
	degradation_level: str = "FULL_CAPACITY"


class ContinuousController:
	"""Event-driven controller: no round boundaries, continuous replanning.

	Architecture:
	  - dispatch loop: assigns work to free workers from the planner backlog
	  - completion processor: handles handoffs via asyncio.Queue, verify+merge,
	    records feedback, checks stopping conditions
	"""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db
		self.running = True
		self._backend: WorkerBackend | None = None
		self._green_branch: GreenBranchManager | None = None
		self._planner: ContinuousPlanner | None = None
		self._notifier: TelegramNotifier | None = None
		self._heartbeat: Heartbeat | None = None
		self._completion_queue: asyncio.Queue[WorkerCompletion] = asyncio.Queue()
		self._active_tasks: set[asyncio.Task[None]] = set()
		self._unit_tasks: dict[str, asyncio.Task[None]] = {}  # unit_id -> task
		self._semaphore: asyncio.Semaphore | None = None
		self._paused: bool = False
		self._start_time: float = 0.0
		self._state_changelog: list[str] = []
		self._total_dispatched: int = 0
		self._total_merged: int = 0
		self._total_failed: int = 0
		self._event_stream: EventStream | None = None
		self._consecutive_all_fail_rounds: int = 0
		self._round_tracker: dict[str, _RoundTracker] = {}  # epoch_id -> tracker
		self._failure_backoff_until: float = 0.0
		self._all_fail_stop_reason: str = ""
		self._in_flight_count: int = 0
		self._deferred_units: list[tuple[WorkUnit, Epoch, Mission]] = []
		self._merged_files: set[str] = set()  # files covered by merged units
		self._completed_unit_ids: set[str] = set()  # units that completed/merged successfully
		self._last_reconcile_count: int = 0
		self.ambition_score: float = 0.0
		self.proposed_by_strategist: bool = False
		self._objective_check_count: int = 0
		self._failed_unit_replan_count: int = 0
		self._strategist: Strategist | None = None
		self._is_cleanup_mission: bool = config.target.objective.startswith("[CLEANUP]")
		self._diff_reviewer: DiffReviewer = DiffReviewer(config)
		self._backlog_manager: BacklogManager = BacklogManager(db, config)
		cont = config.continuous
		self._circuit_breakers: CircuitBreakerManager = CircuitBreakerManager(
			max_failures=cont.circuit_breaker_max_failures,
			cooldown_seconds=float(cont.circuit_breaker_cooldown_seconds),
		)
		self._degradation: DegradationManager = DegradationManager(
			config=config.degradation,
			on_transition=self._on_degradation_transition,
		)
		self._tracer: MissionTracer = MissionTracer(config.tracing)
		budget = config.scheduler.budget
		self._ema: ExponentialMovingAverage = ExponentialMovingAverage(
			alpha=budget.ema_alpha,
			outlier_multiplier=budget.outlier_multiplier,
			conservatism_base=budget.conservatism_base,
		)

	def _task_done_callback(self, task: asyncio.Task[None]) -> None:
		"""Callback for fire-and-forget tasks: discard from tracking set and log exceptions."""
		self._active_tasks.discard(task)
		if task.cancelled():
			return
		exc = task.exception()
		if exc is not None:
			logger.error("Fire-and-forget task failed: %s", exc, exc_info=exc)

	def _record_db_error(self) -> None:
		"""Track a DB error via degradation manager."""
		self._degradation.record_db_error()

	def _record_db_success(self) -> None:
		"""Record a DB success via degradation manager."""
		self._degradation.record_db_success()

	def _on_degradation_transition(self, transition: DegradationTransition) -> None:
		"""Emit event and persist degradation level on transition."""
		if self._event_stream:
			self._event_stream.emit(
				UNIT_EVENT_DEGRADATION_TRANSITION,
				details={
					"from": transition.from_level.name,
					"to": transition.to_level.name,
					"trigger": transition.trigger,
				},
			)
		try:
			self.db.conn.execute(
				"UPDATE missions SET degradation_level = ? WHERE status = 'running'",
				(transition.to_level.name,),
			)
			self.db.conn.commit()
		except Exception as exc:
			logger.debug("Failed to persist degradation level: %s", exc)

	def _log_unit_event(
		self,
		*,
		mission_id: str,
		epoch_id: str,
		work_unit_id: str,
		event_type: str,
		details: str | None = None,
		input_tokens: int = 0,
		output_tokens: int = 0,
		stream_details: dict[str, object] | None = None,
		cost_usd: float = 0.0,
	) -> None:
		"""Insert a UnitEvent into the DB and emit to the JSONL event stream."""
		if not self._degradation.is_db_degraded:
			try:
				self.db.insert_unit_event(UnitEvent(
					mission_id=mission_id,
					epoch_id=epoch_id,
					work_unit_id=work_unit_id,
					event_type=event_type,
					details=details or "",
					input_tokens=input_tokens,
					output_tokens=output_tokens,
				))
				self._record_db_success()
			except Exception as exc:
				logger.debug("Failed to insert unit event: %s", exc)
				self._record_db_error()
		if self._event_stream:
			kwargs: dict[str, object] = {
				"mission_id": mission_id,
				"epoch_id": epoch_id,
				"unit_id": work_unit_id,
			}
			if input_tokens:
				kwargs["input_tokens"] = input_tokens
			if output_tokens:
				kwargs["output_tokens"] = output_tokens
			if cost_usd:
				kwargs["cost_usd"] = cost_usd
			if stream_details is not None:
				kwargs["details"] = stream_details
			self._event_stream.emit(event_type, **kwargs)

	async def run(self, dry_run: bool = False) -> ContinuousMissionResult:
		"""Run the continuous mission loop until objective met or stopping condition."""
		result = ContinuousMissionResult(objective=self.config.target.objective)

		if dry_run:
			await self._init_components()
			if self._planner is None:
				raise RuntimeError("Controller not initialized: call start() first")
			mission = Mission(objective=self.config.target.objective, status="dry_run")
			_, units, _ = await self._planner.get_next_units(
				mission, max_units=10, feedback_context="",
			)
			if not units:
				print("Planner returned no units.")
				return result
			# Compute parallelism levels via topological layering
			unit_ids = {u.id for u in units}
			levels = topological_layers(units)
			print(f"\nDry-run plan: {len(units)} units")
			print(f"{'#':<4} {'Title':<50} {'Priority':>8} {'Files'}")
			print("-" * 90)
			for i, u in enumerate(units, 1):
				files = u.files_hint[:60] if u.files_hint else ""
				deps_str = ""
				if u.depends_on:
					dep_ids = [d.strip() for d in u.depends_on.split(",") if d.strip()]
					valid_deps = [d[:8] for d in dep_ids if d in unit_ids]
					if valid_deps:
						deps_str = f" [depends: {', '.join(valid_deps)}]"
				print(f"{i:<4} {u.title[:48]:<50} {u.priority:>8} {files}{deps_str}")
			print(f"\nEstimated parallelism: {len(levels)} layer(s), max {max(len(lv) for lv in levels)} concurrent")
			if self._backend:
				await self._backend.cleanup()
			return result

		self._start_time = time.monotonic()

		mission = Mission(
			objective=self.config.target.objective,
			status="running",
		)
		try:
			self.db.insert_mission(mission)
		except Exception as exc:
			logger.error("Failed to insert mission: %s", exc, exc_info=True)
			result.stopped_reason = "db_error"
			return result
		result.mission_id = mission.id

		# Clean up orphaned work units from previously killed missions.
		# When a mission is killed, units with status='running' are never
		# updated. New missions see them as in-flight and defer everything.
		try:
			orphaned = self.db.conn.execute(
				"UPDATE work_units SET status='failed' "
				"WHERE status='running' OR status='pending'",
			)
			if orphaned.rowcount:
				self.db.conn.commit()
				logger.info(
					"Cleaned up %d orphaned work units from prior missions",
					orphaned.rowcount,
				)
		except Exception as exc:
			logger.warning("Failed to clean orphaned work units: %s", exc)

		# Generate initial MISSION_STATE.md with objective and empty sections
		try:
			update_mission_state(self.db, mission, self.config, self._state_changelog)
		except Exception as exc:
			logger.warning("Failed to write initial MISSION_STATE.md: %s", exc)

		try:
			await self._init_components()
			self._mission_span_ctx = self._tracer.start_mission_span(mission.id)
			self._mission_span = self._mission_span_ctx.__enter__()

			# Load backlog items as objective if discovery is enabled
			if self.config.discovery.enabled:
				backlog_objective = self._backlog_manager.load_backlog_objective()
				if backlog_objective:
					if self.config.target.objective:
						self.config.target.objective += "\n\n" + backlog_objective
					else:
						self.config.target.objective = backlog_objective
					mission.objective = self.config.target.objective
					result.objective = mission.objective
					try:
						self.db.update_mission(mission)
					except Exception as exc:
						logger.error("Failed to update mission objective: %s", exc)

			# Pass loaded backlog items to planner for richer planning context
			if self._planner and self._backlog_manager.backlog_item_ids:
				items = [self.db.get_backlog_item(bid) for bid in self._backlog_manager.backlog_item_ids]
				self._planner.set_backlog_items([i for i in items if i is not None])

			# Initialize JSONL event stream
			target_dir = Path(self.config.target.resolved_path)
			self._event_stream = EventStream(target_dir / "events.jsonl")
			self._event_stream.open()
			self._event_stream.emit(
				"mission_started",
				mission_id=mission.id,
				details={
					"objective": self.config.target.objective,
					"workers": self.config.scheduler.parallel.num_workers,
				},
			)

			if self._notifier:
				await self._notifier.send_mission_start(
					self.config.target.objective,
					self.config.scheduler.parallel.num_workers,
				)

			# Two concurrent tasks
			dispatch_task = asyncio.create_task(
				self._dispatch_loop(mission, result),
			)
			processor_task = asyncio.create_task(
				self._process_completions(mission, result),
			)

			done, pending = await asyncio.wait(
				[dispatch_task, processor_task],
				return_when=asyncio.FIRST_COMPLETED,
			)

			# Cancel pending tasks
			for task in pending:
				task.cancel()
				try:
					await task
				except asyncio.CancelledError:
					pass

			# Cancel any active worker tasks
			for task in list(self._active_tasks):
				task.cancel()
				try:
					await task
				except asyncio.CancelledError:
					pass

			# Check for exceptions from completed tasks
			for task in done:
				if task.exception():
					logger.error(
						"Controller task failed: %s",
						task.exception(), exc_info=task.exception(),
					)
					if not result.stopped_reason:
						result.stopped_reason = "error"

		except (RuntimeError, OSError) as exc:
			logger.error("Mission infrastructure error: %s", exc, exc_info=True)
			result.stopped_reason = "error"
		except asyncio.CancelledError:
			logger.info("Mission cancelled")
			result.stopped_reason = "cancelled"
		finally:
			# Run final verification on mc/green
			if self._green_branch and self._total_merged > 0:
				try:
					passed, output = await self._run_final_verification()
					result.final_verification_passed = passed
					result.final_verification_output = output
					logger.info(
						"Mission complete. Final verification: %s",
						"PASS" if passed else "FAIL",
					)
				except Exception as exc:
					logger.error(
						"Final verification error: %s", exc, exc_info=True,
					)
					result.final_verification_passed = False
					result.final_verification_output = str(exc)

			# Deploy at mission end if configured and verification passed
			deploy = self.config.deploy
			if (
				self._green_branch
				and deploy.enabled
				and deploy.on_mission_end
				and result.final_verification_passed
			):
				try:
					deploy_ok, deploy_output = await self._green_branch.run_deploy()
					if self._event_stream:
						self._event_stream.emit(
							"deploy_completed" if deploy_ok else "deploy_failed",
							mission_id=mission.id,
							details={"output": deploy_output[:500]},
						)
					if self._notifier:
						status_str = "succeeded" if deploy_ok else "failed"
						await self._notifier.send(
							f"Deploy {status_str}: {deploy_output[:200]}"
						)
					if not deploy_ok:
						logger.warning("Deploy failed: %s", deploy_output[:200])
				except Exception as exc:
					logger.error("Deploy error: %s", exc, exc_info=True)
					if self._event_stream:
						self._event_stream.emit(
							"deploy_failed",
							mission_id=mission.id,
							details={"error": str(exc)},
						)

			# Update backlog items based on mission outcome
			if self._backlog_manager.backlog_item_ids:
				try:
					handoffs = self.db.get_recent_handoffs(mission.id, limit=50)
					self._backlog_manager.update_backlog_on_completion(result.objective_met, handoffs)
				except Exception as exc:
					logger.error(
						"Failed to update backlog on completion: %s", exc, exc_info=True,
					)

			mission.status = "completed" if result.objective_met else "stopped"
			mission.finished_at = _now_iso()
			mission.stopped_reason = result.stopped_reason

			# Set strategist metadata on mission if fields exist
			if hasattr(mission, "ambition_score"):
				mission.ambition_score = self.ambition_score
			if hasattr(mission, "proposed_by_strategist"):
				mission.proposed_by_strategist = self.proposed_by_strategist

			try:
				self.db.update_mission(mission)
			except Exception as exc:
				logger.error(
					"Failed to update mission in finally: %s", exc, exc_info=True,
				)

			if self._backend:
				await self._backend.cleanup()

			result.wall_time_seconds = time.monotonic() - self._start_time
			result.total_units_dispatched = self._total_dispatched
			result.total_units_merged = self._total_merged
			result.total_units_failed = self._total_failed
			result.units_failed_unrecovered = self._total_failed
			result.db_errors = self._degradation.get_status_dict()["db_errors"]
			result.degradation_level = self._degradation.level_name

			if self._backlog_manager.backlog_item_ids:
				result.backlog_item_ids = list(self._backlog_manager.backlog_item_ids)

			try:
				from mission_control.mission_report import generate_mission_report
				generate_mission_report(result, mission, self.db, self.config)
			except Exception as exc:
				logger.error("Failed to generate mission report: %s", exc, exc_info=True)

			# Determine if follow-up work is needed (mission chaining)
			# Chain on both success and failure -- there's always more backlog to work through.
			try:
				remaining_backlog = self.db.get_pending_backlog(limit=5)
				if remaining_backlog:
					top_items = remaining_backlog[:3]
					descriptions = [
						f"[{item.track}] {item.title} (priority={item.priority_score:.1f})"
						for item in top_items
					]
					prefix = "Objective met. " if result.objective_met else ""
					result.next_objective = (
						f"{prefix}Continue with {len(remaining_backlog)} remaining backlog items. "
						f"Top priorities: {'; '.join(descriptions)}"
					)
					mission.next_objective = result.next_objective
					logger.info("Next objective set for mission chaining: %s", result.next_objective[:100])
			except Exception as exc:
				logger.error("Failed to determine next objective: %s", exc, exc_info=True)

			# Append strategic context for future strategist calls
			try:
				merged_summaries: list[str] = []
				failed_summaries: list[str] = []
				handoffs = self.db.get_recent_handoffs(mission.id, limit=50)
				for h in handoffs:
					summary_text = h.summary[:200] if h.summary else ""
					if h.status == "completed" and summary_text:
						merged_summaries.append(summary_text)
					elif summary_text:
						failed_summaries.append(summary_text)
				self.db.append_strategic_context(
					mission_id=mission.id,
					what_attempted=mission.objective[:500],
					what_worked="; ".join(merged_summaries[:10]) or "nothing merged",
					what_failed="; ".join(failed_summaries[:10]) or "no failures",
					recommended_next=result.stopped_reason or "continue",
				)
			except Exception as exc:
				logger.error("Failed to append strategic context: %s", exc, exc_info=True)

			if self._notifier:
				await self._notifier.send_mission_end(
					objective_met=result.objective_met,
					merged=result.total_units_merged,
					failed=result.total_units_failed,
					wall_time=result.wall_time_seconds,
					stopped_reason=result.stopped_reason,
					verification_passed=result.final_verification_passed,
				)

			if self._event_stream:
				self._event_stream.emit(
					"mission_ended",
					mission_id=mission.id,
					details={
						"status": mission.status,
						"stopped_reason": result.stopped_reason,
						"objective_met": result.objective_met,
						"units_merged": result.total_units_merged,
						"units_failed": result.total_units_failed,
						"wall_time_seconds": result.wall_time_seconds,
					},
				)
				self._event_stream.close()

			# Post-mission re-discovery
			if self.config.discovery.enabled and result.objective_met:
				await self._run_post_mission_discovery(mission.id)

			# Close the mission span
			if hasattr(self, "_mission_span_ctx") and self._mission_span_ctx:
				try:
					self._mission_span_ctx.__exit__(None, None, None)
				except Exception:
					pass

		return result

	def _score_ambition(self, units: list[WorkUnit]) -> int:
		"""Score planned work 1-10 based on heuristics.

		Factors: unit count, file diversity, unit_type mix, priority distribution.
		Returns integer 1-10 where 1-3 = busywork, 4-6 = moderate, 7-10 = ambitious.
		"""
		if not units:
			return 1

		count = len(units)

		# Factor 1: Unit count (0-2.5 points)
		if count >= 6:
			count_score = 2.5
		elif count >= 3:
			count_score = 1.5
		else:
			count_score = 0.5

		# Factor 2: File diversity (0-3 points)
		all_files: set[str] = set()
		for u in units:
			if u.files_hint:
				for f in u.files_hint.split(","):
					f = f.strip()
					if f:
						all_files.add(f)
		file_count = len(all_files)
		if file_count >= 10:
			file_score = 3.0
		elif file_count >= 5:
			file_score = 2.0
		elif file_count >= 2:
			file_score = 1.0
		else:
			file_score = 0.5

		# Factor 3: Unit type mix (0-2 points)
		types = {u.unit_type for u in units}
		type_score = 2.0 if len(types) > 1 else 0.5

		# Factor 4: Priority distribution (0-2.5 points, lower number = higher priority)
		avg_priority = sum(u.priority for u in units) / count
		if avg_priority <= 2:
			priority_score = 2.5
		elif avg_priority <= 4:
			priority_score = 1.5
		else:
			priority_score = 0.5

		raw = count_score + file_score + type_score + priority_score
		# raw range: 2.0 - 10.0; round and clamp
		score = max(1, min(10, round(raw)))
		return score

	async def _run_post_mission_discovery(self, mission_id: str = "") -> None:
		"""Run discovery after a successful mission to find new improvements."""
		try:
			from mission_control.auto_discovery import DiscoveryEngine

			engine = DiscoveryEngine(self.config, self.db)
			disc_result, items = await engine.discover()
			if items:
				logger.info(
					"Post-mission discovery found %d new items",
					len(items),
				)
				# Bridge discovery items into persistent backlog
				for disc_item in items:
					backlog_item = BacklogItem(
						title=disc_item.title,
						description=disc_item.description,
						priority_score=disc_item.priority_score,
						impact=disc_item.impact,
						effort=disc_item.effort,
						track=disc_item.track,
						source_mission_id=mission_id,
					)
					try:
						self.db.insert_backlog_item(backlog_item)
					except Exception as exc:
						logger.warning(
							"Failed to insert discovery item '%s' into backlog: %s",
							disc_item.title[:40], exc,
						)
				logger.info(
					"Inserted %d discovery items into persistent backlog",
					len(items),
				)
				if self._notifier:
					summary_lines = [
						f"- [{i.track}] {i.title} "
						f"(priority: {i.priority_score:.1f})"
						for i in items[:5]
					]
					await self._notifier.send(
						"Post-mission discovery: "
						f"{len(items)} new improvement items found.\n"
						+ "\n".join(summary_lines)
					)
			else:
				logger.info("Post-mission discovery found no new items")
		except Exception as exc:
			logger.error(
				"Post-mission discovery failed: %s", exc, exc_info=True,
			)

	async def _init_components(self) -> None:
		"""Initialize backend, green branch manager, and continuous planner."""
		source_repo = str(self.config.target.resolved_path)

		# Backend
		if self.config.backend.type == "ssh":
			self._backend = SSHBackend(self.config.backend.ssh_hosts)
		elif self.config.backend.type == "container":
			pool_dir = (
				self.config.scheduler.parallel.pool_dir
				or str(Path(source_repo).parent / ".mc-pool")
			)
			num_workers = self.config.scheduler.parallel.num_workers
			backend = ContainerBackend(
				source_repo=source_repo,
				pool_dir=pool_dir,
				container_config=self.config.backend.container,
				max_clones=num_workers + 1,
				base_branch=self.config.target.branch,
				max_output_mb=self.config.backend.max_output_mb,
			)
			await backend.initialize(
				warm_count=self.config.scheduler.parallel.warm_clones,
			)
			self._backend = backend
		else:
			pool_dir = (
				self.config.scheduler.parallel.pool_dir
				or str(Path(source_repo).parent / ".mc-pool")
			)
			num_workers = self.config.scheduler.parallel.num_workers
			backend = LocalBackend(
				source_repo=source_repo,
				pool_dir=pool_dir,
				max_clones=num_workers + 1,
				base_branch=self.config.target.branch,
			)
			await backend.initialize(
				warm_count=self.config.scheduler.parallel.warm_clones,
			)
			self._backend = backend

		# Green branch manager
		self._green_branch = GreenBranchManager(self.config, self.db)
		if isinstance(self._backend, (LocalBackend, ContainerBackend)):
			gb_workspace = await self._backend.provision_workspace(
				"green-branch-mgr", source_repo, self.config.target.branch,
			)
			await self._green_branch.initialize(gb_workspace)

			# Symlink source .venv into green branch workspace for verification
			source_venv = Path(source_repo) / ".venv"
			workspace_venv = Path(gb_workspace) / ".venv"
			if source_venv.exists() and not workspace_venv.exists():
				workspace_venv.symlink_to(source_venv)
				logger.info("Symlinked .venv into green branch workspace")

			# Run setup command if configured (e.g., npm install for non-Python projects)
			setup_cmd = self.config.target.verification.setup_command
			if setup_cmd:
				setup_timeout = self.config.target.verification.setup_timeout
				logger.info("Running verification setup: %s", setup_cmd)
				proc = await asyncio.create_subprocess_exec(
					*shlex.split(setup_cmd),
					cwd=gb_workspace,
					stdout=asyncio.subprocess.PIPE,
					stderr=asyncio.subprocess.STDOUT,
				)
				try:
					stdout, _ = await asyncio.wait_for(
						proc.communicate(), timeout=setup_timeout,
					)
				except asyncio.TimeoutError:
					try:
						proc.kill()
						await proc.wait()
					except ProcessLookupError:
						pass
					raise RuntimeError(
						f"Verification setup timed out after {setup_timeout}s: {setup_cmd}"
					)
				if proc.returncode != 0:
					output = stdout.decode() if stdout else ""
					raise RuntimeError(
						f"Verification setup failed (exit {proc.returncode}): {output[:500]}"
					)
		else:
			raise NotImplementedError(
				"Continuous mode requires a local workspace for green branch "
				"operations. SSH backend is not yet supported."
			)

		# Continuous planner (wraps RecursivePlanner)
		self._planner = ContinuousPlanner(self.config, self.db)

		# Telegram notifier (optional)
		tg = self.config.notifications.telegram
		if tg.bot_token and tg.chat_id:
			self._notifier = TelegramNotifier(tg.bot_token, tg.chat_id)

		# HITL approval gate (optional)
		hitl = self.config.hitl
		if hitl.push_gate.enabled or hitl.large_merge_gate.enabled:
			from mission_control.hitl import ApprovalGate
			gate = ApprovalGate(self.config, self._notifier)
			if self._green_branch:
				self._green_branch.configure_hitl(gate)

		# Heartbeat monitor
		hb = self.config.heartbeat
		self._heartbeat = Heartbeat(
			interval=hb.interval,
			idle_threshold=hb.idle_threshold,
			notifier=self._notifier,
			db=self.db,
			enable_recovery=hb.enable_recovery,
		)

	def _get_locked_files(self) -> dict[str, list[str]]:
		"""Return files locked by running units and already merged.

		Keys are file paths, values are lists of reasons (e.g. "in-flight: Build API").
		"""
		locked: dict[str, list[str]] = {}
		# In-flight files
		try:
			rows = self.db.conn.execute(
				"SELECT * FROM work_units WHERE status='running'",
			).fetchall()
			for r in rows:
				unit = Database._row_to_work_unit(r)
				files = _parse_files_hint(unit.files_hint)
				for f in files:
					locked.setdefault(f, []).append(f"in-flight: {unit.title}")
		except Exception as exc:
			logger.warning("Could not query running units for locked files: %s", exc)
		# Merged files
		for f in self._merged_files:
			locked.setdefault(f, []).append("already merged")
		return locked

	def _check_already_merged(self, unit: WorkUnit) -> bool:
		"""Check if a deferred unit's files have already been covered by a merged unit.

		Returns True if the unit should be dropped (work already done).
		"""
		if not self._merged_files:
			return False
		unit_files = _parse_files_hint(unit.files_hint)
		if not unit_files:
			return False
		# Drop if all of this unit's files are already covered by merged work
		return unit_files <= self._merged_files

	def _check_in_flight_overlap(self, unit: WorkUnit) -> bool:
		"""Check if a unit overlaps with any currently running units.

		Returns True if the unit should be deferred (overlap detected).
		"""
		try:
			rows = self.db.conn.execute(
				"SELECT * FROM work_units WHERE status='running' AND id != ?",
				(unit.id,),
			).fetchall()
			running_units = [Database._row_to_work_unit(r) for r in rows]
		except Exception as exc:
			logger.warning("Could not query running units for overlap check: %s", exc)
			return False

		unit_files = _parse_files_hint(unit.files_hint)

		for other in running_units:
			other_files = _parse_files_hint(other.files_hint)

			if unit_files and other_files:
				# Both have populated hints -- check file overlap
				if unit_files & other_files:
					return True
			elif not unit_files and not other_files:
				# Both unknown scope -- block concurrent execution
				return True
			# One empty, one populated -- allow (can't determine overlap)

		return False

	def _check_dependencies_met(self, unit: WorkUnit) -> bool:
		"""Return True if all depends_on IDs have completed (or if there are none)."""
		if not unit.depends_on:
			return True
		dep_ids = [d.strip() for d in unit.depends_on.split(",") if d.strip()]
		return all(dep_id in self._completed_unit_ids for dep_id in dep_ids)

	async def _dispatch_deferred(
		self,
		mission: Mission,
		epoch_map: dict[str, Epoch],
		semaphore: asyncio.Semaphore,
	) -> None:
		"""Try to dispatch previously deferred units that no longer overlap."""
		if not self._deferred_units:
			return

		still_deferred: list[tuple[WorkUnit, Epoch, Mission]] = []
		for unit, epoch, mission_ref in self._deferred_units:
			# Drop deferred units whose files are already covered by merged work
			if self._check_already_merged(unit):
				logger.info(
					"Dropping deferred unit %s (%s): files already covered by merged work",
					unit.id[:12], unit.title[:50],
				)
				continue
			if not self._check_dependencies_met(unit):
				still_deferred.append((unit, epoch, mission_ref))
				continue
			if self._check_in_flight_overlap(unit):
				still_deferred.append((unit, epoch, mission_ref))
				continue

			# No longer overlapping -- dispatch
			logger.info(
				"Dispatching previously deferred unit %s: %s",
				unit.id[:12], unit.title[:60],
			)
			await semaphore.acquire()
			self._in_flight_count += 1
			try:
				self.db.insert_work_unit(unit)
			except Exception as exc:
				logger.error(
					"Failed to insert deferred work unit: %s", exc, exc_info=True,
				)
				self._in_flight_count -= 1
				semaphore.release()
				continue

			self._log_unit_event(
				mission_id=mission_ref.id,
				epoch_id=epoch.id,
				work_unit_id=unit.id,
				event_type=UNIT_EVENT_DISPATCHED,
				stream_details={"title": unit.title, "files": unit.files_hint, "deferred": True},
			)
			self._total_dispatched += 1

			task = asyncio.create_task(
				self._execute_single_unit(unit, epoch, mission_ref, semaphore),
			)
			self._active_tasks.add(task)
			self._unit_tasks[unit.id] = task

			def _on_task_done(t: asyncio.Task[None], uid: str = unit.id) -> None:
				self._active_tasks.discard(t)
				self._unit_tasks.pop(uid, None)
				if not t.cancelled():
					exc = t.exception()
					if exc is not None:
						logger.error("Fire-and-forget task failed: %s", exc, exc_info=exc)

			task.add_done_callback(_on_task_done)

		self._deferred_units = still_deferred

	async def _dispatch_loop(
		self,
		mission: Mission,
		result: ContinuousMissionResult,
	) -> None:
		"""Dispatch work units to free workers as they become available."""
		if self._planner is None:
			raise RuntimeError("Controller not initialized: call start() first")
		if self._backend is None:
			raise RuntimeError("Controller not initialized: call start() first")

		num_workers = self.config.scheduler.parallel.num_workers
		semaphore = asyncio.Semaphore(num_workers)
		self._semaphore = semaphore
		cooldown = self.config.continuous.cooldown_between_units
		epoch_map: dict[str, Epoch] = {}

		while self.running:
			# READ_ONLY guard: skip dispatch, wait for in-flight to drain
			if self._degradation.is_read_only:
				self._degradation.check_in_flight_drained(self._in_flight_count)
				await asyncio.sleep(1)
				continue

			# Try to dispatch previously deferred units before planning new ones
			await self._dispatch_deferred(mission, epoch_map, semaphore)

			# Honor pause state
			if self._paused:
				await asyncio.sleep(1)
				# Still check signals while paused (for resume/stop)
				reason = self._check_signals(mission.id)
				if reason:
					result.stopped_reason = reason
					self.running = False
					break
				continue

			# Honor failure backoff (auto-pause after all-fail round)
			if self._failure_backoff_until > 0 and time.monotonic() < self._failure_backoff_until:
				await asyncio.sleep(1)
				continue
			self._failure_backoff_until = 0.0

			# Expire stale signals
			try:
				self.db.expire_stale_signals(timeout_minutes=10)
			except Exception as exc:
				logger.debug("Failed to expire stale signals: %s", exc)

			# Check stopping conditions before dispatching
			reason = self._should_stop(mission)
			if not reason and self._heartbeat:
				reason = await self._heartbeat.check(
					self._total_merged, self._total_failed,
				)
			if reason == "heartbeat_recovered":
				# Recovery was already attempted in check() -- kill stuck workers and allow one more cycle
				for task in list(self._active_tasks):
					task.cancel()
				logger.warning(
					"Heartbeat recovery: killed %d active tasks, allowing one more cycle",
					len(self._active_tasks),
				)
				self._heartbeat._consecutive_idle = 0
				continue
			if reason:
				result.stopped_reason = reason
				self.running = False
				break

			# Build feedback context for the planner
			feedback_context = build_planner_context(self.db, mission.id)

			# Gather locked files for the planner
			locked_files = self._get_locked_files()

			# Get next batch of units from the planner
			try:
				plan, units, epoch = await self._planner.get_next_units(
					mission,
					max_units=min(num_workers, 3),
					feedback_context=feedback_context,
					locked_files=locked_files,
				)
			except Exception as exc:
				logger.error("Planner failed: %s", exc, exc_info=True)
				await asyncio.sleep(5)
				continue

			if not units:
				running_tasks = {uid: t for uid, t in self._unit_tasks.items() if not t.done()}
				if running_tasks:
					logger.info(
						"Planner returned no new units, but %d units still in-flight -- waiting",
						len(running_tasks),
					)
					await asyncio.gather(*running_tasks.values(), return_exceptions=True)
					continue

				# Gate: if units failed to merge, inform planner before accepting completion
				cont = self.config.continuous
				if self._total_failed > 0 and self._failed_unit_replan_count < cont.max_objective_checks:
					self._failed_unit_replan_count += 1
					try:
						all_units = await self.db.locked_call(
							"get_work_units_for_mission", mission.id,
						)
					except Exception as exc:
						logger.warning("Failed to query units for failure gate: %s", exc)
						all_units = []
					failed_units = [u for u in all_units if u.status == "failed"]
					if failed_units:
						rejection_summary = "\n".join(
							f"- {u.title} (files: {u.files_hint}): FAILED after {u.attempt} attempts"
							for u in failed_units
						)
						feedback_context += (
							f"\n\nWARNING: {len(failed_units)} work unit(s) FAILED and their changes "
							"are NOT in mc/green:\n"
							f"{rejection_summary}\n"
							"You MUST create replacement units for this unfinished work, or the objective "
							"is NOT complete. If you believe the work is truly unnecessary, return empty units "
							"to confirm."
						)
						logger.warning(
							"%d failed units detected, re-entering planner (attempt %d/%d)",
							len(failed_units), self._failed_unit_replan_count, cont.max_objective_checks,
						)
						continue

				# Optional: verify the objective is actually met before declaring done
				if cont.verify_objective_completion:
					check = await self._verify_objective(
						mission, feedback_context,
					)
					if check is not None and not check["met"]:
						feedback_context += (
							f"\n\nOBJECTIVE NOT MET: {check['reason']}. "
							"Continue working toward the objective."
						)
						logger.warning(
							"Objective verification failed: %s -- re-entering planning loop",
							check["reason"],
						)
						continue

				if self._total_failed > 0:
					logger.warning(
						"Completing with %d unrecovered failed units after %d replan attempts",
						self._total_failed, self._failed_unit_replan_count,
					)

				logger.info("Planner returned no units and no in-flight work -- objective complete")
				result.objective_met = True
				result.stopped_reason = "planner_completed"
				self.running = False
				if self._notifier:
					notify_msg = "Mission complete: planner returned no more work units."
					if self._total_failed > 0:
						notify_msg += (
							f" WARNING: {self._total_failed} unit(s) failed and were not recovered."
						)
					await self._notifier.send(notify_msg)
				break

			# Score ambition of planned work -- enforce minimum
			if self._strategist:
				ambition = await self._strategist.evaluate_ambition(units)
			else:
				ambition = self._score_ambition(units)
			self.ambition_score = ambition
			mission.ambition_score = ambition
			result.ambition_score = ambition
			logger.info("Ambition score for planned units: %d/10", ambition)

			min_ambition = self.config.continuous.min_ambition_score
			max_replans = self.config.continuous.max_replan_attempts
			replan_attempts = 0

			while ambition < min_ambition and replan_attempts < max_replans:
				replan_attempts += 1
				logger.warning(
					"Ambition score %d < minimum %d (attempt %d/%d) -- replanning",
					ambition, min_ambition, replan_attempts, max_replans,
				)
				rejection_feedback = (
					f"PREVIOUS PLAN REJECTED: ambition score {ambition}/10 is below the "
					f"minimum threshold of {min_ambition}. Plan more impactful work -- "
					f"avoid trivial lint/quality fixes and target architecture improvements "
					f"or feature additions instead."
				)
				enriched_context = feedback_context + "\n\n" + rejection_feedback
				try:
					plan, units, epoch = await self._planner.get_next_units(
						mission,
						max_units=min(num_workers, 3),
						feedback_context=enriched_context,
						locked_files=locked_files,
					)
				except Exception as exc:
					logger.error("Replan failed: %s", exc, exc_info=True)
					break

				if not units:
					break

				if self._strategist:
					ambition = await self._strategist.evaluate_ambition(units)
				else:
					ambition = self._score_ambition(units)
				self.ambition_score = ambition
				mission.ambition_score = ambition
				result.ambition_score = ambition
				logger.info("Replanned ambition score: %d/10", ambition)

			if ambition < min_ambition:
				logger.info(
					"Proceeding with ambition %d after %d replan attempts (max %d)",
					ambition, replan_attempts, max_replans,
				)

			try:
				self.db.update_mission(mission)
			except Exception as exc:
				logger.error("Failed to persist ambition score: %s", exc)

			# Persist plan and tree
			try:
				self.db.insert_plan(plan)
			except Exception as exc:
				logger.error("Failed to insert plan: %s", exc, exc_info=True)
				continue

			# Persist epoch
			try:
				self.db.insert_epoch(epoch)
			except Exception as exc:
				logger.error("Failed to insert epoch: %s", exc, exc_info=True)

			epoch_map[epoch.id] = epoch

			# Compute topological layers for observability
			layers = topological_layers(units)
			if layers:
				max_concurrent = max(len(lv) for lv in layers)
				logger.info(
					"Plan has %d layer(s), max %d concurrent units in layer 0",
					len(layers), max_concurrent,
				)

			# Track this round for all-fail detection
			self._round_tracker[epoch.id] = _RoundTracker(
				unit_ids={u.id for u in units},
				completed_ids=set(),
				failed_ids=set(),
			)

			# Build unit->layer index mapping for event metadata
			unit_layer_map: dict[str, int] = {}
			for layer_idx, layer_units in enumerate(layers):
				for lu in layer_units:
					unit_layer_map[lu.id] = layer_idx

			# Dispatch each unit
			for unit in units:
				unit.epoch_id = epoch.id

				# Dependency gate: defer if depends_on IDs haven't completed
				if not self._check_dependencies_met(unit):
					logger.info(
						"Deferring unit %s: dependencies not yet met",
						unit.id[:12],
					)
					self._deferred_units.append((unit, epoch, mission))
					continue

				# Skip units whose files are already covered by merged work
				if self._check_already_merged(unit):
					logger.info(
						"Skipping unit %s (%s): files already covered by merged work",
						unit.id[:12], unit.title[:50],
					)
					continue

				# Cross-epoch overlap check: defer if overlapping with in-flight work
				if self._check_in_flight_overlap(unit):
					logger.warning(
						"Deferring unit %s: overlaps with in-flight work",
						unit.id[:12],
					)
					self._deferred_units.append((unit, epoch, mission))
					continue

				logger.info("Waiting for semaphore to dispatch unit %s: %s", unit.id[:12], unit.title[:60])
				await semaphore.acquire()
				self._in_flight_count += 1
				logger.info("Semaphore acquired, dispatching unit %s", unit.id[:12])
				try:
					self.db.insert_work_unit(unit)
				except Exception as exc:
					logger.error(
						"Failed to insert work unit: %s", exc, exc_info=True,
					)
					self._in_flight_count -= 1
					semaphore.release()
					continue

				# Log dispatch event with layer index metadata
				self._log_unit_event(
					mission_id=mission.id,
					epoch_id=epoch.id,
					work_unit_id=unit.id,
					event_type=UNIT_EVENT_DISPATCHED,
					stream_details={
						"title": unit.title,
						"files": unit.files_hint,
						"layer": unit_layer_map.get(unit.id, 0),
					},
				)

				self._total_dispatched += 1

				task = asyncio.create_task(
					self._execute_single_unit(
						unit, epoch, mission, semaphore,
					),
				)
				self._active_tasks.add(task)
				self._unit_tasks[unit.id] = task

				def _on_task_done(t: asyncio.Task[None], uid: str = unit.id) -> None:
					self._active_tasks.discard(t)
					self._unit_tasks.pop(uid, None)
					if not t.cancelled():
						exc = t.exception()
						if exc is not None:
							logger.error("Fire-and-forget task failed: %s", exc, exc_info=exc)

				task.add_done_callback(_on_task_done)

			if cooldown > 0:
				await asyncio.sleep(cooldown)

			# Degradation: check budget fraction for worker reduction
			budget_limit = self.config.scheduler.budget.max_per_run_usd
			if budget_limit > 0 and mission.total_cost_usd > 0:
				self._degradation.check_budget_fraction(mission.total_cost_usd, budget_limit)

			# Adaptive cooldown: increase sleep when costs approach budget
			if budget_limit > 0 and mission.total_cost_usd > 0:
				budget_fraction = mission.total_cost_usd / budget_limit
				if budget_fraction > 0.8:
					adaptive_sleep = cooldown + 30
					logger.info(
						"Adaptive cooldown: %.0f%% of budget used, sleeping %ds",
						budget_fraction * 100, adaptive_sleep,
					)
					await asyncio.sleep(adaptive_sleep)
				elif budget_fraction > 0.5:
					adaptive_sleep = cooldown + 10
					await asyncio.sleep(adaptive_sleep)

	async def _process_completions(
		self,
		mission: Mission,
		result: ContinuousMissionResult,
	) -> None:
		"""Process completed units: verify, merge, record feedback."""
		if self._green_branch is None:
			raise RuntimeError("Controller not initialized: call start() first")

		cont = self.config.continuous

		while self.running or not self._completion_queue.empty():
			try:
				completion = await asyncio.wait_for(
					self._completion_queue.get(), timeout=2.0,
				)
			except asyncio.TimeoutError:
				if not self.running and self._completion_queue.empty():
					break
				continue

			unit = completion.unit
			handoff = completion.handoff
			epoch = completion.epoch
			workspace = completion.workspace

			# Research units: skip merge, just record findings
			if unit.unit_type == "research":
				if unit.status != "completed":
					self._total_failed += 1
					logger.info("Research unit %s failed", unit.id)
					if handoff and self._planner:
						self._planner.ingest_handoff(handoff)
					mission.total_cost_usd += unit.cost_usd
					if unit.cost_usd > 0:
						self._ema.update(unit.cost_usd)
					try:
						self.db.update_mission(mission)
					except Exception as exc:
						logger.error("Failed to update mission cost for unit %s: %s", unit.id, exc)
					self._record_round_outcome(unit, epoch, merged=False)
					continue
				logger.info("Research unit %s completed -- skipping merge", unit.id)
				self._total_merged += 1
				self._completed_unit_ids.add(unit.id)
				self._log_unit_event(
					mission_id=mission.id,
					epoch_id=epoch.id,
					work_unit_id=unit.id,
					event_type=UNIT_EVENT_RESEARCH_COMPLETED,
				)

				if handoff and self._planner:
					self._planner.ingest_handoff(handoff)

				timestamp = unit.finished_at or _now_iso()
				summary = (handoff.summary[:80] if handoff and handoff.summary else unit.output_summary[:80])
				self._state_changelog.append(
					f"- {timestamp} | {unit.id[:8]} research completed -- {summary}"
				)

				try:
					update_mission_state(
						self.db, mission, self.config, self._state_changelog,
						degradation_status=self._degradation.get_status_dict(),
					)
				except Exception as exc:
					logger.warning("Failed to update MISSION_STATE.md: %s", exc)

				mission.total_cost_usd += unit.cost_usd
				if unit.cost_usd > 0:
					self._ema.update(unit.cost_usd)
				try:
					self.db.update_mission(mission)
				except Exception as exc:
					logger.error("Failed to update mission cost for unit %s: %s", unit.id, exc)
				self._record_round_outcome(unit, epoch, merged=True)
				continue

			# Experiment units: skip merge, store comparison report
			if unit.unit_type == "experiment":
				if unit.status != "completed":
					self._total_failed += 1
					logger.info("Experiment unit %s failed", unit.id)
					if handoff and self._planner:
						self._planner.ingest_handoff(handoff)
					mission.total_cost_usd += unit.cost_usd
					if unit.cost_usd > 0:
						self._ema.update(unit.cost_usd)
					try:
						self.db.update_mission(mission)
					except Exception as exc:
						logger.error("Failed to update mission cost for unit %s: %s", unit.id, exc)
					self._record_round_outcome(unit, epoch, merged=False)
					continue
				logger.info("Experiment unit %s completed -- skipping merge", unit.id)
				self._total_merged += 1
				self._completed_unit_ids.add(unit.id)

				# Parse comparison report from handoff and store as ExperimentResult
				comparison_report = ""
				recommended_approach = ""
				approach_count = 2
				if handoff:
					try:
						for item in handoff.discoveries:
							if isinstance(item, str) and "approach" in item.lower():
								comparison_report = item
								break
					except (json.JSONDecodeError, TypeError):
						pass
					# Also check the handoff summary for report data
					if handoff.summary:
						if not comparison_report:
							comparison_report = handoff.summary
						recommended_approach = handoff.summary[:200]

				try:
					experiment_result = ExperimentResult(
						work_unit_id=unit.id,
						epoch_id=epoch.id,
						mission_id=mission.id,
						approach_count=approach_count,
						comparison_report=comparison_report,
						recommended_approach=recommended_approach,
					)
					self.db.insert_experiment_result(experiment_result)
				except Exception as exc:
					logger.warning("Failed to insert experiment result for unit %s: %s", unit.id, exc)

				self._log_unit_event(
					mission_id=mission.id,
					epoch_id=epoch.id,
					work_unit_id=unit.id,
					event_type=UNIT_EVENT_EXPERIMENT_COMPLETED,
				)

				if handoff and self._planner:
					self._planner.ingest_handoff(handoff)

				timestamp = unit.finished_at or _now_iso()
				summary = (handoff.summary[:80] if handoff and handoff.summary else unit.output_summary[:80])
				self._state_changelog.append(
					f"- {timestamp} | {unit.id[:8]} experiment completed -- {summary}"
				)

				try:
					update_mission_state(
						self.db, mission, self.config, self._state_changelog,
						degradation_status=self._degradation.get_status_dict(),
					)
				except Exception as exc:
					logger.warning("Failed to update MISSION_STATE.md: %s", exc)

				mission.total_cost_usd += unit.cost_usd
				if unit.cost_usd > 0:
					self._ema.update(unit.cost_usd)
				try:
					self.db.update_mission(mission)
				except Exception as exc:
					logger.error("Failed to update mission cost for unit %s: %s", unit.id, exc)
				self._record_round_outcome(unit, epoch, merged=True)
				continue

			# Merge if the unit completed with commits
			merged = False
			if unit.status == "completed" and unit.commit_hash:
				try:
					merge_result = await self._green_branch.merge_unit(
						workspace, unit.branch_name,
					)
					merged = merge_result.merged
					self._degradation.record_merge_attempt(conflict=not merged)

					if merged:
						logger.info(
							"Unit %s merged to green",
							unit.id,
						)
						self._total_merged += 1
						self._completed_unit_ids.add(unit.id)
						# Track merged files so deferred duplicates are dropped
						# Use actual git diff files when available, fall back to files_hint
						if merge_result.changed_files:
							merged_files = set(merge_result.changed_files)
						else:
							merged_files = _parse_files_hint(unit.files_hint)
						if merged_files:
							self._merged_files |= merged_files
						# Log merge event
						self._log_unit_event(
							mission_id=mission.id,
							epoch_id=epoch.id,
							work_unit_id=unit.id,
							event_type=UNIT_EVENT_MERGED,
							input_tokens=unit.input_tokens,
							output_tokens=unit.output_tokens,
							cost_usd=unit.cost_usd,
						)
						# Check sync result
						if not merge_result.sync_ok:
							result.sync_failures += 1
							logger.warning("Unit %s merged but sync to source repo failed", unit.id)
							if self._notifier:
								await self._notifier.send(
									f"Sync warning: Unit '{unit.title}' merged to green but failed "
									"to sync back to source repo. Changes may not be visible in the source.",
								)
						# LLM diff review (blocking if gate_completion, else fire-and-forget)
						if self.config.review.enabled:
							if self.config.review.gate_completion:
								review = await self._blocking_review(
									unit, workspace, mission, epoch,
									merge_commit_hash=merge_result.merge_commit_hash,
								)
								if review and review.avg_score < self.config.review.min_review_score:
									logger.warning(
										"Unit %s review score %.1f below threshold %.1f, scheduling retry",
										unit.id, review.avg_score, self.config.review.min_review_score,
									)
									feedback = f"Review feedback (score {review.avg_score}): {review.rationale}"
									self._schedule_retry(unit, epoch, mission, feedback, cont)
									self._record_round_outcome(unit, epoch, merged=False)
									continue
							else:
								task = asyncio.create_task(
									self._review_merged_unit(
										unit, workspace, mission, epoch,
										merge_commit_hash=merge_result.merge_commit_hash,
									),
								)
								self._active_tasks.add(task)
								task.add_done_callback(self._task_done_callback)
					else:
						logger.warning(
							"Unit %s failed merge: %s",
							unit.id, merge_result.failure_output[-200:],
						)

						# Log merge_failed event
						_fail_details = {
							"failure_output": merge_result.failure_output[-2000:],
							"failure_stage": merge_result.failure_stage,
						}
						self._log_unit_event(
							mission_id=mission.id,
							epoch_id=epoch.id,
							work_unit_id=unit.id,
							event_type=UNIT_EVENT_MERGE_FAILED,
							details=json.dumps(_fail_details),
							stream_details=_fail_details,
						)

						# Append merge failure to handoff concerns
						if handoff:
							handoff.concerns.append(
								f"Merge failed: {merge_result.failure_output[-500:]}",
							)

						failure_reason = merge_result.failure_output[-1000:]

						# Check if retryable
						if unit.attempt < unit.max_attempts:
							self._schedule_retry(unit, epoch, mission, failure_reason, cont)
						else:
							self._total_failed += 1
							# Log rejection event
							self._log_unit_event(
								mission_id=mission.id,
								epoch_id=epoch.id,
								work_unit_id=unit.id,
								event_type=UNIT_EVENT_REJECTED,
								details=merge_result.failure_output[-2000:],
								stream_details={"failure_output": merge_result.failure_output[-2000:]},
							)

							# Notify merge conflict via Telegram
							if self._notifier:
								await self._notifier.send_merge_conflict(
									unit.title, merge_result.failure_output[-500:],
								)
				except Exception as exc:
					logger.error(
						"merge_unit failed for %s: %s",
						unit.id, exc, exc_info=True,
					)
					# Log merge_failed event for exception path
					_exc_details = {
						"failure_output": str(exc)[:500],
						"failure_stage": "exception",
					}
					self._log_unit_event(
						mission_id=mission.id,
						epoch_id=epoch.id,
						work_unit_id=unit.id,
						event_type=UNIT_EVENT_MERGE_FAILED,
						details=json.dumps(_exc_details),
						stream_details=_exc_details,
					)
					failure_reason = str(exc)[:300]
					if unit.attempt < unit.max_attempts:
						self._schedule_retry(unit, epoch, mission, failure_reason, cont)
					else:
						self._total_failed += 1
			elif unit.status == "completed":
				# Completed but no commits
				self._total_merged += 1
				merged = True
			else:
				# Unit failed during execution
				failure_reason = (unit.output_summary or "unknown error")[:300]
				if unit.attempt < unit.max_attempts:
					self._schedule_retry(unit, epoch, mission, failure_reason, cont)
				else:
					self._total_failed += 1

			# Update backlog items based on this unit's outcome
			try:
				self._backlog_manager.update_backlog_from_completion(
					unit, merged, handoff, mission.id,
				)
			except Exception as exc:
				logger.warning(
					"Failed to update backlog from completion for unit %s: %s",
					unit.id, exc,
				)

			# Feed handoff to planner for adaptive replanning
			if handoff and self._planner:
				self._planner.ingest_handoff(handoff)

			# Append changelog entry before updating state file
			timestamp = unit.finished_at or _now_iso()
			summary = (handoff.summary[:80] if handoff and handoff.summary else unit.output_summary[:80])
			if merged:
				commit_str = unit.commit_hash or "no-commit"
				self._state_changelog.append(
					f"- {timestamp} | {unit.id[:8]} merged (commit: {commit_str}) -- {summary}"
				)
			elif unit.status == "failed":
				self._state_changelog.append(
					f"- {timestamp} | {unit.id[:8]} failed -- {summary}"
				)

			# Update MISSION_STATE.md in target repo
			try:
				update_mission_state(
					self.db, mission, self.config, self._state_changelog,
					degradation_status=self._degradation.get_status_dict(),
				)
			except Exception as exc:
				logger.warning("Failed to update MISSION_STATE.md: %s", exc)

			# Accumulate cost and feed EMA tracker
			mission.total_cost_usd += unit.cost_usd
			if unit.cost_usd > 0:
				self._ema.update(unit.cost_usd)

			try:
				self.db.update_mission(mission)
				self._record_db_success()
			except Exception:
				self._record_db_error()

			# Track round outcomes for all-fail detection
			self._record_round_outcome(unit, epoch, merged)

			# Circuit breaker: record success/failure per workspace
			if self.config.continuous.circuit_breaker_enabled and workspace:
				if merged:
					self._circuit_breakers.record_success(workspace)
				else:
					self._circuit_breakers.record_failure(workspace)

			# Reconciler sweep: verify combined green state after new merges
			if merged and self._total_merged > self._last_reconcile_count:
				try:
					reconcile_ok, reconcile_output = await self._green_branch.run_reconciliation_check()
					if not reconcile_ok:
						logger.warning(
							"Reconciler sweep failed: %s", reconcile_output[-500:],
						)
						fixup_result = await self._green_branch.run_fixup(reconcile_output)
						if not fixup_result.success:
							logger.warning("Reconciler fixup failed")
							if self._notifier:
								await self._notifier.send(
									f"Reconciler fixup failed: {reconcile_output[-200:]}",
								)
					self._last_reconcile_count = self._total_merged
				except Exception as exc:
					logger.warning("Reconciler sweep error: %s", exc)

	def _record_round_outcome(self, unit: WorkUnit, epoch: Epoch, merged: bool) -> None:
		"""Record a unit outcome in its round tracker and handle all-fail rounds."""
		round_tracker = getattr(self, "_round_tracker", None)
		if round_tracker is None:
			return
		tracker = round_tracker.get(epoch.id)
		if tracker is None or unit.id not in tracker.unit_ids:
			return

		if merged:
			tracker.completed_ids.add(unit.id)
		else:
			tracker.failed_ids.add(unit.id)

		if not tracker.all_resolved:
			return

		# Round fully resolved -- check if all failed
		if tracker.all_failed:
			self._consecutive_all_fail_rounds += 1
			cont = self.config.continuous
			logger.warning(
				"All %d units in round %s failed (consecutive all-fail rounds: %d/%d)",
				len(tracker.unit_ids), epoch.id[:12],
				self._consecutive_all_fail_rounds, cont.max_consecutive_failures,
			)
			if self._consecutive_all_fail_rounds >= cont.max_consecutive_failures:
				logger.error(
					"Stopping mission: %d consecutive all-fail rounds",
					self._consecutive_all_fail_rounds,
				)
				self._all_fail_stop_reason = "repeated_total_failure"
			else:
				self._failure_backoff_until = time.monotonic() + cont.failure_backoff_seconds
				logger.info(
					"Auto-pausing for %ds before retry (all-fail round %d/%d)",
					cont.failure_backoff_seconds,
					self._consecutive_all_fail_rounds, cont.max_consecutive_failures,
				)
		else:
			self._consecutive_all_fail_rounds = 0

		# Circuit breaker stall: if all workspaces are open, stop mission
		if (
			self.config.continuous.circuit_breaker_enabled
			and self._circuit_breakers.all_open()
			and not self._all_fail_stop_reason
		):
			open_workspaces = self._circuit_breakers.get_open_workspaces()
			logger.error(
				"Stopping mission: all %d workspace circuit breakers are open: %s",
				len(open_workspaces),
				{ws: f"{count} failures" for ws, count in open_workspaces.items()},
			)
			self._all_fail_stop_reason = "all_circuit_breakers_open"

		# Clean up resolved round
		del round_tracker[epoch.id]

	async def _verify_objective(
		self,
		mission: Mission,
		feedback_context: str,
	) -> dict[str, Any] | None:
		"""LLM check: is the objective actually met? Returns {"met": bool, "reason": str} or None."""
		max_checks = self.config.continuous.max_objective_checks
		if self._objective_check_count >= max_checks:
			logger.info(
				"Objective check limit reached (%d/%d), accepting completion",
				self._objective_check_count, max_checks,
			)
			return None

		self._objective_check_count += 1
		logger.info(
			"Verifying objective completion (check %d/%d)",
			self._objective_check_count, max_checks,
		)

		# Gather unit summaries
		units = await self.db.locked_call("get_work_units_for_mission", mission.id)
		summaries = []
		for u in units:
			if u.status == "completed" and u.output_summary:
				summaries.append(f"- {u.title}: {u.output_summary[:200]}")
		summary_text = "\n".join(summaries[-20:]) or "No completed units."

		# Read MISSION_STATE.md if available
		state_path = self.config.target.resolved_path / "MISSION_STATE.md"
		mission_state = ""
		if state_path.exists():
			try:
				mission_state = state_path.read_text()[:4000]
			except OSError:
				pass

		prompt = f"""You are verifying whether a mission objective has been fully achieved.

## Objective
{mission.objective}

## Completed Work Summaries
{summary_text}

## Mission State
{mission_state or 'No MISSION_STATE.md found.'}

## Feedback Context
{feedback_context[:2000]}

## Instructions
Determine if the objective has been fully met based on the evidence above.
Consider: are there remaining gaps, unfinished features, or untested scenarios?

End your response with:
OBJECTIVE_CHECK:{{"met": true, "reason": "brief explanation"}}
or
OBJECTIVE_CHECK:{{"met": false, "reason": "what still needs to be done"}}"""

		model = getattr(self.config.models, "planner_model", None) or self.config.scheduler.model
		try:
			proc = await asyncio.create_subprocess_exec(
				"claude", "-p",
				"--output-format", "text",
				"--max-budget-usd", "0.50",
				"--model", model,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(),
				cwd=str(self.config.target.resolved_path),
			)
			stdout, _ = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
				timeout=120,
			)
			output = stdout.decode() if stdout else ""
		except (asyncio.TimeoutError, Exception) as exc:
			logger.warning("Objective verification failed: %s", exc)
			return None

		# Parse OBJECTIVE_CHECK marker
		marker = "OBJECTIVE_CHECK:"
		idx = output.rfind(marker)
		if idx == -1:
			logger.warning("No OBJECTIVE_CHECK marker found in output")
			return None

		remainder = output[idx + len(marker):]
		data = extract_json_from_text(remainder)
		if not isinstance(data, dict):
			logger.warning("Failed to parse OBJECTIVE_CHECK JSON")
			return None

		met = bool(data.get("met", True))
		reason = str(data.get("reason", ""))[:500]
		logger.info("Objective check result: met=%s reason=%s", met, reason)
		return {"met": met, "reason": reason}

	async def _blocking_review(
		self,
		unit: WorkUnit,
		workspace: str,
		mission: Mission,
		epoch: Epoch,
		merge_commit_hash: str = "",
	) -> UnitReview | None:
		"""Blocking review: returns UnitReview for gating decisions."""
		try:
			if merge_commit_hash:
				diff_cmd = ["git", "diff", f"{merge_commit_hash}^..{merge_commit_hash}", "--", "."]
			else:
				green_branch = self.config.green_branch.green_branch
				diff_cmd = ["git", "diff", f"{green_branch}~1..{green_branch}", "--", "."]
			proc = await asyncio.create_subprocess_exec(
				*diff_cmd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=str(self.config.target.resolved_path),
			)
			stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
			diff = stdout.decode() if stdout else ""

			if not diff.strip():
				logger.debug("No diff for unit %s, skipping blocking review", unit.id)
				return None

			review = await self._diff_reviewer.review_unit(
				unit=unit,
				diff=diff,
				objective=mission.objective,
				mission_id=mission.id,
				epoch_id=epoch.id,
			)
			if review:
				await self.db.locked_call("insert_unit_review", review)
				logger.info(
					"Blocking review for unit %s: alignment=%d approach=%d tests=%d criteria=%d avg=%.1f",
					unit.id, review.alignment_score, review.approach_score,
					review.test_score, review.criteria_met_score, review.avg_score,
				)
			return review
		except Exception as exc:
			logger.warning("Blocking review failed for unit %s: %s", unit.id, exc)
			return None

	async def _review_merged_unit(
		self,
		unit: WorkUnit,
		workspace: str,
		mission: Mission,
		epoch: Epoch,
		merge_commit_hash: str = "",
	) -> None:
		"""Fire-and-forget: review a merged unit's diff via LLM."""
		try:
			# Get the diff from the merge commit (not mc/green HEAD which may be MISSION_STATE.md)
			if merge_commit_hash:
				diff_cmd = ["git", "diff", f"{merge_commit_hash}^..{merge_commit_hash}", "--", "."]
			else:
				green_branch = self.config.green_branch.green_branch
				diff_cmd = ["git", "diff", f"{green_branch}~1..{green_branch}", "--", "."]
			proc = await asyncio.create_subprocess_exec(
				*diff_cmd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=str(self.config.target.resolved_path),
			)
			stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
			diff = stdout.decode() if stdout else ""

			if not diff.strip():
				logger.debug("No diff for unit %s, skipping review", unit.id)
				return

			review = await self._diff_reviewer.review_unit(
				unit=unit,
				diff=diff,
				objective=mission.objective,
				mission_id=mission.id,
				epoch_id=epoch.id,
			)
			if review:
				await self.db.locked_call("insert_unit_review", review)
				logger.info(
					"Review for unit %s: alignment=%d approach=%d tests=%d avg=%.1f",
					unit.id, review.alignment_score, review.approach_score,
					review.test_score, review.avg_score,
				)
		except Exception as exc:
			logger.warning("Review failed for unit %s: %s", unit.id, exc)

	def _schedule_retry(
		self,
		unit: WorkUnit,
		epoch: Epoch,
		mission: Mission,
		failure_reason: str,
		cont: ContinuousConfig,
	) -> None:
		"""Prepare a unit for retry and schedule delayed re-dispatch."""
		# Increment attempt BEFORE delay calculation
		unit.attempt += 1

		delay = min(
			cont.retry_base_delay_seconds * (2 ** (unit.attempt - 1)),
			cont.retry_max_delay_seconds,
		)

		# Append failure context to description
		unit.description += (
			f"\n\n[Retry attempt {unit.attempt}] Previous failure: "
			f"{failure_reason[-1000:]}. Avoid the same mistake."
		)

		# Reset unit for re-dispatch
		unit.status = "pending"
		unit.commit_hash = None
		unit.branch_name = ""
		unit.output_summary = ""

		# Persist updated attempt count and status
		try:
			self.db.update_work_unit(unit)
		except Exception as exc:
			logger.error("Failed to persist retry state for %s: %s", unit.id, exc)

		# Log retry event
		_retry_details = {"delay": delay, "failure_reason": failure_reason[-1000:]}
		self._log_unit_event(
			mission_id=mission.id,
			epoch_id=epoch.id,
			work_unit_id=unit.id,
			event_type=UNIT_EVENT_RETRY_QUEUED,
			details=json.dumps(_retry_details),
			stream_details=_retry_details,
		)

		logger.info(
			"Retrying unit %s (attempt %d/%d) after %.0fs delay",
			unit.id, unit.attempt, unit.max_attempts, delay,
		)

		# Schedule delayed re-dispatch
		task = asyncio.create_task(
			self._retry_unit(unit, epoch, mission, delay),
		)
		self._active_tasks.add(task)
		task.add_done_callback(self._task_done_callback)

	async def _retry_unit(
		self,
		unit: WorkUnit,
		epoch: Epoch,
		mission: Mission,
		delay: float,
	) -> None:
		"""Wait for backoff delay, then re-dispatch a failed unit."""
		await asyncio.sleep(delay)
		if not self.running:
			return
		if self._semaphore is None:
			raise RuntimeError("Controller not initialized: call start() first")
		await self._semaphore.acquire()
		self._in_flight_count += 1
		task = asyncio.create_task(
			self._execute_single_unit(unit, epoch, mission, self._semaphore),
		)
		self._active_tasks.add(task)
		task.add_done_callback(self._task_done_callback)

	async def _fail_unit(
		self,
		unit: WorkUnit,
		worker: Worker | None,
		epoch: Epoch,
		reason: str,
		workspace: str,
		put_on_queue: bool = True,
	) -> None:
		"""Mark a unit as failed, update the worker, and optionally queue completion."""
		unit.attempt += 1
		unit.status = "failed"
		unit.output_summary = reason
		unit.finished_at = _now_iso()
		try:
			await self.db.locked_call("update_work_unit", unit)
			self._record_db_success()
		except Exception as exc:
			self._record_db_error()
			logger.error("Failed to persist unit failure for %s: %s", unit.id, exc)
		if worker is not None:
			worker.status = "dead"
			worker.units_failed += 1
			if not self._degradation.is_db_degraded:
				try:
					await self.db.locked_call("update_worker", worker)
					self._record_db_success()
				except Exception:
					self._record_db_error()
		if put_on_queue:
			await self._completion_queue.put(
				WorkerCompletion(
					unit=unit, handoff=None, workspace=workspace, epoch=epoch,
				),
			)

	async def _execute_single_unit(
		self,
		unit: WorkUnit,
		epoch: Epoch,
		mission: Mission,
		semaphore: asyncio.Semaphore,
	) -> None:
		"""Execute a single work unit and put completion on queue."""
		if self._backend is None:
			raise RuntimeError("Controller not initialized: call start() first")

		source_repo = str(self.config.target.resolved_path)
		base_branch = self.config.green_branch.green_branch
		workspace = ""
		worker: Worker | None = None

		try:
			try:
				workspace = await self._backend.provision_workspace(
					unit.id, source_repo, base_branch,
				)
			except RuntimeError as e:
				logger.error("Failed to provision workspace: %s", e)
				await self._fail_unit(unit, None, epoch, str(e), "")
				return

			# Circuit breaker check: skip degraded workspaces
			if self.config.continuous.circuit_breaker_enabled and not self._circuit_breakers.can_dispatch(workspace):
				logger.warning(
					"Circuit breaker open for workspace %s, failing unit %s",
					workspace, unit.id[:12],
				)
				try:
					await self._backend.release_workspace(workspace)
				except Exception:
					pass
				await self._fail_unit(unit, None, epoch, "circuit breaker open", "")
				return

			branch_name = f"mc/unit-{unit.id}"
			unit.branch_name = branch_name
			unit.status = "running"
			unit.started_at = _now_iso()
			await self.db.locked_call("update_work_unit", unit)

			# Build prompt
			from mission_control.memory import load_context_for_mission_worker
			context = load_context_for_mission_worker(unit, self.config)
			experience_context = get_worker_context(self.db, unit)

			# Read MISSION_STATE.md from target repo
			mission_state = ""
			state_path = self.config.target.resolved_path / "MISSION_STATE.md"
			try:
				if state_path.exists():
					mission_state = state_path.read_text()
			except OSError as exc:
				logger.warning("Could not read MISSION_STATE.md: %s", exc)

			# Compute overlap warnings from other running units
			overlap_warnings = ""
			try:
				rows = self.db.conn.execute(
					"SELECT * FROM work_units WHERE status='running' AND id != ?",
					(unit.id,),
				).fetchall()
				running_units = [Database._row_to_work_unit(r) for r in rows]
				current_files = _parse_files_hint(unit.files_hint)
				if current_files:
					conflicts: list[str] = []
					for other in running_units:
						other_files = _parse_files_hint(other.files_hint)
						overlap = current_files & other_files
						if overlap:
							for f in sorted(overlap):
								conflicts.append(f"- {f} (also targeted by unit {other.id[:8]}: {other.title})")
					if conflicts:
						overlap_warnings = "\n".join(conflicts)
			except Exception as exc:
				logger.warning("Could not compute overlap warnings: %s", exc)

			if self._is_cleanup_mission and not unit.specialist:
				unit.specialist = "simplifier"

			specialist_template = load_specialist_template(unit.specialist, self.config)

			prompt = render_mission_worker_prompt(
				unit=unit,
				config=self.config,
				workspace_path=(
					workspace if "::" not in workspace
					else workspace.split("::")[0]
				),
				branch_name=branch_name,
				context=context,
				experience_context=experience_context,
				mission_state=mission_state,
				overlap_warnings=overlap_warnings,
				specialist_template=specialist_template,
			)

			budget = self.config.scheduler.budget.max_per_session_usd
			models_cfg = getattr(self.config, "models", None)
			model = getattr(models_cfg, "worker_model", None) or self.config.scheduler.model
			cmd = [
				"claude", "-p",
				"--output-format", "stream-json",
				"--permission-mode", "bypassPermissions",
				"--model", model,
				"--max-budget-usd", str(budget),
				prompt,
			]

			effective_timeout = unit.timeout or self.config.scheduler.session_timeout
			handle = await self._backend.spawn(
				unit.id, workspace, cmd,
				timeout=effective_timeout,
			)

			# Track worker subprocess lifecycle
			worker = Worker(
				id=unit.id,
				workspace_path=handle.workspace_path or workspace,
				status="working",
				current_unit_id=unit.id,
				pid=handle.pid,
				backend_type=self.config.backend.type,
			)
			try:
				await self.db.locked_call("insert_worker", worker)
			except Exception:
				logger.debug("Worker record insert failed for %s", unit.id)
			if self._event_stream:
				self._event_stream.emit(
					"worker_started",
					mission_id=mission.id,
					epoch_id=epoch.id,
					unit_id=unit.id,
					worker_id=unit.id,
					details={"pid": handle.pid, "workspace": workspace},
				)

			# Wait for completion
			poll_deadline = int(
				effective_timeout * self.config.continuous.timeout_multiplier,
			)
			monitor_interval = self.config.scheduler.monitor_interval
			start = time.monotonic()
			poll_iter = 0
			while time.monotonic() - start < poll_deadline:
				status = await self._backend.check_status(handle)
				if status != "running":
					break
				if not self.running:
					await self._backend.kill(handle)
					await self._fail_unit(unit, worker, epoch, "Stopped by signal", workspace)
					return
				output_so_far = await self._backend.get_output(handle)
				poll_iter += 1
				if poll_iter % 5 == 0:
					worker.last_heartbeat = _now_iso()
					excerpt = (output_so_far or "")[-500:]
					worker.backend_metadata = json.dumps({"output_excerpt": excerpt})
					if not self._degradation.is_db_degraded:
						try:
							await self.db.locked_call("update_worker", worker)
							self._record_db_success()
						except Exception:
							self._record_db_error()
				await asyncio.sleep(monitor_interval)
			else:
				await self._backend.kill(handle)
				await self._fail_unit(
					unit, worker, epoch,
					f"Timed out after {effective_timeout}s", workspace,
				)
				return

			output = await self._backend.get_output(handle)

			# Parse result -- try stream-json first, fall back to plain text
			handoff = None
			stream_result = parse_stream_json(output)
			mc_result = stream_result.mc_result
			if mc_result is None:
				mc_result = parse_mc_result(output)

			# Store token usage
			unit.input_tokens = stream_result.usage.input_tokens
			unit.output_tokens = stream_result.usage.output_tokens
			unit.cost_usd = compute_token_cost(
				stream_result.usage, self.config.pricing,
			)

			if mc_result:
				unit_status = str(mc_result.get("status", "completed"))
				unit.output_summary = str(mc_result.get("summary", ""))
				commits = mc_result.get("commits", [])
				if isinstance(commits, list) and commits:
					unit.commit_hash = str(commits[0])

				# Create handoff
				disc = mc_result.get("discoveries", [])
				conc = mc_result.get("concerns", [])
				fc = mc_result.get("files_changed", [])
				handoff = Handoff(
					work_unit_id=unit.id,
					round_id="",
					epoch_id=epoch.id,
					status=unit_status,
					commits=commits if isinstance(commits, list) else [],
					summary=unit.output_summary,
					discoveries=disc if isinstance(disc, list) else [],
					concerns=conc if isinstance(conc, list) else [],
					files_changed=fc if isinstance(fc, list) else [],
				)
				await self.db.locked_call("insert_handoff", handoff)
				unit.handoff_id = handoff.id
			else:
				unit_status = "completed" if status == "completed" else "failed"
				max_chars = self.config.scheduler.output_summary_max_chars
				unit.output_summary = (
					output[-max_chars:] if output else "No output"
				)

			# Set final status (completion processor handles merge)
			if unit_status in ("completed", "blocked"):
				unit.status = unit_status
			else:
				unit.attempt += 1
				unit.status = "failed"

			unit.finished_at = _now_iso()
			await self.db.locked_call("update_work_unit", unit)

			# Update worker record on completion
			if worker is not None:
				if unit.status == "failed":
					worker.units_failed += 1
				else:
					worker.units_completed += 1
				worker.status = "idle"
				worker.current_unit_id = None
				worker.pid = None
				worker.total_cost_usd += unit.cost_usd
				if not self._degradation.is_db_degraded:
					try:
						await self.db.locked_call("update_worker", worker)
						self._record_db_success()
					except Exception:
						self._record_db_error()

			# Put on completion queue for verify+merge
			await self._completion_queue.put(
				WorkerCompletion(
					unit=unit, handoff=handoff, workspace=workspace, epoch=epoch,
				),
			)

		except (RuntimeError, OSError) as e:
			logger.error("Infrastructure error executing unit %s: %s", unit.id, e)
			await self._fail_unit(unit, worker, epoch, f"Infrastructure error: {e}", workspace)
		except asyncio.CancelledError:
			logger.info("Unit %s execution cancelled", unit.id)
			await self._fail_unit(unit, worker, epoch, "Cancelled", workspace, put_on_queue=False)
		except (ValueError, KeyError, json.JSONDecodeError, sqlite3.IntegrityError) as e:
			logger.error("Data error executing unit %s: %s", unit.id, e)
			await self._fail_unit(unit, worker, epoch, f"Data error: {e}", workspace)
		finally:
			# Safety net: ensure worker marked idle if not already cleaned up
			if worker is not None and worker.status == "working":
				worker.status = "idle"
				worker.current_unit_id = None
				worker.pid = None
				if not self._degradation.is_db_degraded:
					try:
						await self.db.locked_call("update_worker", worker)
						self._record_db_success()
					except Exception:
						self._record_db_error()
			if self._event_stream and worker is not None:
				self._event_stream.emit(
					"worker_stopped",
					mission_id=mission.id,
					epoch_id=epoch.id,
					unit_id=unit.id,
					worker_id=unit.id,
					details={"status": worker.status},
					input_tokens=unit.input_tokens,
					output_tokens=unit.output_tokens,
					cost_usd=unit.cost_usd,
				)
			if workspace:
				await self._backend.release_workspace(workspace)
			self._in_flight_count = max(self._in_flight_count - 1, 0)
			semaphore.release()

	def _should_stop(self, mission: Mission) -> str:
		"""Check stopping conditions. Returns reason string or empty."""
		if not self.running:
			return "user_stopped"

		# Check DB signals
		signal_reason = self._check_signals(mission.id)
		if signal_reason:
			return signal_reason

		cont = self.config.continuous

		# All-fail stop
		if self._all_fail_stop_reason:
			return self._all_fail_stop_reason

		# Wall time limit
		if self._start_time > 0:
			elapsed = time.monotonic() - self._start_time
			if elapsed >= cont.max_wall_time_seconds:
				return "wall_time_exceeded"

		# EMA budget gate: stop if projected next unit cost exceeds remaining budget
		budget_limit = self.config.scheduler.budget.max_per_run_usd
		if budget_limit > 0 and self._ema.would_exceed_budget(mission.total_cost_usd, budget_limit):
			return "ema_budget_exceeded"

		# Degradation safe stop
		if self._degradation.should_stop:
			return "degradation_safe_stop"

		return ""

	def _check_signals(self, mission_id: str) -> str:
		"""Check for pending signals from web/MCP."""
		try:
			signals = self.db.get_pending_signals(mission_id)
		except Exception as exc:
			logger.error("Failed to check signals: %s", exc)
			return ""

		for signal in signals:
			if signal.signal_type == "stop":
				self.db.acknowledge_signal(signal.id)
				self.running = False
				return "signal_stopped"
			elif signal.signal_type == "pause":
				self._paused = True
				self.db.acknowledge_signal(signal.id)
				logger.info("Mission paused by signal")
			elif signal.signal_type == "resume":
				self._paused = False
				self.db.acknowledge_signal(signal.id)
				logger.info("Mission resumed by signal")
			elif signal.signal_type == "cancel_unit":
				self._handle_cancel_unit(signal)
			elif signal.signal_type == "force_retry":
				self._handle_force_retry(signal)
			elif signal.signal_type == "add_objective":
				self._handle_add_objective(signal)
			elif signal.signal_type == "adjust":
				self._handle_adjust_signal(signal)
		return ""

	def _handle_adjust_signal(self, signal: Signal) -> None:
		"""Adjust runtime parameters from signal payload."""
		try:
			params = json.loads(signal.payload) if signal.payload else {}
			if "num_workers" in params:
				new_count = int(params["num_workers"])
				self.config.scheduler.parallel.num_workers = new_count
				if self._semaphore is not None:
					available = max(new_count - self._in_flight_count, 0)
					self._semaphore = asyncio.Semaphore(available)
				logger.info(
					"Adjusted num_workers to %d",
					self.config.scheduler.parallel.num_workers,
				)
			if "max_wall_time" in params:
				self.config.continuous.max_wall_time_seconds = int(
					params["max_wall_time"],
				)
				logger.info(
					"Adjusted max_wall_time to %d",
					self.config.continuous.max_wall_time_seconds,
				)
			self.db.acknowledge_signal(signal.id)
		except Exception as exc:
			logger.error("Failed to handle adjust signal: %s", exc)

	def _handle_cancel_unit(self, signal: Signal) -> None:
		"""Cancel a running unit by its ID."""
		try:
			payload = json.loads(signal.payload) if signal.payload else {}
			unit_id = payload.get("unit_id", "")
			task = self._unit_tasks.get(unit_id)
			if task and not task.done():
				task.cancel()
				logger.info("Cancelled unit %s by signal", unit_id[:12])
			else:
				logger.warning("Cancel signal for unit %s: not found or already done", unit_id[:12])
			self.db.acknowledge_signal(signal.id)
		except Exception as exc:
			logger.error("Failed to handle cancel_unit signal: %s", exc)

	def _handle_force_retry(self, signal: Signal) -> None:
		"""Force retry a unit immediately (no delay, resets attempt counter)."""
		try:
			payload = json.loads(signal.payload) if signal.payload else {}
			unit_id = payload.get("unit_id", "")
			unit = self.db.get_work_unit(unit_id)
			if unit is None:
				logger.warning("Force retry: unit %s not found", unit_id[:12])
				self.db.acknowledge_signal(signal.id)
				return

			# Cancel existing task if running
			task = self._unit_tasks.get(unit_id)
			if task and not task.done():
				task.cancel()

			# Reset unit for immediate re-dispatch
			unit.status = "pending"
			unit.commit_hash = None
			unit.branch_name = ""
			unit.output_summary = ""
			unit.description += "\n\n[Force retry] Manually triggered re-dispatch."
			self.db.update_work_unit(unit)
			logger.info("Force retry queued for unit %s", unit_id[:12])
			self.db.acknowledge_signal(signal.id)
		except Exception as exc:
			logger.error("Failed to handle force_retry signal: %s", exc)

	def _handle_add_objective(self, signal: Signal) -> None:
		"""Append a new objective to the mission for the planner to incorporate."""
		try:
			payload = json.loads(signal.payload) if signal.payload else {}
			new_objective = payload.get("objective", "")
			if new_objective and self._planner:
				# Append to mission objective so the planner sees it
				self.config.target.objective += f"\n\nAdditional objective: {new_objective}"
				logger.info("Added objective: %s", new_objective[:60])
			self.db.acknowledge_signal(signal.id)
		except Exception as exc:
			logger.error("Failed to handle add_objective signal: %s", exc)

	async def _run_final_verification(self) -> tuple[bool, str]:
		"""Run full verification on mc/green at mission end.

		Returns (passed, output) tuple.
		"""
		if self._green_branch is None:
			raise RuntimeError("Controller not initialized: call start() first")
		verify_cmd = self.config.target.verification.command
		gb = self.config.green_branch

		# Checkout mc/green in the green branch workspace
		await self._green_branch._run_git("checkout", gb.green_branch)

		# Run verification
		ok, output = await self._green_branch._run_command(verify_cmd)
		return ok, output

	# Backward-compatible wrappers delegating to extracted modules
	@property
	def _backlog_item_ids(self) -> list[str]:
		try:
			return self._backlog_manager.backlog_item_ids
		except AttributeError:
			return []

	@_backlog_item_ids.setter
	def _backlog_item_ids(self, value: list[str]) -> None:
		try:
			self._backlog_manager.backlog_item_ids = value
		except AttributeError:
			# BacklogManager not yet initialized (e.g. test setup bypassing __init__)
			self._backlog_manager = BacklogManager(self.db, self.config)
			self._backlog_manager.backlog_item_ids = value

	def _load_backlog_objective(self, limit: int = 5) -> str | None:
		return self._backlog_manager.load_backlog_objective(limit=limit)

	def _update_backlog_on_completion(
		self, objective_met: bool, handoffs: list[Handoff],
	) -> None:
		self._backlog_manager.update_backlog_on_completion(objective_met, handoffs)

	def _update_backlog_from_completion(
		self,
		unit: WorkUnit,
		merged: bool,
		handoff: Handoff | None,
		mission_id: str,
	) -> None:
		self._backlog_manager.update_backlog_from_completion(unit, merged, handoff, mission_id)

	def _build_planner_context(self, mission_id: str) -> str:
		return build_planner_context(self.db, mission_id)

	def _update_mission_state(self, mission: Mission) -> None:
		update_mission_state(
			self.db, mission, self.config, self._state_changelog,
			degradation_status=self._degradation.get_status_dict(),
		)

	def stop(self) -> None:
		self.running = False
