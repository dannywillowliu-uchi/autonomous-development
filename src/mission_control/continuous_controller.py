"""Continuous controller -- event-driven development loop without round boundaries."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mission_control.backends import ContainerBackend, LocalBackend, SSHBackend, WorkerBackend
from mission_control.batch_analyzer import BatchAnalyzer
from mission_control.causal import CausalAttributor, CausalSignal
from mission_control.circuit_breaker import CircuitBreakerManager
from mission_control.config import ContinuousConfig, MissionConfig, build_claude_cmd, claude_subprocess_env
from mission_control.constants import (
	UNIT_EVENT_AUDIT_COMPLETED,
	UNIT_EVENT_DEGRADATION_TRANSITION,
	UNIT_EVENT_DESIGN_COMPLETED,
	UNIT_EVENT_DISPATCHED,
	UNIT_EVENT_MERGE_FAILED,
	UNIT_EVENT_MERGED,
	UNIT_EVENT_REJECTED,
	UNIT_EVENT_RESEARCH_COMPLETED,
	UNIT_EVENT_RETRY_QUEUED,
	UNIT_EVENT_SPECULATION_COMPLETED,
)
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.degradation import DegradationManager, DegradationTransition
from mission_control.deliberative_planner import DeliberativePlanner
from mission_control.diff_reviewer import DiffReviewer
from mission_control.ema import ExponentialMovingAverage
from mission_control.event_stream import EventStream
from mission_control.feedback import get_worker_context
from mission_control.file_lock_registry import FileLockRegistry
from mission_control.green_branch import GreenBranchManager, UnitMergeResult
from mission_control.heartbeat import Heartbeat
from mission_control.json_utils import extract_json_from_text
from mission_control.models import (
	Epoch,
	ExperimentResult,
	Handoff,
	Mission,
	Signal,
	SpeculationResult,
	UnitEvent,
	UnitReview,
	Worker,
	WorkUnit,
	_new_id,
	_now_iso,
)
from mission_control.notifier import TelegramNotifier
from mission_control.overlap import _parse_files_hint, topological_layers
from mission_control.planner_context import build_planner_context, update_mission_state
from mission_control.session import parse_mc_result
from mission_control.token_parser import compute_token_cost, parse_stream_json
from mission_control.trace_log import TraceEvent, TraceLogger
from mission_control.tracing import MissionTracer
from mission_control.worker import CONFLICT_RESOLUTION_PROMPT, load_specialist_template, render_mission_worker_prompt

logger = logging.getLogger(__name__)


class DynamicSemaphore:
	"""Asyncio semaphore with dynamically adjustable capacity.

	Unlike creating a new Semaphore (which orphans waiters on the old one),
	adjust() manipulates the existing underlying semaphore so that all
	in-progress acquire() calls remain valid.
	"""

	def __init__(self, value: int = 1) -> None:
		self._sem = asyncio.Semaphore(value)
		self._capacity = value
		self._debt = 0

	async def acquire(self) -> None:
		await self._sem.acquire()

	def release(self) -> None:
		if self._debt > 0:
			self._debt -= 1
		else:
			self._sem.release()

	def locked(self) -> bool:
		return self._sem.locked()

	@property
	def capacity(self) -> int:
		return self._capacity

	@property
	def _value(self) -> int:
		"""Expose internal counter for test introspection."""
		return self._sem._value

	def adjust(self, new_capacity: int) -> None:
		"""Adjust effective capacity without replacing the semaphore object."""
		delta = new_capacity - self._capacity
		self._capacity = new_capacity
		if delta > 0:
			absorb = min(delta, self._debt)
			self._debt -= absorb
			delta -= absorb
			for _ in range(delta):
				self._sem.release()
		elif delta < 0:
			self._debt += abs(delta)


@dataclass
class WorkerCompletion:
	"""A completed unit ready for verification and merge."""

	unit: WorkUnit
	handoff: Handoff | None
	workspace: str
	epoch: Epoch
	prompt_variant_id: str = ""


@dataclass
class _BatchMergeEntry:
	"""A unit queued for batch merge."""

	completion: WorkerCompletion
	mission: Mission
	result: ContinuousMissionResult


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
	evaluator_passed: bool | None = None
	evaluator_evidence: list[str] = field(default_factory=list)
	evaluator_gaps: list[str] = field(default_factory=list)
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

	def __init__(self, config: MissionConfig, db: Database, chain_id: str = "") -> None:
		self.config = config
		self.db = db
		self.chain_id = chain_id
		self.running = True
		self._backend: WorkerBackend | None = None
		self._green_branch: GreenBranchManager | None = None
		self._planner: ContinuousPlanner | None = None
		self._notifier: TelegramNotifier | None = None
		self._heartbeat: Heartbeat | None = None
		self._active_tasks: set[asyncio.Task[None]] = set()
		self._unit_tasks: dict[str, asyncio.Task[None]] = {}  # unit_id -> task
		self._semaphore: DynamicSemaphore | None = None
		self._paused: bool = False
		self._start_time: float = 0.0
		self._state_changelog: list[str] = []
		self._total_dispatched: int = 0
		self._total_merged: int = 0
		self._total_failed: int = 0
		self._event_stream: EventStream | None = None
		self._in_flight_count: int = 0
		self._merged_files: set[str] = set()  # files covered by merged units
		self._completed_unit_ids: set[str] = set()  # units that completed/merged successfully
		self._file_locks: FileLockRegistry = FileLockRegistry()
		self._last_reconcile_count: int = 0
		self._objective_check_count: int = 0
		self._failed_unit_replan_count: int = 0
		self._is_cleanup_mission: bool = config.target.objective.startswith("[CLEANUP]")
		self._diff_reviewer: DiffReviewer = DiffReviewer(config)
		cont = config.continuous
		self._circuit_breakers: CircuitBreakerManager = CircuitBreakerManager(
			max_failures=cont.circuit_breaker_max_failures,
			cooldown_seconds=float(cont.circuit_breaker_cooldown_seconds),
		)
		self._degradation: DegradationManager = DegradationManager(
			config=config.degradation,
			on_transition=self._on_degradation_transition,
		)
		self._causal_attributor: CausalAttributor = CausalAttributor(db)
		self._tracer: MissionTracer = MissionTracer(config.tracing)
		self._trace_logger: TraceLogger = TraceLogger(config.trace_log)
		self._a2a_server: Any = None
		try:
			if config.a2a.enabled is True:
				from mission_control.a2a import A2AServer
				self._a2a_server = A2AServer(config.a2a, db)
		except Exception:
			logger.warning("Failed to initialize A2A server", exc_info=True)
		self._mcp_registry: Any = None
		try:
			if config.mcp_registry.enabled is True:
				from mission_control.mcp_registry import MCPToolRegistry
				self._mcp_registry = MCPToolRegistry(db, config.mcp_registry)
		except Exception:
			logger.warning("Failed to initialize MCP registry", exc_info=True)
		self._prompt_evolution: Any = None
		try:
			if config.prompt_evolution.enabled is True:
				from mission_control.prompt_evolution import PromptEvolutionEngine
				self._prompt_evolution = PromptEvolutionEngine(db, config.prompt_evolution)
		except Exception:
			logger.warning("Failed to initialize prompt evolution", exc_info=True)
		self._memory_manager: Any = None
		try:
			if config.episodic_memory.enabled is True:
				from mission_control.memory import MemoryManager
				self._memory_manager = MemoryManager(db, config.episodic_memory)
		except Exception:
			logger.warning("Failed to initialize memory manager", exc_info=True)
		self._active_fixups: dict[str, asyncio.Task[None]] = {}
		self._batch_analyzer: BatchAnalyzer = BatchAnalyzer(db)
		self._current_strategy: str = ""
		self._speculation_completions: dict[str, list[WorkerCompletion]] = {}
		self._speculation_parent_units: dict[str, WorkUnit] = {}
		self._merge_queue: list[_BatchMergeEntry] = []
		self._merge_queue_timer: asyncio.TimerHandle | None = None
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

	def _trace(self, worker_id: str, unit_id: str, event_type: str, **details: Any) -> None:
		"""Emit a structured trace event to the trace logger."""
		self._trace_logger.write(TraceEvent(
			timestamp=datetime.now(timezone.utc).isoformat(),
			worker_id=worker_id,
			unit_id=unit_id,
			event_type=event_type,
			details=details,
		))

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
			loop = asyncio.get_running_loop()
			loop.create_task(self.db.locked_call("update_degradation_level", transition.to_level.name))
		except RuntimeError:
			self.db.update_degradation_level(transition.to_level.name)

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
			chain_id=self.chain_id,
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
		orphaned_count = await self.db.locked_call("reset_orphaned_units")
		if orphaned_count:
			logger.info(
				"Cleaned up %d orphaned work units from prior missions",
				orphaned_count,
			)

		# Generate initial MISSION_STATE.md with objective and empty sections
		try:
			update_mission_state(self.db, mission, self.config, self._state_changelog)
		except Exception as exc:
			logger.warning("Failed to write initial MISSION_STATE.md: %s", exc)

		try:
			await self._init_components()

			if self._a2a_server is not None:
				try:
					self._a2a_server.start()
				except Exception as exc:
					logger.warning("Failed to start A2A server: %s", exc)
					self._a2a_server = None
			self._mission_span_ctx = self._tracer.start_mission_span(mission.id)
			self._mission_span = self._mission_span_ctx.__enter__()

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

			await self._orchestration_loop(mission, result)

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

			# Evaluator agent: spawn Claude with shell access to actually test the software
			if (
				self.config.evaluator.enabled
				and self._green_branch
				and self._total_merged > 0
				and result.final_verification_passed
			):
				try:
					eval_result = await self._run_evaluator_agent(mission, self._green_branch.workspace)
					result.evaluator_passed = eval_result.get("passed", False)
					result.evaluator_evidence = eval_result.get("evidence", [])
					result.evaluator_gaps = eval_result.get("gaps", [])
					if not result.evaluator_passed:
						result.objective_met = False
						logger.warning(
							"Evaluator agent failed: gaps=%s", result.evaluator_gaps,
						)
				except Exception as exc:
					logger.error("Evaluator agent error: %s", exc, exc_info=True)

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

			mission.status = "completed" if result.objective_met else "stopped"
			mission.finished_at = _now_iso()
			mission.stopped_reason = result.stopped_reason

			try:
				self.db.update_mission(mission)
			except Exception as exc:
				logger.error(
					"Failed to update mission in finally: %s", exc, exc_info=True,
				)

			if self._a2a_server is not None:
				try:
					self._a2a_server.stop()
				except Exception as exc:
					logger.debug("Failed to stop A2A server: %s", exc)

			if self._backend:
				await self._backend.cleanup()

			result.wall_time_seconds = time.monotonic() - self._start_time
			result.total_units_dispatched = self._total_dispatched
			result.total_units_merged = self._total_merged
			result.total_units_failed = self._total_failed
			result.units_failed_unrecovered = self._total_failed
			result.db_errors = self._degradation.get_status_dict()["db_errors"]
			result.degradation_level = self._degradation.level_name

			try:
				from mission_control.mission_report import generate_mission_report
				generate_mission_report(result, mission, self.db, self.config)
			except Exception as exc:
				logger.error("Failed to generate mission report: %s", exc, exc_info=True)

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

			# Prompt evolution: propose mutations for components with high failure rates
			if self._prompt_evolution is not None:
				try:
					total_units = result.total_units_merged + result.total_units_failed
					if total_units > 0:
						fail_rate = result.total_units_failed / total_units
						if fail_rate > 0.2:
							failure_summaries = [
								s for s in (failed_summaries if "failed_summaries" in dir() else [])
							][:5]
							if failure_summaries:
								await self._prompt_evolution.propose_mutation(
									"worker", failure_summaries,
								)
				except Exception as exc:
					logger.debug("Prompt mutation failed: %s", exc)

			# Episodic memory: decay tick and distill high-confidence candidates
			if self._memory_manager is not None:
				try:
					evicted, extended = self._memory_manager.decay_tick()
					if evicted or extended:
						logger.info("Memory decay: evicted=%d, extended=%d", evicted, extended)
					candidates = self._memory_manager.get_promote_candidates()
					if candidates:
						await self._memory_manager.distill_to_semantic(candidates)
				except Exception as exc:
					logger.debug("Memory decay/distill failed: %s", exc)

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

			# Close the mission span
			if hasattr(self, "_mission_span_ctx") and self._mission_span_ctx:
				try:
					self._mission_span_ctx.__exit__(None, None, None)
				except Exception:
					pass

			self._trace_logger.close()

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
				config=self.config,
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
				config=self.config,
			)
			await backend.initialize(
				warm_count=self.config.scheduler.parallel.warm_clones,
			)
			self._backend = backend

		# Green branch manager
		self._green_branch = GreenBranchManager(self.config, self.db, trace_logger=self._trace_logger)
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

		# Deliberative planner (critic/planner dual-agent loop)
		if self.config.deliberation.enabled:
			self._planner = DeliberativePlanner(self.config, self.db)
		else:
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

	async def _process_single_completion(
		self,
		completion: WorkerCompletion,
		mission: Mission,
		result: ContinuousMissionResult,
	) -> None:
		"""Process a single completed unit: verify, merge, record feedback."""
		if self._green_branch is None:
			raise RuntimeError("Controller not initialized: call start() first")

		cont = self.config.continuous
		unit = completion.unit
		handoff = completion.handoff
		epoch = completion.epoch
		workspace = completion.workspace

		# Research units: skip merge, just record findings
		if unit.unit_type == "research":
			if unit.status != "completed":
				self._total_failed += 1
				logger.info("Research unit %s failed", unit.id)
				mission.total_cost_usd += unit.cost_usd
				if unit.cost_usd > 0:
					self._ema.update(unit.cost_usd)
				try:
					self.db.update_mission(mission)
				except Exception as exc:
					logger.error("Failed to update mission cost for unit %s: %s", unit.id, exc)
				try:
					signal = self._build_causal_signal(unit, epoch, mission, merged=False)
					self._causal_attributor.record(signal)
				except Exception:
					pass
				return
			logger.info("Research unit %s completed -- skipping merge", unit.id)
			self._total_merged += 1
			self._completed_unit_ids.add(unit.id)
			self._log_unit_event(
				mission_id=mission.id,
				epoch_id=epoch.id,
				work_unit_id=unit.id,
				event_type=UNIT_EVENT_RESEARCH_COMPLETED,
			)
			self._extract_knowledge(unit, handoff, mission)

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
			try:
				signal = self._build_causal_signal(unit, epoch, mission, merged=True)
				self._causal_attributor.record(signal)
			except Exception:
				pass
			return

		# Experiment units: skip merge, store comparison report
		if unit.unit_type == "experiment":
			if unit.status != "completed":
				self._total_failed += 1
				logger.info("Experiment unit %s failed", unit.id)
				mission.total_cost_usd += unit.cost_usd
				if unit.cost_usd > 0:
					self._ema.update(unit.cost_usd)
				try:
					self.db.update_mission(mission)
				except Exception as exc:
					logger.error("Failed to update mission cost for unit %s: %s", unit.id, exc)
				try:
					signal = self._build_causal_signal(unit, epoch, mission, merged=False)
					self._causal_attributor.record(signal)
				except Exception:
					pass
				return
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
				event_type="experiment_completed",
			)

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
			try:
				signal = self._build_causal_signal(unit, epoch, mission, merged=True)
				self._causal_attributor.record(signal)
			except Exception:
				pass
			return

		# Audit units: skip merge, extract knowledge
		if unit.unit_type == "audit":
			if unit.status != "completed":
				self._total_failed += 1
				logger.info("Audit unit %s failed", unit.id)
				mission.total_cost_usd += unit.cost_usd
				if unit.cost_usd > 0:
					self._ema.update(unit.cost_usd)
				try:
					self.db.update_mission(mission)
				except Exception as exc:
					logger.error("Failed to update mission cost for unit %s: %s", unit.id, exc)
				try:
					signal = self._build_causal_signal(unit, epoch, mission, merged=False)
					self._causal_attributor.record(signal)
				except Exception:
					pass
				return
			logger.info("Audit unit %s completed -- skipping merge", unit.id)
			self._total_merged += 1
			self._completed_unit_ids.add(unit.id)
			self._log_unit_event(
				mission_id=mission.id,
				epoch_id=epoch.id,
				work_unit_id=unit.id,
				event_type=UNIT_EVENT_AUDIT_COMPLETED,
			)
			self._extract_knowledge(unit, handoff, mission)
			timestamp = unit.finished_at or _now_iso()
			summary = (handoff.summary[:80] if handoff and handoff.summary else unit.output_summary[:80])
			self._state_changelog.append(
				f"- {timestamp} | {unit.id[:8]} audit completed -- {summary}"
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
			try:
				signal = self._build_causal_signal(unit, epoch, mission, merged=True)
				self._causal_attributor.record(signal)
			except Exception:
				pass
			return

		# Design units: skip merge, store design decisions
		if unit.unit_type == "design":
			if unit.status != "completed":
				self._total_failed += 1
				logger.info("Design unit %s failed", unit.id)
				mission.total_cost_usd += unit.cost_usd
				if unit.cost_usd > 0:
					self._ema.update(unit.cost_usd)
				try:
					self.db.update_mission(mission)
				except Exception as exc:
					logger.error("Failed to update mission cost for unit %s: %s", unit.id, exc)
				try:
					signal = self._build_causal_signal(unit, epoch, mission, merged=False)
					self._causal_attributor.record(signal)
				except Exception:
					pass
				return
			logger.info("Design unit %s completed -- skipping merge", unit.id)
			self._total_merged += 1
			self._completed_unit_ids.add(unit.id)
			self._log_unit_event(
				mission_id=mission.id,
				epoch_id=epoch.id,
				work_unit_id=unit.id,
				event_type=UNIT_EVENT_DESIGN_COMPLETED,
			)
			self._extract_knowledge(unit, handoff, mission)
			timestamp = unit.finished_at or _now_iso()
			summary = (handoff.summary[:80] if handoff and handoff.summary else unit.output_summary[:80])
			self._state_changelog.append(
				f"- {timestamp} | {unit.id[:8]} design completed -- {summary}"
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
			try:
				signal = self._build_causal_signal(unit, epoch, mission, merged=True)
				self._causal_attributor.record(signal)
			except Exception:
				pass
			return

		# Speculation branch: collect and run selection gate when all done
		if unit.unit_type == "speculation_branch" and unit.speculation_parent_id:
			parent_id = unit.speculation_parent_id
			if parent_id in self._speculation_completions:
				self._speculation_completions[parent_id].append(completion)
				if len(self._speculation_completions[parent_id]) >= self.config.speculation.branch_count:
					await self._speculation_select_winner(parent_id, mission, epoch)
			else:
				logger.warning("Orphan speculation branch %s", unit.id)
				self._total_failed += 1
			mission.total_cost_usd += unit.cost_usd
			if unit.cost_usd > 0:
				self._ema.update(unit.cost_usd)
			try:
				self.db.update_mission(mission)
			except Exception:
				pass
			return

		# Merge if the unit completed with commits
		merged = False
		if unit.status == "completed" and unit.commit_hash:
			try:
				self._trace(unit.id, unit.id, "merge_attempted")

				# Queue for batch merge if batch_merge_min_size >= 2
				gb_config = self.config.green_branch
				if gb_config.batch_merge_min_size >= 2:
					entry = _BatchMergeEntry(
						completion=completion, mission=mission, result=result,
					)
					self._merge_queue.append(entry)
					if len(self._merge_queue) >= gb_config.batch_merge_min_size:
						await self._flush_merge_batch()
					else:
						self._schedule_batch_flush()
					# Bookkeeping runs inside _flush_merge_batch
					return
				else:
					merge_result = await self._green_branch.merge_unit(
						workspace, unit.branch_name,
						acceptance_criteria=unit.acceptance_criteria,
					)

				self._degradation.record_merge_attempt(conflict=not merge_result.merged)

				# Post-merge bookkeeping (review, counters, sync)
				if merge_result.merged:
					self._trace(unit.id, unit.id, "merge_succeeded")
					self._accept_merge(
						unit, merge_result, workspace,
						mission, epoch, result,
					)
					merged = True

				if not merged:
					# Fixup path: fire background task for fixable failures
					fixup_stages = ("pre_merge_verification", "acceptance_criteria", "merge_conflict")
					if (
						merge_result.failure_stage in fixup_stages
						and workspace
						and unit.id not in self._active_fixups
					):
						logger.info(
							"Launching background fixup for unit %s (stage=%s)",
							unit.id[:12], merge_result.failure_stage,
						)
						task = asyncio.create_task(
							self._background_fixup(
								unit, workspace, merge_result, cont,
								mission, epoch, result, handoff,
							),
						)
						self._active_fixups[unit.id] = task
						task.add_done_callback(self._fixup_done_callback)
						# Return early -- background task handles all post-fixup bookkeeping
						return
					else:
						logger.warning(
							"Unit %s failed merge: %s",
							unit.id, merge_result.failure_output[-200:],
						)
						self._trace(
							unit.id, unit.id, "merge_failed",
							failure_stage=merge_result.failure_stage,
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
							self._log_unit_event(
								mission_id=mission.id,
								epoch_id=epoch.id,
								work_unit_id=unit.id,
								event_type=UNIT_EVENT_REJECTED,
								details=merge_result.failure_output[-2000:],
								stream_details={"failure_output": merge_result.failure_output[-2000:]},
							)
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

		# Promote synthesized tools to MCP registry after merge
		if merged and self._mcp_registry and handoff:
			try:
				tools_created = handoff.discoveries  # tools reported via handoff
				if isinstance(tools_created, list):
					from mission_control.tool_synthesis import ToolEntry, promote_to_mcp_registry
					review = self.db.get_unit_review_for_unit(unit.id)
					score = review.avg_score / 10.0 if review else 0.5
					for item in tools_created:
						if isinstance(item, dict) and "name" in item:
							tool = ToolEntry(
								name=item["name"],
								description=item.get("description", ""),
								script_path=Path(item.get("script_path", "")),
							)
							promote_to_mcp_registry(tool, score, mission.id, self._mcp_registry)
			except Exception as exc:
				logger.debug("Failed to promote tools to MCP registry: %s", exc)

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


		# Prompt evolution: record outcome
		if self._prompt_evolution is not None and completion.prompt_variant_id:
			try:
				self._prompt_evolution.record_outcome(
					completion.prompt_variant_id,
					"pass" if merged else "fail",
				)
			except Exception as exc:
				logger.debug("Failed to record prompt outcome: %s", exc)

		# Episodic memory: store episode for this completion
		if self._memory_manager is not None:
			try:
				event_type = "merge_success" if merged else "test_failure"
				scope_tokens = [
					t.strip() for t in (unit.files_hint or "").split(",") if t.strip()
				]
				summary = (
					handoff.summary[:200] if handoff and handoff.summary else unit.output_summary[:200]
				)
				self._memory_manager.store_episode(
					event_type=event_type,
					content=summary,
					outcome="pass" if merged else "fail",
					scope_tokens=scope_tokens,
				)
			except Exception as exc:
				logger.debug("Failed to store episodic memory: %s", exc)

		# Circuit breaker: record success/failure per workspace
		if self.config.continuous.circuit_breaker_enabled and workspace:
			if merged:
				self._circuit_breakers.record_success(workspace)
			else:
				self._circuit_breakers.record_failure(workspace)

		# Causal attribution: record signal for this unit outcome
		try:
			signal = self._build_causal_signal(unit, epoch, mission, merged)
			self._causal_attributor.record(signal)
		except Exception as exc:
			logger.debug("Failed to record causal signal: %s", exc)

		# Reconciler sweep: verify combined green state after new merges
		# Runs when verify_before_merge is off (each merge not pre-verified),
		# or periodically every reconcile_interval merges as an integrity check.
		reconcile_interval = cont.reconcile_interval
		periodic_reconcile = (
			reconcile_interval > 0
			and self._total_merged > 0
			and self._total_merged % reconcile_interval == 0
		)
		should_reconcile = (
			merged
			and self._total_merged > self._last_reconcile_count
			and (not cont.verify_before_merge or periodic_reconcile)
		)
		if should_reconcile:
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

	async def _execute_batch(
		self,
		units: list[WorkUnit],
		epoch: Epoch,
		mission: Mission,
	) -> list[WorkerCompletion]:
		"""Execute a batch of units with topological layering, return completions."""
		if not units:
			return []

		layers = topological_layers(units)
		all_completions: list[WorkerCompletion] = []

		for layer_idx, layer in enumerate(layers):
			logger.info(
				"Batch layer %d/%d (%d units)",
				layer_idx, len(layers) - 1, len(layer),
			)

			layer_tasks: list[asyncio.Task[WorkerCompletion | None]] = []

			for unit in layer:
				unit.epoch_id = epoch.id
				try:
					self.db.insert_work_unit(unit)
				except Exception as exc:
					logger.error("Failed to insert work unit: %s", exc, exc_info=True)
					continue

				# File-scope isolation: skip dispatch if write_scope conflicts
				if unit.write_scope:
					conflicts = self._file_locks.claim(unit.id, unit.write_scope)
					if conflicts:
						logger.warning(
							"Unit %s write_scope conflicts on %s -- holding as pending",
							unit.id[:12], conflicts,
						)
						continue

				self._log_unit_event(
					mission_id=mission.id,
					epoch_id=epoch.id,
					work_unit_id=unit.id,
					event_type=UNIT_EVENT_DISPATCHED,
					stream_details={"title": unit.title, "files": unit.files_hint, "layer": layer_idx},
				)
				self._total_dispatched += 1

				await self._semaphore.acquire()
				self._in_flight_count += 1

				task = asyncio.create_task(
					self._execute_single_unit(unit, epoch, mission),
				)
				layer_tasks.append(task)

			if layer_tasks:
				cont = self.config.continuous
				layer_timeout = (
					cont.layer_drain_timeout_base
					+ cont.layer_drain_timeout_per_unit * len(layer_tasks)
				)
				try:
					results = await asyncio.wait_for(
						asyncio.gather(*layer_tasks, return_exceptions=True),
						timeout=layer_timeout,
					)
				except asyncio.TimeoutError:
					completed = 0
					for t in layer_tasks:
						if t.done():
							r = t.result()
							if isinstance(r, WorkerCompletion):
								all_completions.append(r)
							completed += 1
						else:
							t.cancel()
					logger.warning(
						"Layer %d: completion drain timeout after %ds (%d/%d processed)",
						layer_idx, layer_timeout, completed, len(layer_tasks),
					)
					continue
				for r in results:
					if isinstance(r, WorkerCompletion):
						all_completions.append(r)
					elif isinstance(r, Exception):
						logger.error("Unit execution error in batch: %s", r)

		return all_completions

	async def _process_batch(
		self,
		completions: list[WorkerCompletion],
		mission: Mission,
		result: ContinuousMissionResult,
	) -> None:
		"""Process a batch of completions sequentially."""
		for completion in completions:
			await self._process_single_completion(completion, mission, result)

	async def _orchestration_loop(
		self,
		mission: Mission,
		result: ContinuousMissionResult,
	) -> None:
		"""Sequential orchestration: plan -> execute -> process, repeat."""
		if self._planner is None:
			raise RuntimeError("Controller not initialized")
		if self._backend is None:
			raise RuntimeError("Controller not initialized")

		num_workers = self.config.scheduler.parallel.num_workers
		self._semaphore = DynamicSemaphore(num_workers)

		while self.running:
			reason = self._should_stop(mission)
			if not reason and self._heartbeat:
				reason = await self._heartbeat.check(
					self._total_merged, self._total_failed,
				)
			if reason:
				result.stopped_reason = reason
				self.running = False
				break

			state = build_planner_context(self.db, mission.id)

			knowledge_items = self.db.get_knowledge_for_mission(mission.id)
			knowledge_context = ""
			if knowledge_items:
				knowledge_context = "\n".join(
					f"- [{k.source_unit_type}] {k.title}: {k.content[:200]}"
					for k in knowledge_items[-20:]
				)

			# Gather batch signals for the deliberative planner's critic
			batch_signals = None
			if self._total_dispatched > 0:
				try:
					batch_signals = self._batch_analyzer.analyze(mission.id)
				except Exception as exc:
					logger.debug("Batch analysis failed: %s", exc)

			# Build locked_files: already-merged + in-flight units
			locked_files: dict[str, list[str]] = {}
			for f in self._merged_files:
				locked_files[f] = ["already merged"]
			try:
				running_units = self.db.get_running_units()
				for ru in running_units:
					for f in _parse_files_hint(ru.files_hint):
						locked_files.setdefault(f, []).append(f"in-flight: {ru.title[:40]}")
			except Exception as exc:
				logger.debug("Failed to gather in-flight locked files: %s", exc)

			try:
				plan, units, epoch = await self._planner.get_next_units(
					mission,
					max_units=num_workers,
					feedback_context=state,
					knowledge_context=knowledge_context,
					locked_files=locked_files or None,
					batch_signals=batch_signals,
				)
			except Exception as exc:
				logger.error("Planner failed: %s", exc, exc_info=True)
				await asyncio.sleep(5)
				continue

			if not units:
				logger.info("Planner returned no units -- objective complete")
				result.objective_met = True
				result.stopped_reason = "planner_completed"
				self.running = False
				break

			try:
				self.db.insert_plan(plan)
			except Exception as exc:
				logger.error("Failed to insert plan: %s", exc, exc_info=True)
				continue
			try:
				self.db.insert_epoch(epoch)
			except Exception as exc:
				logger.error("Failed to insert epoch: %s", exc, exc_info=True)

			completions = await self._execute_batch(units, epoch, mission)

			await self._process_batch(completions, mission, result)

			# Flush any remaining batched pushes after processing completions
			if self._green_branch:
				try:
					await self._green_branch.maybe_push(force=True)
				except Exception as exc:
					logger.warning("Batch push flush failed: %s", exc)

			try:
				update_mission_state(
					self.db, mission, self.config, self._state_changelog,
					degradation_status=self._degradation.get_status_dict(),
					strategy=self._current_strategy,
				)
			except Exception as exc:
				logger.warning("Failed to update MISSION_STATE.md: %s", exc)

	# -- Speculation branching --

	SPECULATION_APPROACH_HINTS: tuple[str, ...] = (
		"straightforward: use the most obvious, conventional approach",
		"alternative: try a less common but potentially cleaner approach",
		"defensive: prioritize error handling, edge cases, and robustness",
		"minimal: implement the smallest possible change to satisfy requirements",
		"exploratory: refactor surrounding code as needed for the best long-term design",
	)

	async def _dispatch_speculated_unit(
		self,
		unit: WorkUnit,
		epoch: Epoch,
		mission: Mission,
		unit_layer_map: dict[str, int],
	) -> bool:
		"""Fork a high-uncertainty unit into N parallel branches.

		Returns True if speculation was dispatched, False if fallback to
		single dispatch is needed (e.g. cost cap exceeded).
		"""
		spec_cfg = self.config.speculation
		branch_count = spec_cfg.branch_count

		# Cost guard: check projected cost against remaining budget
		budget_limit = self.config.scheduler.budget.max_per_run_usd
		ema_val = self._ema.value
		if budget_limit > 0 and ema_val is not None and ema_val > 0:
			projected_cost = branch_count * ema_val * spec_cfg.cost_limit_multiplier
			remaining = budget_limit - mission.total_cost_usd
			if projected_cost > remaining:
				logger.info(
					"Speculation cost cap: projected $%.2f > remaining $%.2f, falling back to single dispatch",
					projected_cost, remaining,
				)
				return False

		# Mark parent unit
		unit.unit_type = "speculation_parent"
		unit.status = "running"
		unit.started_at = _now_iso()
		try:
			self.db.insert_work_unit(unit)
		except Exception as exc:
			logger.error("Failed to insert speculation parent unit: %s", exc)
			return False

		self._log_unit_event(
			mission_id=mission.id,
			epoch_id=epoch.id,
			work_unit_id=unit.id,
			event_type=UNIT_EVENT_DISPATCHED,
			stream_details={
				"title": unit.title,
				"files": unit.files_hint,
				"layer": unit_layer_map.get(unit.id, 0),
				"speculation": True,
				"branch_count": branch_count,
			},
		)

		self._speculation_parent_units[unit.id] = unit
		self._speculation_completions[unit.id] = []

		# Generate N branch clones with distinct approach hints
		hints = self.SPECULATION_APPROACH_HINTS[:branch_count]
		for i in range(branch_count):
			branch_id = _new_id()
			hint = hints[i] if i < len(hints) else f"approach_{i}"
			branch = WorkUnit(
				id=branch_id,
				plan_id=unit.plan_id,
				title=f"{unit.title} [branch {i}: {hint.split(':')[0]}]",
				description=f"{unit.description}\n\nAPPROACH HINT: {hint}",
				files_hint=unit.files_hint,
				verification_hint=unit.verification_hint,
				priority=unit.priority,
				epoch_id=epoch.id,
				unit_type="speculation_branch",
				speculation_parent_id=unit.id,
				speculation_score=unit.speculation_score,
				acceptance_criteria=unit.acceptance_criteria,
				specialist=unit.specialist,
			)

			await self._semaphore.acquire()
			self._in_flight_count += 1
			try:
				self.db.insert_work_unit(branch)
			except Exception as exc:
				logger.error("Failed to insert speculation branch: %s", exc)
				self._in_flight_count -= 1
				self._semaphore.release()
				continue

			self._total_dispatched += 1

			task = asyncio.create_task(
				self._execute_single_unit(branch, epoch, mission),
			)
			self._active_tasks.add(task)
			self._unit_tasks[branch.id] = task

			def _on_branch_done(t: asyncio.Task[None], uid: str = branch.id) -> None:
				self._active_tasks.discard(t)
				self._unit_tasks.pop(uid, None)
				if not t.cancelled():
					exc = t.exception()
					if exc is not None:
						logger.error("Speculation branch task failed: %s", exc, exc_info=exc)

			task.add_done_callback(_on_branch_done)

		logger.info(
			"Dispatched %d speculation branches for unit %s (score=%.2f)",
			branch_count, unit.id[:12], unit.speculation_score,
		)
		return True

	async def _speculation_select_winner(
		self,
		parent_id: str,
		mission: Mission,
		epoch: Epoch,
	) -> None:
		"""Select the winning branch and merge it, discard losers."""
		completions = self._speculation_completions.pop(parent_id, [])
		parent_unit = self._speculation_parent_units.pop(parent_id, None)
		if parent_unit is None:
			logger.warning("Speculation parent %s not found", parent_id)
			return

		spec_cfg = self.config.speculation
		metric = spec_cfg.selection_metric

		# Separate passing and failing branches
		passing: list[WorkerCompletion] = []
		failing: list[WorkerCompletion] = []
		for c in completions:
			if c.unit.status == "completed" and c.unit.commit_hash:
				passing.append(c)
			else:
				failing.append(c)

		branch_ids = [c.unit.id for c in completions]
		total_cost = sum(c.unit.cost_usd for c in completions)

		if not passing:
			logger.warning(
				"All %d speculation branches failed for parent %s",
				len(completions), parent_id[:12],
			)
			parent_unit.status = "failed"
			parent_unit.finished_at = _now_iso()
			parent_unit.cost_usd = total_cost
			try:
				self.db.update_work_unit(parent_unit)
			except Exception:
				pass
			self._total_failed += 1
			self._record_speculation_result(
				parent_unit, "", branch_ids, {}, total_cost, metric, mission, epoch,
			)
			# Release all workspaces
			for c in completions:
				if c.workspace and self._backend:
					try:
						await self._backend.release_workspace(c.workspace)
					except Exception:
						pass
			return

		# Score passing branches
		scores: dict[str, float] = {}
		if metric == "review_score":
			for c in passing:
				review = await self._blocking_review(
					c.unit, c.workspace, mission, epoch,
				)
				scores[c.unit.id] = review.avg_score if review else 5.0
		elif metric == "cost":
			for c in passing:
				scores[c.unit.id] = -c.unit.cost_usd  # lower cost = higher score
		else:
			for i, c in enumerate(passing):
				scores[c.unit.id] = float(len(passing) - i)

		# Select winner
		winner_id = max(scores, key=lambda k: scores[k])
		winner = next(c for c in passing if c.unit.id == winner_id)
		losers = [c for c in completions if c.unit.id != winner_id]

		# Merge winner
		merged = False
		if self._green_branch:
			try:
				merge_result = await self._green_branch.merge_unit(
					winner.workspace, winner.unit.branch_name,
				)
				merged = merge_result.merged
			except Exception as exc:
				logger.error("Speculation winner merge failed: %s", exc)

		if merged:
			parent_unit.status = "completed"
			parent_unit.commit_hash = winner.unit.commit_hash
			parent_unit.cost_usd = total_cost
			parent_unit.finished_at = _now_iso()
			try:
				self.db.update_work_unit(parent_unit)
			except Exception:
				pass
			self._completed_unit_ids.add(parent_unit.id)
			self._total_merged += 1
			self._log_unit_event(
				mission_id=mission.id,
				epoch_id=epoch.id,
				work_unit_id=parent_unit.id,
				event_type=UNIT_EVENT_SPECULATION_COMPLETED,
				stream_details={
					"winner": winner_id,
					"branch_count": len(completions),
					"scores": scores,
				},
			)
		else:
			parent_unit.status = "failed"
			parent_unit.cost_usd = total_cost
			parent_unit.finished_at = _now_iso()
			try:
				self.db.update_work_unit(parent_unit)
			except Exception:
				pass
			self._total_failed += 1

		# Release loser workspaces
		for c in losers:
			if c.workspace and self._backend:
				try:
					await self._backend.release_workspace(c.workspace)
				except Exception:
					pass

		self._record_speculation_result(
			parent_unit, winner_id if merged else "", branch_ids,
			scores, total_cost, metric, mission, epoch,
		)

	def _record_speculation_result(
		self,
		parent_unit: WorkUnit,
		winner_branch_id: str,
		branch_ids: list[str],
		branch_scores: dict[str, float],
		total_cost: float,
		selection_metric: str,
		mission: Mission,
		epoch: Epoch,
	) -> None:
		"""Persist a SpeculationResult to the database."""
		try:
			result = SpeculationResult(
				parent_unit_id=parent_unit.id,
				winner_branch_id=winner_branch_id,
				mission_id=mission.id,
				epoch_id=epoch.id,
				branch_count=len(branch_ids),
				branch_ids=",".join(branch_ids),
				branch_scores=json.dumps(branch_scores),
				total_speculation_cost_usd=total_cost,
				selection_metric=selection_metric,
			)
			self.db.insert_speculation_result(result)
		except Exception as exc:
			logger.warning("Failed to record speculation result: %s", exc)

	def _build_causal_signal(
		self,
		unit: WorkUnit,
		epoch: Epoch,
		mission: Mission,
		merged: bool,
	) -> CausalSignal:
		"""Build a CausalSignal from a completed unit's attributes."""
		file_count = len(unit.files_hint.split(",")) if unit.files_hint else 0
		models_cfg = getattr(self.config, "models", None)
		model = getattr(models_cfg, "worker_model", None) or self.config.scheduler.model
		return CausalSignal(
			work_unit_id=unit.id,
			mission_id=mission.id,
			epoch_id=epoch.id,
			specialist=unit.specialist,
			model=model,
			file_count=file_count,
			has_dependencies=bool(unit.depends_on),
			attempt=unit.attempt,
			unit_type=unit.unit_type,
			epoch_size=self._total_dispatched,
			concurrent_units=self._in_flight_count,
			has_overlap=False,
			outcome="merged" if merged else "failed",
			failure_stage="" if merged else "execution",
		)

	def _compute_planner_risks(self) -> str:
		"""Compute a causal risk summary table for planner injection."""
		risk_dims = [
			("specialist", ["test-writer", "refactorer", "debugger", ""]),
			("unit_type", ["implementation", "research", "experiment"]),
		]
		risk_lines: list[str] = []
		for dim_type, values in risk_dims:
			for val in values:
				p = self._causal_attributor.p_failure(dim_type, val)
				if p is not None and p > 0.15:
					label = val or "(general)"
					risk_lines.append(f"- {dim_type}={label}: {p:.0%} failure rate")
		# File count buckets
		for bucket in ["1", "2-3", "4-5", "6+"]:
			p = self._causal_attributor.p_failure("file_count", bucket)
			if p is not None and p > 0.15:
				risk_lines.append(f"- file_count={bucket}: {p:.0%} failure rate")
		if not risk_lines:
			return ""
		return "## Causal Risk Factors\n" + "\n".join(risk_lines)

	async def _run_evaluator_agent(self, mission: Mission, workspace: str) -> dict[str, Any]:
		"""Spawn a Claude subprocess with shell/file access to evaluate the mission result.

		Unlike _verify_objective (which reads summaries), this agent can actually
		run the software: execute tests, start the app, check HTTP endpoints, inspect files.

		Returns {"passed": bool, "evidence": [...], "gaps": [...]}.
		"""
		ev = self.config.evaluator
		prompt = (
			f"You are an evaluator agent. Your job is to determine whether the following "
			f"objective has been achieved by actually running and testing the software.\n\n"
			f"## Objective\n{mission.objective}\n\n"
			f"## Instructions\n"
			f"1. Read the relevant source files to understand what was implemented.\n"
			f"2. Run the test suite to verify tests pass.\n"
			f"3. If applicable, start the application and test it works from a user perspective.\n"
			f"4. Check for any obvious gaps, broken functionality, or missing features.\n"
			f"5. Be thorough but focused on the stated objective.\n\n"
			f"End your response with EXACTLY:\n"
			f'EVALUATION:{{"passed": true/false, "evidence": ["what works"], "gaps": ["what is missing"]}}'
		)

		eval_cmd = build_claude_cmd(
			self.config, model=ev.model, max_turns=ev.max_turns, budget=ev.budget_usd,
		)
		try:
			proc = await asyncio.create_subprocess_exec(
				*eval_cmd,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(self.config),
				cwd=workspace,
			)
			stdout, _ = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
				timeout=ev.timeout,
			)
			output = stdout.decode() if stdout else ""
		except asyncio.TimeoutError:
			logger.warning("Evaluator agent timed out after %ds", ev.timeout)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			return {"passed": False, "evidence": [], "gaps": ["Evaluator timed out"]}
		except (FileNotFoundError, OSError) as exc:
			logger.warning("Evaluator agent failed to start: %s", exc)
			return {"passed": False, "evidence": [], "gaps": [str(exc)]}

		# Parse EVALUATION marker
		marker = "EVALUATION:"
		idx = output.rfind(marker)
		if idx == -1:
			logger.warning("No EVALUATION marker found in evaluator output")
			return {"passed": False, "evidence": [], "gaps": ["No EVALUATION marker in output"]}

		remainder = output[idx + len(marker):]
		data = extract_json_from_text(remainder)
		if not isinstance(data, dict):
			logger.warning("Failed to parse EVALUATION JSON")
			return {"passed": False, "evidence": [], "gaps": ["Failed to parse EVALUATION JSON"]}

		return {
			"passed": bool(data.get("passed", False)),
			"evidence": list(data.get("evidence", [])),
			"gaps": list(data.get("gaps", [])),
		}

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
		verify_cmd = build_claude_cmd(self.config, model=model, budget=0.50)
		try:
			proc = await asyncio.create_subprocess_exec(
				*verify_cmd,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(self.config),
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

			try:
				from mission_control.snapshot import get_project_snapshot
				_snap = get_project_snapshot(self.config.target.resolved_path)
			except Exception:
				_snap = ""

			review = await self._diff_reviewer.review_unit(
				unit=unit,
				diff=diff,
				objective=mission.objective,
				mission_id=mission.id,
				epoch_id=epoch.id,
				project_snapshot=_snap,
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

			try:
				from mission_control.snapshot import get_project_snapshot
				_snap = get_project_snapshot(self.config.target.resolved_path)
			except Exception:
				_snap = ""

			review = await self._diff_reviewer.review_unit(
				unit=unit,
				diff=diff,
				objective=mission.objective,
				mission_id=mission.id,
				epoch_id=epoch.id,
				project_snapshot=_snap,
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

	def _accept_merge(
		self,
		unit: WorkUnit,
		merge_result: UnitMergeResult,
		workspace: str,
		mission: Mission,
		epoch: Epoch,
		result: ContinuousMissionResult,
	) -> None:
		"""Run post-merge bookkeeping: sync check, fire-and-forget review, counter updates."""
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
				asyncio.create_task(self._notifier.send(
					f"Sync warning: Unit '{unit.title}' merged to green but failed "
					"to sync back to source repo. Changes may not be visible in the source.",
				))

		# Fire-and-forget LLM diff review
		if self.config.review.enabled:
			criteria_passed = (
				self.config.review.skip_when_criteria_passed
				and unit.acceptance_criteria
				and merge_result.verification_passed
			)
			if criteria_passed:
				logger.debug("Skipping review for unit %s: acceptance criteria passed", unit.id)
			else:
				task = asyncio.create_task(
					self._review_merged_unit(
						unit, workspace, mission, epoch,
						merge_commit_hash=merge_result.merge_commit_hash,
					),
				)
				self._active_tasks.add(task)
				task.add_done_callback(self._task_done_callback)

		# Update counters
		self._total_merged += 1
		self._completed_unit_ids.add(unit.id)
		if merge_result.changed_files:
			self._merged_files |= set(merge_result.changed_files)
		else:
			mf = _parse_files_hint(unit.files_hint)
			if mf:
				self._merged_files |= mf

	def _schedule_batch_flush(self) -> None:
		"""Schedule a timer to flush the merge queue after batch_merge_wait_seconds."""
		if self._merge_queue_timer is not None:
			return  # timer already running
		wait = self.config.green_branch.batch_merge_wait_seconds
		loop = asyncio.get_event_loop()
		self._merge_queue_timer = loop.call_later(
			wait, lambda: asyncio.ensure_future(self._flush_merge_batch()),
		)

	async def _flush_merge_batch(self) -> None:
		"""Flush the pending merge queue via merge_batch (or merge_unit for single)."""
		if self._merge_queue_timer is not None:
			self._merge_queue_timer.cancel()
			self._merge_queue_timer = None

		entries = list(self._merge_queue)
		self._merge_queue.clear()
		if not entries or self._green_branch is None:
			return

		# Build the units list for merge_batch
		units_for_batch: list[tuple[str, str, str]] = []
		for entry in entries:
			c = entry.completion
			units_for_batch.append(
				(c.workspace, c.unit.branch_name, c.unit.acceptance_criteria),
			)

		try:
			results = await self._green_branch.merge_batch(units_for_batch)
		except Exception as exc:
			logger.error("merge_batch failed: %s", exc, exc_info=True)
			# Mark all entries as failed
			for entry in entries:
				self._total_failed += 1
				_exc_details = {
					"failure_output": str(exc)[:500],
					"failure_stage": "exception",
				}
				self._log_unit_event(
					mission_id=entry.mission.id,
					epoch_id=entry.completion.epoch.id,
					work_unit_id=entry.completion.unit.id,
					event_type=UNIT_EVENT_MERGE_FAILED,
					details=json.dumps(_exc_details),
					stream_details=_exc_details,
				)
			return

		# Process each result with per-unit bookkeeping
		for entry, merge_result in zip(entries, results):
			c = entry.completion
			unit = c.unit
			epoch = c.epoch
			mission = entry.mission
			result = entry.result
			workspace = c.workspace
			handoff = c.handoff
			cont = self.config.continuous

			self._degradation.record_merge_attempt(conflict=not merge_result.merged)

			if merge_result.merged:
				self._trace(unit.id, unit.id, "merge_succeeded")
				self._accept_merge(unit, merge_result, workspace, mission, epoch, result)
			else:
				fixup_stages = ("pre_merge_verification", "acceptance_criteria", "merge_conflict")
				if (
					merge_result.failure_stage in fixup_stages
					and workspace
					and unit.id not in self._active_fixups
				):
					logger.info(
						"Launching background fixup for unit %s (stage=%s)",
						unit.id[:12], merge_result.failure_stage,
					)
					task = asyncio.create_task(
						self._background_fixup(
							unit, workspace, merge_result, cont,
							mission, epoch, result, handoff,
						),
					)
					self._active_fixups[unit.id] = task
					task.add_done_callback(self._fixup_done_callback)
				else:
					logger.warning(
						"Unit %s failed merge: %s",
						unit.id, merge_result.failure_output[-200:],
					)
					self._trace(
						unit.id, unit.id, "merge_failed",
						failure_stage=merge_result.failure_stage,
					)
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
					if handoff:
						handoff.concerns.append(
							f"Merge failed: {merge_result.failure_output[-500:]}",
						)
					failure_reason = merge_result.failure_output[-1000:]
					if unit.attempt < unit.max_attempts:
						self._schedule_retry(unit, epoch, mission, failure_reason, cont)
					else:
						self._total_failed += 1
						self._log_unit_event(
							mission_id=mission.id,
							epoch_id=epoch.id,
							work_unit_id=unit.id,
							event_type=UNIT_EVENT_REJECTED,
							details=merge_result.failure_output[-2000:],
							stream_details={"failure_output": merge_result.failure_output[-2000:]},
						)

	def _fixup_done_callback(self, task: asyncio.Task[None]) -> None:
		"""Remove completed fixup tasks from the active set and log exceptions."""
		# Find and remove the unit_id for this task
		unit_id = None
		for uid, t in self._active_fixups.items():
			if t is task:
				unit_id = uid
				break
		if unit_id is not None:
			del self._active_fixups[unit_id]
		if task.cancelled():
			return
		exc = task.exception()
		if exc is not None:
			logger.error("Background fixup task failed: %s", exc, exc_info=exc)

	async def _background_fixup(
		self,
		unit: WorkUnit,
		workspace: str,
		merge_result: UnitMergeResult,
		cont: ContinuousConfig,
		mission: Mission,
		epoch: Epoch,
		result: ContinuousMissionResult,
		handoff: Handoff | None,
	) -> None:
		"""Run fixup in the background so completion processing is not blocked.

		Calls _resume_worker_for_fixup and then handles all post-fixup
		bookkeeping (accept merge on success, or log failure/retry on failure).
		"""
		try:
			fixed = await self._resume_worker_for_fixup(
				unit, workspace, merge_result, cont,
			)
			if fixed:
				logger.info("Unit %s merged after background fixup", unit.id)
				self._trace(unit.id, unit.id, "merge_succeeded")
				self._accept_merge(
					unit, fixed, workspace,
					mission, epoch, result,
				)
			else:
				logger.warning(
					"Background fixup failed for unit %s: %s",
					unit.id, merge_result.failure_output[-200:],
				)
				self._trace(
					unit.id, unit.id, "merge_failed",
					failure_stage=merge_result.failure_stage,
				)
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
				if handoff:
					handoff.concerns.append(
						f"Merge failed: {merge_result.failure_output[-500:]}",
					)
				failure_reason = merge_result.failure_output[-1000:]
				if unit.attempt < unit.max_attempts:
					self._schedule_retry(unit, epoch, mission, failure_reason, cont)
				else:
					self._total_failed += 1
					self._log_unit_event(
						mission_id=mission.id,
						epoch_id=epoch.id,
						work_unit_id=unit.id,
						event_type=UNIT_EVENT_REJECTED,
						details=merge_result.failure_output[-2000:],
						stream_details={"failure_output": merge_result.failure_output[-2000:]},
					)
					if self._notifier:
						await self._notifier.send_merge_conflict(
							unit.title, merge_result.failure_output[-500:],
						)
		except Exception as exc:
			logger.error("Background fixup crashed for unit %s: %s", unit.id, exc, exc_info=True)
			self._total_failed += 1

	async def _resume_worker_for_fixup(
		self,
		unit: WorkUnit,
		workspace: str,
		merge_result: UnitMergeResult,
		cont: ContinuousConfig,
		max_fixup_attempts: int = 2,
	) -> UnitMergeResult | None:
		"""Resume the worker session (or spawn cold fixup) and re-attempt merge.

		If the unit has a session_id, resumes the original worker session via
		``--resume`` so the fixup agent has full conversation context. Falls back
		to a cold ``-p`` call when no session_id is available (backward compat).

		Returns a successful UnitMergeResult if fixup worked, None otherwise.
		"""
		for attempt in range(max_fixup_attempts):
			self._trace(unit.id, unit.id, "fixup_started", attempt=attempt + 1)
			logger.info(
				"Worker fixup attempt %d/%d for unit %s in %s (resume=%s)",
				attempt + 1, max_fixup_attempts, unit.id[:12], workspace,
				bool(unit.session_id),
			)

			models_cfg = getattr(self.config, "models", None)
			model = getattr(models_cfg, "fixup_model", None) or self.config.scheduler.model

			is_merge_conflict = merge_result.failure_stage == "merge_conflict"
			green_branch = self.config.green_branch.green_branch

			# Sync latest mc/green into the worker's workspace so it can
			# rebase against already-merged work during conflict resolution.
			if is_merge_conflict and self._green_branch:
				green_ws = self._green_branch.workspace
				if green_ws:
					sync_proc = await asyncio.create_subprocess_exec(
						"git", "fetch", green_ws,
						f"+{green_branch}:{green_branch}",
						cwd=workspace,
						stdout=asyncio.subprocess.PIPE,
						stderr=asyncio.subprocess.STDOUT,
					)
					await sync_proc.communicate()

			if unit.session_id:
				# Resume: the session already has the full task context,
				# so we only need to describe the failure.
				if is_merge_conflict:
					prompt = CONFLICT_RESOLUTION_PROMPT.format(
						green_branch=green_branch,
						failure_output=merge_result.failure_output,
						verification_command=self.config.target.verification.command,
					)
				else:
					prompt = (
						f"Your code failed post-merge verification.\n\n"
						f"## Failure Stage\n{merge_result.failure_stage}\n\n"
						f"## Output\n{merge_result.failure_output}\n\n"
						f"## Verification Command\n{self.config.target.verification.command}\n\n"
						f"Fix the issue and commit."
					)
				if unit.acceptance_criteria and merge_result.failure_stage == "acceptance_criteria":
					prompt += f"\n\n## Acceptance Criteria\n{unit.acceptance_criteria}"
				cmd = build_claude_cmd(
					self.config, model=model, max_turns=5,
					permission_mode="bypassPermissions",
					resume_session=unit.session_id, prompt=prompt,
				)
			else:
				# Cold fixup: no session to resume, include full task context.
				if is_merge_conflict:
					prompt = (
						f"## Original Task\n{unit.title}\n{unit.description}\n\n"
						+ CONFLICT_RESOLUTION_PROMPT.format(
							green_branch=green_branch,
							failure_output=merge_result.failure_output,
							verification_command=self.config.target.verification.command,
						)
					)
				else:
					prompt = (
						f"Your previous work on this task failed verification after merging.\n\n"
						f"## Original Task\n{unit.title}\n{unit.description}\n\n"
						f"## Failure Stage\n{merge_result.failure_stage}\n\n"
						f"## Failure Output\n{merge_result.failure_output}\n\n"
						f"## Verification Command\n{self.config.target.verification.command}\n\n"
						f"Fix the issues in your code so verification passes. "
						f"Run the verification command to confirm, then commit your fix."
					)
				if unit.acceptance_criteria and merge_result.failure_stage == "acceptance_criteria":
					prompt += f"\n\n## Acceptance Criteria\n{unit.acceptance_criteria}"
				cmd = build_claude_cmd(
					self.config, model=model, max_turns=5,
					permission_mode="bypassPermissions", prompt=prompt,
				)
			try:
				env = claude_subprocess_env(self.config)
			except Exception:
				env = {}
			full_env = {**os.environ, **env}

			try:
				proc = await asyncio.create_subprocess_exec(
					*cmd,
					cwd=workspace,
					stdout=asyncio.subprocess.PIPE,
					stderr=asyncio.subprocess.PIPE,
					env=full_env,
				)
				await asyncio.wait_for(proc.communicate(), timeout=cont.worker_fixup_timeout)
			except asyncio.TimeoutError:
				logger.warning("Worker fixup timed out for unit %s", unit.id[:12])
				try:
					proc.kill()
					await proc.wait()
				except ProcessLookupError:
					pass
				continue
			except (FileNotFoundError, OSError) as exc:
				logger.warning("Worker fixup failed to spawn for unit %s: %s", unit.id[:12], exc)
				break

			# Re-attempt merge with the fixed code
			if self._green_branch is None:
				break
			new_result = await self._green_branch.merge_unit(
				workspace, unit.branch_name,
				acceptance_criteria=unit.acceptance_criteria,
			)
			if new_result.merged:
				self._trace(unit.id, unit.id, "fixup_succeeded", attempt=attempt + 1)
				return new_result

			# Update failure info for next attempt
			merge_result = new_result
			logger.warning(
				"Worker fixup attempt %d failed for unit %s: %s",
				attempt + 1, unit.id[:12], new_result.failure_output[-200:],
			)

		self._trace(unit.id, unit.id, "fixup_failed", attempts=max_fixup_attempts)
		return None

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
			self._execute_single_unit(unit, epoch, mission),
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
	) -> None:
		"""Mark a unit as failed and update the worker."""
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

	def _extract_knowledge(self, unit: WorkUnit, handoff: Handoff | None, mission: Mission) -> None:
		"""Extract and persist knowledge from a non-code unit completion."""
		if handoff is None:
			return
		discoveries = handoff.discoveries or []
		content = "\n".join(discoveries) if discoveries else handoff.summary
		if not content:
			return
		from mission_control.models import KnowledgeItem
		item = KnowledgeItem(
			mission_id=mission.id,
			source_unit_id=unit.id,
			source_unit_type=unit.unit_type,
			title=unit.title[:200],
			content=content[:2000],
			rationale=handoff.summary[:500] if handoff.summary else "",
			scope=unit.files_hint or "",
		)
		try:
			self.db.insert_knowledge_item(item)
			logger.info(
				"Extracted knowledge from %s unit %s: %s",
				unit.unit_type, unit.id[:12], item.title[:60],
			)
		except Exception as exc:
			logger.warning("Failed to insert knowledge item for unit %s: %s", unit.id, exc)

	async def _execute_single_unit(
		self,
		unit: WorkUnit,
		epoch: Epoch,
		mission: Mission,
	) -> WorkerCompletion | None:
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

			# Prompt evolution: select variant and prepend to context
			selected_variant_id = ""
			if self._prompt_evolution is not None:
				try:
					variant = self._prompt_evolution.select_variant("worker")
					if variant is not None:
						selected_variant_id = variant.variant_id
						context = variant.content + "\n\n" + context if context else variant.content
				except Exception as exc:
					logger.debug("Prompt variant selection failed: %s", exc)

			# Append per-unit causal risk factors
			try:
				models_cfg = getattr(self.config, "models", None)
				worker_model = (
					getattr(models_cfg, "worker_model", None) or self.config.scheduler.model
				)
				risks = self._causal_attributor.top_risk_factors(
					unit, model=worker_model,
					epoch_size=self._total_dispatched,
					concurrent_units=self._in_flight_count,
				)
				risk_section = CausalAttributor.format_risk_section(risks)
				if risk_section:
					experience_context = (
						(experience_context or "") + "\n\n" + risk_section
					).strip()
			except Exception:
				pass

			# Inject accumulated knowledge into worker context
			knowledge_items = self.db.get_knowledge_for_mission(mission.id)
			if knowledge_items:
				from mission_control.overlap import _parse_files_hint as _parse_scope
				unit_files = _parse_scope(unit.files_hint)
				relevant = [
					k for k in knowledge_items
					if unit_files and _parse_scope(k.scope) & unit_files
				]
				if not relevant:
					relevant = knowledge_items[-5:]
				knowledge_section = "\n".join(
					f"- [{k.source_unit_type}] {k.title}: {k.content[:200]}"
					for k in relevant
				)
				context = (context or "") + f"\n\n## Accumulated Knowledge\n{knowledge_section}"

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
			running_units = await self.db.locked_call("get_running_units", unit.id)
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

			if self._is_cleanup_mission and not unit.specialist:
				unit.specialist = "simplifier"

			specialist_template = load_specialist_template(unit.specialist)

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
			session_id = str(uuid.uuid4())
			unit.session_id = session_id
			cmd = build_claude_cmd(
				self.config, model=model, output_format="stream-json",
				permission_mode="bypassPermissions", budget=budget,
				session_id=session_id, prompt=prompt,
			)

			effective_timeout = unit.timeout or self.config.scheduler.session_timeout
			handle = await self._backend.spawn(
				unit.id, workspace, cmd,
				timeout=effective_timeout,
			)
			self._trace(unit.id, unit.id, "worker_spawned", prompt_length=len(prompt), prompt_summary=prompt[:200])

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
			self._trace(unit.id, unit.id, "session_started", pid=handle.pid, workspace=workspace)
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

			self._trace(
				unit.id, unit.id, "session_ended",
				exit_code=unit.status,
				input_tokens=unit.input_tokens,
				output_tokens=unit.output_tokens,
			)

			completion = WorkerCompletion(
				unit=unit, handoff=handoff, workspace=workspace, epoch=epoch,
				prompt_variant_id=selected_variant_id,
			)
			return completion

		except (RuntimeError, OSError) as e:
			logger.error("Infrastructure error executing unit %s: %s", unit.id, e)
			await self._fail_unit(unit, worker, epoch, f"Infrastructure error: {e}", workspace)
		except asyncio.CancelledError:
			logger.info("Unit %s execution cancelled", unit.id)
			await self._fail_unit(unit, worker, epoch, "Cancelled", workspace)
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
			self._file_locks.release(unit.id)
			if workspace:
				await self._backend.release_workspace(workspace)
			self._in_flight_count = max(self._in_flight_count - 1, 0)
			self._semaphore.release()

	def _write_strategy_update(self, revised_strategy: str) -> None:
		"""Overwrite MISSION_STRATEGY.md with revised strategy."""
		target_path = self.config.target.resolved_path
		strategy_path = target_path / "MISSION_STRATEGY.md"
		try:
			strategy_path.write_text(revised_strategy + "\n")
			logger.info("Updated MISSION_STRATEGY.md with revised strategy")
		except OSError as exc:
			logger.warning("Could not write MISSION_STRATEGY.md: %s", exc)

	def _should_stop(self, mission: Mission) -> str:
		"""Check stopping conditions. Returns reason string or empty."""
		if not self.running:
			return "user_stopped"

		# Check DB signals
		signal_reason = self._check_signals(mission.id)
		if signal_reason:
			return signal_reason

		cont = self.config.continuous

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
					self._semaphore.adjust(new_count)
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

	def _build_planner_context(self, mission_id: str) -> str:
		return build_planner_context(self.db, mission_id)

	def _update_mission_state(self, mission: Mission) -> None:
		update_mission_state(
			self.db, mission, self.config, self._state_changelog,
			degradation_status=self._degradation.get_status_dict(),
			strategy=self._current_strategy,
		)

	def stop(self) -> None:
		self.running = False
