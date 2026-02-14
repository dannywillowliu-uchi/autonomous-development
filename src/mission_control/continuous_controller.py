"""Continuous controller -- event-driven development loop without round boundaries."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from mission_control.backends import LocalBackend, SSHBackend, WorkerBackend
from mission_control.config import MissionConfig
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.feedback import get_worker_context
from mission_control.green_branch import GreenBranchManager
from mission_control.models import (
	Epoch,
	Handoff,
	Mission,
	Plan,
	PlanNode,
	Signal,
	UnitEvent,
	WorkUnit,
	_now_iso,
)
from mission_control.session import parse_mc_result
from mission_control.token_parser import compute_token_cost, parse_stream_json
from mission_control.worker import render_mission_worker_prompt

logger = logging.getLogger(__name__)


@dataclass
class WorkerCompletion:
	"""A completed unit ready for verification and merge."""

	unit: WorkUnit
	handoff: Handoff | None
	workspace: str
	epoch: Epoch


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
		self._completion_queue: asyncio.Queue[WorkerCompletion] = asyncio.Queue()
		self._active_tasks: set[asyncio.Task[None]] = set()
		self._start_time: float = 0.0
		self._total_dispatched: int = 0
		self._total_merged: int = 0
		self._total_failed: int = 0

	async def run(self) -> ContinuousMissionResult:
		"""Run the continuous mission loop until objective met or stopping condition."""
		result = ContinuousMissionResult(objective=self.config.target.objective)
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

		try:
			await self._init_components()

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

			mission.status = "completed" if result.objective_met else "stopped"
			mission.finished_at = _now_iso()
			mission.stopped_reason = result.stopped_reason
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

		return result

	async def _init_components(self) -> None:
		"""Initialize backend, green branch manager, and continuous planner."""
		source_repo = str(self.config.target.resolved_path)

		# Backend
		if self.config.backend.type == "ssh":
			self._backend = SSHBackend(self.config.backend.ssh_hosts)
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
		if isinstance(self._backend, LocalBackend):
			gb_workspace = await self._backend.provision_workspace(
				"green-branch-mgr", source_repo, self.config.target.branch,
			)
			await self._green_branch.initialize(gb_workspace)
		else:
			raise NotImplementedError(
				"Continuous mode requires a local workspace for green branch "
				"operations. SSH backend is not yet supported."
			)

		# Continuous planner (wraps RecursivePlanner)
		self._planner = ContinuousPlanner(self.config, self.db)

	async def _dispatch_loop(
		self,
		mission: Mission,
		result: ContinuousMissionResult,
	) -> None:
		"""Dispatch work units to free workers as they become available."""
		assert self._planner is not None
		assert self._backend is not None

		num_workers = self.config.scheduler.parallel.num_workers
		semaphore = asyncio.Semaphore(num_workers)
		cooldown = self.config.continuous.cooldown_between_units

		while self.running:
			# Expire stale signals
			try:
				self.db.expire_stale_signals(timeout_minutes=10)
			except Exception:
				pass

			# Check stopping conditions before dispatching
			reason = self._should_stop(mission)
			if reason:
				result.stopped_reason = reason
				self.running = False
				break

			# Build feedback context for the planner
			feedback_context = self._build_planner_context(mission.id)

			# Get next batch of units from the planner
			try:
				plan, units, epoch = await self._planner.get_next_units(
					mission,
					max_units=min(num_workers, 3),
					feedback_context=feedback_context,
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

			# Dispatch each unit
			for unit in units:
				await semaphore.acquire()
				unit.epoch_id = epoch.id
				try:
					self.db.insert_work_unit(unit)
				except Exception as exc:
					logger.error(
						"Failed to insert work unit: %s", exc, exc_info=True,
					)
					semaphore.release()
					continue

				# Log dispatch event
				try:
					self.db.insert_unit_event(UnitEvent(
						mission_id=mission.id,
						epoch_id=epoch.id,
						work_unit_id=unit.id,
						event_type="dispatched",
					))
				except Exception:
					pass

				self._total_dispatched += 1

				task = asyncio.create_task(
					self._execute_single_unit(
						unit, epoch, mission, semaphore,
					),
				)
				self._active_tasks.add(task)
				task.add_done_callback(self._active_tasks.discard)

			if cooldown > 0:
				await asyncio.sleep(cooldown)

	async def _process_completions(
		self,
		mission: Mission,
		result: ContinuousMissionResult,
	) -> None:
		"""Process completed units: verify, merge, record feedback."""
		assert self._green_branch is not None

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

			# Merge if the unit completed with commits
			merged = False
			if unit.status == "completed" and unit.commit_hash:
				try:
					merge_result = await self._green_branch.merge_unit(
						workspace, unit.branch_name,
					)
					merged = merge_result.merged

					if merged:
						logger.info(
							"Unit %s merged to green",
							unit.id,
						)
						self._total_merged += 1
						# Log merge event
						try:
							self.db.insert_unit_event(UnitEvent(
								mission_id=mission.id,
								epoch_id=epoch.id,
								work_unit_id=unit.id,
								event_type="merged",
								input_tokens=unit.input_tokens,
								output_tokens=unit.output_tokens,
							))
						except Exception:
							pass
					else:
						logger.warning(
							"Unit %s failed merge: %s",
							unit.id, merge_result.failure_output[:200],
						)
						self._total_failed += 1
						# Log rejection event
						try:
							self.db.insert_unit_event(UnitEvent(
								mission_id=mission.id,
								epoch_id=epoch.id,
								work_unit_id=unit.id,
								event_type="rejected",
								details=merge_result.failure_output[:500],
							))
						except Exception:
							pass

						# Append merge failure to handoff concerns
						if handoff:
							try:
								concerns = json.loads(handoff.concerns or "[]")
							except (json.JSONDecodeError, TypeError):
								concerns = []
							concerns.append(
								f"Merge failed: {merge_result.failure_output[:200]}",
							)
							handoff.concerns = json.dumps(concerns)
				except Exception as exc:
					logger.error(
						"merge_unit failed for %s: %s",
						unit.id, exc, exc_info=True,
					)
					self._total_failed += 1
			elif unit.status == "completed":
				# Completed but no commits
				self._total_merged += 1
				merged = True
			else:
				self._total_failed += 1

			# Feed handoff to planner for adaptive replanning
			if handoff and self._planner:
				self._planner.ingest_handoff(handoff)

			# Accumulate cost
			mission.total_cost_usd += unit.cost_usd

			try:
				self.db.update_mission(mission)
			except Exception:
				pass

	async def _execute_single_unit(
		self,
		unit: WorkUnit,
		epoch: Epoch,
		mission: Mission,
		semaphore: asyncio.Semaphore,
	) -> None:
		"""Execute a single work unit and put completion on queue."""
		assert self._backend is not None

		source_repo = str(self.config.target.resolved_path)
		base_branch = self.config.green_branch.green_branch
		workspace = ""

		try:
			try:
				workspace = await self._backend.provision_workspace(
					unit.id, source_repo, base_branch,
				)
			except RuntimeError as e:
				logger.error("Failed to provision workspace: %s", e)
				unit.attempt += 1
				unit.status = "failed"
				unit.output_summary = str(e)
				unit.finished_at = _now_iso()
				await self.db.locked_call("update_work_unit", unit)
				await self._completion_queue.put(
					WorkerCompletion(unit=unit, handoff=None, workspace="", epoch=epoch),
				)
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
			)

			budget = self.config.scheduler.budget.max_per_session_usd
			cmd = [
				"claude", "-p",
				"--output-format", "stream-json",
				"--permission-mode", "bypassPermissions",
				"--model", self.config.scheduler.model,
				"--max-budget-usd", str(budget),
				prompt,
			]

			effective_timeout = unit.timeout or self.config.scheduler.session_timeout
			handle = await self._backend.spawn(
				unit.id, workspace, cmd,
				timeout=effective_timeout,
			)

			# Wait for completion
			poll_deadline = int(
				effective_timeout * self.config.rounds.timeout_multiplier,
			)
			monitor_interval = self.config.scheduler.monitor_interval
			start = time.monotonic()
			while time.monotonic() - start < poll_deadline:
				status = await self._backend.check_status(handle)
				if status != "running":
					break
				if not self.running:
					await self._backend.kill(handle)
					unit.attempt += 1
					unit.status = "failed"
					unit.output_summary = "Stopped by signal"
					unit.finished_at = _now_iso()
					await self.db.locked_call("update_work_unit", unit)
					await self._completion_queue.put(
						WorkerCompletion(
							unit=unit, handoff=None, workspace=workspace, epoch=epoch,
						),
					)
					return
				await self._backend.get_output(handle)
				await asyncio.sleep(monitor_interval)
			else:
				await self._backend.kill(handle)
				unit.attempt += 1
				unit.status = "failed"
				unit.output_summary = f"Timed out after {effective_timeout}s"
				unit.finished_at = _now_iso()
				await self.db.locked_call("update_work_unit", unit)
				await self._completion_queue.put(
					WorkerCompletion(
						unit=unit, handoff=None, workspace=workspace, epoch=epoch,
					),
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
					commits=json.dumps(
						commits if isinstance(commits, list) else [],
					),
					summary=unit.output_summary,
					discoveries=json.dumps(
						disc if isinstance(disc, list) else [],
					),
					concerns=json.dumps(
						conc if isinstance(conc, list) else [],
					),
					files_changed=json.dumps(
						fc if isinstance(fc, list) else [],
					),
				)
				await self.db.locked_call("insert_handoff", handoff)
				unit.handoff_id = handoff.id
			else:
				unit_status = "completed" if status == "completed" else "failed"
				max_chars = self.config.scheduler.output_summary_max_chars
				unit.output_summary = (
					output[-max_chars:] if output else "No output"
				)

			# Set final status (NO merge_to_working -- completion processor handles merge)
			if unit_status in ("completed", "blocked"):
				unit.status = unit_status
			else:
				unit.attempt += 1
				unit.status = "failed"

			unit.finished_at = _now_iso()
			await self.db.locked_call("update_work_unit", unit)

			# Put on completion queue for verify+merge
			await self._completion_queue.put(
				WorkerCompletion(
					unit=unit, handoff=handoff, workspace=workspace, epoch=epoch,
				),
			)

		except (RuntimeError, OSError) as e:
			logger.error("Infrastructure error executing unit %s: %s", unit.id, e)
			unit.attempt += 1
			unit.status = "failed"
			unit.output_summary = f"Infrastructure error: {e}"
			unit.finished_at = _now_iso()
			await self.db.locked_call("update_work_unit", unit)
			await self._completion_queue.put(
				WorkerCompletion(
					unit=unit, handoff=None, workspace=workspace, epoch=epoch,
				),
			)
		except asyncio.CancelledError:
			logger.info("Unit %s execution cancelled", unit.id)
			unit.attempt += 1
			unit.status = "failed"
			unit.output_summary = "Cancelled"
			unit.finished_at = _now_iso()
			await self.db.locked_call("update_work_unit", unit)
			# Don't put on queue -- controller is shutting down
		except (ValueError, KeyError, json.JSONDecodeError) as e:
			logger.error("Data error executing unit %s: %s", unit.id, e)
			unit.attempt += 1
			unit.status = "failed"
			unit.output_summary = f"Data error: {e}"
			unit.finished_at = _now_iso()
			await self.db.locked_call("update_work_unit", unit)
			await self._completion_queue.put(
				WorkerCompletion(
					unit=unit, handoff=None, workspace=workspace, epoch=epoch,
				),
			)
		finally:
			if workspace:
				await self._backend.release_workspace(workspace)
			semaphore.release()

	def _persist_plan_tree(self, node: PlanNode, plan: Plan) -> None:
		"""Persist the in-memory plan tree to the database."""
		try:
			self.db.insert_plan_node(node)
		except Exception as exc:
			logger.error(
				"Failed to insert plan node %s: %s", node.id, exc, exc_info=True,
			)
			raise

		if hasattr(node, "_forced_unit"):
			wu = node._forced_unit
			try:
				self.db.insert_work_unit(wu)
			except Exception as exc:
				logger.error(
					"Failed to insert forced work unit %s: %s",
					wu.id, exc, exc_info=True,
				)
				raise

		if hasattr(node, "_child_leaves"):
			for leaf, wu in node._child_leaves:
				try:
					self.db.insert_plan_node(leaf)
				except Exception as exc:
					logger.error(
						"Failed to insert child plan node %s: %s",
						leaf.id, exc, exc_info=True,
					)
					raise
				try:
					self.db.insert_work_unit(wu)
				except Exception as exc:
					logger.error(
						"Failed to insert child work unit %s: %s",
						wu.id, exc, exc_info=True,
					)
					raise

		if hasattr(node, "_subdivided_children"):
			for child in node._subdivided_children:
				self._persist_plan_tree(child, plan)

	def _build_planner_context(self, mission_id: str) -> str:
		"""Build planner context from recent handoff summaries."""
		try:
			handoffs = self.db.get_recent_handoffs(mission_id, limit=15)
		except Exception as exc:
			logger.error("Failed to get recent handoffs: %s", exc)
			return ""

		if not handoffs:
			return ""

		lines = ["## Recent Handoff Summaries"]
		merged_count = 0
		failed_count = 0

		for h in reversed(handoffs):  # oldest first
			status_label = h.status or "unknown"
			lines.append(f"\n### Unit {h.work_unit_id[:8]} ({status_label})")
			if h.summary:
				lines.append(f"Summary: {h.summary}")

			try:
				discoveries = json.loads(h.discoveries) if h.discoveries else []
			except (json.JSONDecodeError, TypeError):
				discoveries = []
			if discoveries:
				lines.append("Discoveries:")
				for d in discoveries[:5]:
					lines.append(f"  - {d}")

			try:
				concerns = json.loads(h.concerns) if h.concerns else []
			except (json.JSONDecodeError, TypeError):
				concerns = []
			if concerns:
				lines.append("Concerns:")
				for c in concerns[:5]:
					lines.append(f"  - {c}")

			if status_label == "completed":
				merged_count += 1
			else:
				failed_count += 1

		lines.append(
			f"\nMerge stats: {merged_count} merged, {failed_count} failed "
			f"(of last {len(handoffs)} units)",
		)
		return "\n".join(lines)

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
			elif signal.signal_type == "adjust":
				self._handle_adjust_signal(signal)
		return ""

	def _handle_adjust_signal(self, signal: Signal) -> None:
		"""Adjust runtime parameters from signal payload."""
		try:
			params = json.loads(signal.payload) if signal.payload else {}
			if "num_workers" in params:
				self.config.scheduler.parallel.num_workers = int(
					params["num_workers"],
				)
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

	async def _run_final_verification(self) -> tuple[bool, str]:
		"""Run full verification on mc/green at mission end.

		Returns (passed, output) tuple.
		"""
		assert self._green_branch is not None
		verify_cmd = self.config.target.verification.command
		gb = self.config.green_branch

		# Checkout mc/green in the green branch workspace
		await self._green_branch._run_git("checkout", gb.green_branch)

		# Run verification
		ok, output = await self._green_branch._run_command(verify_cmd)
		return ok, output

	def stop(self) -> None:
		self.running = False
