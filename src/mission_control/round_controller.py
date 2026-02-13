"""Round controller -- outer loop for continuous self-driving development."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from mission_control.backends import LocalBackend, SSHBackend, WorkerBackend
from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.evaluator import evaluate_objective
from mission_control.feedback import get_planner_context, get_worker_context, record_round_outcome
from mission_control.green_branch import GreenBranchManager
from mission_control.models import (
	Handoff,
	Mission,
	Plan,
	PlanNode,
	Round,
	Signal,
	WorkUnit,
	_now_iso,
)
from mission_control.recursive_planner import RecursivePlanner
from mission_control.session import parse_mc_result
from mission_control.worker import render_mission_worker_prompt

logger = logging.getLogger(__name__)


@dataclass
class MissionResult:
	"""Summary of a completed mission."""

	mission_id: str = ""
	objective: str = ""
	final_score: float = 0.0
	objective_met: bool = False
	total_rounds: int = 0
	total_cost_usd: float = 0.0
	wall_time_seconds: float = 0.0
	stopped_reason: str = ""
	round_scores: list[float] = field(default_factory=list)


@dataclass
class RoundResult:
	"""Summary of a single round."""

	round_id: str = ""
	number: int = 0
	score: float = 0.0
	objective_met: bool = False
	total_units: int = 0
	completed_units: int = 0
	failed_units: int = 0
	discoveries: list[str] = field(default_factory=list)
	cost_usd: float = 0.0


class RoundController:
	"""Outer loop: plan -> execute -> merge -> fixup -> evaluate, repeat."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db
		self.running = True
		self._backend: WorkerBackend | None = None
		self._green_branch: GreenBranchManager | None = None
		self._planner: RecursivePlanner | None = None

	async def run(self) -> MissionResult:
		"""Run the mission loop until objective met or stopping condition."""
		result = MissionResult(objective=self.config.target.objective)
		start_time = time.monotonic()

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
			# Initialize components
			await self._init_components()

			# Main round loop
			round_number = 0
			discoveries: list[str] = []

			while self.running:
				round_number += 1
				mission.total_rounds = round_number

				# Expire stale signals (unacknowledged >10 min)
				try:
					self.db.expire_stale_signals(timeout_minutes=10)
				except Exception:
					pass

				reason = self._should_stop(mission, result.round_scores)
				if reason:
					result.stopped_reason = reason
					break

				logger.info("Starting round %d", round_number)
				round_result = await self._run_round(
					mission, round_number, discoveries,
				)

				result.round_scores.append(round_result.score)
				result.total_rounds = round_number
				discoveries.extend(round_result.discoveries)

				# Update mission
				mission.total_rounds = round_number
				mission.final_score = round_result.score
				try:
					self.db.update_mission(mission)
				except Exception as exc:
					logger.error("Failed to update mission after round %d: %s", round_number, exc, exc_info=True)

				if round_result.objective_met:
					result.stopped_reason = "objective_met"
					result.objective_met = True
					result.final_score = round_result.score
					break

				# Cooldown between rounds
				cooldown = self.config.rounds.cooldown_between_rounds
				if cooldown > 0:
					await asyncio.sleep(cooldown)

		except (RuntimeError, OSError) as exc:
			logger.error("Mission infrastructure error: %s", exc, exc_info=True)
			result.stopped_reason = "error"
		except asyncio.CancelledError:
			logger.info("Mission cancelled")
			result.stopped_reason = "cancelled"
		except (ValueError, KeyError, json.JSONDecodeError) as exc:
			logger.error("Mission data error: %s", exc, exc_info=True)
			result.stopped_reason = "error"
		finally:
			mission.status = "completed" if result.objective_met else "stopped"
			mission.finished_at = _now_iso()
			mission.stopped_reason = result.stopped_reason
			mission.final_score = result.round_scores[-1] if result.round_scores else 0.0
			try:
				self.db.update_mission(mission)
			except Exception as exc:
				logger.error("Failed to update mission in finally block: %s", exc, exc_info=True)

			if self._backend:
				await self._backend.cleanup()

			result.wall_time_seconds = time.monotonic() - start_time

		return result

	async def _init_components(self) -> None:
		"""Initialize backend, green branch manager, and planner."""
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
		# Use a dedicated workspace for green branch operations
		if isinstance(self._backend, LocalBackend):
			gb_workspace = await self._backend.provision_workspace(
				"green-branch-mgr", source_repo, self.config.target.branch,
			)
			await self._green_branch.initialize(gb_workspace)
		else:
			raise NotImplementedError(
				"Mission mode requires a local workspace for green branch "
				"merge/fixup operations. SSH backend is not yet supported "
				"in mission mode. Use coordinator mode instead."
			)

		# Planner
		self._planner = RecursivePlanner(self.config, self.db)

	async def _run_round(
		self,
		mission: Mission,
		round_number: int,
		prior_discoveries: list[str],
	) -> RoundResult:
		"""Execute a single plan-execute-evaluate cycle."""
		assert self._green_branch is not None
		assert self._planner is not None
		assert self._backend is not None

		rnd = Round(
			mission_id=mission.id,
			number=round_number,
			status="planning",
		)
		rnd.snapshot_hash = await self._green_branch.get_green_hash()
		try:
			self.db.insert_round(rnd)
		except Exception as exc:
			logger.error("Failed to insert round %d: %s", round_number, exc, exc_info=True)
			raise RuntimeError(f"Database error inserting round: {exc}") from exc

		result = RoundResult(round_id=rnd.id, number=round_number)

		# 0. Load feedback context for planner
		planner_context = get_planner_context(self.db, mission.id)

		# 1. Recursive planning
		curated_discoveries = _curate_discoveries(prior_discoveries, self.config.rounds.max_discovery_chars)
		plan, root_node = await self._planner.plan_round(
			objective=mission.objective,
			snapshot_hash=rnd.snapshot_hash,
			prior_discoveries=curated_discoveries,
			round_number=round_number,
			feedback_context=planner_context,
		)
		plan.round_id = rnd.id
		try:
			self.db.insert_plan(plan)
		except Exception as exc:
			logger.error("Failed to insert plan for round %d: %s", round_number, exc, exc_info=True)
			raise RuntimeError(f"Database error inserting plan: {exc}") from exc
		rnd.plan_id = plan.id

		# Persist plan tree
		try:
			self._persist_plan_tree(root_node, plan)
		except Exception as exc:
			logger.error("Failed to persist plan tree for round %d: %s", round_number, exc, exc_info=True)
			raise RuntimeError(f"Database error persisting plan tree: {exc}") from exc

		# Post-planning: detect file overlaps and inject dependency edges
		try:
			from mission_control.overlap import resolve_file_overlaps
			all_units = self.db.get_work_units_for_plan(plan.id)
			resolve_file_overlaps(all_units)
			for u in all_units:
				self.db.update_work_unit(u)
		except Exception as exc:
			logger.warning("Failed to resolve file overlaps: %s", exc)

		result.total_units = plan.total_units
		rnd.total_units = plan.total_units
		rnd.status = "executing"
		try:
			self.db.update_round(rnd)
		except Exception as exc:
			logger.error("Failed to update round %d to executing: %s", round_number, exc, exc_info=True)

		if plan.total_units == 0:
			rnd.status = "completed"
			rnd.finished_at = _now_iso()
			try:
				self.db.update_round(rnd)
			except Exception as exc:
				logger.error("Failed to update empty round %d to completed: %s", round_number, exc, exc_info=True)
			return result

		# Capture pre-round snapshot for feedback
		try:
			snapshot_before = self.db.get_latest_snapshot()
		except Exception as exc:
			logger.error("Failed to get snapshot before round %d: %s", round_number, exc, exc_info=True)
			snapshot_before = None

		# 2. Execute all leaf work units
		await self._execute_units(plan, rnd)

		# 3. Collect handoffs
		try:
			handoffs = self.db.get_handoffs_for_round(rnd.id)
		except Exception as exc:
			logger.error("Failed to get handoffs for round %d: %s", round_number, exc, exc_info=True)
			handoffs = []
		all_discoveries: list[str] = []
		for h in handoffs:
			if h.discoveries:
				try:
					all_discoveries.extend(json.loads(h.discoveries))
				except json.JSONDecodeError:
					pass

		# 4. Fixup: try to promote mc/working to mc/green
		rnd.status = "evaluating"
		try:
			self.db.update_round(rnd)
		except Exception as exc:
			logger.error("Failed to update round %d to evaluating: %s", round_number, exc, exc_info=True)

		fixup_result = await self._green_branch.run_fixup()
		logger.info(
			"Fixup result: promoted=%s attempts=%d",
			fixup_result.promoted, fixup_result.fixup_attempts,
		)

		# 4b. Auto-push mc/green to main if promoted
		if fixup_result.promoted:
			pushed = await self._green_branch.push_green_to_main()
			if pushed:
				logger.info("Auto-pushed mc/green to origin/%s", self.config.green_branch.push_branch)

		# 5. Evaluate objective deterministically
		try:
			snapshot_after = self.db.get_latest_snapshot()
		except Exception as exc:
			logger.error("Failed to get snapshot after round %d: %s", round_number, exc, exc_info=True)
			snapshot_after = None
		try:
			prev_rounds = self.db.get_rounds_for_mission(mission.id)
		except Exception as exc:
			logger.error("Failed to get previous rounds for mission: %s", exc, exc_info=True)
			prev_rounds = []
		# Get score from previous round (exclude current)
		prev_score = 0.0
		for pr in reversed(prev_rounds):
			if pr.id != rnd.id:
				prev_score = pr.objective_score
				break

		# Count completed/failed units
		try:
			units = self.db.get_work_units_for_plan(plan.id)
			completed_count = sum(1 for u in units if u.status == "completed")
			failed_count = sum(1 for u in units if u.status == "failed")
		except Exception as exc:
			logger.error("Failed to get work units for eval: %s", exc, exc_info=True)
			completed_count = 0
			failed_count = 0

		evaluation = evaluate_objective(
			snapshot_before=snapshot_before,
			snapshot_after=snapshot_after,
			completed_units=completed_count,
			total_units=plan.total_units,
			fixup_promoted=fixup_result.promoted,
			prev_score=prev_score,
		)

		# 6. Record feedback
		rnd.objective_score = evaluation.score
		rnd.objective_met = evaluation.met
		try:
			reward = record_round_outcome(
				db=self.db,
				mission_id=mission.id,
				rnd=rnd,
				plan=plan,
				handoffs=handoffs,
				fixup_result=fixup_result,
				snapshot_before=snapshot_before,
				snapshot_after=snapshot_after,
				prev_score=prev_score,
			)
			logger.info("Round %d reward: %.3f", round_number, reward.reward)
		except Exception:
			logger.exception("Failed to record feedback for round %d", round_number)

		# 7. Finalize round
		max_disc = self.config.rounds.max_discoveries_per_round
		rnd.discoveries = json.dumps(all_discoveries[:max_disc])
		rnd.status = "completed"
		rnd.finished_at = _now_iso()

		rnd.completed_units = completed_count
		rnd.failed_units = failed_count
		try:
			self.db.update_round(rnd)
		except Exception as exc:
			logger.error("Failed to finalize round %d: %s", round_number, exc, exc_info=True)

		result.score = evaluation.score
		result.objective_met = evaluation.met
		result.completed_units = rnd.completed_units
		result.failed_units = rnd.failed_units
		result.discoveries = all_discoveries

		return result

	async def _execute_units(self, plan: Plan, rnd: Round) -> None:
		"""Execute work units respecting depends_on edges (topological order).

		Units with no unmet dependencies are dispatched in parallel (bounded
		by the worker semaphore).  When a unit completes, its dependents are
		checked and dispatched if all their dependencies are satisfied.
		If a dependency fails, all downstream units are cascade-failed.
		Circular dependencies are detected and failed as deadlocks.
		"""
		assert self._backend is not None
		units = self.db.get_work_units_for_plan(plan.id)

		# Set round_id on all units
		for unit in units:
			unit.round_id = rnd.id
			await self.db.locked_call("update_work_unit", unit)

		if not units:
			return

		# Build dependency graph
		unit_map: dict[str, WorkUnit] = {u.id: u for u in units}
		# deps_of[uid] = set of unit IDs that uid depends on
		deps_of: dict[str, set[str]] = {}
		# dependents_of[uid] = set of unit IDs that depend on uid
		dependents_of: dict[str, set[str]] = {u.id: set() for u in units}

		for unit in units:
			dep_ids: set[str] = set()
			if unit.depends_on:
				for dep_id in unit.depends_on.split(","):
					dep_id = dep_id.strip()
					if dep_id and dep_id in unit_map:
						dep_ids.add(dep_id)
			deps_of[unit.id] = dep_ids
			for dep_id in dep_ids:
				dependents_of[dep_id].add(unit.id)

		# Detect circular dependencies
		# If after removing all zero-dep nodes we still have nodes left, there's a cycle
		temp_deps = {uid: set(d) for uid, d in deps_of.items()}
		topo_queue = [uid for uid, d in temp_deps.items() if not d]
		visited = set()
		while topo_queue:
			uid = topo_queue.pop()
			visited.add(uid)
			for dependent_id in dependents_of.get(uid, set()):
				temp_deps[dependent_id].discard(uid)
				if not temp_deps[dependent_id]:
					topo_queue.append(dependent_id)

		cycle_units = set(unit_map.keys()) - visited
		if cycle_units:
			logger.error("Circular dependency detected among units: %s", cycle_units)
			for uid in cycle_units:
				unit = unit_map[uid]
				unit.status = "failed"
				unit.output_summary = "Deadlock: circular dependency detected"
				unit.finished_at = _now_iso()
				await self.db.locked_call("update_work_unit", unit)
			# Remove cycle units from processing
			for uid in cycle_units:
				del unit_map[uid]
				deps_of.pop(uid, None)
				dependents_of.pop(uid, None)

		if not unit_map:
			return

		# Execution state
		num_workers = self.config.scheduler.parallel.num_workers
		semaphore = asyncio.Semaphore(num_workers)
		completed: dict[str, bool] = {}  # uid -> success
		pending: set[str] = set(unit_map.keys())
		running: set[str] = set()
		done_event = asyncio.Event()

		def _is_ready(uid: str) -> bool:
			"""Check if all dependencies of uid are satisfied."""
			for dep_id in deps_of.get(uid, set()):
				if dep_id not in completed:
					return False
			return True

		async def _run_unit(unit: WorkUnit) -> None:
			"""Run a single unit and process completion."""
			try:
				await self._execute_single_unit(unit, rnd, semaphore)
			except BaseException as exc:
				logger.error(
					"Unhandled exception in unit %s: %s",
					unit.id, exc,
					exc_info=(type(exc), exc, exc.__traceback__),
				)

			# Check completion status from DB
			refreshed = self.db.get_work_unit(unit.id)
			success = refreshed is not None and refreshed.status == "completed"
			completed[unit.id] = success
			running.discard(unit.id)
			pending.discard(unit.id)

			if not success:
				# Cascade failure to all downstream units
				await _cascade_failure(unit.id)

			# Signal that state changed so dispatcher can check for newly-ready units
			done_event.set()

		async def _cascade_failure(failed_uid: str) -> None:
			"""Mark all transitive dependents of a failed unit as failed."""
			queue = list(dependents_of.get(failed_uid, set()))
			while queue:
				uid = queue.pop(0)
				if uid in completed or uid in running:
					continue
				if uid not in pending:
					continue
				unit = unit_map.get(uid)
				if unit is None:
					continue
				unit.status = "failed"
				unit.output_summary = f"Dependency failed: {failed_uid}"
				unit.finished_at = _now_iso()
				await self.db.locked_call("update_work_unit", unit)
				completed[uid] = False
				pending.discard(uid)
				# Continue cascade
				queue.extend(dependents_of.get(uid, set()))

		# Main dispatch loop
		active_tasks: set[asyncio.Task[None]] = set()

		while pending or running:
			# Find ready units
			ready = [uid for uid in pending if uid not in running and _is_ready(uid)]

			for uid in ready:
				unit = unit_map[uid]
				running.add(uid)
				task = asyncio.create_task(_run_unit(unit))
				active_tasks.add(task)
				task.add_done_callback(active_tasks.discard)

			if not running and not ready and pending:
				# All pending units are blocked but nothing is running
				# This shouldn't happen after cycle detection, but guard anyway
				logger.error("Deadlock: %d units stuck with unmet deps", len(pending))
				for uid in list(pending):
					unit = unit_map[uid]
					unit.status = "failed"
					unit.output_summary = "Deadlock: unmet dependencies"
					unit.finished_at = _now_iso()
					await self.db.locked_call("update_work_unit", unit)
					pending.discard(uid)
				break

			if running:
				done_event.clear()
				await done_event.wait()

		# Wait for any stragglers
		if active_tasks:
			await asyncio.gather(*active_tasks, return_exceptions=True)

	async def _execute_single_unit(
		self,
		unit: WorkUnit,
		rnd: Round,
		semaphore: asyncio.Semaphore,
	) -> None:
		"""Execute a single work unit with concurrency control."""
		assert self._backend is not None
		assert self._green_branch is not None

		async with semaphore:
			source_repo = str(self.config.target.resolved_path)
			base_branch = self.config.green_branch.green_branch

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
				return

			branch_name = f"mc/unit-{unit.id}"
			unit.branch_name = branch_name
			unit.status = "running"
			unit.started_at = _now_iso()
			await self.db.locked_call("update_work_unit", unit)

			# Build prompt (fresh-start pattern)
			from mission_control.memory import load_context_for_mission_worker
			context = load_context_for_mission_worker(unit, self.config)
			experience_context = get_worker_context(self.db, unit)

			prompt = render_mission_worker_prompt(
				unit=unit,
				config=self.config,
				workspace_path=workspace if "::" not in workspace else workspace.split("::")[0],
				branch_name=branch_name,
				context=context,
				experience_context=experience_context,
			)

			budget = self.config.scheduler.budget.max_per_session_usd
			cmd = [
				"claude", "-p",
				"--output-format", "text",
				"--permission-mode", "bypassPermissions",
				"--model", self.config.scheduler.model,
				"--max-budget-usd", str(budget),
				prompt,
			]

			try:
				# Use per-unit timeout if set, otherwise global
				effective_timeout = unit.timeout or self.config.scheduler.session_timeout
				handle = await self._backend.spawn(
					unit.id, workspace, cmd,
					timeout=effective_timeout,
				)

				# Wait for completion with timeout (multiplied for monitoring slack)
				poll_deadline = int(effective_timeout * self.config.rounds.timeout_multiplier)
				monitor_interval = self.config.scheduler.monitor_interval
				start = time.monotonic()
				while time.monotonic() - start < poll_deadline:
					status = await self._backend.check_status(handle)
					if status != "running":
						break
					# Check signals during execution for responsive stop
					if not self.running:
						await self._backend.kill(handle)
						unit.attempt += 1
						unit.status = "failed"
						unit.output_summary = "Stopped by signal"
						unit.finished_at = _now_iso()
						await self.db.locked_call("update_work_unit", unit)
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
					return

				output = await self._backend.get_output(handle)

				# Parse result
				mc_result = parse_mc_result(output)
				if mc_result:
					unit_status = str(mc_result.get("status", "completed"))
					unit.output_summary = str(mc_result.get("summary", ""))
					commits = mc_result.get("commits", [])
					if isinstance(commits, list) and commits:
						unit.commit_hash = str(commits[0])

					# Create handoff (guard non-list values from malformed MC_RESULT)
					disc = mc_result.get("discoveries", [])
					conc = mc_result.get("concerns", [])
					fc = mc_result.get("files_changed", [])
					handoff = Handoff(
						work_unit_id=unit.id,
						round_id=rnd.id,
						status=unit_status,
						commits=json.dumps(commits if isinstance(commits, list) else []),
						summary=unit.output_summary,
						discoveries=json.dumps(disc if isinstance(disc, list) else []),
						concerns=json.dumps(conc if isinstance(conc, list) else []),
						files_changed=json.dumps(fc if isinstance(fc, list) else []),
					)
					await self.db.locked_call("insert_handoff", handoff)
					unit.handoff_id = handoff.id
				else:
					unit_status = "completed" if status == "completed" else "failed"
					max_chars = self.config.scheduler.output_summary_max_chars
					unit.output_summary = output[-max_chars:] if output else "No output"

				if unit_status == "completed" and unit.commit_hash:
					# Merge to working branch
					merged = await self._green_branch.merge_to_working(
						workspace if "::" not in workspace else workspace.split("::")[0],
						branch_name,
					)
					if merged:
						unit.status = "completed"
					else:
						logger.warning("Merge conflict for unit %s", unit.id)
						unit.status = "failed"
						unit.attempt += 1
						unit.output_summary = "Merge conflict: changes could not be integrated"
				elif unit_status == "completed":
					unit.status = "completed"
					logger.info("Unit %s completed with no commits", unit.id)
				elif unit_status == "blocked":
					unit.status = "blocked"
					logger.info(
						"Unit %s blocked: %s", unit.id, unit.output_summary,
					)
				else:
					unit.attempt += 1
					unit.status = "failed"

				unit.finished_at = _now_iso()
				await self.db.locked_call("update_work_unit", unit)

			except (RuntimeError, OSError) as e:
				logger.error("Infrastructure error executing unit %s: %s", unit.id, e)
				unit.attempt += 1
				unit.status = "failed"
				unit.output_summary = f"Infrastructure error: {e}"
				unit.finished_at = _now_iso()
				await self.db.locked_call("update_work_unit", unit)
			except asyncio.CancelledError:
				logger.info("Unit %s execution cancelled", unit.id)
				unit.attempt += 1
				unit.status = "failed"
				unit.output_summary = "Cancelled"
				unit.finished_at = _now_iso()
				await self.db.locked_call("update_work_unit", unit)
			except (ValueError, KeyError, json.JSONDecodeError) as e:
				logger.error("Data error executing unit %s: %s", unit.id, e)
				unit.attempt += 1
				unit.status = "failed"
				unit.output_summary = f"Data error: {e}"
				unit.finished_at = _now_iso()
				await self.db.locked_call("update_work_unit", unit)
			finally:
				await self._backend.release_workspace(workspace)

	def _persist_plan_tree(self, node: PlanNode, plan: Plan) -> None:
		"""Persist the in-memory plan tree to the database."""
		try:
			self.db.insert_plan_node(node)
		except Exception as exc:
			logger.error("Failed to insert plan node %s: %s", node.id, exc, exc_info=True)
			raise

		# Persist forced leaf units
		if hasattr(node, "_forced_unit"):
			wu = node._forced_unit
			try:
				self.db.insert_work_unit(wu)
			except Exception as exc:
				logger.error("Failed to insert forced work unit %s: %s", wu.id, exc, exc_info=True)
				raise

		# Persist child leaves
		if hasattr(node, "_child_leaves"):
			for leaf, wu in node._child_leaves:
				try:
					self.db.insert_plan_node(leaf)
				except Exception as exc:
					logger.error("Failed to insert child plan node %s: %s", leaf.id, exc, exc_info=True)
					raise
				try:
					self.db.insert_work_unit(wu)
				except Exception as exc:
					logger.error("Failed to insert child work unit %s: %s", wu.id, exc, exc_info=True)
					raise

		# Recurse into subdivided children
		if hasattr(node, "_subdivided_children"):
			for child in node._subdivided_children:
				self._persist_plan_tree(child, plan)

	def _should_stop(self, mission: Mission, scores: list[float]) -> str:
		"""Check stopping conditions. Returns reason string or empty."""
		if not self.running:
			return "user_stopped"

		# Check DB signals (from web UI / MCP)
		signal_reason = self._check_signals(mission.id)
		if signal_reason:
			return signal_reason

		if mission.total_rounds > self.config.rounds.max_rounds:
			return "max_rounds"

		# Stall detection: N rounds with negligible score improvement
		threshold = self.config.rounds.stall_threshold
		if len(scores) >= threshold:
			recent = scores[-threshold:]
			if max(recent) - min(recent) < self.config.rounds.stall_score_epsilon:
				return "stalled"

		return ""

	def _check_signals(self, mission_id: str) -> str:
		"""Check for pending signals from web/MCP. Returns stop reason or empty."""
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
			elif signal.signal_type == "retry_unit":
				self._handle_retry_signal(signal)
			elif signal.signal_type == "adjust":
				self._handle_adjust_signal(signal)
		return ""

	def _handle_retry_signal(self, signal: Signal) -> None:
		"""Reset a failed work unit to pending for retry."""
		try:
			unit_id = signal.payload.strip()
			if unit_id:
				unit = self.db.get_work_unit(unit_id)
				if unit and unit.status == "failed" and unit.attempt < unit.max_attempts:
					unit.status = "pending"
					unit.worker_id = None
					unit.claimed_at = None
					unit.heartbeat_at = None
					unit.started_at = None
					unit.finished_at = None
					self.db.update_work_unit(unit)
					logger.info("Retrying work unit %s (attempt %d)", unit_id, unit.attempt + 1)
			self.db.acknowledge_signal(signal.id)
		except Exception as exc:
			logger.error("Failed to handle retry signal: %s", exc)

	def _handle_adjust_signal(self, signal: Signal) -> None:
		"""Adjust runtime parameters from signal payload (JSON)."""
		try:
			import json as _json
			params = _json.loads(signal.payload) if signal.payload else {}
			if "max_rounds" in params:
				self.config.rounds.max_rounds = int(params["max_rounds"])
				logger.info("Adjusted max_rounds to %d", self.config.rounds.max_rounds)
			if "num_workers" in params:
				self.config.scheduler.parallel.num_workers = int(params["num_workers"])
				logger.info("Adjusted num_workers to %d", self.config.scheduler.parallel.num_workers)
			self.db.acknowledge_signal(signal.id)
		except Exception as exc:
			logger.error("Failed to handle adjust signal: %s", exc)

	def stop(self) -> None:
		self.running = False


def _curate_discoveries(discoveries: list[str], max_chars: int = 4000) -> list[str]:
	"""Curate discoveries to fit within budget."""
	if not discoveries:
		return []
	result: list[str] = []
	total = 0
	for d in discoveries:
		if total + len(d) > max_chars:
			break
		result.append(d)
		total += len(d)
	return result
