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
from mission_control.green_branch import FixupResult, GreenBranchManager
from mission_control.models import (
	Handoff,
	Mission,
	Plan,
	PlanNode,
	Round,
	WorkUnit,
	_now_iso,
)
from mission_control.recursive_planner import RecursivePlanner
from mission_control.session import parse_mc_result
from mission_control.worker import render_mission_worker_prompt

logger = logging.getLogger(__name__)

MONITOR_INTERVAL = 5
MAX_DISCOVERY_CHARS = 4000


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
		self.db.insert_mission(mission)
		result.mission_id = mission.id

		try:
			# Initialize components
			await self._init_components()

			# Main round loop
			round_number = 0
			discoveries: list[str] = []

			while self.running:
				round_number += 1
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
				discoveries = round_result.discoveries

				# Update mission
				mission.total_rounds = round_number
				mission.final_score = round_result.score
				self.db.update_mission(mission)

				if round_result.objective_met:
					result.stopped_reason = "objective_met"
					result.objective_met = True
					result.final_score = round_result.score
					break

				# Cooldown between rounds
				cooldown = self.config.rounds.cooldown_between_rounds
				if cooldown > 0:
					await asyncio.sleep(cooldown)

		except Exception:
			logger.exception("Mission error")
			result.stopped_reason = "error"
		finally:
			mission.status = "completed" if result.objective_met else "stopped"
			mission.finished_at = _now_iso()
			mission.stopped_reason = result.stopped_reason
			mission.final_score = result.round_scores[-1] if result.round_scores else 0.0
			self.db.update_mission(mission)

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
		self.db.insert_round(rnd)

		result = RoundResult(round_id=rnd.id, number=round_number)

		# 1. Recursive planning
		curated_discoveries = _curate_discoveries(prior_discoveries)
		plan, root_node = await self._planner.plan_round(
			objective=mission.objective,
			snapshot_hash=rnd.snapshot_hash,
			prior_discoveries=curated_discoveries,
			round_number=round_number,
		)
		plan.round_id = rnd.id
		self.db.insert_plan(plan)
		rnd.plan_id = plan.id

		# Persist plan tree
		self._persist_plan_tree(root_node, plan)

		result.total_units = plan.total_units
		rnd.total_units = plan.total_units
		rnd.status = "executing"
		self.db.update_round(rnd)

		if plan.total_units == 0:
			rnd.status = "completed"
			rnd.finished_at = _now_iso()
			self.db.update_round(rnd)
			return result

		# 2. Execute all leaf work units
		await self._execute_units(plan, rnd)

		# 3. Collect handoffs
		handoffs = self.db.get_handoffs_for_round(rnd.id)
		all_discoveries: list[str] = []
		for h in handoffs:
			if h.discoveries:
				try:
					all_discoveries.extend(json.loads(h.discoveries))
				except json.JSONDecodeError:
					pass

		# 4. Fixup: try to promote mc/working to mc/green
		rnd.status = "evaluating"
		self.db.update_round(rnd)

		fixup_result = await self._green_branch.run_fixup()
		logger.info(
			"Fixup result: promoted=%s attempts=%d",
			fixup_result.promoted, fixup_result.fixup_attempts,
		)

		# 5. Evaluate objective on mc/green
		round_summary = self._build_round_summary(plan, handoffs, fixup_result)
		evaluation = await evaluate_objective(
			config=self.config,
			snapshot_hash=await self._green_branch.get_green_hash(),
			round_summary=round_summary,
			objective=mission.objective,
		)

		# 6. Finalize round
		rnd.objective_score = evaluation.score
		rnd.objective_met = evaluation.met
		rnd.discoveries = json.dumps(all_discoveries[:20])
		rnd.status = "completed"
		rnd.finished_at = _now_iso()

		# Count completed/failed from plan
		units = self.db.get_work_units_for_plan(plan.id)
		rnd.completed_units = sum(1 for u in units if u.status == "completed")
		rnd.failed_units = sum(1 for u in units if u.status == "failed")
		self.db.update_round(rnd)

		result.score = evaluation.score
		result.objective_met = evaluation.met
		result.completed_units = rnd.completed_units
		result.failed_units = rnd.failed_units
		result.discoveries = all_discoveries

		return result

	async def _execute_units(self, plan: Plan, rnd: Round) -> None:
		"""Execute all work units using worker backend."""
		assert self._backend is not None
		units = self.db.get_work_units_for_plan(plan.id)

		# Set round_id on all units
		for unit in units:
			unit.round_id = rnd.id
			self.db.update_work_unit(unit)

		# Spawn workers for each unit
		num_workers = self.config.scheduler.parallel.num_workers
		semaphore = asyncio.Semaphore(num_workers)
		tasks = [self._execute_single_unit(unit, rnd, semaphore) for unit in units]
		await asyncio.gather(*tasks, return_exceptions=True)

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
				unit.status = "failed"
				unit.output_summary = str(e)
				unit.finished_at = _now_iso()
				self.db.update_work_unit(unit)
				return

			branch_name = f"mc/unit-{unit.id}"
			unit.branch_name = branch_name
			unit.status = "running"
			unit.started_at = _now_iso()
			self.db.update_work_unit(unit)

			# Build prompt (fresh-start pattern)
			from mission_control.memory import load_context_for_mission_worker
			context = load_context_for_mission_worker(unit, self.config)

			prompt = render_mission_worker_prompt(
				unit=unit,
				config=self.config,
				workspace_path=workspace if "::" not in workspace else workspace.split("::")[0],
				branch_name=branch_name,
				context=context,
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

			try:
				handle = await self._backend.spawn(
					unit.id, workspace, cmd,
					timeout=self.config.scheduler.session_timeout,
				)

				# Wait for completion with timeout
				timeout = int(self.config.scheduler.session_timeout * 1.2)
				start = time.monotonic()
				while time.monotonic() - start < timeout:
					status = await self._backend.check_status(handle)
					if status != "running":
						break
					await asyncio.sleep(5)
				else:
					await self._backend.kill(handle)
					unit.status = "failed"
					unit.output_summary = f"Timed out after {timeout}s"
					unit.finished_at = _now_iso()
					self.db.update_work_unit(unit)
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

					# Create handoff
					handoff = Handoff(
						work_unit_id=unit.id,
						round_id=rnd.id,
						status=unit_status,
						commits=json.dumps(commits if isinstance(commits, list) else []),
						summary=unit.output_summary,
						discoveries=json.dumps(mc_result.get("discoveries", [])),
						concerns=json.dumps(mc_result.get("concerns", [])),
						files_changed=json.dumps(mc_result.get("files_changed", [])),
					)
					self.db.insert_handoff(handoff)
					unit.handoff_id = handoff.id
				else:
					unit_status = "completed" if status == "completed" else "failed"
					unit.output_summary = output[-500:] if output else "No output"

				if unit_status == "completed" and unit.commit_hash:
					unit.status = "completed"
					# Merge to working branch
					merged = await self._green_branch.merge_to_working(
						workspace if "::" not in workspace else workspace.split("::")[0],
						branch_name,
					)
					if not merged:
						logger.warning("Merge conflict for unit %s", unit.id)
				else:
					unit.status = "failed"

				unit.finished_at = _now_iso()
				self.db.update_work_unit(unit)

			except Exception as e:
				logger.error("Error executing unit %s: %s", unit.id, e)
				unit.status = "failed"
				unit.output_summary = str(e)
				unit.finished_at = _now_iso()
				self.db.update_work_unit(unit)
			finally:
				await self._backend.release_workspace(workspace)

	def _persist_plan_tree(self, node: PlanNode, plan: Plan) -> None:
		"""Persist the in-memory plan tree to the database."""
		self.db.insert_plan_node(node)

		# Persist forced leaf units
		if hasattr(node, "_forced_unit"):
			wu = node._forced_unit
			self.db.insert_work_unit(wu)

		# Persist child leaves
		if hasattr(node, "_child_leaves"):
			for leaf, wu in node._child_leaves:
				self.db.insert_plan_node(leaf)
				self.db.insert_work_unit(wu)

		# Recurse into children (tracked by children_ids)
		# The recursive planner stores children in _child_leaves or via children_ids
		# We need to traverse the in-memory tree structure

	def _should_stop(self, mission: Mission, scores: list[float]) -> str:
		"""Check stopping conditions. Returns reason string or empty."""
		if not self.running:
			return "user_stopped"

		if mission.total_rounds >= self.config.rounds.max_rounds:
			return "max_rounds"

		# Stall detection: N rounds with < 0.01 score improvement
		threshold = self.config.rounds.stall_threshold
		if len(scores) >= threshold:
			recent = scores[-threshold:]
			if max(recent) - min(recent) < 0.01:
				return "stalled"

		return ""

	def _build_round_summary(
		self,
		plan: Plan,
		handoffs: list[Handoff],
		fixup_result: FixupResult,
	) -> str:
		"""Build a summary string for the evaluator."""
		parts = [f"Plan: {plan.total_units} units planned"]

		completed = sum(1 for h in handoffs if h.status == "completed")
		failed = sum(1 for h in handoffs if h.status == "failed")
		parts.append(f"Execution: {completed} completed, {failed} failed")

		summaries = [h.summary for h in handoffs if h.summary]
		if summaries:
			parts.append("Work done:\n" + "\n".join(f"- {s}" for s in summaries[:10]))

		if fixup_result.promoted:
			parts.append(
				f"Fixup: verification passed after {fixup_result.fixup_attempts} attempt(s)"
			)
		else:
			parts.append("Fixup: verification still failing")

		return "\n\n".join(parts)

	def stop(self) -> None:
		self.running = False


def _curate_discoveries(discoveries: list[str], max_chars: int = MAX_DISCOVERY_CHARS) -> list[str]:
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
