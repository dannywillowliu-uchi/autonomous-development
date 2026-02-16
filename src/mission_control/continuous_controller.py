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

from mission_control.backends import LocalBackend, SSHBackend, WorkerBackend
from mission_control.config import ContinuousConfig, MissionConfig
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.event_stream import EventStream
from mission_control.feedback import get_worker_context
from mission_control.green_branch import GreenBranchManager
from mission_control.heartbeat import Heartbeat
from mission_control.models import (
	BacklogItem,
	Epoch,
	Handoff,
	Mission,
	Signal,
	UnitEvent,
	Worker,
	WorkUnit,
	_now_iso,
)
from mission_control.notifier import TelegramNotifier
from mission_control.overlap import _parse_files_hint
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
	backlog_item_ids: list[str] | None = None


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
		self._backlog_item_ids: list[str] = []
		self._state_changelog: list[str] = []
		self._total_dispatched: int = 0
		self._total_merged: int = 0
		self._total_failed: int = 0
		self._event_stream: EventStream | None = None

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
			# Build dependency graph for parallelism estimate
			unit_ids = {u.id for u in units}
			dep_graph: dict[str, list[str]] = {}
			for u in units:
				raw_deps = [d.strip() for d in u.depends_on.split(",") if d.strip()] if u.depends_on else []
				deps = [d for d in raw_deps if d in unit_ids]
				dep_graph[u.id] = deps
			# Compute parallelism levels via topological layering
			remaining = dict(dep_graph)
			levels: list[list[str]] = []
			while remaining:
				layer = [uid for uid, deps in remaining.items() if not deps]
				if not layer:
					layer = list(remaining.keys())[:1]
				levels.append(layer)
				for uid in layer:
					del remaining[uid]
				for deps in remaining.values():
					for uid in layer:
						if uid in deps:
							deps.remove(uid)
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
			print(f"\nEstimated parallelism: {len(levels)} level(s), max {max(len(lv) for lv in levels)} concurrent")
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

		# Generate initial MISSION_STATE.md with objective and empty sections
		try:
			self._update_mission_state(mission)
		except Exception as exc:
			logger.warning("Failed to write initial MISSION_STATE.md: %s", exc)

		try:
			await self._init_components()

			# Load backlog items as objective if discovery is enabled
			if self.config.discovery.enabled:
				backlog_objective = self._load_backlog_objective()
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
			if self._planner and self._backlog_item_ids:
				items = [self.db.get_backlog_item(bid) for bid in self._backlog_item_ids]
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
			if self._backlog_item_ids:
				try:
					handoffs = self.db.get_recent_handoffs(mission.id, limit=50)
					self._update_backlog_on_completion(result.objective_met, handoffs)
				except Exception as exc:
					logger.error(
						"Failed to update backlog on completion: %s", exc, exc_info=True,
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

			if self._backend:
				await self._backend.cleanup()

			result.wall_time_seconds = time.monotonic() - self._start_time
			result.total_units_dispatched = self._total_dispatched
			result.total_units_merged = self._total_merged
			result.total_units_failed = self._total_failed

			if self._backlog_item_ids:
				result.backlog_item_ids = list(self._backlog_item_ids)

			try:
				from mission_control.mission_report import generate_mission_report
				generate_mission_report(result, mission, self.db, self.config)
			except Exception as exc:
				logger.error("Failed to generate mission report: %s", exc, exc_info=True)

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

		return result

	def _load_backlog_objective(self, limit: int = 5) -> str | None:
		"""Load top pending backlog items and compose an objective string.

		Marks selected items as in_progress and stores their IDs for
		post-mission completion tracking.

		Returns the composed objective string, or None if no backlog items found.
		"""
		items = self.db.get_pending_backlog(limit=limit)
		if not items:
			return None

		self._backlog_item_ids = [item.id for item in items]

		# Mark selected items as in_progress
		for item in items:
			item.status = "in_progress"
			item.updated_at = _now_iso()
			self.db.update_backlog_item(item)

		# Compose objective from backlog items
		lines = ["Priority backlog items to address:"]
		for i, item in enumerate(items, 1):
			lines.append(
				f"{i}. [{item.track}] {item.title} "
				f"(backlog_item_id={item.id}, priority={item.priority_score:.1f}): "
				f"{item.description}"
			)

		logger.info(
			"Loaded %d backlog items as mission objective (IDs: %s)",
			len(items),
			", ".join(item.id[:8] for item in items),
		)
		return "\n".join(lines)

	def _update_backlog_on_completion(
		self, objective_met: bool, handoffs: list[Handoff],
	) -> None:
		"""Update backlog items after mission ends.

		If objective_met: mark all targeted items as completed.
		If not: reset to pending, store failure context, increment attempt_count.
		"""
		if not self._backlog_item_ids:
			return

		# Build failure context from handoffs
		failure_reasons: list[str] = []
		for h in handoffs:
			if h.status != "completed":
				try:
					concerns = json.loads(h.concerns) if h.concerns else []
				except (json.JSONDecodeError, TypeError):
					concerns = []
				if concerns:
					failure_reasons.append(concerns[-1][:200])
				elif h.summary:
					failure_reasons.append(h.summary[:200])

		for item_id in self._backlog_item_ids:
			item = self.db.get_backlog_item(item_id)
			if item is None:
				continue

			if objective_met:
				item.status = "completed"
				item.updated_at = _now_iso()
				self.db.update_backlog_item(item)
			else:
				item.status = "pending"
				item.attempt_count += 1
				if failure_reasons:
					item.last_failure_reason = "; ".join(failure_reasons[:3])
				item.updated_at = _now_iso()
				self.db.update_backlog_item(item)

		logger.info(
			"Updated %d backlog items: %s",
			len(self._backlog_item_ids),
			"completed" if objective_met else "reset to pending",
		)

	def _update_backlog_from_completion(
		self,
		unit: WorkUnit,
		merged: bool,
		handoff: Handoff | None,
		mission_id: str,
	) -> None:
		"""Update backlog items based on individual unit completion.

		Searches backlog by title matching against the unit title, then:
		- On successful merge: marks matching item 'completed', sets source_mission_id
		- On failure after max retries: increments attempt_count, sets last_failure_reason
		- On partial completion (retryable): keeps 'in_progress', appends context
		"""
		# Extract keywords from unit title for matching
		title_words = [w for w in unit.title.lower().split() if len(w) > 2]
		if not title_words:
			return

		matching_items = self.db.search_backlog_items(title_words, limit=5)
		if not matching_items:
			return

		# Score matches: count how many title words appear in the backlog item title
		best_item: BacklogItem | None = None
		best_score = 0
		for item in matching_items:
			item_title_lower = item.title.lower()
			score = sum(1 for w in title_words if w in item_title_lower)
			if score > best_score:
				best_score = score
				best_item = item

		if best_item is None or best_score == 0:
			return

		# Extract failure context from handoff
		failure_reason = ""
		context_additions: list[str] = []
		if handoff:
			try:
				concerns = json.loads(handoff.concerns) if handoff.concerns else []
			except (json.JSONDecodeError, TypeError):
				concerns = []
			if concerns:
				failure_reason = "; ".join(str(c)[:200] for c in concerns[:3])

			try:
				discoveries = json.loads(handoff.discoveries) if handoff.discoveries else []
			except (json.JSONDecodeError, TypeError):
				discoveries = []
			if discoveries:
				context_additions.extend(str(d)[:200] for d in discoveries[:3])
			if concerns:
				context_additions.extend(str(c)[:200] for c in concerns[:3])

		if not failure_reason and unit.output_summary:
			failure_reason = unit.output_summary[:300]

		if merged:
			# Successful merge: mark completed
			best_item.status = "completed"
			best_item.source_mission_id = mission_id
			best_item.updated_at = _now_iso()
			self.db.update_backlog_item(best_item)
			logger.info(
				"Backlog item '%s' marked completed (matched unit '%s')",
				best_item.title[:40], unit.title[:40],
			)
		elif unit.attempt >= unit.max_attempts:
			# Failed after max retries
			best_item.attempt_count += 1
			best_item.last_failure_reason = failure_reason or "Max retries exceeded"
			best_item.updated_at = _now_iso()
			self.db.update_backlog_item(best_item)
			logger.info(
				"Backlog item '%s' failure recorded (attempt %d, unit '%s')",
				best_item.title[:40], best_item.attempt_count, unit.title[:40],
			)
		else:
			# Partial completion: keep in_progress, append context
			best_item.status = "in_progress"
			if context_additions:
				separator = "\n\n--- Context from unit " + unit.id[:8] + " ---\n"
				best_item.description += separator + "\n".join(context_additions)
			best_item.updated_at = _now_iso()
			self.db.update_backlog_item(best_item)
			logger.info(
				"Backlog item '%s' updated with partial context (unit '%s')",
				best_item.title[:40], unit.title[:40],
			)

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

		# Heartbeat monitor
		hb = self.config.heartbeat
		self._heartbeat = Heartbeat(
			interval=hb.interval,
			idle_threshold=hb.idle_threshold,
			notifier=self._notifier,
			db=self.db,
			enable_recovery=hb.enable_recovery,
		)

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

		while self.running:
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
				if self._notifier:
					await self._notifier.send(
						"Mission complete: planner returned no more work units.",
					)
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
				logger.info("Waiting for semaphore to dispatch unit %s: %s", unit.id[:12], unit.title[:60])
				await semaphore.acquire()
				logger.info("Semaphore acquired, dispatching unit %s", unit.id[:12])
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
				if self._event_stream:
					self._event_stream.emit(
						"dispatched",
						mission_id=mission.id,
						epoch_id=epoch.id,
						unit_id=unit.id,
						details={"title": unit.title, "files": unit.files_hint},
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

				task.add_done_callback(_on_task_done)

			if cooldown > 0:
				await asyncio.sleep(cooldown)

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
					try:
						self.db.update_mission(mission)
					except Exception as exc:
						logger.error("Failed to update mission cost for unit %s: %s", unit.id, exc)
					continue
				logger.info("Research unit %s completed -- skipping merge", unit.id)
				self._total_merged += 1
				try:
					self.db.insert_unit_event(UnitEvent(
						mission_id=mission.id,
						epoch_id=epoch.id,
						work_unit_id=unit.id,
						event_type="research_completed",
					))
				except Exception as exc:
					logger.warning("Failed to insert research_completed event for unit %s: %s", unit.id, exc)
				if self._event_stream:
					self._event_stream.emit(
						"research_completed",
						mission_id=mission.id,
						epoch_id=epoch.id,
						unit_id=unit.id,
					)

				if handoff and self._planner:
					self._planner.ingest_handoff(handoff)

				timestamp = unit.finished_at or _now_iso()
				summary = (handoff.summary[:80] if handoff and handoff.summary else unit.output_summary[:80])
				self._state_changelog.append(
					f"- {timestamp} | {unit.id[:8]} research completed -- {summary}"
				)

				try:
					self._update_mission_state(mission)
				except Exception as exc:
					logger.warning("Failed to update MISSION_STATE.md: %s", exc)

				mission.total_cost_usd += unit.cost_usd
				try:
					self.db.update_mission(mission)
				except Exception as exc:
					logger.error("Failed to update mission cost for unit %s: %s", unit.id, exc)
				continue

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
						if self._event_stream:
							self._event_stream.emit(
								"merged",
								mission_id=mission.id,
								epoch_id=epoch.id,
								unit_id=unit.id,
								input_tokens=unit.input_tokens,
								output_tokens=unit.output_tokens,
								cost_usd=unit.cost_usd,
							)
					else:
						logger.warning(
							"Unit %s failed merge: %s",
							unit.id, merge_result.failure_output[:200],
						)

						# Log merge_failed event
						try:
							self.db.insert_unit_event(UnitEvent(
								mission_id=mission.id,
								epoch_id=epoch.id,
								work_unit_id=unit.id,
								event_type="merge_failed",
								details=json.dumps({
									"failure_output": merge_result.failure_output[:500],
									"failure_stage": merge_result.failure_stage,
								}),
							))
						except Exception:
							pass
						if self._event_stream:
							self._event_stream.emit(
								"merge_failed",
								mission_id=mission.id,
								epoch_id=epoch.id,
								unit_id=unit.id,
								details={
									"failure_output": merge_result.failure_output[:500],
									"failure_stage": merge_result.failure_stage,
								},
							)

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

						failure_reason = merge_result.failure_output[:300]

						# Check if retryable
						if unit.attempt < unit.max_attempts:
							self._schedule_retry(unit, epoch, mission, failure_reason, cont)
						else:
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
							if self._event_stream:
								self._event_stream.emit(
									"rejected",
									mission_id=mission.id,
									epoch_id=epoch.id,
									unit_id=unit.id,
									details={"failure_output": merge_result.failure_output[:500]},
								)

							# Notify merge conflict via Telegram
							if self._notifier:
								await self._notifier.send_merge_conflict(
									unit.title, merge_result.failure_output[:300],
								)
				except Exception as exc:
					logger.error(
						"merge_unit failed for %s: %s",
						unit.id, exc, exc_info=True,
					)
					# Log merge_failed event for exception path
					try:
						self.db.insert_unit_event(UnitEvent(
							mission_id=mission.id,
							epoch_id=epoch.id,
							work_unit_id=unit.id,
							event_type="merge_failed",
							details=json.dumps({
								"failure_output": str(exc)[:500],
								"failure_stage": "exception",
							}),
						))
					except Exception:
						pass
					if self._event_stream:
						self._event_stream.emit(
							"merge_failed",
							mission_id=mission.id,
							epoch_id=epoch.id,
							unit_id=unit.id,
							details={
								"failure_output": str(exc)[:500],
								"failure_stage": "exception",
							},
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
				self._update_backlog_from_completion(
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
				self._update_mission_state(mission)
			except Exception as exc:
				logger.warning("Failed to update MISSION_STATE.md: %s", exc)

			# Accumulate cost
			mission.total_cost_usd += unit.cost_usd

			try:
				self.db.update_mission(mission)
			except Exception:
				pass

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
			f"{failure_reason[:300]}. Avoid the same mistake."
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
		try:
			self.db.insert_unit_event(UnitEvent(
				mission_id=mission.id,
				epoch_id=epoch.id,
				work_unit_id=unit.id,
				event_type="retry_queued",
				details=json.dumps({"delay": delay, "failure_reason": failure_reason[:300]}),
			))
		except Exception:
			pass
		if self._event_stream:
			self._event_stream.emit(
				"retry_queued",
				mission_id=mission.id,
				epoch_id=epoch.id,
				unit_id=unit.id,
				details={"delay": delay, "failure_reason": failure_reason[:300]},
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
		task.add_done_callback(self._active_tasks.discard)

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
		task = asyncio.create_task(
			self._execute_single_unit(unit, epoch, mission, self._semaphore),
		)
		self._active_tasks.add(task)
		task.add_done_callback(self._active_tasks.discard)

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
		await self.db.locked_call("update_work_unit", unit)
		if worker is not None:
			worker.status = "dead"
			worker.units_failed += 1
			try:
				await self.db.locked_call("update_worker", worker)
			except Exception:
				pass
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
					try:
						await self.db.locked_call("update_worker", worker)
					except Exception:
						pass
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
				try:
					await self.db.locked_call("update_worker", worker)
				except Exception:
					pass

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
				try:
					await self.db.locked_call("update_worker", worker)
				except Exception:
					pass
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
			semaphore.release()

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

	def _update_mission_state(self, mission: Mission) -> None:
		"""Write MISSION_STATE.md in the target repo as a living checklist.

		The planner reads this file to understand what's already been
		completed, avoiding duplicate work and naturally narrowing scope.
		"""
		target_path = self.config.target.resolved_path
		state_path = target_path / "MISSION_STATE.md"

		try:
			handoffs = self.db.get_recent_handoffs(mission.id, limit=50)
		except Exception:
			handoffs = []

		lines = [
			"# Mission State",
			f"Objective: {mission.objective}",
			"",
		]

		completed: list[str] = []
		failed: list[str] = []
		all_files: set[str] = set()

		for h in reversed(handoffs):  # oldest first
			try:
				files = json.loads(h.files_changed) if h.files_changed else []
			except (json.JSONDecodeError, TypeError):
				files = []
			file_str = ", ".join(files[:5]) if files else ""
			for f in files:
				all_files.add(f)

			summary = h.summary[:100] if h.summary else ""

			# Look up work unit for timestamp
			try:
				wu = self.db.get_work_unit(h.work_unit_id)
				timestamp = wu.finished_at if wu and wu.finished_at else ""
			except Exception:
				timestamp = ""

			if h.status == "completed":
				ts_part = f" ({timestamp})" if timestamp else ""
				completed.append(
					f"- [x] {h.work_unit_id[:8]}{ts_part} -- {summary}"
					+ (f" (files: {file_str})" if file_str else ""),
				)
			else:
				try:
					concerns = json.loads(h.concerns) if h.concerns else []
				except (json.JSONDecodeError, TypeError):
					concerns = []
				detail = concerns[-1][:100] if concerns else "unknown"
				ts_part = f" ({timestamp})" if timestamp else ""
				failed.append(f"- [ ] {h.work_unit_id[:8]}{ts_part} -- {detail}")

		if completed:
			lines.append("## Completed")
			lines.extend(completed)
			lines.append("")

		if failed:
			lines.append("## Failed")
			lines.extend(failed)
			lines.append("")

		if all_files:
			lines.append("## Files Modified")
			lines.append(", ".join(sorted(all_files)))
			lines.append("")

		lines.extend([
			"## Remaining",
			"The planner should focus on what hasn't been done yet.",
			"Do NOT re-target files in the 'Files Modified' list unless fixing a failure.",
		])

		if self._state_changelog:
			lines.append("")
			lines.append("## Changelog")
			lines.extend(self._state_changelog)

		try:
			state_path.write_text("\n".join(lines) + "\n")
		except OSError as exc:
			logger.warning("Could not write MISSION_STATE.md: %s", exc)

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
					new_sem = asyncio.Semaphore(new_count)
					in_flight = sum(
						1 for t in self._active_tasks if not t.done()
					)
					for _ in range(in_flight):
						new_sem._value = max(new_sem._value - 1, 0)
					self._semaphore = new_sem
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

	def stop(self) -> None:
		self.running = False
