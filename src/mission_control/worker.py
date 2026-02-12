"""Worker agent -- claim tasks, spawn Claude sessions, push results."""

from __future__ import annotations

import asyncio
import json
import logging

from mission_control.backends.base import WorkerBackend, WorkerHandle
from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import Handoff, MergeRequest, Worker, WorkUnit, _now_iso
from mission_control.session import parse_mc_result

logger = logging.getLogger(__name__)

WORKER_PROMPT_TEMPLATE = """\
You are a parallel worker agent for {target_name} at {workspace_path}.

## Task
{title}

{description}

## Scope
ONLY modify files related to this task.
Files likely involved: {files_hint}

## Current Project State
- Tests: {test_passed}/{test_total} passing
- Lint errors: {lint_errors}
- Type errors: {type_errors}
- Branch: {branch_name}

## Verification Focus
{verification_hint}

## Context
{context_block}

## Instructions
1. Implement the task described above
2. ONLY modify files listed in the scope (or closely related files)
3. Run verification: {verification_command}
4. If verification passes, commit with a descriptive message
5. If verification fails after 3 attempts, stop and report what went wrong
6. Do NOT modify unrelated files or tests

## Output
When done, write a summary as the LAST line of output:
MC_RESULT:{{"status":"completed|failed|blocked","commits":["hash"],\
"summary":"what you did","files_changed":["list"],\
"discoveries":["things discovered during work"],\
"concerns":["potential issues or risks"]}}
"""


def render_worker_prompt(
	unit: WorkUnit,
	config: MissionConfig,
	workspace_path: str,
	branch_name: str,
	test_passed: int = 0,
	test_total: int = 0,
	lint_errors: int = 0,
	type_errors: int = 0,
	context: str = "",
) -> str:
	"""Render the prompt template for a worker session."""
	verify_cmd = unit.verification_command or config.target.verification.command
	return WORKER_PROMPT_TEMPLATE.format(
		target_name=config.target.name,
		workspace_path=workspace_path,
		title=unit.title,
		description=unit.description,
		files_hint=unit.files_hint or "Not specified",
		test_passed=test_passed,
		test_total=test_total,
		lint_errors=lint_errors,
		type_errors=type_errors,
		branch_name=branch_name,
		verification_hint=unit.verification_hint or "Run full verification suite",
		context_block=context or "No additional context.",
		verification_command=verify_cmd,
	)


def _parse_handoff(mc_result: dict[str, object], unit: WorkUnit, round_id: str) -> Handoff:
	"""Build a Handoff from parsed MC_RESULT data."""
	commits = mc_result.get("commits", [])
	discoveries = mc_result.get("discoveries", [])
	concerns = mc_result.get("concerns", [])
	files_changed = mc_result.get("files_changed", [])

	return Handoff(
		work_unit_id=unit.id,
		round_id=round_id,
		status=str(mc_result.get("status", "completed")),
		commits=json.dumps(commits) if isinstance(commits, list) else "[]",
		summary=str(mc_result.get("summary", "")),
		discoveries=json.dumps(discoveries) if isinstance(discoveries, list) else "[]",
		concerns=json.dumps(concerns) if isinstance(concerns, list) else "[]",
		files_changed=json.dumps(files_changed) if isinstance(files_changed, list) else "[]",
	)


MISSION_WORKER_PROMPT_TEMPLATE = """\
You are working on {target_name} at {workspace_path}.

## Task
{title}

{description}

## Constraints
- ONLY modify files in scope: {files_hint}
- No TODOs, no partial implementations
- No modifications to unrelated files
- No refactoring beyond the task scope
- Commit when done or explain why blocked

## Verification
Run: {verification_command}

## Context
{context_block}

## Output
When done, write a summary as the LAST line of output:
MC_RESULT:{{"status":"completed|failed|blocked","commits":["hash"],\
"summary":"what you did","discoveries":["anything unexpected found"],\
"concerns":["potential issues"],"files_changed":["list"]}}
"""


def render_mission_worker_prompt(
	unit: WorkUnit,
	config: MissionConfig,
	workspace_path: str,
	branch_name: str,
	context: str = "",
) -> str:
	"""Render constraint-based prompt for mission mode workers."""
	verify_cmd = unit.verification_command or config.target.verification.command
	return MISSION_WORKER_PROMPT_TEMPLATE.format(
		target_name=config.target.name,
		workspace_path=workspace_path,
		title=unit.title,
		description=unit.description,
		files_hint=unit.files_hint or "Not specified",
		verification_command=verify_cmd,
		context_block=context or "No additional context.",
	)


def parse_handoff(output: str, work_unit_id: str, round_id: str) -> Handoff | None:
	"""Parse MC_RESULT from output and build a Handoff model.

	Returns None if no MC_RESULT found in the output.
	"""
	mc_result = parse_mc_result(output)
	if mc_result is None:
		return None

	commits = mc_result.get("commits", [])
	discoveries = mc_result.get("discoveries", [])
	concerns = mc_result.get("concerns", [])
	files_changed = mc_result.get("files_changed", [])

	return Handoff(
		work_unit_id=work_unit_id,
		round_id=round_id,
		status=str(mc_result.get("status", "completed")),
		commits=json.dumps(commits) if isinstance(commits, list) else "[]",
		summary=str(mc_result.get("summary", "")),
		discoveries=json.dumps(discoveries) if isinstance(discoveries, list) else "[]",
		concerns=json.dumps(concerns) if isinstance(concerns, list) else "[]",
		files_changed=json.dumps(files_changed) if isinstance(files_changed, list) else "[]",
	)


class WorkerAgent:
	"""A parallel worker that claims tasks, spawns Claude sessions, and pushes results."""

	def __init__(
		self,
		worker: Worker,
		db: Database,
		config: MissionConfig,
		backend: WorkerBackend,
		heartbeat_interval: int = 30,
	) -> None:
		self.worker = worker
		self.db = db
		self.config = config
		self.backend = backend
		self.heartbeat_interval = heartbeat_interval
		self.running = True
		self._heartbeat_task: asyncio.Task[None] | None = None
		self._current_handle: WorkerHandle | None = None

	async def run(self) -> None:
		"""Main loop: claim -> execute -> report, until stopped."""
		while self.running:
			unit = self.db.claim_work_unit(self.worker.id)
			if unit is None:
				await asyncio.sleep(self.config.scheduler.polling_interval)
				continue

			self.worker.status = "working"
			self.worker.current_unit_id = unit.id
			self.db.update_worker(self.worker)

			await self._execute_unit(unit)

			self.worker.status = "idle"
			self.worker.current_unit_id = None
			self.db.update_worker(self.worker)

	async def _execute_unit(self, unit: WorkUnit) -> None:
		"""Execute a single work unit via backend: provision, spawn, collect."""
		branch_name = f"mc/unit-{unit.id}"
		unit.branch_name = branch_name
		unit.status = "running"
		unit.started_at = _now_iso()
		self.db.update_work_unit(unit)

		# Start heartbeat
		self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

		try:
			# Provision workspace via backend
			workspace_path = self.worker.workspace_path
			if not workspace_path:
				workspace_path = await self.backend.provision_workspace(
					worker_id=self.worker.id,
					source_repo=str(self.config.target.resolved_path),
					base_branch=self.config.target.branch,
				)
				self.worker.workspace_path = workspace_path
				self.db.update_worker(self.worker)

			# Create branch in workspace
			await self._run_git("checkout", "-b", branch_name, cwd=workspace_path)

			# Build prompt and command
			prompt = render_worker_prompt(
				unit=unit,
				config=self.config,
				workspace_path=workspace_path,
				branch_name=branch_name,
			)
			budget = self.config.scheduler.budget.max_per_session_usd
			cmd = [
				"claude",
				"-p",
				"--output-format", "stream-json",
				"--permission-mode", "bypassPermissions",
				"--model", self.config.scheduler.model,
				"--max-budget-usd", str(budget),
				prompt,
			]

			# Spawn via backend -- use per-unit timeout if set
			effective_timeout = unit.timeout or self.config.scheduler.session_timeout
			handle = await self.backend.spawn(
				worker_id=self.worker.id,
				workspace_path=workspace_path,
				command=cmd,
				timeout=effective_timeout,
			)
			self._current_handle = handle

			# Wait for completion
			try:
				deadline = asyncio.get_event_loop().time() + effective_timeout
				while True:
					status = await self.backend.check_status(handle)
					if status != "running":
						break
					if asyncio.get_event_loop().time() > deadline:
						await self.backend.kill(handle)
						unit.status = "failed"
						unit.output_summary = f"Timed out after {effective_timeout}s"
						unit.finished_at = _now_iso()
						self.db.update_work_unit(unit)
						self.worker.units_failed += 1
						return
					await asyncio.sleep(self.config.scheduler.polling_interval)

				output = await self.backend.get_output(handle)
				unit.exit_code = 0 if status == "completed" else 1

			except Exception as exc:
				logger.error("Backend error for unit %s: %s", unit.id, exc)
				unit.status = "failed"
				unit.output_summary = f"Backend error: {exc}"
				unit.finished_at = _now_iso()
				self.db.update_work_unit(unit)
				self.worker.units_failed += 1
				return

			# Parse result and build handoff
			mc_result = parse_mc_result(output)
			if mc_result:
				result_status = str(mc_result.get("status", "completed"))
				unit.output_summary = str(mc_result.get("summary", ""))
				commits = mc_result.get("commits", [])
				if isinstance(commits, list) and commits:
					unit.commit_hash = str(commits[0])

				# Create enhanced handoff record
				handoff = _parse_handoff(mc_result, unit, round_id=unit.round_id or "")
				self.db.insert_handoff(handoff)
				unit.handoff_id = handoff.id
			else:
				result_status = "completed" if unit.exit_code == 0 else "failed"
				max_chars = self.config.scheduler.output_summary_max_chars
				unit.output_summary = output[-max_chars:]

			if result_status == "completed" and unit.commit_hash:
				unit.status = "completed"
				unit.finished_at = _now_iso()
				self.db.update_work_unit(unit)

				# Submit merge request
				mr = MergeRequest(
					work_unit_id=unit.id,
					worker_id=self.worker.id,
					branch_name=branch_name,
					commit_hash=unit.commit_hash or "",
					position=self.db.get_next_merge_position(),
				)
				self.db.insert_merge_request(mr)
				self.worker.units_completed += 1
			else:
				unit.status = "failed"
				unit.finished_at = _now_iso()
				unit.attempt += 1
				self.db.update_work_unit(unit)
				self.worker.units_failed += 1

				# Reset workspace to base branch for next task
				await self._run_git("checkout", self.config.target.branch, cwd=workspace_path)
				await self._run_git("branch", "-D", branch_name, cwd=workspace_path)

		finally:
			self._current_handle = None
			if self._heartbeat_task:
				self._heartbeat_task.cancel()
				try:
					await self._heartbeat_task
				except asyncio.CancelledError:
					pass
				self._heartbeat_task = None

	async def _heartbeat_loop(self) -> None:
		"""Periodically update heartbeat in the DB."""
		while True:
			await asyncio.sleep(self.heartbeat_interval)
			self.db.update_heartbeat(self.worker.id)

	async def _run_git(self, *args: str, cwd: str) -> bool:
		"""Run a git command."""
		proc = await asyncio.create_subprocess_exec(
			"git", *args,
			cwd=cwd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		await proc.communicate()
		return proc.returncode == 0

	def stop(self) -> None:
		self.running = False
