"""Worker agent -- claim tasks, spawn Claude sessions, push results."""

from __future__ import annotations

import asyncio
import json
import logging
import time

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


def _build_handoff(mc_result: dict[str, object], work_unit_id: str, round_id: str) -> Handoff:
	"""Build a Handoff from parsed MC_RESULT data."""
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
{experience_block}{mission_state_block}{overlap_warnings_block}
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
	experience_context: str = "",
	mission_state: str = "",
	overlap_warnings: str = "",
) -> str:
	"""Render constraint-based prompt for mission mode workers."""
	verify_cmd = unit.verification_command or config.target.verification.command
	exp_block = ""
	if experience_context:
		exp_block = f"\n## Relevant Past Experiences\n{experience_context}\n"
	ms_block = ""
	if mission_state:
		ms_block = f"\n## Mission State\n{mission_state}\n"
	ow_block = ""
	if overlap_warnings:
		ow_block = f"\n## File Locking Warnings\n{overlap_warnings}\n"
	return MISSION_WORKER_PROMPT_TEMPLATE.format(
		target_name=config.target.name,
		workspace_path=workspace_path,
		title=unit.title,
		description=unit.description,
		files_hint=unit.files_hint or "Not specified",
		verification_command=verify_cmd,
		context_block=context or "No additional context.",
		experience_block=exp_block,
		mission_state_block=ms_block,
		overlap_warnings_block=ow_block,
	)


def parse_handoff(output: str, work_unit_id: str, round_id: str) -> Handoff | None:
	"""Parse MC_RESULT from output and build a Handoff model.

	Returns None if no MC_RESULT found in the output.
	"""
	mc_result = parse_mc_result(output)
	if mc_result is None:
		return None
	return _build_handoff(mc_result, work_unit_id, round_id)


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
			unit = await self.db.locked_call("claim_work_unit", self.worker.id)
			if unit is None:
				await asyncio.sleep(self.config.scheduler.polling_interval)
				continue

			self.worker.status = "working"
			self.worker.current_unit_id = unit.id
			await self.db.locked_call("update_worker", self.worker)

			try:
				await self._execute_unit(unit)
			except Exception:
				logger.exception("Unexpected error executing unit %s, worker %s survives", unit.id, self.worker.id)
				try:
					await self._mark_unit_failed(unit)
				except Exception:
					logger.exception("Failed to mark unit %s as failed after unexpected error", unit.id)
			finally:
				self.worker.status = "idle"
				self.worker.current_unit_id = None
				await self.db.locked_call("update_worker", self.worker)

	async def _mark_unit_failed(self, unit: WorkUnit) -> None:
		"""Mark a unit as failed, resetting to pending if retries remain."""
		unit.attempt += 1
		if unit.attempt < unit.max_attempts:
			unit.status = "pending"
			unit.claimed_at = None
			unit.heartbeat_at = None
			unit.started_at = None
			unit.finished_at = None
			logger.info(
				"Unit %s failed (attempt %d/%d), resetting to pending",
				unit.id, unit.attempt, unit.max_attempts,
			)
		else:
			unit.status = "failed"
			unit.finished_at = _now_iso()
			logger.warning(
				"Unit %s permanently failed after %d attempts",
				unit.id, unit.attempt,
			)
		await self.db.locked_call("update_work_unit", unit)
		self.worker.units_failed += 1

	async def _cleanup_merged_branches(self, workspace_path: str) -> None:
		"""Delete feature branches for MRs that have been processed by the merge queue.

		Called before starting a new unit to prevent stale branches from accumulating.
		Only deletes branches for MRs in terminal states (merged/rejected/conflict),
		which means the merge queue has already fetched them.
		"""
		processed_mrs: list[MergeRequest] = await self.db.locked_call(
			"get_processed_merge_requests_for_worker", self.worker.id,
		)
		for mr in processed_mrs:
			if mr.branch_name:
				if not await self._run_git("branch", "-D", mr.branch_name, cwd=workspace_path):
					logger.debug("Branch %s already cleaned up in %s", mr.branch_name, workspace_path)

	async def _execute_unit(self, unit: WorkUnit) -> None:
		"""Execute a single work unit via backend: provision, spawn, collect."""
		branch_name = f"mc/unit-{unit.id}"
		unit.branch_name = branch_name
		unit.status = "running"
		unit.started_at = _now_iso()
		await self.db.locked_call("update_work_unit", unit)

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
				await self.db.locked_call("update_worker", self.worker)

			# Clean up branches from previously processed MRs (deferred cleanup)
			await self._cleanup_merged_branches(workspace_path)

			# Create branch in workspace (try -b, fallback to -B for retry)
			if not await self._run_git("checkout", "-b", branch_name, cwd=workspace_path):
				if not await self._run_git("checkout", "-B", branch_name, cwd=workspace_path):
					unit.output_summary = f"Failed to create branch {branch_name}"
					await self._mark_unit_failed(unit)
					return

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

			# Wait for completion (drain stdout to prevent pipe buffer deadlock)
			try:
				deadline = time.monotonic() + effective_timeout
				while True:
					status = await self.backend.check_status(handle)
					if status != "running":
						break
					if time.monotonic() > deadline:
						await self.backend.kill(handle)
						unit.output_summary = f"Timed out after {effective_timeout}s"
						await self._mark_unit_failed(unit)
						await self._reset_workspace(workspace_path, branch_name)
						return
					await self.backend.get_output(handle)
					await asyncio.sleep(self.config.scheduler.polling_interval)

				output = await self.backend.get_output(handle)
				unit.exit_code = 0 if status == "completed" else 1

			except Exception as exc:
				logger.error("Backend error for unit %s: %s", unit.id, exc)
				unit.output_summary = f"Backend error: {exc}"
				await self._mark_unit_failed(unit)
				await self._reset_workspace(workspace_path, branch_name)
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
				handoff = _build_handoff(mc_result, unit.id, round_id=unit.round_id or "")
				await self.db.locked_call("insert_handoff", handoff)
				unit.handoff_id = handoff.id
			else:
				result_status = "completed" if unit.exit_code == 0 else "failed"
				max_chars = self.config.scheduler.output_summary_max_chars
				unit.output_summary = output[-max_chars:]

			if result_status == "completed" and unit.commit_hash:
				unit.status = "completed"
				unit.finished_at = _now_iso()
				await self.db.locked_call("update_work_unit", unit)

				# Submit merge request (atomic position assignment)
				mr = MergeRequest(
					work_unit_id=unit.id,
					worker_id=self.worker.id,
					branch_name=branch_name,
					commit_hash=unit.commit_hash or "",
				)
				await self.db.locked_call("insert_merge_request_atomic", mr)
				self.worker.units_completed += 1

				# Checkout base branch but keep feature branch alive for merge queue fetch
				base = self.config.target.branch
				if not await self._run_git("checkout", base, cwd=workspace_path):
					logger.warning("Failed to checkout %s in %s", base, workspace_path)
			elif result_status == "completed":
				# No-op: worker completed but produced no commits (task already done)
				logger.info("Unit %s completed with no commits (no-op)", unit.id)
				unit.status = "completed"
				unit.finished_at = _now_iso()
				await self.db.locked_call("update_work_unit", unit)
				self.worker.units_completed += 1

				# Clean up: checkout base and delete feature branch (nothing to merge)
				if not await self._run_git("checkout", self.config.target.branch, cwd=workspace_path):
					logger.warning("Failed to checkout %s in %s", self.config.target.branch, workspace_path)
				if not await self._run_git("branch", "-D", branch_name, cwd=workspace_path):
					logger.warning("Failed to delete branch %s in %s", branch_name, workspace_path)
			else:
				await self._mark_unit_failed(unit)
				await self._reset_workspace(workspace_path, branch_name)

		finally:
			self._current_handle = None
			if self._heartbeat_task:
				self._heartbeat_task.cancel()
				try:
					await self._heartbeat_task
				except asyncio.CancelledError:
					pass
				self._heartbeat_task = None

	async def _reset_workspace(self, workspace_path: str, branch_name: str) -> None:
		"""Reset workspace to clean base branch state after a failure.

		If the reset fails, clear workspace_path to force re-provisioning.
		"""
		base = self.config.target.branch
		# Hard reset to discard any partial changes
		if not await self._run_git("reset", "--hard", cwd=workspace_path):
			logger.warning("Workspace %s corrupted, clearing for re-provision", workspace_path)
			self.worker.workspace_path = ""
			await self.db.locked_call("update_worker", self.worker)
			return
		# Checkout base branch
		if not await self._run_git("checkout", base, cwd=workspace_path):
			logger.warning("Workspace %s corrupted, clearing for re-provision", workspace_path)
			self.worker.workspace_path = ""
			await self.db.locked_call("update_worker", self.worker)
			return
		# Clean untracked files
		await self._run_git("clean", "-fdx", cwd=workspace_path)
		# Delete feature branch
		await self._run_git("branch", "-D", branch_name, cwd=workspace_path)

	async def _heartbeat_loop(self) -> None:
		"""Periodically update heartbeat in the DB."""
		while True:
			await asyncio.sleep(self.heartbeat_interval)
			await self.db.locked_call("update_heartbeat", self.worker.id)

	async def _run_git(self, *args: str, cwd: str) -> bool:
		"""Run a git command, logging output on failure."""
		proc = await asyncio.create_subprocess_exec(
			"git", *args,
			cwd=cwd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await proc.communicate()
		if proc.returncode != 0:
			output = stdout.decode(errors="replace") if stdout else ""
			logger.warning("git %s failed in %s (rc=%d): %s", " ".join(args), cwd, proc.returncode, output)
		return proc.returncode == 0

	def stop(self) -> None:
		self.running = False
