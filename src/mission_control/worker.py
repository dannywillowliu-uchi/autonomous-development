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
from mission_control.session import parse_mc_result, validate_mc_result

logger = logging.getLogger(__name__)


def _sanitize_braces(s: str) -> str:
	"""Escape literal braces so str.format() won't interpret them."""
	return s.replace("{", "{{").replace("}", "}}")


VALID_SPECIALISTS = {"test-writer", "refactorer", "debugger", "simplifier"}


def load_specialist_template(specialist: str, config: MissionConfig) -> str:
	"""Load a specialist template from the specialist_templates directory.

	Returns the template content, or empty string if not found or invalid.
	"""
	if not specialist or specialist not in VALID_SPECIALISTS:
		return ""
	templates_dir = config.target.resolved_path / "specialist_templates"
	template_path = templates_dir / f"{specialist}.md"
	if not template_path.is_file():
		logger.debug("Specialist template not found: %s", template_path)
		return ""
	try:
		return template_path.read_text().strip()
	except OSError as exc:
		logger.warning("Failed to read specialist template %s: %s", template_path, exc)
		return ""


class _SpawnError(Exception):
	"""Raised by _spawn_and_wait on timeout."""

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
		title=_sanitize_braces(unit.title),
		description=_sanitize_braces(unit.description),
		files_hint=_sanitize_braces(unit.files_hint or "Not specified"),
		test_passed=test_passed,
		test_total=test_total,
		lint_errors=lint_errors,
		type_errors=type_errors,
		branch_name=branch_name,
		verification_hint=_sanitize_braces(unit.verification_hint or "Run full verification suite"),
		context_block=_sanitize_braces(context or "No additional context."),
		verification_command=_sanitize_braces(verify_cmd),
	)


def _build_handoff(mc_result: dict[str, object], work_unit_id: str, round_id: str) -> Handoff:
	"""Build a Handoff from parsed MC_RESULT data.

	Validates mc_result against MCResultSchema. On validation error,
	extracts valid fields with degraded defaults via validate_mc_result().
	"""
	validated = validate_mc_result(mc_result)
	commits = validated.get("commits", [])
	discoveries = validated.get("discoveries", [])
	concerns = validated.get("concerns", [])
	files_changed = validated.get("files_changed", [])

	return Handoff(
		work_unit_id=work_unit_id,
		round_id=round_id,
		status=str(validated.get("status", "completed")),
		commits=json.dumps(commits) if isinstance(commits, list) else "[]",
		summary=str(validated.get("summary", "")),
		discoveries=json.dumps(discoveries) if isinstance(discoveries, list) else "[]",
		concerns=json.dumps(concerns) if isinstance(concerns, list) else "[]",
		files_changed=json.dumps(files_changed) if isinstance(files_changed, list) else "[]",
	)


RESEARCH_WORKER_PROMPT_TEMPLATE = """\
You are a research agent for {target_name} at {workspace_path}.

## Research Task
{title}

{description}

## Guidelines
- Focus on EXPLORATION and DISCOVERY, not code changes
- Read files, run tests, analyze patterns, gather information
- Do NOT commit code changes -- your output is the research itself
- Document findings thoroughly in your MC_RESULT discoveries field
- Note any risks, dependencies, or blockers you find in concerns

## Scope
Files to investigate: {files_hint}

## Verification
Run (read-only check): {verification_command}

## Context
{context_block}
{experience_block}{mission_state_block}{overlap_warnings_block}
## Output
When done, write a summary as the LAST line of output:
MC_RESULT:{{"status":"completed|failed|blocked","commits":[],\
"summary":"what you found","discoveries":["key findings"],\
"concerns":["potential issues"],"files_changed":[]}}
"""


EXPERIMENT_WORKER_PROMPT_TEMPLATE = """\
You are an experiment agent for {target_name} at {workspace_path}.

## Experiment Task
{title}

{description}

## Guidelines
- Try each approach described above (default: 2 approaches)
- For EACH approach: implement it, benchmark or evaluate it, record the results
- Do NOT commit code changes -- experiments are informational only
- After testing all approaches, produce a JSON comparison report in your MC_RESULT
- The comparison report should include: approach names, metrics, pros/cons, and a recommendation
- Clean up any experimental code changes before finishing (git checkout -- .)

## Scope
Files to investigate: {files_hint}

## Verification
Run (read-only check): {verification_command}

## Context
{context_block}
{experience_block}{mission_state_block}{overlap_warnings_block}
## Output
When done, write a summary as the LAST line of output:
MC_RESULT:{{"status":"completed|failed|blocked","commits":[],\
"summary":"experiment results summary","discoveries":["key findings"],\
"concerns":["potential issues"],"files_changed":[],\
"comparison_report":{{"approaches":[{{"name":"approach_1","description":"...","metrics":{{}},"pros":[],"cons":[]}}],\
"recommended_approach":"approach_1","rationale":"why this approach is better"}}}}
"""


MISSION_WORKER_PROMPT_TEMPLATE = """\
You are working on {target_name} at {workspace_path}.

## Task
{title}

{description}
{acceptance_criteria_block}
## Constraints
- ONLY modify files in scope: {files_hint}
- No TODOs, no partial implementations
- No modifications to unrelated files
- No refactoring beyond the task scope
- Add tests to existing test files (tests/test_<module>.py), not new files.
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

ARCHITECT_PROMPT_TEMPLATE = """\
You are an architect analyzing {target_name} at {workspace_path}.

## Task
{title}

{description}

## Scope
Files likely involved: {files_hint}

## Verification
Run (read-only check): {verification_command}

## Context
{context_block}
{experience_block}{mission_state_block}{overlap_warnings_block}
## Instructions
Analyze the codebase and describe exactly what changes are needed, which files to modify, and why.
Do NOT write code. Do NOT modify any files. Do NOT commit anything.

For each change, specify:
1. The file path
2. What section/function to modify
3. What the change should accomplish
4. Any edge cases or interactions to consider

## Output
When done, write your analysis as the LAST line of output:
MC_RESULT:{{"status":"completed","commits":[],"summary":"architectural analysis",\
"discoveries":["key findings"],"concerns":["potential issues"],"files_changed":[]}}
"""

EDITOR_PROMPT_TEMPLATE = """\
You are working on {target_name} at {workspace_path}.

## Task
{title}

{description}

## Architect Analysis
The following analysis describes exactly what changes to make:

{architect_output}

## Constraints
- ONLY modify files in scope: {files_hint}
- Follow the architect's analysis above precisely
- No TODOs, no partial implementations
- No modifications to unrelated files
- No refactoring beyond the task scope
- Add tests to existing test files (tests/test_<module>.py), not new files.
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


def _build_context_blocks(
	experience_context: str = "",
	mission_state: str = "",
	overlap_warnings: str = "",
) -> tuple[str, str, str]:
	"""Build optional context block strings."""
	exp_block = ""
	if experience_context:
		exp_block = f"\n## Relevant Past Experiences\n{_sanitize_braces(experience_context)}\n"
	ms_block = ""
	if mission_state:
		ms_block = f"\n## Mission State\n{_sanitize_braces(mission_state)}\n"
	ow_block = ""
	if overlap_warnings:
		ow_block = f"\n## File Locking Warnings\n{_sanitize_braces(overlap_warnings)}\n"
	return exp_block, ms_block, ow_block


def render_mission_worker_prompt(
	unit: WorkUnit,
	config: MissionConfig,
	workspace_path: str,
	branch_name: str,
	context: str = "",
	experience_context: str = "",
	mission_state: str = "",
	overlap_warnings: str = "",
	specialist_template: str = "",
) -> str:
	"""Render constraint-based prompt for mission mode workers."""
	verify_cmd = unit.verification_command or config.target.verification.command
	exp_block, ms_block, ow_block = _build_context_blocks(experience_context, mission_state, overlap_warnings)
	ac_block = ""
	if unit.acceptance_criteria:
		ac_block = f"\n## Acceptance Criteria\n{_sanitize_braces(unit.acceptance_criteria)}\n"
	if unit.unit_type == "research":
		template = RESEARCH_WORKER_PROMPT_TEMPLATE
	elif unit.unit_type == "experiment":
		template = EXPERIMENT_WORKER_PROMPT_TEMPLATE
	else:
		template = MISSION_WORKER_PROMPT_TEMPLATE
	rendered = template.format(
		target_name=config.target.name,
		workspace_path=workspace_path,
		title=_sanitize_braces(unit.title),
		description=_sanitize_braces(unit.description),
		files_hint=_sanitize_braces(unit.files_hint or "Not specified"),
		verification_command=_sanitize_braces(verify_cmd),
		context_block=_sanitize_braces(context or "No additional context."),
		experience_block=exp_block,
		mission_state_block=ms_block,
		overlap_warnings_block=ow_block,
		acceptance_criteria_block=ac_block,
	)
	if specialist_template:
		specialist_section = f"## Specialist Role\n{specialist_template}\n\n"
		rendered = specialist_section + rendered
	return rendered


def render_architect_prompt(
	unit: WorkUnit,
	config: MissionConfig,
	workspace_path: str,
	context: str = "",
	experience_context: str = "",
	mission_state: str = "",
	overlap_warnings: str = "",
) -> str:
	"""Render the architect (analysis-only) prompt for two-pass mode."""
	verify_cmd = unit.verification_command or config.target.verification.command
	exp_block, ms_block, ow_block = _build_context_blocks(experience_context, mission_state, overlap_warnings)
	return ARCHITECT_PROMPT_TEMPLATE.format(
		target_name=config.target.name,
		workspace_path=workspace_path,
		title=_sanitize_braces(unit.title),
		description=_sanitize_braces(unit.description),
		files_hint=_sanitize_braces(unit.files_hint or "Not specified"),
		verification_command=_sanitize_braces(verify_cmd),
		context_block=_sanitize_braces(context or "No additional context."),
		experience_block=exp_block,
		mission_state_block=ms_block,
		overlap_warnings_block=ow_block,
	)


def render_editor_prompt(
	unit: WorkUnit,
	config: MissionConfig,
	workspace_path: str,
	architect_output: str,
	context: str = "",
	experience_context: str = "",
	mission_state: str = "",
	overlap_warnings: str = "",
) -> str:
	"""Render the editor (implementation) prompt for two-pass mode."""
	verify_cmd = unit.verification_command or config.target.verification.command
	exp_block, ms_block, ow_block = _build_context_blocks(experience_context, mission_state, overlap_warnings)
	return EDITOR_PROMPT_TEMPLATE.format(
		target_name=config.target.name,
		workspace_path=workspace_path,
		title=_sanitize_braces(unit.title),
		description=_sanitize_braces(unit.description),
		architect_output=_sanitize_braces(architect_output),
		files_hint=_sanitize_braces(unit.files_hint or "Not specified"),
		verification_command=_sanitize_braces(verify_cmd),
		context_block=_sanitize_braces(context or "No additional context."),
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

	async def _spawn_and_wait(
		self, prompt: str, workspace_path: str, effective_timeout: int,
	) -> tuple[str, int]:
		"""Spawn a Claude session and wait for completion.

		Returns (output, exit_code) where exit_code is 0 for success, 1 for failure.
		Raises _SpawnError on timeout or backend error.
		"""
		budget = self.config.scheduler.budget.max_per_session_usd
		models_cfg = getattr(self.config, "models", None)
		model = getattr(models_cfg, "worker_model", None) or self.config.scheduler.model
		cmd = [
			"claude",
			"-p",
			"--output-format", "stream-json",
			"--permission-mode", "bypassPermissions",
			"--model", model,
			"--max-budget-usd", str(budget),
			prompt,
		]

		handle = await self.backend.spawn(
			worker_id=self.worker.id,
			workspace_path=workspace_path,
			command=cmd,
			timeout=effective_timeout,
		)
		self._current_handle = handle

		deadline = time.monotonic() + effective_timeout
		while True:
			status = await self.backend.check_status(handle)
			if status != "running":
				break
			if time.monotonic() > deadline:
				await self.backend.kill(handle)
				raise _SpawnError(f"Timed out after {effective_timeout}s")
			await self.backend.get_output(handle)
			await asyncio.sleep(self.config.scheduler.polling_interval)

		output = await self.backend.get_output(handle)
		exit_code = 0 if status == "completed" else 1
		return output, exit_code

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

			effective_timeout = unit.timeout or self.config.scheduler.session_timeout
			models_cfg = getattr(self.config, "models", None)
			architect_editor = getattr(models_cfg, "architect_editor_mode", False)
			# Research/experiment units always use single-pass
			use_two_pass = architect_editor and unit.unit_type not in ("research", "experiment")

			if use_two_pass:
				# Two-pass: architect (analysis) then editor (implementation)
				architect_prompt = render_architect_prompt(
					unit=unit, config=self.config, workspace_path=workspace_path,
				)
				architect_succeeded = False
				architect_output = ""
				try:
					architect_output, architect_exit = await self._spawn_and_wait(
						architect_prompt, workspace_path, effective_timeout,
					)
					architect_succeeded = architect_exit == 0
				except _SpawnError:
					logger.warning("Architect pass timed out for unit %s, falling back to single-pass", unit.id)
				except Exception as exc:
					logger.warning("Architect pass failed for unit %s: %s, falling back to single-pass", unit.id, exc)

				if not architect_succeeded:
					logger.info("Falling back to single-pass for unit %s", unit.id)

				if architect_succeeded:
					logger.info("Architect pass completed for unit %s, starting editor pass", unit.id)
					editor_prompt = render_editor_prompt(
						unit=unit, config=self.config, workspace_path=workspace_path,
						architect_output=architect_output,
					)
					try:
						output, exit_code = await self._spawn_and_wait(
							editor_prompt, workspace_path, effective_timeout,
						)
					except _SpawnError as exc:
						unit.output_summary = str(exc)
						await self._mark_unit_failed(unit)
						await self._reset_workspace(workspace_path, branch_name)
						return
					except Exception as exc:
						logger.error("Backend error for unit %s editor pass: %s", unit.id, exc)
						unit.output_summary = f"Backend error: {exc}"
						await self._mark_unit_failed(unit)
						await self._reset_workspace(workspace_path, branch_name)
						return
					unit.exit_code = exit_code
				else:
					# Fallback: single-pass after architect failure
					prompt = render_mission_worker_prompt(
						unit=unit,
						config=self.config,
						workspace_path=workspace_path,
						branch_name=branch_name,
					)
					try:
						output, exit_code = await self._spawn_and_wait(
							prompt, workspace_path, effective_timeout,
						)
					except _SpawnError as exc:
						unit.output_summary = str(exc)
						await self._mark_unit_failed(unit)
						await self._reset_workspace(workspace_path, branch_name)
						return
					except Exception as exc:
						logger.error("Backend error for unit %s fallback pass: %s", unit.id, exc)
						unit.output_summary = f"Backend error: {exc}"
						await self._mark_unit_failed(unit)
						await self._reset_workspace(workspace_path, branch_name)
						return
					unit.exit_code = exit_code
			else:
				# Single-pass (default): build prompt and spawn
				prompt = render_worker_prompt(
					unit=unit,
					config=self.config,
					workspace_path=workspace_path,
					branch_name=branch_name,
				)
				try:
					output, exit_code = await self._spawn_and_wait(
						prompt, workspace_path, effective_timeout,
					)
				except _SpawnError as exc:
					unit.output_summary = str(exc)
					await self._mark_unit_failed(unit)
					await self._reset_workspace(workspace_path, branch_name)
					return
				except Exception as exc:
					logger.error("Backend error for unit %s: %s", unit.id, exc)
					unit.output_summary = f"Backend error: {exc}"
					await self._mark_unit_failed(unit)
					await self._reset_workspace(workspace_path, branch_name)
					return
				unit.exit_code = exit_code

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
