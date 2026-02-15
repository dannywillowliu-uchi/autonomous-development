"""Session spawning -- invoke Claude Code as a subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

from pydantic import ValidationError

from mission_control.config import MissionConfig
from mission_control.json_utils import extract_json_from_text
from mission_control.models import MCResultSchema, Session, Snapshot, TaskRecord

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
You are working on {target_name} at {target_path}.

## Task
{task_description}

## Current Project State
- Tests: {test_passed}/{test_total} passing
- Lint errors: {lint_errors}
- Type errors: {type_errors}
- Branch: {branch_name}

## Context
{context_block}

## Instructions
1. Implement the task described above
2. Run verification: {verification_command}
3. If verification passes, commit with a descriptive message
4. If verification fails after 3 attempts, stop and report what went wrong
5. Do NOT modify or delete existing tests

## Output
When done, write a summary as the LAST line of output:
MC_RESULT:{{"status":"completed|failed|blocked","commits":["hash"],"summary":"what you did","files_changed":["list"]}}
"""


def render_prompt(
	task: TaskRecord,
	snapshot: Snapshot,
	config: MissionConfig,
	branch_name: str,
	context: str = "",
) -> str:
	"""Render the prompt template for a Claude Code session."""
	return PROMPT_TEMPLATE.format(
		target_name=config.target.name,
		target_path=config.target.resolved_path,
		task_description=task.description,
		test_passed=snapshot.test_passed,
		test_total=snapshot.test_total,
		lint_errors=snapshot.lint_errors,
		type_errors=snapshot.type_errors,
		branch_name=branch_name,
		context_block=context or "No additional context.",
		verification_command=config.target.verification.command,
	)


def parse_mc_result(output: str) -> dict[str, object] | None:
	"""Extract MC_RESULT JSON from session output.

	Handles both single-line and multiline JSON after the MC_RESULT: marker.
	"""
	# Find the last MC_RESULT: marker in the output
	marker = "MC_RESULT:"
	idx = output.rfind(marker)
	if idx == -1:
		return None

	# Extract everything after the marker
	remainder = output[idx + len(marker):]

	# Try balanced brace extraction (handles multiline JSON)
	result = extract_json_from_text(remainder)
	if isinstance(result, dict):
		return validate_mc_result(result)

	# Fallback: single-line regex for simple cases
	match = re.search(r"\{.*\}", remainder.split("\n")[0])
	if match:
		try:
			raw = json.loads(match.group(0))
			if isinstance(raw, dict):
				return validate_mc_result(raw)
		except json.JSONDecodeError:
			pass

	return None


_MC_RESULT_DEFAULTS: dict[str, object] = {
	"status": "failed",
	"commits": [],
	"summary": "",
	"files_changed": [],
	"discoveries": [],
	"concerns": [],
}


def validate_mc_result(raw: dict[str, object]) -> dict[str, object]:
	"""Validate an MC_RESULT dict against MCResultSchema.

	On success, returns the validated dict. On ValidationError, extracts
	whatever valid fields exist and returns a degraded dict with defaults
	for missing/invalid fields. Logs a warning on degraded parse.
	"""
	try:
		validated = MCResultSchema.model_validate(raw)
		return validated.model_dump()
	except ValidationError as exc:
		logger.warning("MC_RESULT schema validation failed, extracting valid fields: %s", exc)
		degraded: dict[str, object] = {}
		for field_name in MCResultSchema.model_fields:
			value = raw.get(field_name)
			if value is not None:
				try:
					partial = MCResultSchema.model_validate({
						**_MC_RESULT_DEFAULTS,
						field_name: value,
					})
					degraded[field_name] = getattr(partial, field_name)
				except ValidationError:
					degraded[field_name] = _MC_RESULT_DEFAULTS[field_name]
			else:
				degraded[field_name] = _MC_RESULT_DEFAULTS[field_name]
		return degraded


def build_branch_name(session_id: str) -> str:
	"""Generate a git branch name for a session."""
	return f"mc/session-{session_id}"


async def create_branch(branch_name: str, base_branch: str, cwd: str) -> bool:
	"""Create and checkout a new git branch."""
	proc = await asyncio.create_subprocess_exec(
		"git", "checkout", "-b", branch_name, base_branch,
		cwd=cwd,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.STDOUT,
	)
	await proc.communicate()
	return proc.returncode == 0


async def delete_branch(branch_name: str, base_branch: str, cwd: str) -> bool:
	"""Switch to base branch and delete the session branch."""
	checkout = await asyncio.create_subprocess_exec(
		"git", "checkout", base_branch,
		cwd=cwd,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.STDOUT,
	)
	await checkout.communicate()
	if checkout.returncode != 0:
		return False

	delete = await asyncio.create_subprocess_exec(
		"git", "branch", "-D", branch_name,
		cwd=cwd,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.STDOUT,
	)
	await delete.communicate()
	return delete.returncode == 0


async def merge_branch(branch_name: str, base_branch: str, cwd: str) -> bool:
	"""Merge session branch into base branch."""
	checkout = await asyncio.create_subprocess_exec(
		"git", "checkout", base_branch,
		cwd=cwd,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.STDOUT,
	)
	await checkout.communicate()
	if checkout.returncode != 0:
		return False

	merge = await asyncio.create_subprocess_exec(
		"git", "merge", "--no-ff", branch_name, "-m", f"Merge {branch_name}",
		cwd=cwd,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.STDOUT,
	)
	await merge.communicate()
	return merge.returncode == 0


async def spawn_session(
	task: TaskRecord,
	snapshot: Snapshot,
	config: MissionConfig,
	context: str = "",
) -> Session:
	"""Spawn a Claude Code session as a subprocess.

	Creates a git branch, runs Claude Code with the task prompt,
	captures output, and returns a Session with results.
	"""
	session = Session(
		target_name=config.target.name,
		task_description=task.description,
		status="running",
	)
	session.branch_name = build_branch_name(session.id)

	cwd = str(config.target.resolved_path)

	# Create branch
	if config.scheduler.git.strategy == "branch-per-session":
		branch_ok = await create_branch(session.branch_name, config.target.branch, cwd)
		if not branch_ok:
			session.status = "failed"
			session.output_summary = "Failed to create git branch"
			session.finished_at = datetime.now(timezone.utc).isoformat()
			return session

	prompt = render_prompt(task, snapshot, config, session.branch_name, context)
	budget = config.scheduler.budget.max_per_session_usd

	cmd = [
		"claude",
		"-p",
		"--output-format", "stream-json",
		"--permission-mode", "bypassPermissions",
		"--model", config.scheduler.model,
		"--max-budget-usd", str(budget),
		prompt,
	]

	try:
		proc = await asyncio.create_subprocess_exec(
			*cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
			cwd=cwd,
		)
		stdout_bytes, _ = await asyncio.wait_for(
			proc.communicate(),
			timeout=config.scheduler.session_timeout,
		)
		output = stdout_bytes.decode("utf-8", errors="replace")
		session.exit_code = proc.returncode

	except asyncio.TimeoutError:
		try:
			proc.kill()
			await proc.wait()
		except ProcessLookupError:
			pass
		session.status = "failed"
		session.output_summary = f"Session timed out after {config.scheduler.session_timeout}s"
		session.finished_at = datetime.now(timezone.utc).isoformat()
		return session

	# Parse result
	mc_result = parse_mc_result(output)
	if mc_result:
		session.status = str(mc_result.get("status", "completed"))
		session.output_summary = str(mc_result.get("summary", ""))
		commits = mc_result.get("commits", [])
		if isinstance(commits, list) and commits:
			session.commit_hash = str(commits[0])
	else:
		session.status = "completed" if session.exit_code == 0 else "failed"
		max_chars = config.scheduler.output_summary_max_chars
		session.output_summary = output[-max_chars:]

	session.finished_at = datetime.now(timezone.utc).isoformat()
	return session
