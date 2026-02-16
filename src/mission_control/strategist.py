"""Strategist agent -- proposes mission objectives autonomously.

Gathers context from BACKLOG.md, git history, past missions, strategic context,
and the priority queue, then calls Claude to propose a focused objective.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess

from mission_control.config import MissionConfig, claude_subprocess_env
from mission_control.db import Database
from mission_control.json_utils import extract_json_from_text

log = logging.getLogger(__name__)

STRATEGY_RESULT_MARKER = "STRATEGY_RESULT:"
FOLLOWUP_RESULT_MARKER = "FOLLOWUP_RESULT:"


def _build_strategy_prompt(
	backlog_md: str,
	git_log: str,
	past_missions: str,
	strategic_context: str,
	pending_backlog: str,
) -> str:
	return f"""You are a strategic engineering lead for an autonomous development system.

Your job: propose the SINGLE most impactful mission objective to work on next.

## Context

### BACKLOG.md (project roadmap)
{backlog_md or "(No BACKLOG.md found -- propose based on other context)"}

### Recent Git History (last 20 commits)
{git_log or "(No git history available -- this may be a new project)"}

### Past Mission Reports
{past_missions or "(No prior missions -- this is the first mission)"}

### Rolling Strategic Context
{strategic_context or "(No strategic context yet -- this is the first strategy cycle)"}

### Priority Queue (pending backlog items)
{pending_backlog or "(No pending backlog items)"}

## Instructions

1. Analyze all context to understand what has been done and what needs doing.
2. Identify the highest-impact work that builds on recent progress.
3. Avoid proposing work that overlaps with recently completed missions.
4. Prefer ambitious objectives (architecture changes, new systems) over busywork (lint fixes).
5. The objective should be achievable in a single mission (1-5 work units).

## Output Format

You may reason about your choice, but you MUST end your response with a STRATEGY_RESULT line:

STRATEGY_RESULT:{{"objective": "Actionable objective", "rationale": "Why this matters", "ambition_score": 7}}

- objective: A focused, actionable string describing what to build/fix/improve.
- rationale: 1-3 sentences explaining why this is highest priority.
- ambition_score: Integer 1-10 (1-3 = busywork, 4-6 = moderate, 7-10 = ambitious).

IMPORTANT: The STRATEGY_RESULT line must be the LAST line of your output."""


class Strategist:
	"""Proposes mission objectives by analyzing project context."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db

	def _read_backlog(self) -> str:
		backlog_path = self.config.target.resolved_path / "BACKLOG.md"
		try:
			return backlog_path.read_text()
		except FileNotFoundError:
			log.info("No BACKLOG.md found at %s", backlog_path)
			return ""

	def _get_git_log(self) -> str:
		try:
			result = subprocess.run(
				["git", "log", "--oneline", "-20"],
				capture_output=True,
				text=True,
				cwd=str(self.config.target.resolved_path),
				timeout=10,
			)
			return result.stdout.strip() if result.returncode == 0 else ""
		except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
			log.info("Could not read git log")
			return ""

	def _get_past_missions(self) -> str:
		missions = self.db.get_all_missions(limit=10)
		if not missions:
			return ""
		lines = []
		for m in missions:
			lines.append(
				f"- [{m.status}] {m.objective[:120]} "
				f"(rounds={m.total_rounds}, score={m.final_score}, "
				f"reason={m.stopped_reason or 'n/a'})"
			)
		return "\n".join(lines)

	def _get_strategic_context(self) -> str:
		if not hasattr(self.db, "get_strategic_context"):
			return ""
		try:
			entries = self.db.get_strategic_context(limit=10)
			if not entries:
				return ""
			lines = []
			for e in entries:
				lines.append(f"- {e}")
			return "\n".join(lines)
		except Exception:
			log.debug("get_strategic_context not available", exc_info=True)
			return ""

	def _get_pending_backlog(self) -> str:
		items = self.db.get_pending_backlog(limit=10)
		if not items:
			return ""
		lines = []
		for item in items:
			score = item.pinned_score if item.pinned_score is not None else item.priority_score
			lines.append(f"- [score={score:.1f}] {item.title}: {item.description[:100]}")
		return "\n".join(lines)

	def _parse_strategy_output(self, output: str) -> tuple[str, str, int]:
		"""Parse STRATEGY_RESULT from LLM output. Returns (objective, rationale, ambition_score)."""
		idx = output.rfind(STRATEGY_RESULT_MARKER)
		data = None
		if idx != -1:
			remainder = output[idx + len(STRATEGY_RESULT_MARKER):]
			data = extract_json_from_text(remainder)

		if not isinstance(data, dict):
			data = extract_json_from_text(output)

		if not isinstance(data, dict):
			raise ValueError(f"Could not parse STRATEGY_RESULT from output ({len(output)} chars)")

		objective = str(data.get("objective", "")).strip()
		rationale = str(data.get("rationale", "")).strip()
		raw_score = data.get("ambition_score", 5)
		try:
			ambition_score = max(1, min(10, int(raw_score)))
		except (TypeError, ValueError):
			ambition_score = 5

		if not objective:
			raise ValueError("Empty objective in STRATEGY_RESULT")

		return objective, rationale, ambition_score

	async def propose_objective(self) -> tuple[str, str, int]:
		"""Gather context and propose a mission objective via Claude.

		Returns:
			Tuple of (objective, rationale, ambition_score).
		"""
		backlog_md = self._read_backlog()
		git_log = self._get_git_log()
		past_missions = self._get_past_missions()
		strategic_context = self._get_strategic_context()
		pending_backlog = self._get_pending_backlog()

		prompt = _build_strategy_prompt(
			backlog_md=backlog_md,
			git_log=git_log,
			past_missions=past_missions,
			strategic_context=strategic_context,
			pending_backlog=pending_backlog,
		)

		budget = self.config.planner.budget_per_call_usd
		model = self.config.scheduler.model
		timeout = self.config.target.verification.timeout

		log.info("Invoking strategist LLM to propose objective")

		try:
			proc = await asyncio.create_subprocess_exec(
				"claude", "-p",
				"--output-format", "text",
				"--max-budget-usd", str(budget),
				"--model", model,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(),
				cwd=str(self.config.target.resolved_path),
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
				timeout=timeout,
			)
			output = stdout.decode() if stdout else ""
		except asyncio.TimeoutError:
			log.error("Strategist LLM timed out after %ds", timeout)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			raise

		if proc.returncode != 0:
			err_msg = stderr.decode()[:200] if stderr else "unknown error"
			log.error("Strategist LLM failed (rc=%d): %s", proc.returncode, err_msg)
			raise RuntimeError(f"Strategist subprocess failed (rc={proc.returncode}): {err_msg}")

		return self._parse_strategy_output(output)

	def _build_followup_prompt(self, mission_result: object, strategic_context: str) -> str:
		objective = getattr(mission_result, "objective", "")
		objective_met = getattr(mission_result, "objective_met", False)
		total_dispatched = getattr(mission_result, "total_units_dispatched", 0)
		total_merged = getattr(mission_result, "total_units_merged", 0)
		total_failed = getattr(mission_result, "total_units_failed", 0)
		stopped_reason = getattr(mission_result, "stopped_reason", "")
		wall_time = getattr(mission_result, "wall_time_seconds", 0.0)

		pending_backlog = self._get_pending_backlog()

		return f"""You are a strategic engineering lead evaluating whether a completed mission needs follow-up work.

## Completed Mission
- Objective: {objective}
- Objective met: {objective_met}
- Units dispatched: {total_dispatched}, merged: {total_merged}, failed: {total_failed}
- Stopped reason: {stopped_reason or "normal completion"}
- Wall time: {wall_time:.0f}s

## Rolling Strategic Context
{strategic_context or "(No strategic context available)"}

## Pending Backlog Items
{pending_backlog or "(No pending backlog items)"}

## Instructions

Decide if follow-up work is needed. Follow-up is warranted when:
1. The objective was NOT fully met and there is remaining work to do.
2. Multiple units failed and the failures indicate fixable issues.
3. There are high-priority pending backlog items that build on this mission's progress.

Follow-up is NOT needed when:
1. The objective was fully met with no failures.
2. The remaining work is unrelated to the completed mission.
3. There is no pending backlog.

## Output Format

You MUST end your response with a FOLLOWUP_RESULT line:

FOLLOWUP_RESULT:{{"next_objective": "Follow-up objective or empty string", "rationale": "Why"}}


- next_objective: A focused follow-up objective string. Use empty string "" if no follow-up is needed.
- rationale: 1-2 sentences explaining the decision.

IMPORTANT: The FOLLOWUP_RESULT line must be the LAST line of your output."""

	def _parse_followup_output(self, output: str) -> str:
		"""Parse FOLLOWUP_RESULT from LLM output. Returns next_objective string."""
		idx = output.rfind(FOLLOWUP_RESULT_MARKER)
		data = None
		if idx != -1:
			remainder = output[idx + len(FOLLOWUP_RESULT_MARKER):]
			data = extract_json_from_text(remainder)

		if not isinstance(data, dict):
			data = extract_json_from_text(output)

		if not isinstance(data, dict):
			log.warning("Could not parse FOLLOWUP_RESULT from output (%d chars), assuming no follow-up", len(output))
			return ""

		return str(data.get("next_objective", "")).strip()

	async def suggest_followup(self, mission_result: object, strategic_context: str) -> str:
		"""Call Claude to determine if follow-up work is needed after a mission.

		Args:
			mission_result: The completed mission result with objective_met, stopped_reason, etc.
			strategic_context: Rolling strategic context string from previous missions.

		Returns:
			A next_objective string if follow-up is warranted, or empty string.
		"""
		prompt = self._build_followup_prompt(mission_result, strategic_context)
		budget = self.config.planner.budget_per_call_usd
		model = self.config.scheduler.model
		timeout = self.config.target.verification.timeout

		log.info("Invoking strategist LLM to evaluate follow-up")

		try:
			proc = await asyncio.create_subprocess_exec(
				"claude", "-p",
				"--output-format", "text",
				"--max-budget-usd", str(budget),
				"--model", model,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(),
				cwd=str(self.config.target.resolved_path),
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
				timeout=timeout,
			)
			output = stdout.decode() if stdout else ""
		except asyncio.TimeoutError:
			log.error("Followup LLM timed out after %ds", timeout)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			return ""

		if proc.returncode != 0:
			err_msg = stderr.decode()[:200] if stderr else "unknown error"
			log.error("Followup LLM failed (rc=%d): %s", proc.returncode, err_msg)
			return ""

		return self._parse_followup_output(output)
