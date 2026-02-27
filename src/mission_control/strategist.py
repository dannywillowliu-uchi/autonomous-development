"""Strategist agent -- proposes mission objectives autonomously.

Gathers context from git history, past missions, and strategic context,
then calls Claude to propose a focused objective.
"""

from __future__ import annotations

import asyncio
import logging

from mission_control.config import MissionConfig, build_claude_cmd, claude_subprocess_env
from mission_control.context_gathering import (
	get_episodic_context,
	get_git_log,
	get_human_preferences,
	get_past_missions,
	get_strategic_context,
	read_backlog,
)
from mission_control.db import Database
from mission_control.json_utils import extract_json_from_text
from mission_control.memory import MemoryManager

log = logging.getLogger(__name__)

STRATEGY_RESULT_MARKER = "STRATEGY_RESULT:"


def _build_strategy_prompt(
	backlog_md: str,
	git_log: str,
	past_missions: str,
	strategic_context: str,
	pending_backlog: str,
	human_preferences: str = "",
	project_snapshot: str = "",
	episodic_context: str = "",
) -> str:
	episodic_section = ""
	if episodic_context:
		episodic_section = f"""
### Past Learnings
{episodic_context}
"""

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

### Human Quality Signals
{human_preferences or "(No human ratings available yet)"}

### Project Structure
{project_snapshot or "(No project structure available)"}
{episodic_section}
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

	def __init__(self, config: MissionConfig, db: Database, memory_manager: MemoryManager | None = None) -> None:
		self.config = config
		self.db = db
		self._memory_manager = memory_manager

	def _read_backlog(self) -> str:
		return read_backlog(self.config)

	async def _get_git_log(self) -> str:
		return await get_git_log(self.config)

	def _get_past_missions(self) -> str:
		return get_past_missions(self.db)

	def _get_strategic_context(self) -> str:
		return get_strategic_context(self.db)

	def _get_pending_backlog(self) -> str:
		return ""

	def _get_human_preferences(self) -> str:
		return get_human_preferences(self.db)

	def _get_episodic_context(self) -> str:
		return get_episodic_context(self.db)

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

	async def _invoke_llm(self, prompt: str, label: str, raise_on_failure: bool = True) -> str:
		"""Run a prompt through the Claude subprocess and return raw output."""
		budget = self.config.planner.budget_per_call_usd
		model = self.config.scheduler.model
		timeout = self.config.target.verification.timeout

		log.info("Invoking %s LLM", label)

		cmd = build_claude_cmd(self.config, model=model, budget=budget)
		try:
			proc = await asyncio.create_subprocess_exec(
				*cmd,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(self.config),
				cwd=str(self.config.target.resolved_path),
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
				timeout=timeout,
			)
			output = stdout.decode() if stdout else ""
		except asyncio.TimeoutError:
			log.error("%s LLM timed out after %ds", label, timeout)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			if raise_on_failure:
				raise
			return ""

		if proc.returncode != 0:
			err_msg = stderr.decode()[:200] if stderr else "unknown error"
			log.error("%s LLM failed (rc=%d): %s", label, proc.returncode, err_msg)
			if raise_on_failure:
				raise RuntimeError(f"{label} subprocess failed (rc={proc.returncode}): {err_msg}")
			return ""

		return output

	async def propose_objective(self) -> tuple[str, str, int]:
		"""Gather context and propose a mission objective via Claude.

		Returns:
			Tuple of (objective, rationale, ambition_score).
		"""
		git_log = await self._get_git_log()
		try:
			from mission_control.snapshot import get_project_snapshot
			_snap = get_project_snapshot(self.config.target.resolved_path)
		except Exception:
			_snap = ""

		episodic_ctx = self._get_episodic_context()

		prompt = _build_strategy_prompt(
			backlog_md=self._read_backlog(),
			git_log=git_log,
			past_missions=self._get_past_missions(),
			strategic_context=self._get_strategic_context(),
			pending_backlog=self._get_pending_backlog(),
			human_preferences=self._get_human_preferences(),
			project_snapshot=_snap,
			episodic_context=episodic_ctx,
		)
		output = await self._invoke_llm(prompt, "strategist")
		return self._parse_strategy_output(output)

	def _store_mission_episode(self, mission_result: object) -> None:
		"""Persist a mission summary as an episodic memory for cross-mission learning."""
		if self._memory_manager is None:
			return

		objective = getattr(mission_result, "objective", "")
		objective_met = getattr(mission_result, "objective_met", False)
		total_merged = getattr(mission_result, "total_units_merged", 0)
		total_failed = getattr(mission_result, "total_units_failed", 0)
		stopped_reason = getattr(mission_result, "stopped_reason", "")

		content = (
			f"Mission: {objective[:200]}. "
			f"Merged={total_merged}, Failed={total_failed}, "
			f"Stopped={stopped_reason or 'normal'}"
		)
		outcome = "pass" if objective_met else "fail"
		project_name = self.config.target.name or "unknown"

		try:
			self._memory_manager.store_episode(
				event_type="mission_summary",
				content=content,
				outcome=outcome,
				scope_tokens=["mission", "strategy", project_name],
			)
		except Exception:
			log.warning("Failed to store mission episode", exc_info=True)
