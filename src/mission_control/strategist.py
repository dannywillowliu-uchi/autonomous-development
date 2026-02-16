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
from mission_control.models import BacklogItem, WorkUnit

log = logging.getLogger(__name__)

STRATEGY_RESULT_MARKER = "STRATEGY_RESULT:"


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

	def evaluate_ambition(self, planned_units: list[WorkUnit]) -> int:
		"""Score planned work on a 1-10 scale based on heuristics.

		1-3 = busywork (lint fixes, minor refactors)
		4-6 = moderate (new features, meaningful improvements)
		7-10 = ambitious (architecture changes, new systems, multi-file refactors)
		"""
		if not planned_units:
			return 1

		# Keywords that indicate low-ambition work
		low_keywords = {
			"lint", "typo", "format", "formatting", "whitespace", "style",
			"cleanup", "clean up", "rename", "comment", "docstring",
			"minor", "trivial", "nit", "fixup",
		}
		# Keywords that indicate high-ambition work
		high_keywords = {
			"architecture", "architect", "system", "framework", "engine",
			"redesign", "rewrite", "new module", "new system", "pipeline",
			"infrastructure", "migration", "integrate", "integration",
			"distributed", "concurrent", "async", "multi",
		}
		# Keywords that indicate moderate work
		mid_keywords = {
			"feature", "add", "implement", "create", "build", "test",
			"refactor", "improve", "enhance", "update", "extend", "fix",
		}

		low_count = 0
		mid_count = 0
		high_count = 0
		total_files: set[str] = set()

		for unit in planned_units:
			text = f"{unit.title} {unit.description}".lower()

			if any(kw in text for kw in high_keywords):
				high_count += 1
			elif any(kw in text for kw in low_keywords):
				low_count += 1
			elif any(kw in text for kw in mid_keywords):
				mid_count += 1
			else:
				mid_count += 1  # default to moderate

			if unit.files_hint:
				for f in unit.files_hint.split(","):
					f = f.strip()
					if f:
						total_files.add(f)

		n = len(planned_units)

		# Base score from unit type distribution
		if n > 0:
			high_ratio = high_count / n
			low_ratio = low_count / n
		else:
			high_ratio = 0.0
			low_ratio = 0.0

		if high_ratio >= 0.5:
			type_score = 8.0
		elif high_ratio > 0:
			type_score = 6.0
		elif low_ratio >= 0.7:
			type_score = 2.0
		elif low_ratio >= 0.4:
			type_score = 3.0
		else:
			type_score = 5.0

		# File count modifier: more files = more ambitious
		file_count = len(total_files)
		if file_count >= 10:
			file_mod = 1.5
		elif file_count >= 5:
			file_mod = 1.0
		elif file_count >= 2:
			file_mod = 0.5
		else:
			file_mod = 0.0

		# Unit count modifier
		if n >= 5:
			count_mod = 1.0
		elif n >= 3:
			count_mod = 0.5
		else:
			count_mod = 0.0

		raw = type_score + file_mod + count_mod
		return max(1, min(10, round(raw)))

	def should_replan(self, ambition_score: int, backlog_items: list[BacklogItem]) -> tuple[bool, str]:
		"""Determine if the planner should be re-invoked with a more ambitious objective.

		Returns True if ambition < 4 AND there are higher-priority backlog items available.
		"""
		if ambition_score >= 4:
			return False, ""

		if not backlog_items:
			return False, "No higher-priority backlog items available"

		# Check if any backlog item has meaningful priority (> 5.0)
		high_priority_items = [
			item for item in backlog_items
			if (item.pinned_score if item.pinned_score is not None else item.priority_score) > 5.0
		]

		if not high_priority_items:
			return False, "No high-priority backlog items found"

		top = high_priority_items[0]
		score = top.pinned_score if top.pinned_score is not None else top.priority_score
		return True, (
			f"Ambition score {ambition_score} is low. "
			f"Higher-priority backlog item available: '{top.title}' (priority={score:.1f}). "
			f"Consider replanning with a more ambitious objective."
		)

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
