"""Critic agent -- research, plan review, and chaining via Claude subprocess."""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from typing import Any

from autodev.batch_analyzer import BatchSignals
from autodev.config import MissionConfig, build_claude_cmd, claude_subprocess_env
from autodev.context_gathering import (
	get_episodic_context,
	get_git_log,
	get_human_preferences,
	get_past_missions,
	get_strategic_context,
	read_backlog,
)
from autodev.db import Database
from autodev.json_utils import extract_json_from_text
from autodev.models import CriticFinding, Mission, WorkUnit
from autodev.recursive_planner import _parse_subprocess_cost

logger = logging.getLogger(__name__)

CRITIC_RESULT_MARKER = "CRITIC_RESULT:"


def _parse_critic_output(output: str) -> CriticFinding:
	"""Parse CRITIC_RESULT JSON from LLM output."""
	data = None

	idx = output.rfind(CRITIC_RESULT_MARKER)
	if idx != -1:
		remainder = output[idx + len(CRITIC_RESULT_MARKER):]
		data = extract_json_from_text(remainder)

	if not isinstance(data, dict):
		data = extract_json_from_text(output)

	if not isinstance(data, dict):
		logger.warning("Could not parse CRITIC_RESULT, returning findings from raw output")
		return CriticFinding(
			findings=[output[:2000]] if output else [],
			strategy_text=output[:2000] if output else "",
			verdict="needs_refinement",
		)

	return CriticFinding(
		findings=data.get("findings", []),
		risks=data.get("risks", []),
		gaps=data.get("gaps", []),
		open_questions=data.get("open_questions", []),
		verdict=data.get("verdict", "needs_refinement"),
		confidence=float(data.get("confidence", 0.0)),
		strategy_text=data.get("strategy_text", ""),
		proposed_objective=data.get("proposed_objective", ""),
	)


def _build_batch_signals_text(signals: BatchSignals) -> str:
	"""Format batch signals for inclusion in critic prompts."""
	hotspots = "\n".join(
		f"  - {f} ({c} touches)" for f, c in signals.file_hotspots
	) or "  (none)"
	failures = "\n".join(
		f"  - {k}: {v} failures" for k, v in signals.failure_clusters.items()
	) or "  (none)"
	stalled = "\n".join(f"  - {s}" for s in signals.stalled_areas) or "  (none)"
	effort = "\n".join(
		f"  - {k}: {v:.0%}" for k, v in signals.effort_distribution.items()
	) or "  (none)"

	return f"""## Execution Signals
- File hotspots (3+ touches):
{hotspots}
- Failure clusters:
{failures}
- Stalled areas (2+ attempts, no success):
{stalled}
- Effort distribution:
{effort}"""


def validate_units_preflight(
	units: list[WorkUnit],
	target_path: Path,
) -> list[str]:
	"""Pre-dispatch validation of work units.

	Checks:
	1. files_hint references files that actually exist in the target repo
	2. acceptance_criteria can be parsed as valid shell commands
	3. Unit titles and descriptions are non-empty

	Returns a list of validation failure strings (empty if all valid).
	"""
	failures: list[str] = []

	for i, unit in enumerate(units, 1):
		label = f"Unit {i}"

		# Check title is non-empty
		if not unit.title.strip():
			failures.append(f"{label}: empty title")

		# Check description is non-empty
		if not unit.description.strip():
			failures.append(f"{label} ({unit.title or '?'}): empty description")

		# Check files_hint references existing files
		if unit.files_hint and unit.files_hint.strip():
			for raw in unit.files_hint.split(","):
				path_str = raw.strip()
				if not path_str:
					continue
				full = target_path / path_str
				if not full.exists():
					failures.append(
						f"{label} ({unit.title}): files_hint references "
						f"non-existent path '{path_str}'"
					)

		# Check acceptance_criteria can be parsed as shell commands
		if unit.acceptance_criteria and unit.acceptance_criteria.strip():
			try:
				shlex.split(unit.acceptance_criteria)
			except ValueError as exc:
				failures.append(
					f"{label} ({unit.title}): acceptance_criteria has "
					f"malformed shell syntax: {exc}"
				)

	return failures


class CriticAgent:
	"""Builds critic prompts, spawns Claude subprocess, parses output."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self._config = config
		self._db = db

	async def research(
		self,
		objective: str,
		context: str,
		batch_signals: BatchSignals | None = None,
	) -> CriticFinding:
		"""Legacy research stub -- planner now leads research via web search.

		Returns a minimal finding so callers that still reference this method
		continue to work without errors.
		"""
		return CriticFinding(verdict="needs_refinement", confidence=0.5)

	async def review_plan(
		self,
		objective: str,
		units: list[WorkUnit],
		prev_finding: CriticFinding,
		batch_signals: BatchSignals | None = None,
	) -> tuple[CriticFinding, float]:
		"""Feasibility review: check whether proposed units are achievable."""
		# Pre-dispatch validation -- reject obviously malformed units early
		preflight_failures = validate_units_preflight(
			units, self._config.target.resolved_path,
		)
		if preflight_failures:
			logger.warning(
				"Preflight validation found %d issue(s)", len(preflight_failures),
			)
			return CriticFinding(
				findings=preflight_failures,
				verdict="needs_refinement",
				confidence=1.0,
			), 0.0

		units_text = "\n".join(
			f"  {i+1}. [{u.priority}] {u.title}: {u.description[:200]} "
			f"(files: {u.files_hint or 'unspecified'})"
			for i, u in enumerate(units)
		)

		signals_section = ""
		if batch_signals is not None:
			signals_section = "\n" + _build_batch_signals_text(batch_signals) + "\n"

		prompt = f"""You are a feasibility reviewer for a development plan. \
Your job is NOT to set strategy -- the planner has already done that. \
Your job is to check whether the proposed work units are achievable.

## Objective
{objective}
{signals_section}
## Proposed Work Units
{units_text}

## Review Criteria

For each unit, check:
1. Can one worker complete this in a single session?
2. Are file boundaries clean (no overlap between sibling units)?
3. Are the acceptance criteria testable?
4. Is the scope realistic given the codebase?

## Verdict

If all units pass, set verdict to "sufficient".
If units need adjustment, set verdict to "needs_refinement" with specific, actionable feedback.

Do NOT propose new strategic direction. Do NOT add units. Only refine what exists.

CRITIC_RESULT:{{"findings": ["..."],\
 "risks": ["..."],\
 "gaps": ["..."],\
 "open_questions": [],\
 "verdict": "sufficient|needs_refinement",\
 "confidence": 0.8,\
 "strategy_text": ""}}"""

		output, cost = await self._invoke_llm(prompt, "critic-review", use_mcp=False)
		finding = _parse_critic_output(output)
		return finding, cost

	async def propose_next(
		self,
		mission: Mission,
		result: Any,
		context: str,
	) -> tuple[CriticFinding, float]:
		"""Analyze completed mission and propose next objective for chaining."""
		objective_met = getattr(result, "objective_met", False)
		total_merged = getattr(result, "total_units_merged", 0)
		total_failed = getattr(result, "total_units_failed", 0)
		stopped_reason = getattr(result, "stopped_reason", "")

		prompt = f"""You are a strategic critic analyzing a completed mission to propose the next objective.

## Completed Mission
Objective: {mission.objective}
Objective met: {objective_met}
Units merged: {total_merged}, failed: {total_failed}
Stopped reason: {stopped_reason}

## Project Context
{context}

## Your Task

1. Analyze what was accomplished and what remains
2. Consider the project's strategic direction (backlog, past missions)
3. Propose the single most impactful next objective
4. Explain why this should be next

CRITIC_RESULT:{{"findings": ["what was accomplished..."],\
 "risks": ["risks for next mission..."],\
 "gaps": [],\
 "open_questions": [],\
 "verdict": "sufficient",\
 "confidence": 0.8,\
 "strategy_text": "Rationale for next objective...",\
 "proposed_objective": "The next objective to pursue"}}"""

		output, cost = await self._invoke_llm(prompt, "critic-chaining", use_mcp=True)
		finding = _parse_critic_output(output)
		return finding, cost

	def gather_context(self, mission: Mission | None = None) -> str:
		"""Gather all available project context for critic prompts."""
		sections: list[str] = []

		backlog = read_backlog(self._config)
		if backlog:
			sections.append(f"### BACKLOG.md\n{backlog}")

		past = get_past_missions(self._db)
		if past:
			sections.append(f"### Past Missions\n{past}")

		strategic = get_strategic_context(self._db)
		if strategic:
			sections.append(f"### Strategic Context\n{strategic}")

		episodic = get_episodic_context(self._db)
		if episodic:
			sections.append(f"### Past Learnings\n{episodic}")

		human_prefs = get_human_preferences(self._db)
		if human_prefs:
			sections.append(f"### Human Preferences\n{human_prefs}")

		return "\n\n".join(sections) if sections else "(No project context available)"

	async def gather_context_async(self, mission: Mission | None = None) -> str:
		"""Gather context including async sources (git log)."""
		context = self.gather_context(mission)
		git_log = await get_git_log(self._config)
		if git_log:
			context = f"### Recent Git History\n{git_log}\n\n{context}"
		return context

	async def _invoke_llm(
		self, prompt: str, label: str, *, use_mcp: bool = False,
	) -> tuple[str, float]:
		"""Run a prompt through a Claude subprocess.

		Returns:
			Tuple of (output_text, cost_usd).
		"""
		delib = self._config.deliberation
		budget = delib.critic_budget_usd
		model = delib.critic_model or self._config.scheduler.model
		timeout = delib.timeout
		cwd = str(self._config.target.resolved_path)

		logger.info("Invoking %s LLM (budget=$%.2f, model=%s)", label, budget, model)

		cmd = build_claude_cmd(self._config, model=model, budget=budget)
		try:
			proc = await asyncio.create_subprocess_exec(
				*cmd,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(self._config),
				cwd=cwd,
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
				timeout=timeout,
			)
			output = stdout.decode() if stdout else ""
			stderr_text = stderr.decode() if stderr else ""
		except asyncio.TimeoutError:
			logger.error("%s LLM timed out after %ds", label, timeout)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			return "", budget
		except (FileNotFoundError, OSError) as exc:
			logger.warning("%s LLM failed to start: %s", label, exc)
			return "", 0.0

		if proc.returncode != 0:
			err_msg = stderr.decode()[:200] if stderr else "unknown error"
			logger.warning("%s LLM failed (rc=%d): %s", label, proc.returncode, err_msg)
			return "", budget

		cost = _parse_subprocess_cost(stderr_text, budget)
		return output, cost
