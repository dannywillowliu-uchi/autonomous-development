"""Critic agent -- research, plan review, and chaining via Claude subprocess."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mission_control.batch_analyzer import BatchSignals
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
from mission_control.models import CriticFinding, Mission, WorkUnit

log = logging.getLogger(__name__)

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
		log.warning("Could not parse CRITIC_RESULT, returning findings from raw output")
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
		"""First-pass research: analyze codebase, domain, and prior art.

		Runs with MCP tools in the target directory for codebase access.
		"""
		signals_section = ""
		if batch_signals is not None:
			signals_section = "\n" + _build_batch_signals_text(batch_signals) + "\n"

		prompt = f"""You are a critic agent analyzing a project before planning begins.

## Objective
{objective}

## Project Context
{context}
{signals_section}
## Your Task

1. Analyze the codebase to understand current architecture and patterns
2. Research the problem domain -- how is this type of work typically approached?
3. Check prior art: git history, BACKLOG.md, past missions for related attempts
4. Identify risks, gaps in understanding, and open questions
5. Produce a strategic recommendation for how to approach this objective

## Output Format

Reason in prose first, then emit your structured findings:

CRITIC_RESULT:{{"findings": ["key finding 1", ...],\
 "risks": ["risk 1", ...],\
 "gaps": ["knowledge gap 1", ...],\
 "open_questions": ["question 1", ...],\
 "verdict": "needs_refinement",\
 "confidence": 0.7,\
 "strategy_text": "Recommended approach: ..."}}

IMPORTANT: The CRITIC_RESULT line must contain valid JSON."""

		output = await self._invoke_llm(prompt, "critic-research", use_mcp=True)
		return _parse_critic_output(output)

	async def review_plan(
		self,
		objective: str,
		units: list[WorkUnit],
		prev_finding: CriticFinding,
		batch_signals: BatchSignals | None = None,
	) -> CriticFinding:
		"""Review proposed work units and judge whether they're sufficient."""
		units_text = "\n".join(
			f"  {i+1}. [{u.priority}] {u.title}: {u.description[:200]} "
			f"(files: {u.files_hint or 'unspecified'})"
			for i, u in enumerate(units)
		)

		prev_findings_text = "\n".join(f"  - {f}" for f in prev_finding.findings) or "  (none)"
		prev_risks_text = "\n".join(f"  - {r}" for r in prev_finding.risks) or "  (none)"
		prev_gaps_text = "\n".join(f"  - {g}" for g in prev_finding.gaps) or "  (none)"

		signals_section = ""
		if batch_signals is not None:
			signals_section = "\n" + _build_batch_signals_text(batch_signals) + "\n"

		prompt = f"""You are a critic agent reviewing a proposed plan.

## Objective
{objective}

## Previous Findings
{prev_findings_text}

## Previous Risks
{prev_risks_text}

## Previous Gaps
{prev_gaps_text}
{signals_section}
## Proposed Work Units
{units_text}

## Your Task

1. Do these units address the objective completely?
2. Are there gaps -- important work that's missing?
3. Are there risks the plan doesn't mitigate?
4. Is the file isolation sufficient to avoid merge conflicts?
5. Are dependencies between units correctly ordered?

Set verdict to "sufficient" if the plan is ready for execution.
Set verdict to "needs_refinement" if it needs improvement.

CRITIC_RESULT:{{"findings": ["..."],\
 "risks": ["..."],\
 "gaps": ["..."],\
 "open_questions": ["..."],\
 "verdict": "sufficient|needs_refinement",\
 "confidence": 0.8,\
 "strategy_text": "..."}}"""

		output = await self._invoke_llm(prompt, "critic-review", use_mcp=False)
		return _parse_critic_output(output)

	async def propose_next(
		self,
		mission: Mission,
		result: Any,
		context: str,
	) -> CriticFinding:
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

		output = await self._invoke_llm(prompt, "critic-chaining", use_mcp=True)
		return _parse_critic_output(output)

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
	) -> str:
		"""Run a prompt through a Claude subprocess."""
		delib = self._config.deliberation
		budget = delib.critic_budget_usd
		model = delib.critic_model or self._config.scheduler.model
		timeout = delib.timeout
		cwd = str(self._config.target.resolved_path)

		log.info("Invoking %s LLM (budget=$%.2f, model=%s)", label, budget, model)

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
		except asyncio.TimeoutError:
			log.error("%s LLM timed out after %ds", label, timeout)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			return ""
		except (FileNotFoundError, OSError) as exc:
			log.warning("%s LLM failed to start: %s", label, exc)
			return ""

		if proc.returncode != 0:
			err_msg = stderr.decode()[:200] if stderr else "unknown error"
			log.warning("%s LLM failed (rc=%d): %s", label, proc.returncode, err_msg)
			return ""

		return output
