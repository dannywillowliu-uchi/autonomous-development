"""Strategist agent -- proposes mission objectives autonomously.

Gathers context from BACKLOG.md, git history, past missions, strategic context,
and the priority queue, then calls Claude to propose a focused objective.
"""

from __future__ import annotations

import asyncio
import logging
from enum import IntEnum

from mission_control.config import MissionConfig, claude_subprocess_env
from mission_control.db import Database
from mission_control.json_utils import extract_json_from_text
from mission_control.memory import MemoryManager
from mission_control.models import BacklogItem, WorkUnit

log = logging.getLogger(__name__)

STRATEGY_RESULT_MARKER = "STRATEGY_RESULT:"
FOLLOWUP_RESULT_MARKER = "FOLLOWUP_RESULT:"


class AmbitionLevel(IntEnum):
	"""4-level ambition ladder for strategic objective proposals."""

	BUGS_QUALITY = 1
	IMPROVE_FEATURES = 2
	NEW_CAPABILITIES = 3
	META_IMPROVEMENTS = 4


AMBITION_LEVEL_DESCRIPTIONS: dict[AmbitionLevel, str] = {
	AmbitionLevel.BUGS_QUALITY: (
		"Fix bugs, code quality issues, security vulnerabilities, and technical debt"
	),
	AmbitionLevel.IMPROVE_FEATURES: (
		"Improve existing features: better performance, UX, error handling, adaptive behavior"
	),
	AmbitionLevel.NEW_CAPABILITIES: (
		"Add new capabilities that compound: web research, browser testing, "
		"multi-repo support, external integrations"
	),
	AmbitionLevel.META_IMPROVEMENTS: (
		"Meta-improvements that make the system better at improving itself: "
		"self-calibrating strategy, adaptive verification, outcome-driven planning"
	),
}


CAPABILITY_DOMAINS: list[dict[str, object]] = [
	{
		"name": "Web Research",
		"level": AmbitionLevel.NEW_CAPABILITIES,
		"description": (
			"Workers and strategist can search the web for documentation, "
			"best practices, and library versions"
		),
		"keywords": ["web research", "web search", "websearch", "api docs lookup"],
	},
	{
		"name": "Browser Automation",
		"level": AmbitionLevel.NEW_CAPABILITIES,
		"description": "Workers can validate built software via browser automation and end-to-end UI testing",
		"keywords": ["browser automation", "browser testing", "playwright", "end-to-end test"],
	},
	{
		"name": "Multi-Repository Support",
		"level": AmbitionLevel.NEW_CAPABILITIES,
		"description": (
			"Coordinate work across multiple repositories simultaneously "
			"with cross-repo dependency tracking"
		),
		"keywords": ["multi-repo", "multi repo", "cross-repo", "cross repo"],
	},
	{
		"name": "External Service Integrations",
		"level": AmbitionLevel.NEW_CAPABILITIES,
		"description": "Integration with CI/CD pipelines, issue trackers, deployment platforms, and monitoring systems",
		"keywords": ["ci/cd integration", "issue tracker integration", "deploy hook", "monitoring integration"],
	},
	{
		"name": "Self-Improving Planning",
		"level": AmbitionLevel.META_IMPROVEMENTS,
		"description": (
			"System learns from past mission outcomes to automatically "
			"improve planning quality and unit sizing"
		),
		"keywords": ["self-improving plan", "plan learning", "outcome feedback loop", "adaptive planning"],
	},
	{
		"name": "Adaptive Verification",
		"level": AmbitionLevel.META_IMPROVEMENTS,
		"description": (
			"Verification criteria automatically evolve based on discovered "
			"failure patterns and code complexity"
		),
		"keywords": ["adaptive verification", "verification evolution", "dynamic test generation"],
	},
	{
		"name": "Strategy Self-Calibration",
		"level": AmbitionLevel.META_IMPROVEMENTS,
		"description": "Strategist automatically calibrates ambition and scope based on historical success rates",
		"keywords": ["strategy calibration", "ambition calibration", "self-calibrat"],
	},
]


def _build_strategy_prompt(
	backlog_md: str,
	git_log: str,
	past_missions: str,
	strategic_context: str,
	pending_backlog: str,
	human_preferences: str = "",
	project_snapshot: str = "",
	ambition_level: AmbitionLevel | None = None,
	capability_gaps: str = "",
	web_research_context: str = "",
	episodic_context: str = "",
) -> str:
	ambition_section = ""
	if ambition_level is not None:
		level_desc = AMBITION_LEVEL_DESCRIPTIONS.get(ambition_level, "")
		all_levels = "\n".join(
			f"  Level {lvl.value}: {desc}"
			for lvl, desc in AMBITION_LEVEL_DESCRIPTIONS.items()
		)
		ambition_section = f"""
### Target Ambition Level: {ambition_level.value} - {ambition_level.name}
{level_desc}

The ambition ladder (escalate when lower levels are exhausted):
{all_levels}

You MUST propose an objective at Level {ambition_level.value} or higher.
"""

	capability_section = ""
	if capability_gaps:
		capability_section = f"""
### Capability Gap Analysis
The following capabilities are missing from the system:
{capability_gaps}
Consider proposing objectives that address these gaps, especially at Level 3-4.
"""

	research_section = ""
	if web_research_context:
		research_section = f"""
### Web Research Context
{web_research_context}
"""

	episodic_section = ""
	if episodic_context:
		episodic_section = f"""
### Past Learnings
{episodic_context}
"""

	escalation_instruction = ""
	if ambition_level and ambition_level.value >= 3:
		escalation_instruction = (
			f"\n6. Target ambition Level {ambition_level.value} or higher"
			" -- lower-level work has been exhausted."
		)

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
{ambition_section}{capability_section}{research_section}{episodic_section}
## Instructions

1. Analyze all context to understand what has been done and what needs doing.
2. Identify the highest-impact work that builds on recent progress.
3. Avoid proposing work that overlaps with recently completed missions.
4. Prefer ambitious objectives (architecture changes, new systems) over busywork (lint fixes).
5. The objective should be achievable in a single mission (1-5 work units).{escalation_instruction}

## Output Format

You may reason about your choice, but you MUST end your response with a STRATEGY_RESULT line:

STRATEGY_RESULT:{{"objective": "Actionable objective", "rationale": "Why this matters", "ambition_score": 7}}

- objective: A focused, actionable string describing what to build/fix/improve.
- rationale: 1-3 sentences explaining why this is highest priority.
- ambition_score: Integer 1-10 (1-3 = busywork, 4-6 = moderate, 7-10 = ambitious).

IMPORTANT: The STRATEGY_RESULT line must be the LAST line of your output."""


def _effective_score(item: BacklogItem) -> float:
	"""Return the effective priority score, preferring pinned_score if set."""
	return item.pinned_score if item.pinned_score is not None else item.priority_score


class Strategist:
	"""Proposes mission objectives by analyzing project context."""

	def __init__(self, config: MissionConfig, db: Database, memory_manager: MemoryManager | None = None) -> None:
		self.config = config
		self.db = db
		self._proposed_ambition_score: int | None = None
		self._memory_manager = memory_manager

	def _read_backlog(self) -> str:
		backlog_path = self.config.target.resolved_path / "BACKLOG.md"
		try:
			return backlog_path.read_text()
		except FileNotFoundError:
			log.info("No BACKLOG.md found at %s", backlog_path)
			return ""

	def analyze_capability_gaps(self) -> list[dict[str, object]]:
		"""Compare current system abilities vs potential capabilities.

		Returns capability domains not yet addressed in backlog or past missions.
		Called during propose_objective to inform Level 3-4 escalation.
		"""
		pending = self.db.get_pending_backlog(limit=50)
		missions = self.db.get_all_missions(limit=20)
		return self._compute_capability_gaps(pending, missions)

	def _compute_capability_gaps(
		self,
		pending_items: list[BacklogItem],
		missions: list[object],
	) -> list[dict[str, object]]:
		"""Identify capability domains not addressed by existing backlog or past work."""
		known_parts: list[str] = []
		for item in pending_items:
			known_parts.append(item.title.lower())
			known_parts.append(item.description.lower())
		for m in missions:
			obj = getattr(m, "objective", "")
			known_parts.append(obj.lower())
		known_text = " ".join(known_parts)

		gaps: list[dict[str, object]] = []
		for domain in CAPABILITY_DOMAINS:
			keywords = domain["keywords"]
			assert isinstance(keywords, list)
			addressed = any(kw in known_text for kw in keywords)
			if not addressed:
				gaps.append({
					"name": domain["name"],
					"level": int(domain["level"]),  # type: ignore[arg-type]
					"description": domain["description"],
				})
		return gaps

	def _determine_ambition_level(
		self,
		pending_items: list[BacklogItem],
		capability_gaps: list[dict[str, object]],
	) -> AmbitionLevel:
		"""Determine target ambition level based on backlog state.

		Follows the ambition ladder: address Level 1 first, then Level 2,
		then escalate to Level 3-4 when lower levels are exhausted.
		"""
		level_1_count = 0
		level_2_count = 0

		for item in pending_items:
			if item.status != "pending":
				continue
			if _effective_score(item) < 3.0:
				continue

			track = (item.track or "").lower()
			title_lower = item.title.lower()

			is_level_1 = (
				track in ("quality", "security")
				or any(kw in title_lower for kw in ("fix", "bug", "lint", "typo", "cleanup", "vulnerability"))
			)
			if is_level_1:
				level_1_count += 1
				continue

			is_level_2 = (
				track == "feature"
				or any(kw in title_lower for kw in ("improve", "enhance", "update", "refactor", "optimize"))
			)
			if is_level_2:
				level_2_count += 1

		if level_1_count > 0:
			return AmbitionLevel.BUGS_QUALITY
		if level_2_count > 0:
			return AmbitionLevel.IMPROVE_FEATURES

		# No Level 1-2 work remains -- MUST escalate
		level_3_gaps = [g for g in capability_gaps if g["level"] == int(AmbitionLevel.NEW_CAPABILITIES)]
		if level_3_gaps:
			return AmbitionLevel.NEW_CAPABILITIES
		return AmbitionLevel.META_IMPROVEMENTS

	def _get_web_research_context(self) -> str:
		"""Hook for injecting web research context into objective proposals.

		Returns external research findings to inform strategic decisions.
		Override or extend this method when web research tooling is available.
		"""
		return ""

	async def _get_git_log(self) -> str:
		try:
			proc = await asyncio.create_subprocess_exec(
				"git", "log", "--oneline", "-20",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=str(self.config.target.resolved_path),
			)
			stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
			output = stdout.decode() if stdout else ""
			return output.strip() if proc.returncode == 0 else ""
		except (asyncio.TimeoutError, FileNotFoundError, OSError):
			log.info("Could not read git log")
			return ""

	def _get_past_missions(self) -> str:
		missions = self.db.get_all_missions(limit=10)
		if not missions:
			return ""
		lines = []
		for m in missions:
			rating_str = ""
			try:
				ratings = self.db.get_trajectory_ratings_for_mission(m.id)
				if ratings:
					rating_str = f", human_rating={ratings[0].rating}/10"
			except Exception:
				pass
			lines.append(
				f"- [{m.status}] {m.objective[:120]} "
				f"(rounds={m.total_rounds}, score={m.final_score}, "
				f"reason={m.stopped_reason or 'n/a'}{rating_str})"
			)
		return "\n".join(lines)

	def _get_strategic_context(self) -> str:
		if not hasattr(self.db, "get_strategic_context"):
			return ""
		try:
			entries = self.db.get_strategic_context(limit=10)
			if not entries:
				return ""
			return "\n".join(f"- {e}" for e in entries)
		except Exception:
			log.debug("get_strategic_context not available", exc_info=True)
			return ""

	def _get_pending_backlog(self) -> str:
		items = self.db.get_pending_backlog(limit=10)
		if not items:
			return ""
		return "\n".join(
			f"- [score={_effective_score(item):.1f}] {item.title}: {item.description[:100]}"
			for item in items
		)

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

	def _heuristic_evaluate_ambition(self, planned_units: list[WorkUnit]) -> int:
		"""Score planned work on a 1-10 scale based on keyword heuristics.

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

	async def _zfc_evaluate_ambition(self, planned_units: list[WorkUnit]) -> int | None:
		"""Score ambition via LLM. Returns int 1-10 or None on failure."""
		unit_summaries = "\n".join(
			f"- {u.title}: {u.description[:120]} (files: {u.files_hint or 'n/a'})"
			for u in planned_units
		)
		prompt = f"""Score the ambition level of these planned work units on a 1-10 scale.

## Rubric
1-3 = busywork (lint fixes, typo corrections, minor formatting)
4-6 = moderate (new features, meaningful improvements, bug fixes)
7-10 = ambitious (architecture changes, new systems, multi-file refactors)

## Work Units
{unit_summaries}

You MUST end your response with:
AMBITION_RESULT:{{"score": N, "reasoning": "brief explanation"}}
"""
		zfc = self.config.zfc
		output = await self._invoke_llm(
			prompt, "zfc-ambition",
			raise_on_failure=False,
			budget_override=zfc.llm_budget_usd,
			timeout_override=zfc.llm_timeout,
		)
		if not output:
			return None

		marker = "AMBITION_RESULT:"
		idx = output.rfind(marker)
		if idx == -1:
			log.warning("No AMBITION_RESULT marker in ZFC ambition output")
			return None

		from mission_control.json_utils import extract_json_from_text
		remainder = output[idx + len(marker):]
		data = extract_json_from_text(remainder)
		if not isinstance(data, dict):
			log.warning("Failed to parse ZFC ambition JSON")
			return None

		try:
			score = int(data.get("score", 0))
			return max(1, min(10, score))
		except (TypeError, ValueError):
			log.warning("Invalid score in ZFC ambition result: %s", data.get("score"))
			return None

	async def evaluate_ambition(self, planned_units: list[WorkUnit]) -> int:
		"""Score planned work on a 1-10 scale.

		Uses ZFC LLM-backed scoring if enabled, with heuristic fallback.
		If zfc_propose_objective is enabled and a cached score exists from
		propose_objective, returns that score (consumed once).
		"""
		# ZFC objective passthrough: use cached score from propose_objective
		zfc = self.config.zfc
		if zfc.zfc_propose_objective and self._proposed_ambition_score is not None:
			score = self._proposed_ambition_score
			self._proposed_ambition_score = None  # consume once
			log.info("Using cached ambition score from propose_objective: %d", score)
			return score

		if zfc.zfc_ambition_scoring:
			zfc_score = await self._zfc_evaluate_ambition(planned_units)
			if zfc_score is not None:
				return zfc_score
			log.warning("ZFC ambition scoring failed, falling back to heuristic")

		return self._heuristic_evaluate_ambition(planned_units)

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
			if _effective_score(item) > 5.0
		]

		if not high_priority_items:
			return False, "No high-priority backlog items found"

		top = high_priority_items[0]
		score = _effective_score(top)
		return True, (
			f"Ambition score {ambition_score} is low. "
			f"Higher-priority backlog item available: '{top.title}' (priority={score:.1f}). "
			f"Consider replanning with a more ambitious objective."
		)

	async def _invoke_llm(
		self,
		prompt: str,
		label: str,
		raise_on_failure: bool = True,
		budget_override: float | None = None,
		timeout_override: int | None = None,
	) -> str:
		"""Run a prompt through the Claude subprocess and return raw output.

		Args:
			prompt: The prompt text to send.
			label: Human-readable label for logging (e.g. "strategist", "followup").
			raise_on_failure: If True, raise on non-zero exit. If False, return "".
			budget_override: Override default budget (for ZFC calls).
			timeout_override: Override default timeout (for ZFC calls).
		"""
		budget = budget_override if budget_override is not None else self.config.planner.budget_per_call_usd
		if budget_override is not None:
			model = self.config.zfc.model or self.config.scheduler.model
		else:
			model = self.config.scheduler.model
		timeout = timeout_override if timeout_override is not None else self.config.target.verification.timeout

		log.info("Invoking %s LLM", label)

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

	def _get_episodic_context(self) -> str:
		"""Load episodic + semantic memories and format as context for strategy prompts."""
		lines: list[str] = []
		try:
			semantic = self.db.get_top_semantic_memories(limit=5)
			if semantic:
				lines.append("Learned rules:")
				for sm in semantic:
					lines.append(f"  - [{sm.confidence:.1f}] {sm.content}")
		except Exception:
			log.debug("Failed to load semantic memories", exc_info=True)

		try:
			episodes = self.db.get_episodic_memories_by_scope(["mission", "strategy"], limit=10)
			if episodes:
				lines.append("Recent episodes:")
				for ep in episodes:
					lines.append(f"  - [{ep.event_type}] {ep.content} -> {ep.outcome}")
		except Exception:
			log.debug("Failed to load episodic memories", exc_info=True)

		return "\n".join(lines)

	def _get_human_preferences(self) -> str:
		"""Build a human preference signals section from trajectory ratings."""
		try:
			missions = self.db.get_all_missions(limit=20)
			high_rated: list[str] = []
			low_rated: list[str] = []
			for m in missions:
				ratings = self.db.get_trajectory_ratings_for_mission(m.id)
				if not ratings:
					continue
				r = ratings[0]  # most recent
				label = f"\"{m.objective[:80]}\" ({r.rating}/10)"
				if r.rating >= 7:
					high_rated.append(label)
				elif r.rating <= 4:
					low_rated.append(label)
			if not high_rated and not low_rated:
				return ""
			lines = []
			if high_rated:
				lines.append("Highly-rated: " + ", ".join(high_rated[:5]))
			if low_rated:
				lines.append("Low-rated: " + ", ".join(low_rated[:5]))
			lines.append(
				"Prefer work similar to highly-rated missions."
			)
			return "\n".join(lines)
		except Exception:
			return ""

	async def propose_objective(self) -> tuple[str, str, int]:
		"""Gather context and propose a mission objective via Claude.

		Performs capability gap analysis and determines target ambition level
		before invoking the LLM.

		Returns:
			Tuple of (objective, rationale, ambition_score).
		"""
		git_log = await self._get_git_log()
		try:
			from mission_control.snapshot import get_project_snapshot
			_snap = get_project_snapshot(self.config.target.resolved_path)
		except Exception:
			_snap = ""

		# Capability gap analysis and ambition level determination
		pending = self.db.get_pending_backlog(limit=50)
		missions = self.db.get_all_missions(limit=20)
		capability_gaps = self._compute_capability_gaps(pending, missions)
		ambition_level = self._determine_ambition_level(pending, capability_gaps)

		gaps_text = ""
		if capability_gaps:
			gaps_text = "\n".join(
				f"- [Level {g['level']}] {g['name']}: {g['description']}"
				for g in capability_gaps
			)

		web_research = self._get_web_research_context()
		episodic_ctx = self._get_episodic_context()

		log.info(
			"Ambition level: %s (%d capability gaps identified)",
			ambition_level.name, len(capability_gaps),
		)

		prompt = _build_strategy_prompt(
			backlog_md=self._read_backlog(),
			git_log=git_log,
			past_missions=self._get_past_missions(),
			strategic_context=self._get_strategic_context(),
			pending_backlog=self._get_pending_backlog(),
			human_preferences=self._get_human_preferences(),
			project_snapshot=_snap,
			ambition_level=ambition_level,
			capability_gaps=gaps_text,
			web_research_context=web_research,
			episodic_context=episodic_ctx,
		)
		output = await self._invoke_llm(prompt, "strategist")
		objective, rationale, ambition_score = self._parse_strategy_output(output)

		# ZFC passthrough: cache the LLM-proposed ambition score
		if self.config.zfc.zfc_propose_objective:
			self._proposed_ambition_score = ambition_score

		return objective, rationale, ambition_score

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
		# Persist mission summary as episodic memory for future strategy cycles
		self._store_mission_episode(mission_result)

		prompt = self._build_followup_prompt(mission_result, strategic_context)
		output = await self._invoke_llm(prompt, "followup", raise_on_failure=False)
		if not output:
			return ""
		return self._parse_followup_output(output)

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
