"""Deliberative planner -- ambitious planner + supplementary critic architecture.

The planner is the ambitious, outward-looking central agent (with web search).
The critic is a lightweight supplementary feasibility check.

  Round 1: Planner proposes (with project context + web search) -> Critic checks feasibility
  Round 2+: If critic says "needs_refinement", planner refines -> critic re-checks

The critic declares "sufficient" for early exit. Max rounds configurable.
"""

from __future__ import annotations

import logging
from typing import Any

from autodev.batch_analyzer import BatchSignals
from autodev.config import MissionConfig
from autodev.context_gathering import (
	get_episodic_context,
	get_git_log,
	get_human_preferences,
	get_intel_context,
	get_past_missions,
	get_strategic_context,
	read_backlog,
)
from autodev.critic_agent import CriticAgent
from autodev.db import Database
from autodev.grading import get_epoch_grade_feedback
from autodev.models import (
	CriticFinding,
	Epoch,
	KnowledgeItem,
	Mission,
	Plan,
	WorkUnit,
)
from autodev.overlap import resolve_file_overlaps
from autodev.recursive_planner import RecursivePlanner

logger = logging.getLogger(__name__)


def _format_critic_feedback(finding: CriticFinding) -> str:
	"""Format critic findings as plain text feedback for planner refinement."""
	sections: list[str] = []

	if finding.findings:
		items = "\n".join(f"- {f}" for f in finding.findings)
		sections.append(f"## Critic Findings\n{items}")

	if finding.risks:
		items = "\n".join(f"- {r}" for r in finding.risks)
		sections.append(f"## Risks\n{items}")

	if finding.gaps:
		items = "\n".join(f"- {g}" for g in finding.gaps)
		sections.append(f"## Gaps\n{items}")

	if finding.open_questions:
		items = "\n".join(f"- {q}" for q in finding.open_questions)
		sections.append(f"## Open Questions\n{items}")

	return "\n\n".join(sections)


class DeliberativePlanner:
	"""Ambitious planner + supplementary critic deliberation loop.

	Same interface as ContinuousPlanner so the controller swap is clean.
	"""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self._config = config
		self._db = db
		self._critic = CriticAgent(config, db)
		self._planner = RecursivePlanner(config, db)
		self._epoch_counter: int = 0
		self._current_strategy: str = ""

	def set_strategy(self, strategy: str) -> None:
		"""Store strategy text (for backward compat with controller)."""
		self._current_strategy = strategy

	def set_causal_context(self, risks: str) -> None:
		self._planner.set_causal_context(risks)

	def set_project_snapshot(self, snapshot: str) -> None:
		self._planner.set_project_snapshot(snapshot)

	async def _gather_project_context(self, mission: Mission, knowledge_context: str = "") -> str:
		"""Gather all project context for the planner prompt."""
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

		git_log = await get_git_log(self._config)
		if git_log:
			sections.insert(0, f"### Recent Git History\n{git_log}")

		intel_ctx = await get_intel_context(self._config)
		if intel_ctx:
			sections.append(intel_ctx)

		if knowledge_context:
			sections.append(f"### Accumulated Knowledge\n{knowledge_context}")

		grade_feedback = get_epoch_grade_feedback(self._db, mission.id)
		if grade_feedback:
			sections.append(f"### Decomposition Quality\n{grade_feedback}")

		return "\n\n".join(sections) if sections else "(No project context available)"

	async def get_next_units(
		self,
		mission: Mission,
		max_units: int = 3,
		feedback_context: str = "",
		knowledge_context: str = "",
		batch_signals: BatchSignals | None = None,
		**kwargs: object,
	) -> tuple[Plan, list[WorkUnit], Epoch]:
		"""Run planner/critic deliberation loop. Returns same signature as ContinuousPlanner."""
		self._epoch_counter += 1

		from autodev.snapshot import clear_snapshot_cache
		clear_snapshot_cache()

		epoch = Epoch(
			mission_id=mission.id,
			number=self._epoch_counter,
		)

		# Gather project context for the planner
		project_context = await self._gather_project_context(mission, knowledge_context)

		# Inject project context into planner's feedback
		enriched_feedback = feedback_context
		if project_context:
			enriched_feedback = (
				(enriched_feedback + "\n\n## Project Context\n" + project_context)
				if enriched_feedback
				else ("## Project Context\n" + project_context)
			)

		max_rounds = self._config.deliberation.max_rounds
		total_cost = 0.0

		# Round 1: Planner proposes with full context + web search
		plan, units, planner_cost = await self._planner.plan_round(
			objective=mission.objective,
			round_number=self._epoch_counter,
			feedback_context=enriched_feedback,
		)
		total_cost += planner_cost
		units = resolve_file_overlaps(units)
		plan.status = "active"
		plan.total_units = len(units)

		# Critic does feasibility review
		finding = CriticFinding(verdict="sufficient")
		if units:
			finding, critic_cost = await self._critic.review_plan(
				mission.objective, units, CriticFinding(), batch_signals,
			)
			total_cost += critic_cost

		# Rounds 2-N: If critic says needs_refinement, planner refines -> critic re-checks
		for round_num in range(2, max_rounds + 1):
			if not units:
				break

			if finding.verdict == "sufficient":
				logger.info(
					"Critic approved plan at round %d (confidence=%.2f)",
					round_num - 1, finding.confidence,
				)
				break

			logger.info(
				"Critic requests refinement at round %d: %d gaps, %d risks",
				round_num, len(finding.gaps), len(finding.risks),
			)

			# Planner refines with critic feedback
			critic_feedback_text = _format_critic_feedback(finding)
			refinement_context = enriched_feedback
			if critic_feedback_text:
				refinement_context = (
					(refinement_context + "\n\n" + critic_feedback_text)
					if refinement_context
					else critic_feedback_text
				)

			# Include current units so planner knows what to improve
			current_units_text = "\n".join(
				f"- {u.title}: {u.description[:100]} (files: {u.files_hint or 'unspecified'})"
				for u in units
			)
			refinement_context += f"\n\n## Previous Plan (needs refinement)\n{current_units_text}"

			plan, units, planner_cost = await self._planner.plan_round(
				objective=mission.objective,
				round_number=self._epoch_counter,
				feedback_context=refinement_context,
			)
			total_cost += planner_cost
			units = resolve_file_overlaps(units)
			plan.status = "active"
			plan.total_units = len(units)

			# Critic re-checks
			if units:
				finding, critic_cost = await self._critic.review_plan(
					mission.objective, units, finding, batch_signals,
				)
				total_cost += critic_cost

		# Store knowledge from critic findings
		self._store_knowledge(finding, mission)

		epoch.units_planned = len(units)
		epoch.planner_cost_usd = total_cost

		# Limit to max_units
		units = units[:max_units]

		logger.info(
			"Deliberation complete: epoch=%d, units=%d, planner_cost=$%.2f",
			self._epoch_counter, len(units), total_cost,
		)

		return plan, units, epoch

	async def propose_next_objective(
		self,
		mission: Mission,
		result: Any,
	) -> tuple[str, str]:
		"""For chaining: critic analyzes completed mission, proposes next objective.

		Returns:
			Tuple of (objective, rationale).
		"""
		context = await self._critic.gather_context_async(mission)
		finding, _cost = await self._critic.propose_next(mission, result, context)
		return finding.proposed_objective, finding.strategy_text

	def _write_strategy(self, strategy: str) -> None:
		"""Write MISSION_STRATEGY.md to disk."""
		target_path = self._config.target.resolved_path
		strategy_path = target_path / "MISSION_STRATEGY.md"
		try:
			strategy_path.write_text(strategy + "\n")
			logger.info("Wrote MISSION_STRATEGY.md (%d chars)", len(strategy))
		except OSError as exc:
			logger.warning("Could not write MISSION_STRATEGY.md: %s", exc)

	def _store_knowledge(self, finding: CriticFinding, mission: Mission) -> None:
		"""Store critic findings as KnowledgeItems in DB."""
		for item_text in finding.findings[:10]:
			ki = KnowledgeItem(
				mission_id=mission.id,
				source_unit_id="deliberative_planner",
				source_unit_type="research",
				title="Critic finding",
				content=str(item_text)[:500],
				scope="deliberation",
				confidence=finding.confidence or 0.7,
			)
			try:
				self._db.insert_knowledge_item(ki)
			except Exception as exc:
				logger.debug("Failed to store knowledge item: %s", exc)
