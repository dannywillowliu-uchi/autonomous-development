"""Deliberative planner -- critic/planner dual-agent architecture.

Replaces the separate research phase, strategic reflection, and strategist
with a unified deliberation loop:

  Round 1: Critic researches -> Planner decomposes
  Round 2+: Critic reviews plan -> Planner refines (if needed)

The critic declares "sufficient" for early exit. Max rounds configurable.
"""

from __future__ import annotations

import logging
from typing import Any

from mission_control.batch_analyzer import BatchSignals
from mission_control.config import MissionConfig
from mission_control.critic_agent import CriticAgent
from mission_control.db import Database
from mission_control.models import (
	CriticFinding,
	Epoch,
	KnowledgeItem,
	Mission,
	Plan,
	WorkUnit,
)
from mission_control.planner_agent import PlannerAgent

log = logging.getLogger(__name__)


class DeliberativePlanner:
	"""Dual-agent deliberation loop: critic researches, planner decomposes.

	Same interface as ContinuousPlanner so the controller swap is clean.
	"""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self._config = config
		self._db = db
		self._critic = CriticAgent(config, db)
		self._planner = PlannerAgent(config, db)
		self._epoch_counter: int = 0
		self._current_strategy: str = ""

	def set_strategy(self, strategy: str) -> None:
		"""Store strategy text (for backward compat with controller)."""
		self._current_strategy = strategy

	def set_causal_context(self, risks: str) -> None:
		self._planner.set_causal_context(risks)

	def set_project_snapshot(self, snapshot: str) -> None:
		self._planner.set_project_snapshot(snapshot)

	async def get_next_units(
		self,
		mission: Mission,
		max_units: int = 3,
		feedback_context: str = "",
		knowledge_context: str = "",
		batch_signals: BatchSignals | None = None,
		**kwargs: object,
	) -> tuple[Plan, list[WorkUnit], Epoch]:
		"""Run critic/planner deliberation loop. Returns same signature as ContinuousPlanner."""
		self._epoch_counter += 1

		from mission_control.snapshot import clear_snapshot_cache
		clear_snapshot_cache()

		epoch = Epoch(
			mission_id=mission.id,
			number=self._epoch_counter,
		)

		# Gather context for the critic
		context = await self._critic.gather_context_async(mission)
		if knowledge_context:
			context += f"\n\n## Accumulated Knowledge\n{knowledge_context}"

		max_rounds = self._config.deliberation.max_rounds

		# Round 1: Critic researches, planner decomposes
		finding = await self._critic.research(
			mission.objective, context, batch_signals,
		)
		plan, units = await self._planner.decompose(
			mission.objective, finding, feedback_context,
			round_number=self._epoch_counter,
		)

		# Rounds 2-N: Critic reviews, planner refines (if needed)
		for round_num in range(2, max_rounds + 1):
			if not units:
				break

			finding = await self._critic.review_plan(
				mission.objective, units, finding, batch_signals,
			)
			if finding.verdict == "sufficient":
				log.info(
					"Critic approved plan at round %d (confidence=%.2f)",
					round_num, finding.confidence,
				)
				break

			log.info(
				"Critic requests refinement at round %d: %d gaps, %d risks",
				round_num, len(finding.gaps), len(finding.risks),
			)
			plan, units = await self._planner.refine(
				mission.objective, units, finding, feedback_context,
				round_number=self._epoch_counter,
			)

		# Write strategy and store knowledge
		if finding.strategy_text:
			self._current_strategy = finding.strategy_text
			self._write_strategy(finding.strategy_text)
		self._store_knowledge(finding, mission)

		epoch.units_planned = len(units)

		# Limit to max_units
		units = units[:max_units]

		log.info(
			"Deliberation complete: epoch=%d, units=%d, rounds=%d",
			self._epoch_counter, len(units), min(round_num, max_rounds) if units else 1,
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
		finding = await self._critic.propose_next(mission, result, context)
		return finding.proposed_objective, finding.strategy_text

	def _write_strategy(self, strategy: str) -> None:
		"""Write MISSION_STRATEGY.md to disk."""
		target_path = self._config.target.resolved_path
		strategy_path = target_path / "MISSION_STRATEGY.md"
		try:
			strategy_path.write_text(strategy + "\n")
			log.info("Wrote MISSION_STRATEGY.md (%d chars)", len(strategy))
		except OSError as exc:
			log.warning("Could not write MISSION_STRATEGY.md: %s", exc)

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
				log.debug("Failed to store knowledge item: %s", exc)
