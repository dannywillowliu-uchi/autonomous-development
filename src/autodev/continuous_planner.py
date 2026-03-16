"""Continuous planner -- flat impact-focused planning (no backlog, no recursion)."""

from __future__ import annotations

import logging

from autodev.config import MissionConfig
from autodev.db import Database
from autodev.models import Epoch, Mission, Plan, WorkUnit
from autodev.overlap import resolve_file_overlaps
from autodev.recursive_planner import RecursivePlanner

logger = logging.getLogger(__name__)


class ContinuousPlanner:
	"""Flat impact-focused planner: invokes LLM every iteration with full state."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self._inner = RecursivePlanner(config, db)
		self._config = config
		self._db = db
		self._epoch_count: int = 0
		self._strategy: str = ""

	def set_strategy(self, strategy: str) -> None:
		"""Store the research phase strategy for inclusion in planner context."""
		self._strategy = strategy

	def set_causal_context(self, risks: str) -> None:
		"""Set causal risk factors, delegating to the inner planner."""
		self._inner.set_causal_context(risks)

	def set_project_snapshot(self, snapshot: str) -> None:
		"""Set project structure snapshot, delegating to the inner planner."""
		self._inner.set_project_snapshot(snapshot)

	async def get_next_units(
		self,
		mission: Mission,
		max_units: int = 3,
		feedback_context: str = "",
		knowledge_context: str = "",
		locked_files: dict[str, list[str]] | None = None,
		**kwargs,
	) -> tuple[Plan, list[WorkUnit], Epoch]:
		"""Plan the next batch of units using the flat impact prompt."""
		self._epoch_count += 1
		from autodev.snapshot import clear_snapshot_cache
		clear_snapshot_cache()

		epoch = Epoch(
			mission_id=mission.id,
			number=self._epoch_count,
		)

		# Build the enriched context (planner reads MISSION_STATE.md from disk)
		enriched_context = feedback_context
		if knowledge_context:
			enriched_context = (
				(enriched_context + "\n\n## Accumulated Knowledge\n" + knowledge_context)
				if enriched_context
				else ("## Accumulated Knowledge\n" + knowledge_context)
			)

		plan, units, planner_cost = await self._inner.plan_round(
			objective=mission.objective,
			round_number=self._epoch_count,
			feedback_context=enriched_context,
			locked_files=locked_files,
		)

		# Resolve file overlaps
		units = resolve_file_overlaps(units)

		plan.status = "active"
		plan.total_units = len(units)
		epoch.units_planned = len(units)
		epoch.planner_cost_usd = planner_cost

		# Limit to max_units
		units = units[:max_units]

		logger.info(
			"Planned epoch %d: %d units",
			self._epoch_count, len(units),
		)

		return plan, units, epoch
