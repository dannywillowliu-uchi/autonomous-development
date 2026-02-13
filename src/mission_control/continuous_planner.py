"""Continuous planner -- rolling backlog wrapper around RecursivePlanner."""

from __future__ import annotations

import json
import logging

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import Epoch, Handoff, Mission, Plan, WorkUnit
from mission_control.recursive_planner import RecursivePlanner

log = logging.getLogger(__name__)


class ContinuousPlanner:
	"""Wraps RecursivePlanner to produce small batches of work units on-demand.

	Instead of planning an entire round up-front, this planner maintains a
	rolling backlog. When the backlog runs low, it invokes the LLM to generate
	a small batch of 1-3 new units, incorporating discoveries from recent
	worker feedback.
	"""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self._inner = RecursivePlanner(config, db)
		self._config = config
		self._db = db
		self._backlog: list[WorkUnit] = []
		self._discoveries: list[str] = []
		self._concerns: list[str] = []
		self._epoch_count: int = 0

	def ingest_handoff(self, handoff: Handoff) -> None:
		"""Accumulate discoveries and concerns from worker feedback."""
		if handoff.discoveries:
			try:
				disc_list = json.loads(handoff.discoveries)
				if isinstance(disc_list, list):
					self._discoveries.extend(disc_list)
			except (json.JSONDecodeError, TypeError):
				pass
		if handoff.concerns:
			try:
				conc_list = json.loads(handoff.concerns)
				if isinstance(conc_list, list):
					self._concerns.extend(conc_list)
			except (json.JSONDecodeError, TypeError):
				pass

	@property
	def backlog_size(self) -> int:
		return len(self._backlog)

	async def get_next_units(
		self,
		mission: Mission,
		max_units: int = 3,
		feedback_context: str = "",
	) -> tuple[Plan, list[WorkUnit], Epoch]:
		"""Return next work units from backlog, replanning if needed.

		If the backlog has enough units, returns from it without an LLM call.
		Otherwise, invokes the planner to generate a new batch.

		Returns:
			(plan, units, epoch) -- the plan, selected work units, and the epoch.
		"""
		min_size = self._config.continuous.backlog_min_size

		if len(self._backlog) < min_size:
			# Need to replan
			plan, units, epoch = await self._replan(
				mission, max_units, feedback_context,
			)
			return plan, units, epoch

		# Serve from existing backlog
		serve_count = min(max_units, len(self._backlog))
		units = self._backlog[:serve_count]
		self._backlog = self._backlog[serve_count:]

		# Create a lightweight plan record for these units
		plan = Plan(
			objective=mission.objective,
			status="active",
			total_units=serve_count,
		)

		# Use the current epoch
		epoch = Epoch(
			mission_id=mission.id,
			number=self._epoch_count,
		)

		return plan, units, epoch

	async def _replan(
		self,
		mission: Mission,
		max_units: int,
		feedback_context: str,
	) -> tuple[Plan, list[WorkUnit], Epoch]:
		"""Invoke the planner LLM to generate new work units."""
		self._epoch_count += 1
		epoch = Epoch(
			mission_id=mission.id,
			number=self._epoch_count,
		)

		# Build discovery context from accumulated feedback
		curated_discoveries = self._discoveries[-20:]  # last 20

		# Build concern context
		concern_text = ""
		if self._concerns:
			concern_items = self._concerns[-10:]
			concern_text = "\nConcerns from recent work:\n" + "\n".join(
				f"- {c}" for c in concern_items
			)

		enriched_context = feedback_context
		if concern_text:
			enriched_context = (feedback_context + concern_text) if feedback_context else concern_text

		plan, root_node = await self._inner.plan_round(
			objective=mission.objective,
			snapshot_hash="",  # continuous mode doesn't use snapshot hash
			prior_discoveries=curated_discoveries,
			round_number=self._epoch_count,
			feedback_context=enriched_context,
		)

		# Extract work units from the plan tree
		units = self._extract_units_from_tree(root_node)

		# Post-planning: detect file overlaps and inject dependency edges
		from mission_control.overlap import resolve_file_overlaps
		units = resolve_file_overlaps(units)

		plan.status = "active"
		plan.total_units = len(units)

		# Split: serve up to max_units, rest goes to backlog
		serve_count = min(max_units, len(units))
		serve_units = units[:serve_count]
		self._backlog.extend(units[serve_count:])

		epoch.units_planned = len(units)

		log.info(
			"Replanned epoch %d: %d units (%d served, %d backlogged)",
			self._epoch_count, len(units), serve_count, len(units) - serve_count,
		)

		return plan, serve_units, epoch

	def _extract_units_from_tree(self, node: object) -> list[WorkUnit]:
		"""Extract WorkUnit objects from the in-memory plan tree."""
		units: list[WorkUnit] = []

		# Check for forced unit (leaf at max depth)
		if hasattr(node, "_forced_unit"):
			units.append(node._forced_unit)  # type: ignore[union-attr]

		# Check for child leaves
		if hasattr(node, "_child_leaves"):
			for _leaf, wu in node._child_leaves:  # type: ignore[union-attr]
				units.append(wu)

		# Recurse into subdivided children
		if hasattr(node, "_subdivided_children"):
			for child in node._subdivided_children:  # type: ignore[union-attr]
				units.extend(self._extract_units_from_tree(child))

		return units
