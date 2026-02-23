"""Continuous planner -- rolling backlog wrapper around RecursivePlanner."""

from __future__ import annotations

import logging

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import BacklogItem, Epoch, Handoff, Mission, Plan, WorkUnit
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
		self._unit_to_backlog: dict[str, str] = {}
		self._backlog_items: list[BacklogItem] = []

	def set_causal_context(self, risks: str) -> None:
		"""Set causal risk factors, delegating to the inner planner."""
		self._inner.set_causal_context(risks)

	def set_project_snapshot(self, snapshot: str) -> None:
		"""Set project structure snapshot, delegating to the inner planner."""
		self._inner.set_project_snapshot(snapshot)

	def ingest_handoff(self, handoff: Handoff) -> None:
		"""Accumulate discoveries and concerns from worker feedback."""
		if handoff.discoveries:
			self._discoveries.extend(handoff.discoveries)
		if handoff.concerns:
			self._concerns.extend(handoff.concerns)

	def set_backlog_items(self, items: list[BacklogItem]) -> None:
		"""Store backlog items as context for planning prompts."""
		self._backlog_items = list(items)

	@property
	def backlog_size(self) -> int:
		return len(self._backlog)

	def get_backlog_mapping(self) -> dict[str, str]:
		"""Return a copy of the unit-to-backlog-item mapping."""
		return dict(self._unit_to_backlog)

	async def get_next_units(
		self,
		mission: Mission,
		max_units: int = 3,
		feedback_context: str = "",
		backlog_item_ids: list[str] | None = None,
		decomposition_feedback: str = "",
		locked_files: dict[str, list[str]] | None = None,
	) -> tuple[Plan, list[WorkUnit], Epoch]:
		"""Return next work units from backlog, replanning if needed.

		If the backlog has enough units, returns from it without an LLM call.
		Otherwise, invokes the planner to generate a new batch.

		Args:
			mission: The current mission.
			max_units: Maximum units to return.
			feedback_context: Context from recent worker feedback.
			backlog_item_ids: Optional backlog item IDs being worked on.
			decomposition_feedback: Quality feedback from previous epoch grading.

		Returns:
			(plan, units, epoch) -- the plan, selected work units, and the epoch.
		"""
		min_size = self._config.continuous.backlog_min_size

		# Merge decomposition feedback into the feedback context
		if decomposition_feedback:
			feedback_context = (
				(feedback_context + "\n\n" + decomposition_feedback)
				if feedback_context
				else decomposition_feedback
			)

		if len(self._backlog) < min_size:
			# Need to replan
			plan, units, epoch = await self._replan(
				mission, max_units, feedback_context, backlog_item_ids,
				locked_files=locked_files,
			)
			# Empty replan + empty backlog = objective complete
			if not units and not self._backlog:
				return plan, [], epoch
			# If replan returned nothing but backlog has items, serve from backlog
			if not units and self._backlog:
				serve_count = min(max_units, len(self._backlog))
				units = self._backlog[:serve_count]
				self._backlog = self._backlog[serve_count:]
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
		backlog_item_ids: list[str] | None = None,
		locked_files: dict[str, list[str]] | None = None,
	) -> tuple[Plan, list[WorkUnit], Epoch]:
		"""Invoke the planner LLM to generate new work units."""
		self._epoch_count += 1
		from mission_control.snapshot import clear_snapshot_cache

		clear_snapshot_cache()
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

		# Add backlog item context if IDs provided
		if backlog_item_ids:
			backlog_section = "\nBacklog items being worked on:\n" + "\n".join(
				f"- {bid}" for bid in backlog_item_ids
			)
			enriched_context = (enriched_context + backlog_section) if enriched_context else backlog_section

		# Add rich backlog item details from set_backlog_items()
		if self._backlog_items:
			items_section = "\nPriority backlog items for this mission:\n" + "\n".join(
				f"- [{item.track}] {item.title} (priority={item.priority_score:.1f}): {item.description}"
				for item in self._backlog_items
			)
			enriched_context = (enriched_context + items_section) if enriched_context else items_section

		plan, root_node = await self._inner.plan_round(
			objective=mission.objective,
			snapshot_hash="",  # continuous mode doesn't use snapshot hash
			prior_discoveries=curated_discoveries,
			round_number=self._epoch_count,
			feedback_context=enriched_context,
			locked_files=locked_files,
		)

		# Extract work units from the plan tree
		units = self._extract_units_from_tree(root_node)

		# Post-planning: detect file overlaps and inject dependency edges
		from mission_control.overlap import resolve_file_overlaps
		units = resolve_file_overlaps(units)

		# Track unit-to-backlog-item mapping
		if backlog_item_ids:
			joined_ids = ",".join(backlog_item_ids)
			for unit in units:
				self._unit_to_backlog[unit.id] = joined_ids

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
