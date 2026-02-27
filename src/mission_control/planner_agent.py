"""Planner agent -- wraps RecursivePlanner with critic-enriched context."""

from __future__ import annotations

import logging

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import CriticFinding, Plan, WorkUnit
from mission_control.overlap import resolve_file_overlaps
from mission_control.recursive_planner import RecursivePlanner

log = logging.getLogger(__name__)


def _format_critic_context(finding: CriticFinding) -> str:
	"""Format critic findings as structured context for the planner prompt."""
	sections: list[str] = []

	if finding.strategy_text:
		sections.append(f"## Critic Strategy\n{finding.strategy_text}")

	if finding.findings:
		items = "\n".join(f"- {f}" for f in finding.findings)
		sections.append(f"## Key Findings\n{items}")

	if finding.risks:
		items = "\n".join(f"- {r}" for r in finding.risks)
		sections.append(f"## Risks\n{items}")

	if finding.gaps:
		items = "\n".join(f"- {g}" for g in finding.gaps)
		sections.append(f"## Knowledge Gaps\n{items}")

	if finding.open_questions:
		items = "\n".join(f"- {q}" for q in finding.open_questions)
		sections.append(f"## Open Questions\n{items}")

	return "\n\n".join(sections)


class PlannerAgent:
	"""Wraps RecursivePlanner, injecting critic findings into the planning context."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self._inner = RecursivePlanner(config, db)
		self._config = config
		self._db = db

	def set_causal_context(self, risks: str) -> None:
		self._inner.set_causal_context(risks)

	def set_project_snapshot(self, snapshot: str) -> None:
		self._inner.set_project_snapshot(snapshot)

	async def decompose(
		self,
		objective: str,
		critic_finding: CriticFinding,
		feedback_context: str = "",
		round_number: int = 1,
	) -> tuple[Plan, list[WorkUnit]]:
		"""Decompose objective into work units, enriched by critic findings."""
		critic_context = _format_critic_context(critic_finding)

		enriched_feedback = feedback_context
		if critic_context:
			enriched_feedback = (
				(enriched_feedback + "\n\n" + critic_context)
				if enriched_feedback
				else critic_context
			)

		plan, root_node = await self._inner.plan_round(
			objective=objective,
			snapshot_hash="",
			prior_discoveries=[],
			round_number=round_number,
			feedback_context=enriched_feedback,
		)

		units = self._extract_units(root_node)
		units = resolve_file_overlaps(units)

		plan.status = "active"
		plan.total_units = len(units)

		log.info("Planner decomposed into %d units", len(units))
		return plan, units

	async def refine(
		self,
		objective: str,
		units: list[WorkUnit],
		critic_feedback: CriticFinding,
		feedback_context: str = "",
		round_number: int = 1,
	) -> tuple[Plan, list[WorkUnit]]:
		"""Refine existing units based on critic feedback.

		The critic's gaps and risks become additional context for the planner
		to produce an improved decomposition.
		"""
		refinement_context = _format_critic_context(critic_feedback)

		# Include summaries of current units so planner knows what to improve
		current_units_text = "\n".join(
			f"- {u.title}: {u.description[:100]} (files: {u.files_hint or 'unspecified'})"
			for u in units
		)
		refinement_context += (
			f"\n\n## Previous Plan (needs refinement)\n{current_units_text}"
		)

		enriched_feedback = feedback_context
		if refinement_context:
			enriched_feedback = (
				(enriched_feedback + "\n\n" + refinement_context)
				if enriched_feedback
				else refinement_context
			)

		plan, root_node = await self._inner.plan_round(
			objective=objective,
			snapshot_hash="",
			prior_discoveries=[],
			round_number=round_number,
			feedback_context=enriched_feedback,
		)

		new_units = self._extract_units(root_node)
		new_units = resolve_file_overlaps(new_units)

		plan.status = "active"
		plan.total_units = len(new_units)

		log.info("Planner refined to %d units", len(new_units))
		return plan, new_units

	def _extract_units(self, node: object) -> list[WorkUnit]:
		"""Extract WorkUnit objects from the in-memory plan tree."""
		units: list[WorkUnit] = []

		if hasattr(node, "_forced_unit"):
			units.append(node._forced_unit)  # type: ignore[union-attr]

		if hasattr(node, "_child_leaves"):
			for _leaf, wu in node._child_leaves:  # type: ignore[union-attr]
				units.append(wu)

		if hasattr(node, "_subdivided_children"):
			for child in node._subdivided_children:  # type: ignore[union-attr]
				units.extend(self._extract_units(child))

		return units
