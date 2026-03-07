"""Stagnation detection and pivot recommendation.

Monitors swarm progress across multiple metrics and suggests strategy
pivots when the swarm is not making progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PivotRecommendation:
	"""A recommended strategy change when stagnation is detected."""

	trigger: str  # what triggered the pivot
	strategy: str  # what to do about it
	severity: str  # "warning" | "critical"
	details: str = ""


def analyze_stagnation(
	cycle_number: int,
	test_history: list[int],
	completion_history: list[int],
	failure_history: list[int],
	cost_history: list[float],
	threshold: int = 3,
) -> list[PivotRecommendation]:
	"""Analyze metric histories and recommend pivots if stagnating."""
	pivots: list[PivotRecommendation] = []

	# Flat test count
	if len(test_history) >= threshold:
		recent = test_history[-threshold:]
		if len(set(recent)) == 1 and recent[0] > 0:
			pivots.append(PivotRecommendation(
				trigger=f"Test pass count flat at {recent[0]} for {threshold} cycles",
				strategy="research_before_implement",
				severity="critical",
				details=(
					"Switch implementation agents to research mode. "
					"The current approach is not producing test improvements. "
					"Understand WHY tests are failing before trying more fixes."
				),
			))

	# Rising cost with flat completions
	if len(cost_history) >= threshold and len(completion_history) >= threshold:
		recent_cost = cost_history[-threshold:]
		recent_comp = completion_history[-threshold:]
		cost_rising = recent_cost[-1] > recent_cost[0] * 1.2
		comp_flat = len(set(recent_comp)) <= 2
		if cost_rising and comp_flat:
			pivots.append(PivotRecommendation(
				trigger="Cost rising but completions flat",
				strategy="reduce_and_focus",
				severity="warning",
				details=(
					"Diminishing returns detected. Reduce agent count "
					"and focus remaining agents on fewer, higher-impact tasks."
				),
			))

	# High failure rate
	if len(failure_history) >= 2 and len(completion_history) >= 2:
		recent_fail = sum(failure_history[-3:])
		recent_comp = sum(completion_history[-3:])
		total = recent_fail + recent_comp
		if total > 0 and recent_fail / total > 0.6:
			pivots.append(PivotRecommendation(
				trigger=f"Failure rate {recent_fail}/{total} ({recent_fail/total:.0%})",
				strategy="diagnose_systemic",
				severity="critical",
				details=(
					"Most tasks are failing. This suggests a systemic issue, "
					"not per-task bugs. Spawn a diagnostic agent to find the "
					"root cause before continuing implementation."
				),
			))

	# Repeated failures on same task
	# (this would need task-level history, handled in context.py)

	return pivots


def pivots_to_decisions(pivots: list[PivotRecommendation]) -> list[dict]:
	"""Convert pivot recommendations into planner decision suggestions.

	These are injected into the planner prompt as suggested decisions,
	not executed directly. The planner decides whether to adopt them.
	"""
	decisions: list[dict] = []
	for p in pivots:
		if p.strategy == "research_before_implement":
			decisions.append({
				"type": "create_task",
				"payload": {
					"title": f"Research: {p.trigger}",
					"description": p.details,
					"priority": 3,
				},
				"reasoning": f"Pivot: {p.trigger}",
				"priority": 10,
			})
		elif p.strategy == "reduce_and_focus":
			decisions.append({
				"type": "adjust",
				"payload": {"max_agents": 2},
				"reasoning": f"Pivot: {p.trigger}. Reducing parallelism.",
				"priority": 8,
			})
		elif p.strategy == "diagnose_systemic":
			decisions.append({
				"type": "create_task",
				"payload": {
					"title": "Diagnose systemic failure pattern",
					"description": p.details,
					"priority": 3,
				},
				"reasoning": f"Pivot: {p.trigger}",
				"priority": 10,
			})
	return decisions


def format_pivots_for_planner(pivots: list[PivotRecommendation]) -> str:
	"""Format pivot recommendations as text for the planner prompt."""
	if not pivots:
		return ""

	lines = ["## PIVOT RECOMMENDATIONS (action required)\n"]
	for p in pivots:
		icon = "!!!" if p.severity == "critical" else "!"
		lines.append(f"{icon} **{p.trigger}**")
		lines.append(f"   Strategy: {p.strategy}")
		lines.append(f"   {p.details}")
		lines.append("")
	return "\n".join(lines)
