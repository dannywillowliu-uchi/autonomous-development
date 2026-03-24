"""Stagnation detection and pivot recommendation.

Monitors swarm progress across multiple metrics and suggests strategy
pivots when the swarm is not making progress.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StagnationConfig:
	"""Configurable thresholds for stagnation detection."""

	window: int = 3  # cycles to look back for flat metrics
	cost_rise_ratio: float = 1.2  # cost must exceed start * this to trigger
	failure_rate_threshold: float = 0.6  # fraction of failures to trigger
	repeated_error_min_agents: int = 2  # min agents sharing same error
	agent_churn_min_respawns: int = 2  # min respawns on same task to trigger
	cost_per_completion_threshold: float = 5.0  # max cost per task before flagging
	file_hotspot_threshold: int = 3  # min agents touching same file to flag


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
	*,
	config: StagnationConfig | None = None,
	error_messages: list[str] | None = None,
	task_agent_counts: dict[str, int] | None = None,
	file_changes: dict[str, list[str]] | None = None,
) -> list[PivotRecommendation]:
	"""Analyze metric histories and recommend pivots if stagnating.

	Args:
		cycle_number: Current planner cycle number.
		test_history: Test pass counts per cycle.
		completion_history: Task completions per cycle.
		failure_history: Task failures per cycle.
		cost_history: Cumulative cost per cycle.
		threshold: Legacy parameter -- overridden by config.window if config is provided.
		config: Configurable thresholds. Uses defaults if None.
		error_messages: Error messages from recent agent failures (for repeated-error detection).
		task_agent_counts: Map of task_id -> number of distinct agents that attempted it (for churn detection).
	"""
	cfg = config or StagnationConfig()
	window = cfg.window if config else threshold

	pivots: list[PivotRecommendation] = []

	# Flat test count
	if len(test_history) >= window:
		recent = test_history[-window:]
		if len(set(recent)) == 1 and recent[0] > 0:
			pivots.append(PivotRecommendation(
				trigger=f"Test pass count flat at {recent[0]} for {window} cycles",
				strategy="research_before_implement",
				severity="critical",
				details=(
					"Switch implementation agents to research mode. "
					"The current approach is not producing test improvements. "
					"Understand WHY tests are failing before trying more fixes."
				),
			))

	# Rising cost with flat completions
	if len(cost_history) >= window and len(completion_history) >= window:
		recent_cost = cost_history[-window:]
		recent_comp = completion_history[-window:]
		cost_rising = recent_cost[-1] > recent_cost[0] * cfg.cost_rise_ratio
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
		if total > 0 and recent_fail / total > cfg.failure_rate_threshold:
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

	# Repeated error messages across agents = systemic issue
	if error_messages:
		_check_repeated_errors(error_messages, cfg, pivots)

	# Agent churn: same task attempted by many agents = wrong approach
	if task_agent_counts:
		_check_agent_churn(task_agent_counts, cfg, pivots)

	# Predictive: declining cost efficiency
	if completion_history and cost_history:
		_check_cost_efficiency(completion_history, cost_history, cfg, pivots)

	# Predictive: file hotspots (merge conflict risk)
	if file_changes:
		_check_file_hotspots(file_changes, cfg, pivots)

	return pivots


def _check_repeated_errors(
	error_messages: list[str],
	cfg: StagnationConfig,
	pivots: list[PivotRecommendation],
) -> None:
	"""Detect the same error message repeated across multiple agents."""
	counts = Counter(error_messages)
	for msg, count in counts.most_common(3):
		if count >= cfg.repeated_error_min_agents:
			short_msg = msg[:120] + "..." if len(msg) > 120 else msg
			pivots.append(PivotRecommendation(
				trigger=f"Same error across {count} agents: {short_msg}",
				strategy="research_systemic_error",
				severity="critical",
				details=(
					f"The error \"{short_msg}\" appeared in {count} independent agents. "
					"This is a systemic issue, not a per-agent bug. "
					"Spawn a research agent to investigate the root cause. "
					"Consider environment issues, missing dependencies, or incorrect assumptions."
				),
			))


def _check_agent_churn(
	task_agent_counts: dict[str, int],
	cfg: StagnationConfig,
	pivots: list[PivotRecommendation],
) -> None:
	"""Detect tasks where agents keep getting killed and respawned."""
	churning = {
		task_id: count
		for task_id, count in task_agent_counts.items()
		if count >= cfg.agent_churn_min_respawns
	}
	if churning:
		worst_task = max(churning, key=churning.get)  # type: ignore[arg-type]
		worst_count = churning[worst_task]
		pivots.append(PivotRecommendation(
			trigger=f"Agent churn: task {worst_task} attempted by {worst_count} agents",
			strategy="rethink_approach",
			severity="warning" if worst_count < 4 else "critical",
			details=(
				f"Task {worst_task} has been attempted by {worst_count} different agents. "
				"Repeated agent turnover on the same task suggests the approach is wrong, "
				"not the execution. Pause this task and spawn a research agent to "
				"find an alternative approach before retrying."
			),
		))


def _check_cost_efficiency(
	completion_history: list[int],
	cost_history: list[float],
	cfg: StagnationConfig,
	pivots: list[PivotRecommendation],
) -> None:
	"""Detect declining cost efficiency as a leading stagnation indicator."""
	if len(completion_history) < 4 or len(cost_history) < 4:
		return
	mid = len(completion_history) // 2
	first_completions = sum(completion_history[:mid])
	second_completions = sum(completion_history[mid:])
	first_cost = sum(cost_history[:mid])
	second_cost = sum(cost_history[mid:])
	if first_cost <= 0 or second_cost <= 0:
		return
	first_efficiency = first_completions / first_cost
	second_efficiency = second_completions / second_cost
	if first_efficiency > 0 and second_efficiency < first_efficiency * 0.5:
		pivots.append(PivotRecommendation(
			trigger=f"Cost efficiency dropped from {first_efficiency:.2f} to {second_efficiency:.2f} tasks/$",
			strategy="reduce_and_focus",
			severity="warning",
			details="Reduce agent count and focus on highest-priority tasks",
		))


def _check_file_hotspots(
	file_changes: dict[str, list[str]],
	cfg: StagnationConfig,
	pivots: list[PivotRecommendation],
) -> None:
	"""Detect files touched by too many agents (merge conflict risk)."""
	file_to_agents: dict[str, set[str]] = {}
	for agent, files in file_changes.items():
		for f in files:
			file_to_agents.setdefault(f, set()).add(agent)
	hotspots = {f: agents for f, agents in file_to_agents.items() if len(agents) >= cfg.file_hotspot_threshold}
	if hotspots:
		ranked = sorted(hotspots.items(), key=lambda x: -len(x[1]))
		hotspot_list = ", ".join(f"{f} ({len(a)} agents)" for f, a in ranked)
		pivots.append(PivotRecommendation(
			trigger=f"File hotspots detected: {hotspot_list}",
			strategy="serialize_hotspot",
			severity="warning",
			details="Serialize work on contested files to avoid merge conflicts",
		))


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
		elif p.strategy == "research_systemic_error":
			decisions.append({
				"type": "create_task",
				"payload": {
					"title": f"Research systemic error: {p.trigger[:80]}",
					"description": p.details,
					"priority": 3,
				},
				"reasoning": f"Pivot: {p.trigger}",
				"priority": 10,
			})
		elif p.strategy == "rethink_approach":
			decisions.append({
				"type": "create_task",
				"payload": {
					"title": f"Rethink approach: {p.trigger[:80]}",
					"description": p.details,
					"priority": 2,
				},
				"reasoning": f"Pivot: {p.trigger}. Current approach is failing repeatedly.",
				"priority": 9,
			})
		elif p.strategy == "serialize_hotspot":
			decisions.append({
				"type": "adjust",
				"payload": {"max_agents": 1},
				"reasoning": f"Pivot: {p.trigger}. Serializing to avoid conflicts.",
				"priority": 9,
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
