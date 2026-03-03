"""Heuristic batch signal analysis -- pure DB queries, no LLM."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from mission_control.db import Database

log = logging.getLogger(__name__)

FAILURE_STAGES = frozenset({
	"acceptance_criteria",
	"verification",
	"merge_conflict",
	"timeout",
	"infrastructure",
})


@dataclass
class BatchSignals:
	"""Patterns detected from batch execution data."""

	file_hotspots: list[tuple[str, int]] = field(default_factory=list)
	failure_clusters: dict[str, int] = field(default_factory=dict)
	failure_stages: dict[str, int] = field(default_factory=dict)
	stalled_areas: list[str] = field(default_factory=list)
	effort_distribution: dict[str, float] = field(default_factory=dict)
	retry_depth: dict[str, int] = field(default_factory=dict)
	knowledge_gaps: list[str] = field(default_factory=list)


@dataclass
class EpochCostSummary:
	"""Per-epoch cost and throughput metrics."""

	epoch_id: str = ""
	total_cost_usd: float = 0.0
	total_input_tokens: int = 0
	total_output_tokens: int = 0
	units_dispatched: int = 0
	units_merged: int = 0
	units_failed: int = 0
	avg_cost_per_unit: float = 0.0
	success_rate: float = 0.0


class BatchAnalyzer:
	"""Analyzes execution patterns from DB data -- no LLM calls."""

	def __init__(self, db: Database) -> None:
		self._db = db

	def get_epoch_cost_summary(self, mission_id: str) -> list[EpochCostSummary]:
		"""Return per-epoch cost/throughput metrics from work_units and unit_events."""
		try:
			epochs = self._db.get_epochs_for_mission(mission_id)
		except Exception:
			return []
		if not epochs:
			return []

		try:
			units = self._db.get_work_units_for_mission(mission_id)
		except Exception:
			units = []
		try:
			events = self._db.get_unit_events_for_mission(mission_id, limit=10000)
		except Exception:
			events = []

		# Group work units by epoch_id
		units_by_epoch: dict[str, list] = defaultdict(list)
		for u in units:
			if u.epoch_id:
				units_by_epoch[u.epoch_id].append(u)

		# Count event types by epoch_id
		dispatched_by_epoch: Counter[str] = Counter()
		merged_by_epoch: Counter[str] = Counter()
		failed_by_epoch: Counter[str] = Counter()
		for ev in events:
			if ev.event_type == "dispatched":
				dispatched_by_epoch[ev.epoch_id] += 1
			elif ev.event_type == "merged":
				merged_by_epoch[ev.epoch_id] += 1
			elif ev.event_type in ("failed", "merge_failed"):
				failed_by_epoch[ev.epoch_id] += 1

		summaries: list[EpochCostSummary] = []
		for epoch in epochs:
			eid = epoch.id
			epoch_units = units_by_epoch.get(eid, [])
			planner_cost = getattr(epoch, "planner_cost_usd", 0.0) or 0.0
			total_cost = sum(u.cost_usd for u in epoch_units) + planner_cost
			total_input = sum(u.input_tokens for u in epoch_units)
			total_output = sum(u.output_tokens for u in epoch_units)
			dispatched = dispatched_by_epoch.get(eid, 0)
			merged = merged_by_epoch.get(eid, 0)
			failed = failed_by_epoch.get(eid, 0)
			avg_cost = total_cost / dispatched if dispatched > 0 else 0.0
			success = merged / dispatched if dispatched > 0 else 0.0

			summaries.append(EpochCostSummary(
				epoch_id=eid,
				total_cost_usd=round(total_cost, 4),
				total_input_tokens=total_input,
				total_output_tokens=total_output,
				units_dispatched=dispatched,
				units_merged=merged,
				units_failed=failed,
				avg_cost_per_unit=round(avg_cost, 4),
				success_rate=round(success, 2),
			))

		return summaries

	def analyze(self, mission_id: str) -> BatchSignals:
		"""Compute heuristic signals from mission execution data."""
		try:
			units = self._db.get_work_units_for_mission(mission_id)
		except Exception:
			units = []
		try:
			handoffs = self._db.get_recent_handoffs(mission_id, limit=50)
		except Exception:
			handoffs = []
		try:
			knowledge = self._db.get_knowledge_for_mission(mission_id)
		except Exception:
			knowledge = []
		try:
			events = self._db.get_unit_events_for_mission(mission_id)
		except Exception:
			events = []

		return BatchSignals(
			file_hotspots=self._compute_hotspots(handoffs),
			failure_clusters=self._cluster_failures(units, handoffs),
			failure_stages=self._compute_failure_stages(events),
			stalled_areas=self._find_stalled(units),
			effort_distribution=self._compute_effort(units),
			retry_depth=self._compute_retries(units),
			knowledge_gaps=self._find_gaps(units, knowledge),
		)

	def _compute_hotspots(self, handoffs: list) -> list[tuple[str, int]]:
		"""Files touched by 3+ units."""
		file_counts: Counter[str] = Counter()
		for h in handoffs:
			for f in (h.files_changed or []):
				file_counts[f] += 1
		return [(f, c) for f, c in file_counts.most_common() if c >= 3]

	def _compute_failure_stages(self, events: list) -> dict[str, int]:
		"""Count merge_failed events by failure_stage from details JSON."""
		counts: Counter[str] = Counter()
		for ev in events:
			if ev.event_type != "merge_failed":
				continue
			stage = self._extract_failure_stage(ev.details)
			if stage:
				counts[stage] += 1
		return dict(counts.most_common())

	@staticmethod
	def _extract_failure_stage(details: str) -> str:
		"""Parse failure_stage from a details JSON string, normalizing to known stages."""
		if not details:
			return ""
		try:
			parsed = json.loads(details)
		except (json.JSONDecodeError, TypeError):
			return ""
		raw = parsed.get("failure_stage", "")
		if not raw:
			return ""
		# Normalize known aliases to canonical stage names
		if raw in FAILURE_STAGES:
			return raw
		if raw in ("pre_merge_verification", "post_merge_verification"):
			return "verification"
		if raw in ("fetch", "exception"):
			return "infrastructure"
		if raw == "execution":
			return "infrastructure"
		return raw

	def _cluster_failures(self, units: list, handoffs: list) -> dict[str, int]:
		"""Group failures by area/pattern."""
		clusters: Counter[str] = Counter()
		failed_unit_ids = {u.id for u in units if u.status == "failed"}
		for h in handoffs:
			if h.work_unit_id in failed_unit_ids or h.status != "completed":
				for concern in (h.concerns or []):
					key = concern[:80]
					clusters[key] += 1
		return dict(clusters.most_common(10))

	def _find_stalled(self, units: list) -> list[str]:
		"""Areas attempted 2+ times without success."""
		area_attempts: dict[str, int] = {}
		area_succeeded: set[str] = set()
		for u in units:
			area = u.files_hint or u.title
			area_attempts[area] = area_attempts.get(area, 0) + 1
			if u.status == "completed":
				area_succeeded.add(area)
		return [
			area for area, count in area_attempts.items()
			if count >= 2 and area not in area_succeeded
		]

	def _compute_effort(self, units: list) -> dict[str, float]:
		"""Area -> percentage of total effort."""
		total_cost = sum(u.cost_usd for u in units) or 1.0
		area_cost: dict[str, float] = {}
		for u in units:
			area = u.files_hint or u.title
			area_cost[area] = area_cost.get(area, 0.0) + u.cost_usd
		return {area: round(cost / total_cost, 2) for area, cost in area_cost.items()}

	def _compute_retries(self, units: list) -> dict[str, int]:
		"""Unit ID -> attempt count for retried units."""
		return {u.id: u.attempt for u in units if u.attempt > 0}

	def _find_gaps(self, units: list, knowledge: list) -> list[str]:
		"""Areas where research scope doesn't match implementation."""
		research_scopes = {k.scope for k in knowledge if k.source_unit_type == "research"}
		implementation_areas = set()
		for u in units:
			if u.unit_type in ("implementation", "code") and u.files_hint:
				for f in u.files_hint.split(","):
					f = f.strip()
					if f:
						implementation_areas.add(f)
		gaps = implementation_areas - research_scopes
		return sorted(gaps)[:10]


def format_failure_stages(stages: dict[str, int]) -> str:
	"""Format failure_stages dict into a planner-readable summary.

	Returns a multi-line string describing each failure stage and its count,
	or an empty string if there are no failure stages.
	"""
	if not stages:
		return ""
	total = sum(stages.values())
	lines = [f"Failure breakdown ({total} total):"]
	for stage, count in sorted(stages.items(), key=lambda x: -x[1]):
		lines.append(f"  - {stage}: {count}")
	return "\n".join(lines)


def format_cost_trend(summaries: list[EpochCostSummary], last_n: int = 5) -> str:
	"""Render the last N epoch cost summaries as a compact markdown table.

	Returns an empty string if summaries is empty.
	"""
	if not summaries:
		return ""
	recent = summaries[-last_n:]
	lines = [
		"| Epoch | Cost ($) | In Tokens | Out Tokens | Dispatched | Merged | Failed | Avg $/Unit | Success % |",
		"|-------|----------|-----------|------------|------------|--------|--------|------------|-----------|",
	]
	for s in recent:
		pct = int(s.success_rate * 100)
		lines.append(
			f"| {s.epoch_id[:8]} | {s.total_cost_usd:.2f} | {s.total_input_tokens:,} "
			f"| {s.total_output_tokens:,} | {s.units_dispatched} | {s.units_merged} "
			f"| {s.units_failed} | {s.avg_cost_per_unit:.2f} | {pct}% |"
		)
	return "\n".join(lines)
