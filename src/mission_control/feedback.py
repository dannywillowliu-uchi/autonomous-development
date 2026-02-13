"""Feedback system -- record outcomes, compute rewards, retrieve experiences."""

from __future__ import annotations

import json
import logging
import re

from mission_control.db import Database
from mission_control.green_branch import FixupResult
from mission_control.models import (
	Epoch,
	Experience,
	Handoff,
	Plan,
	Reflection,
	Reward,
	Round,
	Snapshot,
	UnitEvent,
	WorkUnit,
)

log = logging.getLogger(__name__)


def _compute_verification_improvement(
	snapshot_before: Snapshot | None,
	snapshot_after: Snapshot | None,
) -> tuple[float, int, int, int, int, int]:
	"""Compute verification improvement score and deltas from snapshots.

	Returns (score, tests_before, tests_after, tests_delta, lint_delta, type_delta).
	"""
	if snapshot_before is None or snapshot_after is None:
		return 0.5, 0, 0, 0, 0, 0

	tests_before = snapshot_before.test_passed
	tests_after = snapshot_after.test_passed
	tests_delta = tests_after - tests_before
	lint_delta = snapshot_after.lint_errors - snapshot_before.lint_errors
	type_delta = snapshot_after.type_errors - snapshot_before.type_errors
	security_delta = snapshot_after.security_findings - snapshot_before.security_findings

	# Regression check
	tests_broken = snapshot_after.test_failed - snapshot_before.test_failed
	if tests_broken > 0 or security_delta > 0:
		return 0.0, tests_before, tests_after, tests_delta, lint_delta, type_delta

	# Improvement: tests gained and no regressions in lint/type/security
	if tests_delta > 0 and lint_delta <= 0 and type_delta <= 0:
		return 1.0, tests_before, tests_after, tests_delta, lint_delta, type_delta

	# No regression but no strong improvement
	if tests_delta >= 0:
		return 0.7, tests_before, tests_after, tests_delta, lint_delta, type_delta

	return 0.0, tests_before, tests_after, tests_delta, lint_delta, type_delta


def _compute_no_regression(
	snapshot_before: Snapshot | None,
	snapshot_after: Snapshot | None,
) -> float:
	"""1.0 if no tests broken and no security regressions, else 0.0."""
	if snapshot_before is None or snapshot_after is None:
		return 0.5
	tests_delta = snapshot_after.test_passed - snapshot_before.test_passed
	security_delta = snapshot_after.security_findings - snapshot_before.security_findings
	if tests_delta >= 0 and security_delta <= 0:
		return 1.0
	return 0.0


def compute_reward(
	reflection: Reflection,
	prev_score: float,
	snapshot_before: Snapshot | None = None,
	snapshot_after: Snapshot | None = None,
) -> Reward:
	"""Compute composite reward from objective signals only.

	Weights:
	  0.30 * verification_improvement
	  0.25 * completion_rate
	  0.25 * score_progress
	  0.10 * fixup_efficiency
	  0.10 * no_regression
	"""
	# Verification improvement
	vi_score = _compute_verification_improvement(snapshot_before, snapshot_after)[0]

	# Completion rate
	cr = reflection.completion_rate

	# Score progress (normalize: 0.3 expected progress per round)
	score_delta = reflection.objective_score - prev_score
	sp = max(0.0, min(1.0, score_delta / 0.3))

	# Fixup efficiency
	if reflection.fixup_promoted and reflection.fixup_attempts <= 1:
		fe = 1.0
	elif reflection.fixup_promoted:
		fe = 0.5
	else:
		fe = 0.0

	# No regression
	nr = _compute_no_regression(snapshot_before, snapshot_after)

	total = 0.30 * vi_score + 0.25 * cr + 0.25 * sp + 0.10 * fe + 0.10 * nr

	return Reward(
		round_id=reflection.round_id,
		mission_id=reflection.mission_id,
		reward=total,
		verification_improvement=vi_score,
		completion_rate=cr,
		score_progress=sp,
		fixup_efficiency=fe,
		no_regression=nr,
	)


def _get_plan_depth(db: Database, plan_id: str | None) -> int:
	"""Get the max depth of a plan tree."""
	if not plan_id:
		return 0
	nodes = db.get_plan_nodes_for_plan(plan_id)
	if not nodes:
		return 0
	return max(n.depth for n in nodes)


def _get_root_strategy(db: Database, plan: Plan) -> str:
	"""Get the strategy used at the root node."""
	if not plan.root_node_id:
		return ""
	root = db.get_plan_node(plan.root_node_id)
	if root is None:
		return ""
	return root.strategy


def _count_merge_conflicts(handoffs: list[Handoff], units: list[WorkUnit]) -> int:
	"""Count units that failed due to merge conflicts."""
	count = 0
	for u in units:
		if u.status == "failed" and "merge conflict" in u.output_summary.lower():
			count += 1
	return count


def record_round_outcome(
	db: Database,
	mission_id: str,
	rnd: Round,
	plan: Plan,
	handoffs: list[Handoff],
	fixup_result: FixupResult,
	snapshot_before: Snapshot | None,
	snapshot_after: Snapshot | None,
	prev_score: float,
) -> Reward:
	"""Record reflection + reward + experiences after a round completes."""
	units = db.get_work_units_for_plan(plan.id)

	# Count discoveries across handoffs
	disc_count = 0
	for h in handoffs:
		if h.discoveries:
			try:
				disc_count += len(json.loads(h.discoveries))
			except (json.JSONDecodeError, TypeError):
				pass

	# Build verification deltas
	vi_result = _compute_verification_improvement(snapshot_before, snapshot_after)
	_, tests_before, tests_after, tests_delta, lint_delta, type_delta = vi_result

	units_planned = plan.total_units
	units_completed = sum(1 for u in units if u.status == "completed")
	units_failed = sum(1 for u in units if u.status == "failed")
	completion_rate = units_completed / max(units_planned, 1)

	reflection = Reflection(
		mission_id=mission_id,
		round_id=rnd.id,
		round_number=rnd.number,
		tests_before=tests_before,
		tests_after=tests_after,
		tests_delta=tests_delta,
		lint_delta=lint_delta,
		type_delta=type_delta,
		objective_score=rnd.objective_score,
		score_delta=rnd.objective_score - prev_score,
		units_planned=units_planned,
		units_completed=units_completed,
		units_failed=units_failed,
		completion_rate=completion_rate,
		plan_depth=_get_plan_depth(db, plan.id),
		plan_strategy=_get_root_strategy(db, plan),
		fixup_promoted=fixup_result.promoted,
		fixup_attempts=fixup_result.fixup_attempts,
		merge_conflicts=_count_merge_conflicts(handoffs, units),
		discoveries_count=disc_count,
	)
	db.insert_reflection(reflection)

	# Compute reward
	reward = compute_reward(reflection, prev_score, snapshot_before, snapshot_after)
	db.insert_reward(reward)

	# Extract experiences from completed handoffs
	for h in handoffs:
		unit = next((u for u in units if u.id == h.work_unit_id), None)
		if unit is None:
			continue
		exp = Experience(
			round_id=rnd.id,
			work_unit_id=h.work_unit_id,
			title=unit.title,
			scope=unit.description,
			files_hint=unit.files_hint,
			status=h.status,
			summary=h.summary,
			files_changed=h.files_changed,
			discoveries=h.discoveries,
			concerns=h.concerns,
			reward=reward.reward,
		)
		db.insert_experience(exp)

	log.info(
		"Round %d feedback: reward=%.3f (vi=%.2f cr=%.2f sp=%.2f fe=%.2f nr=%.2f)",
		rnd.number, reward.reward,
		reward.verification_improvement, reward.completion_rate,
		reward.score_progress, reward.fixup_efficiency, reward.no_regression,
	)

	return reward


def get_planner_context(
	db: Database,
	mission_id: str,
) -> str:
	"""Build context string for the planner from past reflections."""
	reflections = db.get_recent_reflections(mission_id, limit=5)
	if not reflections:
		return ""

	lines: list[str] = []

	# Score trajectory
	scores = [(r.round_number, r.objective_score) for r in reversed(reflections)]
	if scores:
		trajectory = ", ".join(f"R{n}={s:.2f}" for n, s in scores)
		lines.append(f"Score trajectory: {trajectory}")

	# Strategy success rates
	strategies: dict[str, list[float]] = {}
	for r in reflections:
		if r.plan_strategy:
			strategies.setdefault(r.plan_strategy, []).append(r.completion_rate)
	for strat, rates in strategies.items():
		avg = sum(rates) / len(rates)
		lines.append(f"Strategy '{strat}': avg completion {avg:.0%} ({len(rates)} rounds)")

	# Merge conflict patterns
	conflict_rounds = [r for r in reflections if r.merge_conflicts > 0]
	if conflict_rounds:
		lines.append(
			f"Merge conflicts in {len(conflict_rounds)}/{len(reflections)} recent rounds"
		)

	# Fixup success rate
	fixup_rounds = [r for r in reflections if r.fixup_promoted]
	lines.append(
		f"Fixup promoted: {len(fixup_rounds)}/{len(reflections)} recent rounds"
	)

	# High-value discoveries from top-rewarded experiences
	top_experiences = db.get_top_experiences(limit=10)
	discoveries_from_successes: list[str] = []
	for exp in top_experiences:
		if exp.discoveries:
			try:
				disc_list = json.loads(exp.discoveries)
				for d in disc_list:
					if d not in discoveries_from_successes:
						discoveries_from_successes.append(d)
						if len(discoveries_from_successes) >= 5:
							break
			except (json.JSONDecodeError, TypeError):
				pass
		if len(discoveries_from_successes) >= 5:
			break

	if discoveries_from_successes:
		lines.append("\nKey insights from successful past work:")
		for d in discoveries_from_successes:
			lines.append(f"- {d}")

	return "\n".join(lines)


def _extract_keywords(text: str) -> list[str]:
	"""Extract meaningful keywords from text for experience search."""
	# Split on non-alphanumeric, filter short/common words
	stop_words = {
		"the", "and", "for", "that", "this", "with", "from", "are", "was",
		"will", "have", "has", "been", "not", "but", "can", "all", "its",
		"add", "fix", "update", "implement", "create", "make", "use",
	}
	words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
	return [w for w in words if len(w) > 2 and w not in stop_words]


def get_worker_context(
	db: Database,
	unit: WorkUnit,
) -> str:
	"""Find relevant past experiences for a work unit."""
	keywords = _extract_keywords(f"{unit.title} {unit.description} {unit.files_hint}")
	if not keywords:
		return ""

	# Take top keywords to avoid overly broad search
	search_keywords = keywords[:8]
	experiences = db.search_experiences(search_keywords, limit=3)
	if not experiences:
		return ""

	lines: list[str] = []
	for exp in experiences:
		if exp.status == "completed":
			lines.append(f"- [{exp.title}] succeeded (reward={exp.reward:.2f}): {exp.summary}")
			if exp.discoveries:
				try:
					discoveries = json.loads(exp.discoveries)
					if discoveries:
						lines.append(f"  Insights: {', '.join(discoveries[:2])}")
				except (json.JSONDecodeError, TypeError):
					pass
			if exp.concerns:
				try:
					concerns = json.loads(exp.concerns)
					if concerns:
						lines.append(f"  Pitfalls: {', '.join(concerns[:2])}")
				except (json.JSONDecodeError, TypeError):
					pass
		else:
			lines.append(f"- [{exp.title}] FAILED: {exp.summary}")

	return "\n".join(lines)


def _ensure_continuous_round(
	db: Database,
	mission_id: str,
	epoch: Epoch,
) -> str:
	"""Ensure a virtual round exists for the epoch (FK bridge).

	Continuous mode doesn't use rounds, but reflections/rewards have a
	NOT NULL FK to rounds. We create one lightweight sentinel round per
	epoch to satisfy the constraint.
	"""
	virtual_id = f"continuous-{epoch.id}"
	existing = db.get_round(virtual_id)
	if existing:
		return virtual_id

	from mission_control.models import Round
	rnd = Round(
		id=virtual_id,
		mission_id=mission_id,
		number=epoch.number,
		status="continuous",
	)
	db.insert_round(rnd)
	return virtual_id


def record_unit_outcome(
	db: Database,
	mission_id: str,
	epoch: Epoch,
	unit: WorkUnit,
	handoff: Handoff | None,
	snapshot_before: Snapshot | None,
	snapshot_after: Snapshot | None,
	prev_score: float,
	current_score: float,
) -> tuple[Reflection, Reward, Experience | None]:
	"""Record per-unit feedback for continuous mode.

	Creates a Reflection, computes a Reward, creates an Experience entry,
	and logs a UnitEvent. Operates at unit granularity instead of round.
	"""
	# Ensure FK-compatible round exists for this epoch
	round_id = _ensure_continuous_round(db, mission_id, epoch)

	# Verification deltas
	vi_result = _compute_verification_improvement(snapshot_before, snapshot_after)
	_, tests_before, tests_after, tests_delta, lint_delta, type_delta = vi_result

	# Count discoveries
	disc_count = 0
	if handoff and handoff.discoveries:
		try:
			disc_count = len(json.loads(handoff.discoveries))
		except (json.JSONDecodeError, TypeError):
			pass

	completion_rate = 1.0 if unit.status == "completed" else 0.0

	reflection = Reflection(
		mission_id=mission_id,
		round_id=round_id,
		round_number=epoch.number,
		epoch_id=epoch.id,
		tests_before=tests_before,
		tests_after=tests_after,
		tests_delta=tests_delta,
		lint_delta=lint_delta,
		type_delta=type_delta,
		objective_score=current_score,
		score_delta=current_score - prev_score,
		units_planned=1,
		units_completed=1 if unit.status == "completed" else 0,
		units_failed=1 if unit.status == "failed" else 0,
		completion_rate=completion_rate,
		# No fixup in continuous mode
		fixup_promoted=False,
		fixup_attempts=0,
		merge_conflicts=0,
		discoveries_count=disc_count,
	)
	db.insert_reflection(reflection)

	# Compute reward using existing function
	reward = compute_reward(reflection, prev_score, snapshot_before, snapshot_after)
	reward.epoch_id = epoch.id
	db.insert_reward(reward)

	# Create experience if we have a handoff
	experience = None
	if handoff:
		experience = Experience(
			round_id=round_id,
			work_unit_id=unit.id,
			epoch_id=epoch.id,
			title=unit.title,
			scope=unit.description,
			files_hint=unit.files_hint,
			status=handoff.status,
			summary=handoff.summary,
			files_changed=handoff.files_changed,
			discoveries=handoff.discoveries,
			concerns=handoff.concerns,
			reward=reward.reward,
		)
		db.insert_experience(experience)

	# Log unit event
	event = UnitEvent(
		mission_id=mission_id,
		epoch_id=epoch.id,
		work_unit_id=unit.id,
		event_type="completed" if unit.status == "completed" else "failed",
		score_after=current_score,
	)
	db.insert_unit_event(event)

	log.info(
		"Unit %s outcome: reward=%.3f score=%.2f->%.2f",
		unit.id, reward.reward, prev_score, current_score,
	)

	return reflection, reward, experience


def get_continuous_planner_context(
	db: Database,
	mission_id: str,
	limit: int = 10,
) -> str:
	"""Build context for the continuous planner from recent unit events.

	Similar to get_planner_context() but uses per-unit granularity
	instead of per-round.
	"""
	events = db.get_unit_events_for_mission(mission_id, limit=limit)
	if not events:
		return ""

	lines: list[str] = []

	# Score trajectory from events
	score_points = [(e.timestamp, e.score_after) for e in events if e.score_after > 0]
	if score_points:
		recent = score_points[-5:]
		trajectory = ", ".join(f"{s:.2f}" for _, s in recent)
		lines.append(f"Recent scores: {trajectory}")

	# Event type distribution
	type_counts: dict[str, int] = {}
	for e in events:
		type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
	if type_counts:
		dist = ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items()))
		lines.append(f"Event distribution: {dist}")

	# High-value discoveries from successful experiences
	top_experiences = db.get_top_experiences(limit=5)
	discoveries: list[str] = []
	for exp in top_experiences:
		if exp.discoveries and exp.epoch_id:
			try:
				disc_list = json.loads(exp.discoveries)
				for d in disc_list:
					if d not in discoveries:
						discoveries.append(d)
						if len(discoveries) >= 3:
							break
			except (json.JSONDecodeError, TypeError):
				pass
		if len(discoveries) >= 3:
			break

	if discoveries:
		lines.append("\nKey insights:")
		for d in discoveries:
			lines.append(f"- {d}")

	return "\n".join(lines)
