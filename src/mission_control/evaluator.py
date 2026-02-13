"""Objective evaluator -- deterministic scoring from snapshots and completion rate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from mission_control.models import Snapshot

log = logging.getLogger(__name__)


@dataclass
class ObjectiveEvaluation:
	score: float = 0.0
	met: bool = False
	reasoning: str = ""
	remaining: list[str] = field(default_factory=list)


def _compute_delta_score(
	before: Snapshot | None,
	after: Snapshot | None,
) -> tuple[float, list[str]]:
	"""Score verification delta between two snapshots.

	Returns (score, remaining_issues).
	"""
	if before is None or after is None:
		return 0.5, ["no snapshot data available"]

	remaining: list[str] = []

	tests_broken = max(0, after.test_failed - before.test_failed)
	tests_fixed = max(0, after.test_passed - before.test_passed)
	lint_delta = after.lint_errors - before.lint_errors
	type_delta = after.type_errors - before.type_errors
	security_delta = after.security_findings - before.security_findings

	# Hard regression: broken tests or new security findings
	if tests_broken > 0 or security_delta > 0:
		if tests_broken > 0:
			remaining.append(f"{tests_broken} test(s) newly broken")
		if security_delta > 0:
			remaining.append(f"{security_delta} new security finding(s)")
		return 0.0, remaining

	# Build score from improvements
	score = 0.5  # baseline: no regression

	# Test improvements (up to +0.3)
	if tests_fixed > 0:
		score += min(0.3, tests_fixed * 0.05)

	# Lint improvements (up to +0.1)
	if lint_delta < 0:
		score += min(0.1, abs(lint_delta) * 0.01)
	elif lint_delta > 0:
		remaining.append(f"{lint_delta} new lint error(s)")
		score -= min(0.1, lint_delta * 0.01)

	# Type improvements (up to +0.1)
	if type_delta < 0:
		score += min(0.1, abs(type_delta) * 0.01)
	elif type_delta > 0:
		remaining.append(f"{type_delta} new type error(s)")
		score -= min(0.1, type_delta * 0.01)

	# Track remaining issues from current state
	if after.test_failed > 0:
		remaining.append(f"{after.test_failed} failing test(s)")
	if after.lint_errors > 0:
		remaining.append(f"{after.lint_errors} lint error(s)")
	if after.type_errors > 0:
		remaining.append(f"{after.type_errors} type error(s)")

	return max(0.0, min(1.0, score)), remaining


def evaluate_objective(
	snapshot_before: Snapshot | None,
	snapshot_after: Snapshot | None,
	completed_units: int,
	total_units: int,
	fixup_promoted: bool,
	prev_score: float = 0.0,
) -> ObjectiveEvaluation:
	"""Deterministic objective evaluation from snapshots and completion data.

	Score components (weights):
	  0.50 * verification delta (test/lint/type improvements)
	  0.35 * unit completion rate
	  0.15 * fixup promotion (verification passed on green branch)

	Final score blends with prev_score via momentum (0.3) so scores
	accumulate across rounds. objective_met requires score >= 0.9.
	"""
	# Verification delta
	delta_score, remaining = _compute_delta_score(snapshot_before, snapshot_after)

	# Completion rate
	completion_rate = completed_units / max(total_units, 1)

	# Fixup bonus
	fixup_score = 1.0 if fixup_promoted else 0.0

	# Composite raw score
	raw_score = 0.50 * delta_score + 0.35 * completion_rate + 0.15 * fixup_score
	raw_score = max(0.0, min(1.0, raw_score))

	# Momentum blending: score accumulates across rounds
	if prev_score > 0:
		score = 0.3 * prev_score + 0.7 * raw_score
	else:
		score = raw_score
	score = max(0.0, min(1.0, score))

	# Build reasoning
	parts: list[str] = [
		f"delta={delta_score:.2f}",
		f"completion={completion_rate:.2f}",
		f"fixup={'passed' if fixup_promoted else 'failed'}",
		f"raw={raw_score:.2f}",
	]
	if prev_score > 0:
		parts.append(f"momentum={prev_score:.2f}->{score:.2f}")

	reasoning = ", ".join(parts)
	met = score >= 0.9 and completion_rate >= 0.8

	log.info(
		"Deterministic eval: score=%.2f met=%s (%s)",
		score, met, reasoning,
	)

	return ObjectiveEvaluation(
		score=score,
		met=met,
		reasoning=reasoning,
		remaining=remaining,
	)
