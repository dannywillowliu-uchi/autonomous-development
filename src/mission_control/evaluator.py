"""Deterministic evaluator -- scores rounds using objective signals only."""

from __future__ import annotations

from dataclasses import dataclass

from mission_control.models import Snapshot, SnapshotDelta


@dataclass
class EvalResult:
	"""Result of deterministic round evaluation."""

	score: float
	test_improvement: float
	lint_improvement: float
	completion_rate: float
	no_regression: float


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
	return max(lo, min(hi, value))


def compute_test_improvement(before: Snapshot, after: Snapshot) -> float:
	"""Test improvement ratio clamped to [0, 1].

	(tests_after - tests_before) / max(tests_before, 1)
	Uses test_passed as the primary signal.
	"""
	delta = after.test_passed - before.test_passed
	return _clamp(delta / max(before.test_passed, 1))


def compute_lint_improvement(before: Snapshot, after: Snapshot) -> float:
	"""Lint improvement ratio clamped to [0, 1].

	max(0, (errors_before - errors_after) / max(errors_before, 1))
	"""
	reduction = before.lint_errors - after.lint_errors
	return _clamp(reduction / max(before.lint_errors, 1))


def compute_completion_rate(units_completed: int, units_planned: int) -> float:
	"""Fraction of planned units completed, clamped to [0, 1]."""
	if units_planned <= 0:
		return 0.0
	return _clamp(units_completed / units_planned)


def compute_no_regression(delta: SnapshotDelta) -> float:
	"""1.0 if no test regressions, 0.0 otherwise."""
	return 0.0 if delta.regressed else 1.0


def evaluate_round(
	before: Snapshot,
	after: Snapshot,
	delta: SnapshotDelta,
	units_completed: int,
	units_planned: int,
) -> EvalResult:
	"""Score a round deterministically using objective signals.

	score = 0.4 * test_improvement
	      + 0.2 * lint_improvement
	      + 0.2 * completion_rate
	      + 0.2 * no_regression
	"""
	ti = compute_test_improvement(before, after)
	li = compute_lint_improvement(before, after)
	cr = compute_completion_rate(units_completed, units_planned)
	nr = compute_no_regression(delta)

	score = 0.4 * ti + 0.2 * li + 0.2 * cr + 0.2 * nr
	return EvalResult(
		score=round(score, 6),
		test_improvement=round(ti, 6),
		lint_improvement=round(li, 6),
		completion_rate=round(cr, 6),
		no_regression=round(nr, 6),
	)
