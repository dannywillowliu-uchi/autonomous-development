"""Tests for deterministic evaluator."""

from mission_control.evaluator import (
	EvalResult,
	compute_completion_rate,
	compute_lint_improvement,
	compute_no_regression,
	compute_test_improvement,
	evaluate_round,
)
from mission_control.models import Snapshot, SnapshotDelta


def _snap(test_total: int = 0, test_passed: int = 0, test_failed: int = 0, lint_errors: int = 0) -> Snapshot:
	return Snapshot(test_total=test_total, test_passed=test_passed, test_failed=test_failed, lint_errors=lint_errors)


# -- test_improvement --


class TestTestImprovement:
	def test_zero_before_zero_after(self):
		assert compute_test_improvement(_snap(), _snap()) == 0.0

	def test_zero_before_some_after(self):
		# (5 - 0) / max(0, 1) = 5.0, clamped to 1.0
		assert compute_test_improvement(_snap(), _snap(test_passed=5)) == 1.0

	def test_some_before_more_after(self):
		# (8 - 5) / max(5, 1) = 0.6
		assert compute_test_improvement(_snap(test_passed=5), _snap(test_passed=8)) == 0.6

	def test_no_change(self):
		assert compute_test_improvement(_snap(test_passed=10), _snap(test_passed=10)) == 0.0

	def test_regression_clamps_to_zero(self):
		# (3 - 10) / 10 = -0.7, clamped to 0.0
		assert compute_test_improvement(_snap(test_passed=10), _snap(test_passed=3)) == 0.0

	def test_one_test_added(self):
		# (1 - 0) / 1 = 1.0
		assert compute_test_improvement(_snap(), _snap(test_passed=1)) == 1.0

	def test_double_tests(self):
		# (20 - 10) / 10 = 1.0
		assert compute_test_improvement(_snap(test_passed=10), _snap(test_passed=20)) == 1.0

	def test_large_improvement_clamps(self):
		# (100 - 1) / 1 = 99.0, clamped to 1.0
		assert compute_test_improvement(_snap(test_passed=1), _snap(test_passed=100)) == 1.0


# -- lint_improvement --


class TestLintImprovement:
	def test_zero_before_zero_after(self):
		assert compute_lint_improvement(_snap(), _snap()) == 0.0

	def test_errors_reduced(self):
		# (10 - 3) / 10 = 0.7
		assert compute_lint_improvement(_snap(lint_errors=10), _snap(lint_errors=3)) == 0.7

	def test_all_errors_fixed(self):
		# (5 - 0) / 5 = 1.0
		assert compute_lint_improvement(_snap(lint_errors=5), _snap(lint_errors=0)) == 1.0

	def test_errors_increased(self):
		# (5 - 10) / 5 = -1.0, clamped to 0.0
		assert compute_lint_improvement(_snap(lint_errors=5), _snap(lint_errors=10)) == 0.0

	def test_no_change(self):
		assert compute_lint_improvement(_snap(lint_errors=5), _snap(lint_errors=5)) == 0.0

	def test_zero_before_errors_after(self):
		# (0 - 3) / max(0, 1) = -3.0, clamped to 0.0
		assert compute_lint_improvement(_snap(), _snap(lint_errors=3)) == 0.0

	def test_one_error_fixed(self):
		# (1 - 0) / 1 = 1.0
		assert compute_lint_improvement(_snap(lint_errors=1), _snap(lint_errors=0)) == 1.0


# -- completion_rate --


class TestCompletionRate:
	def test_all_completed(self):
		assert compute_completion_rate(5, 5) == 1.0

	def test_none_completed(self):
		assert compute_completion_rate(0, 5) == 0.0

	def test_partial(self):
		assert compute_completion_rate(3, 5) == 0.6

	def test_zero_planned(self):
		assert compute_completion_rate(0, 0) == 0.0

	def test_zero_planned_nonzero_completed(self):
		# Edge: completed > 0 but planned = 0 => 0.0
		assert compute_completion_rate(3, 0) == 0.0

	def test_negative_planned(self):
		assert compute_completion_rate(1, -1) == 0.0

	def test_overcompleted_clamps(self):
		# 6/5 = 1.2, clamped to 1.0
		assert compute_completion_rate(6, 5) == 1.0


# -- no_regression --


class TestNoRegression:
	def test_no_regression(self):
		delta = SnapshotDelta(tests_broken=0, security_delta=0)
		assert compute_no_regression(delta) == 1.0

	def test_tests_broken(self):
		delta = SnapshotDelta(tests_broken=3)
		assert compute_no_regression(delta) == 0.0

	def test_security_regression(self):
		delta = SnapshotDelta(security_delta=1)
		assert compute_no_regression(delta) == 0.0

	def test_both_regressions(self):
		delta = SnapshotDelta(tests_broken=2, security_delta=3)
		assert compute_no_regression(delta) == 0.0

	def test_improvements_no_regression(self):
		delta = SnapshotDelta(tests_fixed=5, lint_delta=-3, tests_broken=0, security_delta=0)
		assert compute_no_regression(delta) == 1.0


# -- evaluate_round (integration) --


class TestEvaluateRound:
	def test_perfect_score(self):
		before = _snap(test_passed=10, lint_errors=10)
		after = _snap(test_passed=20, lint_errors=0)
		delta = SnapshotDelta(tests_fixed=10, tests_broken=0, lint_delta=-10, security_delta=0)
		result = evaluate_round(before, after, delta, units_completed=5, units_planned=5)
		assert result.test_improvement == 1.0
		assert result.lint_improvement == 1.0
		assert result.completion_rate == 1.0
		assert result.no_regression == 1.0
		assert result.score == 1.0

	def test_zero_score(self):
		before = _snap(test_passed=10, lint_errors=5)
		after = _snap(test_passed=5, lint_errors=10)
		delta = SnapshotDelta(tests_broken=5, security_delta=1)
		result = evaluate_round(before, after, delta, units_completed=0, units_planned=5)
		assert result.test_improvement == 0.0
		assert result.lint_improvement == 0.0
		assert result.completion_rate == 0.0
		assert result.no_regression == 0.0
		assert result.score == 0.0

	def test_test_only_improvement(self):
		before = _snap(test_passed=5)
		after = _snap(test_passed=10)
		delta = SnapshotDelta(tests_fixed=5, tests_broken=0)
		result = evaluate_round(before, after, delta, units_completed=0, units_planned=5)
		# ti=1.0, li=0.0, cr=0.0, nr=1.0
		assert result.score == round(0.4 * 1.0 + 0.2 * 0.0 + 0.2 * 0.0 + 0.2 * 1.0, 6)
		assert result.score == 0.6

	def test_lint_only_improvement(self):
		before = _snap(lint_errors=20)
		after = _snap(lint_errors=10)
		delta = SnapshotDelta(lint_delta=-10, tests_broken=0)
		result = evaluate_round(before, after, delta, units_completed=3, units_planned=5)
		# ti=0.0, li=0.5, cr=0.6, nr=1.0
		expected = round(0.4 * 0.0 + 0.2 * 0.5 + 0.2 * 0.6 + 0.2 * 1.0, 6)
		assert result.score == expected

	def test_partial_improvements(self):
		before = _snap(test_passed=10, lint_errors=10)
		after = _snap(test_passed=13, lint_errors=6)
		delta = SnapshotDelta(tests_fixed=3, lint_delta=-4, tests_broken=0)
		result = evaluate_round(before, after, delta, units_completed=2, units_planned=4)
		# ti = 3/10 = 0.3, li = 4/10 = 0.4, cr = 2/4 = 0.5, nr = 1.0
		assert result.test_improvement == 0.3
		assert result.lint_improvement == 0.4
		assert result.completion_rate == 0.5
		assert result.no_regression == 1.0
		expected = round(0.4 * 0.3 + 0.2 * 0.4 + 0.2 * 0.5 + 0.2 * 1.0, 6)
		assert result.score == expected

	def test_regression_detected(self):
		before = _snap(test_passed=10, lint_errors=10)
		after = _snap(test_passed=15, lint_errors=5)
		delta = SnapshotDelta(tests_fixed=5, tests_broken=2, lint_delta=-5)
		result = evaluate_round(before, after, delta, units_completed=5, units_planned=5)
		# ti=0.5, li=0.5, cr=1.0, nr=0.0 (regression)
		assert result.no_regression == 0.0
		expected = round(0.4 * 0.5 + 0.2 * 0.5 + 0.2 * 1.0 + 0.2 * 0.0, 6)
		assert result.score == expected

	def test_zero_tests_before_and_after(self):
		before = _snap()
		after = _snap()
		delta = SnapshotDelta()
		result = evaluate_round(before, after, delta, units_completed=1, units_planned=1)
		# ti=0.0, li=0.0, cr=1.0, nr=1.0
		expected = round(0.4 * 0.0 + 0.2 * 0.0 + 0.2 * 1.0 + 0.2 * 1.0, 6)
		assert result.score == expected

	def test_all_tests_failing(self):
		before = _snap(test_total=10, test_passed=0, test_failed=10)
		after = _snap(test_total=10, test_passed=0, test_failed=10)
		delta = SnapshotDelta(tests_broken=0)
		result = evaluate_round(before, after, delta, units_completed=0, units_planned=3)
		# ti=0.0, li=0.0, cr=0.0, nr=1.0
		assert result.score == round(0.2 * 1.0, 6)

	def test_returns_eval_result_type(self):
		result = evaluate_round(_snap(), _snap(), SnapshotDelta(), 0, 0)
		assert isinstance(result, EvalResult)

	def test_zero_planned_units(self):
		before = _snap(test_passed=5)
		after = _snap(test_passed=10)
		delta = SnapshotDelta(tests_fixed=5)
		result = evaluate_round(before, after, delta, units_completed=0, units_planned=0)
		# cr=0.0 when nothing planned
		assert result.completion_rate == 0.0
