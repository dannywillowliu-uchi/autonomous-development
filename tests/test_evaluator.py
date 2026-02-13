"""Tests for deterministic objective evaluator."""

from __future__ import annotations

from mission_control.evaluator import ObjectiveEvaluation, _compute_delta_score, evaluate_objective
from mission_control.models import Snapshot


class TestComputeDeltaScore:
	def test_no_snapshots_returns_baseline(self) -> None:
		score, remaining = _compute_delta_score(None, None)
		assert score == 0.5
		assert "no snapshot data available" in remaining

	def test_no_regression_baseline(self) -> None:
		before = Snapshot(test_passed=10, test_failed=0, lint_errors=5, type_errors=2, security_findings=0)
		after = Snapshot(test_passed=10, test_failed=0, lint_errors=5, type_errors=2, security_findings=0)
		score, remaining = _compute_delta_score(before, after)
		assert score == 0.5
		# Remaining reports current state issues even if no delta
		assert "5 lint error(s)" in remaining
		assert "2 type error(s)" in remaining

	def test_tests_improved(self) -> None:
		before = Snapshot(test_passed=10, test_failed=2, lint_errors=0, type_errors=0, security_findings=0)
		after = Snapshot(test_passed=14, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		score, remaining = _compute_delta_score(before, after)
		# 0.5 baseline + 0.2 (4 tests fixed * 0.05)
		assert score == 0.7

	def test_tests_broken_hard_regression(self) -> None:
		before = Snapshot(test_passed=10, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		after = Snapshot(test_passed=8, test_failed=2, lint_errors=0, type_errors=0, security_findings=0)
		score, remaining = _compute_delta_score(before, after)
		assert score == 0.0
		assert "2 test(s) newly broken" in remaining

	def test_security_regression(self) -> None:
		before = Snapshot(test_passed=10, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		after = Snapshot(test_passed=10, test_failed=0, lint_errors=0, type_errors=0, security_findings=3)
		score, remaining = _compute_delta_score(before, after)
		assert score == 0.0
		assert "3 new security finding(s)" in remaining

	def test_lint_improved(self) -> None:
		before = Snapshot(test_passed=10, test_failed=0, lint_errors=10, type_errors=0, security_findings=0)
		after = Snapshot(test_passed=10, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		score, remaining = _compute_delta_score(before, after)
		# 0.5 + 0.1 (lint improvements capped at 0.1)
		assert score == 0.6

	def test_lint_regressed(self) -> None:
		before = Snapshot(test_passed=10, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		after = Snapshot(test_passed=10, test_failed=0, lint_errors=5, type_errors=0, security_findings=0)
		score, remaining = _compute_delta_score(before, after)
		# 0.5 - 0.05 (5 * 0.01)
		assert score == 0.45
		assert "5 new lint error(s)" in remaining

	def test_remaining_includes_current_failures(self) -> None:
		before = Snapshot(test_passed=10, test_failed=5, lint_errors=3, type_errors=1, security_findings=0)
		after = Snapshot(test_passed=10, test_failed=5, lint_errors=3, type_errors=1, security_findings=0)
		score, remaining = _compute_delta_score(before, after)
		assert "5 failing test(s)" in remaining
		assert "3 lint error(s)" in remaining
		assert "1 type error(s)" in remaining

	def test_all_improvements_max_score(self) -> None:
		before = Snapshot(test_passed=0, test_failed=10, lint_errors=20, type_errors=20, security_findings=0)
		after = Snapshot(test_passed=20, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		score, remaining = _compute_delta_score(before, after)
		# 0.5 + 0.3 (tests, capped) + 0.1 (lint, capped) + 0.1 (types, capped) = 1.0
		assert score == 1.0


class TestEvaluateObjective:
	def test_perfect_round(self) -> None:
		before = Snapshot(test_passed=10, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		after = Snapshot(test_passed=15, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		result = evaluate_objective(
			snapshot_before=before,
			snapshot_after=after,
			completed_units=5,
			total_units=5,
			fixup_promoted=True,
		)
		# delta=0.75, completion=1.0, fixup=1.0
		# raw = 0.50*0.75 + 0.35*1.0 + 0.15*1.0 = 0.375 + 0.35 + 0.15 = 0.875
		assert result.score > 0.8
		assert "passed" in result.reasoning

	def test_regression_round(self) -> None:
		before = Snapshot(test_passed=10, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		after = Snapshot(test_passed=8, test_failed=2, lint_errors=0, type_errors=0, security_findings=0)
		result = evaluate_objective(
			snapshot_before=before,
			snapshot_after=after,
			completed_units=3,
			total_units=5,
			fixup_promoted=False,
		)
		# delta=0.0 (regression), completion=0.6, fixup=0.0
		# raw = 0.50*0.0 + 0.35*0.6 + 0.15*0.0 = 0.21
		assert result.score < 0.3
		assert result.met is False

	def test_partial_completion(self) -> None:
		result = evaluate_objective(
			snapshot_before=None,
			snapshot_after=None,
			completed_units=3,
			total_units=10,
			fixup_promoted=False,
		)
		# delta=0.5 (no data), completion=0.3, fixup=0.0
		# raw = 0.50*0.5 + 0.35*0.3 + 0.15*0.0 = 0.25 + 0.105 = 0.355
		assert 0.3 < result.score < 0.4

	def test_momentum_blending(self) -> None:
		result = evaluate_objective(
			snapshot_before=None,
			snapshot_after=None,
			completed_units=5,
			total_units=5,
			fixup_promoted=True,
			prev_score=0.8,
		)
		# raw = 0.50*0.5 + 0.35*1.0 + 0.15*1.0 = 0.75
		# blended = 0.3*0.8 + 0.7*0.75 = 0.24 + 0.525 = 0.765
		assert 0.7 < result.score < 0.8
		assert "momentum" in result.reasoning

	def test_no_momentum_when_prev_zero(self) -> None:
		result = evaluate_objective(
			snapshot_before=None,
			snapshot_after=None,
			completed_units=5,
			total_units=5,
			fixup_promoted=True,
			prev_score=0.0,
		)
		# raw = 0.50*0.5 + 0.35*1.0 + 0.15*1.0 = 0.75
		# No momentum blending (prev_score=0)
		assert abs(result.score - 0.75) < 0.01
		assert "momentum" not in result.reasoning

	def test_objective_met_requires_high_score_and_completion(self) -> None:
		before = Snapshot(test_passed=0, test_failed=10, lint_errors=20, type_errors=20, security_findings=0)
		after = Snapshot(test_passed=20, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		result = evaluate_objective(
			snapshot_before=before,
			snapshot_after=after,
			completed_units=10,
			total_units=10,
			fixup_promoted=True,
			prev_score=0.85,
		)
		# delta=1.0, completion=1.0, fixup=1.0
		# raw=1.0, blended=0.3*0.85+0.7*1.0=0.955
		assert result.met is True
		assert result.score >= 0.9

	def test_objective_not_met_low_completion(self) -> None:
		before = Snapshot(test_passed=0, test_failed=10, lint_errors=20, type_errors=20, security_findings=0)
		after = Snapshot(test_passed=20, test_failed=0, lint_errors=0, type_errors=0, security_findings=0)
		result = evaluate_objective(
			snapshot_before=before,
			snapshot_after=after,
			completed_units=1,
			total_units=10,
			fixup_promoted=True,
			prev_score=0.95,
		)
		# Even with high score, low completion rate blocks met
		assert result.met is False

	def test_zero_units_does_not_divide_by_zero(self) -> None:
		result = evaluate_objective(
			snapshot_before=None,
			snapshot_after=None,
			completed_units=0,
			total_units=0,
			fixup_promoted=False,
		)
		assert isinstance(result, ObjectiveEvaluation)
		assert result.score >= 0.0

	def test_score_always_clamped(self) -> None:
		result = evaluate_objective(
			snapshot_before=None,
			snapshot_after=None,
			completed_units=100,
			total_units=1,
			fixup_promoted=True,
			prev_score=1.5,
		)
		assert result.score <= 1.0
		assert result.score >= 0.0
