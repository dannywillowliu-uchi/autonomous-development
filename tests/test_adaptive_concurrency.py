"""Tests for AdaptiveConcurrencyController."""

from __future__ import annotations

from mission_control.adaptive_concurrency import (
	AdaptiveConcurrencyController,
	MergeOutcome,
)


class TestEmptyWindow:
	def test_empty_window_rates_are_zero(self) -> None:
		ctrl = AdaptiveConcurrencyController()
		assert ctrl.rolling_success_rate() == 0.0
		assert ctrl.rolling_conflict_rate() == 0.0

	def test_step_on_empty_window_scales_down(self) -> None:
		"""Empty window has success_rate=0.0 < 0.5, triggers scale-down."""
		ctrl = AdaptiveConcurrencyController(current_workers=4)
		result = ctrl.step()
		assert result < 4


class TestRampUp:
	def test_scale_up_on_high_success_low_conflict(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=4, max_workers=8)
		result = ctrl.recommend_capacity(success_rate=0.9, conflict_rate=0.05)
		assert result == 5

	def test_repeated_scale_up_to_max(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=6, max_workers=8)
		ctrl.recommend_capacity(success_rate=0.9, conflict_rate=0.0)
		assert ctrl.current_workers == 7
		ctrl.recommend_capacity(success_rate=0.95, conflict_rate=0.0)
		assert ctrl.current_workers == 8

	def test_scale_up_clamped_at_max(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=8, max_workers=8)
		result = ctrl.recommend_capacity(success_rate=1.0, conflict_rate=0.0)
		assert result == 8

	def test_ramp_up_via_recorded_outcomes(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=3, max_workers=8, window_size=10)
		for _ in range(10):
			ctrl.record_outcome(MergeOutcome.SUCCESS)
		result = ctrl.step()
		assert result == 4


class TestRampDown:
	def test_scale_down_on_high_conflict(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=6)
		result = ctrl.recommend_capacity(success_rate=0.7, conflict_rate=0.35)
		assert result < 6

	def test_scale_down_on_low_success(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=5)
		result = ctrl.recommend_capacity(success_rate=0.4, conflict_rate=0.1)
		assert result < 5

	def test_scale_down_clamped_at_min(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=1, min_workers=1)
		result = ctrl.recommend_capacity(success_rate=0.0, conflict_rate=1.0)
		assert result == 1

	def test_exponential_backoff_on_consecutive_downs(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=6, min_workers=1)
		# First scale-down: backoff=1
		r1 = ctrl.recommend_capacity(success_rate=0.3, conflict_rate=0.5)
		assert r1 == 5
		# Second scale-down: backoff=2
		r2 = ctrl.recommend_capacity(success_rate=0.3, conflict_rate=0.5)
		assert r2 == 3
		# Third: backoff=3, but clamped to min
		r3 = ctrl.recommend_capacity(success_rate=0.3, conflict_rate=0.5)
		assert r3 == max(1, 3 - 3)

	def test_ramp_down_via_recorded_conflicts(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=5, window_size=10)
		for _ in range(4):
			ctrl.record_outcome(MergeOutcome.CONFLICT)
		for _ in range(6):
			ctrl.record_outcome(MergeOutcome.FAIL)
		# conflict_rate = 0.4, success_rate = 0.0
		result = ctrl.step()
		assert result < 5


class TestHoldSteady:
	def test_hold_when_rates_are_moderate(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=4)
		result = ctrl.recommend_capacity(success_rate=0.7, conflict_rate=0.15)
		assert result == 4

	def test_hold_at_boundary_success_0_5(self) -> None:
		"""success_rate=0.5 is NOT < 0.5, so no scale-down."""
		ctrl = AdaptiveConcurrencyController(current_workers=4)
		result = ctrl.recommend_capacity(success_rate=0.5, conflict_rate=0.15)
		assert result == 4

	def test_hold_at_boundary_success_0_8(self) -> None:
		"""success_rate=0.8 is NOT > 0.8, so no scale-up."""
		ctrl = AdaptiveConcurrencyController(current_workers=4)
		result = ctrl.recommend_capacity(success_rate=0.8, conflict_rate=0.05)
		assert result == 4

	def test_hold_at_boundary_conflict_0_3(self) -> None:
		"""conflict_rate=0.3 is NOT > 0.3, so no scale-down."""
		ctrl = AdaptiveConcurrencyController(current_workers=4)
		result = ctrl.recommend_capacity(success_rate=0.7, conflict_rate=0.3)
		assert result == 4

	def test_hold_at_boundary_conflict_0_1(self) -> None:
		"""conflict_rate=0.1 is NOT < 0.1, so no scale-up even with high success."""
		ctrl = AdaptiveConcurrencyController(current_workers=4)
		result = ctrl.recommend_capacity(success_rate=0.9, conflict_rate=0.1)
		assert result == 4

	def test_consecutive_scale_down_counter_resets_on_hold(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=5)
		ctrl.recommend_capacity(success_rate=0.3, conflict_rate=0.5)
		assert ctrl._consecutive_scale_downs == 1
		ctrl.recommend_capacity(success_rate=0.7, conflict_rate=0.15)
		assert ctrl._consecutive_scale_downs == 0


class TestBoundaryAndClamping:
	def test_min_equals_max(self) -> None:
		ctrl = AdaptiveConcurrencyController(min_workers=3, max_workers=3, current_workers=3)
		assert ctrl.recommend_capacity(success_rate=1.0, conflict_rate=0.0) == 3
		assert ctrl.recommend_capacity(success_rate=0.0, conflict_rate=1.0) == 3

	def test_current_workers_clamped_on_init(self) -> None:
		ctrl = AdaptiveConcurrencyController(min_workers=2, max_workers=5, current_workers=10)
		assert ctrl.current_workers == 5
		ctrl2 = AdaptiveConcurrencyController(min_workers=3, max_workers=8, current_workers=1)
		assert ctrl2.current_workers == 3

	def test_window_size_respected(self) -> None:
		ctrl = AdaptiveConcurrencyController(window_size=5)
		for _ in range(10):
			ctrl.record_outcome(MergeOutcome.SUCCESS)
		assert len(ctrl._outcomes) == 5

	def test_mixed_outcomes_rates(self) -> None:
		ctrl = AdaptiveConcurrencyController(window_size=10)
		for _ in range(6):
			ctrl.record_outcome(MergeOutcome.SUCCESS)
		for _ in range(3):
			ctrl.record_outcome(MergeOutcome.CONFLICT)
		ctrl.record_outcome(MergeOutcome.FAIL)
		assert ctrl.rolling_success_rate() == 0.6
		assert ctrl.rolling_conflict_rate() == 0.3


class TestReset:
	def test_reset_clears_state(self) -> None:
		ctrl = AdaptiveConcurrencyController(current_workers=4)
		for _ in range(5):
			ctrl.record_outcome(MergeOutcome.SUCCESS)
		ctrl.recommend_capacity(success_rate=0.3, conflict_rate=0.5)
		ctrl.reset()
		assert len(ctrl._outcomes) == 0
		assert ctrl._consecutive_scale_downs == 0
		assert ctrl.rolling_success_rate() == 0.0
