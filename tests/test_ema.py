"""Tests for ExponentialMovingAverage budget tracking."""

from __future__ import annotations

import math

import pytest

from mission_control.ema import ExponentialMovingAverage


class TestBasicEMA:
	def test_first_value_is_ema(self) -> None:
		ema = ExponentialMovingAverage(alpha=0.3)
		result = ema.update(10.0)
		assert result == 10.0
		assert ema.value == 10.0

	def test_second_value_blends(self) -> None:
		ema = ExponentialMovingAverage(alpha=0.3)
		ema.update(10.0)
		result = ema.update(20.0)
		expected = 0.3 * 20.0 + 0.7 * 10.0
		assert result == pytest.approx(expected)

	def test_multiple_values_converge(self) -> None:
		ema = ExponentialMovingAverage(alpha=0.3)
		for _ in range(100):
			ema.update(5.0)
		assert ema.value == pytest.approx(5.0, abs=1e-6)

	def test_alpha_1_takes_latest(self) -> None:
		ema = ExponentialMovingAverage(alpha=1.0)
		ema.update(10.0)
		ema.update(20.0)
		assert ema.value == pytest.approx(20.0)

	def test_alpha_0_keeps_first(self) -> None:
		ema = ExponentialMovingAverage(alpha=0.0)
		ema.update(10.0)
		ema.update(20.0)
		assert ema.value == pytest.approx(10.0)

	def test_count_increments(self) -> None:
		ema = ExponentialMovingAverage()
		assert ema.count == 0
		ema.update(1.0)
		assert ema.count == 1
		ema.update(2.0)
		assert ema.count == 2


class TestOutlierDampening:
	def test_no_dampening_with_few_points(self) -> None:
		"""Outlier dampening only kicks in after 3+ data points."""
		ema = ExponentialMovingAverage(alpha=0.3, outlier_multiplier=3.0)
		ema.update(1.0)  # count=1
		ema.update(1.0)  # count=2
		# 3rd update: count=3, not > 3 so no dampening
		result = ema.update(100.0)
		# After 2 updates of 1.0, ema = 0.3*1 + 0.7*1 = 1.0
		# 3rd update (count=3, not > 3): ema = 0.3*100 + 0.7*1 = 30.7
		assert result == pytest.approx(30.7)

	def test_dampening_after_baseline(self) -> None:
		"""After 3+ data points, spikes >3x EMA are clamped to 2x EMA."""
		ema = ExponentialMovingAverage(alpha=0.3, outlier_multiplier=3.0)
		for _ in range(4):
			ema.update(10.0)
		# EMA should be ~10.0 after 4 identical values
		assert ema.value == pytest.approx(10.0)

		# 5th update with a huge spike (>3x EMA)
		ema.update(100.0)  # 100 > 3*10=30, so clamped to 2*10=20
		# EMA = 0.3*20 + 0.7*10 = 6 + 7 = 13
		assert ema.value == pytest.approx(13.0)

	def test_no_dampening_below_threshold(self) -> None:
		"""Values <= outlier_multiplier * EMA are NOT dampened."""
		ema = ExponentialMovingAverage(alpha=0.3, outlier_multiplier=3.0)
		for _ in range(4):
			ema.update(10.0)

		# 25.0 is < 3*10=30, so no dampening
		ema.update(25.0)
		expected = 0.3 * 25.0 + 0.7 * 10.0
		assert ema.value == pytest.approx(expected)

	def test_dampening_clamps_to_2x(self) -> None:
		"""Verify clamped value is exactly 2x EMA."""
		ema = ExponentialMovingAverage(alpha=0.5, outlier_multiplier=3.0)
		for _ in range(4):
			ema.update(10.0)
		before = ema.value

		ema.update(1000.0)  # massive spike, clamped to 2*10=20
		expected = 0.5 * 20.0 + 0.5 * before
		assert ema.value == pytest.approx(expected)


class TestConservatismFactor:
	def test_factor_decays_with_n(self) -> None:
		"""Conservatism factor should decrease as count increases."""
		ema = ExponentialMovingAverage(conservatism_base=0.5)
		ema.update(10.0)
		k1 = ema._conservatism_factor()

		ema.update(10.0)
		k2 = ema._conservatism_factor()

		ema.update(10.0)
		k3 = ema._conservatism_factor()

		assert k1 > k2 > k3

	def test_factor_formula(self) -> None:
		"""k = 1.0 + base / sqrt(n)."""
		ema = ExponentialMovingAverage(conservatism_base=0.5)
		for _ in range(4):
			ema.update(10.0)
		expected_k = 1.0 + 0.5 / math.sqrt(4)
		assert ema._conservatism_factor() == pytest.approx(expected_k)

	def test_projected_cost_includes_factor(self) -> None:
		ema = ExponentialMovingAverage(alpha=0.3, conservatism_base=0.5)
		for _ in range(4):
			ema.update(10.0)
		k = 1.0 + 0.5 / math.sqrt(4)
		assert ema.projected_cost() == pytest.approx(10.0 * k)

	def test_factor_approaches_1(self) -> None:
		"""With many data points, factor converges to 1.0."""
		ema = ExponentialMovingAverage(conservatism_base=0.5)
		for _ in range(10000):
			ema.update(10.0)
		assert ema._conservatism_factor() == pytest.approx(1.0, abs=0.01)


class TestEdgeCases:
	def test_empty_value_is_none(self) -> None:
		ema = ExponentialMovingAverage()
		assert ema.value is None
		assert ema.count == 0

	def test_empty_projected_cost_is_none(self) -> None:
		ema = ExponentialMovingAverage()
		assert ema.projected_cost() is None

	def test_single_data_point(self) -> None:
		ema = ExponentialMovingAverage(alpha=0.3, conservatism_base=0.5)
		ema.update(5.0)
		assert ema.value == 5.0
		assert ema.count == 1
		k = 1.0 + 0.5 / math.sqrt(1)
		assert ema.projected_cost() == pytest.approx(5.0 * k)

	def test_zero_values(self) -> None:
		ema = ExponentialMovingAverage()
		ema.update(0.0)
		assert ema.value == 0.0
		ema.update(0.0)
		assert ema.value == 0.0

	def test_negative_values(self) -> None:
		"""EMA should handle negative values (though costs shouldn't be negative)."""
		ema = ExponentialMovingAverage(alpha=0.5)
		ema.update(-10.0)
		assert ema.value == -10.0
		ema.update(10.0)
		assert ema.value == pytest.approx(0.0)


class TestBudgetCheck:
	def test_empty_ema_never_exceeds(self) -> None:
		ema = ExponentialMovingAverage()
		assert not ema.would_exceed_budget(spent=100.0, budget=50.0)

	def test_within_budget(self) -> None:
		ema = ExponentialMovingAverage(alpha=0.3, conservatism_base=0.5)
		for _ in range(4):
			ema.update(2.0)
		# projected ~= 2.0 * (1.0 + 0.5/sqrt(4)) = 2.0 * 1.25 = 2.5
		assert not ema.would_exceed_budget(spent=40.0, budget=50.0)

	def test_exceeds_budget(self) -> None:
		ema = ExponentialMovingAverage(alpha=0.3, conservatism_base=0.5)
		for _ in range(4):
			ema.update(5.0)
		# projected ~= 5.0 * 1.25 = 6.25
		# spent=45.0, 45 + 6.25 = 51.25 > 50
		assert ema.would_exceed_budget(spent=45.0, budget=50.0)

	def test_exact_boundary(self) -> None:
		"""Spending exactly at the boundary should not exceed."""
		ema = ExponentialMovingAverage(alpha=1.0, conservatism_base=0.0)
		# With conservatism_base=0, factor at n=0 is 1.0+0.0=1.0
		# But once we add data, factor is 1.0 + 0.0/sqrt(n) = 1.0
		ema.update(10.0)
		# projected = 10.0 * 1.0 = 10.0
		# spent=40, 40+10 = 50 == budget, not exceeded (uses >)
		assert not ema.would_exceed_budget(spent=40.0, budget=50.0)


class TestConfigIntegration:
	def test_ema_from_config_defaults(self) -> None:
		"""EMA can be constructed from BudgetConfig default values."""
		from mission_control.config import BudgetConfig

		bc = BudgetConfig()
		ema = ExponentialMovingAverage(
			alpha=bc.ema_alpha,
			outlier_multiplier=bc.outlier_multiplier,
			conservatism_base=bc.conservatism_base,
		)
		assert ema._alpha == 0.3
		assert ema._outlier_multiplier == 3.0
		assert ema._conservatism_base == 0.5

	def test_ema_from_custom_config(self) -> None:
		"""EMA respects custom BudgetConfig values."""
		from mission_control.config import BudgetConfig

		bc = BudgetConfig(ema_alpha=0.5, outlier_multiplier=2.0, conservatism_base=1.0)
		ema = ExponentialMovingAverage(
			alpha=bc.ema_alpha,
			outlier_multiplier=bc.outlier_multiplier,
			conservatism_base=bc.conservatism_base,
		)
		ema.update(10.0)
		ema.update(10.0)
		k = 1.0 + 1.0 / math.sqrt(2)
		assert ema.projected_cost() == pytest.approx(10.0 * k)
