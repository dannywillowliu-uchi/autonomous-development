"""Tests for cost-aware budget pacing and efficiency tracking in degradation."""

from __future__ import annotations

from mission_control.config import DegradationConfig
from mission_control.degradation import (
	DegradationLevel,
	DegradationManager,
)


class TestRecordUnitCost:
	def test_records_single_unit(self) -> None:
		mgr = DegradationManager()
		mgr.record_unit_cost("u1", 2.0, merged=True)
		assert len(mgr._unit_costs) == 1
		assert mgr._unit_costs[0] == ("u1", 2.0, True)

	def test_records_multiple_units(self) -> None:
		mgr = DegradationManager()
		mgr.record_unit_cost("u1", 1.5, merged=True)
		mgr.record_unit_cost("u2", 3.0, merged=False)
		mgr.record_unit_cost("u3", 2.0, merged=True)
		assert len(mgr._unit_costs) == 3

	def test_records_failed_unit(self) -> None:
		mgr = DegradationManager()
		mgr.record_unit_cost("u1", 4.0, merged=False)
		assert mgr._unit_costs[0] == ("u1", 4.0, False)


class TestCostPerMerge:
	def test_no_units_returns_none(self) -> None:
		mgr = DegradationManager()
		assert mgr.cost_per_merge() is None

	def test_all_failures_returns_none(self) -> None:
		mgr = DegradationManager()
		mgr.record_unit_cost("u1", 3.0, merged=False)
		mgr.record_unit_cost("u2", 5.0, merged=False)
		assert mgr.cost_per_merge() is None

	def test_single_merge(self) -> None:
		mgr = DegradationManager()
		mgr.record_unit_cost("u1", 4.0, merged=True)
		assert mgr.cost_per_merge() == 4.0

	def test_mixed_success_failure(self) -> None:
		mgr = DegradationManager()
		mgr.record_unit_cost("u1", 2.0, merged=True)
		mgr.record_unit_cost("u2", 5.0, merged=False)
		mgr.record_unit_cost("u3", 6.0, merged=True)
		# Average of merged only: (2.0 + 6.0) / 2 = 4.0
		assert mgr.cost_per_merge() == 4.0

	def test_excludes_failed_costs(self) -> None:
		mgr = DegradationManager()
		mgr.record_unit_cost("u1", 1.0, merged=True)
		mgr.record_unit_cost("u2", 100.0, merged=False)
		assert mgr.cost_per_merge() == 1.0


class TestBudgetPace:
	def test_zero_wall_time_returns_zero(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(50.0, 100.0)
		assert mgr.budget_pace(300.0, 0.0) == 0.0

	def test_zero_budget_returns_zero(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(0.0, 0.0)
		assert mgr.budget_pace(300.0, 600.0) == 0.0

	def test_zero_elapsed_returns_zero(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(50.0, 100.0)
		assert mgr.budget_pace(0.0, 600.0) == 0.0

	def test_on_pace(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(50.0, 100.0)
		# 50% budget spent, 50% time elapsed -> pace = 1.0
		pace = mgr.budget_pace(300.0, 600.0)
		assert pace == 1.0

	def test_overspending(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(80.0, 100.0)
		# 80% budget, 40% time -> pace = 2.0
		pace = mgr.budget_pace(240.0, 600.0)
		assert pace == 2.0

	def test_underspending(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(20.0, 100.0)
		# 20% budget, 50% time -> pace = 0.4
		pace = mgr.budget_pace(300.0, 600.0)
		assert abs(pace - 0.4) < 1e-9

	def test_updates_last_budget_pace(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(80.0, 100.0)
		mgr.budget_pace(240.0, 600.0)
		assert mgr._last_budget_pace == 2.0


class TestShouldReduceWorkersForCost:
	def test_false_with_no_data(self) -> None:
		mgr = DegradationManager()
		assert mgr.should_reduce_workers_for_cost() is False

	def test_true_when_cost_per_merge_exceeds_threshold(self) -> None:
		cfg = DegradationConfig(cost_per_merge_threshold=3.0)
		mgr = DegradationManager(config=cfg)
		mgr.record_unit_cost("u1", 4.0, merged=True)
		assert mgr.should_reduce_workers_for_cost() is True

	def test_false_when_cost_per_merge_below_threshold(self) -> None:
		cfg = DegradationConfig(cost_per_merge_threshold=5.0)
		mgr = DegradationManager(config=cfg)
		mgr.record_unit_cost("u1", 3.0, merged=True)
		assert mgr.should_reduce_workers_for_cost() is False

	def test_true_when_budget_pace_exceeds_threshold(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(80.0, 100.0)
		mgr.budget_pace(240.0, 600.0)  # pace = 2.0 > 1.5
		assert mgr.should_reduce_workers_for_cost() is True

	def test_false_when_budget_pace_below_threshold(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(50.0, 100.0)
		mgr.budget_pace(300.0, 600.0)  # pace = 1.0 < 1.5
		assert mgr.should_reduce_workers_for_cost() is False

	def test_all_failures_and_low_pace(self) -> None:
		mgr = DegradationManager()
		mgr.record_unit_cost("u1", 10.0, merged=False)
		# cost_per_merge is None (no merges), pace is 0.0
		assert mgr.should_reduce_workers_for_cost() is False


class TestCostTriggersWorkerReduction:
	def test_high_cost_per_merge_triggers_reduced_workers(self) -> None:
		cfg = DegradationConfig(cost_per_merge_threshold=3.0)
		mgr = DegradationManager(config=cfg)
		# First unit below threshold
		mgr.record_unit_cost("u1", 2.0, merged=True)
		assert mgr.level == DegradationLevel.FULL_CAPACITY
		# Second unit pushes average above threshold: (2+5)/2 = 3.5 > 3.0
		mgr.record_unit_cost("u2", 5.0, merged=True)
		assert mgr.level == DegradationLevel.REDUCED_WORKERS

	def test_budget_pace_triggers_reduced_workers(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(80.0, 100.0)
		mgr.budget_pace(240.0, 600.0)  # pace = 2.0
		# Now record a unit -- should_reduce triggers escalation
		mgr.record_unit_cost("u1", 1.0, merged=True)
		assert mgr.level == DegradationLevel.REDUCED_WORKERS

	def test_no_reduction_below_thresholds(self) -> None:
		cfg = DegradationConfig(cost_per_merge_threshold=10.0)
		mgr = DegradationManager(config=cfg)
		mgr.record_unit_cost("u1", 2.0, merged=True)
		mgr.record_unit_cost("u2", 3.0, merged=True)
		assert mgr.level == DegradationLevel.FULL_CAPACITY

	def test_failed_units_dont_affect_cost_per_merge(self) -> None:
		cfg = DegradationConfig(cost_per_merge_threshold=3.0)
		mgr = DegradationManager(config=cfg)
		mgr.record_unit_cost("u1", 2.0, merged=True)
		mgr.record_unit_cost("u2", 100.0, merged=False)
		# cost_per_merge is still 2.0 (only merged units counted)
		assert mgr.level == DegradationLevel.FULL_CAPACITY

	def test_does_not_escalate_beyond_reduced_workers(self) -> None:
		"""Cost signals only trigger REDUCED_WORKERS, not higher levels."""
		cfg = DegradationConfig(cost_per_merge_threshold=1.0)
		mgr = DegradationManager(config=cfg)
		mgr.record_unit_cost("u1", 10.0, merged=True)
		assert mgr.level == DegradationLevel.REDUCED_WORKERS
		# Recording more expensive units doesn't escalate further
		mgr.record_unit_cost("u2", 50.0, merged=True)
		assert mgr.level == DegradationLevel.REDUCED_WORKERS


class TestCostPerMergeThresholdConfig:
	def test_default_value(self) -> None:
		cfg = DegradationConfig()
		assert cfg.cost_per_merge_threshold == 5.0

	def test_custom_value(self) -> None:
		cfg = DegradationConfig(cost_per_merge_threshold=10.0)
		assert cfg.cost_per_merge_threshold == 10.0

	def test_manager_reads_config(self) -> None:
		cfg = DegradationConfig(cost_per_merge_threshold=7.5)
		mgr = DegradationManager(config=cfg)
		assert mgr._cost_per_merge_threshold == 7.5


class TestStatusDictIncludesCostFields:
	def test_status_dict_with_no_costs(self) -> None:
		mgr = DegradationManager()
		status = mgr.get_status_dict()
		assert status["cost_per_merge"] is None
		assert status["unit_costs_recorded"] == 0
		assert status["budget_pace"] == 0.0

	def test_status_dict_with_costs(self) -> None:
		mgr = DegradationManager()
		mgr.record_unit_cost("u1", 3.0, merged=True)
		mgr.record_unit_cost("u2", 5.0, merged=True)
		status = mgr.get_status_dict()
		assert status["cost_per_merge"] == 4.0
		assert status["unit_costs_recorded"] == 2
