"""Tests for swarm stagnation detection."""

from __future__ import annotations

from autodev.swarm.stagnation import (
	PivotRecommendation,
	analyze_stagnation,
	format_pivots_for_planner,
)


class TestAnalyzeStagnation:
	def test_no_pivots_when_insufficient_data(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=1,
			test_history=[5],
			completion_history=[1],
			failure_history=[0],
			cost_history=[0.1],
		)
		assert pivots == []

	def test_flat_test_count_triggers_research_pivot(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[10, 10, 10],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			threshold=3,
		)
		assert len(pivots) == 1
		assert pivots[0].strategy == "research_before_implement"
		assert pivots[0].severity == "critical"

	def test_flat_test_count_zero_does_not_trigger(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[0, 0, 0],
			completion_history=[0, 0, 0],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.1, 0.1],
			threshold=3,
		)
		# Flat at 0 is not the same as flat at a positive value
		flat_pivots = [p for p in pivots if p.strategy == "research_before_implement"]
		assert flat_pivots == []

	def test_varying_test_count_no_pivot(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[8, 9, 10],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			threshold=3,
		)
		flat_pivots = [p for p in pivots if p.strategy == "research_before_implement"]
		assert flat_pivots == []

	def test_rising_cost_flat_completions_triggers_reduce(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[3, 3, 3],
			failure_history=[0, 0, 0],
			cost_history=[1.0, 1.5, 2.0],
			threshold=3,
		)
		reduce_pivots = [p for p in pivots if p.strategy == "reduce_and_focus"]
		assert len(reduce_pivots) == 1
		assert reduce_pivots[0].severity == "warning"

	def test_cost_not_rising_enough_no_reduce(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[3, 3, 3],
			failure_history=[0, 0, 0],
			cost_history=[1.0, 1.05, 1.1],
			threshold=3,
		)
		reduce_pivots = [p for p in pivots if p.strategy == "reduce_and_focus"]
		assert reduce_pivots == []

	def test_high_failure_rate_triggers_diagnose(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 5, 5],
			completion_history=[0, 0, 1],
			failure_history=[3, 4, 5],
			cost_history=[0.1, 0.2, 0.3],
			threshold=3,
		)
		diag_pivots = [p for p in pivots if p.strategy == "diagnose_systemic"]
		assert len(diag_pivots) == 1
		assert diag_pivots[0].severity == "critical"

	def test_low_failure_rate_no_diagnose(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[3, 4, 5],
			failure_history=[0, 1, 0],
			cost_history=[0.1, 0.2, 0.3],
			threshold=3,
		)
		diag_pivots = [p for p in pivots if p.strategy == "diagnose_systemic"]
		assert diag_pivots == []

	def test_multiple_pivots_can_trigger(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=10,
			test_history=[5, 5, 5],
			completion_history=[2, 2, 2],
			failure_history=[5, 6, 7],
			cost_history=[1.0, 1.5, 2.5],
			threshold=3,
		)
		strategies = {p.strategy for p in pivots}
		assert "research_before_implement" in strategies
		assert "diagnose_systemic" in strategies

	def test_custom_threshold(self) -> None:
		# With threshold=5, 3 data points shouldn't trigger flat test pivot
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[10, 10, 10],
			completion_history=[1, 1, 1],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.1, 0.1],
			threshold=5,
		)
		flat_pivots = [p for p in pivots if p.strategy == "research_before_implement"]
		assert flat_pivots == []


class TestFormatPivotsForPlanner:
	def test_empty_pivots_returns_empty(self) -> None:
		assert format_pivots_for_planner([]) == ""

	def test_critical_pivot_uses_triple_bang(self) -> None:
		pivots = [
			PivotRecommendation(
				trigger="Test count flat",
				strategy="research_before_implement",
				severity="critical",
				details="Switch to research mode.",
			)
		]
		text = format_pivots_for_planner(pivots)
		assert "!!!" in text
		assert "Test count flat" in text
		assert "research_before_implement" in text

	def test_warning_pivot_uses_single_bang(self) -> None:
		pivots = [
			PivotRecommendation(
				trigger="Cost rising",
				strategy="reduce_and_focus",
				severity="warning",
				details="Reduce agents.",
			)
		]
		text = format_pivots_for_planner(pivots)
		assert "!" in text
		assert "!!!" not in text
		assert "Cost rising" in text

	def test_header_present(self) -> None:
		pivots = [
			PivotRecommendation(
				trigger="x", strategy="y", severity="warning"
			)
		]
		text = format_pivots_for_planner(pivots)
		assert "PIVOT RECOMMENDATIONS" in text
