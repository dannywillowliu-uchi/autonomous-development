"""Tests for swarm stagnation detection."""

from __future__ import annotations

from autodev.swarm.stagnation import (
	PivotRecommendation,
	StagnationConfig,
	analyze_stagnation,
	format_pivots_for_planner,
	pivots_to_decisions,
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


class TestStagnationConfig:
	def test_config_window_overrides_threshold(self) -> None:
		cfg = StagnationConfig(window=5)
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[10, 10, 10],
			completion_history=[1, 1, 1],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.1, 0.1],
			threshold=3,  # should be ignored when config is provided
			config=cfg,
		)
		flat_pivots = [p for p in pivots if p.strategy == "research_before_implement"]
		assert flat_pivots == []

	def test_config_cost_rise_ratio(self) -> None:
		# Default ratio=1.2 would trigger; raise to 2.0 so it doesn't
		cfg = StagnationConfig(cost_rise_ratio=2.0)
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[3, 3, 3],
			failure_history=[0, 0, 0],
			cost_history=[1.0, 1.5, 2.0],
			config=cfg,
		)
		reduce_pivots = [p for p in pivots if p.strategy == "reduce_and_focus"]
		assert reduce_pivots == []

	def test_config_failure_rate_threshold(self) -> None:
		# Raise threshold so 70% failures no longer trigger
		cfg = StagnationConfig(failure_rate_threshold=0.8)
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 5, 5],
			completion_history=[1, 2, 3],
			failure_history=[2, 3, 4],
			cost_history=[0.1, 0.2, 0.3],
			config=cfg,
		)
		diag_pivots = [p for p in pivots if p.strategy == "diagnose_systemic"]
		assert diag_pivots == []

	def test_default_config_matches_original_behavior(self) -> None:
		"""StagnationConfig defaults should produce the same results as no config."""
		args = dict(
			cycle_number=5,
			test_history=[10, 10, 10],
			completion_history=[2, 2, 2],
			failure_history=[5, 6, 7],
			cost_history=[1.0, 1.5, 2.5],
			threshold=3,
		)
		pivots_no_config = analyze_stagnation(**args)
		pivots_with_config = analyze_stagnation(**args, config=StagnationConfig())
		assert {p.strategy for p in pivots_no_config} == {p.strategy for p in pivots_with_config}


class TestRepeatedErrors:
	def test_same_error_across_agents_triggers_pivot(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			error_messages=[
				"ModuleNotFoundError: No module named 'foo'",
				"ModuleNotFoundError: No module named 'foo'",
				"Some other error",
			],
		)
		err_pivots = [p for p in pivots if p.strategy == "research_systemic_error"]
		assert len(err_pivots) == 1
		assert "2 agents" in err_pivots[0].trigger
		assert err_pivots[0].severity == "critical"
		assert "ModuleNotFoundError" in err_pivots[0].details

	def test_no_repeated_errors_no_pivot(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			error_messages=["Error A", "Error B", "Error C"],
		)
		err_pivots = [p for p in pivots if p.strategy == "research_systemic_error"]
		assert err_pivots == []

	def test_empty_error_messages_no_pivot(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			error_messages=[],
		)
		err_pivots = [p for p in pivots if p.strategy == "research_systemic_error"]
		assert err_pivots == []

	def test_configurable_min_agents_for_repeated_errors(self) -> None:
		cfg = StagnationConfig(repeated_error_min_agents=3)
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			config=cfg,
			error_messages=["same error", "same error"],  # only 2, need 3
		)
		err_pivots = [p for p in pivots if p.strategy == "research_systemic_error"]
		assert err_pivots == []

	def test_long_error_message_is_truncated(self) -> None:
		long_msg = "x" * 200
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			error_messages=[long_msg, long_msg],
		)
		err_pivots = [p for p in pivots if p.strategy == "research_systemic_error"]
		assert len(err_pivots) == 1
		assert err_pivots[0].trigger.endswith("...")
		assert len(err_pivots[0].trigger) < 200

	def test_multiple_repeated_errors_capped_at_3(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			error_messages=[
				"err_a", "err_a",
				"err_b", "err_b",
				"err_c", "err_c",
				"err_d", "err_d",
			],
		)
		err_pivots = [p for p in pivots if p.strategy == "research_systemic_error"]
		assert len(err_pivots) <= 3


class TestAgentChurn:
	def test_agent_churn_triggers_pivot(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			task_agent_counts={"task-1": 3, "task-2": 1},
		)
		churn_pivots = [p for p in pivots if p.strategy == "rethink_approach"]
		assert len(churn_pivots) == 1
		assert "task-1" in churn_pivots[0].trigger
		assert "3" in churn_pivots[0].trigger

	def test_no_churn_no_pivot(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			task_agent_counts={"task-1": 1, "task-2": 1},
		)
		churn_pivots = [p for p in pivots if p.strategy == "rethink_approach"]
		assert churn_pivots == []

	def test_configurable_churn_threshold(self) -> None:
		cfg = StagnationConfig(agent_churn_min_respawns=5)
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			config=cfg,
			task_agent_counts={"task-1": 3},  # below threshold of 5
		)
		churn_pivots = [p for p in pivots if p.strategy == "rethink_approach"]
		assert churn_pivots == []

	def test_high_churn_is_critical(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			task_agent_counts={"task-1": 5},
		)
		churn_pivots = [p for p in pivots if p.strategy == "rethink_approach"]
		assert len(churn_pivots) == 1
		assert churn_pivots[0].severity == "critical"

	def test_moderate_churn_is_warning(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			task_agent_counts={"task-1": 2},
		)
		churn_pivots = [p for p in pivots if p.strategy == "rethink_approach"]
		assert len(churn_pivots) == 1
		assert churn_pivots[0].severity == "warning"

	def test_worst_task_reported(self) -> None:
		"""When multiple tasks churn, the worst one is reported."""
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			task_agent_counts={"task-1": 3, "task-2": 5},
		)
		churn_pivots = [p for p in pivots if p.strategy == "rethink_approach"]
		assert len(churn_pivots) == 1
		assert "task-2" in churn_pivots[0].trigger
		assert "5" in churn_pivots[0].trigger


class TestCostEfficiency:
	def test_cost_efficiency_decline_detected(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=10,
			test_history=[5, 6, 7, 8],
			completion_history=[5, 5, 1, 1],
			failure_history=[0, 0, 0, 0],
			cost_history=[1.0, 1.0, 1.0, 1.0],
		)
		eff_pivots = [p for p in pivots if "Cost efficiency dropped" in p.trigger]
		assert len(eff_pivots) == 1
		assert eff_pivots[0].strategy == "reduce_and_focus"

	def test_cost_efficiency_insufficient_data(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=3,
			test_history=[5, 5],
			completion_history=[5, 5],
			failure_history=[0, 0],
			cost_history=[1.0, 1.0],
		)
		eff_pivots = [p for p in pivots if "Cost efficiency dropped" in p.trigger]
		assert eff_pivots == []


class TestFileHotspots:
	def test_file_hotspot_detected(self) -> None:
		pivots = analyze_stagnation(
			cycle_number=5,
			test_history=[5, 6, 7],
			completion_history=[1, 2, 3],
			failure_history=[0, 0, 0],
			cost_history=[0.1, 0.2, 0.3],
			file_changes={
				"agent1": ["a.py", "b.py"],
				"agent2": ["a.py", "c.py"],
				"agent3": ["a.py"],
			},
		)
		hotspot_pivots = [p for p in pivots if p.strategy == "serialize_hotspot"]
		assert len(hotspot_pivots) == 1
		assert "a.py" in hotspot_pivots[0].trigger
		assert "3 agents" in hotspot_pivots[0].trigger


class TestPivotsToDecisions:
	def test_research_systemic_error_decision(self) -> None:
		pivots = [PivotRecommendation(
			trigger="Same error across 3 agents: ImportError",
			strategy="research_systemic_error",
			severity="critical",
			details="Investigate the root cause.",
		)]
		decisions = pivots_to_decisions(pivots)
		assert len(decisions) == 1
		assert decisions[0]["type"] == "create_task"
		assert "Research systemic error" in decisions[0]["payload"]["title"]
		assert decisions[0]["priority"] == 10

	def test_serialize_hotspot_decision(self) -> None:
		pivots = [PivotRecommendation(
			trigger="File hotspots detected: a.py (3 agents)",
			strategy="serialize_hotspot",
			severity="warning",
			details="Serialize work on contested files to avoid merge conflicts",
		)]
		decisions = pivots_to_decisions(pivots)
		assert len(decisions) == 1
		assert decisions[0]["type"] == "adjust"
		assert decisions[0]["payload"]["max_agents"] == 1
		assert decisions[0]["priority"] == 9

	def test_rethink_approach_decision(self) -> None:
		pivots = [PivotRecommendation(
			trigger="Agent churn: task-1 attempted by 4 agents",
			strategy="rethink_approach",
			severity="warning",
			details="Find alternative approach.",
		)]
		decisions = pivots_to_decisions(pivots)
		assert len(decisions) == 1
		assert decisions[0]["type"] == "create_task"
		assert "Rethink approach" in decisions[0]["payload"]["title"]
		assert decisions[0]["priority"] == 9


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
