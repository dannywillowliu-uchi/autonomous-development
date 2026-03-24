"""Tests for BudgetConfig cost-cap enforcement in swarm controller."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autodev.config import BudgetConfig, SwarmConfig
from autodev.swarm.budget import BudgetEnforcer
from autodev.swarm.controller import SwarmController
from autodev.swarm.models import (
	AgentStatus,
	SwarmTask,
	TaskStatus,
)


def _make_config(tmp_path: Path, max_per_run_usd: float = 50.0) -> MagicMock:
	config = MagicMock()
	config.target.name = "test-project"
	config.target.objective = "Build a compiler"
	config.target.resolved_path = str(tmp_path)
	config.notification = MagicMock()
	config.scheduler.budget = BudgetConfig(max_per_run_usd=max_per_run_usd)
	return config


def _make_swarm_config(**overrides: object) -> SwarmConfig:
	sc = SwarmConfig()
	for k, v in overrides.items():
		setattr(sc, k, v)
	return sc


def _make_db() -> MagicMock:
	db = MagicMock()
	db.get_knowledge_for_mission.return_value = []
	db.get_tool_failure_summary.return_value = []
	db.get_mcp_health_summary.return_value = {}
	return db


def _make_controller(tmp_path: Path, max_per_run_usd: float = 50.0, **swarm_overrides: object) -> SwarmController:
	return SwarmController(
		_make_config(tmp_path, max_per_run_usd=max_per_run_usd),
		_make_swarm_config(**swarm_overrides),
		_make_db(),
	)


# --- 2a: _check_budget ---


class TestCheckBudget:
	def test_under_limit(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=50.0)
		ctrl._total_cost_usd = 25.0
		assert not ctrl._check_budget()

	def test_at_limit(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=50.0)
		ctrl._total_cost_usd = 50.0
		assert ctrl._check_budget()

	def test_over_limit(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=50.0)
		ctrl._total_cost_usd = 75.0
		assert ctrl._check_budget()

	def test_zero_budget_means_uncapped(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=0.0)
		ctrl._total_cost_usd = 9999.0
		assert not ctrl._check_budget()

	def test_negative_budget_means_uncapped(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=-1.0)
		ctrl._total_cost_usd = 100.0
		assert not ctrl._check_budget()


# --- 2a: spawn blocked on budget exhausted ---


class TestBudgetBlocksSpawning:
	@pytest.mark.asyncio
	async def test_spawn_blocked_when_budget_exhausted(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=10.0)
		ctrl._total_cost_usd = 15.0

		result = await ctrl._handle_spawn({"name": "test-agent", "role": "general"})
		assert result.get("spawned") is False
		assert "Budget exhausted" in result.get("error", "")

	@pytest.mark.asyncio
	async def test_spawn_allowed_when_under_budget(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=100.0, max_agents=5)
		ctrl._total_cost_usd = 10.0

		# Mock the subprocess spawn to avoid actually launching claude
		async def _fake_spawn(*args: object, **kwargs: object) -> None:
			return None
		ctrl._spawn_claude_session = _fake_spawn  # type: ignore[assignment]

		result = await ctrl._handle_spawn({"name": "test-agent", "role": "general"})
		# Agent was created (even if process failed, it got past budget gate)
		assert "Budget exhausted" not in result.get("error", "")


# --- 2a: graceful shutdown ---


class TestGracefulShutdownOnBudget:
	def test_scaling_recommends_full_scaledown_on_exhausted(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=10.0)
		ctrl._total_cost_usd = 15.0

		# Add some active agents
		from autodev.swarm.models import SwarmAgent
		for i in range(3):
			agent = SwarmAgent(name=f"agent-{i}", status=AgentStatus.WORKING)
			ctrl._agents[agent.id] = agent

		rec = ctrl.get_scaling_recommendation()
		assert rec["scale_up"] == 0
		assert rec["scale_down"] == 3  # all active agents


# --- 2b: throughput-based scaling ---


class TestScalingCostRate:
	def test_cost_rate_scaledown_when_unaffordable(self, tmp_path: Path) -> None:
		"""When budget can't cover pending tasks, recommend scale down."""
		ctrl = _make_controller(tmp_path, max_per_run_usd=50.0)
		ctrl._total_cost_usd = 40.0  # $10 remaining

		# 2 completed tasks -> avg $20/task
		for i in range(2):
			task = SwarmTask(title=f"done-{i}", status=TaskStatus.COMPLETED)
			ctrl._tasks[task.id] = task

		# 5 pending tasks (would need ~$100 at $20/task, only $10 left)
		for i in range(5):
			task = SwarmTask(title=f"pending-{i}", status=TaskStatus.PENDING)
			ctrl._tasks[task.id] = task

		# 2 active agents
		from autodev.swarm.models import SwarmAgent
		for i in range(2):
			agent = SwarmAgent(name=f"worker-{i}", status=AgentStatus.WORKING)
			ctrl._agents[agent.id] = agent

		rec = ctrl.get_scaling_recommendation()
		assert rec["scale_down"] > 0
		assert rec["scale_up"] == 0

	def test_default_scaling_when_budget_uncapped(self, tmp_path: Path) -> None:
		"""With no budget cap, use default ratio-based scaling."""
		ctrl = _make_controller(tmp_path, max_per_run_usd=0.0)
		ctrl._total_cost_usd = 500.0

		# 10 pending, 1 active -> should scale up
		for i in range(10):
			task = SwarmTask(title=f"pending-{i}", status=TaskStatus.PENDING)
			ctrl._tasks[task.id] = task

		from autodev.swarm.models import SwarmAgent
		agent = SwarmAgent(name="worker-0", status=AgentStatus.WORKING)
		ctrl._agents[agent.id] = agent

		rec = ctrl.get_scaling_recommendation()
		assert rec["scale_up"] > 0

	def test_default_scaling_when_no_cost_data(self, tmp_path: Path) -> None:
		"""With budget set but no completed tasks, fall back to ratio logic."""
		ctrl = _make_controller(tmp_path, max_per_run_usd=50.0)
		ctrl._total_cost_usd = 5.0

		# 10 pending, 1 active, 0 completed -> no cost_per_task data
		for i in range(10):
			task = SwarmTask(title=f"pending-{i}", status=TaskStatus.PENDING)
			ctrl._tasks[task.id] = task

		from autodev.swarm.models import SwarmAgent
		agent = SwarmAgent(name="worker-0", status=AgentStatus.WORKING)
		ctrl._agents[agent.id] = agent

		rec = ctrl.get_scaling_recommendation()
		# Falls back to default ratio logic
		assert rec["scale_up"] > 0


# --- 2c: planner context includes budget ---


class TestPlannerContextBudget:
	def test_budget_status_in_state(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=50.0)
		ctrl._total_cost_usd = 20.0

		# Add 4 completed tasks
		for i in range(4):
			task = SwarmTask(title=f"done-{i}", status=TaskStatus.COMPLETED)
			ctrl._tasks[task.id] = task

		status = ctrl._get_budget_status()
		assert status["budget_used_usd"] == 20.0
		assert status["budget_limit_usd"] == 50.0
		assert status["budget_remaining_usd"] == 30.0
		assert status["cost_per_task_avg"] == 5.0  # $20 / 4 tasks
		assert status["estimated_tasks_affordable"] == 6  # $30 / $5

	def test_budget_status_no_completed_tasks(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=50.0)
		ctrl._total_cost_usd = 0.0

		status = ctrl._get_budget_status()
		assert status["cost_per_task_avg"] == 0.0
		assert status["estimated_tasks_affordable"] == -1  # unknown

	def test_build_state_includes_budget(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=50.0)
		ctrl._total_cost_usd = 10.0

		state = ctrl.build_state()
		assert "budget_used_usd" in state.budget_status
		assert "budget_limit_usd" in state.budget_status
		assert state.budget_status["budget_used_usd"] == 10.0

	def test_render_includes_budget_section(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=50.0)
		ctrl._total_cost_usd = 20.0

		state = ctrl.build_state()
		rendered = ctrl.render_state(state)
		assert "Budget Status" in rendered
		assert "$20.00" in rendered
		assert "$50.00" in rendered

	def test_render_shows_exhausted_warning(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path, max_per_run_usd=50.0)
		ctrl._total_cost_usd = 60.0

		state = ctrl.build_state()
		rendered = ctrl.render_state(state)
		assert "BUDGET EXHAUSTED" in rendered


# --- BudgetEnforcer unit tests ---


class TestBudgetConfigDefaults:
	def test_budget_config_defaults(self) -> None:
		bc = BudgetConfig()
		assert bc.max_per_run_usd == 50.0
		assert bc.max_agent_spawns == 20
		assert bc.max_cost_rate_usd_per_min == 2.0
		assert bc.warn_threshold_pct == 0.8
		assert bc.enforce is True

	def test_budget_config_custom_values(self) -> None:
		bc = BudgetConfig(
			max_per_run_usd=100.0,
			max_agent_spawns=10,
			max_cost_rate_usd_per_min=5.0,
			warn_threshold_pct=0.9,
			enforce=False,
		)
		assert bc.max_per_run_usd == 100.0
		assert bc.max_agent_spawns == 10
		assert bc.max_cost_rate_usd_per_min == 5.0
		assert bc.warn_threshold_pct == 0.9
		assert bc.enforce is False


class TestIsOverBudget:
	def test_over_budget_when_at_limit(self) -> None:
		bc = BudgetConfig(max_per_run_usd=50.0)
		assert bc.is_over_budget(50.0) is True

	def test_over_budget_when_exceeded(self) -> None:
		bc = BudgetConfig(max_per_run_usd=50.0)
		assert bc.is_over_budget(75.0) is True

	def test_not_over_budget_when_under(self) -> None:
		bc = BudgetConfig(max_per_run_usd=50.0)
		assert bc.is_over_budget(25.0) is False

	def test_zero_budget_never_over(self) -> None:
		bc = BudgetConfig(max_per_run_usd=0.0)
		assert bc.is_over_budget(9999.0) is False

	def test_negative_budget_never_over(self) -> None:
		bc = BudgetConfig(max_per_run_usd=-1.0)
		assert bc.is_over_budget(100.0) is False


class TestIsWarning:
	def test_warning_at_threshold(self) -> None:
		bc = BudgetConfig(max_per_run_usd=100.0, warn_threshold_pct=0.8)
		assert bc.is_warning(80.0) is True

	def test_warning_above_threshold(self) -> None:
		bc = BudgetConfig(max_per_run_usd=100.0, warn_threshold_pct=0.8)
		assert bc.is_warning(90.0) is True

	def test_no_warning_below_threshold(self) -> None:
		bc = BudgetConfig(max_per_run_usd=100.0, warn_threshold_pct=0.8)
		assert bc.is_warning(50.0) is False

	def test_zero_budget_no_warning(self) -> None:
		bc = BudgetConfig(max_per_run_usd=0.0, warn_threshold_pct=0.8)
		assert bc.is_warning(9999.0) is False

	def test_zero_threshold_no_warning(self) -> None:
		bc = BudgetConfig(max_per_run_usd=100.0, warn_threshold_pct=0.0)
		assert bc.is_warning(99.0) is False


class TestEnforceFalseAllowsOverspend:
	def test_is_over_budget_still_returns_true(self) -> None:
		bc = BudgetConfig(max_per_run_usd=50.0, enforce=False)
		assert bc.is_over_budget(60.0) is True

	def test_enforce_flag_is_separate_from_detection(self) -> None:
		bc = BudgetConfig(max_per_run_usd=50.0, enforce=False)
		assert bc.is_over_budget(60.0) is True
		assert bc.enforce is False


class TestCostCeilingStopsSpawning:
	def test_ceiling_triggers_stop_spawning(self) -> None:
		budget = BudgetConfig(max_per_run_usd=50.0)
		enforcer = BudgetEnforcer(budget)
		result = enforcer.check_cycle(cumulative_cost_usd=55.0, current_max_agents=4)
		assert result.stop_spawning is True
		assert enforcer.is_exhausted is True

	def test_under_ceiling_allows_spawning(self) -> None:
		budget = BudgetConfig(max_per_run_usd=50.0)
		enforcer = BudgetEnforcer(budget)
		result = enforcer.check_cycle(cumulative_cost_usd=25.0, current_max_agents=4)
		assert result.stop_spawning is False
		assert enforcer.is_exhausted is False

	def test_exactly_at_ceiling(self) -> None:
		budget = BudgetConfig(max_per_run_usd=50.0)
		enforcer = BudgetEnforcer(budget)
		result = enforcer.check_cycle(cumulative_cost_usd=50.0, current_max_agents=4)
		assert result.stop_spawning is True

	def test_zero_budget_means_uncapped(self) -> None:
		budget = BudgetConfig(max_per_run_usd=0.0)
		enforcer = BudgetEnforcer(budget)
		result = enforcer.check_cycle(cumulative_cost_usd=9999.0, current_max_agents=4)
		assert result.stop_spawning is False


class TestCostRateReducesAgents:
	def test_high_rate_reduces_max_agents(self) -> None:
		budget = BudgetConfig(max_cost_rate_usd_per_min=2.0)
		enforcer = BudgetEnforcer(budget)
		# Set initial state: 0 cost at time T
		enforcer._last_cost = 0.0
		enforcer._last_time = time.monotonic() - 60  # 1 minute ago
		# Spent $5 in 1 minute = $5/min > $2/min limit
		result = enforcer.check_cycle(cumulative_cost_usd=5.0, current_max_agents=4)
		assert result.new_max_agents == 3

	def test_acceptable_rate_no_change(self) -> None:
		budget = BudgetConfig(max_cost_rate_usd_per_min=2.0)
		enforcer = BudgetEnforcer(budget)
		enforcer._last_cost = 0.0
		enforcer._last_time = time.monotonic() - 60  # 1 minute ago
		# Spent $1 in 1 minute = $1/min < $2/min limit
		result = enforcer.check_cycle(cumulative_cost_usd=1.0, current_max_agents=4)
		assert result.new_max_agents is None

	def test_rate_wont_reduce_below_1(self) -> None:
		budget = BudgetConfig(max_cost_rate_usd_per_min=1.0)
		enforcer = BudgetEnforcer(budget)
		enforcer._last_cost = 0.0
		enforcer._last_time = time.monotonic() - 60
		# Already at 1 agent, won't reduce further
		result = enforcer.check_cycle(cumulative_cost_usd=10.0, current_max_agents=1)
		assert result.new_max_agents is None

	def test_zero_rate_limit_disables_check(self) -> None:
		budget = BudgetConfig(max_cost_rate_usd_per_min=0.0)
		enforcer = BudgetEnforcer(budget)
		enforcer._last_cost = 0.0
		enforcer._last_time = time.monotonic() - 60
		result = enforcer.check_cycle(cumulative_cost_usd=100.0, current_max_agents=4)
		assert result.new_max_agents is None


class TestWarnThresholdFiresOnce:
	def test_warning_fires_at_threshold(self) -> None:
		budget = BudgetConfig(max_per_run_usd=100.0, warn_threshold_pct=0.8)
		enforcer = BudgetEnforcer(budget)
		result = enforcer.check_cycle(cumulative_cost_usd=80.0, current_max_agents=4)
		assert result.warning_fired is True
		assert enforcer.warn_fired is True

	def test_warning_does_not_repeat(self) -> None:
		budget = BudgetConfig(max_per_run_usd=100.0, warn_threshold_pct=0.8)
		enforcer = BudgetEnforcer(budget)
		# First cycle crosses threshold
		r1 = enforcer.check_cycle(cumulative_cost_usd=85.0, current_max_agents=4)
		assert r1.warning_fired is True
		# Second cycle still above threshold but should not fire again
		r2 = enforcer.check_cycle(cumulative_cost_usd=90.0, current_max_agents=4)
		assert r2.warning_fired is False

	def test_warning_below_threshold(self) -> None:
		budget = BudgetConfig(max_per_run_usd=100.0, warn_threshold_pct=0.8)
		enforcer = BudgetEnforcer(budget)
		result = enforcer.check_cycle(cumulative_cost_usd=50.0, current_max_agents=4)
		assert result.warning_fired is False
		assert enforcer.warn_fired is False


class TestNoBudgetConfigNoEnforcement:
	def test_none_budget_no_enforcement(self) -> None:
		enforcer = BudgetEnforcer(budget=None)
		result = enforcer.check_cycle(cumulative_cost_usd=9999.0, current_max_agents=4)
		assert result.stop_spawning is False
		assert result.warning_fired is False
		assert result.new_max_agents is None
		assert enforcer.is_exhausted is False

	def test_swarm_config_budget_defaults_none(self) -> None:
		sc = SwarmConfig()
		assert sc.budget is None
