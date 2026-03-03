"""Tests for planner subprocess cost tracking and propagation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import (
	DeliberationConfig,
	MissionConfig,
	PlannerConfig,
	TargetConfig,
)
from mission_control.db import Database
from mission_control.deliberative_planner import DeliberativePlanner
from mission_control.models import CriticFinding, Epoch, Mission, Plan, WorkUnit
from mission_control.recursive_planner import (
	PlannerResult,
	RecursivePlanner,
	_parse_subprocess_cost,
)

# -- _parse_subprocess_cost tests --


class TestParseSubprocessCost:
	def test_explicit_cost_pattern(self) -> None:
		"""Parses 'cost: $X.XX' from stderr."""
		stderr = "Session complete. cost: $0.42\n"
		assert _parse_subprocess_cost(stderr, 1.0) == 0.42

	def test_cost_pattern_case_insensitive(self) -> None:
		"""Case-insensitive matching of cost patterns."""
		stderr = "Cost: $1.23\nDone."
		assert _parse_subprocess_cost(stderr, 5.0) == 1.23

	def test_spent_pattern(self) -> None:
		"""Parses 'spent: $X.XX' from stderr."""
		stderr = "Tokens spent: $0.89\n"
		assert _parse_subprocess_cost(stderr, 1.0) == 0.89

	def test_usage_pattern(self) -> None:
		"""Parses 'usage: $X.XX' from stderr."""
		stderr = "API usage: $2.50\n"
		assert _parse_subprocess_cost(stderr, 1.0) == 2.50

	def test_bare_dollar_amount(self) -> None:
		"""Falls back to bare $X.XX pattern when no keyword match."""
		stderr = "Total: $0.73 for this session\n"
		assert _parse_subprocess_cost(stderr, 1.0) == 0.73

	def test_empty_stderr_returns_fallback(self) -> None:
		"""Empty stderr returns the budget fallback."""
		assert _parse_subprocess_cost("", 1.50) == 1.50

	def test_no_cost_info_returns_fallback(self) -> None:
		"""Stderr with no cost info returns fallback."""
		stderr = "Loading tools...\nReady.\n"
		assert _parse_subprocess_cost(stderr, 2.0) == 2.0

	def test_multiple_dollar_amounts_uses_keyword(self) -> None:
		"""When keyword pattern matches, uses that over bare dollar."""
		stderr = "Loaded $0.01 cache. Session cost: $1.50\n"
		assert _parse_subprocess_cost(stderr, 5.0) == 1.50

	def test_fallback_with_none_cost(self) -> None:
		"""Non-parseable dollar amounts fall through to fallback."""
		stderr = "Some text with no numbers\n"
		assert _parse_subprocess_cost(stderr, 0.10) == 0.10


# -- PlannerResult cost_usd field --


class TestPlannerResultCost:
	def test_default_cost_is_zero(self) -> None:
		"""PlannerResult.cost_usd defaults to 0.0."""
		result = PlannerResult(type="leaves", units=[])
		assert result.cost_usd == 0.0

	def test_cost_can_be_set(self) -> None:
		"""PlannerResult.cost_usd can be set explicitly."""
		result = PlannerResult(type="leaves", units=[], cost_usd=1.50)
		assert result.cost_usd == 1.50


# -- plan_round returns cost --


def _config() -> MissionConfig:
	config = MissionConfig(
		planner=PlannerConfig(budget_per_call_usd=0.50),
	)
	config.target.path = "/tmp/test-target-project"
	return config


class TestPlanRoundCost:
	@pytest.mark.asyncio
	async def test_plan_round_returns_cost(self) -> None:
		"""plan_round returns (Plan, list[WorkUnit], float) with cost."""
		planner = RecursivePlanner(_config(), MagicMock())
		result = PlannerResult(
			type="leaves",
			units=[{"title": "Task", "description": "Do it", "files_hint": "a.py", "priority": 1}],
			cost_usd=0.35,
		)
		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=result):
			plan, units, cost = await planner.plan_round("Test objective")

		assert cost == 0.35
		assert len(units) == 1

	@pytest.mark.asyncio
	async def test_plan_round_accumulates_retry_cost(self) -> None:
		"""When planner retries (fallback parse), costs are accumulated."""
		planner = RecursivePlanner(_config(), MagicMock())
		# First call returns fallback (triggers retry), second returns real result
		fallback_result = PlannerResult(
			type="leaves",
			units=[{"title": "Execute scope", "description": "fallback", "files_hint": "", "priority": 1}],
			cost_usd=0.20,
		)
		real_result = PlannerResult(
			type="leaves",
			units=[{"title": "Real task", "description": "real", "files_hint": "a.py", "priority": 1}],
			cost_usd=0.30,
		)
		with patch.object(
			planner, "_run_planner_subprocess",
			new_callable=AsyncMock,
			side_effect=[fallback_result, real_result],
		):
			plan, units, cost = await planner.plan_round("Test objective")

		# Both calls' costs should be accumulated
		assert cost == pytest.approx(0.50)


# -- Epoch planner_cost_usd --


class TestEpochPlannerCost:
	def test_default_is_zero(self) -> None:
		"""Epoch.planner_cost_usd defaults to 0.0."""
		epoch = Epoch()
		assert epoch.planner_cost_usd == 0.0

	def test_can_be_set(self) -> None:
		"""Epoch.planner_cost_usd can be set."""
		epoch = Epoch(planner_cost_usd=1.25)
		assert epoch.planner_cost_usd == 1.25


# -- Deliberative planner cost accumulation --


def _delib_config(tmp_path: Path) -> MissionConfig:
	return MissionConfig(
		target=TargetConfig(path=str(tmp_path), name="test"),
		planner=PlannerConfig(budget_per_call_usd=0.50),
		deliberation=DeliberationConfig(
			max_rounds=3, critic_budget_usd=1.0, timeout=30,
		),
	)


def _mission() -> Mission:
	return Mission(id="m1", objective="Build a REST API")


def _mock_units(count: int = 2) -> list[WorkUnit]:
	return [
		WorkUnit(
			id=f"wu-{i}", plan_id="p1",
			title=f"Task {i}", files_hint=f"src/mod{i}.py",
			priority=i + 1,
		)
		for i in range(count)
	]


class TestDeliberativeCostAccumulation:
	@pytest.mark.asyncio
	async def test_single_round_accumulates_planner_and_critic(self, tmp_path: Path) -> None:
		"""Single round: planner cost + critic cost = epoch.planner_cost_usd."""
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_delib_config(tmp_path), db)

		units = _mock_units(2)
		mock_plan = Plan(id="p1", objective="test")

		async def mock_plan_round(**kwargs):
			return mock_plan, list(units), 0.40

		with (
			patch.object(
				planner._planner, "plan_round",
				new_callable=AsyncMock,
				side_effect=mock_plan_round,
			),
			patch.object(
				planner._critic, "review_plan",
				new_callable=AsyncMock,
				return_value=(CriticFinding(verdict="sufficient", confidence=0.9), 0.25),
			),
			patch.object(
				planner, "_gather_project_context",
				new_callable=AsyncMock,
				return_value="context",
			),
		):
			plan, result_units, epoch = await planner.get_next_units(_mission(), max_units=5)

		# planner: $0.40 + critic: $0.25 = $0.65
		assert epoch.planner_cost_usd == pytest.approx(0.65)

	@pytest.mark.asyncio
	async def test_multi_round_accumulates_all_costs(self, tmp_path: Path) -> None:
		"""Multi-round deliberation accumulates all planner + critic costs."""
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_delib_config(tmp_path), db)

		units = _mock_units(2)
		mock_plan = Plan(id="p1", objective="test")

		# Round 1: planner $0.30 + critic $0.15 (needs refinement)
		# Round 2: planner $0.25 + critic $0.10 (sufficient)
		plan_results = [
			(mock_plan, list(units), 0.30),
			(mock_plan, list(units), 0.25),
		]
		review_results = [
			(CriticFinding(verdict="needs_refinement", confidence=0.4), 0.15),
			(CriticFinding(verdict="sufficient", confidence=0.9), 0.10),
		]

		with (
			patch.object(
				planner._planner, "plan_round",
				new_callable=AsyncMock,
				side_effect=plan_results,
			),
			patch.object(
				planner._critic, "review_plan",
				new_callable=AsyncMock,
				side_effect=review_results,
			),
			patch.object(
				planner, "_gather_project_context",
				new_callable=AsyncMock,
				return_value="context",
			),
		):
			plan, result_units, epoch = await planner.get_next_units(_mission(), max_units=5)

		# Total: $0.30 + $0.15 + $0.25 + $0.10 = $0.80
		assert epoch.planner_cost_usd == pytest.approx(0.80)

	@pytest.mark.asyncio
	async def test_empty_units_no_critic_cost(self, tmp_path: Path) -> None:
		"""Empty units skips critic, only planner cost counted."""
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_delib_config(tmp_path), db)

		mock_plan = Plan(id="p1", objective="test")

		with (
			patch.object(
				planner._planner, "plan_round",
				new_callable=AsyncMock,
				return_value=(mock_plan, [], 0.20),
			),
			patch.object(
				planner._critic, "review_plan",
				new_callable=AsyncMock,
			) as mock_review,
			patch.object(
				planner, "_gather_project_context",
				new_callable=AsyncMock,
				return_value="context",
			),
		):
			plan, result_units, epoch = await planner.get_next_units(_mission(), max_units=3)

		# Only planner cost, no critic
		assert epoch.planner_cost_usd == pytest.approx(0.20)
		mock_review.assert_not_called()

	@pytest.mark.asyncio
	async def test_cost_appears_in_epoch(self, tmp_path: Path) -> None:
		"""Verify cost is accessible on the returned Epoch object."""
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_delib_config(tmp_path), db)

		units = _mock_units(1)
		mock_plan = Plan(id="p1", objective="test")

		with (
			patch.object(
				planner._planner, "plan_round",
				new_callable=AsyncMock,
				return_value=(mock_plan, units, 1.00),
			),
			patch.object(
				planner._critic, "review_plan",
				new_callable=AsyncMock,
				return_value=(CriticFinding(verdict="sufficient"), 0.50),
			),
			patch.object(
				planner, "_gather_project_context",
				new_callable=AsyncMock,
				return_value="context",
			),
		):
			_, _, epoch = await planner.get_next_units(_mission())

		assert isinstance(epoch, Epoch)
		assert epoch.planner_cost_usd == pytest.approx(1.50)
		assert epoch.planner_cost_usd > 0
