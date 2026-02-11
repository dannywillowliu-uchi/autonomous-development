"""Tests for round_controller module."""

from __future__ import annotations

import pytest

from mission_control.config import MissionConfig, RoundsConfig
from mission_control.db import Database
from mission_control.green_branch import FixupResult
from mission_control.models import Handoff, Mission, Plan
from mission_control.round_controller import (
	MissionResult,
	RoundController,
	RoundResult,
	_curate_discoveries,
)


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


@pytest.fixture()
def config() -> MissionConfig:
	cfg = MissionConfig()
	cfg.rounds = RoundsConfig(max_rounds=10, stall_threshold=3)
	return cfg


@pytest.fixture()
def controller(config: MissionConfig, db: Database) -> RoundController:
	return RoundController(config, db)


@pytest.fixture()
def mission() -> Mission:
	return Mission(objective="Build feature X", status="running")


# -- Dataclass defaults --


class TestMissionResultDefaults:
	def test_defaults(self) -> None:
		r = MissionResult()
		assert r.mission_id == ""
		assert r.objective == ""
		assert r.final_score == 0.0
		assert r.objective_met is False
		assert r.total_rounds == 0
		assert r.total_cost_usd == 0.0
		assert r.wall_time_seconds == 0.0
		assert r.stopped_reason == ""
		assert r.round_scores == []

	def test_round_scores_isolation(self) -> None:
		"""Each instance gets its own list."""
		a = MissionResult()
		b = MissionResult()
		a.round_scores.append(0.5)
		assert b.round_scores == []


class TestRoundResultDefaults:
	def test_defaults(self) -> None:
		r = RoundResult()
		assert r.round_id == ""
		assert r.number == 0
		assert r.score == 0.0
		assert r.objective_met is False
		assert r.total_units == 0
		assert r.completed_units == 0
		assert r.failed_units == 0
		assert r.discoveries == []
		assert r.cost_usd == 0.0

	def test_discoveries_isolation(self) -> None:
		"""Each instance gets its own list."""
		a = RoundResult()
		b = RoundResult()
		a.discoveries.append("found something")
		assert b.discoveries == []


# -- _curate_discoveries --


class TestCurateDiscoveries:
	def test_empty_list(self) -> None:
		assert _curate_discoveries([]) == []

	def test_short_list_fits(self) -> None:
		items = ["alpha", "beta", "gamma"]
		assert _curate_discoveries(items, max_chars=100) == items

	def test_exact_fit(self) -> None:
		"""Items that exactly fill the budget are all included."""
		items = ["aaa", "bbb"]  # 3 + 3 = 6
		assert _curate_discoveries(items, max_chars=6) == ["aaa", "bbb"]

	def test_exceeds_max_chars(self) -> None:
		items = ["aaa", "bbb", "ccc"]  # 3 + 3 + 3 = 9
		result = _curate_discoveries(items, max_chars=7)
		assert result == ["aaa", "bbb"]

	def test_single_item_over_budget(self) -> None:
		"""A first item that already exceeds the budget yields empty."""
		items = ["this is way too long"]
		result = _curate_discoveries(items, max_chars=5)
		assert result == []

	def test_preserves_order(self) -> None:
		items = ["first", "second", "third"]
		result = _curate_discoveries(items, max_chars=50)
		assert result == ["first", "second", "third"]


# -- _should_stop --


class TestShouldStop:
	def test_no_stop_conditions(self, controller: RoundController, mission: Mission) -> None:
		"""No stopping when running, under max_rounds, and not stalled."""
		mission.total_rounds = 0
		result = controller._should_stop(mission, [0.1, 0.3, 0.5])
		assert result == ""

	def test_user_stopped(self, controller: RoundController, mission: Mission) -> None:
		controller.running = False
		result = controller._should_stop(mission, [])
		assert result == "user_stopped"

	def test_max_rounds(self, controller: RoundController, mission: Mission) -> None:
		mission.total_rounds = 10  # config.rounds.max_rounds == 10
		result = controller._should_stop(mission, [0.5])
		assert result == "max_rounds"

	def test_max_rounds_exceeded(self, controller: RoundController, mission: Mission) -> None:
		mission.total_rounds = 15
		result = controller._should_stop(mission, [0.5])
		assert result == "max_rounds"

	def test_stalled_flat_scores(self, controller: RoundController, mission: Mission) -> None:
		"""Stall detected when last N scores have < 0.01 spread."""
		mission.total_rounds = 0
		scores = [0.5, 0.5, 0.5]  # stall_threshold = 3
		result = controller._should_stop(mission, scores)
		assert result == "stalled"

	def test_stalled_near_identical(self, controller: RoundController, mission: Mission) -> None:
		"""Scores within 0.01 range count as stalled."""
		mission.total_rounds = 0
		scores = [0.50, 0.505, 0.509]  # spread = 0.009 < 0.01
		result = controller._should_stop(mission, scores)
		assert result == "stalled"

	def test_not_stalled_with_improvement(self, controller: RoundController, mission: Mission) -> None:
		"""Scores with > 0.01 spread are not stalled."""
		mission.total_rounds = 0
		scores = [0.5, 0.5, 0.52]  # spread = 0.02 >= 0.01
		result = controller._should_stop(mission, scores)
		assert result == ""

	def test_not_stalled_too_few_scores(self, controller: RoundController, mission: Mission) -> None:
		"""Fewer scores than stall_threshold cannot trigger stall."""
		mission.total_rounds = 0
		scores = [0.5, 0.5]  # only 2, threshold is 3
		result = controller._should_stop(mission, scores)
		assert result == ""

	def test_user_stopped_takes_priority(self, controller: RoundController, mission: Mission) -> None:
		"""user_stopped is checked first, even if max_rounds also applies."""
		controller.running = False
		mission.total_rounds = 10
		result = controller._should_stop(mission, [0.5, 0.5, 0.5])
		assert result == "user_stopped"


# -- stop() --


class TestStop:
	def test_stop_sets_running_false(self, controller: RoundController) -> None:
		assert controller.running is True
		controller.stop()
		assert controller.running is False


# -- _build_round_summary --


class TestBuildRoundSummary:
	def test_basic_summary(self, controller: RoundController) -> None:
		plan = Plan(total_units=3)
		handoffs = [
			Handoff(status="completed", summary="Implemented auth"),
			Handoff(status="completed", summary="Added tests"),
			Handoff(status="failed", summary="Lint fix failed"),
		]
		fixup = FixupResult(promoted=True, fixup_attempts=2)

		result = controller._build_round_summary(plan, handoffs, fixup)

		assert "3 units planned" in result
		assert "2 completed, 1 failed" in result
		assert "Implemented auth" in result
		assert "Added tests" in result
		assert "Lint fix failed" in result
		assert "verification passed after 2 attempt(s)" in result

	def test_fixup_failing(self, controller: RoundController) -> None:
		plan = Plan(total_units=1)
		handoffs = [Handoff(status="completed", summary="Did work")]
		fixup = FixupResult(promoted=False, fixup_attempts=3)

		result = controller._build_round_summary(plan, handoffs, fixup)

		assert "verification still failing" in result

	def test_no_handoffs(self, controller: RoundController) -> None:
		plan = Plan(total_units=0)
		fixup = FixupResult(promoted=False, fixup_attempts=0)

		result = controller._build_round_summary(plan, [], fixup)

		assert "0 units planned" in result
		assert "0 completed, 0 failed" in result

	def test_handoffs_without_summaries(self, controller: RoundController) -> None:
		"""Handoffs with empty summary are excluded from work done section."""
		plan = Plan(total_units=2)
		handoffs = [
			Handoff(status="completed", summary=""),
			Handoff(status="completed", summary="Real work"),
		]
		fixup = FixupResult(promoted=True, fixup_attempts=1)

		result = controller._build_round_summary(plan, handoffs, fixup)

		assert "Work done:" in result
		assert "Real work" in result
		# The empty-summary handoff should not add a blank bullet
		lines = result.split("\n")
		work_bullets = [line for line in lines if line.startswith("- ")]
		assert len(work_bullets) == 1
