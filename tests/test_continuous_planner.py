"""Tests for ContinuousPlanner."""

from __future__ import annotations

from unittest.mock import AsyncMock

from mission_control.config import ContinuousConfig, MissionConfig, PlannerConfig, TargetConfig
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.models import Handoff, Mission, Plan, PlanNode, WorkUnit


def _config() -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(name="test", path="/tmp/test", objective="Build API")
	mc.planner = PlannerConfig(max_depth=2)
	mc.continuous = ContinuousConfig(backlog_min_size=2)
	return mc


def _mission() -> Mission:
	return Mission(id="m1", objective="Build a production API")


class TestIngestHandoff:
	def test_accumulates_discoveries(self) -> None:
		config = _config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		h = Handoff(
			discoveries=["Found auth module", "Tests need fixtures"],
			concerns=["Rate limiting missing"],
		)
		planner.ingest_handoff(h)

		assert len(planner._discoveries) == 2
		assert "Found auth module" in planner._discoveries
		assert len(planner._concerns) == 1

	def test_handles_empty_lists(self) -> None:
		config = _config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		h = Handoff(discoveries=[], concerns=[])
		planner.ingest_handoff(h)

		assert len(planner._discoveries) == 0

	def test_handles_empty_handoff(self) -> None:
		config = _config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		h = Handoff()
		planner.ingest_handoff(h)

		assert len(planner._discoveries) == 0
		assert len(planner._concerns) == 0


class TestBacklogSize:
	def test_initial_backlog_empty(self) -> None:
		planner = ContinuousPlanner(_config(), Database(":memory:"))
		assert planner.backlog_size == 0


class TestGetNextUnits:
	async def test_empty_backlog_triggers_replan(self) -> None:
		"""When backlog is empty (< min_size), LLM is invoked."""
		config = _config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		# Mock the inner planner
		mock_wu = WorkUnit(id="wu1", plan_id="p1", title="Task 1")
		mock_plan = Plan(id="p1", objective="test")
		mock_root = PlanNode(id="root", plan_id="p1", node_type="branch", strategy="leaves")
		mock_root._child_leaves = [  # type: ignore[attr-defined]
			(PlanNode(id="leaf1", node_type="leaf"), mock_wu),
		]

		planner._inner.plan_round = AsyncMock(return_value=(mock_plan, mock_root))

		mission = _mission()
		plan, units, epoch = await planner.get_next_units(mission, max_units=3)

		assert len(units) == 1
		assert units[0].title == "Task 1"
		assert epoch.number == 1
		assert epoch.mission_id == "m1"
		planner._inner.plan_round.assert_called_once()

	async def test_populated_backlog_returns_without_llm(self) -> None:
		"""When backlog has enough units, returns from backlog without LLM call."""
		config = _config()
		config.continuous.backlog_min_size = 2
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		# Pre-populate backlog
		planner._backlog = [
			WorkUnit(id="wu1", title="Task 1"),
			WorkUnit(id="wu2", title="Task 2"),
			WorkUnit(id="wu3", title="Task 3"),
		]

		planner._inner.plan_round = AsyncMock()  # Should NOT be called

		mission = _mission()
		plan, units, epoch = await planner.get_next_units(mission, max_units=2)

		assert len(units) == 2
		assert units[0].title == "Task 1"
		assert units[1].title == "Task 2"
		assert planner.backlog_size == 1  # 1 remains
		planner._inner.plan_round.assert_not_called()

	async def test_epoch_increments(self) -> None:
		"""Each replan creates a new epoch with incrementing number."""
		config = _config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		mock_wu1 = WorkUnit(id="wu1", title="T1")
		mock_wu2 = WorkUnit(id="wu2", title="T2")

		call_count = 0

		async def mock_plan_round(**kwargs):
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			root = PlanNode(id=f"root{call_count}", plan_id=plan.id, strategy="leaves")
			wu = mock_wu1 if call_count == 1 else mock_wu2
			root._child_leaves = [(PlanNode(id=f"l{call_count}", node_type="leaf"), wu)]  # type: ignore[attr-defined]
			return plan, root

		planner._inner.plan_round = AsyncMock(side_effect=mock_plan_round)

		mission = _mission()

		# First call -> epoch 1
		_, _, epoch1 = await planner.get_next_units(mission)
		assert epoch1.number == 1

		# Second call -> epoch 2
		_, _, epoch2 = await planner.get_next_units(mission)
		assert epoch2.number == 2

	async def test_replan_backlog_overflow(self) -> None:
		"""Extra units from replanning are stored in backlog."""
		config = _config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		# Create a plan with 4 units
		mock_units = [WorkUnit(id=f"wu{i}", title=f"Task {i}") for i in range(4)]
		mock_plan = Plan(id="p1", objective="test")
		mock_root = PlanNode(id="root", plan_id="p1", strategy="leaves")
		mock_root._child_leaves = [  # type: ignore[attr-defined]
			(PlanNode(id=f"l{i}", node_type="leaf"), wu)
			for i, wu in enumerate(mock_units)
		]

		planner._inner.plan_round = AsyncMock(return_value=(mock_plan, mock_root))

		mission = _mission()
		plan, units, epoch = await planner.get_next_units(mission, max_units=2)

		# Should serve 2, backlog 2
		assert len(units) == 2
		assert planner.backlog_size == 2
