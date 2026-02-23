"""Tests for ContinuousPlanner."""

from __future__ import annotations

import time
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


class TestCausalContextAndSnapshotDelegation:
	def test_set_causal_context_delegates(self) -> None:
		"""set_causal_context delegates to inner planner."""
		config = _config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)
		planner.set_causal_context("model=opus: 9% failure")
		assert planner._inner._causal_risks == "model=opus: 9% failure"

	def test_set_project_snapshot_delegates(self) -> None:
		"""set_project_snapshot delegates to inner planner."""
		config = _config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)
		planner.set_project_snapshot("src/ has 20 files")
		assert planner._inner._project_snapshot == "src/ has 20 files"


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


class TestBacklogStalenessEviction:
	async def test_time_based_eviction(self) -> None:
		"""Units older than backlog_max_age_seconds are evicted."""
		config = _config()
		config.continuous.backlog_max_age_seconds = 60
		config.continuous.backlog_min_size = 1  # won't trigger replan with 2 units
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		# Pre-populate backlog with 3 units
		planner._backlog = [
			WorkUnit(id="old1", title="Old Task 1", description="Old task one desc"),
			WorkUnit(id="old2", title="Old Task 2", description="Old task two desc"),
			WorkUnit(id="fresh", title="Fresh Task"),
		]
		# Mark old units as queued 120 seconds ago (exceeds 60s threshold)
		now = time.monotonic()
		planner._backlog_queued_at["old1"] = now - 120
		planner._backlog_queued_at["old2"] = now - 120
		planner._backlog_queued_at["fresh"] = now - 10  # recent

		planner._inner.plan_round = AsyncMock()  # should not be called

		mission = _mission()
		_, units, _ = await planner.get_next_units(mission, max_units=3)

		# Only the fresh unit should survive eviction
		assert len(units) == 1
		assert units[0].id == "fresh"
		planner._inner.plan_round.assert_not_called()

	async def test_divergence_based_eviction(self) -> None:
		"""Units with >50% file overlap with merged files are evicted."""
		config = _config()
		config.continuous.backlog_min_size = 1
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		# Unit touches 3 files; 2 are already merged (>50% overlap)
		planner._backlog = [
			WorkUnit(
				id="diverged",
				title="Diverged Task",
				description="Diverged task desc",
				files_hint="src/a.py, src/b.py, src/c.py",
			),
			WorkUnit(
				id="safe",
				title="Safe Task",
				description="Safe task desc",
				files_hint="src/x.py, src/y.py",
			),
		]
		planner._backlog_queued_at["diverged"] = time.monotonic()
		planner._backlog_queued_at["safe"] = time.monotonic()

		# Mark src/a.py and src/b.py as already merged (2/3 = 67% > 50%)
		planner.set_merged_files({"src/a.py", "src/b.py"})

		planner._inner.plan_round = AsyncMock()

		mission = _mission()
		_, units, _ = await planner.get_next_units(mission, max_units=3)

		# Only "safe" should survive (0/2 overlap)
		assert len(units) == 1
		assert units[0].id == "safe"

	async def test_stale_context_forwarded_to_replan(self) -> None:
		"""Evicted unit descriptions appear in stale_context for next replan."""
		config = _config()
		config.continuous.backlog_max_age_seconds = 60
		config.continuous.backlog_min_size = 2
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		# Pre-populate with one stale unit (will be evicted, leaving backlog < min_size)
		planner._backlog = [
			WorkUnit(id="stale1", title="Stale Unit", description="Refactor auth module"),
		]
		planner._backlog_queued_at["stale1"] = time.monotonic() - 120

		# Mock replan to return a fresh unit
		mock_wu = WorkUnit(id="new1", title="New Task")
		mock_plan = Plan(id="p1", objective="test")
		mock_root = PlanNode(id="root", plan_id="p1", strategy="leaves")
		mock_root._child_leaves = [  # type: ignore[attr-defined]
			(PlanNode(id="leaf1", node_type="leaf"), mock_wu),
		]
		planner._inner.plan_round = AsyncMock(return_value=(mock_plan, mock_root))

		mission = _mission()
		_, units, _ = await planner.get_next_units(mission, max_units=3)

		# Should have replanned and returned the new unit
		assert len(units) == 1
		assert units[0].id == "new1"
		planner._inner.plan_round.assert_called_once()

		# Verify stale context was included in the feedback_context
		call_kwargs = planner._inner.plan_round.call_args[1]
		feedback = call_kwargs.get("feedback_context", "")
		assert "Refactor auth module" in feedback
		assert "evicted as stale" in feedback

		# Stale context should be cleared after replan
		assert planner._stale_context == []

	async def test_no_eviction_when_under_threshold(self) -> None:
		"""Units within the max age are not evicted."""
		config = _config()
		config.continuous.backlog_max_age_seconds = 1800
		config.continuous.backlog_min_size = 1
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		planner._backlog = [
			WorkUnit(id="wu1", title="Task 1"),
			WorkUnit(id="wu2", title="Task 2"),
		]
		now = time.monotonic()
		planner._backlog_queued_at["wu1"] = now - 100  # 100s < 1800s
		planner._backlog_queued_at["wu2"] = now - 200  # 200s < 1800s

		planner._inner.plan_round = AsyncMock()

		mission = _mission()
		_, units, _ = await planner.get_next_units(mission, max_units=2)

		assert len(units) == 2
		planner._inner.plan_round.assert_not_called()

	async def test_units_without_queued_at_not_evicted_by_time(self) -> None:
		"""Units without a queued_at timestamp are not time-evicted."""
		config = _config()
		config.continuous.backlog_max_age_seconds = 60
		config.continuous.backlog_min_size = 1
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		# No queued_at entry for this unit
		planner._backlog = [
			WorkUnit(id="wu1", title="Task without timestamp"),
		]

		planner._inner.plan_round = AsyncMock()

		mission = _mission()
		_, units, _ = await planner.get_next_units(mission, max_units=1)

		assert len(units) == 1
		assert units[0].id == "wu1"
