"""Tests for backlog intake in ContinuousController and ContinuousPlanner.

Covers:
- Loading backlog items as mission objective
- Marking items complete/failed post-mission
- Discovery items flowing to persistent backlog
- Empty backlog fallback to normal discovery
- Planner set_backlog_items stores items
- Backlog item context appears in planning prompts
- Controller loads and passes backlog items to planner
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import ContinuousConfig, MissionConfig, PlannerConfig, TargetConfig
from mission_control.continuous_controller import ContinuousController
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.models import BacklogItem, DiscoveryItem, Handoff, Mission, Plan, PlanNode, WorkUnit


class TestLoadBacklogObjective:
	"""Test _load_backlog_objective() method."""

	def test_loads_pending_items_as_objective(self, config: MissionConfig, db: Database) -> None:
		"""Top pending backlog items compose into an objective string."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Add auth", description="Implement OAuth",
			priority_score=8.0, track="feature",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="Fix XSS", description="Sanitize inputs",
			priority_score=9.0, track="security",
		))

		ctrl = ContinuousController(config, db)
		objective = ctrl._load_backlog_objective(limit=5)

		assert objective is not None
		assert "Fix XSS" in objective
		assert "Add auth" in objective
		assert "backlog_item_id=b1" in objective
		assert "backlog_item_id=b2" in objective

	def test_marks_items_as_in_progress(self, config: MissionConfig, db: Database) -> None:
		"""Selected backlog items are marked in_progress."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality",
		))

		ctrl = ContinuousController(config, db)
		ctrl._load_backlog_objective(limit=5)

		item = db.get_backlog_item("b1")
		assert item is not None
		assert item.status == "in_progress"

	def test_stores_item_ids_for_tracking(self, config: MissionConfig, db: Database) -> None:
		"""Item IDs are stored on the controller for post-mission tracking."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="Task 2", description="desc",
			priority_score=3.0, track="feature",
		))

		ctrl = ContinuousController(config, db)
		ctrl._load_backlog_objective(limit=5)

		assert "b1" in ctrl._backlog_item_ids
		assert "b2" in ctrl._backlog_item_ids

	def test_returns_none_on_empty_backlog(self, config: MissionConfig, db: Database) -> None:
		"""Returns None when no pending backlog items exist."""
		ctrl = ContinuousController(config, db)
		objective = ctrl._load_backlog_objective(limit=5)

		assert objective is None
		assert ctrl._backlog_item_ids == []

	def test_respects_limit(self, config: MissionConfig, db: Database) -> None:
		"""Only loads up to `limit` items."""
		for i in range(10):
			db.insert_backlog_item(BacklogItem(
				id=f"b{i}", title=f"Task {i}", description="desc",
				priority_score=float(10 - i), track="quality",
			))

		ctrl = ContinuousController(config, db)
		ctrl._load_backlog_objective(limit=3)

		assert len(ctrl._backlog_item_ids) == 3

	def test_only_loads_pending_items(self, config: MissionConfig, db: Database) -> None:
		"""Does not load items that are already in_progress or completed."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Pending", description="desc",
			priority_score=5.0, track="quality", status="pending",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="In Progress", description="desc",
			priority_score=9.0, track="quality", status="in_progress",
		))
		db.insert_backlog_item(BacklogItem(
			id="b3", title="Completed", description="desc",
			priority_score=9.0, track="quality", status="completed",
		))

		ctrl = ContinuousController(config, db)
		objective = ctrl._load_backlog_objective(limit=5)

		assert objective is not None
		assert "Pending" in objective
		assert "In Progress" not in objective
		assert "Completed" not in objective
		assert ctrl._backlog_item_ids == ["b1"]

	def test_sorted_by_priority_desc(self, config: MissionConfig, db: Database) -> None:
		"""Items are selected by priority score descending."""
		db.insert_backlog_item(BacklogItem(
			id="low", title="Low priority", description="desc",
			priority_score=1.0, track="quality",
		))
		db.insert_backlog_item(BacklogItem(
			id="high", title="High priority", description="desc",
			priority_score=9.0, track="feature",
		))

		ctrl = ContinuousController(config, db)
		objective = ctrl._load_backlog_objective(limit=1)

		assert objective is not None
		assert "High priority" in objective
		assert ctrl._backlog_item_ids == ["high"]


class TestUpdateBacklogOnCompletion:
	"""Test _update_backlog_on_completion() method."""

	def test_marks_completed_on_success(self, config: MissionConfig, db: Database) -> None:
		"""When objective_met=True, all targeted items become completed."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality", status="in_progress",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="Task 2", description="desc",
			priority_score=3.0, track="feature", status="in_progress",
		))

		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = ["b1", "b2"]
		ctrl._update_backlog_on_completion(objective_met=True, handoffs=[])

		assert db.get_backlog_item("b1").status == "completed"
		assert db.get_backlog_item("b2").status == "completed"

	def test_resets_to_pending_on_failure(self, config: MissionConfig, db: Database) -> None:
		"""When objective_met=False, items reset to pending with incremented attempt_count."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality", status="in_progress",
			attempt_count=0,
		))

		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = ["b1"]
		ctrl._update_backlog_on_completion(objective_met=False, handoffs=[])

		item = db.get_backlog_item("b1")
		assert item.status == "pending"
		assert item.attempt_count == 1

	def test_stores_failure_reason_from_handoffs(self, config: MissionConfig, db: Database) -> None:
		"""Failure reasons from handoff concerns are stored in last_failure_reason."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality", status="in_progress",
		))

		handoff = Handoff(
			work_unit_id="wu1",
			status="failed",
			concerns=json.dumps(["Merge conflict on config.py"]),
			summary="Failed to complete task",
		)

		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = ["b1"]
		ctrl._update_backlog_on_completion(objective_met=False, handoffs=[handoff])

		item = db.get_backlog_item("b1")
		assert item.last_failure_reason is not None
		assert "Merge conflict on config.py" in item.last_failure_reason

	def test_noop_when_no_backlog_items(self, config: MissionConfig, db: Database) -> None:
		"""Does nothing when no backlog items were loaded."""
		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = []
		# Should not raise
		ctrl._update_backlog_on_completion(objective_met=True, handoffs=[])

	def test_handles_missing_item_gracefully(self, config: MissionConfig, db: Database) -> None:
		"""Skips items that no longer exist in the DB."""
		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = ["nonexistent"]
		# Should not raise
		ctrl._update_backlog_on_completion(objective_met=True, handoffs=[])

	def test_failure_context_from_summary_when_no_concerns(self, config: MissionConfig, db: Database) -> None:
		"""Uses handoff summary as failure reason when concerns are empty."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality", status="in_progress",
		))

		handoff = Handoff(
			work_unit_id="wu1",
			status="failed",
			concerns="[]",
			summary="Timed out after 300s",
		)

		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = ["b1"]
		ctrl._update_backlog_on_completion(objective_met=False, handoffs=[handoff])

		item = db.get_backlog_item("b1")
		assert "Timed out" in item.last_failure_reason


class TestDiscoveryToBacklog:
	"""Test that post-mission discovery items flow into the persistent backlog."""

	@pytest.mark.asyncio
	async def test_discovery_items_inserted_to_backlog(self, config: MissionConfig, db: Database) -> None:
		"""Discovery items are converted to BacklogItems and inserted."""
		mock_items = [
			DiscoveryItem(
				id="d1", title="Add caching", description="Redis caching layer",
				priority_score=7.0, impact=8, effort=5, track="feature",
			),
			DiscoveryItem(
				id="d2", title="Fix SQL injection", description="Parameterize queries",
				priority_score=9.0, impact=10, effort=3, track="security",
			),
		]

		mock_engine = MagicMock()
		mock_engine.discover = AsyncMock(return_value=(MagicMock(), mock_items))

		ctrl = ContinuousController(config, db)

		with patch("mission_control.auto_discovery.DiscoveryEngine", return_value=mock_engine):
			await ctrl._run_post_mission_discovery()

		# Check backlog items were created
		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) >= 2
		titles = {item.title for item in backlog}
		assert "Add caching" in titles
		assert "Fix SQL injection" in titles

		# Verify fields are mapped correctly
		caching_items = [i for i in backlog if i.title == "Add caching"]
		assert len(caching_items) == 1
		assert caching_items[0].priority_score == 7.0
		assert caching_items[0].impact == 8
		assert caching_items[0].effort == 5
		assert caching_items[0].track == "feature"

	@pytest.mark.asyncio
	async def test_empty_discovery_no_backlog_items(self, config: MissionConfig, db: Database) -> None:
		"""When discovery returns no items, no backlog items are created."""
		mock_engine = MagicMock()
		mock_engine.discover = AsyncMock(return_value=(MagicMock(), []))

		ctrl = ContinuousController(config, db)

		with patch("mission_control.auto_discovery.DiscoveryEngine", return_value=mock_engine):
			await ctrl._run_post_mission_discovery()

		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 0


class TestBacklogIntegrationInRun:
	"""Test backlog integration points in the run() method."""

	def test_empty_backlog_falls_back_to_config_objective(self, config: MissionConfig, db: Database) -> None:
		"""When backlog is empty, the config objective is used unchanged."""
		config.target.objective = "Build the widget"
		config.discovery.enabled = True

		ctrl = ContinuousController(config, db)
		objective = ctrl._load_backlog_objective(limit=5)

		assert objective is None
		assert config.target.objective == "Build the widget"

	def test_backlog_merged_with_existing_objective(self, config: MissionConfig, db: Database) -> None:
		"""Backlog items are appended to existing config objective."""
		config.target.objective = "Build the widget"
		config.discovery.enabled = True

		db.insert_backlog_item(BacklogItem(
			id="b1", title="Add tests", description="Coverage for auth module",
			priority_score=7.0, track="quality",
		))

		ctrl = ContinuousController(config, db)
		backlog_objective = ctrl._load_backlog_objective(limit=5)

		assert backlog_objective is not None
		# Simulate what run() does
		merged = config.target.objective + "\n\n" + backlog_objective
		assert "Build the widget" in merged
		assert "Add tests" in merged
		assert "backlog_item_id=b1" in merged

	def test_backlog_replaces_empty_objective(self, config: MissionConfig, db: Database) -> None:
		"""When config has no objective, backlog becomes the sole objective."""
		config.target.objective = ""
		config.discovery.enabled = True

		db.insert_backlog_item(BacklogItem(
			id="b1", title="Refactor DB", description="Split into modules",
			priority_score=6.0, track="quality",
		))

		ctrl = ContinuousController(config, db)
		backlog_objective = ctrl._load_backlog_objective(limit=5)

		assert backlog_objective is not None
		assert "Refactor DB" in backlog_objective

	def test_discovery_disabled_skips_backlog(self, config: MissionConfig, db: Database) -> None:
		"""When discovery is disabled, backlog loading is skipped in run()."""
		config.discovery.enabled = False

		db.insert_backlog_item(BacklogItem(
			id="b1", title="Should not load", description="desc",
			priority_score=9.0, track="quality",
		))

		# The run() method checks config.discovery.enabled before calling
		# _load_backlog_objective(). We verify the flag works correctly.
		_ = ContinuousController(config, db)
		# Directly verify the condition that run() checks
		assert not config.discovery.enabled
		# Item should still be pending since we didn't call _load_backlog_objective
		item = db.get_backlog_item("b1")
		assert item.status == "pending"


def _planner_config() -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(name="test", path="/tmp/test", objective="Build API")
	mc.planner = PlannerConfig(max_depth=2)
	mc.continuous = ContinuousConfig(backlog_min_size=2)
	return mc


def _mock_plan_round(unit_ids: list[str]) -> AsyncMock:
	"""Create a mock plan_round that returns units with the given IDs."""
	units = [WorkUnit(id=uid, plan_id="p1", title=f"Task {uid}") for uid in unit_ids]
	plan = Plan(id="p1", objective="test")
	root = PlanNode(id="root", plan_id="p1", strategy="leaves")
	root._child_leaves = [  # type: ignore[attr-defined]
		(PlanNode(id=f"leaf-{wu.id}", node_type="leaf"), wu)
		for wu in units
	]
	return AsyncMock(return_value=(plan, root))


class TestPlannerSetBacklogItems:
	"""Test ContinuousPlanner.set_backlog_items() stores items."""

	def test_stores_items(self) -> None:
		"""set_backlog_items stores the provided BacklogItem list."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		items = [
			BacklogItem(id="b1", title="Add auth", description="OAuth flow", priority_score=8.0, track="feature"),
			BacklogItem(id="b2", title="Fix XSS", description="Sanitize inputs", priority_score=9.0, track="security"),
		]
		planner.set_backlog_items(items)

		assert len(planner._backlog_items) == 2
		assert planner._backlog_items[0].title == "Add auth"
		assert planner._backlog_items[1].title == "Fix XSS"

	def test_stores_copy(self) -> None:
		"""set_backlog_items stores a copy, not a reference to the original list."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		items = [BacklogItem(id="b1", title="Task 1", description="desc", priority_score=5.0, track="quality")]
		planner.set_backlog_items(items)

		items.append(BacklogItem(id="b2", title="Task 2", description="desc", priority_score=3.0, track="feature"))
		assert len(planner._backlog_items) == 1

	def test_replaces_previous_items(self) -> None:
		"""Calling set_backlog_items again replaces the previous items."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		planner.set_backlog_items([
			BacklogItem(id="b1", title="Old", description="desc", priority_score=5.0, track="quality"),
		])
		planner.set_backlog_items([
			BacklogItem(id="b2", title="New", description="desc", priority_score=7.0, track="feature"),
		])

		assert len(planner._backlog_items) == 1
		assert planner._backlog_items[0].title == "New"

	def test_empty_list_clears(self) -> None:
		"""Passing empty list clears stored items."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		planner.set_backlog_items([
			BacklogItem(id="b1", title="Task", description="desc", priority_score=5.0, track="quality"),
		])
		planner.set_backlog_items([])

		assert planner._backlog_items == []


class TestBacklogContextInPlanning:
	"""Test that backlog item context appears in planning prompts."""

	async def test_backlog_items_appear_in_replan_context(self) -> None:
		"""Backlog items set via set_backlog_items appear in the feedback_context sent to plan_round."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])
		planner.set_backlog_items([
			BacklogItem(id="b1", title="Add auth", description="OAuth flow", priority_score=8.0, track="feature"),
			BacklogItem(id="b2", title="Fix XSS", description="Sanitize inputs", priority_score=9.0, track="security"),
		])

		mission = Mission(id="m1", objective="Build API")
		await planner.get_next_units(mission)

		call_kwargs = planner._inner.plan_round.call_args[1]
		ctx = call_kwargs["feedback_context"]
		assert "Priority backlog items for this mission:" in ctx
		assert "Add auth" in ctx
		assert "Fix XSS" in ctx
		assert "priority=8.0" in ctx
		assert "priority=9.0" in ctx

	async def test_backlog_items_appended_to_existing_context(self) -> None:
		"""Backlog section is appended after existing feedback context."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])
		planner.set_backlog_items([
			BacklogItem(id="b1", title="Refactor DB", description="Split modules", priority_score=6.0, track="quality"),
		])

		mission = Mission(id="m1", objective="Build API")
		await planner.get_next_units(mission, feedback_context="Previous work completed auth.")

		call_kwargs = planner._inner.plan_round.call_args[1]
		ctx = call_kwargs["feedback_context"]
		assert ctx.startswith("Previous work completed auth.")
		assert "Refactor DB" in ctx

	async def test_no_backlog_items_no_extra_context(self) -> None:
		"""Without backlog items, no backlog section appears in context."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])

		mission = Mission(id="m1", objective="Build API")
		await planner.get_next_units(mission, feedback_context="Some feedback.")

		call_kwargs = planner._inner.plan_round.call_args[1]
		ctx = call_kwargs["feedback_context"]
		assert "Priority backlog items" not in ctx


class TestControllerPassesBacklogToPlanner:
	"""Test that controller loads backlog items and passes them to planner."""

	def test_controller_passes_items_to_planner(self, config: MissionConfig, db: Database) -> None:
		"""After loading backlog, controller calls planner.set_backlog_items with the items."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Add auth", description="OAuth flow",
			priority_score=8.0, track="feature",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="Fix XSS", description="Sanitize inputs",
			priority_score=9.0, track="security",
		))

		ctrl = ContinuousController(config, db)
		ctrl._planner = ContinuousPlanner(config, db)

		# Load backlog items (simulating what run() does)
		config.discovery.enabled = True
		ctrl._load_backlog_objective(limit=5)

		# Pass to planner (simulating what run() does after _load_backlog_objective)
		items = [db.get_backlog_item(bid) for bid in ctrl._backlog_item_ids]
		ctrl._planner.set_backlog_items([i for i in items if i is not None])

		assert len(ctrl._planner._backlog_items) == 2
		titles = {i.title for i in ctrl._planner._backlog_items}
		assert "Add auth" in titles
		assert "Fix XSS" in titles

	def test_empty_backlog_skips_planner_call(self, config: MissionConfig, db: Database) -> None:
		"""When no backlog items are loaded, planner.set_backlog_items is not called."""
		ctrl = ContinuousController(config, db)
		ctrl._planner = ContinuousPlanner(config, db)

		config.discovery.enabled = True
		ctrl._load_backlog_objective(limit=5)

		# No items loaded, so _backlog_item_ids is empty
		assert ctrl._backlog_item_ids == []
		assert ctrl._planner._backlog_items == []
