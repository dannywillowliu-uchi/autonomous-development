"""Tests for round_controller module."""

from __future__ import annotations

import pytest

from mission_control.config import MissionConfig, RoundsConfig
from mission_control.db import Database
from mission_control.models import Mission, Plan, PlanNode, Round, WorkUnit
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

	def test_max_rounds_allows_current(self, controller: RoundController, mission: Mission) -> None:
		"""When total_rounds == max_rounds, continue (current round should run)."""
		mission.total_rounds = 10  # config.rounds.max_rounds == 10
		result = controller._should_stop(mission, [0.5])
		assert result == ""

	def test_max_rounds_exceeded(self, controller: RoundController, mission: Mission) -> None:
		mission.total_rounds = 11  # > max_rounds (10), should stop
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

	def test_custom_stall_epsilon(self, db: Database, mission: Mission) -> None:
		"""Custom stall_score_epsilon changes stall sensitivity."""
		cfg = MissionConfig()
		cfg.rounds = RoundsConfig(max_rounds=10, stall_threshold=3, stall_score_epsilon=0.05)
		ctrl = RoundController(cfg, db)

		# Spread of 0.02 would not trigger default (0.01), but is < custom 0.05
		mission.total_rounds = 0
		scores = [0.5, 0.5, 0.52]
		result = ctrl._should_stop(mission, scores)
		assert result == "stalled"


# -- stop() --


class TestStop:
	def test_stop_sets_running_false(self, controller: RoundController) -> None:
		assert controller.running is True
		controller.stop()
		assert controller.running is False


# -- _persist_plan_tree --


class TestPersistPlanTree:
	def test_flat_tree_with_child_leaves(self, controller: RoundController, db: Database) -> None:
		"""Root with _child_leaves persists root, leaves, and work units."""
		plan = Plan(objective="Test")
		db.insert_plan(plan)

		root = PlanNode(
			plan_id=plan.id, depth=0, scope="Root",
			node_type="branch", strategy="leaves", status="expanded",
		)
		leaf1 = PlanNode(
			plan_id=plan.id, parent_id=root.id, depth=1,
			scope="Task A", node_type="leaf", status="expanded",
		)
		leaf2 = PlanNode(
			plan_id=plan.id, parent_id=root.id, depth=1,
			scope="Task B", node_type="leaf", status="expanded",
		)
		wu1 = WorkUnit(plan_id=plan.id, title="Task A", plan_node_id=leaf1.id)
		wu2 = WorkUnit(plan_id=plan.id, title="Task B", plan_node_id=leaf2.id)
		leaf1.work_unit_id = wu1.id
		leaf2.work_unit_id = wu2.id
		root._child_leaves = [(leaf1, wu1), (leaf2, wu2)]  # type: ignore[attr-defined]

		controller._persist_plan_tree(root, plan)

		# Verify all nodes persisted
		nodes = db.get_plan_nodes_for_plan(plan.id)
		assert len(nodes) == 3  # root + 2 leaves

		# Verify work units persisted
		units = db.get_work_units_for_plan(plan.id)
		assert len(units) == 2

	def test_subdivided_tree_persists_all_nodes(self, controller: RoundController, db: Database) -> None:
		"""Subdivided tree with _subdivided_children persists the full tree."""
		plan = Plan(objective="Test subdivide")
		db.insert_plan(plan)

		root = PlanNode(
			plan_id=plan.id, depth=0, scope="Root",
			node_type="branch", strategy="subdivide", status="expanded",
		)
		child_a = PlanNode(
			plan_id=plan.id, parent_id=root.id, depth=1, scope="Backend",
			node_type="branch", strategy="leaves", status="expanded",
		)
		child_b = PlanNode(
			plan_id=plan.id, parent_id=root.id, depth=1, scope="Frontend",
			node_type="branch", strategy="leaves", status="expanded",
		)

		leaf_a = PlanNode(
			plan_id=plan.id, parent_id=child_a.id, depth=2,
			scope="API", node_type="leaf", status="expanded",
		)
		leaf_b = PlanNode(
			plan_id=plan.id, parent_id=child_b.id, depth=2,
			scope="UI", node_type="leaf", status="expanded",
		)
		wu_a = WorkUnit(plan_id=plan.id, title="API endpoint", plan_node_id=leaf_a.id)
		wu_b = WorkUnit(plan_id=plan.id, title="UI component", plan_node_id=leaf_b.id)
		leaf_a.work_unit_id = wu_a.id
		leaf_b.work_unit_id = wu_b.id

		child_a._child_leaves = [(leaf_a, wu_a)]  # type: ignore[attr-defined]
		child_b._child_leaves = [(leaf_b, wu_b)]  # type: ignore[attr-defined]
		root._subdivided_children = [child_a, child_b]  # type: ignore[attr-defined]
		root.children_ids = f"{child_a.id},{child_b.id}"

		controller._persist_plan_tree(root, plan)

		# Verify all 5 nodes persisted: root + 2 branch children + 2 leaves
		nodes = db.get_plan_nodes_for_plan(plan.id)
		assert len(nodes) == 5

		# Verify 2 work units persisted
		units = db.get_work_units_for_plan(plan.id)
		assert len(units) == 2
		titles = {u.title for u in units}
		assert titles == {"API endpoint", "UI component"}


class TestExecuteUnitsExceptionLogging:
	async def test_gather_logs_unhandled_exceptions(
		self, controller: RoundController, db: Database, mission: Mission, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Unhandled exceptions from asyncio.gather are logged, not silently swallowed."""
		import logging
		from unittest.mock import AsyncMock

		db.insert_mission(mission)
		plan = Plan(objective="test")
		db.insert_plan(plan)
		rnd = Round(mission_id=mission.id, number=1)
		db.insert_round(rnd)

		unit = WorkUnit(plan_id=plan.id, title="Failing unit")
		db.insert_work_unit(unit)

		# _green_branch is None, so _execute_single_unit will hit an
		# AssertionError -- this exercises the gather exception logging
		controller._backend = AsyncMock()

		with caplog.at_level(logging.ERROR, logger="mission_control.round_controller"):
			await controller._execute_units(plan, rnd)

		# Verify the exception was logged (not silently swallowed)
		assert any("Unhandled exception" in msg for msg in caplog.messages)

	def test_forced_leaf_persists(self, controller: RoundController, db: Database) -> None:
		"""Node with _forced_unit (max depth forced leaf) persists correctly."""
		plan = Plan(objective="Test forced")
		db.insert_plan(plan)

		node = PlanNode(
			plan_id=plan.id, depth=3, scope="Forced leaf",
			node_type="leaf", strategy="leaves", status="expanded",
		)
		wu = WorkUnit(plan_id=plan.id, title="Forced task", plan_node_id=node.id)
		node.work_unit_id = wu.id
		node._forced_unit = wu  # type: ignore[attr-defined]

		controller._persist_plan_tree(node, plan)

		nodes = db.get_plan_nodes_for_plan(plan.id)
		assert len(nodes) == 1

		units = db.get_work_units_for_plan(plan.id)
		assert len(units) == 1
		assert units[0].title == "Forced task"

	def test_deeply_nested_subdivide_persists(self, controller: RoundController, db: Database) -> None:
		"""Three-level tree persists all nodes via recursion."""
		plan = Plan(objective="Deep tree")
		db.insert_plan(plan)

		root = PlanNode(
			plan_id=plan.id, depth=0, scope="Root",
			node_type="branch", strategy="subdivide", status="expanded",
		)
		mid = PlanNode(
			plan_id=plan.id, parent_id=root.id, depth=1, scope="Mid",
			node_type="branch", strategy="subdivide", status="expanded",
		)
		bottom = PlanNode(
			plan_id=plan.id, parent_id=mid.id, depth=2, scope="Bottom",
			node_type="branch", strategy="leaves", status="expanded",
		)

		leaf = PlanNode(
			plan_id=plan.id, parent_id=bottom.id, depth=3,
			scope="Deep task", node_type="leaf", status="expanded",
		)
		wu = WorkUnit(plan_id=plan.id, title="Deep work", plan_node_id=leaf.id)
		leaf.work_unit_id = wu.id

		bottom._child_leaves = [(leaf, wu)]  # type: ignore[attr-defined]
		mid._subdivided_children = [bottom]  # type: ignore[attr-defined]
		mid.children_ids = bottom.id
		root._subdivided_children = [mid]  # type: ignore[attr-defined]
		root.children_ids = mid.id

		controller._persist_plan_tree(root, plan)

		nodes = db.get_plan_nodes_for_plan(plan.id)
		assert len(nodes) == 4  # root + mid + bottom + leaf

		units = db.get_work_units_for_plan(plan.id)
		assert len(units) == 1
		assert units[0].title == "Deep work"

		# Verify round-trip: retrieve nodes and check hierarchy
		retrieved_root = db.get_plan_node(root.id)
		assert retrieved_root is not None
		assert retrieved_root.children_ids == mid.id

		retrieved_leaf = db.get_plan_node(leaf.id)
		assert retrieved_leaf is not None
		assert retrieved_leaf.node_type == "leaf"
		assert retrieved_leaf.work_unit_id == wu.id


class TestShouldStopOffByOne:
	"""Verify max_rounds boundary is exact (no off-by-one).

	The main loop sets total_rounds = round_number BEFORE calling _should_stop.
	With max_rounds=5, rounds 1-5 should execute. _should_stop should return
	"max_rounds" only when total_rounds > max_rounds (i.e., round 6).
	"""

	def test_stops_after_max_rounds(self, db: Database) -> None:
		"""When total_rounds exceeds max_rounds, should stop."""
		cfg = MissionConfig()
		cfg.rounds = RoundsConfig(max_rounds=5)
		ctrl = RoundController(cfg, db)
		mission = Mission(objective="test")
		mission.total_rounds = 6
		assert ctrl._should_stop(mission, []) == "max_rounds"

	def test_does_not_stop_at_max_rounds(self, db: Database) -> None:
		"""When total_rounds equals max_rounds, should NOT stop (current round runs)."""
		cfg = MissionConfig()
		cfg.rounds = RoundsConfig(max_rounds=5)
		ctrl = RoundController(cfg, db)
		mission = Mission(objective="test")
		mission.total_rounds = 5
		assert ctrl._should_stop(mission, []) == ""

	def test_does_not_stop_before_max_rounds(self, db: Database) -> None:
		"""When total_rounds is less than max_rounds, should not stop."""
		cfg = MissionConfig()
		cfg.rounds = RoundsConfig(max_rounds=5)
		ctrl = RoundController(cfg, db)
		mission = Mission(objective="test")
		mission.total_rounds = 4
		assert ctrl._should_stop(mission, []) == ""


class TestBlockedUnitStatus:
	"""Blocked units should not have attempt counter incremented."""

	async def test_blocked_unit_attempt_not_incremented(
		self, controller: RoundController, db: Database, mission: Mission,
	) -> None:
		"""Unit with MC_RESULT status 'blocked' should keep attempt unchanged."""
		import asyncio
		from unittest.mock import AsyncMock

		db.insert_mission(mission)
		plan = Plan(objective="test")
		db.insert_plan(plan)
		rnd = Round(mission_id=mission.id, number=1)
		db.insert_round(rnd)

		unit = WorkUnit(plan_id=plan.id, title="Blocked unit", attempt=0, max_attempts=3)
		db.insert_work_unit(unit)

		# Mock backend
		mock_backend = AsyncMock()
		mock_backend.provision_workspace = AsyncMock(return_value="/tmp/ws")
		mock_backend.spawn = AsyncMock()
		mock_backend.check_status = AsyncMock(return_value="completed")
		mock_backend.get_output = AsyncMock(
			return_value='MC_RESULT:{"status":"blocked","summary":"Waiting for dependency"}'
		)
		mock_backend.release_workspace = AsyncMock()
		controller._backend = mock_backend

		# Mock green branch
		mock_gb = AsyncMock()
		controller._green_branch = mock_gb

		sem = asyncio.Semaphore(1)
		await controller._execute_single_unit(unit, rnd, sem)

		refreshed = db.get_work_unit(unit.id)
		assert refreshed is not None
		assert refreshed.status == "blocked"
		assert refreshed.attempt == 0  # NOT incremented

	async def test_failed_unit_attempt_incremented(
		self, controller: RoundController, db: Database, mission: Mission,
	) -> None:
		"""Unit with MC_RESULT status 'failed' should have attempt incremented."""
		import asyncio
		from unittest.mock import AsyncMock

		db.insert_mission(mission)
		plan = Plan(objective="test")
		db.insert_plan(plan)
		rnd = Round(mission_id=mission.id, number=1)
		db.insert_round(rnd)

		unit = WorkUnit(plan_id=plan.id, title="Failing unit", attempt=0, max_attempts=3)
		db.insert_work_unit(unit)

		mock_backend = AsyncMock()
		mock_backend.provision_workspace = AsyncMock(return_value="/tmp/ws")
		mock_backend.spawn = AsyncMock()
		mock_backend.check_status = AsyncMock(return_value="completed")
		mock_backend.get_output = AsyncMock(
			return_value='MC_RESULT:{"status":"failed","summary":"Could not fix"}'
		)
		mock_backend.release_workspace = AsyncMock()
		controller._backend = mock_backend

		mock_gb = AsyncMock()
		controller._green_branch = mock_gb

		sem = asyncio.Semaphore(1)
		await controller._execute_single_unit(unit, rnd, sem)

		refreshed = db.get_work_unit(unit.id)
		assert refreshed is not None
		assert refreshed.status == "failed"
		assert refreshed.attempt == 1  # Incremented


class TestMergeToWorkingFailureMarksUnitFailed:
	"""When merge_to_working fails, the unit should be marked failed, not completed."""

	async def test_merge_failure_marks_unit_failed(
		self, controller: RoundController, db: Database, mission: Mission,
	) -> None:
		"""Unit completes with commits but merge_to_working returns False -- unit should be failed."""
		import asyncio
		from unittest.mock import AsyncMock

		db.insert_mission(mission)
		plan = Plan(objective="test")
		db.insert_plan(plan)
		rnd = Round(mission_id=mission.id, number=1)
		db.insert_round(rnd)

		unit = WorkUnit(plan_id=plan.id, title="Merge conflict unit", attempt=0, max_attempts=3)
		db.insert_work_unit(unit)

		mock_backend = AsyncMock()
		mock_backend.provision_workspace = AsyncMock(return_value="/tmp/ws")
		mock_backend.spawn = AsyncMock()
		mock_backend.check_status = AsyncMock(return_value="completed")
		mock_backend.get_output = AsyncMock(
			return_value='MC_RESULT:{"status":"completed","summary":"Added feature","commits":["abc123"]}'
		)
		mock_backend.release_workspace = AsyncMock()
		controller._backend = mock_backend

		mock_gb = AsyncMock()
		mock_gb.merge_to_working = AsyncMock(return_value=False)
		controller._green_branch = mock_gb

		sem = asyncio.Semaphore(1)
		await controller._execute_single_unit(unit, rnd, sem)

		refreshed = db.get_work_unit(unit.id)
		assert refreshed is not None
		assert refreshed.status == "failed"
		assert refreshed.attempt == 1  # Incremented on merge failure
		assert "merge" in refreshed.output_summary.lower()

	async def test_merge_success_marks_unit_completed(
		self, controller: RoundController, db: Database, mission: Mission,
	) -> None:
		"""Unit completes with commits and merge_to_working succeeds -- unit should be completed."""
		import asyncio
		from unittest.mock import AsyncMock

		db.insert_mission(mission)
		plan = Plan(objective="test")
		db.insert_plan(plan)
		rnd = Round(mission_id=mission.id, number=1)
		db.insert_round(rnd)

		unit = WorkUnit(plan_id=plan.id, title="Successful unit", attempt=0, max_attempts=3)
		db.insert_work_unit(unit)

		mock_backend = AsyncMock()
		mock_backend.provision_workspace = AsyncMock(return_value="/tmp/ws")
		mock_backend.spawn = AsyncMock()
		mock_backend.check_status = AsyncMock(return_value="completed")
		mock_backend.get_output = AsyncMock(
			return_value='MC_RESULT:{"status":"completed","summary":"Added feature","commits":["abc123"]}'
		)
		mock_backend.release_workspace = AsyncMock()
		controller._backend = mock_backend

		mock_gb = AsyncMock()
		mock_gb.merge_to_working = AsyncMock(return_value=True)
		controller._green_branch = mock_gb

		sem = asyncio.Semaphore(1)
		await controller._execute_single_unit(unit, rnd, sem)

		refreshed = db.get_work_unit(unit.id)
		assert refreshed is not None
		assert refreshed.status == "completed"
		assert refreshed.attempt == 0  # NOT incremented on success


class TestSSHBackendMissionModeRejection:
	"""Phase 61: SSH backend raises NotImplementedError in mission mode."""

	async def test_ssh_backend_raises_not_implemented(self, db: Database) -> None:
		"""Mission mode with SSH backend should raise NotImplementedError at init."""
		from mission_control.config import BackendConfig, SSHHostConfig

		cfg = MissionConfig()
		cfg.backend = BackendConfig(type="ssh", ssh_hosts=[SSHHostConfig(hostname="host1")])

		controller = RoundController(cfg, db)

		with pytest.raises(NotImplementedError, match="SSH backend is not yet supported"):
			await controller._init_components()  # noqa: SLF001


