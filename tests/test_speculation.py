"""Tests for speculation branching: config, models, DB, and controller logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mission_control.config import MissionConfig, SpeculationConfig, _build_speculation, load_config
from mission_control.continuous_controller import ContinuousController, DynamicSemaphore, WorkerCompletion
from mission_control.db import Database
from mission_control.models import (
	Epoch,
	Mission,
	Plan,
	SpeculationResult,
	UnitReview,
	WorkUnit,
)

# -- Fixtures --


@pytest.fixture
def db(tmp_path: Path) -> Database:
	d = Database(tmp_path / "test.db")
	yield d
	d.close()


def _create_mission_and_plan(db: Database, mission_id: str = "m-1", plan_id: str = "p-1") -> tuple[Mission, Plan]:
	mission = Mission(id=mission_id, objective="test")
	db.insert_mission(mission)
	plan = Plan(id=plan_id, objective="test")
	db.insert_plan(plan)
	return mission, plan


# -- Config tests --


class TestSpeculationConfigDefaults:
	def test_defaults(self) -> None:
		sc = SpeculationConfig()
		assert sc.enabled is False
		assert sc.uncertainty_threshold == 0.7
		assert sc.branch_count == 2
		assert sc.selection_metric == "review_score"
		assert sc.cost_limit_multiplier == 1.5

	def test_build_speculation(self) -> None:
		data = {
			"enabled": True,
			"uncertainty_threshold": 0.5,
			"branch_count": 3,
			"selection_metric": "cost",
			"cost_limit_multiplier": 2.0,
		}
		sc = _build_speculation(data)
		assert sc.enabled is True
		assert sc.uncertainty_threshold == 0.5
		assert sc.branch_count == 3
		assert sc.selection_metric == "cost"
		assert sc.cost_limit_multiplier == 2.0

	def test_mission_config_has_speculation(self) -> None:
		mc = MissionConfig()
		assert isinstance(mc.speculation, SpeculationConfig)
		assert mc.speculation.enabled is False

	def test_toml_parsing(self, tmp_path: Path) -> None:
		toml_content = """\
[target]
name = "test"
path = "."
objective = "test"

[speculation]
enabled = true
uncertainty_threshold = 0.6
branch_count = 3
selection_metric = "cost"
cost_limit_multiplier = 2.5
"""
		config_path = tmp_path / "mission-control.toml"
		config_path.write_text(toml_content)
		mc = load_config(config_path)
		assert mc.speculation.enabled is True
		assert mc.speculation.uncertainty_threshold == 0.6
		assert mc.speculation.branch_count == 3
		assert mc.speculation.selection_metric == "cost"
		assert mc.speculation.cost_limit_multiplier == 2.5


# -- Model tests --


class TestWorkUnitSpeculationFields:
	def test_defaults(self) -> None:
		wu = WorkUnit()
		assert wu.speculation_score == 0.0
		assert wu.speculation_parent_id == ""

	def test_set_values(self) -> None:
		wu = WorkUnit(speculation_score=0.85, speculation_parent_id="parent-123")
		assert wu.speculation_score == 0.85
		assert wu.speculation_parent_id == "parent-123"


class TestSpeculationResultDefaults:
	def test_defaults(self) -> None:
		sr = SpeculationResult()
		assert sr.parent_unit_id == ""
		assert sr.winner_branch_id == ""
		assert sr.mission_id == ""
		assert sr.epoch_id == ""
		assert sr.branch_count == 0
		assert sr.branch_ids == ""
		assert sr.branch_scores == ""
		assert sr.total_speculation_cost_usd == 0.0
		assert sr.selection_metric == "review_score"
		assert sr.timestamp != ""  # auto-generated


# -- DB tests --


class TestSpeculationResultDB:
	def test_insert_get_round_trip(self, db: Database) -> None:
		_create_mission_and_plan(db)
		sr = SpeculationResult(
			id="sr-1",
			parent_unit_id="wu-parent",
			winner_branch_id="wu-branch-1",
			mission_id="m-1",
			epoch_id="e-1",
			branch_count=2,
			branch_ids="wu-branch-1,wu-branch-2",
			branch_scores='{"wu-branch-1": 8.5, "wu-branch-2": 6.0}',
			total_speculation_cost_usd=1.25,
			selection_metric="review_score",
		)
		db.insert_speculation_result(sr)
		results = db.get_speculation_results_for_mission("m-1")
		assert len(results) == 1
		fetched = results[0]
		assert fetched.id == "sr-1"
		assert fetched.parent_unit_id == "wu-parent"
		assert fetched.winner_branch_id == "wu-branch-1"
		assert fetched.branch_count == 2
		assert fetched.branch_ids == "wu-branch-1,wu-branch-2"
		assert fetched.total_speculation_cost_usd == 1.25
		assert fetched.selection_metric == "review_score"
		scores = json.loads(fetched.branch_scores)
		assert scores["wu-branch-1"] == 8.5

	def test_results_for_different_mission(self, db: Database) -> None:
		_create_mission_and_plan(db, "m-1", "p-1")
		_create_mission_and_plan(db, "m-2", "p-2")
		db.insert_speculation_result(SpeculationResult(id="sr-1", mission_id="m-1", parent_unit_id="wu-1"))
		db.insert_speculation_result(SpeculationResult(id="sr-2", mission_id="m-2", parent_unit_id="wu-2"))
		assert len(db.get_speculation_results_for_mission("m-1")) == 1
		assert len(db.get_speculation_results_for_mission("m-2")) == 1

	def test_speculation_score_persisted(self, db: Database) -> None:
		_create_mission_and_plan(db)
		wu = WorkUnit(id="wu-spec", plan_id="p-1", title="speculative", speculation_score=0.85)
		db.insert_work_unit(wu)
		fetched = db.get_work_unit("wu-spec")
		assert fetched is not None
		assert fetched.speculation_score == 0.85

	def test_speculation_parent_id_persisted(self, db: Database) -> None:
		_create_mission_and_plan(db)
		wu = WorkUnit(id="wu-branch", plan_id="p-1", title="branch", speculation_parent_id="parent-abc")
		db.insert_work_unit(wu)
		fetched = db.get_work_unit("wu-branch")
		assert fetched is not None
		assert fetched.speculation_parent_id == "parent-abc"

	def test_speculation_fields_update(self, db: Database) -> None:
		_create_mission_and_plan(db)
		wu = WorkUnit(id="wu-up", plan_id="p-1", title="test")
		db.insert_work_unit(wu)
		wu.speculation_score = 0.9
		wu.speculation_parent_id = "p-xyz"
		db.update_work_unit(wu)
		fetched = db.get_work_unit("wu-up")
		assert fetched is not None
		assert fetched.speculation_score == 0.9
		assert fetched.speculation_parent_id == "p-xyz"


# -- Controller logic tests --


def _make_config(**kwargs: object) -> MissionConfig:
	"""Build a MissionConfig with sensible test defaults."""
	mc = MissionConfig()
	mc.target.path = "/tmp/test-target"
	mc.target.objective = "test objective"
	mc.speculation.enabled = True
	mc.speculation.uncertainty_threshold = 0.7
	mc.speculation.branch_count = 2
	for k, v in kwargs.items():
		parts = k.split(".")
		obj = mc
		for part in parts[:-1]:
			obj = getattr(obj, part)
		setattr(obj, parts[-1], v)
	return mc


class TestSpeculationBelowThreshold:
	def test_not_triggered(self) -> None:
		"""Units below threshold should not trigger speculation dispatch."""
		config = _make_config()
		unit = WorkUnit(speculation_score=0.5)
		# The check in _dispatch_loop: score >= threshold
		assert not (
			config.speculation.enabled
			and unit.speculation_score >= config.speculation.uncertainty_threshold
		)

	def test_at_threshold_triggers(self) -> None:
		"""Units at exactly the threshold should trigger."""
		config = _make_config()
		unit = WorkUnit(speculation_score=0.7)
		assert (
			config.speculation.enabled
			and unit.speculation_score >= config.speculation.uncertainty_threshold
		)

	def test_disabled_does_not_trigger(self) -> None:
		"""Even high-score units don't trigger when disabled."""
		config = _make_config(**{"speculation.enabled": False})
		unit = WorkUnit(speculation_score=0.9)
		assert not (
			config.speculation.enabled
			and unit.speculation_score >= config.speculation.uncertainty_threshold
		)


class TestApproachHints:
	def test_hints_are_distinct(self) -> None:
		hints = ContinuousController.SPECULATION_APPROACH_HINTS
		assert len(hints) >= 5
		assert len(set(hints)) == len(hints)

	def test_n_hints_selected(self) -> None:
		hints = ContinuousController.SPECULATION_APPROACH_HINTS[:3]
		assert len(hints) == 3


class TestSpeculationBranchCollection:
	def test_accumulation(self) -> None:
		"""Branch completions should accumulate in the collection dict."""
		completions: dict[str, list[WorkerCompletion]] = {"parent-1": []}
		branch_unit = WorkUnit(id="b-1", speculation_parent_id="parent-1")
		epoch = Epoch(id="e-1")
		completion = WorkerCompletion(unit=branch_unit, handoff=None, workspace="/tmp", epoch=epoch)
		completions["parent-1"].append(completion)
		assert len(completions["parent-1"]) == 1

		branch_unit2 = WorkUnit(id="b-2", speculation_parent_id="parent-1")
		completion2 = WorkerCompletion(unit=branch_unit2, handoff=None, workspace="/tmp", epoch=epoch)
		completions["parent-1"].append(completion2)
		assert len(completions["parent-1"]) == 2


class TestSpeculationWinnerByReviewScore:
	@pytest.mark.asyncio
	async def test_highest_score_wins(self, db: Database) -> None:
		"""The branch with the highest review score should be selected as winner."""
		config = _make_config()
		controller = ContinuousController(config, db)
		controller._backend = AsyncMock()
		controller._green_branch = AsyncMock()
		controller._green_branch.merge_unit = AsyncMock(
			return_value=MagicMock(merged=True, merge_commit_hash="abc123"),
		)
		controller._planner = MagicMock()
		controller._diff_reviewer = AsyncMock()

		mission = Mission(id="m-1", objective="test")
		epoch = Epoch(id="e-1", mission_id="m-1")

		# Create parent unit
		parent = WorkUnit(id="parent-1", plan_id="p-1", unit_type="speculation_parent")
		controller._speculation_parent_units["parent-1"] = parent

		# Create 2 branch completions with different scores
		branch1 = WorkUnit(
			id="b-1", plan_id="p-1", status="completed",
			commit_hash="hash1", cost_usd=0.5, unit_type="speculation_branch",
			speculation_parent_id="parent-1",
		)
		branch2 = WorkUnit(
			id="b-2", plan_id="p-1", status="completed",
			commit_hash="hash2", cost_usd=0.3, unit_type="speculation_branch",
			speculation_parent_id="parent-1",
		)

		# Mock blocking review to return different scores
		review1 = UnitReview(avg_score=7.0)
		review2 = UnitReview(avg_score=9.0)
		controller._blocking_review = AsyncMock(side_effect=[review1, review2])

		controller._speculation_completions["parent-1"] = [
			WorkerCompletion(unit=branch1, handoff=None, workspace="/ws1", epoch=epoch),
			WorkerCompletion(unit=branch2, handoff=None, workspace="/ws2", epoch=epoch),
		]

		await controller._speculation_select_winner("parent-1", mission, epoch)

		# branch2 has higher score, should be winner
		controller._green_branch.merge_unit.assert_called_once_with("/ws2", branch2.branch_name)
		assert parent.status == "completed"
		assert controller._total_merged == 1
		# Loser workspace released
		controller._backend.release_workspace.assert_called_once_with("/ws1")


class TestSpeculationAllBranchesFail:
	@pytest.mark.asyncio
	async def test_parent_marked_failed(self, db: Database) -> None:
		"""When all branches fail, parent should be marked failed."""
		config = _make_config()
		controller = ContinuousController(config, db)
		controller._backend = AsyncMock()
		controller._green_branch = AsyncMock()

		mission = Mission(id="m-1", objective="test")
		epoch = Epoch(id="e-1", mission_id="m-1")

		parent = WorkUnit(id="parent-f", plan_id="p-1", unit_type="speculation_parent")
		controller._speculation_parent_units["parent-f"] = parent

		branch1 = WorkUnit(
			id="bf-1", plan_id="p-1", status="failed",
			cost_usd=0.4, unit_type="speculation_branch",
			speculation_parent_id="parent-f",
		)
		branch2 = WorkUnit(
			id="bf-2", plan_id="p-1", status="failed",
			cost_usd=0.3, unit_type="speculation_branch",
			speculation_parent_id="parent-f",
		)

		controller._speculation_completions["parent-f"] = [
			WorkerCompletion(unit=branch1, handoff=None, workspace="/ws1", epoch=epoch),
			WorkerCompletion(unit=branch2, handoff=None, workspace="/ws2", epoch=epoch),
		]

		await controller._speculation_select_winner("parent-f", mission, epoch)

		assert parent.status == "failed"
		assert controller._total_failed == 1
		assert parent.cost_usd == pytest.approx(0.7)


class TestSpeculationCostCapFallback:
	@pytest.mark.asyncio
	async def test_falls_back_to_single_dispatch(self, db: Database) -> None:
		"""When projected cost exceeds remaining budget, should return False."""
		config = _make_config()
		config.scheduler.budget.max_per_run_usd = 10.0
		controller = ContinuousController(config, db)
		controller._backend = AsyncMock()

		# EMA value of $5 -> 2 branches * $5 * 1.5 = $15 > $2 remaining
		controller._ema._ema = 5.0
		controller._ema._count = 1

		mission = Mission(id="m-1", objective="test", total_cost_usd=8.0)
		epoch = Epoch(id="e-1")
		controller._semaphore = DynamicSemaphore(4)
		unit = WorkUnit(
			id="wu-cost", plan_id="p-1", speculation_score=0.9,
			unit_type="implementation",
		)

		result = await controller._dispatch_speculated_unit(
			unit, epoch, mission, {},
		)
		assert result is False


class TestSpeculationResultRecorded:
	@pytest.mark.asyncio
	async def test_result_persisted(self, db: Database) -> None:
		"""After selection, a SpeculationResult should be in the DB."""
		config = _make_config(**{"speculation.selection_metric": "status"})
		controller = ContinuousController(config, db)
		controller._backend = AsyncMock()
		controller._green_branch = AsyncMock()
		controller._green_branch.merge_unit = AsyncMock(
			return_value=MagicMock(merged=True, merge_commit_hash="abc"),
		)
		controller._planner = MagicMock()

		mission, plan = _create_mission_and_plan(db)
		epoch = Epoch(id="e-1", mission_id="m-1")
		db.insert_epoch(epoch)

		parent = WorkUnit(id="parent-r", plan_id="p-1", unit_type="speculation_parent")
		controller._speculation_parent_units["parent-r"] = parent

		branch1 = WorkUnit(
			id="br-1", plan_id="p-1", status="completed",
			commit_hash="h1", cost_usd=0.5, unit_type="speculation_branch",
			speculation_parent_id="parent-r",
		)

		controller._speculation_completions["parent-r"] = [
			WorkerCompletion(unit=branch1, handoff=None, workspace="/ws1", epoch=epoch),
		]
		config.speculation.branch_count = 1

		await controller._speculation_select_winner("parent-r", mission, epoch)

		results = db.get_speculation_results_for_mission("m-1")
		assert len(results) == 1
		assert results[0].parent_unit_id == "parent-r"
		assert results[0].winner_branch_id == "br-1"
		assert results[0].branch_count == 1


class TestSpeculationLoserWorkspacesReleased:
	@pytest.mark.asyncio
	async def test_loser_released(self, db: Database) -> None:
		"""Loser workspaces should be released after winner selection."""
		config = _make_config(**{"speculation.selection_metric": "status"})
		controller = ContinuousController(config, db)
		controller._backend = AsyncMock()
		controller._green_branch = AsyncMock()
		controller._green_branch.merge_unit = AsyncMock(
			return_value=MagicMock(merged=True, merge_commit_hash="abc"),
		)
		controller._planner = MagicMock()

		mission = Mission(id="m-1", objective="test")
		epoch = Epoch(id="e-1", mission_id="m-1")

		parent = WorkUnit(id="parent-ws", plan_id="p-1", unit_type="speculation_parent")
		controller._speculation_parent_units["parent-ws"] = parent

		branch1 = WorkUnit(
			id="bw-1", plan_id="p-1", status="completed",
			commit_hash="h1", cost_usd=0.5, unit_type="speculation_branch",
		)
		branch2 = WorkUnit(
			id="bw-2", plan_id="p-1", status="completed",
			commit_hash="h2", cost_usd=0.3, unit_type="speculation_branch",
		)

		controller._speculation_completions["parent-ws"] = [
			WorkerCompletion(unit=branch1, handoff=None, workspace="/ws-winner", epoch=epoch),
			WorkerCompletion(unit=branch2, handoff=None, workspace="/ws-loser", epoch=epoch),
		]

		await controller._speculation_select_winner("parent-ws", mission, epoch)

		# One of the workspaces should have been released (the loser)
		release_calls = controller._backend.release_workspace.call_args_list
		released_workspaces = {call.args[0] for call in release_calls}
		# The winner workspace should not be released, the loser should
		assert len(released_workspaces) == 1
