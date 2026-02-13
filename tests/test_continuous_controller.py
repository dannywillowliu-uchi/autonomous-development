"""Tests for ContinuousController."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import ContinuousConfig, MissionConfig
from mission_control.continuous_controller import (
	ContinuousController,
	ContinuousMissionResult,
	WorkerCompletion,
)
from mission_control.db import Database
from mission_control.green_branch import UnitMergeResult
from mission_control.models import Epoch, Handoff, Mission, Plan, Signal, WorkUnit


def _db() -> Database:
	return Database(":memory:")


def _config() -> MissionConfig:
	mc = MissionConfig()
	mc.continuous = ContinuousConfig(
		stall_threshold_units=5,
		stall_score_epsilon=0.01,
		backlog_min_size=2,
	)
	return mc


class TestShouldStop:
	def test_running_false(self) -> None:
		ctrl = ContinuousController(_config(), _db())
		ctrl.running = False
		assert ctrl._should_stop(Mission(id="m1")) == "user_stopped"

	def test_no_stop_condition(self) -> None:
		ctrl = ContinuousController(_config(), _db())
		assert ctrl._should_stop(Mission(id="m1")) == ""

	def test_stall_detection(self) -> None:
		ctrl = ContinuousController(_config(), _db())
		ctrl._units_since_improvement = 5  # equals threshold
		assert ctrl._should_stop(Mission(id="m1")) == "stalled"

	def test_stall_below_threshold(self) -> None:
		ctrl = ContinuousController(_config(), _db())
		ctrl._units_since_improvement = 4
		assert ctrl._should_stop(Mission(id="m1")) == ""

	def test_objective_met(self) -> None:
		ctrl = ContinuousController(_config(), _db())
		ctrl._current_score = 0.95
		assert ctrl._should_stop(Mission(id="m1")) == "objective_met"

	def test_score_below_threshold(self) -> None:
		ctrl = ContinuousController(_config(), _db())
		ctrl._current_score = 0.89
		assert ctrl._should_stop(Mission(id="m1")) == ""

	def test_signal_stopped(self) -> None:
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(mission_id="m1", signal_type="stop")
		db.insert_signal(signal)

		ctrl = ContinuousController(_config(), db)
		assert ctrl._should_stop(Mission(id="m1")) == "signal_stopped"
		assert not ctrl.running


class TestCheckSignals:
	def test_stop_signal(self) -> None:
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(mission_id="m1", signal_type="stop")
		db.insert_signal(signal)

		ctrl = ContinuousController(_config(), db)
		result = ctrl._check_signals("m1")
		assert result == "signal_stopped"
		assert not ctrl.running

	def test_adjust_signal(self) -> None:
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 4}',
		)
		db.insert_signal(signal)

		config = _config()
		ctrl = ContinuousController(config, db)
		result = ctrl._check_signals("m1")
		assert result == ""  # adjust doesn't stop
		assert config.scheduler.parallel.num_workers == 4

	def test_no_signals(self) -> None:
		ctrl = ContinuousController(_config(), _db())
		assert ctrl._check_signals("m1") == ""


class TestHandleAdjustSignal:
	def test_adjust_num_workers(self) -> None:
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 8}',
		)
		db.insert_signal(signal)

		config = _config()
		ctrl = ContinuousController(config, db)
		ctrl._handle_adjust_signal(signal)
		assert config.scheduler.parallel.num_workers == 8

	def test_adjust_wall_time(self) -> None:
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"max_wall_time": 3600}',
		)
		db.insert_signal(signal)

		config = _config()
		ctrl = ContinuousController(config, db)
		ctrl._handle_adjust_signal(signal)
		assert config.continuous.max_wall_time_seconds == 3600


class TestProcessCompletions:
	@pytest.mark.asyncio
	async def test_merged_unit_updates_score(self) -> None:
		"""Merged unit should update the running score."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		config = _config()
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		# Mock green branch manager
		mock_gbm = MagicMock()
		mock_gbm.verify_and_merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		ctrl._green_branch = mock_gbm

		# Create epoch
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)

		# Create plan for work unit FK
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		# Queue a completed unit
		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
		)
		db.insert_work_unit(unit)

		# Handoff is already inserted by _execute_single_unit; here we just
		# pass the Python object to the completion (no DB insert needed).
		handoff = Handoff(
			work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="completed", summary="Done",
		)

		completion = WorkerCompletion(
			unit=unit, handoff=handoff, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)

		# Stop after processing one
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		assert len(result.unit_scores) == 1
		assert ctrl._total_merged == 1
		mock_gbm.verify_and_merge_unit.assert_called_once()

	@pytest.mark.asyncio
	async def test_failed_merge_counts_as_failure(self) -> None:
		"""Unit that fails verify+merge should increment failure count."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		config = _config()
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.verify_and_merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=False, rebase_ok=False,
				failure_output="Merge conflict",
			),
		)
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		assert ctrl._total_failed == 1
		assert ctrl._total_merged == 0

	@pytest.mark.asyncio
	async def test_completed_no_commits_counts_as_merged(self) -> None:
		"""Unit completed without commits should count as merged."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(_config(), db)
		result = ContinuousMissionResult(mission_id="m1")
		ctrl._green_branch = MagicMock()

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash=None,
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		assert ctrl._total_merged == 1

	@pytest.mark.asyncio
	async def test_failed_unit_counts_as_failure(self) -> None:
		"""Unit with failed status should count as failure."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(_config(), db)
		result = ContinuousMissionResult(mission_id="m1")
		ctrl._green_branch = MagicMock()

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="failed",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		assert ctrl._total_failed == 1

	@pytest.mark.asyncio
	async def test_stall_tracking(self) -> None:
		"""Units with no score improvement should increment stall counter."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		config = _config()
		config.continuous.stall_score_epsilon = 1.0  # very large: any change is within epsilon
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")
		ctrl._green_branch = MagicMock()

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash=None,
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		# Score change should be within epsilon (1.0) -> stall incremented
		assert ctrl._units_since_improvement >= 1

	@pytest.mark.asyncio
	async def test_objective_met_stops(self) -> None:
		"""When score >= 0.9, processor should set objective_met and stop."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		config = _config()
		ctrl = ContinuousController(config, db)
		ctrl._current_score = 0.89  # close to threshold
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.verify_and_merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc",
			branch_name="mc/unit-wu1",
		)
		db.insert_work_unit(unit)

		# Patch compute_running_score to return high score
		with patch(
			"mission_control.continuous_controller.compute_running_score",
		) as mock_score:
			from mission_control.evaluator import ObjectiveEvaluation
			mock_score.return_value = ObjectiveEvaluation(
				score=0.95, met=True, reasoning="test",
			)

			completion = WorkerCompletion(
				unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
			)
			ctrl._completion_queue.put_nowait(completion)

			await ctrl._process_completions(Mission(id="m1"), result)

			assert result.objective_met is True
			assert result.stopped_reason == "objective_met"
			assert not ctrl.running


class TestStop:
	def test_stop_sets_running_false(self) -> None:
		ctrl = ContinuousController(_config(), _db())
		assert ctrl.running is True
		ctrl.stop()
		assert ctrl.running is False


class TestWorkerCompletion:
	def test_dataclass_fields(self) -> None:
		unit = WorkUnit(id="wu1", title="Task")
		epoch = Epoch(id="ep1", mission_id="m1")
		wc = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		assert wc.unit.id == "wu1"
		assert wc.workspace == "/tmp/ws"
		assert wc.handoff is None


class TestContinuousMissionResult:
	def test_defaults(self) -> None:
		r = ContinuousMissionResult()
		assert r.mission_id == ""
		assert r.final_score == 0.0
		assert r.objective_met is False
		assert r.total_units_dispatched == 0
		assert r.unit_scores == []
