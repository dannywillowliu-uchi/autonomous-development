"""Tests for ContinuousController."""

from __future__ import annotations

import asyncio
import time
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

	def test_wall_time_exceeded(self) -> None:
		config = _config()
		config.continuous.max_wall_time_seconds = 10
		ctrl = ContinuousController(config, _db())
		ctrl._start_time = time.monotonic() - 20  # 20s elapsed, limit is 10s
		assert ctrl._should_stop(Mission(id="m1")) == "wall_time_exceeded"

	def test_wall_time_not_exceeded(self) -> None:
		config = _config()
		config.continuous.max_wall_time_seconds = 100
		ctrl = ContinuousController(config, _db())
		ctrl._start_time = time.monotonic() - 10  # 10s elapsed, limit is 100s
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
		mock_gbm.merge_unit = AsyncMock(
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
		mock_gbm.merge_unit.assert_called_once()

	@pytest.mark.asyncio
	async def test_failed_merge_counts_as_failure(self) -> None:
		"""Unit that fails verify+merge should increment failure count."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		config = _config()
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
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
		mock_gbm.merge_unit = AsyncMock(
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


class TestExecuteSingleUnit:
	@pytest.mark.asyncio
	async def test_provision_failure_queues_failed_completion(self) -> None:
		"""When workspace provisioning fails, a failed completion is queued."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)

		config = _config()
		config.target.path = "/tmp/test"
		ctrl = ContinuousController(config, db)

		mock_backend = AsyncMock()
		mock_backend.provision_workspace.side_effect = RuntimeError("Pool exhausted")
		ctrl._backend = mock_backend

		async def mock_locked_call(method: str, *args: object) -> None:
			getattr(db, method)(*args)
		db.locked_call = mock_locked_call  # type: ignore[attr-defined]

		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)
		semaphore = asyncio.Semaphore(1)

		await ctrl._execute_single_unit(unit, epoch, Mission(id="m1"), semaphore)

		assert not ctrl._completion_queue.empty()
		completion = ctrl._completion_queue.get_nowait()
		assert completion.unit.status == "failed"
		assert "Pool exhausted" in completion.unit.output_summary


class TestEndToEnd:
	@pytest.mark.asyncio
	async def test_units_flow_dispatch_to_completion(self) -> None:
		"""Integration: dispatch loop feeds units, completion processor scores and stops."""
		db = _db()
		config = _config()
		config.target.path = "/tmp/test"
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 30
		ctrl = ContinuousController(config, db)

		executed_ids: list[str] = []

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			executed_ids.append(unit.id)
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(
					unit=unit, handoff=None,
					workspace="/tmp/ws", epoch=epoch,
				),
			)
			semaphore.release()

		# Mock planner: returns 2 units per call
		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(
				id=f"ep{call_count}", mission_id=mission.id,
				number=call_count,
			)
			units = [
				WorkUnit(
					id=f"wu-{call_count}-{i}", plan_id=plan.id,
					title=f"Task {call_count}.{i}",
				)
				for i in range(2)
			]
			return plan, units, epoch

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()

		# Scores escalate: 0.5, 0.7, 0.95 (met) -- stops on 3rd unit
		score_iter = iter([0.5, 0.7, 0.95, 0.95, 0.95])

		def mock_score(**kwargs: object) -> object:
			from mission_control.evaluator import ObjectiveEvaluation
			s = next(score_iter, 0.95)
			return ObjectiveEvaluation(score=s, met=(s >= 0.9), reasoning="test")

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch(
				"mission_control.continuous_controller.compute_running_score",
				side_effect=mock_score,
			),
		):
			result = await ctrl.run()

		assert result.objective_met is True
		assert result.stopped_reason == "objective_met"
		assert len(executed_ids) >= 2
		assert result.total_units_merged >= 2
		assert result.final_score >= 0.9
