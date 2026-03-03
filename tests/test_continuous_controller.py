"""Tests for ContinuousController."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.backends import LocalBackend
from mission_control.config import MissionConfig
from mission_control.continuous_controller import (
	ContinuousController,
	ContinuousMissionResult,
	DynamicSemaphore,
	WorkerCompletion,
)
from mission_control.db import Database
from mission_control.green_branch import UnitMergeResult
from mission_control.models import Epoch, Handoff, Mission, Plan, Signal, Worker, WorkUnit


def _assert_semaphore_available(sem: asyncio.Semaphore | DynamicSemaphore, expected: int) -> None:
	"""Assert the number of available permits on a semaphore.

	For 0 expected, checks locked(). For N > 0, reads the internal _value
	counter. Works with both asyncio.Semaphore and DynamicSemaphore.
	"""
	if expected == 0:
		assert sem.locked(), "Expected 0 available permits but semaphore is not locked"
		return
	actual = sem._value
	assert actual == expected, f"Expected {expected} available permits, got {actual}"


class TestShouldStop:
	def test_running_false(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		ctrl.running = False
		assert ctrl._should_stop(Mission(id="m1")) == "user_stopped"

	def test_no_stop_condition(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl._should_stop(Mission(id="m1")) == ""

	def test_wall_time_exceeded(self, config: MissionConfig, db: Database) -> None:
		config.continuous.max_wall_time_seconds = 10
		ctrl = ContinuousController(config, db)
		ctrl._start_time = time.monotonic() - 20  # 20s elapsed, limit is 10s
		assert ctrl._should_stop(Mission(id="m1")) == "wall_time_exceeded"

	def test_signal_stopped(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(mission_id="m1", signal_type="stop")
		db.insert_signal(signal)

		ctrl = ContinuousController(config, db)
		assert ctrl._should_stop(Mission(id="m1")) == "signal_stopped"
		assert not ctrl.running


class TestCheckSignals:
	def test_stop_signal(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(mission_id="m1", signal_type="stop")
		db.insert_signal(signal)

		ctrl = ContinuousController(config, db)
		result = ctrl._check_signals("m1")
		assert result == "signal_stopped"
		assert not ctrl.running

	def test_adjust_signal(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 4}',
		)
		db.insert_signal(signal)

		ctrl = ContinuousController(config, db)
		result = ctrl._check_signals("m1")
		assert result == ""  # adjust doesn't stop
		assert config.scheduler.parallel.num_workers == 4



class TestHandleAdjustSignal:
	def test_adjust_num_workers(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 8}',
		)
		db.insert_signal(signal)

		ctrl = ContinuousController(config, db)
		ctrl._handle_adjust_signal(signal)
		assert config.scheduler.parallel.num_workers == 8

	def test_adjust_wall_time(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"max_wall_time": 3600}',
		)
		db.insert_signal(signal)

		ctrl = ContinuousController(config, db)
		ctrl._handle_adjust_signal(signal)
		assert config.continuous.max_wall_time_seconds == 3600


class TestPauseResumeSignals:
	def test_pause_then_resume(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))

		ctrl = ContinuousController(config, db)

		# Pause
		pause_sig = Signal(mission_id="m1", signal_type="pause")
		db.insert_signal(pause_sig)
		ctrl._check_signals("m1")
		assert ctrl._paused is True

		# Resume
		resume_sig = Signal(mission_id="m1", signal_type="resume")
		db.insert_signal(resume_sig)
		ctrl._check_signals("m1")
		assert ctrl._paused is False


class TestCancelUnitSignal:
	def test_cancel_running_unit(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))

		ctrl = ContinuousController(config, db)

		# Create a mock task
		mock_task = MagicMock(spec=asyncio.Task)
		mock_task.done.return_value = False
		ctrl._unit_tasks["unit123"] = mock_task

		signal = Signal(
			mission_id="m1",
			signal_type="cancel_unit",
			payload='{"unit_id": "unit123"}',
		)
		db.insert_signal(signal)
		ctrl._handle_cancel_unit(signal)

		mock_task.cancel.assert_called_once()

	def test_cancel_already_done_unit(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))

		ctrl = ContinuousController(config, db)

		mock_task = MagicMock(spec=asyncio.Task)
		mock_task.done.return_value = True
		ctrl._unit_tasks["unit123"] = mock_task

		signal = Signal(
			mission_id="m1",
			signal_type="cancel_unit",
			payload='{"unit_id": "unit123"}',
		)
		db.insert_signal(signal)
		ctrl._handle_cancel_unit(signal)

		mock_task.cancel.assert_not_called()


class TestForceRetrySignal:
	def test_force_retry_resets_unit(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="failed", attempt=2, max_attempts=3,
			output_summary="Some error",
		)
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		signal = Signal(
			mission_id="m1",
			signal_type="force_retry",
			payload='{"unit_id": "wu1"}',
		)
		db.insert_signal(signal)
		ctrl._handle_force_retry(signal)

		db_unit = db.get_work_unit("wu1")
		assert db_unit is not None
		assert db_unit.status == "pending"
		assert "[Force retry]" in db_unit.description

	def test_force_retry_nonexistent_unit(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))

		ctrl = ContinuousController(config, db)
		signal = Signal(
			mission_id="m1",
			signal_type="force_retry",
			payload='{"unit_id": "nonexistent"}',
		)
		db.insert_signal(signal)
		# Should not raise
		ctrl._handle_force_retry(signal)


class TestAddObjectiveSignal:
	def test_add_objective_appends_to_config(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))

		config.target.objective = "Original objective"
		ctrl = ContinuousController(config, db)
		ctrl._planner = MagicMock()

		signal = Signal(
			mission_id="m1",
			signal_type="add_objective",
			payload='{"objective": "Also add logging"}',
		)
		db.insert_signal(signal)
		ctrl._handle_add_objective(signal)

		assert "Also add logging" in config.target.objective
		assert "Original objective" in config.target.objective

	def test_add_objective_no_planner(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))

		config.target.objective = "Original"
		ctrl = ContinuousController(config, db)
		# _planner is None

		signal = Signal(
			mission_id="m1",
			signal_type="add_objective",
			payload='{"objective": "New obj"}',
		)
		db.insert_signal(signal)
		# Should not raise even without planner
		ctrl._handle_add_objective(signal)
		# Objective should NOT be appended since planner is None
		assert "New obj" not in config.target.objective


class TestProcessCompletions:
	@pytest.mark.asyncio
	async def test_merged_unit_updates_score(self, config: MissionConfig, db: Database) -> None:
		"""Merged unit should update the running score."""
		db.insert_mission(Mission(id="m1", objective="test"))
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

		await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		assert ctrl._total_merged == 1
		mock_gbm.merge_unit.assert_called_once()

	@pytest.mark.asyncio
	async def test_failed_merge_counts_as_failure(self, config: MissionConfig, db: Database) -> None:
		"""Unit that fails verify+merge should increment failure count."""
		db.insert_mission(Mission(id="m1", objective="test"))
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
			attempt=3, max_attempts=3,
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		assert ctrl._total_failed == 1
		assert ctrl._total_merged == 0

	@pytest.mark.asyncio
	async def test_failed_unit_counts_as_failure(self, config: MissionConfig, db: Database) -> None:
		"""Unit with failed status should count as failure."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")
		ctrl._green_branch = MagicMock()

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="failed", attempt=3, max_attempts=3,
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		assert ctrl._total_failed == 1

	@pytest.mark.asyncio
	async def test_research_unit_skips_merge(self, config: MissionConfig, db: Database) -> None:
		"""Research units should skip merge_unit() but count as merged."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()
		ctrl._green_branch = mock_gbm

		mock_planner = MagicMock()
		mock_planner.ingest_handoff = MagicMock()
		ctrl._planner = mock_planner

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Research task",
			status="completed", commit_hash=None,
			unit_type="research",
		)
		db.insert_work_unit(unit)

		handoff = Handoff(
			work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="completed", summary="Found useful info",
			discoveries=["pattern X exists"],
		)

		completion = WorkerCompletion(
			unit=unit, handoff=handoff, workspace="/tmp/ws", epoch=epoch,
		)

		await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		# Should count as merged, not failed
		assert ctrl._total_merged == 1
		assert ctrl._total_failed == 0
		# merge_unit should NOT have been called
		mock_gbm.merge_unit.assert_not_called()



class TestStop:
	def test_stop_sets_running_false(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl.running is True
		ctrl.stop()
		assert ctrl.running is False


class TestExecuteSingleUnit:
	@pytest.mark.asyncio
	async def test_provision_failure_queues_failed_completion(self, config: MissionConfig, db: Database) -> None:
		"""When workspace provisioning fails, a failed completion is queued."""
		db.insert_mission(Mission(id="m1", objective="test"))
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)

		ctrl = ContinuousController(config, db)

		mock_backend = AsyncMock()
		mock_backend.provision_workspace.side_effect = RuntimeError("Pool exhausted")
		ctrl._backend = mock_backend

		async def mock_locked_call(method: str, *args: object) -> None:
			getattr(db, method)(*args)
		db.locked_call = mock_locked_call  # type: ignore[attr-defined]

		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)
		ctrl._semaphore = DynamicSemaphore(1)

		await ctrl._execute_single_unit(unit, epoch, Mission(id="m1"))

		# _fail_unit handles the failure internally (no queue)
		assert unit.status == "failed"
		assert "Pool exhausted" in unit.output_summary


class TestEndToEnd:
	@pytest.mark.asyncio
	async def test_units_flow_dispatch_to_completion(self, config: MissionConfig, db: Database) -> None:
		"""Integration: dispatch loop feeds units, completion processor merges them."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1

		ctrl = ContinuousController(config, db)

		executed_ids: list[str] = []

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
		) -> WorkerCompletion | None:
			executed_ids.append(unit.id)
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			completion = WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch)
			ctrl._semaphore.release()
			ctrl._in_flight_count = max(ctrl._in_flight_count - 1, 0)
			return completion

		# Mock planner: returns 2 units first call, then empty (mission done)
		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(
				id=f"ep{call_count}", mission_id=mission.id,
				number=call_count,
			)
			if call_count == 1:
				units = [
					WorkUnit(
						id=f"wu-{call_count}-{i}", plan_id=plan.id,
						title=f"Task {call_count}.{i}",
					)
					for i in range(2)
				]
			else:
				units = []
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
			ctrl._notifier = None
			ctrl._heartbeat = None
			ctrl._event_stream = None

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=5.0)

		assert len(executed_ids) >= 2
		assert result.objective_met is True
		assert result.stopped_reason == "planner_completed"
		assert result.total_units_dispatched >= 2


class TestRetry:
	def _make_ctrl(self, config: MissionConfig, db: Database) -> tuple[ContinuousController, Database]:
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._green_branch = MagicMock()
		ctrl._semaphore = DynamicSemaphore(2)
		return ctrl, db

	@pytest.mark.asyncio
	async def test_failed_unit_retried_when_under_max_attempts(self, config: MissionConfig, db: Database) -> None:
		"""Unit fails with attempt=1, max_attempts=3 -> gets re-queued, not counted as failed."""
		ctrl, db = self._make_ctrl(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="failed", attempt=1, max_attempts=3,
			output_summary="Some error occurred",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		with patch.object(ctrl, "_retry_unit", new_callable=AsyncMock):
			await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		assert ctrl._total_failed == 0
		assert unit.status == "pending"

	@pytest.mark.asyncio
	async def test_failed_unit_not_retried_at_max_attempts(self, config: MissionConfig, db: Database) -> None:
		"""Unit fails with attempt=3, max_attempts=3 -> counted as failed, no retry."""
		ctrl, db = self._make_ctrl(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="failed", attempt=3, max_attempts=3,
			output_summary="Some error occurred",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		assert ctrl._total_failed == 1
		assert unit.status == "failed"

	@pytest.mark.asyncio
	async def test_retry_appends_failure_context(self, config: MissionConfig, db: Database) -> None:
		"""Verify the description is augmented with failure info on retry."""
		ctrl, db = self._make_ctrl(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			description="Original description",
			status="failed", attempt=1, max_attempts=3,
			output_summary="Import error in main.py",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		with patch.object(ctrl, "_retry_unit", new_callable=AsyncMock):
			await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		assert "[Retry attempt 2]" in unit.description  # attempt incremented from 1->2 inside _schedule_retry
		assert "Import error in main.py" in unit.description
		assert "Avoid the same mistake" in unit.description
		assert unit.description.startswith("Original description")

	def test_retry_delay_exponential_backoff(self, config: MissionConfig, db: Database) -> None:
		"""Verify delay computation: base_delay * 2^(attempt-1), capped at max.

		_schedule_retry increments attempt first, so starting at attempt=0:
		  -> incremented to 1 -> delay = 30 * 2^0 = 30
		"""
		ctrl, _ = self._make_ctrl(config, db)
		config = ctrl.config
		config.continuous.retry_base_delay_seconds = 30
		config.continuous.retry_max_delay_seconds = 300

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		mission = Mission(id="m1")

		# attempt=0 -> _schedule_retry increments to 1 -> delay = 30 * 2^0 = 30
		plan = Plan(id="p1", objective="test")
		ctrl.db.insert_plan(plan)
		unit1 = WorkUnit(id="wu1", plan_id="p1", title="T", attempt=0, max_attempts=5)
		ctrl.db.insert_work_unit(unit1)
		with patch("asyncio.create_task"):
			ctrl._schedule_retry(unit1, epoch, mission, "err", config.continuous)
		assert unit1.status == "pending"
		assert unit1.attempt == 1  # incremented from 0

		# Verify delay values directly (after increment, delay uses attempt-1)
		assert min(30 * (2 ** 0), 300) == 30   # attempt=1 -> 2^0
		assert min(30 * (2 ** 1), 300) == 60   # attempt=2 -> 2^1
		assert min(30 * (2 ** 2), 300) == 120  # attempt=3 -> 2^2
		assert min(30 * (2 ** 3), 300) == 240  # attempt=4 -> 2^3
		assert min(30 * (2 ** 4), 300) == 300  # attempt=5 -> 2^4, capped

	@pytest.mark.asyncio
	async def test_merge_failure_triggers_retry(self, config: MissionConfig, db: Database) -> None:
		"""Unit completes but merge fails -> retried if under max_attempts."""
		ctrl, db = self._make_ctrl(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=False, rebase_ok=False,
				failure_output="Merge conflict in utils.py",
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
			attempt=1, max_attempts=3,
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		with patch.object(ctrl, "_retry_unit", new_callable=AsyncMock):
			await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		assert ctrl._total_failed == 0
		assert ctrl._total_merged == 0
		assert unit.status == "pending"
		assert "[Retry attempt 2]" in unit.description  # attempt incremented from 1->2 inside _schedule_retry


class TestVenvSymlink:
	"""Verify .venv is symlinked into green branch workspace."""

	@pytest.mark.asyncio
	async def test_venv_symlinked_when_exists(self, config: MissionConfig, db: Database) -> None:
		"""Source .venv should be symlinked into green branch workspace."""
		import tempfile

		ctrl = ContinuousController(config, db)

		with tempfile.TemporaryDirectory() as source_repo:
			source_venv = Path(source_repo) / ".venv"
			source_venv.mkdir()
			(source_venv / "bin").mkdir()

			with tempfile.TemporaryDirectory() as gb_workspace:
				config.target.path = source_repo

				mock_backend = AsyncMock(spec=LocalBackend)
				mock_backend.provision_workspace = AsyncMock(return_value=gb_workspace)
				mock_backend.initialize = AsyncMock()
				ctrl._backend = mock_backend

				mock_gbm = MagicMock()
				mock_gbm.initialize = AsyncMock()

				with (
					patch("mission_control.continuous_controller.GreenBranchManager", return_value=mock_gbm),
					patch("mission_control.continuous_controller.LocalBackend", return_value=mock_backend),
					patch.object(ctrl, "_backend", mock_backend),
				):
					# Simulate the relevant part of _init_components
					ctrl._green_branch = mock_gbm
					await mock_gbm.initialize(gb_workspace)

					workspace_venv = Path(gb_workspace) / ".venv"
					if source_venv.exists() and not workspace_venv.exists():
						workspace_venv.symlink_to(source_venv)

					assert workspace_venv.is_symlink()
					assert workspace_venv.resolve() == source_venv.resolve()


class TestWorkerRecordPersistence:
	"""Tests that Worker DB records are created/updated during _execute_single_unit."""

	def _setup(self, config: MissionConfig, db: Database) -> tuple:
		db.insert_mission(Mission(id="m1", objective="test"))
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)

		ctrl = ContinuousController(config, db)

		async def mock_locked_call(method: str, *args: object) -> object:
			return getattr(db, method)(*args)
		db.locked_call = mock_locked_call  # type: ignore[attr-defined]

		return db, config, ctrl, epoch

	@pytest.mark.asyncio
	async def test_worker_created_on_spawn(self, config: MissionConfig, db: Database) -> None:
		"""Worker record is inserted after backend.spawn() succeeds."""
		db, config, ctrl, epoch = self._setup(config, db)

		mock_backend = AsyncMock()
		mock_handle = MagicMock()
		mock_handle.pid = 42
		mock_handle.workspace_path = "/tmp/ws/wu1"
		mock_backend.spawn.return_value = mock_handle
		mock_backend.check_status.return_value = "completed"
		mock_backend.get_output.return_value = ""
		ctrl._backend = mock_backend

		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)
		ctrl._semaphore = DynamicSemaphore(1)

		await ctrl._execute_single_unit(unit, epoch, Mission(id="m1"))

		worker = db.get_worker("wu1")
		assert worker is not None
		assert worker.workspace_path == "/tmp/ws/wu1"
		assert worker.backend_type == "local"

	@pytest.mark.asyncio
	async def test_worker_idle_on_completion(self, config: MissionConfig, db: Database) -> None:
		"""Worker status set to idle after successful completion."""
		db, config, ctrl, epoch = self._setup(config, db)

		mc_result = json.dumps({
			"status": "completed", "commits": ["abc123"],
			"summary": "Done", "files_changed": [], "discoveries": [], "concerns": [],
		})
		output = f"MC_RESULT:{mc_result}"

		mock_backend = AsyncMock()
		mock_handle = MagicMock()
		mock_handle.pid = 42
		mock_handle.workspace_path = "/tmp/ws"
		mock_backend.spawn.return_value = mock_handle
		mock_backend.check_status.return_value = "completed"
		mock_backend.get_output.return_value = output
		ctrl._backend = mock_backend

		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)
		ctrl._semaphore = DynamicSemaphore(1)

		await ctrl._execute_single_unit(unit, epoch, Mission(id="m1"))

		worker = db.get_worker("wu1")
		assert worker is not None
		assert worker.status == "idle"
		assert worker.current_unit_id is None
		assert worker.pid is None
		assert worker.units_completed == 1
		assert worker.units_failed == 0



class TestFailUnitHelper:
	@pytest.mark.asyncio
	async def test_updates_unit_and_worker_and_queues(self, config: MissionConfig, db: Database) -> None:
		"""_fail_unit should update unit status, worker status, and put completion on queue."""
		db.insert_mission(Mission(id="m1", objective="test"))
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)

		ctrl = ContinuousController(config, db)

		async def mock_locked_call(method: str, *args: object) -> object:
			return getattr(db, method)(*args)
		db.locked_call = mock_locked_call  # type: ignore[attr-defined]

		unit = WorkUnit(id="wu1", plan_id="p1", title="Task", attempt=0)
		db.insert_work_unit(unit)

		worker = Worker(id="w1", workspace_path="/tmp/ws", status="working")
		db.insert_worker(worker)

		await ctrl._fail_unit(unit, worker, epoch, "test failure", "/tmp/ws")

		assert unit.status == "failed"
		assert unit.attempt == 1
		assert unit.output_summary == "test failure"
		assert unit.finished_at is not None
		assert worker.status == "dead"
		assert worker.units_failed == 1



class TestHandleAdjustSemaphoreRebuild:
	def test_semaphore_rebuilt_with_new_value(self, config: MissionConfig, db: Database) -> None:
		"""_handle_adjust_signal should adjust semaphore capacity in-place."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = DynamicSemaphore(2)
		ctrl._in_flight_count = 0

		original_sem = ctrl._semaphore
		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 6}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		assert config.scheduler.parallel.num_workers == 6
		assert ctrl._semaphore is original_sem  # same object, not replaced
		assert ctrl._semaphore.capacity == 6
		_assert_semaphore_available(ctrl._semaphore, 6)

	def test_semaphore_accounts_for_in_flight(self, config: MissionConfig, db: Database) -> None:
		"""Semaphore adjust increases available slots even with in-flight workers."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = DynamicSemaphore(2)
		ctrl._in_flight_count = 3  # 3 workers currently executing

		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 5}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		assert config.scheduler.parallel.num_workers == 5
		assert ctrl._semaphore is not None
		assert ctrl._semaphore.capacity == 5
		# adjust(5) adds 3 slots to underlying semaphore (5-2=3)
		# so available = original 2 + 3 = 5
		_assert_semaphore_available(ctrl._semaphore, 5)

	def test_semaphore_clamps_to_zero_when_in_flight_exceeds(self, config: MissionConfig, db: Database) -> None:
		"""When reducing capacity, debt absorbs future releases."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = DynamicSemaphore(4)
		ctrl._in_flight_count = 4  # all 4 slots used

		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 2}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		assert config.scheduler.parallel.num_workers == 2
		assert ctrl._semaphore is not None
		assert ctrl._semaphore.capacity == 2
		# Debt of 2 means 2 future releases will be swallowed
		assert ctrl._semaphore._debt == 2



class TestSequentialOrchestration:
	@pytest.mark.asyncio
	async def test_sequential_dispatch_and_completion(self, config: MissionConfig, db: Database) -> None:
		"""Orchestration loop dispatches units and processes completions sequentially."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 10
		config.continuous.cooldown_between_units = 0

		ctrl = ContinuousController(config, db)

		events: list[str] = []

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
		) -> WorkerCompletion | None:
			events.append(f"exec:{unit.id}")
			await asyncio.sleep(0.01)
			unit.status = "completed"
			unit.commit_hash = "abc123"
			unit.branch_name = f"mc/unit-{unit.id}"
			unit.finished_at = "2025-01-01T00:00:00"
			completion = WorkerCompletion(
				unit=unit, handoff=None,
				workspace="/tmp/ws", epoch=epoch,
			)
			ctrl._semaphore.release()
			ctrl._in_flight_count = max(ctrl._in_flight_count - 1, 0)
			return completion

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [
					WorkUnit(
						id=f"wu-{i}", plan_id=plan.id,
						title=f"Task {i}", max_attempts=1,
					)
					for i in range(3)
				]
				return plan, units, epoch
			return plan, [], epoch

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(merged=True, rebase_ok=True, verification_passed=True),
		)

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()
			ctrl._notifier = None
			ctrl._heartbeat = None
			ctrl._event_stream = None

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=10.0)

		assert result.total_units_dispatched >= 3
		assert ctrl._total_merged + ctrl._total_failed == 3
		assert ctrl._total_merged == 3
		assert ctrl._total_failed == 0
		assert len([e for e in events if e.startswith("exec:")]) == 3


class TestAdjustWorkersDuringDispatch:
	@pytest.mark.asyncio
	async def test_adjust_workers_during_dispatch(self, config: MissionConfig, db: Database) -> None:
		"""Adjust workers signal mid-dispatch resizes the semaphore without orphaning tasks."""
		config.scheduler.parallel.num_workers = 2
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)

		# Simulate active dispatch state: capacity=2 with 0 available (both acquired), 2 in-flight
		ctrl._semaphore = DynamicSemaphore(2)
		await ctrl._semaphore.acquire()
		await ctrl._semaphore.acquire()
		ctrl._in_flight_count = 2
		original_sem = ctrl._semaphore

		mock_task1 = MagicMock(spec=asyncio.Task)
		mock_task1.done.return_value = False
		mock_task2 = MagicMock(spec=asyncio.Task)
		mock_task2.done.return_value = False
		ctrl._active_tasks = {mock_task1, mock_task2}

		# Now adjust workers from 2 -> 4 while both are in-flight
		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 4}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		# Verify config updated
		assert config.scheduler.parallel.num_workers == 4
		# Same semaphore object, adjusted in-place
		assert ctrl._semaphore is original_sem
		assert ctrl._semaphore.capacity == 4
		# 2 new slots added: should be able to acquire 2 more
		_assert_semaphore_available(ctrl._semaphore, 2)

		# Verify new capacity allows 2 more dispatches
		acquired = ctrl._semaphore.acquire()
		done = asyncio.ensure_future(acquired)
		await asyncio.sleep(0)
		assert done.done()

		# Verify that no tasks were cancelled (no orphans)
		mock_task1.cancel.assert_not_called()
		mock_task2.cancel.assert_not_called()


class TestCancelUnitDuringMerge:
	@pytest.mark.asyncio
	async def test_cancel_unit_during_merge(self, config: MissionConfig, db: Database) -> None:
		"""Cancel signal while _process_unit_completion is mid-merge cleans up properly."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		merge_started = asyncio.Event()
		merge_proceed = asyncio.Event()

		async def slow_merge(workspace: str, branch: str, **kwargs: object) -> UnitMergeResult:
			merge_started.set()
			await merge_proceed.wait()
			return UnitMergeResult(merged=True, rebase_ok=True, verification_passed=True)

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(side_effect=slow_merge)
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

		# Create a mock task for the unit so cancel_unit can find it
		mock_task = MagicMock(spec=asyncio.Task)
		mock_task.done.return_value = False
		ctrl._unit_tasks["wu1"] = mock_task

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		# Run _process_single_completion in background
		async def run_processor() -> None:
			await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		processor_task = asyncio.create_task(run_processor())

		# Wait for merge to start
		await asyncio.wait_for(merge_started.wait(), timeout=5.0)

		# Send cancel signal while merge is in progress
		cancel_signal = Signal(
			mission_id="m1",
			signal_type="cancel_unit",
			payload='{"unit_id": "wu1"}',
		)
		db.insert_signal(cancel_signal)
		ctrl._handle_cancel_unit(cancel_signal)

		# Verify cancel was called on the task
		mock_task.cancel.assert_called_once()

		# Let merge complete (simulates the race where merge finishes despite cancel)
		merge_proceed.set()

		await asyncio.wait_for(processor_task, timeout=5.0)

		# The merge completed successfully, so it counts as merged
		assert ctrl._total_merged == 1
		mock_gbm.merge_unit.assert_called_once()


class TestDryRun:
	@pytest.mark.asyncio
	async def test_dry_run_returns_early(self, config: MissionConfig, db: Database) -> None:
		"""dry_run=True should call planner once and return empty result."""
		ctrl = ContinuousController(config, db)

		plan = Plan(id="p1", objective="test")
		epoch = Epoch(id="ep1", mission_id="m1", number=0)
		units = [
			WorkUnit(id="wu1", plan_id="p1", title="Task A", priority=1, files_hint="src/a.py"),
			WorkUnit(id="wu2", plan_id="p1", title="Task B", priority=2, files_hint="src/b.py"),
		]

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(return_value=(plan, units, epoch))

		mock_backend = AsyncMock()

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._backend = mock_backend

		with patch.object(ctrl, "_init_components", mock_init):
			result = await ctrl.run(dry_run=True)

		mock_planner.get_next_units.assert_called_once()
		assert result.mission_id == ""
		assert result.total_units_dispatched == 0
		mock_backend.cleanup.assert_called_once()



class TestScoreAmbition:
	def test_empty_units_returns_1(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl._score_ambition([]) == 1

	def test_many_diverse_units_scores_high(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		units = [
			WorkUnit(id=f"wu{i}", title=f"Task {i}", priority=1,
				files_hint=f"src/mod{i}.py, src/mod{i}_test.py",
				unit_type="implementation" if i % 2 == 0 else "research")
			for i in range(8)
		]
		score = ctrl._score_ambition(units)
		# 8 units, 16 files, mixed types, high priority -> high score
		assert score >= 7

	def test_score_clamped_to_range(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		# Even with extreme inputs, score stays 1-10
		units = [
			WorkUnit(id=f"wu{i}", title=f"T{i}", priority=1,
				files_hint=", ".join(f"f{j}.py" for j in range(20)),
				unit_type="research" if i % 2 else "implementation")
			for i in range(20)
		]
		score = ctrl._score_ambition(units)
		assert 1 <= score <= 10



class TestAmbitionScoringInDispatch:
	@pytest.mark.asyncio
	async def test_units_dispatched_and_completed(self, config: MissionConfig, db: Database) -> None:
		"""Units should be dispatched, executed, and processed via _orchestration_loop."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1

		ctrl = ContinuousController(config, db)

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [
					WorkUnit(id=f"wu-{i}", plan_id=plan.id, title=f"Task {i}", priority=2,
						files_hint=f"src/mod{i}.py, src/mod{i}_test.py",
						unit_type="implementation" if i % 2 == 0 else "research")
					for i in range(5)
				]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
		) -> WorkerCompletion | None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			completion = WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch)
			ctrl._semaphore.release()
			ctrl._in_flight_count = max(ctrl._in_flight_count - 1, 0)
			return completion

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
			ctrl._notifier = None
			ctrl._heartbeat = None
			ctrl._event_stream = None

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=5.0)

		assert result.objective_met is True
		assert result.stopped_reason == "planner_completed"
		assert result.total_units_dispatched >= 5



class TestInFlightUnitsPreventPrematureCompletion:
	@pytest.mark.asyncio
	async def test_waits_for_inflight_before_declaring_complete(self, config: MissionConfig, db: Database) -> None:
		"""Planner returns empty but units are still running -- should wait, not declare complete."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 10

		ctrl = ContinuousController(config, db)

		inflight_completed = asyncio.Event()

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
		) -> WorkerCompletion | None:
			# Simulate slow execution
			await asyncio.sleep(0.1)
			inflight_completed.set()
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			completion = WorkerCompletion(
				unit=unit, handoff=None,
				workspace="/tmp/ws", epoch=epoch,
			)
			ctrl._semaphore.release()
			ctrl._in_flight_count = max(ctrl._in_flight_count - 1, 0)
			return completion

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [
					WorkUnit(id="wu-1", plan_id=plan.id, title="Task 1"),
				]
				return plan, units, epoch
			# Second call: planner returns empty while unit is still running
			# Third call (after gather): still empty, now truly complete
			return plan, [], epoch

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
			ctrl._notifier = None
			ctrl._heartbeat = None
			ctrl._event_stream = None

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=5.0)

		# Planner was called at least twice (first returns units, second returns empty)
		assert call_count >= 2
		# The in-flight unit should have completed before mission ended
		assert inflight_completed.is_set()
		assert result.objective_met is True
		assert result.stopped_reason == "planner_completed"
		assert result.total_units_dispatched >= 1


class TestTaskDoneCallback:
	@pytest.mark.asyncio
	async def test_exception_is_logged(
		self, config: MissionConfig, db: Database, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Exceptions from fire-and-forget tasks should be logged, not swallowed."""
		ctrl = ContinuousController(config, db)

		async def failing_coro() -> None:
			raise RuntimeError("boom")

		task = asyncio.create_task(failing_coro())
		ctrl._active_tasks.add(task)
		task.add_done_callback(ctrl._task_done_callback)
		# Let the task complete and callback fire
		await asyncio.sleep(0)
		await asyncio.sleep(0)

		assert task not in ctrl._active_tasks
		assert any("Fire-and-forget task failed" in r.message and "boom" in r.message for r in caplog.records)

	@pytest.mark.asyncio
	async def test_successful_task_no_error_logged(
		self, config: MissionConfig, db: Database, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Successful tasks should not produce error log messages."""
		ctrl = ContinuousController(config, db)

		async def ok_coro() -> None:
			pass

		task = asyncio.create_task(ok_coro())
		ctrl._active_tasks.add(task)
		task.add_done_callback(ctrl._task_done_callback)
		await asyncio.sleep(0)
		await asyncio.sleep(0)

		assert task not in ctrl._active_tasks
		assert not any("Fire-and-forget task failed" in r.message for r in caplog.records)

	@pytest.mark.asyncio
	async def test_cancelled_task_no_error_logged(
		self, config: MissionConfig, db: Database, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Cancelled tasks should not produce error log messages."""
		ctrl = ContinuousController(config, db)

		async def slow_coro() -> None:
			await asyncio.sleep(100)

		task = asyncio.create_task(slow_coro())
		ctrl._active_tasks.add(task)
		task.add_done_callback(ctrl._task_done_callback)
		task.cancel()
		try:
			await task
		except asyncio.CancelledError:
			pass

		assert task not in ctrl._active_tasks
		assert not any("Fire-and-forget task failed" in r.message for r in caplog.records)


class TestStrategicContextRegression:
	def test_append_strategic_context_uses_mission_id(self, config: MissionConfig, db: Database) -> None:
		"""Regression: strategic context must use mission.id, not mission.mission_id."""
		mission = Mission(id="m-test-123", objective="Build feature X", status="completed")
		db.insert_mission(mission)

		# Verify the controller code references mission.id (not mission.mission_id)
		# by exercising the same DB call the finally block makes
		# This previously used mission.mission_id which raised AttributeError
		merged_summaries = ["Added auth module", "Fixed tests"]
		failed_summaries = ["Linting failed"]
		db.append_strategic_context(
			mission_id=mission.id,
			what_attempted=mission.objective[:500],
			what_worked="; ".join(merged_summaries) or "nothing merged",
			what_failed="; ".join(failed_summaries) or "no failures",
			recommended_next="continue",
		)

		# Verify it was persisted with the correct mission_id
		contexts = db.get_strategic_context(limit=5)
		assert len(contexts) == 1
		assert contexts[0].mission_id == "m-test-123"
		assert "Build feature X" in contexts[0].what_attempted
		assert "Added auth module" in contexts[0].what_worked
		assert "Linting failed" in contexts[0].what_failed

	def test_mission_has_id_not_mission_id(self) -> None:
		"""Verify Mission dataclass has 'id' field and not 'mission_id'."""
		m = Mission(id="test-id")
		assert m.id == "test-id"
		assert not hasattr(m, "mission_id")


class TestLogUnitEvent:
	def test_inserts_event_and_emits_to_stream(self, config: MissionConfig, db: Database) -> None:
		"""_log_unit_event should insert into DB and emit to event stream."""
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		mock_stream = MagicMock()
		ctrl._event_stream = mock_stream

		ctrl._log_unit_event(
			mission_id="m1",
			epoch_id="ep1",
			work_unit_id="wu1",
			event_type="dispatched",
			stream_details={"title": "Task", "files": "a.py"},
		)

		# Verify DB insert
		events = db.conn.execute(
			"SELECT * FROM unit_events WHERE work_unit_id = 'wu1' AND event_type = 'dispatched'"
		).fetchall()
		assert len(events) == 1

		# Verify event stream emit
		mock_stream.emit.assert_called_once()
		call_kwargs = mock_stream.emit.call_args
		assert call_kwargs[0][0] == "dispatched"
		assert call_kwargs[1]["mission_id"] == "m1"
		assert call_kwargs[1]["details"] == {"title": "Task", "files": "a.py"}

	def test_db_error_does_not_prevent_stream_emit(self, config: MissionConfig, db: Database) -> None:
		"""If DB insert fails, event stream emit should still happen."""
		ctrl = ContinuousController(config, db)
		mock_stream = MagicMock()
		ctrl._event_stream = mock_stream

		# Patch insert_unit_event to fail
		with patch.object(db, "insert_unit_event", side_effect=Exception("db error")):
			ctrl._log_unit_event(
				mission_id="m1",
				epoch_id="ep1",
				work_unit_id="wu1",
				event_type="merged",
			)

		# Stream emit should still be called
		mock_stream.emit.assert_called_once()

	def test_no_stream_does_not_raise(self, config: MissionConfig, db: Database) -> None:
		"""_log_unit_event should work without an event stream."""
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		# _event_stream is None by default

		ctrl._log_unit_event(
			mission_id="m1",
			epoch_id="ep1",
			work_unit_id="wu1",
			event_type="dispatched",
		)

		events = db.conn.execute(
			"SELECT * FROM unit_events WHERE work_unit_id = 'wu1' AND event_type = 'dispatched'"
		).fetchall()
		assert len(events) == 1

	def test_token_and_cost_fields_forwarded(self, config: MissionConfig, db: Database) -> None:
		"""Token counts and cost should be forwarded to both DB and stream."""
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		mock_stream = MagicMock()
		ctrl._event_stream = mock_stream

		ctrl._log_unit_event(
			mission_id="m1",
			epoch_id="ep1",
			work_unit_id="wu1",
			event_type="merged",
			input_tokens=1000,
			output_tokens=500,
			cost_usd=0.05,
		)

		# Verify DB has token counts
		row = db.conn.execute(
			"SELECT input_tokens, output_tokens FROM unit_events WHERE work_unit_id = 'wu1'"
		).fetchone()
		assert row is not None
		assert row[0] == 1000
		assert row[1] == 500

		# Verify stream emit includes tokens and cost
		call_kwargs = mock_stream.emit.call_args[1]
		assert call_kwargs["input_tokens"] == 1000
		assert call_kwargs["output_tokens"] == 500
		assert call_kwargs["cost_usd"] == 0.05


class TestInFlightCount:
	def test_initial_in_flight_is_zero(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl._in_flight_count == 0

	def test_adjust_modifies_existing_semaphore_in_place(self, config: MissionConfig, db: Database) -> None:
		"""DynamicSemaphore.adjust() modifies the same object, not replacing it."""
		db.insert_mission(Mission(id="m1", objective="test"))
		config.scheduler.parallel.num_workers = 2
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = DynamicSemaphore(2)
		original_sem = ctrl._semaphore

		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 4}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		assert ctrl._semaphore is original_sem  # same object
		assert ctrl._semaphore.capacity == 4
		_assert_semaphore_available(ctrl._semaphore, 4)

	def test_adjust_without_semaphore_skips_rebuild(self, config: MissionConfig, db: Database) -> None:
		"""When semaphore is None, adjust only updates config."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = None
		ctrl._in_flight_count = 2

		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 8}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		assert config.scheduler.parallel.num_workers == 8
		assert ctrl._semaphore is None


class TestDbErrorBudget:
	def test_initial_state(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl._degradation._db_error_count == 0
		assert ctrl._degradation.is_db_degraded is False

	def test_record_db_error_increments_count(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		ctrl._record_db_error()
		assert ctrl._degradation._db_error_count == 1
		assert ctrl._degradation.is_db_degraded is False

	def test_degraded_mode_after_threshold(self, config: MissionConfig, db: Database) -> None:
		"""After db_error_threshold consecutive errors, degraded mode activates."""
		ctrl = ContinuousController(config, db)
		threshold = config.degradation.db_error_threshold
		for _ in range(threshold):
			ctrl._record_db_error()

		assert ctrl._degradation._db_error_count == threshold
		assert ctrl._degradation.is_db_degraded is True

	def test_success_recovers_from_degraded(self, config: MissionConfig, db: Database) -> None:
		"""Consecutive DB successes recover from degraded mode."""
		ctrl = ContinuousController(config, db)
		threshold = config.degradation.db_error_threshold
		for _ in range(threshold):
			ctrl._record_db_error()
		assert ctrl._degradation.is_db_degraded is True

		recovery = config.degradation.recovery_success_threshold
		for _ in range(recovery):
			ctrl._record_db_success()
		assert ctrl._degradation.is_db_degraded is False

	def test_success_noop_when_no_errors(self, config: MissionConfig, db: Database) -> None:
		"""_record_db_success with no prior errors is a no-op."""
		ctrl = ContinuousController(config, db)
		ctrl._record_db_success()
		assert ctrl._degradation._db_error_count == 0
		assert ctrl._degradation.is_db_degraded is False

	def test_log_unit_event_skips_db_in_degraded_mode(self, config: MissionConfig, db: Database) -> None:
		"""In degraded mode, _log_unit_event skips DB insert but still emits to stream."""
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		# Force degraded via repeated errors
		threshold = config.degradation.db_error_threshold
		for _ in range(threshold):
			ctrl._record_db_error()
		mock_stream = MagicMock()
		ctrl._event_stream = mock_stream

		with patch.object(db, "insert_unit_event") as mock_insert:
			ctrl._log_unit_event(
				mission_id="m1",
				epoch_id="ep1",
				work_unit_id="wu1",
				event_type="dispatched",
			)

		# DB insert should NOT be called
		mock_insert.assert_not_called()
		# Stream emit should still fire
		mock_stream.emit.assert_called_once()

	def test_log_unit_event_tracks_errors(self, config: MissionConfig, db: Database) -> None:
		"""DB insert failure in _log_unit_event should increment error count."""
		ctrl = ContinuousController(config, db)
		assert ctrl._degradation._db_error_count == 0

		with patch.object(db, "insert_unit_event", side_effect=Exception("db error")):
			ctrl._log_unit_event(
				mission_id="m1",
				epoch_id="ep1",
				work_unit_id="wu1",
				event_type="merged",
			)

		assert ctrl._degradation._db_error_count == 1

	def test_log_unit_event_resets_on_success(self, config: MissionConfig, db: Database) -> None:
		"""Successful DB insert in _log_unit_event should decrement error count."""
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		# Set some prior errors via the degradation manager
		for _ in range(3):
			ctrl._record_db_error()

		ctrl._log_unit_event(
			mission_id="m1",
			epoch_id="ep1",
			work_unit_id="wu1",
			event_type="dispatched",
		)

		assert ctrl._degradation._db_error_count == 2  # decremented by 1

	def test_db_errors_on_result(self, config: MissionConfig, db: Database) -> None:
		"""ContinuousMissionResult should have db_errors field."""
		result = ContinuousMissionResult(mission_id="m1")
		assert result.db_errors == 0

		result.db_errors = 3
		assert result.db_errors == 3

	def test_degraded_mode_still_attempts_critical_writes(self, config: MissionConfig, db: Database) -> None:
		"""In degraded mode, critical writes (update_mission) are still attempted."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)

		# Enter degraded mode
		threshold = config.degradation.db_error_threshold
		for _ in range(threshold):
			ctrl._record_db_error()
		assert ctrl._degradation.is_db_degraded is True

		# _log_unit_event (non-critical) should skip
		with patch.object(db, "insert_unit_event") as mock_insert:
			ctrl._log_unit_event(
				mission_id="m1",
				epoch_id="ep1",
				work_unit_id="wu1",
				event_type="dispatched",
			)
		mock_insert.assert_not_called()

		# Verify recovery after consecutive successes
		mission = Mission(id="m1", objective="test")
		try:
			db.update_mission(mission)
			recovery = config.degradation.recovery_success_threshold
			for _ in range(recovery):
				ctrl._record_db_success()
		except Exception:
			ctrl._record_db_error()
		assert ctrl._degradation.is_db_degraded is False


class TestLayerByLayerDispatch:
	"""Verify _execute_batch dispatches units grouped by topological layer."""

	@pytest.mark.asyncio
	async def test_layer_barrier_enforces_ordering(self, config: MissionConfig, db: Database) -> None:
		"""Units in layer 1 only start after layer 0 tasks complete via gather barrier."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 10
		config.continuous.cooldown_between_units = 0

		config.scheduler.parallel.num_workers = 4
		ctrl = ContinuousController(config, db)

		events: list[str] = []

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
		) -> WorkerCompletion | None:
			events.append(f"start:{unit.id}")
			await asyncio.sleep(0.02)
			events.append(f"end:{unit.id}")
			unit.status = "completed"
			unit.commit_hash = "abc123"
			unit.branch_name = f"mc/unit-{unit.id}"
			unit.finished_at = "2025-01-01T00:00:00"
			completion = WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch)
			ctrl._semaphore.release()
			ctrl._in_flight_count = max(ctrl._in_flight_count - 1, 0)
			return completion

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				# A and B in layer 0 (no deps), C depends on A (layer 1)
				units = [
					WorkUnit(id="wu-a", plan_id=plan.id, title="A", max_attempts=1, depends_on=""),
					WorkUnit(id="wu-b", plan_id=plan.id, title="B", max_attempts=1, depends_on=""),
					WorkUnit(id="wu-c", plan_id=plan.id, title="C", max_attempts=1, depends_on="wu-a"),
				]
				return plan, units, epoch
			return plan, [], epoch

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(merged=True, rebase_ok=True, verification_passed=True),
		)

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()
			ctrl._notifier = None
			ctrl._heartbeat = None
			ctrl._event_stream = None

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=10.0)

		assert result.total_units_dispatched >= 3
		assert ctrl._total_merged == 3

		# Layer 0 units (A, B) must both end before layer 1 unit (C) starts
		a_end = events.index("end:wu-a")
		b_end = events.index("end:wu-b")
		c_start = events.index("start:wu-c")
		assert a_end < c_start, "A must finish before C starts"
		assert b_end < c_start, "B must finish before C starts"

	@pytest.mark.asyncio
	async def test_same_layer_units_run_concurrently(self, config: MissionConfig, db: Database) -> None:
		"""Units in the same layer are dispatched before any completes (concurrent)."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 10
		config.continuous.cooldown_between_units = 0

		config.scheduler.parallel.num_workers = 4
		ctrl = ContinuousController(config, db)

		events: list[str] = []

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
		) -> WorkerCompletion | None:
			events.append(f"start:{unit.id}")
			await asyncio.sleep(0.02)
			events.append(f"end:{unit.id}")
			unit.status = "completed"
			unit.commit_hash = "abc123"
			unit.branch_name = f"mc/unit-{unit.id}"
			unit.finished_at = "2025-01-01T00:00:00"
			completion = WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch)
			ctrl._semaphore.release()
			ctrl._in_flight_count = max(ctrl._in_flight_count - 1, 0)
			return completion

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				# All units in the same layer (no dependencies)
				units = [
					WorkUnit(id=f"wu-{i}", plan_id=plan.id, title=f"T{i}", max_attempts=1)
					for i in range(3)
				]
				return plan, units, epoch
			return plan, [], epoch

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(merged=True, rebase_ok=True, verification_passed=True),
		)

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()
			ctrl._notifier = None
			ctrl._heartbeat = None
			ctrl._event_stream = None

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=10.0)

		assert result.total_units_dispatched >= 3

		# All 3 units should start before any ends (concurrent dispatch in same layer)
		starts = [e for e in events if e.startswith("start:")]
		ends = [e for e in events if e.startswith("end:")]
		first_end_idx = events.index(ends[0])
		all_starts_before_first_end = all(events.index(s) < first_end_idx for s in starts)
		assert all_starts_before_first_end, "All same-layer units should start before any finishes"

class TestDynamicSemaphore:
	def test_basic_acquire_release(self) -> None:
		sem = DynamicSemaphore(2)
		assert sem.capacity == 2
		assert sem._value == 2

	def test_adjust_increase(self) -> None:
		sem = DynamicSemaphore(2)
		sem.adjust(4)
		assert sem.capacity == 4
		assert sem._value == 4  # 2 original + 2 released

	def test_adjust_decrease_with_debt(self) -> None:
		sem = DynamicSemaphore(4)
		sem.adjust(2)
		assert sem.capacity == 2
		assert sem._debt == 2
		# Two releases get absorbed by debt
		sem.release()
		sem.release()
		assert sem._debt == 0
		assert sem._value == 4  # unchanged since releases were swallowed
		# Third release goes through normally
		sem.release()
		assert sem._value == 5

	def test_adjust_increase_absorbs_debt(self) -> None:
		sem = DynamicSemaphore(4)
		sem.adjust(2)  # debt = 2
		sem.adjust(3)  # increase by 1, absorbs 1 debt
		assert sem._debt == 1
		assert sem.capacity == 3

	@pytest.mark.asyncio
	async def test_adjust_increases_concurrent_capacity(self) -> None:
		"""After adjusting from 2->4, 4 concurrent acquires should succeed."""
		sem = DynamicSemaphore(2)
		sem.adjust(4)

		acquired = 0
		for _ in range(4):
			await asyncio.wait_for(sem.acquire(), timeout=0.1)
			acquired += 1
		assert acquired == 4

		# 5th acquire should block (timeout)
		with pytest.raises(asyncio.TimeoutError):
			await asyncio.wait_for(sem.acquire(), timeout=0.05)


class TestAdjustSignalSemaphore:
	@pytest.mark.asyncio
	async def test_adjust_signal_increases_semaphore_capacity(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""After adjust signal from 2->4 workers, 4 concurrent dispatches are possible."""
		config.scheduler.parallel.num_workers = 2
		db.insert_mission(Mission(id="m1", objective="test"))

		ctrl = ContinuousController(config, db)
		ctrl._semaphore = DynamicSemaphore(2)

		# Verify initial capacity: 2 acquires succeed, 3rd blocks
		await asyncio.wait_for(ctrl._semaphore.acquire(), timeout=0.1)
		await asyncio.wait_for(ctrl._semaphore.acquire(), timeout=0.1)
		with pytest.raises(asyncio.TimeoutError):
			await asyncio.wait_for(ctrl._semaphore.acquire(), timeout=0.05)

		# Release the 2 slots back
		ctrl._semaphore.release()
		ctrl._semaphore.release()

		# Send adjust signal to increase to 4
		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 4}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		assert config.scheduler.parallel.num_workers == 4
		assert ctrl._semaphore.capacity == 4

		# Now 4 concurrent acquires should succeed
		acquired = 0
		for _ in range(4):
			await asyncio.wait_for(ctrl._semaphore.acquire(), timeout=0.1)
			acquired += 1
		assert acquired == 4

		# 5th should block
		with pytest.raises(asyncio.TimeoutError):
			await asyncio.wait_for(ctrl._semaphore.acquire(), timeout=0.05)

	@pytest.mark.asyncio
	async def test_adjust_signal_decreases_semaphore_capacity(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""After adjust signal from 4->2 workers, excess releases are absorbed."""
		config.scheduler.parallel.num_workers = 4
		db.insert_mission(Mission(id="m1", objective="test"))

		ctrl = ContinuousController(config, db)
		ctrl._semaphore = DynamicSemaphore(4)

		# Simulate 4 in-flight workers (acquire all 4)
		for _ in range(4):
			await asyncio.wait_for(ctrl._semaphore.acquire(), timeout=0.1)

		# Adjust down to 2
		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 2}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		assert ctrl._semaphore.capacity == 2

		# Release 4 slots (simulating 4 workers finishing)
		for _ in range(4):
			ctrl._semaphore.release()

		# Only 2 should be available (2 releases were absorbed by debt)
		assert ctrl._semaphore._value == 2


# -- Evaluator agent tests --


class TestEvaluatorAgent:
	"""Tests for _run_evaluator_agent()."""

	async def test_evaluator_passes(self, config: MissionConfig, db: Database) -> None:
		"""Evaluator returning passed=true produces correct result."""
		config.evaluator.enabled = True
		ctrl = ContinuousController(config, db)
		mission = Mission(id="m1", objective="Build API")

		eval_output = 'Tests pass. EVALUATION:{"passed": true, "evidence": ["all tests green"], "gaps": []}'
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(eval_output.encode(), b""))

		mod = "mission_control.continuous_controller"
		with patch(f"{mod}.claude_subprocess_env", return_value={}):
			with patch(f"{mod}.asyncio.create_subprocess_exec", return_value=mock_proc):
				with patch(f"{mod}.asyncio.wait_for", return_value=(eval_output.encode(), b"")):
					result = await ctrl._run_evaluator_agent(mission, "/tmp/workspace")

		assert result["passed"] is True
		assert "all tests green" in result["evidence"]
		assert result["gaps"] == []

	async def test_evaluator_fails(self, config: MissionConfig, db: Database) -> None:
		"""Evaluator returning passed=false produces correct result."""
		config.evaluator.enabled = True
		ctrl = ContinuousController(config, db)
		mission = Mission(id="m1", objective="Build API")

		eval_output = 'EVALUATION:{"passed": false, "evidence": [], "gaps": ["API endpoint missing"]}'
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(eval_output.encode(), b""))

		mod = "mission_control.continuous_controller"
		with patch(f"{mod}.claude_subprocess_env", return_value={}):
			with patch(f"{mod}.asyncio.create_subprocess_exec", return_value=mock_proc):
				with patch(f"{mod}.asyncio.wait_for", return_value=(eval_output.encode(), b"")):
					result = await ctrl._run_evaluator_agent(mission, "/tmp/workspace")

		assert result["passed"] is False
		assert "API endpoint missing" in result["gaps"]

	async def test_evaluator_timeout(self, config: MissionConfig, db: Database) -> None:
		"""Evaluator timeout returns failure result."""
		config.evaluator.enabled = True
		config.evaluator.timeout = 1
		ctrl = ContinuousController(config, db)
		mission = Mission(id="m1", objective="Build API")

		mock_proc = AsyncMock()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch("mission_control.continuous_controller.claude_subprocess_env", return_value={}):
			with patch("mission_control.continuous_controller.asyncio.create_subprocess_exec", return_value=mock_proc):
				with patch("mission_control.continuous_controller.asyncio.wait_for", side_effect=asyncio.TimeoutError):
					result = await ctrl._run_evaluator_agent(mission, "/tmp/workspace")

		assert result["passed"] is False
		assert "timed out" in result["gaps"][0].lower()

	async def test_evaluator_skipped_when_disabled(self, config: MissionConfig, db: Database) -> None:
		"""Evaluator config disabled by default -- verify default."""
		assert config.evaluator.enabled is False


# -- Reviewer skip tests --


class TestReviewerSkip:
	"""Tests for reviewer skip when acceptance criteria passed."""

	def test_review_default_model_is_haiku(self) -> None:
		"""Verify the default review model changed to haiku."""
		from mission_control.config import ReviewConfig
		rc = ReviewConfig()
		assert rc.model == "haiku"
		assert rc.budget_per_review_usd == 0.05
		assert rc.skip_when_criteria_passed is True

	def test_review_skip_when_criteria_passed_default(self) -> None:
		"""skip_when_criteria_passed defaults to True."""
		from mission_control.config import ReviewConfig
		rc = ReviewConfig()
		assert rc.skip_when_criteria_passed is True


class TestResumeWorkerForFixup:
	"""Tests for _resume_worker_for_fixup (session resume fixup)."""

	def _make_ctrl(self, config: MissionConfig, db: Database) -> tuple[ContinuousController, Database]:
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._green_branch = MagicMock()
		ctrl._semaphore = DynamicSemaphore(2)
		return ctrl, db

	@pytest.mark.asyncio
	async def test_fixup_succeeds_with_resume(self, config: MissionConfig, db: Database) -> None:
		"""Worker fixup uses --resume when session_id is set and re-merge succeeds."""
		ctrl, db = self._make_ctrl(config, db)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task", description="Do stuff",
			branch_name="mc/unit-wu1", session_id="test-session-123",
		)
		fail_result = UnitMergeResult(
			merged=False,
			failure_output="2 tests failed",
			failure_stage="pre_merge_verification",
		)
		success_result = UnitMergeResult(merged=True, verification_passed=True)

		ctrl._green_branch.merge_unit = AsyncMock(return_value=success_result)

		with patch("asyncio.create_subprocess_exec") as mock_exec:
			mock_proc = AsyncMock()
			mock_proc.communicate = AsyncMock(return_value=(b"Fixed", b""))
			mock_proc.kill = AsyncMock()
			mock_exec.return_value = mock_proc

			result = await ctrl._resume_worker_for_fixup(
				unit, "/tmp/ws", fail_result, config.continuous,
			)

			# Verify --resume flag is in the command
			call_args = mock_exec.call_args[0]
			assert "--resume" in call_args
			assert "test-session-123" in call_args
			# Should NOT contain the original task context
			prompt_arg = call_args[-1]
			assert "Do stuff" not in prompt_arg

		assert result is not None
		assert result.merged is True

	@pytest.mark.asyncio
	async def test_fixup_cold_fallback_without_session_id(self, config: MissionConfig, db: Database) -> None:
		"""Worker fixup falls back to cold -p when no session_id is set."""
		ctrl, db = self._make_ctrl(config, db)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task", description="Do stuff",
			branch_name="mc/unit-wu1", session_id="",
		)
		fail_result = UnitMergeResult(
			merged=False,
			failure_output="2 tests failed",
			failure_stage="pre_merge_verification",
		)
		success_result = UnitMergeResult(merged=True, verification_passed=True)

		ctrl._green_branch.merge_unit = AsyncMock(return_value=success_result)

		with patch("asyncio.create_subprocess_exec") as mock_exec:
			mock_proc = AsyncMock()
			mock_proc.communicate = AsyncMock(return_value=(b"Fixed", b""))
			mock_proc.kill = AsyncMock()
			mock_exec.return_value = mock_proc

			result = await ctrl._resume_worker_for_fixup(
				unit, "/tmp/ws", fail_result, config.continuous,
			)

			# Verify --resume is NOT in the command
			call_args = mock_exec.call_args[0]
			assert "--resume" not in call_args
			# Cold fixup should include the original task context
			prompt_arg = call_args[-1]
			assert "Do stuff" in prompt_arg

		assert result is not None
		assert result.merged is True

	@pytest.mark.asyncio
	async def test_fixup_fails_returns_none(self, config: MissionConfig, db: Database) -> None:
		"""Worker fixup cannot fix the code, returns None."""
		ctrl, db = self._make_ctrl(config, db)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task", description="Do stuff",
			branch_name="mc/unit-wu1", session_id="test-session-123",
		)
		fail_result = UnitMergeResult(
			merged=False,
			failure_output="2 tests failed",
			failure_stage="pre_merge_verification",
		)

		ctrl._green_branch.merge_unit = AsyncMock(
			return_value=UnitMergeResult(merged=False, failure_output="still failing"),
		)

		with patch("asyncio.create_subprocess_exec") as mock_exec:
			mock_proc = AsyncMock()
			mock_proc.communicate = AsyncMock(return_value=(b"Tried", b""))
			mock_proc.kill = AsyncMock()
			mock_exec.return_value = mock_proc

			result = await ctrl._resume_worker_for_fixup(
				unit, "/tmp/ws", fail_result, config.continuous,
				max_fixup_attempts=1,
			)

		assert result is None

	@pytest.mark.asyncio
	async def test_fixup_timeout_continues(self, config: MissionConfig, db: Database) -> None:
		"""Worker fixup that times out retries next attempt."""
		ctrl, db = self._make_ctrl(config, db)
		config.continuous.worker_fixup_timeout = 1

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task", description="Do stuff",
			branch_name="mc/unit-wu1", session_id="test-session-123",
		)
		fail_result = UnitMergeResult(
			merged=False,
			failure_output="lint errors",
			failure_stage="pre_merge_verification",
		)

		with patch("asyncio.create_subprocess_exec") as mock_exec:
			mock_proc = AsyncMock()
			mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
			mock_proc.kill = AsyncMock()
			mock_proc.wait = AsyncMock()
			mock_exec.return_value = mock_proc

			result = await ctrl._resume_worker_for_fixup(
				unit, "/tmp/ws", fail_result, config.continuous,
				max_fixup_attempts=1,
			)

		assert result is None

	@pytest.mark.asyncio
	async def test_fixup_triggers_on_verification_failure(self, config: MissionConfig, db: Database) -> None:
		"""Completion processor launches background fixup when merge fails at pre_merge_verification."""
		ctrl, db = self._make_ctrl(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		fail_result = UnitMergeResult(
			merged=False,
			failure_output="1 test failed",
			failure_stage="pre_merge_verification",
		)

		ctrl._green_branch.merge_unit = AsyncMock(return_value=fail_result)

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
			attempt=1, max_attempts=3,
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		bg_fixup = AsyncMock()
		with patch.object(ctrl, "_background_fixup", bg_fixup):
			await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		# Background task was created for this unit
		assert "wu1" in ctrl._active_fixups

	@pytest.mark.asyncio
	async def test_fixup_not_triggered_on_fetch_failure(self, config: MissionConfig, db: Database) -> None:
		"""Completion processor does NOT launch background fixup for non-fixable stages."""
		ctrl, db = self._make_ctrl(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		fail_result = UnitMergeResult(
			merged=False,
			failure_output="Failed to fetch unit branch",
			failure_stage="fetch",
		)
		ctrl._green_branch.merge_unit = AsyncMock(return_value=fail_result)

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
			attempt=1, max_attempts=3,
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		with patch.object(ctrl, "_retry_unit", new_callable=AsyncMock):
			await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		assert "wu1" not in ctrl._active_fixups
		assert ctrl._total_merged == 0

	@pytest.mark.asyncio
	async def test_fixup_triggered_on_merge_conflict(self, config: MissionConfig, db: Database) -> None:
		"""Completion processor launches background fixup for merge_conflict."""
		ctrl, db = self._make_ctrl(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		fail_result = UnitMergeResult(
			merged=False,
			failure_output="Merge conflict: CONFLICT in utils.py",
			failure_stage="merge_conflict",
		)

		ctrl._green_branch.merge_unit = AsyncMock(return_value=fail_result)

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
			attempt=1, max_attempts=3,
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		bg_fixup = AsyncMock()
		with patch.object(ctrl, "_background_fixup", bg_fixup):
			await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		# Background task was created for merge_conflict
		assert "wu1" in ctrl._active_fixups

	@pytest.mark.asyncio
	async def test_merge_conflict_resume_prompt_contains_rebase(self, config: MissionConfig, db: Database) -> None:
		"""Merge conflict fixup prompt instructs the worker to rebase, not just fix verification."""
		ctrl, db = self._make_ctrl(config, db)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task", description="Do stuff",
			branch_name="mc/unit-wu1", session_id="test-session-456",
		)
		fail_result = UnitMergeResult(
			merged=False,
			failure_output="CONFLICT (content): Merge conflict in utils.py",
			failure_stage="merge_conflict",
		)
		success_result = UnitMergeResult(merged=True, verification_passed=True)

		ctrl._green_branch.merge_unit = AsyncMock(return_value=success_result)

		with patch("asyncio.create_subprocess_exec") as mock_exec:
			mock_proc = AsyncMock()
			mock_proc.communicate = AsyncMock(return_value=(b"Resolved", b""))
			mock_proc.kill = AsyncMock()
			mock_exec.return_value = mock_proc

			result = await ctrl._resume_worker_for_fixup(
				unit, "/tmp/ws", fail_result, config.continuous,
			)

			call_args = mock_exec.call_args[0]
			assert "--resume" in call_args
			prompt_arg = call_args[-1]
			assert "rebase" in prompt_arg.lower()
			assert "conflict" in prompt_arg.lower()
			# Should NOT contain the original task (session has it)
			assert "Do stuff" not in prompt_arg

		assert result is not None
		assert result.merged is True


class TestAcceptMerge:
	"""Tests for _accept_merge: counter updates and fire-and-forget review."""

	def _make_ctrl(self, config: MissionConfig, db: Database) -> tuple[ContinuousController, Database]:
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._green_branch = MagicMock()
		ctrl._semaphore = DynamicSemaphore(2)
		return ctrl, db

	def test_accept_merge_increments_counters(self, config: MissionConfig, db: Database) -> None:
		"""Basic merge acceptance updates counters and file tracking."""
		ctrl, db = self._make_ctrl(config, db)
		config.review.enabled = False

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			files_hint="src/foo.py", input_tokens=100, output_tokens=50,
		)
		merge_result = UnitMergeResult(
			merged=True, verification_passed=True,
			changed_files=["src/foo.py"],
		)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		result = ContinuousMissionResult(mission_id="m1")

		ctrl._accept_merge(
			unit, merge_result, "/tmp/ws",
			Mission(id="m1"), epoch, result,
		)

		assert ctrl._total_merged == 1
		assert "wu1" in ctrl._completed_unit_ids
		assert "src/foo.py" in ctrl._merged_files

	@pytest.mark.asyncio
	async def test_fixup_merge_goes_through_accept(self, config: MissionConfig, db: Database) -> None:
		"""Background fixup calls _accept_merge when fixup succeeds."""
		ctrl, db = self._make_ctrl(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		fail_result = UnitMergeResult(
			merged=False,
			failure_output="2 tests failed",
			failure_stage="pre_merge_verification",
		)
		success_result = UnitMergeResult(merged=True, verification_passed=True)

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
			attempt=1, max_attempts=3,
		)
		db.insert_work_unit(unit)

		with patch.object(
			ctrl, "_resume_worker_for_fixup",
			new_callable=AsyncMock,
			return_value=success_result,
		), patch.object(
			ctrl, "_accept_merge",
		) as mock_accept:
			await ctrl._background_fixup(
				unit, "/tmp/ws", fail_result, config.continuous,
				Mission(id="m1"), epoch, result, None,
			)

		# Verify _accept_merge was called (fixup path uses unified acceptance)
		mock_accept.assert_called_once()

	@pytest.mark.asyncio
	async def test_background_fixup_prevents_duplicates(self, config: MissionConfig, db: Database) -> None:
		"""Completion processor skips fixup when one is already in progress for the same unit."""
		ctrl, db = self._make_ctrl(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		fail_result = UnitMergeResult(
			merged=False,
			failure_output="1 test failed",
			failure_stage="pre_merge_verification",
		)

		ctrl._green_branch.merge_unit = AsyncMock(return_value=fail_result)

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
			attempt=1, max_attempts=3,
		)
		db.insert_work_unit(unit)

		# Simulate an already-active fixup for this unit
		ctrl._active_fixups["wu1"] = asyncio.create_task(asyncio.sleep(999))

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		bg_fixup = AsyncMock()
		with patch.object(ctrl, "_background_fixup", bg_fixup):
			await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		# Should NOT have launched a second fixup
		bg_fixup.assert_not_awaited()

		# Cleanup
		ctrl._active_fixups["wu1"].cancel()
		try:
			await ctrl._active_fixups["wu1"]
		except asyncio.CancelledError:
			pass

	@pytest.mark.asyncio
	async def test_background_fixup_failure_schedules_retry(self, config: MissionConfig, db: Database) -> None:
		"""Background fixup schedules retry when fixup returns None (failure)."""
		ctrl, db = self._make_ctrl(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		fail_result = UnitMergeResult(
			merged=False,
			failure_output="2 tests failed",
			failure_stage="pre_merge_verification",
		)

		epoch = Epoch(id="ep1", mission_id="m1", number=1)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
			attempt=1, max_attempts=3,
		)

		with patch.object(
			ctrl, "_resume_worker_for_fixup",
			new_callable=AsyncMock,
			return_value=None,
		), patch.object(
			ctrl, "_schedule_retry",
		) as mock_retry:
			await ctrl._background_fixup(
				unit, "/tmp/ws", fail_result, config.continuous,
				Mission(id="m1"), epoch, result, None,
			)

		mock_retry.assert_called_once()


class TestExtractKnowledge:
	def test_extracts_from_handoff_discoveries(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Research auth",
			unit_type="research", files_hint="src/auth.py",
		)
		handoff = Handoff(
			work_unit_id="wu1", status="completed",
			summary="Analyzed auth module",
			discoveries=["JWT is used", "No refresh tokens"],
		)
		mission = Mission(id="m1", objective="test")
		ctrl._extract_knowledge(unit, handoff, mission)
		items = db.get_knowledge_for_mission("m1")
		assert len(items) == 1
		assert items[0].source_unit_type == "research"
		assert "JWT is used" in items[0].content
		assert items[0].scope == "src/auth.py"

	def test_skips_when_no_handoff(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task", unit_type="audit")
		mission = Mission(id="m1", objective="test")
		ctrl._extract_knowledge(unit, None, mission)
		assert db.get_knowledge_for_mission("m1") == []

	def test_skips_when_no_content(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task", unit_type="design")
		handoff = Handoff(work_unit_id="wu1", status="completed", summary="", discoveries=[])
		mission = Mission(id="m1", objective="test")
		ctrl._extract_knowledge(unit, handoff, mission)
		assert db.get_knowledge_for_mission("m1") == []

	def test_uses_summary_when_no_discoveries(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Audit security", unit_type="audit")
		handoff = Handoff(
			work_unit_id="wu1", status="completed",
			summary="Found 3 security issues",
			discoveries=[],
		)
		mission = Mission(id="m1", objective="test")
		ctrl._extract_knowledge(unit, handoff, mission)
		items = db.get_knowledge_for_mission("m1")
		assert len(items) == 1
		assert items[0].content == "Found 3 security issues"


class TestOrchestrationLoop:
	"""Tests for the sequential orchestration loop."""

	@pytest.mark.asyncio
	async def test_stops_on_empty_plan(self, config: MissionConfig, db: Database) -> None:
		"""Loop should stop and set objective_met when planner returns no units."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._planner = MagicMock()
		ctrl._planner.get_next_units = AsyncMock(
			return_value=(Plan(objective="test"), [], Epoch(mission_id="m1")),
		)
		ctrl._backend = MagicMock()
		ctrl._semaphore = DynamicSemaphore(2)

		mission = Mission(id="m1", objective="test")
		result = ContinuousMissionResult()
		await ctrl._orchestration_loop(mission, result)

		assert result.objective_met is True
		assert result.stopped_reason == "planner_completed"

	@pytest.mark.asyncio
	async def test_respects_stopping_conditions(self, config: MissionConfig, db: Database) -> None:
		"""Loop should stop when _should_stop returns a reason."""
		db.insert_mission(Mission(id="m1", objective="test"))
		config.continuous.max_wall_time_seconds = 1
		ctrl = ContinuousController(config, db)
		ctrl._start_time = time.monotonic() - 10
		ctrl._planner = MagicMock()
		ctrl._backend = MagicMock()
		ctrl._semaphore = DynamicSemaphore(2)

		mission = Mission(id="m1", objective="test")
		result = ContinuousMissionResult()
		await ctrl._orchestration_loop(mission, result)

		assert result.stopped_reason == "wall_time_exceeded"
		assert not ctrl.running


class TestExecuteBatch:
	@pytest.mark.asyncio
	async def test_returns_empty_for_empty_units(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		ctrl._backend = MagicMock()
		ctrl._semaphore = DynamicSemaphore(2)
		epoch = Epoch(id="ep1", mission_id="m1")
		mission = Mission(id="m1", objective="test")

		result = await ctrl._execute_batch([], epoch, mission)
		assert result == []

	@pytest.mark.asyncio
	async def test_collects_completions_from_units(self, config: MissionConfig, db: Database) -> None:
		"""Batch should collect WorkerCompletion objects returned by _execute_single_unit."""
		db.insert_mission(Mission(id="m1", objective="test"))
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		ctrl = ContinuousController(config, db)
		ctrl._backend = MagicMock()
		ctrl._semaphore = DynamicSemaphore(2)

		completion_obj = WorkerCompletion(
			unit=WorkUnit(id="wu1", plan_id="p1", title="Task"),
			handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		async def mock_execute(unit: WorkUnit, epoch: Epoch, mission: Mission) -> WorkerCompletion | None:
			return completion_obj

		units = [WorkUnit(id="wu1", plan_id="p1", title="Task")]

		with patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute):
			result = await ctrl._execute_batch(units, epoch, Mission(id="m1", objective="test"))

		assert len(result) == 1
		assert result[0] is completion_obj


class TestProcessBatch:
	@pytest.mark.asyncio
	async def test_processes_all_completions(self, config: MissionConfig, db: Database) -> None:
		"""_process_batch should call _process_single_completion for each item."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._green_branch = MagicMock()

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		db.insert_epoch(epoch)

		completions = [
			WorkerCompletion(
				unit=WorkUnit(id=f"wu{i}", plan_id="p1", title=f"Task {i}", status="completed"),
				handoff=None, workspace="/tmp/ws", epoch=epoch,
			)
			for i in range(3)
		]

		processed: list[str] = []

		async def mock_process(completion: WorkerCompletion, mission: Mission, result: ContinuousMissionResult) -> None:
			processed.append(completion.unit.id)

		mission = Mission(id="m1", objective="test")
		result = ContinuousMissionResult()

		with patch.object(ctrl, "_process_single_completion", side_effect=mock_process):
			await ctrl._process_batch(completions, mission, result)

		assert processed == ["wu0", "wu1", "wu2"]



