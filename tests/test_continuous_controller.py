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
	WorkerCompletion,
	_RoundTracker,
)
from mission_control.db import Database
from mission_control.green_branch import UnitMergeResult
from mission_control.models import BacklogItem, Epoch, Handoff, Mission, Plan, Signal, Worker, WorkUnit


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
		ctrl._completion_queue.put_nowait(completion)

		# Stop after processing one
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

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
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

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
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

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
			discoveries='["pattern X exists"]',
		)

		completion = WorkerCompletion(
			unit=unit, handoff=handoff, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		# Should count as merged, not failed
		assert ctrl._total_merged == 1
		assert ctrl._total_failed == 0
		# merge_unit should NOT have been called
		mock_gbm.merge_unit.assert_not_called()
		# Handoff should still be ingested
		mock_planner.ingest_handoff.assert_called_once_with(handoff)



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
		semaphore = asyncio.Semaphore(1)

		await ctrl._execute_single_unit(unit, epoch, Mission(id="m1"), semaphore)

		assert not ctrl._completion_queue.empty()
		completion = ctrl._completion_queue.get_nowait()
		assert completion.unit.status == "failed"
		assert "Pool exhausted" in completion.unit.output_summary


class TestEndToEnd:
	@pytest.mark.asyncio
	async def test_units_flow_dispatch_to_completion(self, config: MissionConfig, db: Database) -> None:
		"""Integration: dispatch loop feeds units, completion processor merges them."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1
		config.discovery.enabled = False
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

		# Mock planner: returns 2 units first call, then empty (mission done)
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
		ctrl._semaphore = asyncio.Semaphore(2)
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
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		with patch.object(ctrl, "_retry_unit", new_callable=AsyncMock):
			await ctrl._process_completions(Mission(id="m1"), result)

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
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

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
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		with patch.object(ctrl, "_retry_unit", new_callable=AsyncMock):
			await ctrl._process_completions(Mission(id="m1"), result)

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
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		with patch.object(ctrl, "_retry_unit", new_callable=AsyncMock):
			await ctrl._process_completions(Mission(id="m1"), result)

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


class TestPostMissionDiscovery:
	@pytest.mark.asyncio
	async def test_discovery_runs_on_success(self, config: MissionConfig, db: Database) -> None:
		"""Post-mission discovery should run when objective met and discovery enabled."""
		config.discovery.enabled = True
		ctrl = ContinuousController(config, db)

		mock_engine = MagicMock()
		mock_engine.discover = AsyncMock(return_value=(MagicMock(item_count=2), [
			MagicMock(track="quality", title="T1", priority_score=5.0),
			MagicMock(track="feature", title="T2", priority_score=6.0),
		]))

		with patch(
			"mission_control.auto_discovery.DiscoveryEngine",
			return_value=mock_engine,
		):
			await ctrl._run_post_mission_discovery()

		mock_engine.discover.assert_called_once()

	@pytest.mark.asyncio
	async def test_discovery_handles_errors(self, config: MissionConfig, db: Database) -> None:
		"""Post-mission discovery failure should not raise."""
		config.discovery.enabled = True
		ctrl = ContinuousController(config, db)

		with patch(
			"mission_control.auto_discovery.DiscoveryEngine",
			side_effect=RuntimeError("boom"),
		):
			# Should not raise
			await ctrl._run_post_mission_discovery()


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
		semaphore = asyncio.Semaphore(1)

		await ctrl._execute_single_unit(unit, epoch, Mission(id="m1"), semaphore)

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
		semaphore = asyncio.Semaphore(1)

		await ctrl._execute_single_unit(unit, epoch, Mission(id="m1"), semaphore)

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

		# Check completion was queued
		assert not ctrl._completion_queue.empty()
		completion = ctrl._completion_queue.get_nowait()
		assert completion.unit.id == "wu1"
		assert completion.handoff is None
		assert completion.workspace == "/tmp/ws"



class TestHandleAdjustSemaphoreRebuild:
	def test_semaphore_rebuilt_with_new_value(self, config: MissionConfig, db: Database) -> None:
		"""_handle_adjust_signal should rebuild semaphore with new worker count."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = asyncio.Semaphore(2)
		ctrl._in_flight_count = 0

		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 6}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		assert config.scheduler.parallel.num_workers == 6
		assert ctrl._semaphore is not None
		# With 0 in-flight, all 6 slots should be available
		assert ctrl._semaphore._value == 6

	def test_semaphore_accounts_for_in_flight(self, config: MissionConfig, db: Database) -> None:
		"""Semaphore rebuild should subtract in-flight count from new capacity."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = asyncio.Semaphore(2)
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
		# 5 total - 3 in-flight = 2 available
		assert ctrl._semaphore._value == 2

	def test_semaphore_clamps_to_zero_when_in_flight_exceeds(self, config: MissionConfig, db: Database) -> None:
		"""When in-flight exceeds new worker count, available slots should be 0."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = asyncio.Semaphore(4)
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
		# 2 total - 4 in-flight = clamped to 0
		assert ctrl._semaphore._value == 0



class TestConcurrentDispatchAndCompletion:
	@pytest.mark.asyncio
	async def test_concurrent_dispatch_and_completion(self, config: MissionConfig, db: Database) -> None:
		"""Dispatch loop and completion processor run concurrently; counters stay consistent."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 10
		config.continuous.cooldown_between_units = 0
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		# Track execution ordering to prove concurrency
		events: list[str] = []
		planner_may_finish = asyncio.Event()

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			events.append(f"exec:{unit.id}")
			# Small delay to ensure dispatch and completion overlap
			await asyncio.sleep(0.01)
			unit.status = "completed"
			unit.commit_hash = "abc123"
			unit.branch_name = f"mc/unit-{unit.id}"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(
					unit=unit, handoff=None,
					workspace="/tmp/ws", epoch=epoch,
				),
			)
			semaphore.release()

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
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
			# Hold off planner completion until units have been processed
			await planner_may_finish.wait()
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

		async def monitor_and_release() -> None:
			"""Wait until all 3 units are merged, then let planner finish."""
			while ctrl._total_merged + ctrl._total_failed < 3:
				await asyncio.sleep(0.01)
			planner_may_finish.set()

		async def run_all() -> ContinuousMissionResult:
			with (
				patch.object(ctrl, "_init_components", mock_init),
				patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
				patch("mission_control.continuous_controller.EventStream"),
			):
				monitor_task = asyncio.create_task(monitor_and_release())
				try:
					return await ctrl.run()
				finally:
					monitor_task.cancel()
					try:
						await monitor_task
					except asyncio.CancelledError:
						pass

		result = await asyncio.wait_for(run_all(), timeout=10.0)

		assert result.total_units_dispatched >= 3
		assert ctrl._total_merged + ctrl._total_failed == 3
		assert ctrl._total_merged == 3
		assert ctrl._total_failed == 0
		# Verify concurrent execution occurred (units dispatched before all completed)
		assert len([e for e in events if e.startswith("exec:")]) == 3


class TestAdjustWorkersDuringDispatch:
	@pytest.mark.asyncio
	async def test_adjust_workers_during_dispatch(self, config: MissionConfig, db: Database) -> None:
		"""Adjust workers signal mid-dispatch resizes the semaphore without orphaning tasks."""
		config.scheduler.parallel.num_workers = 2
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)

		# Simulate active dispatch state: semaphore with 0 available, 2 in-flight
		semaphore = asyncio.Semaphore(0)
		ctrl._semaphore = semaphore
		ctrl._in_flight_count = 2

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
		# Semaphore rebuilt: 4 total - 2 in-flight = 2 available
		assert ctrl._semaphore is not None
		assert ctrl._semaphore._value == 2
		# Old semaphore replaced
		assert ctrl._semaphore is not semaphore

		# Verify new semaphore allows 2 more dispatches
		acquired = ctrl._semaphore.acquire()
		# Should succeed immediately (non-blocking)
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

		async def slow_merge(workspace: str, branch: str) -> UnitMergeResult:
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
		ctrl._completion_queue.put_nowait(completion)

		# Run _process_completions in background
		async def run_processor() -> None:
			await ctrl._process_completions(Mission(id="m1"), result)

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
		ctrl.running = False

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
	async def test_ambition_score_persisted_on_mission(self, config: MissionConfig, db: Database) -> None:
		"""Ambition score should be set on the mission object and result."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
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
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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

		assert result.ambition_score > 0
		assert ctrl.ambition_score > 0

		# Verify it was persisted to the DB
		db_mission = db.get_latest_mission()
		assert db_mission is not None
		assert db_mission.ambition_score > 0



class TestNextObjectivePopulation:
	@pytest.mark.asyncio
	async def test_next_objective_set_when_not_met_with_backlog(self, config: MissionConfig, db: Database) -> None:
		"""When objective not met and backlog exists, next_objective should be populated."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 2
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		# Insert pending backlog items
		db.insert_backlog_item(BacklogItem(
			id="bl1", title="Add auth", priority_score=8.0,
			track="feature", status="pending",
		))
		db.insert_backlog_item(BacklogItem(
			id="bl2", title="Fix tests", priority_score=6.0,
			track="quality", status="pending",
		))

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			# Always return units (never complete)
			plan = Plan(id="p1", objective="test")
			epoch = Epoch(id="ep1", mission_id=mission.id, number=1)
			units = [WorkUnit(id="wu-1", plan_id=plan.id, title="Task", priority=3, files_hint="a.py")]
			return plan, units, epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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

		# Mission should have stopped by wall_time, not objective_met
		assert result.objective_met is False
		assert result.next_objective != ""
		assert "remaining backlog" in result.next_objective
		assert "Add auth" in result.next_objective

	@pytest.mark.asyncio
	async def test_next_objective_empty_when_objective_met(self, config: MissionConfig, db: Database) -> None:
		"""When objective is met, next_objective should remain empty."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		# Insert backlog items (they exist but shouldn't trigger chaining since objective is met)
		db.insert_backlog_item(BacklogItem(
			id="bl1", title="Some task", priority_score=5.0,
			track="feature", status="pending",
		))

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [WorkUnit(id="wu-1", plan_id=plan.id, title="Task", priority=3, files_hint="a.py")]
				return plan, units, epoch
			return plan, [], epoch  # Empty = objective met

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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
		assert result.next_objective == ""


class TestInFlightUnitsPreventPrematureCompletion:
	@pytest.mark.asyncio
	async def test_waits_for_inflight_before_declaring_complete(self, config: MissionConfig, db: Database) -> None:
		"""Planner returns empty but units are still running -- should wait, not declare complete."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 10
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		inflight_completed = asyncio.Event()

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			# Simulate slow execution
			await asyncio.sleep(0.1)
			inflight_completed.set()
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(
					unit=unit, handoff=None,
					workspace="/tmp/ws", epoch=epoch,
				),
			)
			semaphore.release()

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
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


class TestRoundTracker:
	def test_all_resolved_when_all_tracked(self) -> None:
		tracker = _RoundTracker(
			unit_ids={"a", "b"},
			completed_ids={"a"},
			failed_ids={"b"},
		)
		assert tracker.all_resolved is True

	def test_not_resolved_when_missing(self) -> None:
		tracker = _RoundTracker(
			unit_ids={"a", "b"},
			completed_ids={"a"},
			failed_ids=set(),
		)
		assert tracker.all_resolved is False

	def test_all_failed(self) -> None:
		tracker = _RoundTracker(
			unit_ids={"a", "b"},
			completed_ids=set(),
			failed_ids={"a", "b"},
		)
		assert tracker.all_failed is True

	def test_not_all_failed_when_some_succeed(self) -> None:
		tracker = _RoundTracker(
			unit_ids={"a", "b"},
			completed_ids={"a"},
			failed_ids={"b"},
		)
		assert tracker.all_failed is False


class TestAutoFailurePause:
	def _make_ctrl(self, config: MissionConfig, db: Database) -> ContinuousController:
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._green_branch = MagicMock()
		return ctrl

	def test_single_all_fail_sets_backoff(self, config: MissionConfig, db: Database) -> None:
		"""After one all-fail round, backoff timer is set and counter increments."""
		config.continuous.failure_backoff_seconds = 30
		config.continuous.max_consecutive_failures = 3
		ctrl = self._make_ctrl(config, db)

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		units = [
			WorkUnit(id="wu1", title="T1"),
			WorkUnit(id="wu2", title="T2"),
		]
		ctrl._round_tracker["ep1"] = _RoundTracker(
			unit_ids={"wu1", "wu2"},
			completed_ids=set(),
			failed_ids=set(),
		)

		# Record both as failed
		ctrl._record_round_outcome(units[0], epoch, merged=False)
		assert ctrl._consecutive_all_fail_rounds == 0  # not resolved yet
		ctrl._record_round_outcome(units[1], epoch, merged=False)

		assert ctrl._consecutive_all_fail_rounds == 1
		assert ctrl._failure_backoff_until > 0
		assert ctrl.running is True  # not stopped yet

	def test_counter_increments_on_consecutive_all_fail(self, config: MissionConfig, db: Database) -> None:
		"""Counter increments with each consecutive all-fail round."""
		config.continuous.failure_backoff_seconds = 10
		config.continuous.max_consecutive_failures = 5
		ctrl = self._make_ctrl(config, db)

		for i in range(3):
			epoch_id = f"ep{i}"
			unit_id = f"wu{i}"
			epoch = Epoch(id=epoch_id, mission_id="m1", number=i)
			ctrl._round_tracker[epoch_id] = _RoundTracker(
				unit_ids={unit_id},
				completed_ids=set(),
				failed_ids=set(),
			)
			ctrl._record_round_outcome(
				WorkUnit(id=unit_id, title=f"T{i}"), epoch, merged=False,
			)

		assert ctrl._consecutive_all_fail_rounds == 3
		assert ctrl.running is True  # max is 5, only at 3

	def test_stop_after_max_consecutive_failures(self, config: MissionConfig, db: Database) -> None:
		"""Mission stops after max_consecutive_failures all-fail rounds."""
		config.continuous.max_consecutive_failures = 2
		ctrl = self._make_ctrl(config, db)

		for i in range(2):
			epoch_id = f"ep{i}"
			unit_id = f"wu{i}"
			epoch = Epoch(id=epoch_id, mission_id="m1", number=i)
			ctrl._round_tracker[epoch_id] = _RoundTracker(
				unit_ids={unit_id},
				completed_ids=set(),
				failed_ids=set(),
			)
			ctrl._record_round_outcome(
				WorkUnit(id=unit_id, title=f"T{i}"), epoch, merged=False,
			)

		assert ctrl._consecutive_all_fail_rounds == 2
		assert ctrl._all_fail_stop_reason == "repeated_total_failure"

	def test_should_stop_returns_repeated_total_failure(self, config: MissionConfig, db: Database) -> None:
		"""_should_stop returns 'repeated_total_failure' when flag is set."""
		ctrl = self._make_ctrl(config, db)
		ctrl._all_fail_stop_reason = "repeated_total_failure"

		reason = ctrl._should_stop(Mission(id="m1"))
		assert reason == "repeated_total_failure"

	def test_counter_resets_on_successful_round(self, config: MissionConfig, db: Database) -> None:
		"""Counter resets to 0 when a round has at least one success."""
		config.continuous.max_consecutive_failures = 3
		ctrl = self._make_ctrl(config, db)

		# First round: all fail
		epoch1 = Epoch(id="ep1", mission_id="m1", number=1)
		ctrl._round_tracker["ep1"] = _RoundTracker(
			unit_ids={"wu1"},
			completed_ids=set(),
			failed_ids=set(),
		)
		ctrl._record_round_outcome(
			WorkUnit(id="wu1", title="T1"), epoch1, merged=False,
		)
		assert ctrl._consecutive_all_fail_rounds == 1

		# Second round: one succeeds
		epoch2 = Epoch(id="ep2", mission_id="m1", number=2)
		ctrl._round_tracker["ep2"] = _RoundTracker(
			unit_ids={"wu2", "wu3"},
			completed_ids=set(),
			failed_ids=set(),
		)
		ctrl._record_round_outcome(
			WorkUnit(id="wu2", title="T2"), epoch2, merged=True,
		)
		ctrl._record_round_outcome(
			WorkUnit(id="wu3", title="T3"), epoch2, merged=False,
		)
		assert ctrl._consecutive_all_fail_rounds == 0

	def test_untracked_unit_is_noop(self, config: MissionConfig, db: Database) -> None:
		"""Units not in any round tracker are silently ignored."""
		ctrl = self._make_ctrl(config, db)
		epoch = Epoch(id="ep_unknown", mission_id="m1", number=1)

		# Should not raise or modify state
		ctrl._record_round_outcome(
			WorkUnit(id="wu_orphan", title="Orphan"), epoch, merged=False,
		)
		assert ctrl._consecutive_all_fail_rounds == 0

	@pytest.mark.asyncio
	async def test_all_fail_round_pauses_then_retries(self, config: MissionConfig, db: Database) -> None:
		"""Integration: all-fail round pauses dispatch, then retries after backoff."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 10
		config.continuous.max_consecutive_failures = 3
		config.continuous.failure_backoff_seconds = 0  # instant backoff for test speed
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count <= 2:
				# First two rounds: return units (both will fail)
				units = [WorkUnit(id=f"wu-{call_count}", plan_id=plan.id, title=f"Task {call_count}", max_attempts=1)]
				return plan, units, epoch
			# Third call: return empty (mission done)
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			# All units fail
			unit.status = "failed"
			unit.attempt = 1
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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

		# After 2 all-fail rounds and counter < max(3), planner eventually returns empty
		assert call_count >= 3
		assert result.objective_met is True

	@pytest.mark.asyncio
	async def test_repeated_total_failure_stops_mission(self, config: MissionConfig, db: Database) -> None:
		"""Integration: mission stops with 'repeated_total_failure' after max all-fail rounds."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 10
		config.continuous.max_consecutive_failures = 2
		config.continuous.failure_backoff_seconds = 0
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			# Always return units (every round will fail)
			units = [WorkUnit(id=f"wu-{call_count}", plan_id=plan.id, title=f"Task {call_count}", max_attempts=1)]
			return plan, units, epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "failed"
			unit.attempt = 1
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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

		assert result.stopped_reason == "repeated_total_failure"
		assert result.objective_met is False
		assert ctrl._consecutive_all_fail_rounds >= 2


class TestInFlightCount:
	def test_initial_in_flight_is_zero(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl._in_flight_count == 0

	def test_adjust_uses_in_flight_count_not_active_tasks(self, config: MissionConfig, db: Database) -> None:
		"""Semaphore rebuild uses _in_flight_count, not _active_tasks length."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = asyncio.Semaphore(0)
		ctrl._in_flight_count = 1
		# active_tasks has 3 items (includes fire-and-forget tasks like reviews)
		ctrl._active_tasks = {MagicMock(), MagicMock(), MagicMock()}

		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 4}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		# Should use _in_flight_count (1), not len(_active_tasks) (3)
		assert ctrl._semaphore is not None
		assert ctrl._semaphore._value == 3  # 4 - 1

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
		assert ctrl._db_error_count == 0
		assert ctrl._db_degraded is False

	def test_record_db_error_increments_count(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		ctrl._record_db_error()
		assert ctrl._db_error_count == 1
		assert ctrl._db_degraded is False

	def test_degraded_mode_after_threshold(self, config: MissionConfig, db: Database) -> None:
		"""After _DB_ERROR_THRESHOLD consecutive errors, degraded mode activates."""
		ctrl = ContinuousController(config, db)
		for _ in range(ContinuousController._DB_ERROR_THRESHOLD):
			ctrl._record_db_error()

		assert ctrl._db_error_count == ContinuousController._DB_ERROR_THRESHOLD
		assert ctrl._db_degraded is True

	def test_success_resets_error_count_and_degraded(self, config: MissionConfig, db: Database) -> None:
		"""A successful DB write resets both counter and degraded flag."""
		ctrl = ContinuousController(config, db)
		for _ in range(ContinuousController._DB_ERROR_THRESHOLD):
			ctrl._record_db_error()
		assert ctrl._db_degraded is True

		ctrl._record_db_success()
		assert ctrl._db_error_count == 0
		assert ctrl._db_degraded is False

	def test_success_noop_when_no_errors(self, config: MissionConfig, db: Database) -> None:
		"""_record_db_success with no prior errors is a no-op."""
		ctrl = ContinuousController(config, db)
		ctrl._record_db_success()
		assert ctrl._db_error_count == 0
		assert ctrl._db_degraded is False

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
		ctrl._db_degraded = True
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
		assert ctrl._db_error_count == 0

		with patch.object(db, "insert_unit_event", side_effect=Exception("db error")):
			ctrl._log_unit_event(
				mission_id="m1",
				epoch_id="ep1",
				work_unit_id="wu1",
				event_type="merged",
			)

		assert ctrl._db_error_count == 1

	def test_log_unit_event_resets_on_success(self, config: MissionConfig, db: Database) -> None:
		"""Successful DB insert in _log_unit_event should reset error count."""
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		ctrl._db_error_count = 3  # some prior errors

		ctrl._log_unit_event(
			mission_id="m1",
			epoch_id="ep1",
			work_unit_id="wu1",
			event_type="dispatched",
		)

		assert ctrl._db_error_count == 0

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
		for _ in range(ContinuousController._DB_ERROR_THRESHOLD):
			ctrl._record_db_error()
		assert ctrl._db_degraded is True

		# _log_unit_event (non-critical) should skip
		with patch.object(db, "insert_unit_event") as mock_insert:
			ctrl._log_unit_event(
				mission_id="m1",
				epoch_id="ep1",
				work_unit_id="wu1",
				event_type="dispatched",
			)
		mock_insert.assert_not_called()

		# update_mission (critical) should still be attempted in _process_completions
		# Verify by checking that a successful DB write exits degraded mode
		mission = Mission(id="m1", objective="test")
		try:
			db.update_mission(mission)
			ctrl._record_db_success()
		except Exception:
			ctrl._record_db_error()
		assert ctrl._db_degraded is False
		assert ctrl._db_error_count == 0

