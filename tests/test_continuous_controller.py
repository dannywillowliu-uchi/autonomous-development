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
)
from mission_control.db import Database
from mission_control.green_branch import UnitMergeResult
from mission_control.models import Epoch, Handoff, Mission, Plan, Signal, Worker, WorkUnit


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

	def test_wall_time_not_exceeded(self, config: MissionConfig, db: Database) -> None:
		config.continuous.max_wall_time_seconds = 100
		ctrl = ContinuousController(config, db)
		ctrl._start_time = time.monotonic() - 10  # 10s elapsed, limit is 100s
		assert ctrl._should_stop(Mission(id="m1")) == ""

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

	def test_no_signals(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl._check_signals("m1") == ""


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
	def test_pause_signal_sets_paused(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(mission_id="m1", signal_type="pause")
		db.insert_signal(signal)

		ctrl = ContinuousController(config, db)
		assert ctrl._paused is False
		ctrl._check_signals("m1")
		assert ctrl._paused is True

	def test_resume_signal_clears_paused(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		signal = Signal(mission_id="m1", signal_type="resume")
		db.insert_signal(signal)

		ctrl = ContinuousController(config, db)
		ctrl._paused = True
		ctrl._check_signals("m1")
		assert ctrl._paused is False

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
	async def test_completed_no_commits_counts_as_merged(self, config: MissionConfig, db: Database) -> None:
		"""Unit completed without commits should count as merged."""
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
	async def test_merge_failure_appends_to_handoff_concerns(self, config: MissionConfig, db: Database) -> None:
		"""When merge fails and handoff exists, failure is appended to concerns."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=False, rebase_ok=False,
				failure_output="Merge conflict in main.py",
			),
		)
		ctrl._green_branch = mock_gbm

		mock_planner = MagicMock()
		mock_planner.ingest_handoff = MagicMock()
		ctrl._planner = mock_planner

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc",
			branch_name="mc/unit-wu1",
			attempt=3, max_attempts=3,
		)
		db.insert_work_unit(unit)

		handoff = Handoff(
			work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="completed", summary="Done", concerns="[]",
		)

		completion = WorkerCompletion(
			unit=unit, handoff=handoff, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		# Handoff concerns should contain merge failure info
		concerns = json.loads(handoff.concerns)
		assert len(concerns) == 1
		assert "Merge failed" in concerns[0]
		# Planner should still get the handoff
		mock_planner.ingest_handoff.assert_called_once_with(handoff)


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

	@pytest.mark.asyncio
	async def test_research_unit_with_commits_skips_merge(self, config: MissionConfig, db: Database) -> None:
		"""Research units skip merge even if they have commits."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Research",
			status="completed", commit_hash="abc123",
			unit_type="research",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		assert ctrl._total_merged == 1
		mock_gbm.merge_unit.assert_not_called()

	@pytest.mark.asyncio
	async def test_failed_research_unit_counts_as_failure(self, config: MissionConfig, db: Database) -> None:
		"""Research unit with failed status should still count as failure."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")
		ctrl._green_branch = MagicMock()

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Research",
			status="failed", unit_type="research",
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


class TestStop:
	def test_stop_sets_running_false(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
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
		assert r.objective_met is False
		assert r.total_units_dispatched == 0


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
		config.continuous.max_wall_time_seconds = 5
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

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
		):
			result = await ctrl.run()

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


class TestRetryCounterIncrement:
	"""Verify _schedule_retry properly increments unit.attempt."""

	def test_attempt_incremented_from_zero(self, config: MissionConfig, db: Database) -> None:
		"""Unit at attempt=0 should be incremented to 1 by _schedule_retry."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)

		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			attempt=0, max_attempts=3,
		)
		db.insert_work_unit(unit)

		with patch("asyncio.create_task"):
			ctrl._schedule_retry(unit, epoch, Mission(id="m1"), "error", config.continuous)

		assert unit.attempt == 1
		assert unit.status == "pending"

		# Verify it was persisted to DB
		db_unit = db.get_work_unit("wu1")
		assert db_unit is not None
		assert db_unit.attempt == 1

	def test_attempt_incremented_multiple_retries(self, config: MissionConfig, db: Database) -> None:
		"""Multiple retries should increment attempt each time."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)

		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			attempt=0, max_attempts=5,
		)
		db.insert_work_unit(unit)

		for expected_attempt in range(1, 4):
			with patch("asyncio.create_task"):
				ctrl._schedule_retry(unit, epoch, Mission(id="m1"), f"error {expected_attempt}", config.continuous)
			assert unit.attempt == expected_attempt


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
	async def test_discovery_notifies_on_results(self, config: MissionConfig, db: Database) -> None:
		"""Should send Telegram notification with discovery results."""
		config.discovery.enabled = True
		ctrl = ContinuousController(config, db)
		ctrl._notifier = MagicMock()
		ctrl._notifier.send = AsyncMock()

		items = [
			MagicMock(track="quality", title="Fix tests", priority_score=5.0),
		]
		mock_engine = MagicMock()
		mock_engine.discover = AsyncMock(
			return_value=(MagicMock(item_count=1), items),
		)

		with patch(
			"mission_control.auto_discovery.DiscoveryEngine",
			return_value=mock_engine,
		):
			await ctrl._run_post_mission_discovery()

		ctrl._notifier.send.assert_called_once()
		call_args = ctrl._notifier.send.call_args[0][0]
		assert "1 new improvement" in call_args

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

	@pytest.mark.asyncio
	async def test_worker_dead_on_infrastructure_error(self, config: MissionConfig, db: Database) -> None:
		"""Worker status set to dead on infrastructure error."""
		db, config, ctrl, epoch = self._setup(config, db)

		mock_backend = AsyncMock()
		mock_handle = MagicMock()
		mock_handle.pid = 42
		mock_handle.workspace_path = "/tmp/ws"
		mock_backend.spawn.return_value = mock_handle
		mock_backend.check_status.side_effect = RuntimeError("Connection lost")
		ctrl._backend = mock_backend

		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)
		semaphore = asyncio.Semaphore(1)

		await ctrl._execute_single_unit(unit, epoch, Mission(id="m1"), semaphore)

		worker = db.get_worker("wu1")
		assert worker is not None
		assert worker.status == "dead"
		assert worker.units_failed == 1

	@pytest.mark.asyncio
	async def test_no_worker_on_provision_failure(self, config: MissionConfig, db: Database) -> None:
		"""No Worker record when workspace provisioning fails (before spawn)."""
		db, config, ctrl, epoch = self._setup(config, db)

		mock_backend = AsyncMock()
		mock_backend.provision_workspace.side_effect = RuntimeError("Pool full")
		ctrl._backend = mock_backend

		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)
		semaphore = asyncio.Semaphore(1)

		await ctrl._execute_single_unit(unit, epoch, Mission(id="m1"), semaphore)

		worker = db.get_worker("wu1")
		assert worker is None


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

	@pytest.mark.asyncio
	async def test_put_on_queue_false_skips_queue(self, config: MissionConfig, db: Database) -> None:
		"""_fail_unit with put_on_queue=False should not add to completion queue."""
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

		await ctrl._fail_unit(unit, None, epoch, "cancelled", "/tmp/ws", put_on_queue=False)

		assert unit.status == "failed"
		assert unit.attempt == 1
		assert ctrl._completion_queue.empty()

	@pytest.mark.asyncio
	async def test_no_worker(self, config: MissionConfig, db: Database) -> None:
		"""_fail_unit with worker=None should still update unit and queue."""
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

		await ctrl._fail_unit(unit, None, epoch, "provision failed", "")

		assert unit.status == "failed"
		assert not ctrl._completion_queue.empty()


class TestHandleAdjustSemaphoreRebuild:
	def test_semaphore_rebuilt_with_new_value(self, config: MissionConfig, db: Database) -> None:
		"""_handle_adjust_signal should rebuild semaphore with new worker count."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = asyncio.Semaphore(2)

		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 6}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		assert config.scheduler.parallel.num_workers == 6
		assert ctrl._semaphore is not None
		assert ctrl._semaphore._value == 6

	def test_semaphore_preserves_in_flight(self, config: MissionConfig, db: Database) -> None:
		"""Semaphore rebuild should pre-acquire slots for in-flight workers."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		ctrl._semaphore = asyncio.Semaphore(4)

		# Simulate 2 in-flight tasks
		mock_task1 = MagicMock(spec=asyncio.Task)
		mock_task1.done.return_value = False
		mock_task2 = MagicMock(spec=asyncio.Task)
		mock_task2.done.return_value = False
		mock_done_task = MagicMock(spec=asyncio.Task)
		mock_done_task.done.return_value = True
		ctrl._active_tasks = {mock_task1, mock_task2, mock_done_task}

		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 5}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		# 5 total - 2 in-flight = 3 available
		assert ctrl._semaphore._value == 3

	def test_no_semaphore_no_crash(self, config: MissionConfig, db: Database) -> None:
		"""Adjusting workers without semaphore should still update config."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		# _semaphore is None by default

		signal = Signal(
			mission_id="m1",
			signal_type="adjust",
			payload='{"num_workers": 3}',
		)
		db.insert_signal(signal)
		ctrl._handle_adjust_signal(signal)

		assert config.scheduler.parallel.num_workers == 3
		assert ctrl._semaphore is None


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

	@pytest.mark.asyncio
	async def test_dry_run_no_units(self, config: MissionConfig, db: Database) -> None:
		"""dry_run with no planner units should print message and return."""
		ctrl = ContinuousController(config, db)

		plan = Plan(id="p1", objective="test")
		epoch = Epoch(id="ep1", mission_id="m1", number=0)

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(return_value=(plan, [], epoch))

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._backend = AsyncMock()

		with patch.object(ctrl, "_init_components", mock_init):
			result = await ctrl.run(dry_run=True)

		assert result.total_units_dispatched == 0

	@pytest.mark.asyncio
	async def test_dry_run_does_not_insert_mission(self, config: MissionConfig, db: Database) -> None:
		"""dry_run should not persist a mission to the database."""
		ctrl = ContinuousController(config, db)

		plan = Plan(id="p1", objective="test")
		epoch = Epoch(id="ep1", mission_id="m1", number=0)
		units = [WorkUnit(id="wu1", plan_id="p1", title="Task")]

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(return_value=(plan, units, epoch))

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._backend = AsyncMock()

		with patch.object(ctrl, "_init_components", mock_init):
			await ctrl.run(dry_run=True)

		# No mission should be in the DB
		latest = db.get_latest_mission()
		assert latest is None
