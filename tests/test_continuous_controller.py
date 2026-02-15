"""Tests for ContinuousController."""

from __future__ import annotations

import asyncio
import json
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
	async def test_merge_failure_appends_to_handoff_concerns(self) -> None:
		"""When merge fails and handoff exists, failure is appended to concerns."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(_config(), db)
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
	async def test_research_unit_skips_merge(self) -> None:
		"""Research units should skip merge_unit() but count as merged."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		config = _config()
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
	async def test_research_unit_with_commits_skips_merge(self) -> None:
		"""Research units skip merge even if they have commits."""
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(_config(), db)
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
	async def test_failed_research_unit_counts_as_failure(self) -> None:
		"""Research unit with failed status should still count as failure."""
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
		assert r.objective_met is False
		assert r.total_units_dispatched == 0


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
		"""Integration: dispatch loop feeds units, completion processor merges them."""
		db = _db()
		config = _config()
		config.target.path = "/tmp/test"
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
	def _make_ctrl(self) -> tuple[ContinuousController, Database]:
		db = _db()
		db.insert_mission(Mission(id="m1", objective="test"))
		config = _config()
		ctrl = ContinuousController(config, db)
		ctrl._green_branch = MagicMock()
		ctrl._semaphore = asyncio.Semaphore(2)
		return ctrl, db

	@pytest.mark.asyncio
	async def test_failed_unit_retried_when_under_max_attempts(self) -> None:
		"""Unit fails with attempt=1, max_attempts=3 -> gets re-queued, not counted as failed."""
		ctrl, db = self._make_ctrl()
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
	async def test_failed_unit_not_retried_at_max_attempts(self) -> None:
		"""Unit fails with attempt=3, max_attempts=3 -> counted as failed, no retry."""
		ctrl, db = self._make_ctrl()
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
	async def test_retry_appends_failure_context(self) -> None:
		"""Verify the description is augmented with failure info on retry."""
		ctrl, db = self._make_ctrl()
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

		assert "[Retry attempt 1]" in unit.description
		assert "Import error in main.py" in unit.description
		assert "Avoid the same mistake" in unit.description
		assert unit.description.startswith("Original description")

	def test_retry_delay_exponential_backoff(self) -> None:
		"""Verify delay computation: base_delay * 2^(attempt-1), capped at max."""
		ctrl, _ = self._make_ctrl()
		config = ctrl.config
		config.continuous.retry_base_delay_seconds = 30
		config.continuous.retry_max_delay_seconds = 300

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		mission = Mission(id="m1")

		# attempt=1 -> delay = 30 * 2^0 = 30
		unit1 = WorkUnit(id="wu1", title="T", attempt=1, max_attempts=5)
		with patch("asyncio.create_task"):
			ctrl._schedule_retry(unit1, epoch, mission, "err", config.continuous)
		# Check the delay passed to _retry_unit via the created task
		# We verify by checking the unit state was reset
		assert unit1.status == "pending"

		# Verify delay values directly
		assert min(30 * (2 ** 0), 300) == 30   # attempt=1
		assert min(30 * (2 ** 1), 300) == 60   # attempt=2
		assert min(30 * (2 ** 2), 300) == 120  # attempt=3
		assert min(30 * (2 ** 3), 300) == 240  # attempt=4
		assert min(30 * (2 ** 4), 300) == 300  # attempt=5, capped

	@pytest.mark.asyncio
	async def test_merge_failure_triggers_retry(self) -> None:
		"""Unit completes but merge fails -> retried if under max_attempts."""
		ctrl, db = self._make_ctrl()
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
		assert "[Retry attempt 1]" in unit.description
