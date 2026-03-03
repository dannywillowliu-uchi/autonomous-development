"""Tests for automatic unit retry with failure diagnosis."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import ContinuousConfig, MissionConfig
from mission_control.continuous_controller import (
	ContinuousController,
	ContinuousMissionResult,
	WorkerCompletion,
)
from mission_control.db import Database
from mission_control.models import Epoch, Handoff, Mission, Plan, WorkUnit


def _make_controller(config: MissionConfig, db: Database) -> ContinuousController:
	"""Create a ContinuousController with mocked internals for unit testing."""
	ctrl = ContinuousController(config, db)
	ctrl._green_branch = MagicMock()
	ctrl._backend = AsyncMock()
	ctrl._semaphore = MagicMock()
	ctrl._semaphore.acquire = AsyncMock()
	ctrl._semaphore.release = MagicMock()
	ctrl._active_tasks = set()
	ctrl._in_flight_count = 0
	ctrl._total_failed = 0
	ctrl._total_merged = 0
	ctrl._total_dispatched = 0
	ctrl.running = True
	return ctrl


class TestFailedUnitTriggersRetry:
	"""Test that a failed unit triggers retry when attempt < max_retry_attempts."""

	def test_schedule_retry_creates_new_unit(self, config: MissionConfig, db: Database) -> None:
		"""When a unit fails and attempt < max_retry_attempts, a new WorkUnit is created."""
		config.continuous.max_retry_attempts = 2
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		original_unit = WorkUnit(
			id="orig1", plan_id="p1", title="Fix bug",
			description="Fix the login bug",
			files_hint="src/auth.py",
			acceptance_criteria="pytest tests/test_auth.py passes",
			status="failed", attempt=0, epoch_id="e1",
		)
		db.insert_work_unit(original_unit)

		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		ctrl = _make_controller(config, db)
		cont = config.continuous

		with patch("asyncio.create_task") as mock_create_task:
			mock_task = MagicMock(spec=asyncio.Task)
			mock_create_task.return_value = mock_task

			ctrl._schedule_retry(original_unit, epoch, mission, "SyntaxError in auth.py", cont)

		# Verify a new unit was inserted in the DB with parent_unit_id
		all_units = db.get_work_units_for_plan("p1")
		assert len(all_units) == 2  # original + retry

		retry_unit = [u for u in all_units if u.id != "orig1"][0]
		assert retry_unit.parent_unit_id == "orig1"
		assert retry_unit.title == "Fix bug"
		assert retry_unit.files_hint == "src/auth.py"
		assert retry_unit.acceptance_criteria == "pytest tests/test_auth.py passes"
		assert retry_unit.attempt == 1
		assert retry_unit.status == "pending"
		assert retry_unit.epoch_id == "e1"

	@pytest.mark.asyncio
	async def test_retry_gate_in_process_completion_triggers(self, config: MissionConfig, db: Database) -> None:
		"""_process_single_completion triggers retry for failed execution units."""
		config.continuous.max_retry_attempts = 2

		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="failed", attempt=0,
			output_summary="ImportError: no module named foo",
			epoch_id="e1",
		)
		db.insert_work_unit(unit)

		ctrl = _make_controller(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		with patch.object(ctrl, "_schedule_retry") as mock_retry, \
			patch.object(ctrl, "_handle_completion_common"):
			await ctrl._process_single_completion(completion, mission, result)
			mock_retry.assert_called_once()
			call_args = mock_retry.call_args
			assert call_args[0][0] is unit  # original unit
			assert "ImportError" in call_args[0][3]  # failure_reason


class TestRetryUsesDiagnoseFailure:
	"""Test that retry uses diagnose_failure hints in the new unit."""

	def test_diagnosis_included_in_retry_event(self, config: MissionConfig, db: Database) -> None:
		"""The retry event log includes diagnosis from diagnose_failure()."""
		config.continuous.max_retry_attempts = 2
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		unit = WorkUnit(
			id="diag1", plan_id="p1", title="Task",
			status="failed", attempt=0, epoch_id="e1",
		)
		db.insert_work_unit(unit)

		ctrl = _make_controller(config, db)
		cont = config.continuous

		with patch("asyncio.create_task") as mock_ct:
			mock_ct.return_value = MagicMock(spec=asyncio.Task)
			ctrl._schedule_retry(
				unit, epoch, mission,
				"ModuleNotFoundError: No module named 'nonexistent'",
				cont,
			)

		# Check that the retry event was logged with diagnosis
		events = db.get_unit_events_for_epoch("e1")
		retry_events = [e for e in events if e.event_type == "retry_queued"]
		assert len(retry_events) == 1
		details = json.loads(retry_events[0].details)
		assert "diagnosis" in details
		assert "Import error" in details["diagnosis"]


class TestRetryStopsAfterMaxAttempts:
	"""Test that retry stops after max_retry_attempts is reached."""

	@pytest.mark.asyncio
	async def test_no_retry_at_max_attempts(self, config: MissionConfig, db: Database) -> None:
		"""When attempt >= max_retry_attempts, no retry is scheduled and _total_failed increments."""
		config.continuous.max_retry_attempts = 2

		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		unit = WorkUnit(
			id="max1", plan_id="p1", title="Task",
			status="failed", attempt=2,  # already at max
			output_summary="persistent error",
			epoch_id="e1",
		)
		db.insert_work_unit(unit)

		ctrl = _make_controller(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		with patch.object(ctrl, "_schedule_retry") as mock_retry, \
			patch.object(ctrl, "_handle_completion_common"):
			await ctrl._process_single_completion(completion, mission, result)
			mock_retry.assert_not_called()
			assert ctrl._total_failed == 1

	@pytest.mark.asyncio
	async def test_retry_allowed_below_max(self, config: MissionConfig, db: Database) -> None:
		"""When attempt < max_retry_attempts, retry IS scheduled."""
		config.continuous.max_retry_attempts = 3

		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		unit = WorkUnit(
			id="below1", plan_id="p1", title="Task",
			status="failed", attempt=1,  # below max
			output_summary="some error",
			epoch_id="e1",
		)
		db.insert_work_unit(unit)

		ctrl = _make_controller(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		with patch.object(ctrl, "_schedule_retry") as mock_retry, \
			patch.object(ctrl, "_handle_completion_common"):
			await ctrl._process_single_completion(completion, mission, result)
			mock_retry.assert_called_once()


class TestSuccessfulRetryCountsAsMerged:
	"""Test that a successful retry unit flows through normal merge path."""

	@pytest.mark.asyncio
	async def test_retry_unit_completed_merges(self, config: MissionConfig, db: Database) -> None:
		"""A retry unit that completes goes through the merge path like any unit."""
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		retry_unit = WorkUnit(
			id="retry1", plan_id="p1", title="Fix bug",
			status="completed", attempt=1,
			commit_hash="abc123",
			parent_unit_id="orig1",
			epoch_id="e1",
			branch_name="mc/unit-retry1",
		)
		db.insert_work_unit(retry_unit)

		ctrl = _make_controller(config, db)
		ctrl._green_branch.merge_unit = AsyncMock(
			return_value=MagicMock(merged=True, commit_hash="abc123", failure_output="", failure_stage=""),
		)
		result = ContinuousMissionResult(mission_id="m1")

		handoff = Handoff(
			work_unit_id="retry1", status="completed",
			commits=["abc123"], summary="Fixed", files_changed=["src/auth.py"],
		)

		completion = WorkerCompletion(
			unit=retry_unit, handoff=handoff, workspace="/tmp/ws", epoch=epoch,
		)

		with patch.object(ctrl, "_accept_merge") as mock_accept, \
			patch.object(ctrl, "_handle_completion_common"):
			await ctrl._process_single_completion(completion, mission, result)
			mock_accept.assert_called_once()


class TestRetryInheritsParentContext:
	"""Test that retry units inherit files_hint and acceptance_criteria from parent."""

	def test_retry_inherits_files_hint_and_criteria(self, config: MissionConfig, db: Database) -> None:
		"""The retry WorkUnit inherits files_hint and acceptance_criteria from the parent."""
		config.continuous.max_retry_attempts = 2
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		parent = WorkUnit(
			id="parent1", plan_id="p1", title="Add feature X",
			description="Implement feature X",
			files_hint="src/feature.py, tests/test_feature.py",
			acceptance_criteria="pytest tests/test_feature.py -q",
			verification_hint="Check feature.py and test file",
			specialist="debugger",
			status="failed", attempt=0, epoch_id="e1",
		)
		db.insert_work_unit(parent)

		ctrl = _make_controller(config, db)

		with patch("asyncio.create_task") as mock_ct:
			mock_ct.return_value = MagicMock(spec=asyncio.Task)
			ctrl._schedule_retry(parent, epoch, mission, "test failure", config.continuous)

		all_units = db.get_work_units_for_plan("p1")
		retry_unit = [u for u in all_units if u.id != "parent1"][0]

		assert retry_unit.files_hint == "src/feature.py, tests/test_feature.py"
		assert retry_unit.acceptance_criteria == "pytest tests/test_feature.py -q"
		assert retry_unit.verification_hint == "Check feature.py and test file"
		assert retry_unit.specialist == "debugger"
		assert retry_unit.description == "Implement feature X"
		assert retry_unit.parent_unit_id == "parent1"


class TestConfigMaxRetryAttempts:
	"""Test the max_retry_attempts config field."""

	def test_default_value(self) -> None:
		"""ContinuousConfig.max_retry_attempts defaults to 2."""
		cfg = ContinuousConfig()
		assert cfg.max_retry_attempts == 2

	def test_custom_value(self) -> None:
		"""max_retry_attempts can be set to a custom value."""
		cfg = ContinuousConfig(max_retry_attempts=5)
		assert cfg.max_retry_attempts == 5

	@pytest.mark.asyncio
	async def test_zero_disables_retry(self, config: MissionConfig, db: Database) -> None:
		"""When max_retry_attempts=0, failed units go straight to _total_failed."""
		config.continuous.max_retry_attempts = 0

		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		unit = WorkUnit(
			id="nort1", plan_id="p1", title="Task",
			status="failed", attempt=0,
			output_summary="error",
			epoch_id="e1",
		)
		db.insert_work_unit(unit)

		ctrl = _make_controller(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		with patch.object(ctrl, "_schedule_retry") as mock_retry, \
			patch.object(ctrl, "_handle_completion_common"):
			await ctrl._process_single_completion(completion, mission, result)
			mock_retry.assert_not_called()
			assert ctrl._total_failed == 1
