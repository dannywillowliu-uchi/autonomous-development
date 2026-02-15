"""Integration tests for parallel execution with real DB and mocked subprocesses."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mission_control.config import (
	GreenBranchConfig,
	MissionConfig,
	ParallelConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.coordinator import Coordinator
from mission_control.db import Database
from mission_control.green_branch import GreenBranchManager
from mission_control.merge_queue import MergeQueue
from mission_control.models import Epoch, MergeRequest, Mission, Plan, Snapshot, UnitEvent, WorkUnit
from mission_control.planner import _parse_plan_output
from mission_control.worker import render_worker_prompt


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


@pytest.fixture()
def config() -> MissionConfig:
	cfg = MissionConfig()
	cfg.target = TargetConfig(
		name="test-proj",
		path="/tmp/test-proj",
		branch="main",
		objective="Fix 3 failing tests and 2 lint errors",
		verification=VerificationConfig(command="pytest -q && ruff check src/"),
	)
	cfg.scheduler.parallel = ParallelConfig(
		num_workers=2,
		heartbeat_timeout=600,
		max_rebase_attempts=3,
	)
	cfg.scheduler.session_timeout = 300
	cfg.scheduler.budget.max_per_session_usd = 2.0
	return cfg


class TestPlannerToWorkUnits:
	"""Test planner output -> work units -> DB lifecycle."""

	def test_full_plan_parse_and_insert(self, db: Database) -> None:
		"""Simulate planner producing 3 units and inserting them into DB."""
		planner_output = """Here are the work units:
```json
[
  {
    "title": "Fix test_calculator_add",
    "description": "The add function returns wrong value. Fix in src/calc.py",
    "files_hint": "src/calc.py,tests/test_calc.py",
    "verification_hint": "Run pytest tests/test_calc.py::test_add",
    "priority": 1,
    "depends_on_indices": []
  },
  {
    "title": "Fix test_calculator_divide",
    "description": "Division by zero not handled. Add guard in src/calc.py",
    "files_hint": "src/calc.py",
    "verification_hint": "Run pytest tests/test_calc.py::test_divide",
    "priority": 1,
    "depends_on_indices": []
  },
  {
    "title": "Fix ruff lint errors",
    "description": "2 unused imports in src/utils.py",
    "files_hint": "src/utils.py",
    "verification_hint": "Run ruff check src/",
    "priority": 2,
    "depends_on_indices": []
  }
]
```
"""
		plan = Plan(id="p1", objective="Fix everything")
		db.insert_plan(plan)

		units = _parse_plan_output(planner_output, plan.id)
		assert len(units) == 3

		for u in units:
			db.insert_work_unit(u)

		# Verify all are in DB
		db_units = db.get_work_units_for_plan("p1")
		assert len(db_units) == 3
		assert db_units[0].title == "Fix test_calculator_add"
		assert db_units[0].priority == 1

	def test_plan_with_dependencies(self, db: Database) -> None:
		"""Units with depends_on_indices are resolved correctly."""
		planner_output = json.dumps([
			{"title": "Setup infra", "description": "Create base", "priority": 1, "depends_on_indices": []},
			{"title": "Add feature", "description": "Needs infra", "priority": 2, "depends_on_indices": [0]},
		])
		plan = Plan(id="p2", objective="Build feature")
		db.insert_plan(plan)

		units = _parse_plan_output(planner_output, plan.id)
		assert len(units) == 2

		for u in units:
			db.insert_work_unit(u)

		# Second unit depends on first
		db_units = db.get_work_units_for_plan("p2")
		assert db_units[1].depends_on == db_units[0].id

		# Can claim first but not second
		w_id = "worker-1"
		claimed = db.claim_work_unit(w_id)
		assert claimed is not None
		assert claimed.title == "Setup infra"

		# Second is blocked
		claimed2 = db.claim_work_unit("worker-2")
		assert claimed2 is None

		# Complete first, now second is claimable
		claimed.status = "completed"
		db.update_work_unit(claimed)
		claimed2 = db.claim_work_unit("worker-2")
		assert claimed2 is not None
		assert claimed2.title == "Add feature"


class TestWorkerLifecycle:
	"""Test worker claiming -> executing -> reporting flow."""

	def test_claim_execute_complete_flow(self, db: Database, config: MissionConfig) -> None:
		"""Worker claims a unit, completes it, and a merge request is created."""
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		wu = WorkUnit(id="wu1", plan_id="p1", title="Fix tests", description="Fix them")
		db.insert_work_unit(wu)

		from mission_control.models import Worker
		w = Worker(id="w1", workspace_path="/tmp/w1")
		db.insert_worker(w)

		# Claim
		claimed = db.claim_work_unit("w1")
		assert claimed is not None
		assert claimed.status == "claimed"

		# Simulate completion
		claimed.status = "completed"
		claimed.commit_hash = "abc123"
		claimed.output_summary = "Fixed 2 tests"
		db.update_work_unit(claimed)

		# Create merge request
		mr = MergeRequest(
			work_unit_id="wu1",
			worker_id="w1",
			branch_name="mc/unit-wu1",
			commit_hash="abc123",
			position=db.get_next_merge_position(),
		)
		db.insert_merge_request(mr)

		# Verify state
		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "completed"
		assert result.commit_hash == "abc123"

		pending_mr = db.get_next_merge_request()
		assert pending_mr is not None
		assert pending_mr.work_unit_id == "wu1"

	def test_two_workers_claim_different_units(self, db: Database) -> None:
		"""Two workers claiming simultaneously get different units."""
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		for i in range(3):
			db.insert_work_unit(WorkUnit(id=f"wu{i}", plan_id="p1", title=f"Task {i}"))

		c1 = db.claim_work_unit("w1")
		c2 = db.claim_work_unit("w2")

		assert c1 is not None
		assert c2 is not None
		assert c1.id != c2.id

	def test_worker_failure_increments_attempt(self, db: Database, config: MissionConfig) -> None:
		"""Failed unit has attempt incremented and can be retried."""
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		wu = WorkUnit(id="wu1", plan_id="p1", title="Flaky task", max_attempts=3)
		db.insert_work_unit(wu)

		# First attempt fails
		claimed = db.claim_work_unit("w1")
		assert claimed is not None
		claimed.status = "failed"
		claimed.attempt = 1
		claimed.output_summary = "Timed out"
		db.update_work_unit(claimed)

		# Release for retry
		claimed.status = "pending"
		claimed.worker_id = None
		claimed.claimed_at = None
		claimed.heartbeat_at = None
		db.update_work_unit(claimed)

		# Second attempt
		claimed2 = db.claim_work_unit("w2")
		assert claimed2 is not None
		assert claimed2.id == "wu1"
		assert claimed2.attempt == 1  # was set before release


class TestMergeQueueFlow:
	"""Test merge queue verdict-based merge/reject flow."""

	def test_merge_request_lifecycle(self, db: Database) -> None:
		"""MR goes through pending -> verifying -> merged."""
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		wu = WorkUnit(id="wu1", plan_id="p1", title="Fix", status="completed", commit_hash="abc")
		db.insert_work_unit(wu)

		mr = MergeRequest(
			id="mr1",
			work_unit_id="wu1",
			worker_id="w1",
			branch_name="mc/unit-wu1",
			commit_hash="abc",
			position=1,
		)
		db.insert_merge_request(mr)

		# Get next MR
		pending = db.get_next_merge_request()
		assert pending is not None
		assert pending.id == "mr1"

		# Mark as verifying
		pending.status = "verifying"
		db.update_merge_request(pending)

		# Mark as merged
		pending.status = "merged"
		from mission_control.models import _now_iso
		pending.merged_at = _now_iso()
		db.update_merge_request(pending)

		# No more pending
		assert db.get_next_merge_request() is None

	async def test_rejected_mr_releases_unit(self, db: Database) -> None:
		"""Rejected merge request releases work unit for retry."""
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		wu = WorkUnit(id="wu1", plan_id="p1", title="Fix", status="completed", commit_hash="abc")
		db.insert_work_unit(wu)

		mr = MergeRequest(
			work_unit_id="wu1",
			worker_id="w1",
			branch_name="mc/unit-wu1",
			commit_hash="abc",
			position=1,
		)
		db.insert_merge_request(mr)

		# Simulate MergeQueue._release_unit_for_retry
		mq = MergeQueue.__new__(MergeQueue)
		mq.db = db
		mq.config = MissionConfig()
		await mq._release_unit_for_retry(mr)

		# Unit should be back to pending
		unit = db.get_work_unit("wu1")
		assert unit is not None
		assert unit.status == "pending"
		assert unit.worker_id is None
		assert unit.attempt == 1

	def test_merge_queue_ordering(self, db: Database) -> None:
		"""Merge requests are processed in position order."""
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		for i in range(3):
			wu = WorkUnit(id=f"wu{i}", plan_id="p1", title=f"Task {i}", status="completed")
			db.insert_work_unit(wu)
			mr = MergeRequest(
				id=f"mr{i}",
				work_unit_id=f"wu{i}",
				worker_id="w1",
				branch_name=f"mc/unit-wu{i}",
				position=i + 1,
			)
			db.insert_merge_request(mr)

		# Should get position 1 first
		first = db.get_next_merge_request()
		assert first is not None
		assert first.position == 1


class TestCoordinatorIntegration:
	"""Integration tests for the full coordinator flow."""

	async def test_coordinator_full_flow_with_mocks(
		self, db: Database, config: MissionConfig,
	) -> None:
		"""Full flow: snapshot -> plan -> workers -> merge -> report."""
		# Setup: pre-populate plan with 2 units (skip actual planner call)
		plan = Plan(id="p1", objective="Fix tests", status="active", total_units=2)
		db.insert_plan(plan)
		wu1 = WorkUnit(id="wu1", plan_id="p1", title="Fix test A", status="completed", commit_hash="aaa")
		wu2 = WorkUnit(id="wu2", plan_id="p1", title="Fix test B", status="completed", commit_hash="bbb")
		db.insert_work_unit(wu1)
		db.insert_work_unit(wu2)

		# Create merge requests for both
		mr1 = MergeRequest(
			work_unit_id="wu1", worker_id="w1", branch_name="mc/unit-wu1",
			commit_hash="aaa", status="merged", position=1,
		)
		mr2 = MergeRequest(
			work_unit_id="wu2", worker_id="w1", branch_name="mc/unit-wu2",
			commit_hash="bbb", status="merged", position=2,
		)
		db.insert_merge_request(mr1)
		db.insert_merge_request(mr2)

		mock_pool = AsyncMock()
		mock_pool.acquire = AsyncMock(side_effect=[
			Path("/tmp/merge"),
			Path("/tmp/w1"),
			Path("/tmp/w2"),
		])
		mock_pool.initialize = AsyncMock()
		mock_pool.cleanup = AsyncMock()

		mock_backend = AsyncMock()
		mock_backend._pool = mock_pool
		mock_backend.initialize = AsyncMock()
		mock_backend.cleanup = AsyncMock()

		with (
			patch("mission_control.coordinator.snapshot_project_health", new_callable=AsyncMock) as mock_snap,
			patch("mission_control.coordinator.create_plan", new_callable=AsyncMock) as mock_create,
			patch("mission_control.coordinator.LocalBackend", return_value=mock_backend),
			patch("mission_control.coordinator.MergeQueue") as mock_mq_cls,
			patch("mission_control.coordinator.WorkerAgent") as mock_wa_cls,
		):
			mock_snap.return_value = Snapshot(test_total=10, test_passed=7, test_failed=3)
			mock_create.return_value = plan

			mock_mq = AsyncMock()
			mock_mq.run = AsyncMock()
			mock_mq.stop = lambda: None
			mock_mq_cls.return_value = mock_mq

			mock_agent = AsyncMock()
			mock_agent.run = AsyncMock()
			mock_agent.running = False
			mock_agent.stop = lambda: None
			mock_wa_cls.return_value = mock_agent

			coord = Coordinator(config, db, num_workers=2)
			report = await coord.run()

		assert report.plan_id == "p1"
		assert report.total_units == 2
		assert report.units_merged == 2
		assert report.workers_spawned == 2
		assert report.wall_time_seconds > 0

	async def test_stale_recovery_during_monitoring(self, db: Database, config: MissionConfig) -> None:
		"""Stale units are recovered during coordinator monitoring."""
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		# Insert a unit that looks stale (old heartbeat)
		wu = WorkUnit(
			id="wu1", plan_id="p1", title="Stale task",
			status="claimed", worker_id="dead-worker",
			heartbeat_at="2020-01-01T00:00:00+00:00",
		)
		db.insert_work_unit(wu)

		# Recover with a very short timeout
		recovered = db.recover_stale_units(1)
		assert len(recovered) == 1
		assert recovered[0].id == "wu1"

		# Unit is back to pending
		unit = db.get_work_unit("wu1")
		assert unit is not None
		assert unit.status == "pending"
		assert unit.worker_id is None


class TestPromptRendering:
	"""Test that worker prompts contain all necessary information."""

	def test_full_prompt_contains_all_fields(self, config: MissionConfig) -> None:
		unit = WorkUnit(
			title="Fix auth bug",
			description="The login endpoint returns 500",
			files_hint="src/auth.py,tests/test_auth.py",
			verification_hint="Run pytest tests/test_auth.py",
		)
		prompt = render_worker_prompt(
			unit, config, "/tmp/clone", "mc/unit-abc",
			test_passed=8, test_total=10, lint_errors=2, type_errors=1,
			context="Previous worker fixed config loading",
		)
		assert "Fix auth bug" in prompt
		assert "The login endpoint returns 500" in prompt
		assert "src/auth.py,tests/test_auth.py" in prompt
		assert "Run pytest tests/test_auth.py" in prompt
		assert "8/10" in prompt
		assert "mc/unit-abc" in prompt
		assert "Previous worker fixed config loading" in prompt
		assert "pytest -q" in prompt  # verification command from config


class TestParallelMergeConflictIntegration:
	"""Integration: 3 workers targeting same file, verify DB state after merges."""

	async def test_three_workers_same_file_db_state(self, db: Database) -> None:
		"""Create 3 units targeting same file, simulate merges, verify DB state."""
		# Setup mission infrastructure
		mission = Mission(id="m1", objective="test merge conflicts", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test", status="active", total_units=3)
		db.insert_plan(plan)

		# 3 work units all targeting the same file
		units = []
		for i in range(3):
			wu = WorkUnit(
				id=f"wu{i}",
				plan_id="p1",
				title=f"Modify shared.py (worker {i})",
				description="Edit the same shared module",
				files_hint="src/shared.py",
				status="running",
				worker_id=f"w{i}",
				branch_name=f"mc/unit-wu{i}",
				commit_hash=f"commit{i}abc",
				epoch_id="e1",
			)
			db.insert_work_unit(wu)
			units.append(wu)

		# Setup GreenBranchManager with stateful mock
		cfg = MissionConfig()
		cfg.target = TargetConfig(
			name="test", path="/tmp/test", branch="main",
			verification=VerificationConfig(command="pytest -q"),
		)
		cfg.green_branch = GreenBranchConfig(
			working_branch="mc/working", green_branch="mc/green",
		)
		mgr = GreenBranchManager(cfg, db)
		mgr.workspace = "/tmp/workspace"
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		# Track which merge is first (succeeds), rest conflict
		merge_attempt = 0

		async def stateful_git(*args: str) -> tuple[bool, str]:
			nonlocal merge_attempt
			if args[0] == "merge" and len(args) > 1 and args[1] == "--no-ff":
				merge_attempt += 1
				if merge_attempt == 1:
					return (True, "Merge made by the 'ort' strategy.")
				return (False, "CONFLICT (content): Merge conflict in src/shared.py")
			if args[0] == "merge" and "--abort" in args:
				return (True, "")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=stateful_git)

		# Simulate sequential merge attempts (as controller would do)
		merge_results = []
		for wu in units:
			result = await mgr.merge_unit(f"/tmp/w{wu.id[-1]}", wu.branch_name)
			merge_results.append((wu, result))

		# Record events in DB based on merge results
		merged_count = 0
		rejected_count = 0
		for wu, result in merge_results:
			if result.merged:
				merged_count += 1
				wu.status = "completed"
				db.update_work_unit(wu)
				event = UnitEvent(
					mission_id="m1", epoch_id="e1", work_unit_id=wu.id,
					event_type="merged",
				)
				db.insert_unit_event(event)
			else:
				rejected_count += 1
				wu.status = "failed"
				wu.output_summary = result.failure_output
				db.update_work_unit(wu)
				event = UnitEvent(
					mission_id="m1", epoch_id="e1", work_unit_id=wu.id,
					event_type="rejected",
					details=result.failure_output,
				)
				db.insert_unit_event(event)

		# Verify: exactly 1 merged, 2 rejected
		assert merged_count == 1
		assert rejected_count == 2

		# Verify DB state: unit_events
		events = db.get_unit_events_for_mission("m1")
		merged_events = [e for e in events if e.event_type == "merged"]
		rejected_events = [e for e in events if e.event_type == "rejected"]
		assert len(merged_events) == 1
		assert len(rejected_events) == 2

		# Verify work unit statuses in DB
		for wu in units:
			db_unit = db.get_work_unit(wu.id)
			assert db_unit is not None
			if wu.id == merge_results[0][0].id and merge_results[0][1].merged:
				assert db_unit.status == "completed"
			else:
				# Check it's either completed (the winner) or failed (the losers)
				assert db_unit.status in ("completed", "failed")

		# Verify rejected events contain conflict details
		for e in rejected_events:
			assert "conflict" in e.details.lower() or "Merge conflict" in e.details
