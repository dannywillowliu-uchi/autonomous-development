"""Tests for the sequential merge queue."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from mission_control.config import MissionConfig, TargetConfig
from mission_control.db import Database
from mission_control.merge_queue import MergeQueue
from mission_control.models import MergeRequest, Plan, Snapshot, Worker, WorkUnit


def _config() -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(name="test", path="/tmp/test", branch="main")
	return mc


def _db_with_mr(
	mr_status: str = "pending",
	unit_attempt: int = 0,
	unit_max_attempts: int = 3,
	position: int = 1,
) -> tuple[Database, MergeRequest, WorkUnit, Worker]:
	"""Create an in-memory DB pre-loaded with a merge request + unit + worker."""
	db = Database(":memory:")
	plan = Plan(id="plan1", objective="Test plan")
	db.insert_plan(plan)
	worker = Worker(id="w1234567", workspace_path="/tmp/worker1")
	db.insert_worker(worker)

	unit = WorkUnit(
		id="unit1",
		plan_id="plan1",
		title="Fix tests",
		status="running",
		worker_id="w1234567",
		branch_name="fix/tests",
		attempt=unit_attempt,
		max_attempts=unit_max_attempts,
	)
	db.insert_work_unit(unit)

	mr = MergeRequest(
		id="mr1",
		work_unit_id="unit1",
		worker_id="w1234567",
		branch_name="fix/tests",
		commit_hash="abc123",
		status=mr_status,
		position=position,
	)
	db.insert_merge_request(mr)

	return db, mr, unit, worker


class TestMergeQueue:
	@patch("mission_control.merge_queue.snapshot_project_health", new_callable=AsyncMock)
	async def test_process_helped_merge(self, mock_snapshot: AsyncMock) -> None:
		"""Verdict=helped -> MR merged, unit completed."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		# Mock snapshots (before/after identical = neutral but reviewer sees improvement)
		before_snap = Snapshot(test_total=10, test_passed=8, test_failed=2)
		after_snap = Snapshot(test_total=10, test_passed=10, test_failed=0)
		mock_snapshot.side_effect = [before_snap, after_snap]

		# Mock all git operations to succeed
		queue._run_git = AsyncMock(return_value=True)
		queue._fetch_worker_branch = AsyncMock(return_value=True)
		queue._rebase_onto_base = AsyncMock(return_value=True)
		queue._merge_into_base = AsyncMock(return_value=True)

		await queue._process_merge_request(mr)

		# MR should be merged
		updated_mr = db.get_next_merge_request()
		# No pending MR left since it was processed
		assert updated_mr is None
		# Check the MR directly
		row = db.conn.execute("SELECT * FROM merge_requests WHERE id=?", ("mr1",)).fetchone()
		assert row["status"] == "merged"
		assert row["merged_at"] is not None

		# Work unit should be completed
		updated_unit = db.get_work_unit("unit1")
		assert updated_unit is not None
		assert updated_unit.status == "completed"
		assert updated_unit.finished_at is not None

	@patch("mission_control.merge_queue.snapshot_project_health", new_callable=AsyncMock)
	async def test_process_rejected_merge(self, mock_snapshot: AsyncMock) -> None:
		"""Verdict=hurt -> MR rejected, unit released for retry."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		# After is worse: tests broken
		before_snap = Snapshot(test_total=10, test_passed=10, test_failed=0)
		after_snap = Snapshot(test_total=10, test_passed=8, test_failed=2)
		mock_snapshot.side_effect = [before_snap, after_snap]

		queue._run_git = AsyncMock(return_value=True)
		queue._fetch_worker_branch = AsyncMock(return_value=True)
		queue._rebase_onto_base = AsyncMock(return_value=True)

		await queue._process_merge_request(mr)

		row = db.conn.execute("SELECT * FROM merge_requests WHERE id=?", ("mr1",)).fetchone()
		assert row["status"] == "rejected"
		assert "Verification failed" in row["rejection_reason"]

		# Unit should be released back to pending (attempt < max)
		updated_unit = db.get_work_unit("unit1")
		assert updated_unit is not None
		assert updated_unit.status == "pending"
		assert updated_unit.attempt == 1

	async def test_rebase_conflict_handling(self) -> None:
		"""Rebase failure -> MR status=conflict, unit released for retry."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		queue._fetch_worker_branch = AsyncMock(return_value=True)
		queue._rebase_onto_base = AsyncMock(return_value=False)

		await queue._process_merge_request(mr)

		row = db.conn.execute("SELECT * FROM merge_requests WHERE id=?", ("mr1",)).fetchone()
		assert row["status"] == "conflict"
		assert row["rejection_reason"] == "Rebase conflict"
		assert row["rebase_attempts"] == 1

		# Unit released for retry
		updated_unit = db.get_work_unit("unit1")
		assert updated_unit is not None
		assert updated_unit.status == "pending"

	async def test_release_unit_for_retry(self) -> None:
		"""Unit with attempt < max_attempts goes back to pending."""
		db, mr, unit, worker = _db_with_mr(unit_attempt=1, unit_max_attempts=3)
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		await queue._release_unit_for_retry(mr)

		updated_unit = db.get_work_unit("unit1")
		assert updated_unit is not None
		assert updated_unit.status == "pending"
		assert updated_unit.worker_id is None
		assert updated_unit.claimed_at is None
		assert updated_unit.heartbeat_at is None
		assert updated_unit.attempt == 2

	async def test_release_unit_exhausted(self) -> None:
		"""Unit at max_attempts gets status=failed."""
		db, mr, unit, worker = _db_with_mr(unit_attempt=3, unit_max_attempts=3)
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		await queue._release_unit_for_retry(mr)

		updated_unit = db.get_work_unit("unit1")
		assert updated_unit is not None
		assert updated_unit.status == "failed"
		assert updated_unit.finished_at is not None

	async def test_queue_processes_in_order(self) -> None:
		"""Two MRs: lower position is processed first."""
		db = Database(":memory:")
		plan = Plan(id="p1", objective="Test plan")
		db.insert_plan(plan)
		worker = Worker(id="w1234567", workspace_path="/tmp/worker1")
		db.insert_worker(worker)

		unit1 = WorkUnit(
			id="u1", plan_id="p1", title="First",
			status="running", worker_id="w1234567", branch_name="b1",
		)
		unit2 = WorkUnit(
			id="u2", plan_id="p1", title="Second",
			status="running", worker_id="w1234567", branch_name="b2",
		)
		db.insert_work_unit(unit1)
		db.insert_work_unit(unit2)

		mr1 = MergeRequest(
			id="mr1", work_unit_id="u1", worker_id="w1234567",
			branch_name="b1", position=2, status="pending",
		)
		mr2 = MergeRequest(
			id="mr2", work_unit_id="u2", worker_id="w1234567",
			branch_name="b2", position=1, status="pending",
		)
		db.insert_merge_request(mr1)
		db.insert_merge_request(mr2)

		# get_next_merge_request should return position=1 first
		next_mr = db.get_next_merge_request()
		assert next_mr is not None
		assert next_mr.id == "mr2"
		assert next_mr.position == 1

	async def test_empty_queue_waits(self) -> None:
		"""No pending MRs: run loop sleeps and continues."""
		db = Database(":memory:")
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		call_count = 0

		async def fake_sleep(seconds: float) -> None:
			nonlocal call_count
			call_count += 1
			if call_count >= 2:
				queue.stop()

		with patch("mission_control.merge_queue.asyncio.sleep", side_effect=fake_sleep):
			await queue.run()

		# Should have slept at least twice before stopping
		assert call_count >= 2

	@patch("mission_control.merge_queue.snapshot_project_health", new_callable=AsyncMock)
	async def test_neutral_verdict_merges(self, mock_snapshot: AsyncMock) -> None:
		"""Neutral verdict also merges (not just helped)."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		# Identical snapshots -> neutral verdict
		snap = Snapshot(test_total=10, test_passed=10, test_failed=0)
		mock_snapshot.side_effect = [snap, snap]

		queue._run_git = AsyncMock(return_value=True)
		queue._fetch_worker_branch = AsyncMock(return_value=True)
		queue._rebase_onto_base = AsyncMock(return_value=True)
		queue._merge_into_base = AsyncMock(return_value=True)

		await queue._process_merge_request(mr)

		row = db.conn.execute("SELECT * FROM merge_requests WHERE id=?", ("mr1",)).fetchone()
		assert row["status"] == "merged"

		updated_unit = db.get_work_unit("unit1")
		assert updated_unit is not None
		assert updated_unit.status == "completed"


class TestFetchWorkerBranchRemoteCleanup:
	async def test_remote_removed_after_fetch(self) -> None:
		"""_fetch_worker_branch should remove the git remote after fetching."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		git_calls: list[tuple[str, ...]] = []

		async def mock_run_git(*args: str) -> bool:
			git_calls.append(args)
			return True

		queue._run_git = mock_run_git  # type: ignore[assignment]

		result = await queue._fetch_worker_branch(mr)

		assert result is True
		# Verify remote was added then removed
		remote_add_calls = [c for c in git_calls if c[0] == "remote" and c[1] == "add"]
		remote_remove_calls = [c for c in git_calls if c[0] == "remote" and c[1] == "remove"]
		assert len(remote_add_calls) == 1
		assert len(remote_remove_calls) == 1
		# Same remote name used for add and remove
		assert remote_add_calls[0][2] == remote_remove_calls[0][2]

	async def test_local_branch_created_from_fetch_head(self) -> None:
		"""After fetching, a local branch must be created from FETCH_HEAD."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		git_calls: list[tuple[str, ...]] = []

		async def mock_run_git(*args: str) -> bool:
			git_calls.append(args)
			return True

		queue._run_git = mock_run_git  # type: ignore[assignment]

		await queue._fetch_worker_branch(mr)

		# Should create local branch from FETCH_HEAD
		branch_calls = [c for c in git_calls if c[0] == "branch" and c[1] == "-f"]
		assert len(branch_calls) == 1
		assert branch_calls[0][2] == mr.branch_name
		assert branch_calls[0][3] == "FETCH_HEAD"
		# Branch creation must happen before remote removal
		branch_idx = git_calls.index(branch_calls[0])
		remove_idx = next(i for i, c in enumerate(git_calls) if c[0] == "remote" and c[1] == "remove")
		assert branch_idx < remove_idx

	async def test_fetch_failure_skips_branch_creation(self) -> None:
		"""If fetch fails, don't try to create a local branch."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		git_calls: list[tuple[str, ...]] = []

		async def mock_run_git(*args: str) -> bool:
			git_calls.append(args)
			if args[0] == "fetch":
				return False
			return True

		queue._run_git = mock_run_git  # type: ignore[assignment]

		result = await queue._fetch_worker_branch(mr)

		assert result is False
		# Should NOT have created a branch
		branch_calls = [c for c in git_calls if c[0] == "branch"]
		assert len(branch_calls) == 0


class TestSnapshotComparison:
	@patch("mission_control.merge_queue.snapshot_project_health", new_callable=AsyncMock)
	async def test_before_snapshot_taken_on_base_branch(self, mock_snapshot: AsyncMock) -> None:
		"""Before snapshot must be taken on the base branch, not the feature branch."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		before_snap = Snapshot(test_total=10, test_passed=10, test_failed=0)
		after_snap = Snapshot(test_total=10, test_passed=9, test_failed=1)
		mock_snapshot.side_effect = [before_snap, after_snap]

		git_calls: list[tuple[str, ...]] = []

		async def mock_run_git(*args: str) -> bool:
			git_calls.append(args)
			return True

		queue._run_git = mock_run_git  # type: ignore[assignment]
		queue._fetch_worker_branch = AsyncMock(return_value=True)
		queue._rebase_onto_base = AsyncMock(return_value=True)

		await queue._process_merge_request(mr)

		# After rebase, should checkout base branch BEFORE taking before snapshot
		# Then checkout feature branch for after snapshot
		assert git_calls[0] == ("checkout", "main")  # base branch for before
		assert git_calls[1] == ("checkout", mr.branch_name)  # feature branch for after

	@patch("mission_control.merge_queue.snapshot_project_health", new_callable=AsyncMock)
	async def test_regression_detected_by_snapshot_diff(self, mock_snapshot: AsyncMock) -> None:
		"""Feature branch with more failures than base is rejected (not neutral)."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		# Base branch: all passing. Feature: introduces failure
		before_snap = Snapshot(test_total=10, test_passed=10, test_failed=0)
		after_snap = Snapshot(test_total=10, test_passed=8, test_failed=2)
		mock_snapshot.side_effect = [before_snap, after_snap]

		queue._run_git = AsyncMock(return_value=True)
		queue._fetch_worker_branch = AsyncMock(return_value=True)
		queue._rebase_onto_base = AsyncMock(return_value=True)

		await queue._process_merge_request(mr)

		row = db.conn.execute("SELECT * FROM merge_requests WHERE id=?", ("mr1",)).fetchone()
		assert row["status"] == "rejected"
		assert "Verification failed" in row["rejection_reason"]

		# Unit should be released for retry
		updated_unit = db.get_work_unit("unit1")
		assert updated_unit is not None
		assert updated_unit.status == "pending"


class TestRebaseOntoBase:
	async def test_rebase_uses_local_base_branch(self) -> None:
		"""_rebase_onto_base should rebase onto local base branch (not origin/)."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		git_calls: list[tuple[str, ...]] = []

		async def mock_run_git(*args: str) -> bool:
			git_calls.append(args)
			return True

		queue._run_git = mock_run_git  # type: ignore[assignment]

		result = await queue._rebase_onto_base(mr)

		assert result is True
		# Should checkout feature branch first
		assert git_calls[0] == ("checkout", mr.branch_name)
		# Then rebase onto LOCAL base branch (not origin/main)
		assert git_calls[1] == ("rebase", "main")
		# Verify it does NOT use origin/ prefix (local merges would be lost)
		rebase_calls = [c for c in git_calls if c[0] == "rebase"]
		for call in rebase_calls:
			assert not call[1].startswith("origin/"), \
				f"Rebase should use local ref, not {call[1]}"

	async def test_sequential_merges_preserve_prior_changes(self) -> None:
		"""Two MRs merged sequentially: second MR rebases onto post-first-merge state."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		git_calls: list[tuple[str, ...]] = []

		async def mock_run_git(*args: str) -> bool:
			git_calls.append(args)
			return True

		queue._run_git = mock_run_git  # type: ignore[assignment]

		# First rebase
		await queue._rebase_onto_base(mr)

		git_calls.clear()

		# Second rebase (different MR)
		mr2 = MergeRequest(
			id="mr2", work_unit_id="unit1", worker_id="w1234567",
			branch_name="fix/other", status="pending", position=2,
		)
		await queue._rebase_onto_base(mr2)

		# Second rebase should also use local base branch
		rebase_calls = [c for c in git_calls if c[0] == "rebase"]
		assert len(rebase_calls) == 1
		assert rebase_calls[0] == ("rebase", "main")

	async def test_rebase_failure_returns_to_base_branch(self) -> None:
		"""On rebase failure, workspace should be on base branch after abort."""
		db, mr, unit, worker = _db_with_mr()
		config = _config()
		queue = MergeQueue(config, db, "/tmp/merge-workspace")

		git_calls: list[tuple[str, ...]] = []

		async def mock_run_git(*args: str) -> bool:
			git_calls.append(args)
			if args[0] == "rebase" and args[1] != "--abort":
				return False
			return True

		queue._run_git = mock_run_git  # type: ignore[assignment]

		result = await queue._rebase_onto_base(mr)

		assert result is False
		# Should have: checkout branch, rebase (fail), abort, checkout base
		abort_calls = [c for c in git_calls if c == ("rebase", "--abort")]
		assert len(abort_calls) == 1
		# Last call should checkout the base branch
		assert git_calls[-1] == ("checkout", config.target.branch)
