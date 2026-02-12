"""Tests for the scheduler main loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mission_control.config import (
	BudgetConfig,
	GitConfig,
	MissionConfig,
	SchedulerConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.db import Database
from mission_control.models import Session, Snapshot
from mission_control.scheduler import Scheduler, SchedulerReport


def _config() -> MissionConfig:
	return MissionConfig(
		target=TargetConfig(
			name="test",
			path="/tmp/test",
			branch="main",
			objective="Build stuff",
			verification=VerificationConfig(command="pytest -q", timeout=60),
		),
		scheduler=SchedulerConfig(
			session_timeout=60,
			cooldown=0,
			max_sessions_per_run=3,
			model="sonnet",
			git=GitConfig(strategy="branch-per-session", auto_merge=False),
			budget=BudgetConfig(max_per_session_usd=1.0, max_per_run_usd=10.0),
		),
	)


def _healthy_snapshot() -> Snapshot:
	return Snapshot(test_total=10, test_passed=10, test_failed=0)


def _improved_snapshot() -> Snapshot:
	return Snapshot(test_total=12, test_passed=12, test_failed=0)


def _session(sid: str = "test-session", status: str = "completed") -> Session:
	return Session(
		id=sid,
		target_name="test",
		task_description="Fix tests",
		status=status,
		branch_name=f"mc/session-{sid}",
		exit_code=0,
		output_summary="Done",
	)


class TestScheduler:
	@pytest.mark.asyncio
	async def test_no_work_stops(self) -> None:
		"""Scheduler stops when no work is discovered."""
		config = _config()
		config.target.objective = ""
		db = Database(":memory:")

		with (
			patch("mission_control.scheduler.snapshot_project_health", new_callable=AsyncMock) as mock_snap,
		):
			mock_snap.return_value = _healthy_snapshot()

			scheduler = Scheduler(config, db)
			report = await scheduler.run(max_sessions=3)

		assert report.sessions_run == 0
		assert report.stopped_reason == "no_work"

	@pytest.mark.asyncio
	async def test_max_sessions_stops(self) -> None:
		"""Scheduler stops after max_sessions."""
		config = _config()
		db = Database(":memory:")

		call_count = 0

		async def mock_snapshot(_cfg: MissionConfig) -> Snapshot:
			nonlocal call_count
			call_count += 1
			if call_count % 2 == 1:
				return Snapshot(test_total=10, test_passed=8, test_failed=2)
			return Snapshot(test_total=10, test_passed=10, test_failed=0)

		spawn_count = 0

		async def mock_spawn_fn(*args: object, **kwargs: object) -> Session:
			nonlocal spawn_count
			spawn_count += 1
			return _session(f"s{spawn_count}")

		with (
			patch("mission_control.scheduler.snapshot_project_health", side_effect=mock_snapshot),
			patch("mission_control.scheduler.spawn_session", side_effect=mock_spawn_fn),
			patch("mission_control.scheduler.merge_branch", new_callable=AsyncMock),
			patch("mission_control.scheduler.delete_branch", new_callable=AsyncMock),
		):
			scheduler = Scheduler(config, db)
			report = await scheduler.run(max_sessions=2)

		assert report.sessions_run == 2
		assert report.stopped_reason == "max_sessions"

	@pytest.mark.asyncio
	async def test_hurt_session_reverted(self) -> None:
		"""Sessions that hurt are reverted."""
		config = _config()
		db = Database(":memory:")

		call_count = 0

		async def mock_snapshot(_cfg: MissionConfig) -> Snapshot:
			nonlocal call_count
			call_count += 1
			if call_count == 1:
				return Snapshot(test_total=10, test_passed=10, test_failed=0)
			return Snapshot(test_total=10, test_passed=8, test_failed=2)

		with (
			patch("mission_control.scheduler.snapshot_project_health", side_effect=mock_snapshot),
			patch("mission_control.scheduler.spawn_session", new_callable=AsyncMock) as mock_spawn,
			patch("mission_control.scheduler.delete_branch", new_callable=AsyncMock) as mock_delete,
			patch("mission_control.scheduler.merge_branch", new_callable=AsyncMock),
		):
			mock_spawn.return_value = _session()

			scheduler = Scheduler(config, db)
			report = await scheduler.run(max_sessions=1)

		assert report.sessions_hurt == 1
		mock_delete.assert_called_once()

	@pytest.mark.asyncio
	async def test_stop_method(self) -> None:
		"""Scheduler.stop() stops the loop."""
		config = _config()
		db = Database(":memory:")
		scheduler = Scheduler(config, db)
		scheduler.stop()
		assert scheduler.running is False

	@pytest.mark.asyncio
	async def test_report_structure(self) -> None:
		report = SchedulerReport()
		assert report.sessions_run == 0
		assert report.total_cost_usd == 0.0
		assert report.stopped_reason == ""

	@pytest.mark.asyncio
	async def test_previous_snapshot_is_from_prior_run(self) -> None:
		"""previous snapshot should be from the prior run, not the just-inserted before."""
		config = _config()
		db = Database(":memory:")

		# Insert a "prior run" snapshot first
		prior = Snapshot(id="prior-snap", test_total=5, test_passed=3, test_failed=2)
		db.insert_snapshot(prior)

		captured_previous: list[Snapshot | None] = []

		call_count = 0

		async def mock_snapshot(_cfg: MissionConfig) -> Snapshot:
			nonlocal call_count
			call_count += 1
			return Snapshot(test_total=10, test_passed=10, test_failed=0)

		def mock_discover(before, cfg, recent, previous):
			captured_previous.append(previous)
			return []  # no work -> stop

		with (
			patch("mission_control.scheduler.snapshot_project_health", side_effect=mock_snapshot),
			patch("mission_control.scheduler.discover_from_snapshot", side_effect=mock_discover),
		):
			scheduler = Scheduler(config, db)
			await scheduler.run(max_sessions=1)

		# previous should be the prior snapshot, NOT the just-inserted before
		assert len(captured_previous) == 1
		assert captured_previous[0] is not None
		assert captured_previous[0].id == "prior-snap"

	@pytest.mark.asyncio
	async def test_failed_delete_branch_sets_revert_failed(self) -> None:
		"""Failed delete_branch sets session status to revert_failed."""
		config = _config()
		db = Database(":memory:")

		call_count = 0

		async def mock_snapshot(_cfg: MissionConfig) -> Snapshot:
			nonlocal call_count
			call_count += 1
			if call_count == 1:
				return Snapshot(test_total=10, test_passed=10, test_failed=0)
			# After session: worse results trigger revert
			return Snapshot(test_total=10, test_passed=8, test_failed=2)

		mock_recover_proc = AsyncMock()
		mock_recover_proc.communicate = AsyncMock(return_value=(b"", b""))

		with (
			patch("mission_control.scheduler.snapshot_project_health", side_effect=mock_snapshot),
			patch("mission_control.scheduler.spawn_session", new_callable=AsyncMock) as mock_spawn,
			patch("mission_control.scheduler.delete_branch", new_callable=AsyncMock) as mock_delete,
			patch("mission_control.scheduler.merge_branch", new_callable=AsyncMock),
			patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
		):
			mock_spawn.return_value = _session()
			mock_delete.return_value = False  # delete_branch fails
			mock_exec.return_value = mock_recover_proc

			scheduler = Scheduler(config, db)
			report = await scheduler.run(max_sessions=1)

		assert report.sessions_run == 1
		# Verify the session was persisted with revert_failed status
		sessions = db.get_recent_sessions(1)
		assert sessions[0].status == "revert_failed"

	@pytest.mark.asyncio
	async def test_failed_merge_branch_sets_merge_failed(self) -> None:
		"""Failed merge_branch sets session status to merge_failed."""
		config = _config()
		config.scheduler.git.auto_merge = True
		db = Database(":memory:")

		call_count = 0

		async def mock_snapshot(_cfg: MissionConfig) -> Snapshot:
			nonlocal call_count
			call_count += 1
			if call_count == 1:
				return Snapshot(test_total=10, test_passed=8, test_failed=2)
			# After session: improvement triggers merge
			return Snapshot(test_total=10, test_passed=10, test_failed=0)

		with (
			patch("mission_control.scheduler.snapshot_project_health", side_effect=mock_snapshot),
			patch("mission_control.scheduler.spawn_session", new_callable=AsyncMock) as mock_spawn,
			patch("mission_control.scheduler.delete_branch", new_callable=AsyncMock),
			patch("mission_control.scheduler.merge_branch", new_callable=AsyncMock) as mock_merge,
		):
			mock_spawn.return_value = _session()
			mock_merge.return_value = False  # merge_branch fails

			scheduler = Scheduler(config, db)
			report = await scheduler.run(max_sessions=1)

		assert report.sessions_run == 1
		sessions = db.get_recent_sessions(1)
		assert sessions[0].status == "merge_failed"

	@pytest.mark.asyncio
	async def test_spawn_session_oserror_caught(self) -> None:
		"""OSError from spawn_session is caught and scheduler stops gracefully."""
		config = _config()
		db = Database(":memory:")

		async def mock_snapshot(_cfg: MissionConfig) -> Snapshot:
			return Snapshot(test_total=10, test_passed=8, test_failed=2)

		with (
			patch("mission_control.scheduler.snapshot_project_health", side_effect=mock_snapshot),
			patch("mission_control.scheduler.spawn_session", new_callable=AsyncMock) as mock_spawn,
		):
			mock_spawn.side_effect = OSError("No such file or directory: 'claude'")

			scheduler = Scheduler(config, db)
			report = await scheduler.run(max_sessions=3)

		assert report.sessions_run == 0
		assert report.stopped_reason == "spawn_error"
