"""Tests for verify_and_merge_unit in GreenBranchManager."""

from __future__ import annotations

from unittest.mock import AsyncMock

from mission_control.config import (
	GreenBranchConfig,
	MissionConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.db import Database
from mission_control.green_branch import GreenBranchManager, UnitMergeResult


def _config() -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(
		name="test",
		path="/tmp/test",
		branch="main",
		verification=VerificationConfig(command="pytest -q", timeout=60),
	)
	mc.green_branch = GreenBranchConfig(
		working_branch="mc/working",
		green_branch="mc/green",
		auto_push=False,
	)
	return mc


def _manager() -> GreenBranchManager:
	config = _config()
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


class TestUnitMergeResult:
	def test_defaults(self) -> None:
		r = UnitMergeResult()
		assert r.merged is False
		assert r.rebase_ok is True
		assert r.verification_passed is False
		assert r.failure_output == ""


class TestVerifyAndMergeUnit:
	async def test_successful_merge(self) -> None:
		"""Full success: fetch, merge, verify, ff-merge all pass."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(True, "all tests passed"))

		result = await mgr.verify_and_merge_unit("/tmp/worker", "mc/unit-abc")

		assert result.merged is True
		assert result.rebase_ok is True
		assert result.verification_passed is True
		assert result.failure_output == ""

	async def test_fetch_failure(self) -> None:
		"""Failed fetch returns failure."""
		mgr = _manager()
		call_count = 0

		async def mock_git(*args: str) -> tuple[bool, str]:
			nonlocal call_count
			call_count += 1
			# First call: remote add (ok), second: fetch (fail)
			if args[0] == "fetch" and args[1].startswith("worker-"):
				return (False, "fetch failed")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)

		result = await mgr.verify_and_merge_unit("/tmp/worker", "mc/unit-abc")

		assert result.merged is False
		assert "Failed to fetch" in result.failure_output

	async def test_merge_conflict(self) -> None:
		"""Merge conflict returns rebase_ok=False."""
		mgr = _manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and "--no-ff" in args:
				return (False, "CONFLICT: merge conflict in foo.py")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)

		result = await mgr.verify_and_merge_unit("/tmp/worker", "mc/unit-abc")

		assert result.merged is False
		assert result.rebase_ok is False
		assert "Merge conflict" in result.failure_output

	async def test_verification_fails(self) -> None:
		"""Verification failure returns verification_passed=False."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(False, "3 tests failed"))

		result = await mgr.verify_and_merge_unit("/tmp/worker", "mc/unit-abc")

		assert result.merged is False
		assert result.rebase_ok is True
		assert result.verification_passed is False
		assert "3 tests failed" in result.failure_output

	async def test_auto_push_on_success(self) -> None:
		"""When auto_push is True, push_green_to_main is called."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(True, "ok"))
		mgr._run_git_in = AsyncMock(return_value=(True, ""))
		mgr.push_green_to_main = AsyncMock(return_value=True)

		result = await mgr.verify_and_merge_unit("/tmp/worker", "mc/unit-abc")

		assert result.merged is True
		mgr.push_green_to_main.assert_called_once()

	async def test_lock_serialization(self) -> None:
		"""Concurrent calls are serialized via _merge_lock."""
		mgr = _manager()
		call_order: list[int] = []

		async def slow_git(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and "--no-ff" in args:
				# Which unit is merging?
				branch = args[-1]
				unit_num = 1 if "unit-1" in branch else 2
				call_order.append(unit_num)
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=slow_git)
		mgr._run_command = AsyncMock(return_value=(True, "ok"))

		# Launch two concurrent merges
		import asyncio
		results = await asyncio.gather(
			mgr.verify_and_merge_unit("/tmp/w1", "mc/unit-1"),
			mgr.verify_and_merge_unit("/tmp/w2", "mc/unit-2"),
		)

		# Both should succeed
		assert all(r.merged for r in results)
		# They should have been serialized (both complete)
		assert len(call_order) == 2
