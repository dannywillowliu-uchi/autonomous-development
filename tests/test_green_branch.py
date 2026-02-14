"""Tests for the green branch manager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from mission_control.config import (
	GreenBranchConfig,
	MissionConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.db import Database
from mission_control.green_branch import GreenBranchManager


def _config() -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(
		name="test",
		path="/tmp/test",
		branch="main",
		verification=VerificationConfig(command="pytest -q"),
	)
	mc.green_branch = GreenBranchConfig(
		working_branch="mc/working",
		green_branch="mc/green",
	)
	return mc


def _manager() -> GreenBranchManager:
	config = _config()
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


class TestGreenBranchManagerInit:
	def test_init_sets_config_and_db(self) -> None:
		config = _config()
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		assert mgr.config is config
		assert mgr.db is db
		assert mgr.workspace == ""


class TestMergeUnit:
	"""Tests for merge_unit() -- merge-only path without verification."""

	async def test_successful_merge(self) -> None:
		"""Successful merge: merged=True, no verification run."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert result.rebase_ok is True
		# Verify no _run_command was called (no verification)
		assert not hasattr(mgr, "_run_command") or not getattr(mgr._run_command, "called", False)
		mgr._sync_to_source.assert_awaited_once()

	async def test_merge_conflict(self) -> None:
		"""Merge conflict returns rebase_ok=False with failure output."""
		mgr = _manager()

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and args[1] == "--no-ff":
				return (False, "CONFLICT (content): Merge conflict in file.py")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)

		result = await mgr.merge_unit("/tmp/worker", "feat/conflict")

		assert result.merged is False
		assert result.rebase_ok is False
		assert "Merge conflict" in result.failure_output

	async def test_fetch_failure(self) -> None:
		"""Fetch failure returns failure output."""
		mgr = _manager()

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "fetch":
				return (False, "fatal: couldn't find remote ref")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)

		result = await mgr.merge_unit("/tmp/worker", "feat/broken")

		assert result.merged is False
		assert result.failure_output == "Failed to fetch unit branch"

	async def test_auto_push(self) -> None:
		"""When auto_push is configured, push_green_to_main is called."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]
		mgr.push_green_to_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		mgr.push_green_to_main.assert_awaited_once()

	async def test_ff_only_failure(self) -> None:
		"""When ff-only merge fails, returns failure."""
		mgr = _manager()

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and "--ff-only" in args:
				return (False, "fatal: Not possible to fast-forward")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is False
		assert result.failure_output == "ff-only merge failed"

	async def test_sync_called_on_success(self) -> None:
		"""merge_unit calls _sync_to_source after successful merge."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		mgr._sync_to_source.assert_awaited_once()


class TestGetGreenHash:
	async def test_returns_hash(self) -> None:
		"""Returns stripped commit hash from git rev-parse."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, "abc123def456\n"))

		result = await mgr.get_green_hash()

		assert result == "abc123def456"
		mgr._run_git.assert_called_once_with("rev-parse", "mc/green")

	async def test_returns_empty_on_failure(self) -> None:
		"""Returns empty string when git rev-parse fails."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(False, "fatal: bad revision"))

		result = await mgr.get_green_hash()

		assert result == ""


class TestRunCommandTimeout:
	async def test_timeout_kills_subprocess(self) -> None:
		"""_run_command should kill the subprocess on timeout."""
		mgr = _manager()
		mgr.config.target.verification.timeout = 1

		mock_proc = AsyncMock()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch("mission_control.green_branch.asyncio.create_subprocess_shell", return_value=mock_proc):
			with patch("mission_control.green_branch.asyncio.wait_for", side_effect=asyncio.TimeoutError):
				ok, output = await mgr._run_command("pytest -q")

		assert ok is False
		assert "timed out" in output
		mock_proc.kill.assert_called_once()
		mock_proc.wait.assert_called_once()


class TestInitializeResetOnInit:
	"""Existing branches are reset to base on init when reset_on_init=True."""

	async def test_existing_branches_reset_to_base(self) -> None:
		"""When branches exist and reset_on_init is True, update-ref is called."""
		config = _config()
		config.green_branch.reset_on_init = True
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		source_calls: list[tuple[str, tuple[str, ...]]] = []
		workspace_calls: list[tuple[str, ...]] = []

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			source_calls.append((cwd, args))
			if args[0] == "rev-parse":
				return True, "abc123"
			return True, ""

		async def mock_run_git(*args: str) -> tuple[bool, str]:
			workspace_calls.append(args)
			if args[0] == "rev-parse":
				return True, "abc123"
			return True, ""

		with patch.object(mgr, "_run_git_in", side_effect=mock_run_git_in), \
			patch.object(mgr, "_run_git", side_effect=mock_run_git):
			await mgr.initialize("/tmp/workspace")

		# Source repo: should call update-ref for both branches
		src_update_refs = [c[1] for c in source_calls if c[1][0] == "update-ref"]
		assert len(src_update_refs) == 2
		assert src_update_refs[0] == ("update-ref", "refs/heads/mc/working", "main")
		assert src_update_refs[1] == ("update-ref", "refs/heads/mc/green", "main")

		# Workspace: should also call update-ref for both branches
		ws_update_refs = [c for c in workspace_calls if c[0] == "update-ref"]
		assert len(ws_update_refs) == 2

		# Should NOT call branch (branches already exist)
		src_branch_calls = [c[1] for c in source_calls if c[1][0] == "branch"]
		assert len(src_branch_calls) == 0

	async def test_reset_on_init_false_preserves_branches(self) -> None:
		"""When reset_on_init is False, existing branches are not touched."""
		config = _config()
		config.green_branch.reset_on_init = False
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		source_calls: list[tuple[str, tuple[str, ...]]] = []
		workspace_calls: list[tuple[str, ...]] = []

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			source_calls.append((cwd, args))
			if args[0] == "rev-parse":
				return True, "abc123"
			return True, ""

		async def mock_run_git(*args: str) -> tuple[bool, str]:
			workspace_calls.append(args)
			if args[0] == "rev-parse":
				return True, "abc123"
			return True, ""

		with patch.object(mgr, "_run_git_in", side_effect=mock_run_git_in), \
			patch.object(mgr, "_run_git", side_effect=mock_run_git):
			await mgr.initialize("/tmp/workspace")

		# No update-ref calls in source or workspace
		src_update_refs = [c[1] for c in source_calls if c[1][0] == "update-ref"]
		assert len(src_update_refs) == 0
		ws_update_refs = [c for c in workspace_calls if c[0] == "update-ref"]
		assert len(ws_update_refs) == 0

	async def test_new_branches_created_regardless_of_setting(self) -> None:
		"""New branches are created even when reset_on_init is True."""
		config = _config()
		config.green_branch.reset_on_init = True
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		source_calls: list[tuple[str, tuple[str, ...]]] = []
		workspace_calls: list[tuple[str, ...]] = []

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			source_calls.append((cwd, args))
			if args[0] == "rev-parse":
				return False, ""
			return True, ""

		async def mock_run_git(*args: str) -> tuple[bool, str]:
			workspace_calls.append(args)
			if args[0] == "rev-parse":
				return False, ""
			return True, ""

		with patch.object(mgr, "_run_git_in", side_effect=mock_run_git_in), \
			patch.object(mgr, "_run_git", side_effect=mock_run_git):
			await mgr.initialize("/tmp/workspace")

		# Source repo: should create branches, not update-ref
		src_branch_calls = [c[1] for c in source_calls if c[1][0] == "branch"]
		src_update_refs = [c[1] for c in source_calls if c[1][0] == "update-ref"]
		assert len(src_branch_calls) == 2
		assert len(src_update_refs) == 0

		# Workspace: should also create branches from origin
		ws_branch_calls = [c for c in workspace_calls if c[0] == "branch"]
		assert len(ws_branch_calls) == 2


class TestSyncToSource:
	"""Tests for _sync_to_source: pushing refs from workspace clone back to source repo."""

	async def test_syncs_both_branches(self) -> None:
		"""_sync_to_source fetches both mc/green and mc/working into source repo."""
		mgr = _manager()
		calls: list[tuple[str, tuple[str, ...]]] = []

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			calls.append((cwd, args))
			return (True, "")

		mgr._run_git_in = mock_run_git_in  # type: ignore[assignment]

		await mgr._sync_to_source()

		assert len(calls) == 2
		# First call: fetch mc/green
		assert calls[0][0] == "/tmp/test"  # source repo path
		assert calls[0][1] == ("fetch", "/tmp/test-workspace", "mc/green:mc/green")
		# Second call: fetch mc/working
		assert calls[1][0] == "/tmp/test"
		assert calls[1][1] == ("fetch", "/tmp/test-workspace", "mc/working:mc/working")

	async def test_warns_on_failure_but_does_not_raise(self) -> None:
		"""Failed sync logs a warning but doesn't propagate the error."""
		mgr = _manager()

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			return (False, "fetch failed")

		mgr._run_git_in = mock_run_git_in  # type: ignore[assignment]

		# Should not raise
		await mgr._sync_to_source()

	async def test_partial_failure_continues(self) -> None:
		"""If first branch sync fails, second is still attempted."""
		mgr = _manager()
		calls: list[tuple[str, tuple[str, ...]]] = []

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			calls.append((cwd, args))
			if "mc/green:mc/green" in args:
				return (False, "failed")
			return (True, "")

		mgr._run_git_in = mock_run_git_in  # type: ignore[assignment]

		await mgr._sync_to_source()

		# Both branches attempted despite first failure
		assert len(calls) == 2
