"""Tests for the green branch manager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from mission_control.config import (
	GreenBranchConfig,
	MissionConfig,
	SchedulerConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.db import Database
from mission_control.green_branch import FixupResult, GreenBranchManager


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
		fixup_max_attempts=3,
	)
	return mc


def _manager() -> GreenBranchManager:
	config = _config()
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


class TestFixupResult:
	def test_defaults(self) -> None:
		result = FixupResult()
		assert result.promoted is False
		assert result.fixup_attempts == 0
		assert result.failure_output == ""

	def test_custom_values(self) -> None:
		result = FixupResult(promoted=True, fixup_attempts=2, failure_output="error")
		assert result.promoted is True
		assert result.fixup_attempts == 2
		assert result.failure_output == "error"


class TestGreenBranchManagerInit:
	def test_init_sets_config_and_db(self) -> None:
		config = _config()
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		assert mgr.config is config
		assert mgr.db is db
		assert mgr.workspace == ""


class TestMergeToWorking:
	async def test_successful_merge(self) -> None:
		"""Successful merge returns True and cleans up remote."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.merge_to_working("/tmp/worker", "feat/add-tests")

		assert result is True
		# Verify the sequence: remote add, fetch, checkout, merge, remote remove
		calls = mgr._run_git.call_args_list
		assert calls[0].args == ("remote", "add", "worker-feat/add-tests", "/tmp/worker")
		assert calls[1].args == ("fetch", "worker-feat/add-tests", "feat/add-tests")
		assert calls[2].args == ("checkout", "mc/working")
		assert calls[3].args[:2] == ("merge", "--no-ff")
		assert calls[4].args == ("remote", "remove", "worker-feat/add-tests")

	async def test_fetch_failure_cleans_up_and_returns_false(self) -> None:
		"""Fetch failure removes the remote and returns False."""
		mgr = _manager()

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "fetch":
				return (False, "fatal: couldn't find remote ref")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)

		result = await mgr.merge_to_working("/tmp/worker", "feat/broken")

		assert result is False
		# Should have called remote remove to clean up
		calls = mgr._run_git.call_args_list
		remove_calls = [c for c in calls if c.args[0] == "remote" and c.args[1] == "remove"]
		assert len(remove_calls) == 1

	async def test_merge_conflict_aborts_and_returns_false(self) -> None:
		"""Merge conflict aborts the merge, cleans up remote, returns False."""
		mgr = _manager()

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and args[1] == "--no-ff":
				return (False, "CONFLICT (content): Merge conflict in file.py")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)

		result = await mgr.merge_to_working("/tmp/worker", "feat/conflict")

		assert result is False
		# Should have called merge --abort
		calls = mgr._run_git.call_args_list
		abort_calls = [c for c in calls if c.args == ("merge", "--abort")]
		assert len(abort_calls) == 1
		# Should have cleaned up the remote
		remove_calls = [c for c in calls if c.args[0] == "remote" and c.args[1] == "remove"]
		assert len(remove_calls) == 1


	async def test_concurrent_merges_are_serialized(self) -> None:
		"""Concurrent merge_to_working calls are serialized by the lock."""
		mgr = _manager()
		call_order: list[str] = []

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "checkout":
				call_order.append(f"checkout-{args[1]}")
				await asyncio.sleep(0.01)  # Simulate git latency
			elif args[0] == "merge" and args[1] == "--no-ff":
				call_order.append(f"merge-{args[2]}")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)

		# Launch two concurrent merges
		results = await asyncio.gather(
			mgr.merge_to_working("/tmp/w1", "branch-a"),
			mgr.merge_to_working("/tmp/w2", "branch-b"),
		)
		assert all(results)

		# Checkout calls should not interleave -- all of one merge's
		# operations should complete before the other starts
		checkout_indices = [i for i, c in enumerate(call_order) if c.startswith("checkout")]
		merge_indices = [i for i, c in enumerate(call_order) if c.startswith("merge")]
		# With serialization, merge-a should come before checkout for branch-b
		# (or vice versa), meaning they don't interleave
		assert len(merge_indices) == 2
		# The first merge should complete (checkout + merge) before second starts
		assert merge_indices[0] < checkout_indices[1]


class TestRunFixup:
	async def test_verification_passes_first_try(self) -> None:
		"""Verification passes immediately: promoted=True, 0 fixup attempts."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(True, "10 passed"))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.run_fixup()

		assert result.promoted is True
		assert result.fixup_attempts == 0
		assert result.failure_output == ""
		# Should have checked out working, run verify, checked out green, merged ff-only
		git_calls = mgr._run_git.call_args_list
		assert git_calls[0].args == ("checkout", "mc/working")
		assert git_calls[1].args == ("checkout", "mc/green")
		assert git_calls[2].args == ("merge", "--ff-only", "mc/working")

	async def test_verification_fails_fixup_succeeds(self) -> None:
		"""Verification fails, Claude fixup runs, re-verify passes."""
		mgr = _manager()

		git_results = [
			(True, ""),              # checkout mc/working
			(True, "abc123\n"),      # rev-parse HEAD (save state)
			(True, ""),              # checkout mc/green
			(True, ""),              # merge --ff-only
		]
		mgr._run_git = AsyncMock(side_effect=git_results)
		mgr._run_command = AsyncMock(side_effect=[
			(False, "2 failed, 8 passed"),  # initial verification
			(True, "10 passed"),              # re-verification
		])
		mgr._run_claude = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.run_fixup()

		assert result.promoted is True
		assert result.fixup_attempts == 1
		assert result.failure_output == ""

	async def test_verification_fails_all_fixups_exhausted(self) -> None:
		"""Verification fails, all fixup attempts exhausted, state restored each time."""
		mgr = _manager()
		mgr.config.green_branch.fixup_max_attempts = 2
		mgr._run_git = AsyncMock(return_value=(True, "abc123\n"))
		mgr._run_command = AsyncMock(side_effect=[
			(False, "2 failed"),   # initial verification
			(False, "1 failed"),   # re-verification attempt 1
			(False, "1 failed"),   # re-verification attempt 2 (final output)
		])
		mgr._run_claude = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup()

		assert result.promoted is False
		assert result.fixup_attempts == 2
		assert result.failure_output == "1 failed"

	async def test_fixup_succeeds_on_second_attempt(self) -> None:
		"""Verification fails, first fixup re-verify fails (restore), second succeeds."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, "abc123\n"))
		mgr._run_command = AsyncMock(side_effect=[
			(False, "3 failed"),   # initial verification
			(False, "1 failed"),   # re-verification attempt 1 still fails
			(True, "10 passed"),   # re-verification attempt 2 passes
		])
		mgr._run_claude = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.run_fixup()

		assert result.promoted is True
		assert result.fixup_attempts == 2

	async def test_fixup_agent_crash_restores_state(self) -> None:
		"""When fixup agent raises an exception, state is restored and next attempt runs."""
		mgr = _manager()
		mgr.config.green_branch.fixup_max_attempts = 2
		mgr._run_git = AsyncMock(return_value=(True, "abc123\n"))
		mgr._run_command = AsyncMock(side_effect=[
			(False, "2 failed"),   # initial verification
			(True, "10 passed"),   # re-verification after attempt 2
		])
		mgr._run_claude = AsyncMock(side_effect=[
			RuntimeError("Claude crashed"),  # fixup agent crash attempt 1
			(True, ""),                        # fixup attempt 2 succeeds
		])
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.run_fixup()

		assert result.promoted is True
		assert result.fixup_attempts == 2

	async def test_fixup_agent_returns_failure_restores_state(self) -> None:
		"""When fixup agent returns non-zero exit, state is restored."""
		mgr = _manager()
		mgr.config.green_branch.fixup_max_attempts = 1
		mgr._run_git = AsyncMock(return_value=(True, "abc123\n"))
		mgr._run_command = AsyncMock(return_value=(False, "2 failed"))
		mgr._run_claude = AsyncMock(return_value=(False, "claude error"))

		result = await mgr.run_fixup()

		assert result.promoted is False
		assert result.fixup_attempts == 1
		# Verify restore was called (reset --hard + clean -fd)
		git_calls = mgr._run_git.call_args_list
		reset_calls = [c for c in git_calls if len(c.args) >= 2 and c.args[0] == "reset"]
		assert len(reset_calls) >= 1

	async def test_claude_receives_prompt_via_method(self) -> None:
		"""Verify the prompt with shell metacharacters is passed safely via _run_claude."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, "abc123\n"))

		malicious_output = 'test $(rm -rf /) `whoami` "quotes" && echo pwned'
		mgr._run_command = AsyncMock(side_effect=[
			(False, malicious_output),  # initial verification with dangerous output
			(True, "10 passed"),        # re-verification passes
		])
		mgr._run_claude = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.run_fixup()

		assert result.promoted is True
		# Verify the prompt was passed to _run_claude, not _run_command
		mgr._run_claude.assert_called_once()
		prompt_arg = mgr._run_claude.call_args.args[0]
		assert malicious_output in prompt_arg

	async def test_promotion_fails_on_ff_merge_failure(self) -> None:
		"""When ff-only merge fails, run_fixup returns promoted=False."""
		mgr = _manager()
		# _run_git succeeds for checkout but fails for merge
		call_count = 0

		async def _mock_run_git(*args: str) -> tuple[bool, str]:
			nonlocal call_count
			call_count += 1
			if args[0] == "merge" and "--ff-only" in args:
				return (False, "fatal: Not possible to fast-forward")
			return (True, "ok\n")

		mgr._run_git = _mock_run_git  # type: ignore[assignment]
		mgr._run_command = AsyncMock(return_value=(True, "10 passed"))

		result = await mgr.run_fixup()

		assert result.promoted is False
		assert result.failure_output == "ff-only merge failed"

	async def test_promotion_fails_on_ff_merge_in_fixup_loop(self) -> None:
		"""When ff-only merge fails inside the fixup loop, returns promoted=False."""
		mgr = _manager()
		mgr.config.green_branch.fixup_max_attempts = 1

		async def _mock_run_git(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and "--ff-only" in args:
				return (False, "fatal: Not possible to fast-forward")
			return (True, "abc123\n")

		mgr._run_git = _mock_run_git  # type: ignore[assignment]
		mgr._run_command = AsyncMock(side_effect=[
			(False, "2 failed"),  # initial verification fails
			(True, "10 passed"),  # re-verification after fixup passes
		])
		mgr._run_claude = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup()

		assert result.promoted is False
		assert result.fixup_attempts == 1
		assert result.failure_output == "ff-only merge failed"


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


class TestRunClaudeTimeout:
	async def test_timeout_kills_subprocess(self) -> None:
		"""_run_claude should kill the subprocess on timeout."""
		mgr = _manager()
		mgr.config.scheduler = SchedulerConfig(session_timeout=1)

		mock_proc = AsyncMock()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch("mission_control.green_branch.asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch("mission_control.green_branch.asyncio.wait_for", side_effect=asyncio.TimeoutError):
				ok, output = await mgr._run_claude("fix stuff", 1.0)

		assert ok is False
		assert "timed out" in output
		mock_proc.kill.assert_called_once()
		mock_proc.wait.assert_called_once()

	async def test_normal_completion(self) -> None:
		"""_run_claude returns output on normal completion."""
		mgr = _manager()
		mgr.config.scheduler = SchedulerConfig(session_timeout=60)

		mock_proc = AsyncMock()
		mock_proc.returncode = 0

		with patch("mission_control.green_branch.asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch(
				"mission_control.green_branch.asyncio.wait_for",
				return_value=(b"fixed it", None),
			):
				ok, output = await mgr._run_claude("fix stuff", 1.0)

		assert ok is True
		assert output == "fixed it"


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
	"""Phase 65: Existing branches are reset to base on init when reset_on_init=True."""

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


class TestSyncCalledOnPromotion:
	"""Verify _sync_to_source is called at all 3 promotion sites."""

	async def test_clean_pass_syncs(self) -> None:
		"""Verification passes immediately: _sync_to_source called."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(True, "10 passed"))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.run_fixup()

		assert result.promoted is True
		mgr._sync_to_source.assert_awaited_once()

	async def test_fixup_success_syncs(self) -> None:
		"""Verification fails then fixup succeeds: _sync_to_source called."""
		mgr = _manager()
		git_results = [
			(True, ""),              # checkout mc/working
			(True, "abc123\n"),      # rev-parse HEAD
			(True, ""),              # checkout mc/green
			(True, ""),              # merge --ff-only
		]
		mgr._run_git = AsyncMock(side_effect=git_results)
		mgr._run_command = AsyncMock(side_effect=[
			(False, "2 failed"),
			(True, "10 passed"),
		])
		mgr._run_claude = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.run_fixup()

		assert result.promoted is True
		mgr._sync_to_source.assert_awaited_once()

	async def test_verify_and_merge_unit_syncs(self) -> None:
		"""verify_and_merge_unit: _sync_to_source called after successful merge."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(True, "10 passed"))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.verify_and_merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		mgr._sync_to_source.assert_awaited_once()

	async def test_failed_promotion_does_not_sync(self) -> None:
		"""When all fixups fail, _sync_to_source is NOT called."""
		mgr = _manager()
		mgr.config.green_branch.fixup_max_attempts = 1
		mgr._run_git = AsyncMock(return_value=(True, "abc123\n"))
		mgr._run_command = AsyncMock(return_value=(False, "2 failed"))
		mgr._run_claude = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.run_fixup()

		assert result.promoted is False
		mgr._sync_to_source.assert_not_awaited()
