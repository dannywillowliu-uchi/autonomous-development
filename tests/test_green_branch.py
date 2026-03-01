"""Tests for the green branch manager."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mission_control.config import (
	DeployConfig,
	GreenBranchConfig,
	MissionConfig,
	TargetConfig,
	TraceLogConfig,
	VerificationConfig,
)
from mission_control.continuous_controller import (
	ContinuousController,
	ContinuousMissionResult,
	WorkerCompletion,
)
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.green_branch import (
	FIXUP_PROMPTS,
	FixupCandidate,
	FixupResult,
	GreenBranchManager,
	UnitMergeResult,
)
from mission_control.models import (
	Epoch,
	Handoff,
	Mission,
	Plan,
	VerificationNodeKind,
	VerificationReport,
	VerificationResult,
	WorkUnit,
)
from mission_control.trace_log import TraceLogger


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
	# Disable pre-merge verification by default in tests to avoid
	# needing to mock _run_verification in every merge test
	mc.continuous.verify_before_merge = False
	return mc


def _manager() -> GreenBranchManager:
	config = _config()
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


def _fixup_config(fixup_candidates: int = 3) -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(
		name="test",
		path="/tmp/test",
		branch="main",
		verification=VerificationConfig(command="pytest -q && ruff check src/"),
	)
	mc.green_branch = GreenBranchConfig(
		working_branch="mc/working",
		green_branch="mc/green",
		fixup_candidates=fixup_candidates,
	)
	return mc


def _fixup_manager(fixup_candidates: int = 3) -> GreenBranchManager:
	config = _fixup_config(fixup_candidates)
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


def _candidate(
	index: int,
	passed: bool = False,
	tests: int = 0,
	lint: int = 0,
	diff: int = 0,
) -> FixupCandidate:
	return FixupCandidate(
		branch=f"mc/fixup-candidate-{index}",
		verification_passed=passed,
		tests_passed=tests,
		lint_errors=lint,
		diff_lines=diff,
	)


def _conflict_config() -> MissionConfig:
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


def _conflict_manager() -> GreenBranchManager:
	config = _conflict_config()
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


class TestMergeUnit:
	"""Tests for merge_unit() -- fetch, merge, push."""

	async def test_successful_merge(self) -> None:
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		mgr._sync_to_source.assert_awaited_once()

	async def test_merge_conflict(self) -> None:
		mgr = _manager()

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and "--no-ff" in args:
				return (False, "CONFLICT (content): Merge conflict in file.py")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)

		result = await mgr.merge_unit("/tmp/worker", "feat/conflict")

		assert result.merged is False
		assert result.failure_stage == "merge_conflict"

	async def test_merge_target_is_fetched_remote(self) -> None:
		mgr = _manager()
		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		await mgr.merge_unit("/tmp/worker", "feat/branch")

		# Rebase-first flow: merge target is the rebased branch, not the remote
		merge_calls = [c for c in git_calls if c[0] == "merge" and "--no-ff" in c]
		assert len(merge_calls) == 1
		assert "mc/rebase-feat/branch" in merge_calls[0][2]

	async def test_auto_push_batched(self) -> None:
		"""Push only fires after push_batch_size merges, not every merge."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_batch_size = 3
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]
		mgr.push_green_to_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

		# First two merges: no push
		for i in range(2):
			result = await mgr.merge_unit("/tmp/worker", f"feat/branch-{i}")
			assert result.merged is True
		mgr.push_green_to_main.assert_not_awaited()

		# Third merge: push fires (batch_size reached)
		result = await mgr.merge_unit("/tmp/worker", "feat/branch-2")
		assert result.merged is True
		mgr.push_green_to_main.assert_awaited_once()

	async def test_auto_push_batch_size_1_pushes_every_merge(self) -> None:
		"""With push_batch_size=1, every merge triggers a push (legacy behavior)."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_batch_size = 1
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]
		mgr.push_green_to_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")
		assert result.merged is True
		mgr.push_green_to_main.assert_awaited_once()

	async def test_fetch_failure(self) -> None:
		mgr = _manager()

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "fetch":
				return (False, "fatal: remote not found")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is False
		assert result.failure_stage == "fetch"

	async def test_cleanup_on_failure(self) -> None:
		mgr = _manager()
		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			if args[0] == "merge" and "--no-ff" in args:
				return (False, "conflict")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)

		await mgr.merge_unit("/tmp/worker", "feat/branch")

		remote_removes = [c for c in git_calls if c[0] == "remote" and c[1] == "remove"]
		assert len(remote_removes) >= 1


	async def test_merge_conflict_returns_failure(self) -> None:
		"""Merge conflict aborts and returns merge_conflict failure stage."""
		mgr = _manager()

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and "--no-ff" in args:
				return (False, "CONFLICT (content): Merge conflict in file.py")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)

		result = await mgr.merge_unit("/tmp/worker", "feat/conflict")

		assert result.merged is False
		assert result.failure_stage == "merge_conflict"


class TestMaybePush:
	"""Tests for maybe_push() batch push logic."""

	async def test_maybe_push_below_threshold_no_push(self) -> None:
		"""No push when merges_since_push < push_batch_size."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_batch_size = 5
		mgr._merges_since_push = 3
		mgr.push_green_to_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

		result = await mgr.maybe_push()

		assert result is False
		mgr.push_green_to_main.assert_not_awaited()

	async def test_maybe_push_at_threshold_pushes(self) -> None:
		"""Push fires when merges_since_push >= push_batch_size."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_batch_size = 5
		mgr._merges_since_push = 5
		mgr.push_green_to_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

		result = await mgr.maybe_push()

		assert result is True
		mgr.push_green_to_main.assert_awaited_once()
		assert mgr._merges_since_push == 0

	async def test_maybe_push_force_flushes(self) -> None:
		"""force=True pushes regardless of counter."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_batch_size = 100
		mgr._merges_since_push = 1
		mgr.push_green_to_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

		result = await mgr.maybe_push(force=True)

		assert result is True
		mgr.push_green_to_main.assert_awaited_once()
		assert mgr._merges_since_push == 0

	async def test_maybe_push_auto_push_disabled(self) -> None:
		"""Returns False when auto_push is disabled."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = False
		mgr.push_green_to_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

		result = await mgr.maybe_push(force=True)

		assert result is False
		mgr.push_green_to_main.assert_not_awaited()

	async def test_maybe_push_counter_resets_on_success(self) -> None:
		"""Counter resets to 0 after a successful push."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_batch_size = 2
		mgr._merges_since_push = 2
		mgr.push_green_to_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

		await mgr.maybe_push()
		assert mgr._merges_since_push == 0

	async def test_maybe_push_counter_not_reset_on_push_failure(self) -> None:
		"""Counter is NOT reset when push_green_to_main returns False."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_batch_size = 2
		mgr._merges_since_push = 2
		mgr.push_green_to_main = AsyncMock(return_value=False)  # type: ignore[method-assign]

		result = await mgr.maybe_push()

		assert result is False
		assert mgr._merges_since_push == 2


class TestPushGreenToMain:
	"""Tests for push_green_to_main() stash/checkout/push flow."""

	async def test_stashes_dirty_tree_before_checkout(self) -> None:
		"""Dirty working tree is stashed before checkout and restored after."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_branch = "main"

		calls: list[tuple[str, tuple[str, ...]]] = []

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			calls.append((cwd, args))
			if args[0] == "stash" and args[1] == "--include-untracked":
				return (True, "Saved working directory")
			return (True, "")

		mgr._run_git_in = mock_run_git_in  # type: ignore[assignment]

		result = await mgr.push_green_to_main()

		assert result is True
		git_cmds = [c[1] for c in calls]
		# Stash should come before checkout
		stash_idx = next(i for i, c in enumerate(git_cmds) if c[0] == "stash" and c[1] == "--include-untracked")
		checkout_idx = next(i for i, c in enumerate(git_cmds) if c[0] == "checkout")
		assert stash_idx < checkout_idx
		# Stash pop should come after push
		pop_idx = next(i for i, c in enumerate(git_cmds) if c[0] == "stash" and c[1] == "pop")
		push_idx = next(i for i, c in enumerate(git_cmds) if c[0] == "push")
		assert pop_idx > push_idx

	async def test_checkout_failure_returns_false_and_restores_stash(self) -> None:
		"""If checkout fails, return False and still pop stash."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_branch = "main"

		popped = []

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			if args[0] == "stash" and args[1] == "--include-untracked":
				return (True, "Saved working directory")
			if args[0] == "stash" and args[1] == "pop":
				popped.append(True)
				return (True, "")
			if args[0] == "checkout":
				return (False, "error: Your local changes would be overwritten")
			return (True, "")

		mgr._run_git_in = mock_run_git_in  # type: ignore[assignment]

		result = await mgr.push_green_to_main()

		assert result is False
		assert popped == [True]  # stash pop was called despite failure

	async def test_no_stash_when_tree_clean(self) -> None:
		"""When working tree is clean, stash pop is not called."""
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_branch = "main"

		calls: list[tuple[str, tuple[str, ...]]] = []

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			calls.append((cwd, args))
			if args[0] == "stash" and args[1] == "--include-untracked":
				return (True, "No local changes to save")
			return (True, "")

		mgr._run_git_in = mock_run_git_in  # type: ignore[assignment]

		result = await mgr.push_green_to_main()

		assert result is True
		git_cmds = [c[1] for c in calls]
		# No stash pop should be called
		assert not any(c[0] == "stash" and c[1] == "pop" for c in git_cmds)


class TestGetGreenHash:
	async def test_returns_hash(self) -> None:
		"""Returns stripped commit hash from git rev-parse."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, "abc123def456\n"))

		result = await mgr.get_green_hash()

		assert result == "abc123def456"
		mgr._run_git.assert_called_once_with("rev-parse", "mc/green")


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


class TestInitializeSetupCommand:
	"""Tests for running setup_command during initialize()."""

	async def test_setup_command_runs_on_init(self) -> None:
		"""When setup_command is configured, it runs at the end of initialize()."""
		config = _config()
		config.target.verification.setup_command = "npm install"
		config.target.verification.setup_timeout = 30
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"ok", None))

		with patch.object(mgr, "_run_git_in", AsyncMock(return_value=(False, ""))), \
			patch.object(mgr, "_run_git", AsyncMock(return_value=(True, ""))), \
			patch("mission_control.green_branch.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_shell:
			await mgr.initialize("/tmp/workspace")

		mock_shell.assert_called_once()
		assert mock_shell.call_args[0] == ("npm", "install")

	async def test_setup_command_failure_raises(self) -> None:
		"""Setup command failure raises RuntimeError."""
		config = _config()
		config.target.verification.setup_command = "npm install"
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		mock_proc = AsyncMock()
		mock_proc.returncode = 1
		mock_proc.communicate = AsyncMock(return_value=(b"ERR: not found", None))

		with patch.object(mgr, "_run_git_in", AsyncMock(return_value=(False, ""))), \
			patch.object(mgr, "_run_git", AsyncMock(return_value=(True, ""))), \
			patch("mission_control.green_branch.asyncio.create_subprocess_exec", return_value=mock_proc):
			with pytest.raises(RuntimeError, match="Workspace setup failed"):
				await mgr.initialize("/tmp/workspace")


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
		assert calls[0][1] == ("fetch", "/tmp/test-workspace", "+mc/green:mc/green")
		# Second call: fetch mc/working
		assert calls[1][0] == "/tmp/test"
		assert calls[1][1] == ("fetch", "/tmp/test-workspace", "+mc/working:mc/working")


class TestParallelMergeConflicts:
	"""Tests for concurrent merge scenarios and conflict handling."""

	async def test_concurrent_merge_only_one_succeeds(self) -> None:
		"""3 workers modify the same file; only 1 merge succeeds, 2 get conflicts."""
		mgr = _manager()
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		merge_count = 0

		async def stateful_git(*args: str) -> tuple[bool, str]:
			nonlocal merge_count
			if args[0] == "merge" and "--no-ff" in args:
				merge_count += 1
				if merge_count == 1:
					return (True, "")
				return (False, "CONFLICT (content): Merge conflict in shared.py")
			if args[0] == "merge" and "--abort" in args:
				return (True, "")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=stateful_git)

		results = await asyncio.gather(
			mgr.merge_unit("/tmp/w1", "feat/unit-1"),
			mgr.merge_unit("/tmp/w2", "feat/unit-2"),
			mgr.merge_unit("/tmp/w3", "feat/unit-3"),
		)

		succeeded = [r for r in results if r.merged]
		failed = [r for r in results if not r.merged]

		assert len(succeeded) == 1
		assert len(failed) == 2
		for r in failed:
			assert r.failure_stage == "merge_conflict"

	async def test_concurrent_merge_lock_serialization(self) -> None:
		"""_merge_lock serializes concurrent merge attempts."""
		mgr = _manager()
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		execution_order: list[str] = []

		async def ordered_git(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and "--no-ff" in args:
				branch = args[2].split("/")[-1] if len(args) > 2 else "unknown"
				execution_order.append(branch)
				await asyncio.sleep(0.01)
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=ordered_git)

		await asyncio.gather(
			mgr.merge_unit("/tmp/w1", "unit-a"),
			mgr.merge_unit("/tmp/w2", "unit-b"),
			mgr.merge_unit("/tmp/w3", "unit-c"),
		)

		assert len(execution_order) == 3
		# Rebase-first flow: merge target is mc/rebase-{branch}, not the branch itself
		assert set(execution_order) == {"rebase-unit-a", "rebase-unit-b", "rebase-unit-c"}


class TestVerificationOutsideLock:
	"""Tests for optimistic concurrency: verification runs outside _merge_lock."""

	async def test_lock_not_held_during_verification(self) -> None:
		"""Verification runs outside _merge_lock so other merges can proceed."""
		mgr = _gate_manager()
		lock_held_during_verify = False

		async def tracking_git(*args: str) -> tuple[bool, str]:
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		async def slow_verify() -> VerificationReport:
			nonlocal lock_held_during_verify
			lock_held_during_verify = mgr._merge_lock.locked()
			return VerificationReport(
				results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True)],
				raw_output="ok",
			)

		mgr._run_verification = AsyncMock(side_effect=slow_verify)  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert lock_held_during_verify is False

	async def test_rollback_uses_git_revert_not_reset(self) -> None:
		"""Rollback uses git revert (safe when HEAD advanced) instead of reset --hard."""
		mgr = _gate_manager()
		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			if args[0] == "rev-parse" and args[1] == "mc/green":
				return (True, "abc123def456\n")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)

		failing_report = VerificationReport(
			results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=False)],
			raw_output="FAIL",
		)
		mgr._run_verification = AsyncMock(return_value=failing_report)  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is False
		revert_calls = [c for c in git_calls if c[0] == "revert"]
		assert len(revert_calls) >= 1
		# The revert should target the specific merge commit hash
		assert "abc123def456" in revert_calls[0]

	async def test_finalize_sync_under_lock(self) -> None:
		"""Phase 3 (sync+push) runs under _merge_lock."""
		mgr = _gate_manager()
		lock_held_during_sync = False

		async def tracking_git(*args: str) -> tuple[bool, str]:
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)

		async def check_lock_sync() -> bool:
			nonlocal lock_held_during_sync
			lock_held_during_sync = mgr._merge_lock.locked()
			return True

		mgr._sync_to_source = AsyncMock(side_effect=check_lock_sync)  # type: ignore[method-assign]

		passing_report = VerificationReport(
			results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True)],
			raw_output="ok",
		)
		mgr._run_verification = AsyncMock(return_value=passing_report)  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert lock_held_during_sync is True


class TestMergeBatch:
	"""Tests for merge_batch() -- speculative batch merge with bisection."""

	async def test_batch_single_falls_back_to_merge_unit(self) -> None:
		"""A single-item batch delegates to merge_unit."""
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		results = await mgr.merge_batch([("/tmp/w1", "feat/a", "")])

		assert len(results) == 1
		assert results[0].merged is True

	async def test_batch_empty_returns_empty(self) -> None:
		"""An empty batch returns an empty list."""
		mgr = _manager()
		results = await mgr.merge_batch([])
		assert results == []

	async def test_batch_all_pass_single_verification(self) -> None:
		"""Multiple units merged + verified in one pass."""
		mgr = _gate_manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		passing_report = VerificationReport(
			results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True)],
			raw_output="ok",
		)
		mgr._run_verification = AsyncMock(return_value=passing_report)  # type: ignore[method-assign]

		units = [
			("/tmp/w1", "feat/a", ""),
			("/tmp/w2", "feat/b", ""),
			("/tmp/w3", "feat/c", ""),
		]
		results = await mgr.merge_batch(units)

		assert len(results) == 3
		for r in results:
			assert r.merged is True
			assert r.verification_passed is True
		# Verification should have been called only once for the batch
		mgr._run_verification.assert_awaited_once()

	async def test_batch_verification_disabled(self) -> None:
		"""When verify_before_merge=False, no verification runs."""
		mgr = _manager()  # verify_before_merge=False
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]
		mgr._run_verification = AsyncMock()  # type: ignore[method-assign]

		units = [
			("/tmp/w1", "feat/a", ""),
			("/tmp/w2", "feat/b", ""),
		]
		results = await mgr.merge_batch(units)

		assert len(results) == 2
		for r in results:
			assert r.merged is True
		mgr._run_verification.assert_not_awaited()

	async def test_batch_fetch_failure_partial(self) -> None:
		"""If one unit fails to fetch, others still succeed."""
		mgr = _manager()
		call_count = 0

		async def selective_git(*args: str) -> tuple[bool, str]:
			nonlocal call_count
			if args[0] == "fetch" and "feat/bad" in args:
				return (False, "fetch failed")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=selective_git)
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		units = [
			("/tmp/w1", "feat/good", ""),
			("/tmp/w2", "feat/bad", ""),
		]
		results = await mgr.merge_batch(units)

		assert len(results) == 2
		# The good unit should succeed
		assert results[0].merged is True
		# The bad unit should fail with fetch error
		assert results[1].merged is False
		assert results[1].failure_stage == "fetch"

	async def test_batch_bisection_on_verification_failure(self) -> None:
		"""When batch verification fails, bisection runs and uses merge_unit."""
		mgr = _gate_manager()

		async def tracking_git(*args: str) -> tuple[bool, str]:
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		failing_report = VerificationReport(
			results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=False)],
			raw_output="FAIL",
		)
		passing_report = VerificationReport(
			results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True)],
			raw_output="ok",
		)

		verify_calls = 0

		async def conditional_verify() -> VerificationReport:
			nonlocal verify_calls
			verify_calls += 1
			if verify_calls == 1:
				# First call (batch) fails
				return failing_report
			# Subsequent calls (bisection / individual) pass
			return passing_report

		mgr._run_verification = AsyncMock(side_effect=conditional_verify)  # type: ignore[method-assign]

		units = [
			("/tmp/w1", "feat/a", ""),
			("/tmp/w2", "feat/b", ""),
		]
		results = await mgr.merge_batch(units)

		assert len(results) == 2
		# Both should eventually succeed via bisection -> individual merge
		# (since individual verifications pass)
		assert verify_calls >= 2  # at least batch + bisect calls


class TestRunDeploy:
	"""Tests for run_deploy() with mock subprocess."""

	def _deploy_config(self) -> MissionConfig:
		mc = _config()
		mc.deploy = DeployConfig(
			enabled=True,
			command="vercel deploy --prod",
			timeout=60,
		)
		return mc

	async def test_successful_deploy(self) -> None:
		config = self._deploy_config()
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"Deployed!", None))

		with patch(
			"mission_control.green_branch.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		):
			ok, output = await mgr.run_deploy()

		assert ok is True
		assert "Deployed!" in output

	async def test_deploy_failure(self) -> None:
		config = self._deploy_config()
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		mock_proc = AsyncMock()
		mock_proc.returncode = 1
		mock_proc.communicate = AsyncMock(return_value=(b"Error: auth failed", None))

		with patch(
			"mission_control.green_branch.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		):
			ok, output = await mgr.run_deploy()

		assert ok is False
		assert "exit 1" in output

	async def test_deploy_timeout(self) -> None:
		config = self._deploy_config()
		config.deploy.timeout = 1
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		mock_proc = AsyncMock()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch(
			"mission_control.green_branch.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		), patch(
			"mission_control.green_branch.asyncio.wait_for",
			side_effect=asyncio.TimeoutError,
		):
			ok, output = await mgr.run_deploy()

		assert ok is False
		assert "timed out" in output

	async def test_deploy_after_auto_push(self) -> None:
		"""When auto_push and on_auto_push, deploy runs after push (batch_size=1)."""
		config = self._deploy_config()
		config.green_branch.auto_push = True
		config.green_branch.push_batch_size = 1  # push every merge so deploy fires
		config.deploy.on_auto_push = True
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)
		mgr.workspace = "/tmp/test-workspace"
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]
		mgr.push_green_to_main = AsyncMock(return_value=True)  # type: ignore[method-assign]
		mgr.run_deploy = AsyncMock(return_value=(True, "ok"))  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		mgr.push_green_to_main.assert_awaited_once()
		mgr.run_deploy.assert_awaited_once()


class TestPollHealthCheck:
	"""Tests for the httpx-based _poll_health_check implementation."""

	async def test_success_on_first_try(self) -> None:
		"""Health check returns True immediately on HTTP 200."""
		mgr = _manager()
		mock_response = httpx.Response(200)

		mock_client = AsyncMock(spec=httpx.AsyncClient)
		mock_client.get = AsyncMock(return_value=mock_response)
		mock_client.__aenter__ = AsyncMock(return_value=mock_client)
		mock_client.__aexit__ = AsyncMock(return_value=False)

		with patch("mission_control.green_branch.httpx.AsyncClient", return_value=mock_client):
			result = await mgr._poll_health_check("https://example.com/health", 30)

		assert result is True
		mock_client.get.assert_awaited_once()


def _run(cmd: list[str], cwd: str | Path) -> str:
	"""Run a git command synchronously, raise on failure."""
	result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
	return result.stdout.strip()


def _setup_source_repo(tmp_path: Path) -> tuple[Path, Path]:
	"""Create a bare source repo and a workspace clone with initial commits.

	Returns (source_repo, workspace) paths.
	"""
	source = tmp_path / "source.git"
	source.mkdir()
	_run(["git", "init", "--bare"], source)

	# Create a temporary clone to make initial commits
	setup_clone = tmp_path / "setup-clone"
	_run(["git", "clone", str(source), str(setup_clone)], tmp_path)
	_run(["git", "config", "user.email", "test@test.com"], setup_clone)
	_run(["git", "config", "user.name", "Test"], setup_clone)

	# Initial commit on main
	(setup_clone / "README.md").write_text("# Test Project\n")
	_run(["git", "add", "README.md"], setup_clone)
	_run(["git", "commit", "-m", "Initial commit"], setup_clone)
	_run(["git", "push", "origin", "main"], setup_clone)

	# Second commit so there's history
	(setup_clone / "app.py").write_text("print('hello')\n")
	_run(["git", "add", "app.py"], setup_clone)
	_run(["git", "commit", "-m", "Add app.py"], setup_clone)
	_run(["git", "push", "origin", "main"], setup_clone)

	# Create mc/green branch in source repo pointing to main
	main_hash = _run(["git", "rev-parse", "main"], setup_clone)
	_run(["git", "branch", "mc/green", main_hash], setup_clone)
	_run(["git", "push", "origin", "mc/green"], setup_clone)
	_run(["git", "branch", "mc/working", main_hash], setup_clone)
	_run(["git", "push", "origin", "mc/working"], setup_clone)

	# Create the workspace clone (this is what GreenBranchManager operates on)
	workspace = tmp_path / "workspace"
	_run(["git", "clone", str(source), str(workspace)], tmp_path)
	_run(["git", "config", "user.email", "test@test.com"], workspace)
	_run(["git", "config", "user.name", "Test"], workspace)
	# Track mc/green and mc/working locally
	_run(["git", "branch", "mc/green", "origin/mc/green"], workspace)
	_run(["git", "branch", "mc/working", "origin/mc/working"], workspace)

	return source, workspace


def _make_worker_clone(tmp_path: Path, source: Path, name: str) -> Path:
	"""Create a worker clone from the source repo and return its path."""
	worker = tmp_path / name
	_run(["git", "clone", str(source), str(worker)], tmp_path)
	_run(["git", "config", "user.email", "worker@test.com"], worker)
	_run(["git", "config", "user.name", "Worker"], worker)
	return worker


def _real_config(source: Path) -> MissionConfig:
	"""Build a MissionConfig pointing at real repos."""
	mc = MissionConfig()
	mc.target = TargetConfig(
		name="test",
		path=str(source),
		branch="main",
		verification=VerificationConfig(command="true"),
	)
	mc.green_branch = GreenBranchConfig(
		working_branch="mc/working",
		green_branch="mc/green",
		reset_on_init=False,
	)
	return mc


class TestGreenBranchRealGit:
	"""Integration tests using real git repos -- no mocks."""

	async def test_merge_unit_creates_merge_commit_on_green(self, tmp_path: Path) -> None:
		"""merge_unit produces a merge commit on mc/green with correct parents."""
		source, workspace = _setup_source_repo(tmp_path)
		config = _real_config(source)
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)
		mgr.workspace = str(workspace)

		# Get mc/green hash before merge
		green_before = _run(["git", "rev-parse", "mc/green"], workspace)

		# Create a worker clone and make a commit on a unit branch
		worker = _make_worker_clone(tmp_path, source, "worker1")
		_run(["git", "checkout", "-b", "unit/feature-a"], worker)
		(worker / "feature_a.py").write_text("# Feature A\n")
		_run(["git", "add", "feature_a.py"], worker)
		_run(["git", "commit", "-m", "Add feature A"], worker)
		unit_hash = _run(["git", "rev-parse", "HEAD"], worker)

		result = await mgr.merge_unit(str(worker), "unit/feature-a")

		assert result.merged is True
		assert result.rebase_ok is True
		assert result.failure_output == ""

		# Verify mc/green advanced
		green_after = _run(["git", "rev-parse", "mc/green"], workspace)
		assert green_after != green_before

		# Verify it's a merge commit (2 parents)
		parents = _run(["git", "log", "-1", "--format=%P", "mc/green"], workspace)
		parent_list = parents.split()
		assert len(parent_list) == 2
		assert green_before in parent_list
		# The other parent should be the unit commit
		assert unit_hash in parent_list

		# Verify the file exists on mc/green
		_run(["git", "checkout", "mc/green"], workspace)
		assert (workspace / "feature_a.py").exists()

	async def test_rebase_resolves_stale_branch(self, tmp_path: Path) -> None:
		"""Two workers branch from same mc/green, edit different files -- both merge."""
		source, workspace = _setup_source_repo(tmp_path)
		config = _real_config(source)
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)
		mgr.workspace = str(workspace)

		# Worker 1: branch from mc/green, add feature_a.py
		worker1 = _make_worker_clone(tmp_path, source, "worker-rebase-1")
		_run(["git", "checkout", "-b", "unit/rebase-a"], worker1)
		(worker1 / "feature_a.py").write_text("# Feature A\n")
		_run(["git", "add", "feature_a.py"], worker1)
		_run(["git", "commit", "-m", "Add feature A"], worker1)

		# Worker 2: branch from mc/green (same base), add feature_b.py (different file)
		worker2 = _make_worker_clone(tmp_path, source, "worker-rebase-2")
		_run(["git", "checkout", "-b", "unit/rebase-b"], worker2)
		(worker2 / "feature_b.py").write_text("# Feature B\n")
		_run(["git", "add", "feature_b.py"], worker2)
		_run(["git", "commit", "-m", "Add feature B"], worker2)

		# Merge worker 1 -- succeeds, mc/green advances
		result1 = await mgr.merge_unit(str(worker1), "unit/rebase-a")
		assert result1.merged is True

		# Merge worker 2 -- previously would fail, now rebase resolves divergence
		result2 = await mgr.merge_unit(str(worker2), "unit/rebase-b")
		assert result2.merged is True
		assert result2.rebase_ok is True

		# Verify mc/green has both files
		_run(["git", "checkout", "mc/green"], workspace)
		assert (workspace / "feature_a.py").exists()
		assert (workspace / "feature_b.py").exists()

	async def test_merge_conflict_detection_real(self, tmp_path: Path) -> None:
		"""Two workers modifying the same line -- first merges, second gets conflict."""
		source, workspace = _setup_source_repo(tmp_path)
		config = _real_config(source)
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)
		mgr.workspace = str(workspace)

		# Worker 1: modify app.py line 1
		worker1 = _make_worker_clone(tmp_path, source, "worker-conflict-1")
		_run(["git", "checkout", "-b", "unit/change-1"], worker1)
		(worker1 / "app.py").write_text("print('worker 1 was here')\n")
		_run(["git", "add", "app.py"], worker1)
		_run(["git", "commit", "-m", "Worker 1 change"], worker1)

		# Worker 2: modify the same line differently
		worker2 = _make_worker_clone(tmp_path, source, "worker-conflict-2")
		_run(["git", "checkout", "-b", "unit/change-2"], worker2)
		(worker2 / "app.py").write_text("print('worker 2 was here')\n")
		_run(["git", "add", "app.py"], worker2)
		_run(["git", "commit", "-m", "Worker 2 change"], worker2)

		# First merge succeeds
		result1 = await mgr.merge_unit(str(worker1), "unit/change-1")
		assert result1.merged is True

		# Second merge should fail with merge conflict
		result2 = await mgr.merge_unit(str(worker2), "unit/change-2")
		assert result2.merged is False
		assert result2.failure_stage == "merge_conflict"
		assert "conflict" in result2.failure_output.lower()

		# Verify mc/green still has worker 1's content (not corrupted)
		_run(["git", "checkout", "mc/green"], workspace)
		content = (workspace / "app.py").read_text()
		assert "worker 1 was here" in content

	async def test_sync_to_source_updates_source_repo(self, tmp_path: Path) -> None:
		"""After merge_unit, source repo's mc/green matches workspace's mc/green."""
		source, workspace = _setup_source_repo(tmp_path)
		config = _real_config(source)
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)
		mgr.workspace = str(workspace)

		worker = _make_worker_clone(tmp_path, source, "worker-sync")
		_run(["git", "checkout", "-b", "unit/sync-test"], worker)
		(worker / "sync.py").write_text("# sync\n")
		_run(["git", "add", "sync.py"], worker)
		_run(["git", "commit", "-m", "Sync test"], worker)

		await mgr.merge_unit(str(worker), "unit/sync-test")

		# Source repo's mc/green should match workspace's mc/green
		ws_green = _run(["git", "rev-parse", "mc/green"], workspace)
		src_green = _run(["git", "rev-parse", "mc/green"], source)
		assert ws_green == src_green


class TestFixupSessionModel:
	"""Tests for _run_fixup_session passing the correct --model flag."""

	async def test_fixup_uses_fixup_model_from_config(self) -> None:
		"""When config.models.fixup_model is set, it is passed as --model."""
		mgr = _manager()
		# Simulate a ModelsConfig-like object on config
		mgr.config.models = type("ModelsConfig", (), {"fixup_model": "haiku"})()  # type: ignore[attr-defined]
		mgr._run_command = AsyncMock(return_value=(True, "fixed"))  # type: ignore[method-assign]

		await mgr._run_fixup_session("fix the tests")

		cmd = mgr._run_command.call_args[0][0]
		assert "--model" in cmd
		model_idx = cmd.index("--model")
		assert cmd[model_idx + 1] == "haiku"

	async def test_fixup_falls_back_to_scheduler_model(self) -> None:
		"""When config.models.fixup_model is default, uses scheduler.model as fallback."""
		mgr = _manager()
		mgr.config.scheduler.model = "sonnet"
		# Clear the fixup_model so _get_fixup_model falls back to scheduler.model
		mgr.config.models.fixup_model = ""  # type: ignore[attr-defined]
		mgr._run_command = AsyncMock(return_value=(True, "fixed"))  # type: ignore[method-assign]

		await mgr._run_fixup_session("fix the tests")

		cmd = mgr._run_command.call_args[0][0]
		assert "--model" in cmd
		model_idx = cmd.index("--model")
		assert cmd[model_idx + 1] == "sonnet"

	async def test_fixup_falls_back_when_fixup_model_empty(self) -> None:
		"""When config.models exists but fixup_model is empty, falls back to scheduler.model."""
		mgr = _manager()
		mgr.config.scheduler.model = "opus"
		mgr.config.models = type("ModelsConfig", (), {"fixup_model": ""})()  # type: ignore[attr-defined]
		mgr._run_command = AsyncMock(return_value=(True, "fixed"))  # type: ignore[method-assign]

		await mgr._run_fixup_session("fix the tests")

		cmd = mgr._run_command.call_args[0][0]
		model_idx = cmd.index("--model")
		assert cmd[model_idx + 1] == "opus"


def _state_config(target_path: str = "/tmp/test") -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(
		name="test",
		path=target_path,
		branch="main",
		verification=VerificationConfig(command="pytest -q"),
	)
	mc.green_branch = GreenBranchConfig(
		working_branch="mc/working",
		green_branch="mc/green",
	)
	return mc


def _state_manager(target_path: str = "/tmp/test", workspace: str = "/tmp/test-workspace") -> GreenBranchManager:
	config = _state_config(target_path)
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = workspace
	return mgr


# --- MISSION_STATE.md generation tests (from test_mission_state.py) ---


def _insert_unit_with_handoff(
	db: Database,
	*,
	unit_id: str,
	plan_id: str = "p1",
	epoch_id: str = "e1",
	mission_id: str = "m1",
	title: str = "",
	unit_status: str = "completed",
	commit_hash: str | None = None,
	finished_at: str | None = None,
	handoff_status: str = "completed",
	summary: str = "",
	files_changed: list[str] | None = None,
	concerns: list[str] | None = None,
) -> None:
	"""Helper to insert a Plan, WorkUnit, and Handoff with proper FK relationships."""
	# Ensure plan exists (ignore if already inserted)
	try:
		db.insert_plan(Plan(id=plan_id, objective="test"))
	except Exception:
		pass

	unit = WorkUnit(
		id=unit_id, plan_id=plan_id, title=title,
		status=unit_status, commit_hash=commit_hash,
		finished_at=finished_at, epoch_id=epoch_id,
	)
	db.insert_work_unit(unit)

	handoff = Handoff(
		work_unit_id=unit_id, epoch_id=epoch_id,
		status=handoff_status, summary=summary,
		files_changed=files_changed or [],
		concerns=concerns or [],
	)
	db.insert_handoff(handoff)


class TestInitialGeneration:
	"""Test that initial _update_mission_state produces valid markdown."""

	def test_initial_state_has_objective(self, config: MissionConfig, db: Database) -> None:
		"""Initial MISSION_STATE.md contains the objective."""
		mission = Mission(id="m1", objective="Build the widget", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "# Mission State" in content
		assert "Build the widget" in content

	def test_initial_state_has_progress_section(self, config: MissionConfig, db: Database) -> None:
		"""Initial state includes the Progress section."""
		mission = Mission(id="m1", objective="Do stuff", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Progress" in content
		assert "0 tasks complete, 0 failed" in content


class TestCompletionWithTimestamp:
	"""Test that completed units appear in progress counts and files."""

	def test_completed_unit_counted(self, config: MissionConfig, db: Database) -> None:
		"""Completed unit is reflected in progress counts."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		_insert_unit_with_handoff(
			db,
			unit_id="unit1234abcd",
			title="Add feature",
			unit_status="completed",
			commit_hash="abc123",
			finished_at="2024-01-15T10:30:00+00:00",
			summary="Added the feature",
			files_changed=["src/foo.py"],
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "1 tasks complete" in content
		assert "Last completed:" in content
		assert "Add feature" in content

	def test_completed_entry_has_files(self, config: MissionConfig, db: Database) -> None:
		"""Completed entry includes file list in Files Modified section."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		_insert_unit_with_handoff(
			db,
			unit_id="unitabcd1234",
			title="Fix bug",
			unit_status="completed",
			commit_hash="def456",
			finished_at="2024-02-20T14:00:00+00:00",
			summary="Fixed the bug",
			files_changed=["src/bar.py", "tests/test_bar.py"],
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Files Modified" in content
		assert "bar.py" in content
		assert "test_bar.py" in content

	def test_no_changelog_in_new_format(self, config: MissionConfig, db: Database) -> None:
		"""New fixed-size format does not include changelog section."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._state_changelog.append(
			"- 2024-01-15T10:30:00 | unit1234 merged (commit: abc123) -- Added feature"
		)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		# Changelog is no longer part of the fixed-size format
		# (state_changelog is still passed but update_mission_state ignores it now)
		assert "# Mission State" in content


class TestFailureWithTimestamp:
	"""Test that failed units appear in active issues."""

	def test_failed_entry_in_active_issues(self, config: MissionConfig, db: Database) -> None:
		"""Failed handoff appears in Active Issues section."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		_insert_unit_with_handoff(
			db,
			unit_id="failunit1234",
			title="Broken thing",
			unit_status="failed",
			finished_at="2024-03-10T08:45:00+00:00",
			handoff_status="failed",
			summary="It broke",
			concerns=["Merge conflict on models.py"],
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Active Issues" in content
		assert "Merge conflict on models.py" in content

	def test_failed_counted_in_progress(self, config: MissionConfig, db: Database) -> None:
		"""Failed units are counted in progress section."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		_insert_unit_with_handoff(
			db,
			unit_id="failunit1234",
			title="Broken thing",
			unit_status="failed",
			handoff_status="failed",
			summary="It broke",
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "1 failed" in content


class TestProgressCounts:
	"""Test that progress counts are accurate across multiple updates."""

	def test_multiple_units_counted(self, config: MissionConfig, db: Database) -> None:
		"""Multiple units of different statuses are counted correctly."""
		mission = Mission(id="m1", objective="Test accumulation", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		for i in range(3):
			_insert_unit_with_handoff(
				db,
				unit_id=f"unit{i:04d}1234",
				title=f"Task {i}",
				unit_status="completed",
				commit_hash=f"hash{i}",
				finished_at=f"2024-01-15T{10+i}:00:00+00:00",
				summary=f"Feature {i}",
				files_changed=[f"src/mod{i}.py"],
			)
		_insert_unit_with_handoff(
			db,
			unit_id="failunit01234",
			title="Failed task",
			unit_status="failed",
			handoff_status="failed",
			summary="Broke",
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "3 tasks complete, 1 failed" in content

	def test_files_modified_grouped_by_dir(self, config: MissionConfig, db: Database) -> None:
		"""Files are grouped by directory in Files Modified section."""
		mission = Mission(id="m1", objective="Test order", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		_insert_unit_with_handoff(
			db,
			unit_id="unit00011234",
			title="First",
			unit_status="completed",
			files_changed=["src/a.py", "src/b.py", "tests/test_a.py"],
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Files Modified" in content
		assert "src/" in content
		assert "a.py" in content


class TestTimestampFallback:
	"""Test timestamp handling when finished_at is missing."""

	def test_no_timestamp_omits_parenthesized_time(self, config: MissionConfig, db: Database) -> None:
		"""Last completed entry omits timestamp when finished_at is None."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		_insert_unit_with_handoff(
			db,
			unit_id="notime123456",
			title="No time",
			unit_status="completed",
			commit_hash="xyz",
			finished_at=None,
			summary="Done without timestamp",
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "No time" in content
		assert "1 tasks complete" in content
		# No parenthesized timestamp
		assert "()" not in content


# -- Pre-merge verification gate tests --


def _gate_manager() -> GreenBranchManager:
	"""Manager with verify_before_merge enabled."""
	config = _config()
	config.continuous.verify_before_merge = True
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


class TestPreMergeVerificationGate:
	"""Tests for the pre-merge verification gate in merge_unit()."""

	async def test_pre_merge_verification_pass(self) -> None:
		"""When verification passes, merge succeeds."""
		mgr = _gate_manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		passing_report = VerificationReport(
			results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True)],
			raw_output="all passed",
		)
		mgr._run_verification = AsyncMock(return_value=passing_report)  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert result.verification_passed is True
		assert result.verification_report is passing_report
		mgr._run_verification.assert_awaited_once()

	async def test_pre_merge_verification_fail_rollback(self) -> None:
		"""When verification fails, merge is rolled back via git revert."""
		mgr = _gate_manager()
		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)

		failing_report = VerificationReport(
			results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=False)],
			raw_output="2 tests failed",
		)
		mgr._run_verification = AsyncMock(return_value=failing_report)  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is False
		assert result.verification_passed is False
		assert result.failure_stage == "pre_merge_verification"
		assert "2 tests failed" in result.failure_output
		assert result.verification_report is failing_report
		# Verify git revert was called (safe rollback even if HEAD advanced)
		revert_calls = [c for c in git_calls if c[0] == "revert"]
		assert len(revert_calls) >= 1

	async def test_pre_merge_verification_skipped_when_disabled(self) -> None:
		"""When verify_before_merge is False, _run_verification is not called."""
		mgr = _manager()  # verify_before_merge=False by default
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]
		mgr._run_verification = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		mgr._run_verification.assert_not_awaited()


# -- Acceptance criteria tests --


class TestAcceptanceCriteria:
	"""Tests for executable acceptance criteria in merge_unit()."""

	async def test_acceptance_criteria_pass(self) -> None:
		"""When criteria command exits 0, merge succeeds."""
		mgr = _gate_manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		passing_report = VerificationReport(
			results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True)],
			raw_output="ok",
		)
		mgr._run_verification = AsyncMock(return_value=passing_report)  # type: ignore[method-assign]
		mgr._run_acceptance_criteria = AsyncMock(return_value=(True, "ok"))  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch", acceptance_criteria="pytest tests/test_x.py -q")

		assert result.merged is True
		mgr._run_acceptance_criteria.assert_awaited_once_with("pytest tests/test_x.py -q")

	async def test_acceptance_criteria_fail_rollback(self) -> None:
		"""When acceptance criteria fail, merge is rolled back via git revert."""
		mgr = _gate_manager()
		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)

		passing_report = VerificationReport(
			results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True)],
			raw_output="ok",
		)
		mgr._run_verification = AsyncMock(return_value=passing_report)  # type: ignore[method-assign]
		mgr._run_acceptance_criteria = AsyncMock(return_value=(False, "assertion failed"))  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch", acceptance_criteria="python -c 'assert False'")

		assert result.merged is False
		assert result.failure_stage == "acceptance_criteria"
		assert "assertion failed" in result.failure_output
		# Verify git revert was called (safe rollback even if HEAD advanced)
		revert_calls = [c for c in git_calls if c[0] == "revert"]
		assert len(revert_calls) >= 1

	async def test_acceptance_criteria_empty_skips(self) -> None:
		"""When acceptance_criteria is empty, it's skipped."""
		mgr = _gate_manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		passing_report = VerificationReport(
			results=[VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True)],
			raw_output="ok",
		)
		mgr._run_verification = AsyncMock(return_value=passing_report)  # type: ignore[method-assign]
		mgr._run_acceptance_criteria = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch", acceptance_criteria="")

		assert result.merged is True
		mgr._run_acceptance_criteria.assert_not_awaited()

	async def test_acceptance_criteria_timeout(self) -> None:
		"""Acceptance criteria timeout returns failure."""
		mgr = _gate_manager()
		mgr.workspace = "/tmp/test-workspace"

		mock_proc = AsyncMock()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch("mission_control.green_branch.asyncio.create_subprocess_shell", return_value=mock_proc):
			with patch("mission_control.green_branch.asyncio.wait_for", side_effect=asyncio.TimeoutError):
				passed, output = await mgr._run_acceptance_criteria("sleep 999", timeout=1)

		assert passed is False
		assert "timed out" in output


# ---------------------------------------------------------------------------
# Fixup candidate dataclasses
# ---------------------------------------------------------------------------


class TestFixupCandidateDataclass:
	def test_defaults(self) -> None:
		c = FixupCandidate()
		assert c.branch == ""
		assert c.verification_passed is False
		assert c.tests_passed == 0
		assert c.lint_errors == 0
		assert c.diff_lines == 0

	def test_fields_set(self) -> None:
		c = FixupCandidate(
			branch="mc/fixup-candidate-0",
			verification_passed=True,
			tests_passed=42,
			lint_errors=2,
			diff_lines=15,
		)
		assert c.branch == "mc/fixup-candidate-0"
		assert c.verification_passed is True
		assert c.tests_passed == 42
		assert c.lint_errors == 2
		assert c.diff_lines == 15


class TestFixupResultDataclass:
	def test_defaults(self) -> None:
		r = FixupResult()
		assert r.success is False
		assert r.winner is None
		assert r.candidates == []

	def test_with_winner(self) -> None:
		winner = FixupCandidate(
			branch="mc/fixup-candidate-1", verification_passed=True,
		)
		r = FixupResult(success=True, winner=winner, candidates=[winner])
		assert r.success is True
		assert r.winner is winner


# ---------------------------------------------------------------------------
# Diff line counting
# ---------------------------------------------------------------------------


class TestCountDiffLines:
	def test_normal_stat(self) -> None:
		output = " 3 files changed, 10 insertions(+), 5 deletions(-)\n"
		assert GreenBranchManager._count_diff_lines(output) == 15

	def test_insertions_only(self) -> None:
		output = " 1 file changed, 7 insertions(+)\n"
		assert GreenBranchManager._count_diff_lines(output) == 7

	def test_deletions_only(self) -> None:
		output = " 2 files changed, 3 deletions(-)\n"
		assert GreenBranchManager._count_diff_lines(output) == 3

	def test_empty_output(self) -> None:
		assert GreenBranchManager._count_diff_lines("") == 0

	def test_no_changes(self) -> None:
		assert GreenBranchManager._count_diff_lines("no diff\n") == 0

	def test_multiline_stat(self) -> None:
		output = (
			" src/foo.py | 10 +++++++---\n"
			" src/bar.py |  5 ++---\n"
			" 2 files changed, 9 insertions(+), 6 deletions(-)\n"
		)
		assert GreenBranchManager._count_diff_lines(output) == 15


# ---------------------------------------------------------------------------
# N-of-M fixup candidate selection
# ---------------------------------------------------------------------------


class TestRunFixup:
	"""Tests for the N-of-M fixup candidate selection."""

	async def test_best_candidate_selected_by_tests(self) -> None:
		"""Candidate with most tests passing wins."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=True, tests=8, lint=0, diff=10),
			_candidate(1, passed=True, tests=10, lint=0, diff=10),
			_candidate(2, passed=True, tests=5, lint=0, diff=10),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert result.winner is not None
		assert result.winner.tests_passed == 10
		assert result.winner.branch == "mc/fixup-candidate-1"

	async def test_tiebreak_by_lint_errors(self) -> None:
		"""When tests_passed tie, fewer lint errors wins."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=5, diff=10),
			_candidate(1, passed=True, tests=10, lint=2, diff=10),
			_candidate(2, passed=True, tests=10, lint=8, diff=10),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert result.winner is not None
		assert result.winner.lint_errors == 2
		assert result.winner.branch == "mc/fixup-candidate-1"

	async def test_tiebreak_by_diff_size(self) -> None:
		"""When tests and lint tie, smallest diff wins."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=0, diff=50),
			_candidate(1, passed=True, tests=10, lint=0, diff=5),
			_candidate(2, passed=True, tests=10, lint=0, diff=30),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert result.winner is not None
		assert result.winner.diff_lines == 5
		assert result.winner.branch == "mc/fixup-candidate-1"

	async def test_all_candidates_fail(self) -> None:
		"""When all candidates fail verification, result is failure."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=False, tests=3, lint=5, diff=10),
			_candidate(1, passed=False, tests=5, lint=2, diff=8),
			_candidate(2, passed=False, tests=0, lint=10, diff=20),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)

		result = await mgr.run_fixup("some failure")

		assert result.success is False
		assert result.winner is None
		assert len(result.candidates) == 3

	async def test_single_pass(self) -> None:
		"""When only one candidate passes, it wins regardless of metrics."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=False, tests=0, lint=0, diff=0),
			_candidate(1, passed=True, tests=3, lint=5, diff=100),
			_candidate(2, passed=False, tests=0, lint=0, diff=0),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert result.winner is not None
		assert result.winner.branch == "mc/fixup-candidate-1"

	async def test_winner_merged_into_green(self) -> None:
		"""Winning candidate branch is merged into mc/green."""
		mgr = _fixup_manager(fixup_candidates=2)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=0, diff=5),
			_candidate(1, passed=False, tests=3, lint=2, diff=10),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			return (True, "")

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(side_effect=tracking_git)

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		ff_calls = [
			c for c in git_calls
			if c[0] == "merge" and "--ff-only" in c
		]
		assert len(ff_calls) == 1
		assert "mc/fixup-candidate-0" in ff_calls[0]

	async def test_candidate_branches_cleaned_up(self) -> None:
		"""All candidate branches are deleted after fixup."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=0, diff=5),
			_candidate(1, passed=False, tests=3, lint=2, diff=10),
			_candidate(2, passed=False, tests=0, lint=5, diff=20),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			return (True, "")

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(side_effect=tracking_git)

		await mgr.run_fixup("some failure")

		delete_calls = [
			c for c in git_calls if c[0] == "branch" and c[1] == "-D"
		]
		deleted_branches = {c[2] for c in delete_calls}
		assert "mc/fixup-candidate-0" in deleted_branches
		assert "mc/fixup-candidate-1" in deleted_branches
		assert "mc/fixup-candidate-2" in deleted_branches

	async def test_merge_failure_returns_no_winner(self) -> None:
		"""If merging the winner branch fails, result has no winner."""
		mgr = _fixup_manager(fixup_candidates=2)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=0, diff=5),
			_candidate(1, passed=False, tests=3, lint=2, diff=10),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		async def failing_merge(*args: str) -> tuple[bool, str]:
			if args[0] == "merge":
				return (False, "merge failed")
			return (True, "")

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(side_effect=failing_merge)

		result = await mgr.run_fixup("some failure")

		assert result.success is False
		assert result.winner is None

	async def test_configurable_candidate_count(self) -> None:
		"""fixup_candidates config controls how many candidates are spawned."""
		mgr = _fixup_manager(fixup_candidates=5)

		call_count = 0

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			nonlocal call_count
			call_count += 1
			return FixupCandidate(
				branch=f"mc/fixup-candidate-{index}",
				verification_passed=(index == 0),
				tests_passed=10 if index == 0 else 3,
				lint_errors=0,
				diff_lines=5,
			)

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert call_count == 5
		assert len(result.candidates) == 5

	async def test_prompts_cycle_when_n_exceeds_prompts(self) -> None:
		"""When fixup_candidates > len(FIXUP_PROMPTS), prompts padded."""
		mgr = _fixup_manager(fixup_candidates=5)

		prompts_received: list[str] = []

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			prompts_received.append(prompt)
			return FixupCandidate(
				branch=f"mc/fixup-candidate-{index}",
				verification_passed=False,
			)

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)

		await mgr.run_fixup("some failure")

		assert len(prompts_received) == 5
		assert prompts_received[0] == FIXUP_PROMPTS[0]
		assert prompts_received[1] == FIXUP_PROMPTS[1]
		assert prompts_received[2] == FIXUP_PROMPTS[2]
		# Extra candidates get the first prompt
		assert prompts_received[3] == FIXUP_PROMPTS[0]
		assert prompts_received[4] == FIXUP_PROMPTS[0]

	async def test_only_passing_candidates_considered(self) -> None:
		"""Failed candidates never selected even with better metrics."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=False, tests=100, lint=0, diff=1),
			_candidate(1, passed=True, tests=5, lint=3, diff=50),
			_candidate(2, passed=False, tests=50, lint=0, diff=2),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert result.winner is not None
		assert result.winner.branch == "mc/fixup-candidate-1"
		assert result.winner.tests_passed == 5

	async def test_fallback_to_no_ff_merge(self) -> None:
		"""If ff-only merge fails, falls back to --no-ff merge."""
		mgr = _fixup_manager(fixup_candidates=1)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=0, diff=5),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		merge_attempts: list[tuple[str, ...]] = []

		async def selective_git(*args: str) -> tuple[bool, str]:
			if args[0] == "merge":
				merge_attempts.append(args)
				if "--ff-only" in args:
					return (False, "not possible to fast-forward")
				return (True, "")
			return (True, "")

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(side_effect=selective_git)

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert len(merge_attempts) == 2
		assert "--ff-only" in merge_attempts[0]
		assert "--no-ff" in merge_attempts[1]


# ---------------------------------------------------------------------------
# Individual candidate execution
# ---------------------------------------------------------------------------


class TestRunFixupCandidate:
	"""Tests for the individual candidate execution."""

	async def test_candidate_creates_branch_from_green(self) -> None:
		"""Candidate creates branch from mc/green and runs verification."""
		mgr = _fixup_manager()

		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			if args[0] == "diff":
				return (True, " 1 file changed, 3 insertions(+)\n")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)
		mgr._run_fixup_session = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(
			return_value=(True, "10 passed\nAll checks passed"),
		)

		await mgr._run_fixup_candidate(0, "fix it", "failure", "mc/green")

		checkout_calls = [c for c in git_calls if c[0] == "checkout"]
		assert any("mc/green" in c for c in checkout_calls)
		assert any("mc/fixup-candidate-0" in c for c in checkout_calls)

	async def test_candidate_parses_verification_results(self) -> None:
		"""Candidate parses pytest and ruff output from verification."""
		mgr = _fixup_manager()

		mgr._run_git = AsyncMock(
			return_value=(True, " 1 file changed, 5 insertions(+)\n"),
		)
		mgr._run_fixup_session = AsyncMock(return_value=(True, ""))
		verify_output = (
			"42 passed, 3 failed\n"
			"src/foo.py:1:1: E501 line too long\n"
			"src/bar.py:2:3: F401 unused import"
		)
		mgr._run_command = AsyncMock(return_value=(True, verify_output))

		candidate = await mgr._run_fixup_candidate(
			0, "fix it", "failure", "mc/green",
		)

		assert candidate.tests_passed == 42
		assert candidate.lint_errors == 2
		assert candidate.diff_lines == 5

	async def test_candidate_handles_branch_creation_failure(self) -> None:
		"""If branch creation fails, returns empty candidate."""
		mgr = _fixup_manager()

		async def failing_checkout(*args: str) -> tuple[bool, str]:
			if args[0] == "checkout" and "-b" in args:
				return (False, "branch already exists")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=failing_checkout)

		candidate = await mgr._run_fixup_candidate(
			0, "fix it", "failure", "mc/green",
		)

		assert candidate.verification_passed is False
		assert candidate.tests_passed == 0

	async def test_candidate_measures_diff(self) -> None:
		"""Candidate uses git diff --stat to measure diff size."""
		mgr = _fixup_manager()

		diff_output = (
			" src/a.py | 10 +++++-----\n"
			" 2 files changed, 15 insertions(+), 8 deletions(-)\n"
		)

		async def selective_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and "--stat" in args:
				return (True, diff_output)
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=selective_git)
		mgr._run_fixup_session = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(
			return_value=(True, "5 passed\nAll checks passed"),
		)

		candidate = await mgr._run_fixup_candidate(
			0, "fix it", "failure", "mc/green",
		)

		assert candidate.diff_lines == 23


# ---------------------------------------------------------------------------
# Claude Code subprocess spawning for fixup
# ---------------------------------------------------------------------------


class TestRunFixupSession:
	"""Tests for the Claude Code subprocess spawning."""

	async def test_spawns_claude_with_correct_args(self) -> None:
		"""_run_fixup_session calls _run_command with claude CLI args."""
		mgr = _fixup_manager()
		mgr._run_command = AsyncMock(return_value=(True, "done"))

		await mgr._run_fixup_session("fix the bug")

		mgr._run_command.assert_awaited_once()
		cmd = mgr._run_command.call_args[0][0]
		assert cmd[0] == "claude"
		assert "--print" in cmd
		assert "--output-format" in cmd
		assert "text" in cmd
		assert "-p" in cmd
		assert "fix the bug" in cmd


# ---------------------------------------------------------------------------
# ZFC fixup prompts
# ---------------------------------------------------------------------------


class TestZFCFixupPrompts:
	def _zfc_manager(self, zfc_fixup: bool = False) -> GreenBranchManager:
		mgr = _fixup_manager()
		mgr.config.zfc.zfc_fixup_prompts = zfc_fixup
		mgr.config.zfc.llm_timeout = 5
		return mgr

	async def test_zfc_disabled_uses_static_prompts(self) -> None:
		"""toggle off -> FIXUP_PROMPTS used."""
		mgr = self._zfc_manager(zfc_fixup=False)

		mgr._run_fixup_candidate = AsyncMock(return_value=MagicMock(
			verification_passed=False, branch="mc/fixup-0",
		))
		mgr._run_git = AsyncMock(return_value=(True, ""))

		with patch.object(mgr, "_zfc_generate_fixup_strategies") as mock_zfc:
			await mgr.run_fixup("test failure")
			mock_zfc.assert_not_called()

	async def test_zfc_enabled_calls_llm(self) -> None:
		"""toggle on -> _zfc_generate_fixup_strategies called."""
		mgr = self._zfc_manager(zfc_fixup=True)

		mgr._run_fixup_candidate = AsyncMock(return_value=MagicMock(
			verification_passed=False, branch="mc/fixup-0",
		))
		mgr._run_git = AsyncMock(return_value=(True, ""))

		with patch.object(
			mgr, "_zfc_generate_fixup_strategies",
			return_value=["Fix A", "Fix B", "Fix C"],
		) as mock_zfc:
			await mgr.run_fixup("test failure")
			mock_zfc.assert_called_once_with("test failure", 3)

	async def test_zfc_llm_failure_falls_back(self) -> None:
		"""LLM returns None -> static prompts used as fallback."""
		mgr = self._zfc_manager(zfc_fixup=True)

		mgr._run_fixup_candidate = AsyncMock(return_value=MagicMock(
			verification_passed=False, branch="mc/fixup-0",
		))
		mgr._run_git = AsyncMock(return_value=(True, ""))

		with patch.object(mgr, "_zfc_generate_fixup_strategies", return_value=None):
			result = await mgr.run_fixup("test failure")
			assert result is not None

	async def test_zfc_parse_strategies_output(self) -> None:
		"""Correct FIXUP_STRATEGIES marker is parsed."""
		mgr = self._zfc_manager(zfc_fixup=True)

		output = (
			'Some reasoning about fixes.\n'
			'FIXUP_STRATEGIES:{"strategies": ["Fix the import", "Update the mock", "Rewrite test"]}'
		)

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (output.encode(), b"")
		mock_proc.returncode = 0

		with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
			strategies = await mgr._zfc_generate_fixup_strategies("test failure", 3)

		assert strategies is not None
		assert len(strategies) == 3
		assert "Fix the import" in strategies[0]

	async def test_zfc_timeout_returns_none(self) -> None:
		"""Subprocess timeout -> returns None."""
		mgr = self._zfc_manager(zfc_fixup=True)
		mgr.config.zfc.llm_timeout = 0

		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = TimeoutError()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch("asyncio.wait_for", side_effect=TimeoutError()):
				strategies = await mgr._zfc_generate_fixup_strategies("test failure", 3)

		assert strategies is None


# ---------------------------------------------------------------------------
# Track actual merged files via git diff
# ---------------------------------------------------------------------------


class TestMergedFilesFromGitDiff:
	"""UnitMergeResult.changed_files is populated from git diff, not files_hint."""

	async def test_changed_files_populated_on_merge(self) -> None:
		"""Successful merge populates changed_files from git diff --name-only."""
		mgr = _conflict_manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and args[1] == "--name-only":
				return (True, "src/app.py\nsrc/utils.py\n")
			if args[0] == "rev-parse" and args[1] == "HEAD":
				return (True, "abc123\n")
			if args[0] == "rev-list" and "--count" in args:
				return (True, "1")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(True, ""))  # type: ignore[method-assign]
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert result.changed_files == ["src/app.py", "src/utils.py"]

	async def test_changed_files_empty_when_diff_empty(self) -> None:
		"""If git diff returns empty output, changed_files is empty list."""
		mgr = _conflict_manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and args[1] == "--name-only":
				return (True, "")
			if args[0] == "rev-list" and "--count" in args:
				return (True, "1")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(True, ""))  # type: ignore[method-assign]
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert result.changed_files == []

	async def test_controller_uses_changed_files_not_files_hint(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Controller tracks actual changed files, not files_hint."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		# Mock green branch manager to return changed_files different from files_hint
		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
				changed_files=["src/app.py"],  # actual diff
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, ""))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
			files_hint="src/app.py, src/index.html",  # declared scope includes index.html
		)
		db.insert_work_unit(unit)

		handoff = Handoff(
			work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="completed", summary="Done",
		)

		completion = WorkerCompletion(
			unit=unit, handoff=handoff, workspace="/tmp/ws", epoch=epoch,
		)

		await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		# _merged_files should contain only app.py (from git diff), not index.html
		assert "src/app.py" in ctrl._merged_files
		assert "src/index.html" not in ctrl._merged_files

	async def test_controller_falls_back_to_files_hint(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When changed_files is empty, controller falls back to files_hint."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
				changed_files=[],  # empty -- fallback to files_hint
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, ""))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu2", plan_id="p1", title="Task",
			status="completed", commit_hash="def456",
			branch_name="mc/unit-wu2",
			files_hint="src/models.py",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		assert "src/models.py" in ctrl._merged_files


# ---------------------------------------------------------------------------
# Locked files in planner prompt
# ---------------------------------------------------------------------------


class TestLockedFilesInPlannerPrompt:
	"""Locked files section is injected into the planner prompt."""

	@pytest.mark.asyncio
	async def test_locked_files_forwarded_to_planner(self) -> None:
		"""ContinuousPlanner forwards locked_files to RecursivePlanner.plan_round()."""
		config = _conflict_config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		captured_kwargs: dict = {}

		async def mock_plan_round(**kwargs):
			captured_kwargs.update(kwargs)
			plan = Plan(objective="test")
			return plan, []

		planner._inner.plan_round = mock_plan_round  # type: ignore[assignment]

		locked = {"src/app.py": ["already merged"], "src/api.py": ["in-flight: Build API"]}
		mission = Mission(id="m1", objective="test")

		await planner.get_next_units(mission, locked_files=locked)

		assert captured_kwargs.get("locked_files") == locked

	@pytest.mark.asyncio
	async def test_locked_section_in_prompt(self) -> None:
		"""RecursivePlanner injects ## Locked Files section into the LLM prompt."""
		from mission_control.recursive_planner import PlannerResult, RecursivePlanner

		config = _conflict_config()
		db = Database(":memory:")
		rp = RecursivePlanner(config, db)

		captured_prompt = ""

		async def mock_subprocess(prompt: str) -> PlannerResult:
			nonlocal captured_prompt
			captured_prompt = prompt
			return PlannerResult(type="leaves", units=[])

		rp._run_planner_subprocess = mock_subprocess  # type: ignore[assignment]

		locked = {"src/app.py": ["already merged"], "src/api.py": ["in-flight: Build API"]}
		await rp.plan_round(
			objective="test",
			round_number=1,
			locked_files=locked,
		)

		assert "## Locked Files" in captured_prompt
		assert "src/app.py (already merged)" in captured_prompt
		assert "src/api.py (in-flight: Build API)" in captured_prompt
		assert "AUTOMATICALLY DROPPED" in captured_prompt

	@pytest.mark.asyncio
	async def test_no_locked_section_when_empty(self) -> None:
		"""No ## Locked Files section when locked_files is empty."""
		from mission_control.recursive_planner import PlannerResult, RecursivePlanner

		config = _conflict_config()
		db = Database(":memory:")
		rp = RecursivePlanner(config, db)

		captured_prompt = ""

		async def mock_subprocess(prompt: str) -> PlannerResult:
			nonlocal captured_prompt
			captured_prompt = prompt
			return PlannerResult(type="leaves", units=[])

		rp._run_planner_subprocess = mock_subprocess  # type: ignore[assignment]

		await rp.plan_round(
			objective="test",
			round_number=1,
			locked_files={},
		)

		assert "## Locked Files" not in captured_prompt


# ---------------------------------------------------------------------------
# Changed files from real git integration
# ---------------------------------------------------------------------------


class TestChangedFilesRealGit:
	"""Integration: changed_files populated from actual git diff."""

	async def test_changed_files_from_real_merge(self, tmp_path: Path) -> None:
		"""changed_files reflects actual files changed, not declared scope."""
		source, workspace = _setup_source_repo(tmp_path)
		config = MissionConfig()
		config.target = TargetConfig(
			name="test", path=str(source), branch="main",
			verification=VerificationConfig(command="true"),
		)
		config.green_branch = GreenBranchConfig(
			working_branch="mc/working",
			green_branch="mc/green",
			reset_on_init=False,
		)

		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)
		mgr.workspace = str(workspace)

		# Create worker that only touches feature_a.py
		worker = tmp_path / "worker"
		_run(["git", "clone", str(source), str(worker)], tmp_path)
		_run(["git", "config", "user.email", "w@test.com"], worker)
		_run(["git", "config", "user.name", "Worker"], worker)
		_run(["git", "checkout", "-b", "unit/feature-a"], worker)
		(worker / "feature_a.py").write_text("# Feature A\n")
		_run(["git", "add", "feature_a.py"], worker)
		_run(["git", "commit", "-m", "Add feature A"], worker)

		result = await mgr.merge_unit(str(worker), "unit/feature-a")

		assert result.merged is True
		assert "feature_a.py" in result.changed_files
		# Should NOT include files not actually changed
		assert "app.py" not in result.changed_files
		assert "README.md" not in result.changed_files


# ---------------------------------------------------------------------------
# Reconciler sweep
# ---------------------------------------------------------------------------


class TestReconciliationCheck:
	"""GreenBranchManager.run_reconciliation_check() runs verification on green."""

	async def test_reconciliation_check_passes(self) -> None:
		mgr = _conflict_manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(True, "all tests passed"))  # type: ignore[method-assign]

		ok, output = await mgr.run_reconciliation_check()

		assert ok is True
		assert output == "all tests passed"
		# Should checkout mc/green before running verification
		mgr._run_git.assert_any_call("checkout", "mc/green")

	async def test_reconciliation_check_fails(self) -> None:
		mgr = _conflict_manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(False, "2 tests failed"))  # type: ignore[method-assign]

		ok, output = await mgr.run_reconciliation_check()

		assert ok is False
		assert "failed" in output


class TestReconcilerSweep:
	"""Controller triggers reconciler after merges."""

	@pytest.mark.asyncio
	async def test_reconciler_runs_after_merge(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Reconciler sweep fires after a successful merge."""
		config.continuous.verify_before_merge = False
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, "all pass"))
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

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		mock_gbm.run_reconciliation_check.assert_awaited_once()

	@pytest.mark.asyncio
	async def test_reconciler_triggers_fixup_on_failure(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When reconciler verification fails, run_fixup is called."""
		config.continuous.verify_before_merge = False
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(
			return_value=(False, "ImportError: cannot import name 'foo'"),
		)
		mock_gbm.run_fixup = AsyncMock(
			return_value=FixupResult(success=True),
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
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)

		await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		mock_gbm.run_fixup.assert_awaited_once()
		# Fixup should receive the failure output
		call_args = mock_gbm.run_fixup.call_args
		assert "ImportError" in call_args[0][0]

	@pytest.mark.asyncio
	async def test_reconciler_skipped_when_no_new_merges(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Reconciler doesn't re-run if no new merges since last check."""
		config.continuous.verify_before_merge = False
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, ""))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		# Process first merge
		unit1 = WorkUnit(
			id="wu1", plan_id="p1", title="Task 1",
			status="completed", commit_hash="abc", branch_name="mc/unit-wu1",
		)
		db.insert_work_unit(unit1)
		completion1 = WorkerCompletion(unit=unit1, handoff=None, workspace="/tmp/ws", epoch=epoch)

		# Process a failed unit (no merge)
		unit2 = WorkUnit(
			id="wu2", plan_id="p1", title="Task 2",
			status="failed",
		)
		db.insert_work_unit(unit2)
		completion2 = WorkerCompletion(unit=unit2, handoff=None, workspace="/tmp/ws", epoch=epoch)

		await ctrl._process_single_completion(completion1, Mission(id="m1"), result)
		await ctrl._process_single_completion(completion2, Mission(id="m1"), result)

		# Reconciler should have run once (after first merge) but not again for the failure
		assert mock_gbm.run_reconciliation_check.await_count == 1

	@pytest.mark.asyncio
	async def test_reconciler_runs_periodically_with_verify_before_merge(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Reconciler fires every reconcile_interval merges even when verify_before_merge=True."""
		config.continuous.verify_before_merge = True
		config.continuous.reconcile_interval = 3
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, "all pass"))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		# Process 3 successful merges; reconciler should fire on the 3rd
		for i in range(3):
			unit = WorkUnit(
				id=f"wu{i}", plan_id="p1", title=f"Task {i}",
				status="completed", commit_hash=f"abc{i}",
				branch_name=f"mc/unit-wu{i}",
			)
			db.insert_work_unit(unit)
			completion = WorkerCompletion(
				unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
			)
			await ctrl._process_single_completion(completion, Mission(id="m1"), result)

		# With verify_before_merge=True, reconciler only fires at interval (3rd merge)
		assert mock_gbm.run_reconciliation_check.await_count == 1


def _traced_manager(tmp_path: Path) -> tuple[GreenBranchManager, Path]:
	"""Create a GreenBranchManager with an enabled TraceLogger backed by a temp file."""
	config = _config()
	trace_file = tmp_path / "trace.jsonl"
	trace_cfg = TraceLogConfig(enabled=True, path=str(trace_file))
	trace_logger = TraceLogger(trace_cfg)
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db, trace_logger=trace_logger)
	mgr.workspace = "/tmp/test-workspace"
	return mgr, trace_file


def _read_trace_events(trace_file: Path) -> list[dict]:
	import json
	lines = trace_file.read_text().strip().splitlines()
	return [json.loads(line) for line in lines]


class TestTraceLogging:
	"""Verify TraceLogger integration in GreenBranchManager."""

	async def test_merge_unit_emits_rebase_events(self, tmp_path: Path) -> None:
		"""Successful merge emits rebase_attempted and rebase_result events."""
		mgr, trace_file = _traced_manager(tmp_path)
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/traced")
		assert result.merged is True

		events = _read_trace_events(trace_file)
		event_types = [e["event_type"] for e in events]
		assert "rebase_attempted" in event_types
		assert "rebase_result" in event_types

		rebase_attempted = next(e for e in events if e["event_type"] == "rebase_attempted")
		assert rebase_attempted["details"]["branch"] == "feat/traced"

		rebase_result = next(e for e in events if e["event_type"] == "rebase_result")
		assert rebase_result["details"]["success"] is True
		assert rebase_result["details"]["branch"] == "feat/traced"

	async def test_merge_rebase_failure_emits_rebase_result_false(self, tmp_path: Path) -> None:
		"""Rebase failure emits rebase_result with success=False."""
		mgr, trace_file = _traced_manager(tmp_path)

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "rebase":
				return (False, "CONFLICT")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)

		result = await mgr.merge_unit("/tmp/worker", "feat/conflict")
		assert result.merged is False

		events = _read_trace_events(trace_file)
		rebase_result = next(e for e in events if e["event_type"] == "rebase_result")
		assert rebase_result["details"]["success"] is False

	async def test_sync_to_source_emits_git_push(self, tmp_path: Path) -> None:
		"""Successful sync emits git_push events for green and working branches."""
		mgr, trace_file = _traced_manager(tmp_path)

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			return (True, "")

		mgr._run_git_in = mock_run_git_in  # type: ignore[assignment]

		result = await mgr._sync_to_source()
		assert result is True

		events = _read_trace_events(trace_file)
		push_events = [e for e in events if e["event_type"] == "git_push"]
		assert len(push_events) == 2
		targets = {e["details"]["target"] for e in push_events}
		assert targets == {"source"}

	async def test_push_green_to_main_emits_git_push(self, tmp_path: Path) -> None:
		"""Successful push emits git_push event with target=origin."""
		mgr, trace_file = _traced_manager(tmp_path)
		mgr.config.green_branch.auto_push = True
		mgr.config.green_branch.push_branch = "main"

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			return (True, "")

		mgr._run_git_in = mock_run_git_in  # type: ignore[assignment]

		result = await mgr.push_green_to_main()
		assert result is True

		events = _read_trace_events(trace_file)
		push_events = [e for e in events if e["event_type"] == "git_push"]
		assert len(push_events) >= 1
		origin_push = next(e for e in push_events if e["details"]["target"] == "origin")
		assert origin_push["details"]["branch"] == "main"

	async def test_no_trace_when_logger_is_none(self) -> None:
		"""Without a trace logger, merge_unit works normally with no trace output."""
		mgr = _manager()  # No trace logger
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/no-trace")
		assert result.merged is True
		assert mgr._trace_logger is None

	async def test_initialize_emits_branch_created(self, tmp_path: Path) -> None:
		"""Branch creation during initialize() emits branch_created events."""
		mgr, trace_file = _traced_manager(tmp_path)
		mgr.config.green_branch.reset_on_init = False

		async def mock_run_git_in(cwd: str, *args: str) -> tuple[bool, str]:
			# rev-parse --verify returns False (branch doesn't exist)
			if args[0] == "rev-parse" and args[1] == "--verify":
				return (False, "")
			return (True, "")

		async def mock_run_git(*args: str) -> tuple[bool, str]:
			if args[0] == "rev-parse" and args[1] == "--verify":
				return (False, "")
			return (True, "")

		mgr._run_git_in = mock_run_git_in  # type: ignore[assignment]
		mgr._run_git = AsyncMock(side_effect=mock_run_git)

		await mgr.initialize("/tmp/workspace")

		events = _read_trace_events(trace_file)
		created_events = [e for e in events if e["event_type"] == "branch_created"]
		# Should have at least 2 branch_created events (source + workspace for each branch)
		assert len(created_events) >= 2
		repos = [e["details"]["repo"] for e in created_events]
		assert "source" in repos
		assert "workspace" in repos
