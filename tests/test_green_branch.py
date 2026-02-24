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
	VerificationConfig,
)
from mission_control.continuous_controller import ContinuousController
from mission_control.db import Database
from mission_control.green_branch import GreenBranchManager
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

		merge_calls = [c for c in git_calls if c[0] == "merge" and "--no-ff" in c]
		assert len(merge_calls) == 1
		assert "worker-feat/branch/feat/branch" in merge_calls[0][2]

	async def test_auto_push(self) -> None:
		mgr = _manager()
		mgr.config.green_branch.auto_push = True
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


	async def test_merge_conflict_resolved_by_llm(self) -> None:
		"""When conflict resolution succeeds, result.merged=True and conflict_resolved=True."""
		mgr = _manager()

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and "--no-ff" in args:
				return (False, "CONFLICT (content): Merge conflict in file.py")
			if args[0] == "rev-parse" and args[1] == "HEAD":
				return (True, "abc123\n")
			if args[0] == "diff" and args[1] == "--name-only" and "HEAD~1" in args:
				return (True, "file.py\n")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)
		mgr._resolve_merge_conflict = AsyncMock(return_value=(True, "resolved"))  # type: ignore[method-assign]
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/conflict")

		assert result.merged is True
		assert result.conflict_resolved is True
		mgr._resolve_merge_conflict.assert_awaited_once()

	async def test_merge_conflict_resolution_fails_falls_back(self) -> None:
		"""When conflict resolution fails, merge aborts as before."""
		mgr = _manager()

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and "--no-ff" in args:
				return (False, "CONFLICT (content): Merge conflict in file.py")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)
		mgr._resolve_merge_conflict = AsyncMock(return_value=(False, "failed"))  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/conflict")

		assert result.merged is False
		assert result.failure_stage == "merge_conflict"

	async def test_merge_conflict_resolution_disabled(self) -> None:
		"""When conflict_resolution is disabled, skips resolution attempt."""
		mgr = _manager()
		mgr.config.green_branch.conflict_resolution = False

		async def side_effect(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and "--no-ff" in args:
				return (False, "CONFLICT (content): Merge conflict in file.py")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=side_effect)
		mgr._resolve_merge_conflict = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/conflict")

		assert result.merged is False
		assert result.failure_stage == "merge_conflict"
		mgr._resolve_merge_conflict.assert_not_awaited()


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
		assert set(execution_order) == {"unit-a", "unit-b", "unit-c"}


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
		"""When auto_push and on_auto_push, deploy runs after push."""
		config = self._deploy_config()
		config.green_branch.auto_push = True
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

	def test_initial_state_has_remaining_section(self, config: MissionConfig, db: Database) -> None:
		"""Initial state includes the Remaining section for the planner."""
		mission = Mission(id="m1", objective="Do stuff", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Remaining" in content
		assert "planner should focus" in content


class TestCompletionWithTimestamp:
	"""Test that completed units get timestamped entries."""

	def test_completed_entry_has_timestamp(self, config: MissionConfig, db: Database) -> None:
		"""Completed handoff entry includes the work unit's finished_at timestamp."""
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
		assert "## Completed" in content
		assert "unit1234" in content
		assert "2024-01-15T10:30:00" in content
		assert "Added the feature" in content

	def test_completed_entry_has_files(self, config: MissionConfig, db: Database) -> None:
		"""Completed entry includes file list."""
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
		assert "src/bar.py" in content
		assert "tests/test_bar.py" in content

	def test_changelog_entry_for_merged_unit(self, config: MissionConfig, db: Database) -> None:
		"""Changelog section appears when entries have been appended."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._state_changelog.append(
			"- 2024-01-15T10:30:00 | unit1234 merged (commit: abc123) -- Added feature"
		)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Changelog" in content
		assert "unit1234 merged (commit: abc123)" in content
		assert "2024-01-15T10:30:00" in content


class TestFailureWithTimestamp:
	"""Test that failed units get timestamped entries."""

	def test_failed_entry_has_timestamp(self, config: MissionConfig, db: Database) -> None:
		"""Failed handoff entry includes the work unit's finished_at timestamp."""
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
		assert "## Failed" in content
		assert "failunit" in content
		assert "2024-03-10T08:45:00" in content
		assert "Merge conflict on models.py" in content

	def test_changelog_entry_for_failed_unit(self, config: MissionConfig, db: Database) -> None:
		"""Changelog includes failed unit entries."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._state_changelog.append(
			"- 2024-03-10T08:45:00 | failunit failed -- It broke"
		)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Changelog" in content
		assert "failunit failed" in content


class TestChangelogAccumulation:
	"""Test that changelog accumulates entries across multiple updates."""

	def test_changelog_accumulates_multiple_entries(self, config: MissionConfig, db: Database) -> None:
		"""Multiple changelog entries persist across _update_mission_state calls."""
		mission = Mission(id="m1", objective="Test accumulation", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)

		# First update
		ctrl._state_changelog.append(
			"- 2024-01-15T10:00:00 | unit0001 merged (commit: aaa111) -- First feature"
		)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Changelog" in content
		assert "unit0001 merged" in content

		# Second update
		ctrl._state_changelog.append(
			"- 2024-01-15T11:00:00 | unit0002 merged (commit: bbb222) -- Second feature"
		)
		ctrl._update_mission_state(mission)

		content = state_path.read_text()
		assert "unit0001 merged" in content
		assert "unit0002 merged" in content

		# Third update (failure)
		ctrl._state_changelog.append(
			"- 2024-01-15T12:00:00 | unit0003 failed -- Merge conflict"
		)
		ctrl._update_mission_state(mission)

		content = state_path.read_text()
		assert "unit0001 merged" in content
		assert "unit0002 merged" in content
		assert "unit0003 failed" in content

	def test_changelog_preserves_order(self, config: MissionConfig, db: Database) -> None:
		"""Changelog entries appear in chronological order (insertion order)."""
		mission = Mission(id="m1", objective="Test order", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._state_changelog.append("- 2024-01-15T10:00:00 | first entry")
		ctrl._state_changelog.append("- 2024-01-15T11:00:00 | second entry")
		ctrl._state_changelog.append("- 2024-01-15T12:00:00 | third entry")
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		first_pos = content.index("first entry")
		second_pos = content.index("second entry")
		third_pos = content.index("third entry")
		assert first_pos < second_pos < third_pos


class TestTimestampFallback:
	"""Test timestamp handling when finished_at is missing."""

	def test_no_timestamp_when_finished_at_is_none(self, config: MissionConfig, db: Database) -> None:
		"""Entry omits timestamp when work unit's finished_at is None."""
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
		assert "notime12" in content
		assert "Done without timestamp" in content
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
		"""When verification fails, merge is rolled back."""
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
		# Verify git reset was called
		reset_calls = [c for c in git_calls if c[0] == "reset" and "--hard" in c]
		assert len(reset_calls) >= 1

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
		"""When criteria command exits non-zero, merge is rolled back."""
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
		# Verify git reset was called
		reset_calls = [c for c in git_calls if c[0] == "reset" and "--hard" in c]
		assert len(reset_calls) >= 1

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


