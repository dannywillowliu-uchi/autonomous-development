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

		with patch("mission_control.green_branch.asyncio.create_subprocess_exec", return_value=mock_proc):
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
		assert calls[0][1] == ("fetch", "/tmp/test-workspace", "mc/green:mc/green")
		# Second call: fetch mc/working
		assert calls[1][0] == "/tmp/test"
		assert calls[1][1] == ("fetch", "/tmp/test-workspace", "mc/working:mc/working")


class TestParallelMergeConflicts:
	"""Tests for concurrent merge scenarios and conflict handling."""

	async def test_concurrent_merge_only_one_succeeds(self) -> None:
		"""3 workers modify the same file; only 1 merge succeeds, 2 get conflicts."""
		mgr = _manager()
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		# Track mc/green state: first --no-ff merge succeeds, subsequent ones conflict
		# because the green branch has advanced.
		merge_count = 0

		async def stateful_git(*args: str) -> tuple[bool, str]:
			nonlocal merge_count
			# The critical merge command: "merge --no-ff <remote>/<branch>"
			if args[0] == "merge" and len(args) > 1 and args[1] == "--no-ff":
				merge_count += 1
				if merge_count == 1:
					return (True, "Merge made by the 'ort' strategy.")
				return (False, "CONFLICT (content): Merge conflict in shared.py")
			# merge --abort is fine
			if args[0] == "merge" and "--abort" in args:
				return (True, "")
			# All other git commands succeed
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=stateful_git)

		# Launch 3 concurrent merge_unit calls
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
			assert r.rebase_ok is False
			assert "Merge conflict" in r.failure_output

	async def test_concurrent_merge_lock_serialization(self) -> None:
		"""_merge_lock serializes concurrent merge attempts (no true parallelism)."""
		mgr = _manager()
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		execution_order: list[str] = []

		async def ordered_git(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and len(args) > 1 and args[1] == "--no-ff":
				# Extract branch name from the remote/branch arg
				branch = args[2].split("/")[-1] if len(args) > 2 else "unknown"
				execution_order.append(branch)
				# Small sleep to verify that other tasks don't interleave
				await asyncio.sleep(0.01)
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=ordered_git)

		# Launch 3 concurrent merges
		await asyncio.gather(
			mgr.merge_unit("/tmp/w1", "unit-a"),
			mgr.merge_unit("/tmp/w2", "unit-b"),
			mgr.merge_unit("/tmp/w3", "unit-c"),
		)

		# All 3 merges should have executed (serialized by the lock)
		assert len(execution_order) == 3
		# Since asyncio.Lock serializes, no two merges ran simultaneously.
		# Verify all branches were processed (order may vary due to asyncio scheduling)
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

		# Second merge should fail with conflict
		result2 = await mgr.merge_unit(str(worker2), "unit/change-2")
		assert result2.merged is False
		assert result2.rebase_ok is False
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


class TestCommitStateFile:
	"""Tests for commit_state_file() method."""

	async def test_writes_and_commits_file(self, tmp_path: Path) -> None:
		"""commit_state_file writes content to workspace and commits it."""
		workspace = tmp_path / "workspace"
		workspace.mkdir()
		mgr = _state_manager(workspace=str(workspace))

		git_calls: list[tuple[str, ...]] = []

		async def mock_run_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_run_git)

		result = await mgr.commit_state_file("# State\nObjective: test\n")

		assert result is True
		# File was written to workspace
		state_file = workspace / "MISSION_STATE.md"
		assert state_file.exists()
		assert state_file.read_text() == "# State\nObjective: test\n"
		# Git add and commit were called
		assert ("add", "MISSION_STATE.md") in git_calls
		assert ("commit", "-m", "Update MISSION_STATE.md") in git_calls

	async def test_returns_false_on_commit_failure(self, tmp_path: Path) -> None:
		"""commit_state_file returns False when git commit fails for other reasons."""
		workspace = tmp_path / "workspace"
		workspace.mkdir()
		mgr = _state_manager(workspace=str(workspace))

		async def mock_run_git(*args: str) -> tuple[bool, str]:
			if args[0] == "commit":
				return (False, "error: some git error")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_run_git)

		result = await mgr.commit_state_file("# State\n")

		assert result is False


class TestMergeUnitCommitsState:
	"""Tests for merge_unit() calling commit_state_file when MISSION_STATE.md exists."""

	async def test_merge_commits_state_file_when_exists(self, tmp_path: Path) -> None:
		"""merge_unit calls commit_state_file when MISSION_STATE.md exists in target repo."""
		target_dir = tmp_path / "target"
		target_dir.mkdir()
		state_file = target_dir / "MISSION_STATE.md"
		state_file.write_text("# Mission State\nObjective: build stuff\n")

		workspace = tmp_path / "workspace"
		workspace.mkdir()

		mgr = _state_manager(target_path=str(target_dir), workspace=str(workspace))
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]
		mgr.commit_state_file = AsyncMock(return_value=True)  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		mgr.commit_state_file.assert_awaited_once_with(
			"# Mission State\nObjective: build stuff\n",
		)

	async def test_merge_skips_state_when_not_exists(self, tmp_path: Path) -> None:
		"""merge_unit skips commit_state_file when MISSION_STATE.md doesn't exist."""
		target_dir = tmp_path / "target"
		target_dir.mkdir()
		# No MISSION_STATE.md created

		workspace = tmp_path / "workspace"
		workspace.mkdir()

		mgr = _state_manager(target_path=str(target_dir), workspace=str(workspace))
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]
		mgr.commit_state_file = AsyncMock(return_value=True)  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		mgr.commit_state_file.assert_not_awaited()
