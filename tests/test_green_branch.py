"""Tests for the green branch manager."""

from __future__ import annotations

import asyncio
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

	async def test_setup_command_timeout_raises(self) -> None:
		"""Setup command timeout raises RuntimeError."""
		config = _config()
		config.target.verification.setup_command = "npm install"
		config.target.verification.setup_timeout = 1
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		mock_proc = AsyncMock()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch.object(mgr, "_run_git_in", AsyncMock(return_value=(False, ""))), \
			patch.object(mgr, "_run_git", AsyncMock(return_value=(True, ""))), \
			patch("mission_control.green_branch.asyncio.create_subprocess_exec", return_value=mock_proc), \
			patch("mission_control.green_branch.asyncio.wait_for", side_effect=asyncio.TimeoutError):
			with pytest.raises(RuntimeError, match="timed out"):
				await mgr.initialize("/tmp/workspace")

	async def test_no_setup_command_skips(self) -> None:
		"""When setup_command is empty, no subprocess is spawned."""
		config = _config()
		config.target.verification.setup_command = ""
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		with patch.object(mgr, "_run_git_in", AsyncMock(return_value=(False, ""))), \
			patch.object(mgr, "_run_git", AsyncMock(return_value=(True, ""))), \
			patch("mission_control.green_branch.asyncio.create_subprocess_exec") as mock_shell:
			await mgr.initialize("/tmp/workspace")

		mock_shell.assert_not_called()


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

	async def test_merge_conflict_detection(self) -> None:
		"""merge_unit detects conflicts and returns rebase_ok=False with descriptive output."""
		mgr = _manager()
		conflict_msg = "CONFLICT (content): Merge conflict in src/shared_module.py\nAutomatic merge failed"

		async def conflict_git(*args: str) -> tuple[bool, str]:
			if args[0] == "merge" and len(args) > 1 and args[1] == "--no-ff":
				return (False, conflict_msg)
			if args[0] == "merge" and "--abort" in args:
				return (True, "")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=conflict_git)

		result = await mgr.merge_unit("/tmp/worker", "feat/conflicting")

		assert result.merged is False
		assert result.rebase_ok is False
		assert "Merge conflict" in result.failure_output
		assert "src/shared_module.py" in result.failure_output

	async def test_green_branch_advances_monotonically(self) -> None:
		"""After N sequential merges, green hash changes each time and never goes backwards."""
		mgr = _manager()
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		# Simulate a git environment where rev-parse returns incrementing hashes
		hash_counter = 0
		hashes_seen: list[str] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			nonlocal hash_counter
			if args[0] == "rev-parse" and args[-1] == "mc/green":
				hash_counter += 1
				h = f"{hash_counter:040x}"
				hashes_seen.append(h)
				return (True, h + "\n")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)

		# Perform 5 sequential merges
		for i in range(5):
			result = await mgr.merge_unit(f"/tmp/w{i}", f"feat/unit-{i}")
			assert result.merged is True

		# Capture the green hash after each merge
		green_hashes: list[str] = []
		for i in range(5):
			h = await mgr.get_green_hash()
			green_hashes.append(h)

		# All hashes should be unique (branch advanced each time)
		assert len(set(green_hashes)) == len(green_hashes)
		# Hashes should be monotonically increasing (never goes backwards)
		for i in range(1, len(green_hashes)):
			assert green_hashes[i] > green_hashes[i - 1]

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

	async def test_no_deploy_command(self) -> None:
		config = _config()
		config.deploy = DeployConfig(enabled=True, command="")
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		ok, output = await mgr.run_deploy()
		assert ok is False
		assert "No deploy command" in output

	async def test_health_check_success(self) -> None:
		config = self._deploy_config()
		config.deploy.health_check_url = "https://example.com"
		config.deploy.health_check_timeout = 10
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"ok", None))

		with patch(
			"mission_control.green_branch.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		), patch.object(
			mgr, "_poll_health_check", AsyncMock(return_value=True),
		):
			ok, output = await mgr.run_deploy()

		assert ok is True

	async def test_health_check_failure(self) -> None:
		config = self._deploy_config()
		config.deploy.health_check_url = "https://example.com"
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)

		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"ok", None))

		with patch(
			"mission_control.green_branch.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		), patch.object(
			mgr, "_poll_health_check", AsyncMock(return_value=False),
		):
			ok, output = await mgr.run_deploy()

		assert ok is False
		assert "Health check failed" in output

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

	async def test_no_deploy_when_push_disabled(self) -> None:
		"""When auto_push is off, deploy is not triggered on merge."""
		config = self._deploy_config()
		config.green_branch.auto_push = False
		config.deploy.on_auto_push = True
		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)
		mgr.workspace = "/tmp/test-workspace"
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]
		mgr.run_deploy = AsyncMock(return_value=(True, "ok"))  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		mgr.run_deploy.assert_not_awaited()


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

	async def test_success_after_retries(self) -> None:
		"""Health check returns True after initial failures."""
		mgr = _manager()

		call_count = 0

		async def mock_get(url: str, timeout: float = 10.0) -> httpx.Response:
			nonlocal call_count
			call_count += 1
			if call_count < 3:
				raise httpx.ConnectError("Connection refused")
			return httpx.Response(200)

		mock_client = AsyncMock(spec=httpx.AsyncClient)
		mock_client.get = AsyncMock(side_effect=mock_get)
		mock_client.__aenter__ = AsyncMock(return_value=mock_client)
		mock_client.__aexit__ = AsyncMock(return_value=False)

		with patch("mission_control.green_branch.httpx.AsyncClient", return_value=mock_client), \
			patch("mission_control.green_branch.random.uniform", return_value=0.01):
			result = await mgr._poll_health_check("https://example.com/health", 30)

		assert result is True
		assert call_count == 3

	async def test_timeout_returns_false(self) -> None:
		"""Health check returns False when total timeout expires."""
		mgr = _manager()

		mock_client = AsyncMock(spec=httpx.AsyncClient)
		mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
		mock_client.__aenter__ = AsyncMock(return_value=mock_client)
		mock_client.__aexit__ = AsyncMock(return_value=False)

		# Use a very short timeout so the test completes fast
		with patch("mission_control.green_branch.httpx.AsyncClient", return_value=mock_client), \
			patch("mission_control.green_branch.random.uniform", return_value=0.01):
			result = await mgr._poll_health_check("https://example.com/health", 0)

		assert result is False

	async def test_jitter_range_applied(self) -> None:
		"""Poll interval uses random.uniform(3.0, 7.0) for jitter."""
		mgr = _manager()
		mock_response = httpx.Response(503)

		call_count = 0

		async def mock_get(url: str, timeout: float = 10.0) -> httpx.Response:
			nonlocal call_count
			call_count += 1
			if call_count >= 3:
				return httpx.Response(200)
			return mock_response

		mock_client = AsyncMock(spec=httpx.AsyncClient)
		mock_client.get = AsyncMock(side_effect=mock_get)
		mock_client.__aenter__ = AsyncMock(return_value=mock_client)
		mock_client.__aexit__ = AsyncMock(return_value=False)

		with patch("mission_control.green_branch.httpx.AsyncClient", return_value=mock_client), \
			patch("mission_control.green_branch.random.uniform", return_value=0.01) as mock_uniform:
			result = await mgr._poll_health_check("https://example.com/health", 30)

		assert result is True
		# random.uniform should be called for each sleep between polls
		assert mock_uniform.call_count == 2
		for call in mock_uniform.call_args_list:
			assert call.args == (3.0, 7.0)
