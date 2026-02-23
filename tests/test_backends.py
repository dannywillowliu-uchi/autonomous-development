"""Tests for worker execution backends (local, ssh, container)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.backends.base import WorkerHandle
from mission_control.backends.container import ContainerBackend
from mission_control.backends.local import _MB, LocalBackend
from mission_control.backends.ssh import SSHBackend
from mission_control.config import ContainerConfig, SSHHostConfig

# ---------------------------------------------------------------------------
# WorkerHandle dataclass
# ---------------------------------------------------------------------------

class TestWorkerHandle:
	def test_defaults(self) -> None:
		"""WorkerHandle sets correct defaults for optional fields."""
		handle = WorkerHandle(worker_id="w1")
		assert handle.worker_id == "w1"
		assert handle.pid is None
		assert handle.workspace_path == ""
		assert handle.backend_metadata == ""

	def test_full_construction(self) -> None:
		"""WorkerHandle stores all provided values."""
		handle = WorkerHandle(
			worker_id="w2",
			pid=12345,
			workspace_path="/tmp/ws",
			backend_metadata='{"host": "server1"}',
		)
		assert handle.worker_id == "w2"
		assert handle.pid == 12345
		assert handle.workspace_path == "/tmp/ws"
		assert handle.backend_metadata == '{"host": "server1"}'


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------

class TestLocalBackend:
	@pytest.fixture()
	def backend(self) -> LocalBackend:
		"""LocalBackend with a mocked WorkspacePool."""
		with patch("mission_control.backends.local.WorkspacePool") as mock_pool_cls:
			mock_pool_instance = MagicMock()
			mock_pool_cls.return_value = mock_pool_instance
			# Ensure async methods are AsyncMock
			mock_pool_instance.acquire = AsyncMock()
			mock_pool_instance.release = AsyncMock()
			mock_pool_instance.cleanup = AsyncMock()
			mock_pool_instance.initialize = AsyncMock()
			b = LocalBackend(
				source_repo="/repo",
				pool_dir="/pool",
				max_clones=5,
				base_branch="main",
			)
		return b

	def test_init_creates_pool(self) -> None:
		"""Constructor instantiates a WorkspacePool with correct args."""
		with patch("mission_control.backends.local.WorkspacePool") as mock_pool_cls:
			b = LocalBackend(
				source_repo="/repo",
				pool_dir="/pool",
				max_clones=8,
				base_branch="develop",
			)
			mock_pool_cls.assert_called_once_with(
				source_repo="/repo",
				pool_dir="/pool",
				max_clones=8,
				base_branch="develop",
			)
		assert b._processes == {}
		assert b._stdout_bufs == {}

	@patch("mission_control.backends.local.asyncio.create_subprocess_exec")
	async def test_provision_workspace(
		self, mock_exec: AsyncMock, backend: LocalBackend,
	) -> None:
		"""provision_workspace acquires from pool, checks out base_branch, then creates feature branch."""
		backend._pool.acquire.return_value = Path("/pool/clone-1")

		# Mock the git checkout subprocess (both calls succeed)
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.return_value = mock_proc

		result = await backend.provision_workspace("w1", "/repo", "main")

		assert result == "/pool/clone-1"
		backend._pool.acquire.assert_awaited_once()
		# Three subprocess calls: fetch, checkout base_branch, checkout -B feature branch
		assert mock_exec.await_count == 3
		fetch_call = mock_exec.call_args_list[0]
		assert fetch_call[0] == ("git", "fetch", "origin")
		assert fetch_call[1]["cwd"] == "/pool/clone-1"
		base_call = mock_exec.call_args_list[1]
		assert base_call[0] == ("git", "checkout", "-B", "main", "origin/main")
		assert base_call[1]["cwd"] == "/pool/clone-1"
		branch_call = mock_exec.call_args_list[2]
		assert branch_call[0] == ("git", "checkout", "-B", "mc/unit-w1")
		assert branch_call[1]["cwd"] == "/pool/clone-1"

	@patch("mission_control.backends.local.asyncio.create_subprocess_exec")
	async def test_provision_creates_marker_file(
		self, mock_exec: AsyncMock, backend: LocalBackend, tmp_path: Path,
	) -> None:
		"""provision_workspace creates .editable-install-protected marker."""
		workspace = tmp_path / "clone-1"
		workspace.mkdir()
		backend._pool.acquire.return_value = workspace

		# Create a source .venv so the symlink code path runs
		source_venv = tmp_path / "source" / ".venv"
		source_venv.mkdir(parents=True)

		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.return_value = mock_proc

		await backend.provision_workspace("w1", str(tmp_path / "source"), "main")

		marker = workspace / ".editable-install-protected"
		assert marker.exists()
		assert "Do not run pip install" in marker.read_text()

	@patch("mission_control.backends.local.asyncio.create_subprocess_exec")
	async def test_provision_workspace_uses_force_create_branch(
		self, mock_exec: AsyncMock, backend: LocalBackend,
	) -> None:
		"""provision_workspace uses checkout -B (force) so retried units succeed."""
		backend._pool.acquire.return_value = Path("/pool/clone-1")

		mock_proc = AsyncMock()
		mock_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_proc.returncode = 0
		mock_exec.return_value = mock_proc

		await backend.provision_workspace("w1", "/repo", "main")

		# call 0 = fetch, call 1 = base checkout, call 2 = feature branch
		args = mock_exec.call_args_list[2][0]
		assert args == ("git", "checkout", "-B", "mc/unit-w1")

	@patch("mission_control.backends.local.asyncio.create_subprocess_exec")
	async def test_provision_workspace_checkout_failure_raises(
		self, mock_exec: AsyncMock, backend: LocalBackend,
	) -> None:
		"""provision_workspace raises and releases workspace if feature branch checkout fails."""
		backend._pool.acquire.return_value = Path("/pool/clone-1")

		# call 0 = fetch (succeeds), call 1 = base checkout (succeeds),
		# call 2 = feature branch (fails)
		mock_fetch_proc = AsyncMock()
		mock_fetch_proc.returncode = 0
		mock_fetch_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_base_proc = AsyncMock()
		mock_base_proc.returncode = 0
		mock_base_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_branch_proc = AsyncMock()
		mock_branch_proc.communicate = AsyncMock(return_value=(b"fatal: error", None))
		mock_branch_proc.returncode = 1
		mock_exec.side_effect = [mock_fetch_proc, mock_base_proc, mock_branch_proc]

		with pytest.raises(RuntimeError, match="Failed to create branch"):
			await backend.provision_workspace("w1", "/repo", "main")

		# Should have released the workspace back to the pool
		backend._pool.release.assert_awaited_once_with(Path("/pool/clone-1"))

	@patch("mission_control.backends.local.asyncio.create_subprocess_exec")
	async def test_provision_workspace_base_checkout_failure_raises(
		self, mock_exec: AsyncMock, backend: LocalBackend,
	) -> None:
		"""provision_workspace raises and releases workspace if base branch checkout fails."""
		backend._pool.acquire.return_value = Path("/pool/clone-1")

		# Fetch succeeds, base checkout fails
		mock_fetch_proc = AsyncMock()
		mock_fetch_proc.returncode = 0
		mock_fetch_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_fail_proc = AsyncMock()
		mock_fail_proc.communicate = AsyncMock(return_value=(b"error: pathspec", None))
		mock_fail_proc.returncode = 1
		mock_exec.side_effect = [mock_fetch_proc, mock_fail_proc]

		with pytest.raises(RuntimeError, match="Failed to checkout base branch"):
			await backend.provision_workspace("w1", "/repo", "mc/green")

		backend._pool.release.assert_awaited_once_with(Path("/pool/clone-1"))

	async def test_provision_workspace_no_workspace_available(
		self, backend: LocalBackend,
	) -> None:
		"""provision_workspace raises RuntimeError when pool returns None."""
		backend._pool.acquire.return_value = None

		with pytest.raises(RuntimeError, match="No workspace available"):
			await backend.provision_workspace("w1", "/repo", "main")


	@patch("mission_control.backends.local.asyncio.create_subprocess_exec")
	async def test_provision_workspace_uses_specified_base_branch(
		self, mock_exec: AsyncMock, backend: LocalBackend,
	) -> None:
		"""provision_workspace checks out the specified base_branch, not hardcoded main."""
		backend._pool.acquire.return_value = Path("/pool/clone-1")

		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.return_value = mock_proc

		await backend.provision_workspace("w1", "/repo", "mc/green")

		# call 0 = fetch, call 1 = base checkout (should be mc/green, not main)
		base_call = mock_exec.call_args_list[1]
		assert base_call[0] == ("git", "checkout", "-B", "mc/green", "origin/mc/green")

	@patch("mission_control.backends.local.asyncio.create_subprocess_exec")
	async def test_provision_workspace_resets_base_branch_to_origin(
		self, mock_exec: AsyncMock, backend: LocalBackend,
	) -> None:
		"""provision_workspace uses -B to force-reset the local base branch to origin.

		Without -B, a stale local mc/green would be checked out instead of the
		latest origin/mc/green, causing merge conflicts when multiple workers
		fork from outdated code.
		"""
		backend._pool.acquire.return_value = Path("/pool/clone-1")

		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.return_value = mock_proc

		await backend.provision_workspace("w1", "/repo", "mc/green")

		# The base branch checkout (call 1) must use -B to reset to origin
		base_call = mock_exec.call_args_list[1]
		assert base_call[0] == (
			"git", "checkout", "-B", "mc/green", "origin/mc/green",
		), "Base branch checkout must use -B to reset local branch to origin"

	@patch("mission_control.backends.local.asyncio.create_subprocess_exec")
	async def test_spawn(self, mock_exec: AsyncMock, backend: LocalBackend) -> None:
		"""spawn creates a subprocess and returns a WorkerHandle."""
		mock_proc = AsyncMock()
		mock_proc.pid = 9999
		mock_exec.return_value = mock_proc

		handle = await backend.spawn("w1", "/ws", ["claude", "code", "--task", "fix"], 120)

		assert handle.worker_id == "w1"
		assert handle.pid == 9999
		assert handle.workspace_path == "/ws"
		call_kwargs = mock_exec.call_args[1]
		assert call_kwargs["cwd"] == "/ws"
		assert call_kwargs["stdout"] == asyncio.subprocess.PIPE
		assert call_kwargs["stderr"] == asyncio.subprocess.STDOUT
		assert "env" in call_kwargs
		assert "ANTHROPIC_API_KEY" not in call_kwargs["env"]
		assert "w1" in backend._processes
		assert backend._stdout_bufs["w1"] == b""

	@patch("mission_control.backends.local.asyncio.create_subprocess_exec")
	async def test_spawn_clears_stdout_collected(self, mock_exec: AsyncMock, backend: LocalBackend) -> None:
		"""spawn should clear _stdout_collected so a reused worker collects output."""
		mock_proc = AsyncMock()
		mock_proc.pid = 1111
		mock_exec.return_value = mock_proc

		# Simulate a prior spawn that already collected output
		backend._stdout_collected.add("w1")

		await backend.spawn("w1", "/ws", ["echo", "hi"], 60)

		# _stdout_collected should no longer contain w1
		assert "w1" not in backend._stdout_collected

	async def test_check_status_running(self, backend: LocalBackend) -> None:
		"""check_status returns 'running' when returncode is None."""
		mock_proc = MagicMock()
		mock_proc.returncode = None
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		assert await backend.check_status(handle) == "running"

	async def test_check_status_completed(self, backend: LocalBackend) -> None:
		"""check_status returns 'completed' when returncode is 0."""
		mock_proc = MagicMock()
		mock_proc.returncode = 0
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		assert await backend.check_status(handle) == "completed"

	async def test_check_status_failed_nonzero(self, backend: LocalBackend) -> None:
		"""check_status returns 'failed' when returncode is non-zero."""
		mock_proc = MagicMock()
		mock_proc.returncode = 1
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		assert await backend.check_status(handle) == "failed"

	async def test_check_status_unknown_worker(self, backend: LocalBackend) -> None:
		"""check_status returns 'failed' for unknown worker_id."""
		handle = WorkerHandle(worker_id="unknown")
		assert await backend.check_status(handle) == "failed"

	async def test_get_output_finished_process(self, backend: LocalBackend) -> None:
		"""get_output returns decoded stdout for a finished process."""
		mock_proc = MagicMock()
		mock_proc.returncode = 0
		mock_stdout = AsyncMock()
		mock_stdout.read = AsyncMock(return_value=b"hello world\n")
		mock_proc.stdout = mock_stdout
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		output = await backend.get_output(handle)
		assert output == "hello world\n"

	async def test_get_output_finished_appends_remaining(self, backend: LocalBackend) -> None:
		"""get_output appends remaining stdout to partial buffer for finished process."""
		mock_proc = MagicMock()
		mock_proc.returncode = 0
		mock_stdout = AsyncMock()
		mock_stdout.read = AsyncMock(return_value=b" world\n")
		mock_proc.stdout = mock_stdout
		backend._processes["w1"] = mock_proc
		# Simulate partial data already read during polling
		backend._stdout_bufs["w1"] = b"hello"

		handle = WorkerHandle(worker_id="w1")
		output = await backend.get_output(handle)
		assert output == "hello world\n"
		# Second call should not re-read (already collected)
		mock_stdout.read.reset_mock()
		output2 = await backend.get_output(handle)
		assert output2 == "hello world\n"
		mock_stdout.read.assert_not_awaited()

	async def test_get_output_unknown_worker(self, backend: LocalBackend) -> None:
		"""get_output returns empty string for unknown worker_id."""
		handle = WorkerHandle(worker_id="unknown")
		assert await backend.get_output(handle) == ""

	async def test_kill(self, backend: LocalBackend) -> None:
		"""kill terminates a running process."""
		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		await backend.kill(handle)

		mock_proc.kill.assert_called_once()
		mock_proc.wait.assert_awaited_once()

	async def test_kill_already_finished(self, backend: LocalBackend) -> None:
		"""kill is a no-op for an already-finished process."""
		mock_proc = MagicMock()
		mock_proc.returncode = 0
		mock_proc.kill = MagicMock()
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		await backend.kill(handle)

		mock_proc.kill.assert_not_called()

	async def test_initialize_delegates_to_pool(self, backend: LocalBackend) -> None:
		"""initialize delegates to pool.initialize with warm_count."""
		await backend.initialize(warm_count=3)
		backend._pool.initialize.assert_awaited_once_with(warm_count=3)

	async def test_initialize_default_warm_count(self, backend: LocalBackend) -> None:
		"""initialize passes default warm_count=0."""
		await backend.initialize()
		backend._pool.initialize.assert_awaited_once_with(warm_count=0)

	async def test_release_workspace(self, backend: LocalBackend) -> None:
		"""release_workspace delegates to pool.release with a Path."""
		await backend.release_workspace("/pool/clone-1")
		backend._pool.release.assert_awaited_once_with(Path("/pool/clone-1"))

	async def test_cleanup(self, backend: LocalBackend) -> None:
		"""cleanup kills all running processes and clears state."""
		running = MagicMock()
		running.returncode = None
		running.kill = MagicMock()
		running.wait = AsyncMock()

		finished = MagicMock()
		finished.returncode = 0

		backend._processes["w1"] = running
		backend._processes["w2"] = finished
		backend._stdout_bufs["w1"] = b"data"

		await backend.cleanup()

		running.kill.assert_called_once()
		running.wait.assert_awaited_once()
		assert backend._processes == {}
		assert backend._stdout_bufs == {}
		backend._pool.cleanup.assert_awaited_once()


# ---------------------------------------------------------------------------
# Worker output overflow
# ---------------------------------------------------------------------------

class TestWorkerOutputOverflow:
	"""Tests for output size limits in LocalBackend."""

	@pytest.fixture()
	def backend(self) -> LocalBackend:
		"""LocalBackend with 1MB output limit for fast testing."""
		with patch("mission_control.backends.local.WorkspacePool") as mock_pool_cls:
			mock_pool_instance = MagicMock()
			mock_pool_cls.return_value = mock_pool_instance
			mock_pool_instance.acquire = AsyncMock()
			mock_pool_instance.release = AsyncMock()
			mock_pool_instance.cleanup = AsyncMock()
			mock_pool_instance.initialize = AsyncMock()
			b = LocalBackend(
				source_repo="/repo",
				pool_dir="/pool",
				max_clones=5,
				base_branch="main",
				max_output_mb=1,
			)
		return b

	async def test_output_accumulates_normally_below_limit(self, backend: LocalBackend) -> None:
		"""Output under the limit accumulates without truncation or killing."""
		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.pid = 100
		mock_stdout = AsyncMock()
		# 500KB chunk -- well under 1MB limit
		chunk = b"x" * (500 * 1024)
		mock_stdout.read = AsyncMock(return_value=chunk)
		mock_proc.stdout = mock_stdout
		mock_proc.kill = MagicMock()

		backend._processes["w1"] = mock_proc
		backend._stdout_bufs["w1"] = b""
		backend._output_warnings_fired["w1"] = set()

		handle = WorkerHandle(worker_id="w1", pid=100)
		output = await backend.get_output(handle)

		assert len(output) == 500 * 1024
		mock_proc.kill.assert_not_called()

	async def test_warning_logged_at_thresholds(self, backend: LocalBackend, caplog: pytest.LogCaptureFixture) -> None:
		"""Warnings are logged when output crosses threshold boundaries."""
		mock_proc = MagicMock()
		mock_proc.returncode = 0  # finished
		mock_proc.pid = 200
		mock_stdout = AsyncMock()
		mock_stdout.read = AsyncMock(return_value=b"")
		mock_proc.stdout = mock_stdout

		backend._processes["w1"] = mock_proc
		# Pre-fill buffer to 11MB to cross the 10MB threshold
		backend._stdout_bufs["w1"] = b"x" * (11 * _MB)
		backend._output_warnings_fired["w1"] = set()
		backend._stdout_collected.add("w1")  # already collected

		handle = WorkerHandle(worker_id="w1", pid=200)
		with caplog.at_level(logging.WARNING, logger="mission_control.backends.local"):
			await backend.get_output(handle)

		assert any("10MB" in msg for msg in caplog.messages)

	async def test_worker_killed_and_output_truncated_at_limit(
		self, backend: LocalBackend, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Worker is killed and output truncated when output exceeds max_output_mb."""
		mock_proc = MagicMock()
		mock_proc.returncode = None  # still running
		mock_proc.pid = 300
		mock_stdout = AsyncMock()
		# Return a chunk that pushes over 1MB limit
		mock_stdout.read = AsyncMock(return_value=b"y" * (100 * 1024))
		mock_proc.stdout = mock_stdout
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		backend._processes["w1"] = mock_proc
		# Pre-fill to just under 1MB
		backend._stdout_bufs["w1"] = b"x" * (1 * _MB - 10 * 1024)
		backend._output_warnings_fired["w1"] = set()

		handle = WorkerHandle(worker_id="w1", pid=300)
		with caplog.at_level(logging.ERROR, logger="mission_control.backends.local"):
			output = await backend.get_output(handle)

		# Output should be truncated to exactly max_output_bytes
		assert len(output.encode()) == 1 * _MB
		mock_proc.kill.assert_called_once()
		mock_proc.wait.assert_awaited_once()
		assert any("exceeded limit" in msg for msg in caplog.messages)

	async def test_finished_process_output_truncated(self, backend: LocalBackend) -> None:
		"""Output from a finished process is truncated if final read pushes over limit."""
		mock_proc = MagicMock()
		mock_proc.returncode = 0
		mock_proc.pid = 400
		mock_stdout = AsyncMock()
		# Return large remaining output
		mock_stdout.read = AsyncMock(return_value=b"z" * (2 * _MB))
		mock_proc.stdout = mock_stdout

		backend._processes["w1"] = mock_proc
		backend._stdout_bufs["w1"] = b""
		backend._output_warnings_fired["w1"] = set()

		handle = WorkerHandle(worker_id="w1", pid=400)
		output = await backend.get_output(handle)

		assert len(output.encode()) == 1 * _MB


# ---------------------------------------------------------------------------
# SSHBackend
# ---------------------------------------------------------------------------

class TestSSHBackend:
	@pytest.fixture()
	def hosts(self) -> list[SSHHostConfig]:
		return [
			SSHHostConfig(hostname="host-a", user="deploy", max_workers=2),
			SSHHostConfig(hostname="host-b", user="deploy", max_workers=3),
		]

	@pytest.fixture()
	def backend(self, hosts: list[SSHHostConfig]) -> SSHBackend:
		return SSHBackend(hosts=hosts)

	def test_init_empty_hosts_raises(self) -> None:
		"""SSHBackend requires at least one host."""
		with pytest.raises(ValueError, match="at least one host"):
			SSHBackend(hosts=[])

	def test_init_sets_worker_counts(self, backend: SSHBackend) -> None:
		"""Constructor initializes worker counts to zero."""
		assert backend._worker_count == {"host-a": 0, "host-b": 0}

	def test_select_host_picks_least_loaded(self, backend: SSHBackend) -> None:
		"""_select_host returns the host with fewest active workers."""
		backend._worker_count["host-a"] = 2  # at capacity
		backend._worker_count["host-b"] = 1

		chosen = backend._select_host()
		assert chosen.hostname == "host-b"

	def test_select_host_all_at_capacity(self, backend: SSHBackend) -> None:
		"""_select_host raises when all hosts are at capacity."""
		backend._worker_count["host-a"] = 2
		backend._worker_count["host-b"] = 3

		with pytest.raises(RuntimeError, match="at capacity"):
			backend._select_host()

	def test_select_host_prefers_lower_count(self, backend: SSHBackend) -> None:
		"""When both have room, picks the one with fewer workers."""
		backend._worker_count["host-a"] = 1
		backend._worker_count["host-b"] = 0

		chosen = backend._select_host()
		assert chosen.hostname == "host-b"

	def test_ssh_target_with_user(self, backend: SSHBackend) -> None:
		host = SSHHostConfig(hostname="server1", user="admin")
		assert backend._ssh_target(host) == "admin@server1"

	def test_ssh_target_without_user(self, backend: SSHBackend) -> None:
		host = SSHHostConfig(hostname="server1", user="")
		assert backend._ssh_target(host) == "server1"

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_provision_workspace(
		self, mock_exec: AsyncMock, backend: SSHBackend,
	) -> None:
		"""provision_workspace runs ssh git clone and returns packed path."""
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"Cloning...", None))
		mock_exec.return_value = mock_proc

		result = await backend.provision_workspace("w1", "git@github.com:repo.git", "main")

		# Result should contain remote_path::metadata
		assert "::" in result
		remote_path, metadata_str = result.split("::", 1)
		assert remote_path == "/tmp/mc-worker-w1"
		metadata = json.loads(metadata_str)
		assert metadata["hostname"] in ("host-a", "host-b")

		# Should have called ssh with git clone
		mock_exec.assert_awaited_once()
		call_args = mock_exec.call_args[0]
		assert call_args[0] == "ssh"
		# ssh target should be user@hostname
		assert "@" in call_args[1]

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_provision_workspace_quotes_arguments(
		self, mock_exec: AsyncMock, backend: SSHBackend,
	) -> None:
		"""provision_workspace uses shlex.quote on all interpolated values."""
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"Cloning...", None))
		mock_exec.return_value = mock_proc

		# Source repo with spaces and metacharacters
		await backend.provision_workspace("w1", "git@host:repo with spaces.git", "main; rm -rf /")

		mock_exec.assert_awaited_once()
		call_args = mock_exec.call_args[0]
		remote_cmd = call_args[2]
		import shlex
		# All three values should be quoted
		assert shlex.quote("main; rm -rf /") in remote_cmd
		assert shlex.quote("git@host:repo with spaces.git") in remote_cmd
		assert shlex.quote("/tmp/mc-worker-w1") in remote_cmd

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_release_workspace_quotes_path(
		self, mock_exec: AsyncMock, backend: SSHBackend,
	) -> None:
		"""release_workspace uses shlex.quote on the remote path."""
		mock_proc = AsyncMock()
		mock_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.return_value = mock_proc

		backend._worker_count["host-a"] = 1
		metadata = json.dumps({"hostname": "host-a", "user": "deploy"})
		workspace_path = f"/tmp/path with spaces::{metadata}"

		await backend.release_workspace(workspace_path)

		call_args = mock_exec.call_args[0]
		import shlex
		assert shlex.quote("/tmp/path with spaces") in call_args[2]

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_kill_quotes_worker_id_in_pkill(
		self, mock_exec: AsyncMock, backend: SSHBackend,
	) -> None:
		"""kill uses shlex.quote for worker_id in the pkill command."""
		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()
		backend._processes["w1; evil"] = mock_proc

		mock_cleanup = AsyncMock()
		mock_cleanup.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.return_value = mock_cleanup

		metadata = json.dumps({"hostname": "host-a", "user": "deploy"})
		handle = WorkerHandle(
			worker_id="w1; evil",
			workspace_path="/tmp/mc-worker-w1",
			backend_metadata=metadata,
		)

		await backend.kill(handle)

		call_args = mock_exec.call_args[0]
		import shlex
		assert shlex.quote("mc-worker-w1; evil") in call_args[2]

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_provision_workspace_failure(
		self, mock_exec: AsyncMock, backend: SSHBackend,
	) -> None:
		"""provision_workspace raises on non-zero exit code."""
		mock_proc = AsyncMock()
		mock_proc.returncode = 1
		mock_proc.communicate = AsyncMock(return_value=(b"fatal: repo not found", None))
		mock_exec.return_value = mock_proc

		with pytest.raises(RuntimeError, match="Failed to provision"):
			await backend.provision_workspace("w1", "bad-repo", "main")

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_spawn_quotes_command(
		self, mock_exec: AsyncMock, backend: SSHBackend,
	) -> None:
		"""spawn uses shlex.quote to prevent shell injection."""
		mock_proc = AsyncMock()
		mock_proc.pid = 999
		mock_proc.returncode = None
		mock_exec.return_value = mock_proc

		workspace = '/tmp/mc-worker-w1::{"hostname":"host-a","user":"deploy"}'
		# Command contains shell metacharacters
		cmd = ["claude", "--prompt", 'fix $(rm -rf /); echo "pwned"']
		handle = await backend.spawn("w1", workspace, cmd, timeout=60)

		assert handle.worker_id == "w1"
		mock_exec.assert_awaited_once()
		call_args = mock_exec.call_args[0]
		# The remote command (third arg to ssh) should have quoted parts
		remote_cmd = call_args[2]
		# The dangerous metacharacters should be quoted/escaped
		assert "$(rm -rf /)" not in remote_cmd or "'" in remote_cmd
		# Verify shlex quoting is present
		import shlex
		assert shlex.quote('fix $(rm -rf /); echo "pwned"') in remote_cmd

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_get_output_consistent_on_repeated_calls(
		self, mock_exec: AsyncMock, backend: SSHBackend,
	) -> None:
		"""get_output returns same data on repeated calls (output is buffered)."""
		mock_proc = AsyncMock()
		mock_proc.pid = 888
		mock_proc.returncode = 0
		mock_stdout = AsyncMock()
		mock_stdout.read = AsyncMock(return_value=b"test output data")
		mock_proc.stdout = mock_stdout
		mock_exec.return_value = mock_proc

		workspace = '/tmp/mc-worker-w2::{"hostname":"host-a","user":"deploy"}'
		handle = await backend.spawn("w2", workspace, ["echo", "hello"], timeout=60)

		# First call reads from stream
		result1 = await backend.get_output(handle)
		assert result1 == "test output data"

		# Second call should return same data (not empty)
		mock_stdout.read = AsyncMock(return_value=b"")  # stream exhausted
		result2 = await backend.get_output(handle)
		assert result2 == "test output data"

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_spawn(self, mock_exec: AsyncMock, backend: SSHBackend) -> None:
		"""spawn ssh-es into the remote host and runs the command."""
		mock_proc = AsyncMock()
		mock_proc.pid = 5555
		mock_exec.return_value = mock_proc

		metadata = json.dumps({"hostname": "host-a", "user": "deploy"})
		workspace_path = f"/tmp/mc-worker-w1::{metadata}"

		handle = await backend.spawn("w1", workspace_path, ["claude", "--task", "fix"], 60)

		assert handle.worker_id == "w1"
		assert handle.pid == 5555
		assert handle.workspace_path == "/tmp/mc-worker-w1"
		assert handle.backend_metadata == metadata

		call_args = mock_exec.call_args[0]
		assert call_args[0] == "ssh"
		assert call_args[1] == "deploy@host-a"

	async def test_check_status_running(self, backend: SSHBackend) -> None:
		mock_proc = MagicMock()
		mock_proc.returncode = None
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		assert await backend.check_status(handle) == "running"

	async def test_check_status_completed(self, backend: SSHBackend) -> None:
		mock_proc = MagicMock()
		mock_proc.returncode = 0
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		assert await backend.check_status(handle) == "completed"

	async def test_check_status_failed(self, backend: SSHBackend) -> None:
		mock_proc = MagicMock()
		mock_proc.returncode = 1
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		assert await backend.check_status(handle) == "failed"

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_kill(self, mock_exec: AsyncMock, backend: SSHBackend) -> None:
		"""kill terminates local ssh process and sends remote pkill."""
		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()
		backend._processes["w1"] = mock_proc

		# Mock the remote cleanup subprocess
		mock_cleanup = AsyncMock()
		mock_cleanup.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.return_value = mock_cleanup

		metadata = json.dumps({"hostname": "host-a", "user": "deploy"})
		handle = WorkerHandle(
			worker_id="w1",
			workspace_path="/tmp/mc-worker-w1",
			backend_metadata=metadata,
		)

		await backend.kill(handle)

		mock_proc.kill.assert_called_once()
		mock_proc.wait.assert_awaited_once()
		# Should also ssh pkill the remote process
		mock_exec.assert_awaited_once()
		call_args = mock_exec.call_args[0]
		assert call_args[0] == "ssh"
		assert "pkill" in call_args[2]

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_release_workspace_with_metadata(
		self, mock_exec: AsyncMock, backend: SSHBackend,
	) -> None:
		"""release_workspace ssh rm -rf the remote directory."""
		mock_proc = AsyncMock()
		mock_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.return_value = mock_proc

		# Bump count so we can verify it decrements
		backend._worker_count["host-a"] = 1

		metadata = json.dumps({"hostname": "host-a", "user": "deploy"})
		workspace_path = f"/tmp/mc-worker-w1::{metadata}"

		await backend.release_workspace(workspace_path)

		mock_exec.assert_awaited_once()
		call_args = mock_exec.call_args[0]
		assert call_args[0] == "ssh"
		assert "rm -rf" in call_args[2] and "/tmp/mc-worker-w1" in call_args[2]
		assert backend._worker_count["host-a"] == 0

	async def test_release_workspace_without_metadata(
		self, backend: SSHBackend,
	) -> None:
		"""release_workspace with no metadata is a no-op (no SSH call)."""
		# No :: separator, so host_info is None and nothing happens
		await backend.release_workspace("/tmp/mc-worker-w1")
		# Should not raise -- just silently skip

	async def test_cleanup(self, backend: SSHBackend) -> None:
		"""cleanup kills all running processes and resets counts."""
		running = MagicMock()
		running.returncode = None
		running.kill = MagicMock()
		running.wait = AsyncMock()

		backend._processes["w1"] = running
		backend._worker_count["host-a"] = 1

		await backend.cleanup()

		running.kill.assert_called_once()
		running.wait.assert_awaited_once()
		assert backend._processes == {}
		assert backend._worker_count == {"host-a": 0, "host-b": 0}

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_get_output_drains_stdout_while_running(
		self, mock_exec: AsyncMock, backend: SSHBackend,
	) -> None:
		"""get_output reads chunks incrementally while process is running to prevent pipe deadlock."""
		mock_proc = AsyncMock()
		mock_proc.pid = 777
		mock_proc.returncode = None  # Still running
		mock_stdout = AsyncMock()
		# Simulate incremental reads: each call returns a chunk
		mock_stdout.read = AsyncMock(return_value=b"chunk1")
		mock_proc.stdout = mock_stdout
		mock_exec.return_value = mock_proc

		workspace = '/tmp/mc-worker-w3::{"hostname":"host-a","user":"deploy"}'
		handle = await backend.spawn("w3", workspace, ["cmd"], timeout=60)

		# First call while running -- should read a chunk
		out1 = await backend.get_output(handle)
		assert out1 == "chunk1"
		mock_stdout.read.assert_awaited_once()

		# Second call while still running -- reads another chunk
		mock_stdout.read = AsyncMock(return_value=b"chunk2")
		out2 = await backend.get_output(handle)
		assert out2 == "chunk1chunk2"  # Accumulated

		# Process finishes -- get remaining output
		mock_proc.returncode = 0
		mock_stdout.read = AsyncMock(return_value=b"final")
		out3 = await backend.get_output(handle)
		assert out3 == "chunk1chunk2final"

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_spawn_clears_stdout_collected_on_reuse(
		self, mock_exec: AsyncMock, backend: SSHBackend,
	) -> None:
		"""spawn clears _stdout_collected so reused workers get fresh output."""
		mock_proc1 = AsyncMock()
		mock_proc1.pid = 100
		mock_proc1.returncode = 0
		mock_stdout1 = AsyncMock()
		mock_stdout1.read = AsyncMock(return_value=b"output-1")
		mock_proc1.stdout = mock_stdout1

		mock_proc2 = AsyncMock()
		mock_proc2.pid = 200
		mock_proc2.returncode = 0
		mock_stdout2 = AsyncMock()
		mock_stdout2.read = AsyncMock(return_value=b"output-2")
		mock_proc2.stdout = mock_stdout2

		mock_exec.side_effect = [mock_proc1, mock_proc2]

		workspace = '/tmp/mc-worker-w1::{"hostname":"host-a","user":"deploy"}'

		# First spawn + get_output
		handle1 = await backend.spawn("w1", workspace, ["cmd1"], timeout=60)
		out1 = await backend.get_output(handle1)
		assert out1 == "output-1"
		assert "w1" in backend._stdout_collected

		# Second spawn (same worker_id) should clear _stdout_collected
		handle2 = await backend.spawn("w1", workspace, ["cmd2"], timeout=60)
		assert "w1" not in backend._stdout_collected

		out2 = await backend.get_output(handle2)
		assert out2 == "output-2"

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_kill_ssh_timeout_logs_warning(
		self, mock_exec: AsyncMock, backend: SSHBackend, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""kill logs a warning and continues when SSH pkill hangs past timeout."""
		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()
		backend._processes["w1"] = mock_proc

		# SSH cleanup process that hangs forever
		mock_cleanup = AsyncMock()
		mock_cleanup.communicate = AsyncMock(side_effect=asyncio.Future)
		mock_exec.return_value = mock_cleanup

		metadata = json.dumps({"hostname": "host-a", "user": "deploy"})
		handle = WorkerHandle(
			worker_id="w1",
			workspace_path="/tmp/mc-worker-w1",
			backend_metadata=metadata,
		)

		with patch("mission_control.backends.ssh.asyncio.wait_for", side_effect=asyncio.TimeoutError):
			with caplog.at_level(logging.WARNING, logger="mission_control.backends.ssh"):
				await backend.kill(handle)

		# Local process was still killed
		mock_proc.kill.assert_called_once()
		mock_proc.wait.assert_awaited_once()
		# Warning was logged about the timeout
		assert any("SSH pkill timed out" in msg for msg in caplog.messages)

	@patch("mission_control.backends.ssh.asyncio.create_subprocess_exec")
	async def test_release_workspace_timeout_logs_warning(
		self, mock_exec: AsyncMock, backend: SSHBackend, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""release_workspace logs a warning and continues when SSH rm -rf hangs."""
		mock_proc = AsyncMock()
		mock_proc.communicate = AsyncMock(side_effect=asyncio.Future)
		mock_exec.return_value = mock_proc

		backend._worker_count["host-a"] = 1
		metadata = json.dumps({"hostname": "host-a", "user": "deploy"})
		workspace_path = f"/tmp/mc-worker-w1::{metadata}"

		with patch("mission_control.backends.ssh.asyncio.wait_for", side_effect=asyncio.TimeoutError):
			with caplog.at_level(logging.WARNING, logger="mission_control.backends.ssh"):
				await backend.release_workspace(workspace_path)

		assert any("SSH rm -rf timed out" in msg for msg in caplog.messages)
		# Worker count still decremented (best-effort cleanup continues)
		assert backend._worker_count["host-a"] == 0

	async def test_kill_no_metadata_skips_remote(self, backend: SSHBackend) -> None:
		"""kill works when backend_metadata is empty -- only kills local process."""
		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(
			worker_id="w1",
			workspace_path="/tmp/mc-worker-w1",
			backend_metadata="",
		)

		# Should not raise -- just kills local process, skips SSH
		await backend.kill(handle)

		mock_proc.kill.assert_called_once()
		mock_proc.wait.assert_awaited_once()

	async def test_cleanup_kills_with_timeout(self, backend: SSHBackend) -> None:
		"""cleanup handles hanging processes gracefully."""
		running = MagicMock()
		running.returncode = None
		running.kill = MagicMock()
		# Simulate a process.wait() that completes normally
		running.wait = AsyncMock()

		finished = MagicMock()
		finished.returncode = 0

		backend._processes["w1"] = running
		backend._processes["w2"] = finished
		backend._stdout_bufs["w1"] = b"partial"
		backend._worker_count["host-a"] = 2

		await backend.cleanup()

		running.kill.assert_called_once()
		running.wait.assert_awaited_once()
		assert backend._processes == {}
		assert backend._stdout_bufs == {}
		assert backend._stdout_collected == set()
		assert backend._worker_count == {"host-a": 0, "host-b": 0}


# ---------------------------------------------------------------------------
# ContainerBackend
# ---------------------------------------------------------------------------

class TestContainerBackend:
	@pytest.fixture()
	def container_config(self) -> ContainerConfig:
		return ContainerConfig(
			image="mission-control-worker:latest",
			claude_config_dir="/home/user/.config/claude",
		)

	@pytest.fixture()
	def backend(self, container_config: ContainerConfig) -> ContainerBackend:
		"""ContainerBackend with a mocked WorkspacePool."""
		with patch("mission_control.backends.container.WorkspacePool") as mock_pool_cls:
			mock_pool_instance = MagicMock()
			mock_pool_cls.return_value = mock_pool_instance
			mock_pool_instance.acquire = AsyncMock()
			mock_pool_instance.release = AsyncMock()
			mock_pool_instance.cleanup = AsyncMock()
			mock_pool_instance.initialize = AsyncMock()
			b = ContainerBackend(
				source_repo="/repo",
				pool_dir="/pool",
				container_config=container_config,
				max_clones=5,
				base_branch="main",
			)
		return b

	def test_init_creates_pool(self) -> None:
		"""Constructor instantiates a WorkspacePool with correct args."""
		cc = ContainerConfig()
		with patch("mission_control.backends.container.WorkspacePool") as mock_pool_cls:
			ContainerBackend(
				source_repo="/repo",
				pool_dir="/pool",
				container_config=cc,
				max_clones=8,
				base_branch="develop",
			)
			mock_pool_cls.assert_called_once_with(
				source_repo="/repo",
				pool_dir="/pool",
				max_clones=8,
				base_branch="develop",
			)

	def test_build_docker_command_volumes(self, backend: ContainerBackend) -> None:
		"""docker run command includes workspace and claude config mounts."""
		cmd = backend._build_docker_command("w1", "/host/workspace", ["claude", "--task", "fix"])

		assert "-v" in cmd
		# Workspace mount
		assert "/host/workspace:/workspace" in cmd
		# Claude config mount (read-only)
		assert "/home/user/.config/claude:/home/mcworker/.config/claude:ro" in cmd
		# Image and command at the end
		assert cmd[-3] == "claude"
		assert cmd[-1] == "fix"

	def test_build_docker_command_caps_and_security(self, backend: ContainerBackend) -> None:
		"""docker run command drops capabilities and sets security options."""
		cmd = backend._build_docker_command("w1", "/host/ws", ["echo"])

		# --cap-drop ALL
		cap_idx = cmd.index("--cap-drop")
		assert cmd[cap_idx + 1] == "ALL"

		# --security-opt no-new-privileges:true
		sec_idx = cmd.index("--security-opt")
		assert cmd[sec_idx + 1] == "no-new-privileges:true"

		# --user
		user_idx = cmd.index("--user")
		assert cmd[user_idx + 1] == "10000:10000"

	def test_build_docker_command_env_vars(self, backend: ContainerBackend, monkeypatch: pytest.MonkeyPatch) -> None:
		"""docker run passes allowlisted env vars, not secrets."""
		monkeypatch.setenv("HOME", "/home/test")
		monkeypatch.setenv("PATH", "/usr/bin")
		monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")

		cmd = backend._build_docker_command("w1", "/host/ws", ["echo"])
		cmd_str = " ".join(cmd)

		# HOME and PATH should be passed
		assert "HOME=/home/test" in cmd_str or "HOME=/home/mcworker" in cmd_str
		# ANTHROPIC_API_KEY must NOT be passed
		assert "sk-secret" not in cmd_str

	def test_build_docker_command_skips_claude_config_when_empty(self) -> None:
		"""No claude config volume when claude_config_dir is empty and env unset."""
		cc = ContainerConfig(claude_config_dir="")
		with patch("mission_control.backends.container.WorkspacePool"):
			b = ContainerBackend(
				source_repo="/repo", pool_dir="/pool",
				container_config=cc,
			)
		with patch.dict("os.environ", {}, clear=True):
			import os
			os.environ.pop("CLAUDE_CONFIG_DIR", None)
			cmd = b._build_docker_command("w1", "/host/ws", ["echo"])

		# Should not contain the mcworker config mount
		assert "/home/mcworker/.config/claude:ro" not in " ".join(cmd)

	@patch("mission_control.backends.container.asyncio.create_subprocess_exec")
	async def test_spawn_builds_docker_command(
		self, mock_exec: AsyncMock, backend: ContainerBackend,
	) -> None:
		"""spawn launches docker run and returns handle with host workspace path."""
		mock_proc = AsyncMock()
		mock_proc.pid = 8888
		mock_exec.return_value = mock_proc

		handle = await backend.spawn("w1", "/host/workspace", ["claude", "--task", "fix"], 120)

		assert handle.worker_id == "w1"
		assert handle.pid == 8888
		assert handle.workspace_path == "/host/workspace"

		# Verify docker run was called
		call_args = mock_exec.call_args[0]
		assert call_args[0] == "docker"
		assert call_args[1] == "run"
		assert "--rm" in call_args

	@patch("mission_control.backends.container.asyncio.create_subprocess_exec")
	async def test_spawn_returns_host_workspace_path(
		self, mock_exec: AsyncMock, backend: ContainerBackend,
	) -> None:
		"""spawn returns the HOST workspace path, not the container mount path."""
		mock_proc = AsyncMock()
		mock_proc.pid = 7777
		mock_exec.return_value = mock_proc

		handle = await backend.spawn("w1", "/host/pool/clone-1", ["cmd"], 60)

		# Must be host path, not /workspace
		assert handle.workspace_path == "/host/pool/clone-1"
		assert handle.workspace_path != "/workspace"

	async def test_check_status_running(self, backend: ContainerBackend) -> None:
		mock_proc = MagicMock()
		mock_proc.returncode = None
		backend._processes["w1"] = mock_proc
		handle = WorkerHandle(worker_id="w1")
		assert await backend.check_status(handle) == "running"

	async def test_check_status_completed(self, backend: ContainerBackend) -> None:
		mock_proc = MagicMock()
		mock_proc.returncode = 0
		backend._processes["w1"] = mock_proc
		handle = WorkerHandle(worker_id="w1")
		assert await backend.check_status(handle) == "completed"

	async def test_check_status_failed(self, backend: ContainerBackend) -> None:
		mock_proc = MagicMock()
		mock_proc.returncode = 1
		backend._processes["w1"] = mock_proc
		handle = WorkerHandle(worker_id="w1")
		assert await backend.check_status(handle) == "failed"

	async def test_check_status_unknown(self, backend: ContainerBackend) -> None:
		handle = WorkerHandle(worker_id="unknown")
		assert await backend.check_status(handle) == "failed"

	async def test_get_output_finished_process(self, backend: ContainerBackend) -> None:
		mock_proc = MagicMock()
		mock_proc.returncode = 0
		mock_stdout = AsyncMock()
		mock_stdout.read = AsyncMock(return_value=b"container output\n")
		mock_proc.stdout = mock_stdout
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		output = await backend.get_output(handle)
		assert output == "container output\n"

	async def test_get_output_unknown_worker(self, backend: ContainerBackend) -> None:
		handle = WorkerHandle(worker_id="unknown")
		assert await backend.get_output(handle) == ""

	@patch("mission_control.backends.container.asyncio.create_subprocess_exec")
	async def test_kill_stops_container(
		self, mock_exec: AsyncMock, backend: ContainerBackend,
	) -> None:
		"""kill terminates docker process and calls docker stop as fallback."""
		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()
		backend._processes["w1"] = mock_proc
		backend._container_names["w1"] = "mc-worker-w1"

		mock_stop_proc = AsyncMock()
		mock_stop_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.return_value = mock_stop_proc

		handle = WorkerHandle(worker_id="w1")
		await backend.kill(handle)

		mock_proc.kill.assert_called_once()
		mock_proc.wait.assert_awaited_once()
		# docker stop should be called as fallback
		mock_exec.assert_awaited_once()
		call_args = mock_exec.call_args[0]
		assert "stop" in call_args
		assert "mc-worker-w1" in call_args

	async def test_kill_already_finished(self, backend: ContainerBackend) -> None:
		"""kill is a no-op for process that already finished."""
		mock_proc = MagicMock()
		mock_proc.returncode = 0
		mock_proc.kill = MagicMock()
		backend._processes["w1"] = mock_proc

		handle = WorkerHandle(worker_id="w1")
		await backend.kill(handle)
		mock_proc.kill.assert_not_called()

	async def test_release_workspace(self, backend: ContainerBackend) -> None:
		"""release_workspace delegates to pool.release."""
		await backend.release_workspace("/pool/clone-1")
		backend._pool.release.assert_awaited_once_with(Path("/pool/clone-1"))

	async def test_cleanup(self, backend: ContainerBackend) -> None:
		"""cleanup kills all running processes and clears state."""
		running = MagicMock()
		running.returncode = None
		running.kill = MagicMock()
		running.wait = AsyncMock()

		finished = MagicMock()
		finished.returncode = 0

		backend._processes["w1"] = running
		backend._processes["w2"] = finished
		backend._container_names["w1"] = "mc-worker-w1"
		backend._stdout_bufs["w1"] = b"data"

		await backend.cleanup()

		running.kill.assert_called_once()
		running.wait.assert_awaited_once()
		assert backend._processes == {}
		assert backend._container_names == {}
		assert backend._stdout_bufs == {}
		backend._pool.cleanup.assert_awaited_once()

	async def test_initialize_delegates_to_pool(self, backend: ContainerBackend) -> None:
		"""initialize delegates to pool.initialize."""
		await backend.initialize(warm_count=3)
		backend._pool.initialize.assert_awaited_once_with(warm_count=3)

	def test_build_docker_command_extra_volumes(self) -> None:
		"""Extra volumes are passed through to docker run."""
		cc = ContainerConfig(
			claude_config_dir="",
			extra_volumes=["/data:/data:ro", "/logs:/logs"],
		)
		with patch("mission_control.backends.container.WorkspacePool"):
			b = ContainerBackend(
				source_repo="/repo", pool_dir="/pool",
				container_config=cc,
			)
		with patch.dict("os.environ", {}, clear=True):
			import os
			os.environ.pop("CLAUDE_CONFIG_DIR", None)
			cmd = b._build_docker_command("w1", "/host/ws", ["echo"])

		cmd_str = " ".join(cmd)
		assert "/data:/data:ro" in cmd_str
		assert "/logs:/logs" in cmd_str

	def test_build_docker_command_network(self, backend: ContainerBackend) -> None:
		"""docker run includes --network flag."""
		cmd = backend._build_docker_command("w1", "/host/ws", ["echo"])
		net_idx = cmd.index("--network")
		assert cmd[net_idx + 1] == "bridge"

	def test_build_docker_command_container_name(self, backend: ContainerBackend) -> None:
		"""Container name follows mc-worker-{id} pattern."""
		backend._build_docker_command("abc123", "/host/ws", ["echo"])
		assert backend._container_names["abc123"] == "mc-worker-abc123"
