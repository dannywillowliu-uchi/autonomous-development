"""Tests for worker execution backends (local, ssh, container)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.backends.base import WorkerHandle
from mission_control.backends.container import ContainerBackend
from mission_control.backends.local import LocalBackend
from mission_control.backends.ssh import SSHBackend
from mission_control.config import SSHHostConfig

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
# ContainerBackend -- all methods raise NotImplementedError
# ---------------------------------------------------------------------------

class TestContainerBackend:
	@pytest.fixture()
	def backend(self) -> ContainerBackend:
		return ContainerBackend()

	async def test_provision_workspace_raises(self, backend: ContainerBackend) -> None:
		with pytest.raises(NotImplementedError):
			await backend.provision_workspace("w1", "/repo", "main")

	async def test_spawn_raises(self, backend: ContainerBackend) -> None:
		with pytest.raises(NotImplementedError):
			await backend.spawn("w1", "/ws", ["echo", "hi"], 60)

	async def test_check_status_raises(self, backend: ContainerBackend) -> None:
		handle = WorkerHandle(worker_id="w1")
		with pytest.raises(NotImplementedError):
			await backend.check_status(handle)

	async def test_get_output_raises(self, backend: ContainerBackend) -> None:
		handle = WorkerHandle(worker_id="w1")
		with pytest.raises(NotImplementedError):
			await backend.get_output(handle)

	async def test_kill_raises(self, backend: ContainerBackend) -> None:
		handle = WorkerHandle(worker_id="w1")
		with pytest.raises(NotImplementedError):
			await backend.kill(handle)

	async def test_release_workspace_raises(self, backend: ContainerBackend) -> None:
		with pytest.raises(NotImplementedError):
			await backend.release_workspace("/ws")

	async def test_cleanup_raises(self, backend: ContainerBackend) -> None:
		with pytest.raises(NotImplementedError):
			await backend.cleanup()


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
		"""provision_workspace acquires from pool and runs git checkout."""
		backend._pool.acquire.return_value = Path("/pool/clone-1")

		# Mock the git checkout subprocess
		mock_proc = AsyncMock()
		mock_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.return_value = mock_proc

		result = await backend.provision_workspace("w1", "/repo", "main")

		assert result == "/pool/clone-1"
		backend._pool.acquire.assert_awaited_once()
		mock_exec.assert_awaited_once()
		# The command should be git checkout -b mc/unit-w1
		args = mock_exec.call_args
		assert args[0] == ("git", "checkout", "-b", "mc/unit-w1")
		assert args[1]["cwd"] == "/pool/clone-1"

	async def test_provision_workspace_no_workspace_available(
		self, backend: LocalBackend,
	) -> None:
		"""provision_workspace raises RuntimeError when pool returns None."""
		backend._pool.acquire.return_value = None

		with pytest.raises(RuntimeError, match="No workspace available"):
			await backend.provision_workspace("w1", "/repo", "main")

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
		mock_exec.assert_awaited_once_with(
			"claude", "code", "--task", "fix",
			cwd="/ws",
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		assert "w1" in backend._processes
		assert backend._stdout_bufs["w1"] == b""

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
		# Ensure the buffer key does NOT exist so the branch reads from proc.stdout
		backend._stdout_bufs.pop("w1", None)

		handle = WorkerHandle(worker_id="w1")
		output = await backend.get_output(handle)
		assert output == "hello world\n"

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
		assert "rm -rf /tmp/mc-worker-w1" in call_args[2]
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
