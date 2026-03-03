"""Tests for ContainerBackend health pre-flight checks and recovery."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.backends.container import ContainerBackend
from mission_control.backends.local import HealthCheckResult
from mission_control.config import ContainerConfig


@pytest.fixture()
def container_config() -> ContainerConfig:
	return ContainerConfig(
		image="mission-control-worker:latest",
		docker_executable="docker",
		workspace_mount="/workspace",
	)


@pytest.fixture()
def backend(container_config: ContainerConfig) -> ContainerBackend:
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
			container_health_timeout=10,
		)
	return b


class TestContainerHealthCheck:
	"""Tests for _verify_container_workspace()."""

	async def test_healthy_workspace_passes(self, backend: ContainerBackend, tmp_path: Path) -> None:
		"""No stale container + valid .git/HEAD + git status ok -> passes."""
		backend._container_names["w1"] = "mc-worker-w1"

		# Set up valid git workspace on host
		git_dir = tmp_path / ".git"
		git_dir.mkdir()
		(git_dir / "HEAD").write_text("ref: refs/heads/main\n")

		with patch("mission_control.backends.container.asyncio.create_subprocess_exec") as mock_exec:
			# docker inspect returns non-zero (no stale container)
			inspect_proc = AsyncMock()
			inspect_proc.returncode = 1
			inspect_proc.communicate = AsyncMock(return_value=(b"Error: No such object", None))
			# git status returns 0
			status_proc = AsyncMock()
			status_proc.returncode = 0
			status_proc.communicate = AsyncMock(return_value=(b"", None))
			mock_exec.side_effect = [inspect_proc, status_proc]

			result = await backend._verify_container_workspace("w1", str(tmp_path))

		assert result.passed is True
		assert result.issues == []

	async def test_stale_container_fails(self, backend: ContainerBackend, tmp_path: Path) -> None:
		"""docker inspect finds a stale container -> health check fails."""
		backend._container_names["w1"] = "mc-worker-w1"

		git_dir = tmp_path / ".git"
		git_dir.mkdir()
		(git_dir / "HEAD").write_text("ref: refs/heads/main\n")

		with patch("mission_control.backends.container.asyncio.create_subprocess_exec") as mock_exec:
			# docker inspect returns 0 (stale container exists)
			inspect_proc = AsyncMock()
			inspect_proc.returncode = 0
			inspect_proc.wait = AsyncMock(return_value=0)
			# git status ok
			status_proc = AsyncMock()
			status_proc.returncode = 0
			status_proc.wait = AsyncMock(return_value=0)
			mock_exec.side_effect = [inspect_proc, status_proc]

			result = await backend._verify_container_workspace("w1", str(tmp_path))

		assert result.passed is False
		assert any("Stale container" in issue for issue in result.issues)

	async def test_git_head_missing_fails(self, backend: ContainerBackend, tmp_path: Path) -> None:
		""".git/HEAD missing on host -> health check fails."""
		backend._container_names["w1"] = "mc-worker-w1"
		# No .git directory at all

		with patch("mission_control.backends.container.asyncio.create_subprocess_exec") as mock_exec:
			inspect_proc = AsyncMock()
			inspect_proc.returncode = 1
			inspect_proc.communicate = AsyncMock(return_value=(b"", None))
			status_proc = AsyncMock()
			status_proc.returncode = 0
			status_proc.communicate = AsyncMock(return_value=(b"", None))
			mock_exec.side_effect = [inspect_proc, status_proc]

			result = await backend._verify_container_workspace("w1", str(tmp_path))

		assert result.passed is False
		assert any(".git/HEAD missing" in issue for issue in result.issues)

	async def test_git_status_fails(self, backend: ContainerBackend, tmp_path: Path) -> None:
		"""git status --porcelain non-zero on host -> health check fails."""
		backend._container_names["w1"] = "mc-worker-w1"

		git_dir = tmp_path / ".git"
		git_dir.mkdir()
		(git_dir / "HEAD").write_text("ref: refs/heads/main\n")

		with patch("mission_control.backends.container.asyncio.create_subprocess_exec") as mock_exec:
			inspect_proc = AsyncMock()
			inspect_proc.returncode = 1
			inspect_proc.communicate = AsyncMock(return_value=(b"", None))
			status_proc = AsyncMock()
			status_proc.returncode = 128
			status_proc.communicate = AsyncMock(return_value=(b"fatal: bad", None))
			mock_exec.side_effect = [inspect_proc, status_proc]

			result = await backend._verify_container_workspace("w1", str(tmp_path))

		assert result.passed is False
		assert any("git status failed" in issue for issue in result.issues)

	async def test_timeout_handling(self, backend: ContainerBackend, tmp_path: Path) -> None:
		"""Subprocess timeout -> health check reports timeout issue."""
		backend._container_names["w1"] = "mc-worker-w1"
		backend._container_health_timeout = 1

		git_dir = tmp_path / ".git"
		git_dir.mkdir()
		(git_dir / "HEAD").write_text("ref: refs/heads/main\n")

		with patch("mission_control.backends.container.asyncio.create_subprocess_exec") as mock_exec:
			mock_proc = AsyncMock()
			mock_proc.communicate = AsyncMock(return_value=(b"", None))
			mock_exec.return_value = mock_proc

			with patch("mission_control.backends.container.asyncio.wait_for", side_effect=asyncio.TimeoutError):
				result = await backend._verify_container_workspace("w1", str(tmp_path))

		assert result.passed is False
		assert any("timed out" in issue for issue in result.issues)

	async def test_logs_health_check_result(
		self, backend: ContainerBackend, tmp_path: Path, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""Health check results are logged at INFO with container ID and workspace."""
		backend._container_names["w1"] = "mc-worker-w1"

		git_dir = tmp_path / ".git"
		git_dir.mkdir()
		(git_dir / "HEAD").write_text("ref: refs/heads/main\n")

		with patch("mission_control.backends.container.asyncio.create_subprocess_exec") as mock_exec:
			inspect_proc = AsyncMock()
			inspect_proc.returncode = 1
			inspect_proc.communicate = AsyncMock(return_value=(b"", None))
			status_proc = AsyncMock()
			status_proc.returncode = 0
			status_proc.communicate = AsyncMock(return_value=(b"", None))
			mock_exec.side_effect = [inspect_proc, status_proc]

			with caplog.at_level(logging.INFO, logger="mission_control.backends.container"):
				await backend._verify_container_workspace("w1", str(tmp_path))

		ws_path = str(tmp_path)
		assert any(
			"mc-worker-w1" in msg and ws_path in msg and "passed=True" in msg
			for msg in caplog.messages
		)

	async def test_uses_default_container_name(self, backend: ContainerBackend, tmp_path: Path) -> None:
		"""When no container name is stored, uses default mc-worker-{id} pattern."""
		git_dir = tmp_path / ".git"
		git_dir.mkdir()
		(git_dir / "HEAD").write_text("ref: refs/heads/main\n")

		with patch("mission_control.backends.container.asyncio.create_subprocess_exec") as mock_exec:
			inspect_proc = AsyncMock()
			inspect_proc.returncode = 1
			inspect_proc.communicate = AsyncMock(return_value=(b"", None))
			status_proc = AsyncMock()
			status_proc.returncode = 0
			status_proc.communicate = AsyncMock(return_value=(b"", None))
			mock_exec.side_effect = [inspect_proc, status_proc]

			result = await backend._verify_container_workspace("w1", str(tmp_path))

		assert result.passed is True
		calls = mock_exec.call_args_list
		assert any("mc-worker-w1" in str(call) for call in calls)


class TestContainerSpawnHealthCheck:
	"""Tests for health check integration in spawn()."""

	@patch("mission_control.backends.container.asyncio.create_subprocess_exec")
	async def test_healthy_spawn_proceeds(
		self, mock_exec: AsyncMock, backend: ContainerBackend,
	) -> None:
		"""spawn() succeeds when health check passes."""
		mock_proc = AsyncMock()
		mock_proc.pid = 9999
		mock_exec.return_value = mock_proc

		backend._verify_container_workspace = AsyncMock(
			return_value=HealthCheckResult(passed=True)
		)

		handle = await backend.spawn("w1", "/host/ws", ["cmd"], 120)

		assert handle.worker_id == "w1"
		assert handle.pid == 9999
		backend._verify_container_workspace.assert_awaited_once_with("w1", "/host/ws")

	@patch("mission_control.backends.container.asyncio.create_subprocess_exec")
	async def test_unhealthy_triggers_repair_then_succeeds(
		self, mock_exec: AsyncMock, backend: ContainerBackend,
	) -> None:
		"""spawn() repairs container when first health check fails, then succeeds."""
		mock_proc = AsyncMock()
		mock_proc.pid = 8888
		mock_exec.return_value = mock_proc

		bad_result = HealthCheckResult(passed=False, issues=["stale container found"])
		good_result = HealthCheckResult(passed=True)
		backend._verify_container_workspace = AsyncMock(side_effect=[bad_result, good_result])
		backend._repair_container = AsyncMock()

		handle = await backend.spawn("w1", "/host/ws", ["cmd"], 120)

		assert handle.worker_id == "w1"
		backend._repair_container.assert_awaited_once_with("w1")
		assert backend._verify_container_workspace.await_count == 2

	@patch("mission_control.backends.container.asyncio.create_subprocess_exec")
	async def test_repair_failure_raises(
		self, mock_exec: AsyncMock, backend: ContainerBackend,
	) -> None:
		"""spawn() raises RuntimeError when repair fails to restore health."""
		bad1 = HealthCheckResult(passed=False, issues=["stale container"])
		bad2 = HealthCheckResult(passed=False, issues=["still stale"])
		backend._verify_container_workspace = AsyncMock(side_effect=[bad1, bad2])
		backend._repair_container = AsyncMock()

		with pytest.raises(RuntimeError, match="health check failed after repair"):
			await backend.spawn("w1", "/host/ws", ["cmd"], 120)

		backend._repair_container.assert_awaited_once_with("w1")

	@patch("mission_control.backends.container.asyncio.create_subprocess_exec")
	async def test_repair_stops_and_removes_container(
		self, mock_exec: AsyncMock, backend: ContainerBackend,
	) -> None:
		"""_repair_container issues docker stop and docker rm."""
		backend._container_names["w1"] = "mc-worker-w1"

		stop_proc = AsyncMock()
		stop_proc.communicate = AsyncMock(return_value=(b"", None))
		rm_proc = AsyncMock()
		rm_proc.communicate = AsyncMock(return_value=(b"", None))
		mock_exec.side_effect = [stop_proc, rm_proc]

		await backend._repair_container("w1")

		assert mock_exec.await_count == 2
		# First call: docker stop
		stop_call = mock_exec.call_args_list[0]
		assert "stop" in stop_call[0]
		assert "mc-worker-w1" in stop_call[0]
		# Second call: docker rm --force
		rm_call = mock_exec.call_args_list[1]
		assert "rm" in rm_call[0]
		assert "--force" in rm_call[0]
		assert "mc-worker-w1" in rm_call[0]

	@patch("mission_control.backends.container.asyncio.create_subprocess_exec")
	async def test_repair_handles_timeout(
		self, mock_exec: AsyncMock, backend: ContainerBackend, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""_repair_container handles timeouts from docker stop/rm gracefully."""
		backend._container_names["w1"] = "mc-worker-w1"

		async def timeout_exec(*args, **kwargs):
			proc = AsyncMock()
			proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
			return proc

		mock_exec.side_effect = timeout_exec

		with patch(
			"mission_control.backends.container.asyncio.wait_for",
			side_effect=asyncio.TimeoutError,
		):
			with caplog.at_level(logging.WARNING, logger="mission_control.backends.container"):
				await backend._repair_container("w1")

		assert any("docker stop failed" in msg or "docker rm failed" in msg for msg in caplog.messages)


class TestContainerHealthTimeout:
	"""Tests for container_health_timeout config parameter."""

	def test_default_timeout(self) -> None:
		"""Default container_health_timeout is 10 seconds."""
		with patch("mission_control.backends.container.WorkspacePool"):
			b = ContainerBackend(
				source_repo="/repo",
				pool_dir="/pool",
				container_config=ContainerConfig(),
			)
		assert b._container_health_timeout == 10

	def test_custom_timeout(self) -> None:
		"""container_health_timeout can be overridden."""
		with patch("mission_control.backends.container.WorkspacePool"):
			b = ContainerBackend(
				source_repo="/repo",
				pool_dir="/pool",
				container_config=ContainerConfig(),
				container_health_timeout=30,
			)
		assert b._container_health_timeout == 30
