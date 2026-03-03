"""Tests for SSHBackend retry logic, workspace verification, and timeout handling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.backends.ssh import _SSH_CONNECTION_FAILURE_RC, SSHBackend
from mission_control.config import SSHHostConfig


def _make_backend(**kwargs) -> SSHBackend:
	host = SSHHostConfig(hostname="remote1", user="deploy", max_workers=4)
	defaults = {"hosts": [host], "connection_timeout": 5.0, "base_delay": 0.01}
	defaults.update(kwargs)
	return SSHBackend(**defaults)


def _fake_proc(returncode: int, stdout: bytes = b"") -> MagicMock:
	proc = MagicMock()
	proc.returncode = returncode
	fut: asyncio.Future[tuple[bytes, bytes]] = asyncio.get_event_loop().create_future()
	fut.set_result((stdout, b""))
	proc.communicate = MagicMock(return_value=fut)
	proc.kill = MagicMock()
	proc.wait = AsyncMock()
	return proc


# ---------------------------------------------------------------------------
# _run_ssh_command -- retry on transient failure
# ---------------------------------------------------------------------------


class TestRunSSHCommandRetry:
	@pytest.mark.asyncio
	async def test_success_no_retry(self) -> None:
		"""Successful command returns immediately without retrying."""
		backend = _make_backend()
		proc = _fake_proc(0, b"ok")
		with patch("asyncio.create_subprocess_exec", return_value=proc):
			rc, output = await backend._run_ssh_command("deploy@remote1", "echo hi")
		assert rc == 0
		assert output == "ok"

	@pytest.mark.asyncio
	async def test_retry_on_connection_failure(self) -> None:
		"""rc=255 triggers retries up to max_retries, then returns failure."""
		backend = _make_backend(max_retries=2)
		proc = _fake_proc(_SSH_CONNECTION_FAILURE_RC, b"Connection refused")
		call_count = 0

		async def fake_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			return proc

		with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
			rc, output = await backend._run_ssh_command("deploy@remote1", "ls")

		assert rc == _SSH_CONNECTION_FAILURE_RC
		# 1 initial + 2 retries = 3 total
		assert call_count == 3

	@pytest.mark.asyncio
	async def test_transient_then_success(self) -> None:
		"""Transient failure followed by success returns success."""
		backend = _make_backend(max_retries=3)
		fail_proc = _fake_proc(_SSH_CONNECTION_FAILURE_RC, b"fail")
		ok_proc = _fake_proc(0, b"done")
		call_count = 0

		async def fake_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			return fail_proc if call_count <= 2 else ok_proc

		with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
			rc, output = await backend._run_ssh_command("deploy@remote1", "test")

		assert rc == 0
		assert output == "done"
		assert call_count == 3

	@pytest.mark.asyncio
	async def test_no_retry_on_permanent_failure(self) -> None:
		"""Non-255, non-zero rc fails immediately without retry."""
		backend = _make_backend(max_retries=3)
		proc = _fake_proc(1, b"command not found")
		call_count = 0

		async def fake_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			return proc

		with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
			rc, output = await backend._run_ssh_command("deploy@remote1", "bad-cmd")

		assert rc == 1
		assert call_count == 1

	@pytest.mark.asyncio
	async def test_no_retry_on_other_nonzero(self) -> None:
		"""rc=127 (command not found) is permanent -- no retry."""
		backend = _make_backend(max_retries=3)
		proc = _fake_proc(127, b"not found")
		call_count = 0

		async def fake_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			return proc

		with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
			rc, output = await backend._run_ssh_command("deploy@remote1", "missing")

		assert rc == 127
		assert call_count == 1


# ---------------------------------------------------------------------------
# _run_ssh_command -- timeout handling
# ---------------------------------------------------------------------------


class TestRunSSHCommandTimeout:
	@pytest.mark.asyncio
	async def test_timeout_retries_then_raises(self) -> None:
		"""Timeout on every attempt raises TimeoutError after exhausting retries."""
		backend = _make_backend(max_retries=1, connection_timeout=0.01)
		proc = MagicMock()
		proc.returncode = None
		proc.kill = MagicMock()
		proc.wait = AsyncMock()

		async def slow_communicate():
			await asyncio.sleep(10)
			return (b"", b"")

		proc.communicate = slow_communicate
		call_count = 0

		async def fake_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			return proc

		with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
			with pytest.raises(asyncio.TimeoutError, match="timed out after 2 attempts"):
				await backend._run_ssh_command("deploy@remote1", "slow")

		assert call_count == 2

	@pytest.mark.asyncio
	async def test_timeout_then_success(self) -> None:
		"""Timeout on first attempt, success on second."""
		backend = _make_backend(max_retries=2, connection_timeout=0.05)

		slow_proc = MagicMock()
		slow_proc.returncode = None
		slow_proc.kill = MagicMock()
		slow_proc.wait = AsyncMock()

		async def slow_communicate():
			await asyncio.sleep(10)
			return (b"", b"")

		slow_proc.communicate = slow_communicate

		ok_proc = _fake_proc(0, b"ok")
		call_count = 0

		async def fake_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			return slow_proc if call_count == 1 else ok_proc

		with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
			rc, output = await backend._run_ssh_command("deploy@remote1", "cmd")

		assert rc == 0
		assert output == "ok"
		assert call_count == 2

	@pytest.mark.asyncio
	async def test_custom_timeout_override(self) -> None:
		"""Per-call timeout overrides the default connection_timeout."""
		backend = _make_backend(connection_timeout=100.0)
		proc = MagicMock()
		proc.returncode = None
		proc.kill = MagicMock()
		proc.wait = AsyncMock()

		async def slow_communicate():
			await asyncio.sleep(10)
			return (b"", b"")

		proc.communicate = slow_communicate

		async def fake_exec(*args, **kwargs):
			return proc

		with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
			with pytest.raises(asyncio.TimeoutError):
				await backend._run_ssh_command(
					"deploy@remote1", "cmd", timeout=0.01, max_retries=0
				)


# ---------------------------------------------------------------------------
# _verify_remote_workspace
# ---------------------------------------------------------------------------


class TestVerifyRemoteWorkspace:
	_PATCH_TARGET = "asyncio.subprocess.create_subprocess_exec"

	@pytest.mark.asyncio
	async def test_verify_success(self) -> None:
		"""Returns True when remote dir exists and git status is clean."""
		backend = _make_backend()
		proc = _fake_proc(0, b"")
		with patch(self._PATCH_TARGET, return_value=proc):
			result = await backend._verify_remote_workspace(
				"deploy@remote1", "/tmp/mc-worker-w1"
			)
		assert result is True

	@pytest.mark.asyncio
	async def test_verify_failure_nonzero(self) -> None:
		"""Returns False when remote command fails (dir missing, etc.)."""
		backend = _make_backend()
		proc = _fake_proc(1, b"not a git repo")
		with patch(self._PATCH_TARGET, return_value=proc):
			result = await backend._verify_remote_workspace(
				"deploy@remote1", "/tmp/mc-worker-w1"
			)
		assert result is False

	@pytest.mark.asyncio
	async def test_verify_timeout_returns_false(self) -> None:
		"""Returns False on timeout rather than raising."""
		backend = _make_backend(connection_timeout=0.01)
		proc = MagicMock()
		proc.returncode = None
		proc.kill = MagicMock()
		proc.wait = AsyncMock()

		async def slow_communicate():
			await asyncio.sleep(10)
			return (b"", b"")

		proc.communicate = slow_communicate

		async def fake_exec(*args, **kwargs):
			return proc

		with patch(self._PATCH_TARGET, side_effect=fake_exec):
			result = await backend._verify_remote_workspace(
				"deploy@remote1", "/tmp/mc-worker-w1"
			)
		assert result is False


# ---------------------------------------------------------------------------
# _repair_workspace
# ---------------------------------------------------------------------------


class TestRepairWorkspace:
	@pytest.mark.asyncio
	async def test_repair_success(self) -> None:
		"""Successful re-clone returns True."""
		backend = _make_backend()
		proc = _fake_proc(0, b"Cloning into...")
		with patch("asyncio.create_subprocess_exec", return_value=proc):
			result = await backend._repair_workspace(
				"deploy@remote1", "/tmp/mc-worker-w1", "git@host:repo.git", "main"
			)
		assert result is True

	@pytest.mark.asyncio
	async def test_repair_clone_failure(self) -> None:
		"""Clone failure returns False."""
		backend = _make_backend()
		proc = _fake_proc(128, b"fatal: repo not found")
		with patch("asyncio.create_subprocess_exec", return_value=proc):
			result = await backend._repair_workspace(
				"deploy@remote1", "/tmp/mc-worker-w1", "git@host:repo.git", "main"
			)
		assert result is False

	@pytest.mark.asyncio
	async def test_repair_timeout(self) -> None:
		"""Timeout during repair returns False."""
		backend = _make_backend(connection_timeout=0.01)
		proc = MagicMock()
		proc.returncode = None
		proc.kill = MagicMock()
		proc.wait = AsyncMock()

		async def slow_communicate():
			await asyncio.sleep(10)
			return (b"", b"")

		proc.communicate = slow_communicate

		async def fake_exec(*args, **kwargs):
			return proc

		with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
			result = await backend._repair_workspace(
				"deploy@remote1", "/tmp/mc-worker-w1", "git@host:repo.git", "main"
			)
		assert result is False


# ---------------------------------------------------------------------------
# spawn calls _verify_remote_workspace
# ---------------------------------------------------------------------------


class TestSpawnVerification:
	@pytest.mark.asyncio
	async def test_spawn_verifies_workspace(self) -> None:
		"""spawn() calls _verify_remote_workspace before starting the worker."""
		backend = _make_backend()
		metadata = '{"hostname": "remote1", "user": "deploy"}'
		workspace_path = f"/tmp/mc-worker-w1::{metadata}"

		verify_called = False

		async def mock_verify(ssh_target, remote_path):
			nonlocal verify_called
			verify_called = True
			return True

		backend._verify_remote_workspace = mock_verify

		proc = MagicMock()
		proc.pid = 12345
		proc.returncode = None
		proc.stdout = AsyncMock()

		with patch("asyncio.create_subprocess_exec", return_value=proc):
			handle = await backend.spawn("w1", workspace_path, ["echo", "hi"], 60)

		assert verify_called
		assert handle.worker_id == "w1"


# ---------------------------------------------------------------------------
# provision_workspace uses _run_ssh_command
# ---------------------------------------------------------------------------


class TestProvisionWorkspace:
	@pytest.mark.asyncio
	async def test_provision_retries_on_ssh_failure(self) -> None:
		"""provision_workspace retries via _run_ssh_command on rc=255."""
		backend = _make_backend(max_retries=1)
		fail_proc = _fake_proc(_SSH_CONNECTION_FAILURE_RC, b"Connection refused")
		ok_proc = _fake_proc(0, b"Cloning...")
		call_count = 0

		async def fake_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			return fail_proc if call_count == 1 else ok_proc

		with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
			result = await backend.provision_workspace("w1", "git@host:repo.git", "main")

		assert "/tmp/mc-worker-w1" in result
		assert call_count == 2

	@pytest.mark.asyncio
	async def test_provision_raises_on_permanent_failure(self) -> None:
		"""provision_workspace raises RuntimeError on non-retryable failure."""
		backend = _make_backend()
		proc = _fake_proc(128, b"fatal: repo not found")

		with patch("asyncio.create_subprocess_exec", return_value=proc):
			with pytest.raises(RuntimeError, match="Failed to provision workspace"):
				await backend.provision_workspace("w1", "git@host:repo.git", "main")


# ---------------------------------------------------------------------------
# connection_timeout parameter
# ---------------------------------------------------------------------------


class TestConnectionTimeout:
	def test_default_timeout(self) -> None:
		"""Default connection_timeout is 30s."""
		host = SSHHostConfig(hostname="remote1", user="deploy", max_workers=4)
		backend = SSHBackend(hosts=[host])
		assert backend._connection_timeout == 30.0

	def test_custom_timeout(self) -> None:
		"""Custom connection_timeout is stored correctly."""
		backend = _make_backend(connection_timeout=60.0)
		assert backend._connection_timeout == 60.0
