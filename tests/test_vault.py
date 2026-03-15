"""Tests for KeychainVault."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from autodev.auth.vault import KeychainVault


@pytest.fixture
def vault():
	return KeychainVault()


def _make_proc(returncode=0, stdout=b"", stderr=b""):
	"""Create a mock subprocess with given return values."""
	proc = AsyncMock()
	proc.returncode = returncode
	proc.communicate = AsyncMock(return_value=(stdout, stderr))
	return proc


class TestStore:
	@pytest.mark.asyncio
	async def test_store_passes_secret_via_stdin(self, vault):
		proc = _make_proc(returncode=0)
		with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
			await vault.store("github", "danny", "my-secret-token")

			mock_exec.assert_called_once_with(
				"security", "add-generic-password",
				"-U",
				"-s", "autodev/github",
				"-a", "danny",
				"-w",
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
			)
			proc.communicate.assert_called_once_with(input=b"my-secret-token")

	@pytest.mark.asyncio
	async def test_store_raises_on_failure(self, vault):
		proc = _make_proc(returncode=1)
		with patch("asyncio.create_subprocess_exec", return_value=proc):
			with pytest.raises(RuntimeError, match="Failed to store credential"):
				await vault.store("github", "danny", "secret")


class TestGet:
	@pytest.mark.asyncio
	async def test_get_returns_credential(self, vault):
		proc = _make_proc(returncode=0, stdout=b"my-secret-token\n")
		with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
			result = await vault.get("github", "danny")

			assert result == "my-secret-token"
			mock_exec.assert_called_once_with(
				"security", "find-generic-password",
				"-s", "autodev/github",
				"-a", "danny",
				"-w",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
			)

	@pytest.mark.asyncio
	async def test_get_returns_none_when_not_found(self, vault):
		proc = _make_proc(returncode=44)
		with patch("asyncio.create_subprocess_exec", return_value=proc):
			result = await vault.get("nonexistent", "nobody")
			assert result is None


class TestDelete:
	@pytest.mark.asyncio
	async def test_delete_returns_true_on_success(self, vault):
		proc = _make_proc(returncode=0)
		with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
			result = await vault.delete("github", "danny")

			assert result is True
			mock_exec.assert_called_once_with(
				"security", "delete-generic-password",
				"-s", "autodev/github",
				"-a", "danny",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
			)

	@pytest.mark.asyncio
	async def test_delete_returns_false_when_not_found(self, vault):
		proc = _make_proc(returncode=44)
		with patch("asyncio.create_subprocess_exec", return_value=proc):
			result = await vault.delete("nonexistent", "nobody")
			assert result is False


class TestListServices:
	@pytest.mark.asyncio
	async def test_list_services_parses_output(self, vault):
		keychain_output = (
			'keychain: "/Users/danny/Library/Keychains/login.keychain-db"\n'
			'class: "genp"\n'
			'attributes:\n'
			'    "svce"<blob>="autodev/github"\n'
			'    "acct"<blob>="danny"\n'
			'keychain: "/Users/danny/Library/Keychains/login.keychain-db"\n'
			'class: "genp"\n'
			'attributes:\n'
			'    "svce"<blob>="autodev/openai"\n'
			'    "acct"<blob>="api-user"\n'
		)
		proc = _make_proc(returncode=0, stdout=keychain_output.encode())
		with patch("asyncio.create_subprocess_exec", return_value=proc):
			result = await vault.list_services()

			assert result == [
				{"service": "autodev/github", "account": "danny"},
				{"service": "autodev/openai", "account": "api-user"},
			]

	@pytest.mark.asyncio
	async def test_list_services_returns_empty_on_failure(self, vault):
		proc = _make_proc(returncode=44)
		with patch("asyncio.create_subprocess_exec", return_value=proc):
			result = await vault.list_services()
			assert result == []


class TestServicePrefix:
	@pytest.mark.asyncio
	async def test_service_prefix_applied(self, vault):
		"""All methods namespace services under autodev/ prefix."""
		proc = _make_proc(returncode=0, stdout=b"token\n")
		with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
			await vault.get("my-service", "user")
			args = mock_exec.call_args[0]
			assert "autodev/my-service" in args

	def test_service_name_helper(self, vault):
		assert vault._service_name("github") == "autodev/github"
		assert vault._service_name("openai") == "autodev/openai"
