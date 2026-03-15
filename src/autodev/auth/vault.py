"""macOS Keychain wrapper for credential storage."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class KeychainVault:
	"""Store and retrieve credentials from macOS Keychain."""

	SERVICE_PREFIX = "autodev"

	def _service_name(self, service: str) -> str:
		return f"{self.SERVICE_PREFIX}/{service}"

	async def store(self, service: str, account: str, secret: str) -> None:
		"""Store a credential in Keychain.

		Uses stdin to pass the secret (not CLI args) to avoid exposure in ps output.
		"""
		proc = await asyncio.create_subprocess_exec(
			"security", "add-generic-password",
			"-U",
			"-s", self._service_name(service),
			"-a", account,
			"-w",
			stdin=asyncio.subprocess.PIPE,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		await proc.communicate(input=secret.encode())
		if proc.returncode != 0:
			raise RuntimeError(f"Failed to store credential for {service}/{account}")

	async def get(self, service: str, account: str) -> str | None:
		"""Retrieve a credential from Keychain. Returns None if not found."""
		proc = await asyncio.create_subprocess_exec(
			"security", "find-generic-password",
			"-s", self._service_name(service),
			"-a", account,
			"-w",
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, _ = await proc.communicate()
		if proc.returncode != 0:
			return None
		return stdout.decode().strip()

	async def delete(self, service: str, account: str) -> bool:
		"""Delete a credential from Keychain. Returns True on success."""
		proc = await asyncio.create_subprocess_exec(
			"security", "delete-generic-password",
			"-s", self._service_name(service),
			"-a", account,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		await proc.communicate()
		return proc.returncode == 0

	async def list_services(self) -> list[dict[str, str]]:
		"""List all autodev-managed credentials.

		Uses security find-generic-password -l (not dump-keychain which prompts).
		"""
		proc = await asyncio.create_subprocess_exec(
			"security", "find-generic-password",
			"-l", self.SERVICE_PREFIX,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, _ = await proc.communicate()
		if proc.returncode != 0:
			return []
		return self._parse_keychain_output(stdout.decode())

	@staticmethod
	def _parse_keychain_output(output: str) -> list[dict[str, str]]:
		"""Parse security CLI output for service/account pairs."""
		results: list[dict[str, str]] = []
		current: dict[str, str] = {}
		for line in output.splitlines():
			line = line.strip()
			if line.startswith('"svce"'):
				val = line.split("=", 1)
				if len(val) == 2:
					current["service"] = val[1].strip().strip('"')
			elif line.startswith('"acct"'):
				val = line.split("=", 1)
				if len(val) == 2:
					current["account"] = val[1].strip().strip('"')
			elif line.startswith("keychain:") and current:
				if "service" in current and "account" in current:
					results.append(current)
				current = {}
		if "service" in current and "account" in current:
			results.append(current)
		return results
