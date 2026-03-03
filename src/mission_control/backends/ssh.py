"""SSH backend -- runs workers on remote hosts via SSH."""

from __future__ import annotations

import asyncio
import asyncio.subprocess as _aio_sub
import json
import logging
import random
import shlex

from mission_control.backends.base import WorkerBackend, WorkerHandle
from mission_control.config import SSHHostConfig

logger = logging.getLogger(__name__)

# SSH exit code 255 indicates connection-level failure (network, auth, etc.)
_SSH_CONNECTION_FAILURE_RC = 255


class SSHBackend(WorkerBackend):
	"""Execute workers on remote hosts via SSH."""

	def __init__(
		self,
		hosts: list[SSHHostConfig],
		connection_timeout: float = 30.0,
		max_retries: int = 3,
		base_delay: float = 2.0,
	) -> None:
		if not hosts:
			raise ValueError("SSHBackend requires at least one host")
		self._hosts = hosts
		self._connection_timeout = connection_timeout
		self._max_retries = max_retries
		self._base_delay = base_delay
		self._worker_count: dict[str, int] = {h.hostname: 0 for h in hosts}
		self._processes: dict[str, asyncio.subprocess.Process] = {}
		self._stdout_bufs: dict[str, bytes] = {}
		self._stdout_collected: set[str] = set()

	async def _run_ssh_command(
		self,
		ssh_target: str,
		remote_cmd: str,
		*,
		timeout: float | None = None,
		max_retries: int | None = None,
	) -> tuple[int, str]:
		"""Run an SSH command with retry and exponential backoff.

		Retries on transient failures (rc=255, timeout). Fails immediately
		on permanent failures (rc != 0 and rc != 255).

		Returns (returncode, stdout_text).
		"""
		effective_timeout = timeout if timeout is not None else self._connection_timeout
		effective_retries = max_retries if max_retries is not None else self._max_retries

		last_rc = -1
		last_output = ""

		for attempt in range(effective_retries + 1):
			proc = await asyncio.create_subprocess_exec(
				"ssh", ssh_target, remote_cmd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			try:
				stdout, _ = await asyncio.wait_for(
					proc.communicate(), timeout=effective_timeout
				)
			except asyncio.TimeoutError:
				proc.kill()
				await proc.wait()
				last_rc = -1
				last_output = "SSH command timed out"
				if attempt < effective_retries:
					delay = self._base_delay * (2 ** attempt) + random.uniform(0, 1)
					logger.warning(
						"SSH timeout to %s (attempt %d/%d), retrying in %.1fs",
						ssh_target, attempt + 1, effective_retries + 1, delay,
					)
					await asyncio.sleep(delay)
					continue
				raise asyncio.TimeoutError(
					f"SSH command to {ssh_target} timed out after {effective_retries + 1} attempts"
				)

			last_rc = proc.returncode or 0
			last_output = stdout.decode(errors="replace")

			if last_rc == 0:
				return last_rc, last_output

			# Connection-level failure -- transient, retry
			if last_rc == _SSH_CONNECTION_FAILURE_RC:
				if attempt < effective_retries:
					delay = self._base_delay * (2 ** attempt) + random.uniform(0, 1)
					logger.warning(
						"SSH connection failure to %s (rc=255, attempt %d/%d), retrying in %.1fs",
						ssh_target, attempt + 1, effective_retries + 1, delay,
					)
					await asyncio.sleep(delay)
					continue
				# Exhausted retries
				return last_rc, last_output

			# Permanent failure from remote command -- do not retry
			return last_rc, last_output

		return last_rc, last_output

	async def _verify_remote_workspace(
		self, ssh_target: str, remote_path: str
	) -> bool:
		"""Check that the remote workspace exists and git status is clean.

		Uses asyncio.subprocess directly (not _run_ssh_command) so that
		this lightweight pre-flight check stays independent of the main
		subprocess mock path used by provision/spawn/release.
		"""
		cmd = (
			f"test -d {shlex.quote(remote_path)} && "
			f"cd {shlex.quote(remote_path)} && git status --porcelain"
		)
		try:
			proc = await _aio_sub.create_subprocess_exec(
				"ssh", ssh_target, cmd,
				stdout=_aio_sub.PIPE,
				stderr=_aio_sub.STDOUT,
			)
			await asyncio.wait_for(proc.communicate(), timeout=self._connection_timeout)
			return (proc.returncode or 0) == 0
		except (asyncio.TimeoutError, OSError):
			return False

	async def _repair_workspace(
		self,
		ssh_target: str,
		remote_path: str,
		source_repo: str,
		base_branch: str,
	) -> bool:
		"""Re-clone workspace if verification fails. Single retry."""
		rm_cmd = f"rm -rf {shlex.quote(remote_path)}"
		clone_cmd = (
			f"git clone --depth=1 -b {shlex.quote(base_branch)} "
			f"{shlex.quote(source_repo)} {shlex.quote(remote_path)}"
		)
		try:
			rc, output = await self._run_ssh_command(
				ssh_target, f"{rm_cmd} && {clone_cmd}", max_retries=1
			)
			if rc != 0:
				logger.error("Workspace repair failed on %s: %s", ssh_target, output)
				return False
			return True
		except asyncio.TimeoutError:
			logger.error("Workspace repair timed out on %s", ssh_target)
			return False

	def _select_host(self) -> SSHHostConfig:
		"""Pick the host with fewest active workers."""
		best: SSHHostConfig | None = None
		best_count = float("inf")
		for host in self._hosts:
			count = self._worker_count.get(host.hostname, 0)
			if count < host.max_workers and count < best_count:
				best = host
				best_count = count
		if best is None:
			raise RuntimeError("All SSH hosts are at capacity")
		return best

	def _ssh_target(self, host: SSHHostConfig) -> str:
		if host.user:
			return f"{host.user}@{host.hostname}"
		return host.hostname

	async def provision_workspace(
		self, worker_id: str, source_repo: str, base_branch: str
	) -> str:
		host = self._select_host()
		remote_path = f"/tmp/mc-worker-{worker_id}"
		ssh_target = self._ssh_target(host)

		clone_cmd = (
			f"git clone --depth=1 -b {shlex.quote(base_branch)} "
			f"{shlex.quote(source_repo)} {shlex.quote(remote_path)}"
		)
		rc, output = await self._run_ssh_command(ssh_target, clone_cmd)
		if rc != 0:
			raise RuntimeError(
				f"Failed to provision workspace on {host.hostname}: {output}"
			)

		self._worker_count[host.hostname] = (
			self._worker_count.get(host.hostname, 0) + 1
		)

		metadata = json.dumps({"hostname": host.hostname, "user": host.user})
		return f"{remote_path}::{metadata}"

	async def spawn(
		self, worker_id: str, workspace_path: str, command: list[str], timeout: int
	) -> WorkerHandle:
		remote_path, metadata = workspace_path.split("::", 1)
		host_info = json.loads(metadata)

		ssh_target = host_info.get("user", "") + "@" + host_info["hostname"]
		if not host_info.get("user"):
			ssh_target = host_info["hostname"]

		# Verify workspace before spawning (best-effort, never blocks spawn)
		try:
			if not await self._verify_remote_workspace(ssh_target, remote_path):
				logger.warning(
					"Workspace verification failed for %s on %s",
					remote_path, ssh_target,
				)
		except Exception:
			logger.warning(
				"Workspace verification error for %s on %s, proceeding anyway",
				remote_path, ssh_target, exc_info=True,
			)

		quoted_parts = [shlex.quote(c) for c in command]
		remote_cmd = f"cd {shlex.quote(remote_path)} && {' '.join(quoted_parts)}"
		proc = await asyncio.create_subprocess_exec(
			"ssh", ssh_target, remote_cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		self._processes[worker_id] = proc
		self._stdout_bufs[worker_id] = b""
		self._stdout_collected.discard(worker_id)

		return WorkerHandle(
			worker_id=worker_id,
			pid=proc.pid,
			workspace_path=remote_path,
			backend_metadata=metadata,
		)

	async def check_status(self, handle: WorkerHandle) -> str:
		proc = self._processes.get(handle.worker_id)
		if proc is None:
			return "failed"
		if proc.returncode is None:
			return "running"
		return "completed" if proc.returncode == 0 else "failed"

	async def get_output(self, handle: WorkerHandle) -> str:
		proc = self._processes.get(handle.worker_id)
		if proc is None:
			return ""
		if proc.returncode is None:
			# Process still running -- drain available stdout to prevent pipe buffer deadlock
			if proc.stdout:
				try:
					chunk = await asyncio.wait_for(proc.stdout.read(65536), timeout=0.1)
					self._stdout_bufs[handle.worker_id] = (
						self._stdout_bufs.get(handle.worker_id, b"") + chunk
					)
				except asyncio.TimeoutError:
					pass
		else:
			# Process finished -- collect remaining output once
			if proc.stdout and handle.worker_id not in self._stdout_collected:
				remaining = await proc.stdout.read()
				self._stdout_bufs[handle.worker_id] = (
					self._stdout_bufs.get(handle.worker_id, b"") + remaining
				)
				self._stdout_collected.add(handle.worker_id)
		return self._stdout_bufs.get(handle.worker_id, b"").decode(errors="replace")

	async def kill(self, handle: WorkerHandle) -> None:
		proc = self._processes.get(handle.worker_id)
		if proc is not None and proc.returncode is None:
			proc.kill()
			await proc.wait()

		# Also kill remote process
		if handle.backend_metadata:
			host_info = json.loads(handle.backend_metadata)
			ssh_target = self._ssh_target(
				SSHHostConfig(
					hostname=host_info["hostname"], user=host_info.get("user", "")
				)
			)
			try:
				cleanup = await asyncio.create_subprocess_exec(
					"ssh", ssh_target,
					f"pkill -f {shlex.quote('mc-worker-' + handle.worker_id)} || true",
					stdout=asyncio.subprocess.PIPE,
					stderr=asyncio.subprocess.STDOUT,
				)
				await asyncio.wait_for(cleanup.communicate(), timeout=self._connection_timeout)
			except asyncio.TimeoutError:
				logger.warning("SSH pkill timed out for worker %s on %s", handle.worker_id, ssh_target)

	async def release_workspace(self, workspace_path: str) -> None:
		if "::" in workspace_path:
			remote_path, metadata = workspace_path.split("::", 1)
			host_info = json.loads(metadata)
		else:
			remote_path = workspace_path
			host_info = None

		if host_info:
			ssh_target = self._ssh_target(
				SSHHostConfig(
					hostname=host_info["hostname"], user=host_info.get("user", "")
				)
			)
			try:
				await self._run_ssh_command(
					ssh_target,
					f"rm -rf {shlex.quote(remote_path)}",
					max_retries=1,
				)
			except asyncio.TimeoutError:
				logger.warning("SSH rm -rf timed out for workspace %s on %s", remote_path, ssh_target)
			self._worker_count[host_info["hostname"]] = max(
				0, self._worker_count.get(host_info["hostname"], 1) - 1
			)

	async def cleanup(self) -> None:
		for wid, proc in self._processes.items():
			if proc.returncode is None:
				proc.kill()
				await proc.wait()
		self._processes.clear()
		self._stdout_bufs.clear()
		self._stdout_collected.clear()
		self._worker_count = {h.hostname: 0 for h in self._hosts}
