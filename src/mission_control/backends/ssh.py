"""SSH backend -- runs workers on remote hosts via SSH."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex

from mission_control.backends.base import WorkerBackend, WorkerHandle
from mission_control.config import SSHHostConfig

logger = logging.getLogger(__name__)


class SSHBackend(WorkerBackend):
	"""Execute workers on remote hosts via SSH."""

	def __init__(self, hosts: list[SSHHostConfig]) -> None:
		if not hosts:
			raise ValueError("SSHBackend requires at least one host")
		self._hosts = hosts
		self._worker_count: dict[str, int] = {h.hostname: 0 for h in hosts}
		self._processes: dict[str, asyncio.subprocess.Process] = {}
		self._stdout_bufs: dict[str, bytes] = {}
		self._stdout_collected: set[str] = set()

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

		proc = await asyncio.create_subprocess_exec(
			"ssh", self._ssh_target(host),
			f"git clone --depth=1 -b {shlex.quote(base_branch)} {shlex.quote(source_repo)} {shlex.quote(remote_path)}",
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await proc.communicate()
		if proc.returncode != 0:
			raise RuntimeError(
				f"Failed to provision workspace on {host.hostname}: "
				f"{stdout.decode(errors='replace')}"
			)

		self._worker_count[host.hostname] = (
			self._worker_count.get(host.hostname, 0) + 1
		)

		metadata = json.dumps({"hostname": host.hostname, "user": host.user})
		# Store metadata as {remote_path}::{json} so spawn can recover the host
		return f"{remote_path}::{metadata}"

	async def spawn(
		self, worker_id: str, workspace_path: str, command: list[str], timeout: int
	) -> WorkerHandle:
		remote_path, metadata = workspace_path.split("::", 1)
		host_info = json.loads(metadata)

		ssh_target = host_info.get("user", "") + "@" + host_info["hostname"]
		if not host_info.get("user"):
			ssh_target = host_info["hostname"]

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
		if proc.returncode is not None and proc.stdout:
			if handle.worker_id not in self._stdout_collected:
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
			cleanup = await asyncio.create_subprocess_exec(
				"ssh", ssh_target,
				f"pkill -f {shlex.quote('mc-worker-' + handle.worker_id)} || true",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			await cleanup.communicate()

	async def release_workspace(self, workspace_path: str) -> None:
		# workspace_path here is the raw remote path (not the packed form)
		# We need to figure out the host -- check all handles
		# For simplicity, parse from the workspace_path if it has metadata
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
			proc = await asyncio.create_subprocess_exec(
				"ssh", ssh_target, f"rm -rf {shlex.quote(remote_path)}",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			await proc.communicate()
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
