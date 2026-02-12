"""Local backend -- runs workers as subprocesses with WorkspacePool."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from mission_control.backends.base import WorkerBackend, WorkerHandle
from mission_control.workspace import WorkspacePool

logger = logging.getLogger(__name__)


class LocalBackend(WorkerBackend):
	"""Execute workers as local subprocesses with shared git clones."""

	def __init__(
		self,
		source_repo: str,
		pool_dir: str,
		max_clones: int = 10,
		base_branch: str = "main",
	) -> None:
		self._pool = WorkspacePool(
			source_repo=source_repo,
			pool_dir=pool_dir,
			max_clones=max_clones,
			base_branch=base_branch,
		)
		self._processes: dict[str, asyncio.subprocess.Process] = {}
		self._stdout_bufs: dict[str, bytes] = {}
		self._stdout_collected: set[str] = set()

	async def initialize(self, warm_count: int = 0) -> None:
		await self._pool.initialize(warm_count=warm_count)

	async def provision_workspace(
		self, worker_id: str, source_repo: str, base_branch: str
	) -> str:
		workspace = await self._pool.acquire()
		if workspace is None:
			raise RuntimeError(f"No workspace available for worker {worker_id}")

		branch_name = f"mc/unit-{worker_id}"
		proc = await asyncio.create_subprocess_exec(
			"git", "checkout", "-B", branch_name,
			cwd=str(workspace),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await proc.communicate()
		if proc.returncode != 0:
			await self._pool.release(workspace)
			raise RuntimeError(
				f"Failed to create branch {branch_name}: "
				f"{stdout.decode(errors='replace')}"
			)
		return str(workspace)

	async def spawn(
		self, worker_id: str, workspace_path: str, command: list[str], timeout: int
	) -> WorkerHandle:
		proc = await asyncio.create_subprocess_exec(
			*command,
			cwd=workspace_path,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		self._processes[worker_id] = proc
		self._stdout_bufs[worker_id] = b""
		self._stdout_collected.discard(worker_id)
		return WorkerHandle(
			worker_id=worker_id,
			pid=proc.pid,
			workspace_path=workspace_path,
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
			# Process still running -- read what we have so far
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

	async def release_workspace(self, workspace_path: str) -> None:
		await self._pool.release(Path(workspace_path))

	async def cleanup(self) -> None:
		for wid, proc in self._processes.items():
			if proc.returncode is None:
				proc.kill()
				await proc.wait()
		self._processes.clear()
		self._stdout_bufs.clear()
		self._stdout_collected.clear()
		await self._pool.cleanup()
