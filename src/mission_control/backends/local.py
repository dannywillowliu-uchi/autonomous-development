"""Local backend -- runs workers as subprocesses with WorkspacePool."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from mission_control.backends.base import WorkerBackend, WorkerHandle
from mission_control.config import MissionConfig, claude_subprocess_env
from mission_control.workspace import WorkspacePool

logger = logging.getLogger(__name__)


_MB = 1024 * 1024
_OUTPUT_WARNING_THRESHOLDS_MB = (10, 25, 50)


class LocalBackend(WorkerBackend):
	"""Execute workers as local subprocesses with shared git clones."""

	def __init__(
		self,
		source_repo: str,
		pool_dir: str,
		max_clones: int = 10,
		base_branch: str = "main",
		max_output_mb: int = 50,
		config: MissionConfig | None = None,
	) -> None:
		self._config = config
		self._pool = WorkspacePool(
			source_repo=source_repo,
			pool_dir=pool_dir,
			max_clones=max_clones,
			base_branch=base_branch,
		)
		self._processes: dict[str, asyncio.subprocess.Process] = {}
		self._stdout_bufs: dict[str, bytes] = {}
		self._stdout_collected: set[str] = set()
		self._max_output_bytes: int = max_output_mb * _MB
		self._output_warnings_fired: dict[str, set[int]] = {}

	async def initialize(self, warm_count: int = 0) -> None:
		await self._pool.initialize(warm_count=warm_count)

	async def provision_workspace(
		self, worker_id: str, source_repo: str, base_branch: str
	) -> str:
		workspace = await self._pool.acquire()
		if workspace is None:
			raise RuntimeError(f"No workspace available for worker {worker_id}")

		# Fetch latest refs so that branches created after the clone (e.g.
		# mc/green from GreenBranchManager.initialize()) are visible.
		fetch_proc = await asyncio.create_subprocess_exec(
			"git", "fetch", "origin",
			cwd=str(workspace),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		await fetch_proc.communicate()

		# Checkout the target base branch before creating the feature branch.
		# Without this, the feature branch starts from whatever HEAD the clone
		# was reset to (origin/main), ignoring the caller's base_branch.
		base_proc = await asyncio.create_subprocess_exec(
			"git", "checkout", "-B", base_branch, f"origin/{base_branch}",
			cwd=str(workspace),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		base_stdout, _ = await base_proc.communicate()
		if base_proc.returncode != 0:
			await self._pool.release(workspace)
			raise RuntimeError(
				f"Failed to checkout base branch {base_branch}: "
				f"{base_stdout.decode(errors='replace')}"
			)

		# Sync uncommitted working-tree changes from source into clone.
		# This ensures fixes not yet committed (e.g. config changes) are
		# available to workers immediately.
		src_dir = Path(source_repo) / "src"
		dst_dir = Path(workspace) / "src"
		if src_dir.is_dir() and dst_dir.is_dir():
			sync_proc = await asyncio.create_subprocess_exec(
				"rsync", "-a", "--delete",
				str(src_dir) + "/",
				str(dst_dir) + "/",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			sync_out, _ = await sync_proc.communicate()
			if sync_proc.returncode != 0:
				logger.warning(
					"rsync src/ to workspace failed (rc=%d): %s",
					sync_proc.returncode,
					sync_out.decode(errors="replace") if sync_out else "",
				)
			else:
				logger.info(
					"Synced source working tree to workspace %s",
					workspace.name if isinstance(workspace, Path) else workspace,
				)

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

		# Symlink source .venv into worker workspace for verification commands.
		# _reset_clone() runs git clean -fdx which deletes symlinks, so this
		# must be recreated on every provision.
		# IMPORTANT: Workers must NOT run `pip install` or `uv pip install` --
		# doing so overwrites the editable install to point at the clone instead
		# of the source tree, breaking imports after mc/green reset.
		source_venv = Path(source_repo) / ".venv"
		workspace_venv = Path(workspace) / ".venv"
		if source_venv.exists() and not workspace_venv.exists():
			workspace_venv.symlink_to(source_venv)

		# Drop a marker so tooling can detect a protected symlinked venv
		marker = Path(workspace) / ".editable-install-protected"
		try:
			if not marker.exists():
				marker.write_text("This venv is symlinked from the source repo. Do not run pip install.\n")
		except OSError:
			pass

		# NOTE: Workers may corrupt the editable .pth file by running
		# `pip install -e .` in their clones. This is harmless because
		# _workspace_env() sets PYTHONPATH to the workspace src/ directory,
		# bypassing the .pth file entirely for verification/acceptance.

		return str(workspace)

	async def spawn(
		self, worker_id: str, workspace_path: str, command: list[str], timeout: int
	) -> WorkerHandle:
		proc = await asyncio.create_subprocess_exec(
			*command,
			cwd=workspace_path,
			stdin=asyncio.subprocess.DEVNULL,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
			env=claude_subprocess_env(self._config),
		)
		self._processes[worker_id] = proc
		self._stdout_bufs[worker_id] = b""
		self._stdout_collected.discard(worker_id)
		self._output_warnings_fired[worker_id] = set()
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

	def _check_output_thresholds(self, worker_id: str, size: int) -> None:
		"""Log warnings when output size crosses configured thresholds."""
		fired = self._output_warnings_fired.get(worker_id, set())
		for threshold_mb in _OUTPUT_WARNING_THRESHOLDS_MB:
			threshold_bytes = threshold_mb * _MB
			if size >= threshold_bytes and threshold_mb not in fired:
				fired.add(threshold_mb)
				logger.warning(
					"Worker %s output reached %dMB (%d bytes)",
					worker_id, threshold_mb, size,
				)
		self._output_warnings_fired[worker_id] = fired

	async def get_output(self, handle: WorkerHandle) -> str:
		proc = self._processes.get(handle.worker_id)
		if proc is None:
			return ""
		wid = handle.worker_id
		if proc.returncode is None:
			# Process still running -- read what we have so far
			if proc.stdout:
				try:
					chunk = await asyncio.wait_for(proc.stdout.read(65536), timeout=0.1)
					self._stdout_bufs[wid] = self._stdout_bufs.get(wid, b"") + chunk
				except asyncio.TimeoutError:
					pass
			buf_size = len(self._stdout_bufs.get(wid, b""))
			self._check_output_thresholds(wid, buf_size)
			if buf_size >= self._max_output_bytes:
				logger.error(
					"Worker %s output exceeded limit (%dMB). Killing worker.",
					wid, self._max_output_bytes // _MB,
				)
				self._stdout_bufs[wid] = self._stdout_bufs[wid][:self._max_output_bytes]
				proc.kill()
				await proc.wait()
		else:
			# Process finished -- collect remaining output once
			if proc.stdout and wid not in self._stdout_collected:
				remaining = await proc.stdout.read()
				self._stdout_bufs[wid] = self._stdout_bufs.get(wid, b"") + remaining
				# Truncate if final collection pushes over limit
				if len(self._stdout_bufs[wid]) > self._max_output_bytes:
					self._stdout_bufs[wid] = self._stdout_bufs[wid][:self._max_output_bytes]
				self._stdout_collected.add(wid)
			self._check_output_thresholds(wid, len(self._stdout_bufs.get(wid, b"")))
		return self._stdout_bufs.get(wid, b"").decode(errors="replace")

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
		self._output_warnings_fired.clear()
		await self._pool.cleanup()
