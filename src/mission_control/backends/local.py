"""Local backend -- runs workers as subprocesses with WorkspacePool."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from mission_control.backends.base import WorkerBackend, WorkerHandle
from mission_control.config import MissionConfig, claude_subprocess_env
from mission_control.workspace import WorkspacePool

logger = logging.getLogger(__name__)


_MB = 1024 * 1024
_OUTPUT_WARNING_THRESHOLDS_MB = (10, 25, 50)
_HEALTH_CHECK_TIMEOUT = 5


@dataclass
class HealthCheckResult:
	"""Result of a workspace health pre-flight check."""

	passed: bool
	issues: list[str] = field(default_factory=list)


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
		gb = getattr(config, "green_branch", None) if config else None
		green_branch_name = getattr(gb, "green_branch", None) if gb else None
		self._pool = WorkspacePool(
			source_repo=source_repo,
			pool_dir=pool_dir,
			max_clones=max_clones,
			base_branch=base_branch,
			green_branch=green_branch_name,
		)
		self._processes: dict[str, asyncio.subprocess.Process] = {}
		self._stdout_bufs: dict[str, bytes] = {}
		self._stdout_collected: set[str] = set()
		self._max_output_bytes: int = max_output_mb * _MB
		self._output_warnings_fired: dict[str, set[int]] = {}

	async def initialize(self, warm_count: int = 0) -> None:
		await self._pool.initialize(warm_count=warm_count)

	async def _verify_workspace_health(self, workspace: Path) -> HealthCheckResult:
		"""Run pre-flight health checks on a workspace before use.

		Checks:
		1. .git/HEAD exists and is readable
		2. git status --porcelain exits cleanly (non-zero = corrupted)
		3. .venv symlink target exists (if symlink present)
		"""
		issues: list[str] = []

		# 1. Check .git/HEAD exists and is readable
		git_head = workspace / ".git" / "HEAD"
		try:
			git_head.read_text()
		except (OSError, FileNotFoundError):
			issues.append(f".git/HEAD missing or unreadable at {workspace}")

		# 2. Run git status --porcelain with timeout
		try:
			status_proc = await asyncio.create_subprocess_exec(
				"git", "status", "--porcelain",
				cwd=str(workspace),
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			await asyncio.wait_for(status_proc.communicate(), timeout=_HEALTH_CHECK_TIMEOUT)
			if status_proc.returncode != 0:
				issues.append(f"git status failed (rc={status_proc.returncode}) at {workspace}")
		except asyncio.TimeoutError:
			issues.append(f"git status timed out after {_HEALTH_CHECK_TIMEOUT}s at {workspace}")
		except OSError as exc:
			issues.append(f"git status failed with OS error: {exc}")

		# 3. Verify .venv symlink target exists
		workspace_venv = workspace / ".venv"
		if workspace_venv.is_symlink():
			if not workspace_venv.resolve().exists():
				issues.append(f".venv symlink target missing: {workspace_venv} -> {workspace_venv.resolve()}")

		return HealthCheckResult(passed=len(issues) == 0, issues=issues)

	async def provision_workspace(
		self, worker_id: str, source_repo: str, base_branch: str
	) -> str:
		workspace = await self._pool.acquire()
		if workspace is None:
			raise RuntimeError(f"No workspace available for worker {worker_id}")

		# Pre-flight health check: verify workspace integrity before use.
		health = await self._verify_workspace_health(workspace)
		if not health.passed:
			for issue in health.issues:
				logger.warning("Workspace health check failed: %s", issue)
			# Release the bad workspace and try to acquire a fresh one.
			await self._pool.release(workspace)
			workspace = await self._pool.acquire()
			if workspace is None:
				raise RuntimeError(f"No workspace available for worker {worker_id} after re-provision")
			# Retry health check once on the fresh workspace.
			health = await self._verify_workspace_health(workspace)
			if not health.passed:
				for issue in health.issues:
					logger.warning("Workspace health check failed after re-provision: %s", issue)
				await self._pool.release(workspace)
				raise RuntimeError(
					f"Workspace health check failed after re-provision for {worker_id}: "
					+ "; ".join(health.issues)
				)

		# Repair editable install if workers from a previous mission corrupted
		# the .pth file to point at a (now-deleted) worker clone.
		await self._repair_editable_install(source_repo)

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

		await self._write_worker_claude_md(Path(workspace) if not isinstance(workspace, Path) else workspace)

		# Block workers from pushing directly to origin.
		# The orchestrator handles all pushes via the green branch manager.
		await asyncio.create_subprocess_exec(
			"git", "config", "remote.origin.pushUrl", "no_push_allowed",
			cwd=str(workspace),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
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

	async def _write_worker_claude_md(self, workspace: Path) -> None:
		"""Write worker-specific CLAUDE.md into the workspace.

		Overwrites the cloned project CLAUDE.md with a stripped-down version
		that omits human-dev config (Telegram, Obsidian, etc.).
		"""
		template_path = Path(__file__).parent.parent / "worker_claude_md.md"
		try:
			template = template_path.read_text()
		except FileNotFoundError:
			logger.warning("Worker CLAUDE.md template not found at %s", template_path)
			return
		verification_cmd = ".venv/bin/python -m pytest -q && .venv/bin/ruff check src/ tests/"
		if self._config:
			v = getattr(self._config, "target", None)
			v = getattr(v, "verification", None) if v else None
			v = getattr(v, "command", None) if v else None
			if v:
				verification_cmd = v
		content = template.replace("{verification_command}", verification_cmd)
		(workspace / "CLAUDE.md").write_text(content)

	async def _repair_editable_install(self, source_repo: str) -> None:
		"""Repair the editable .pth if it points to a stale worker clone."""
		source_venv = Path(source_repo) / ".venv"
		pth_files = list(source_venv.glob("lib/*/site-packages/__editable__.mission_control*.pth"))
		if not pth_files:
			return
		pth = pth_files[0]
		try:
			current = pth.read_text().strip()
		except OSError:
			return
		expected = str(Path(source_repo) / "src")
		if current == expected:
			return
		logger.info("Repairing editable install: %s -> %s", current, expected)
		venv_python = source_venv / "bin" / "python"
		if not venv_python.exists():
			return
		proc = await asyncio.create_subprocess_exec(
			"uv", "pip", "install", "-e", ".",
			"--python", str(venv_python),
			cwd=source_repo,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await proc.communicate()
		if proc.returncode != 0:
			output = stdout.decode(errors="replace") if stdout else ""
			logger.warning("Failed to repair editable install: %s", output[:300])

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
