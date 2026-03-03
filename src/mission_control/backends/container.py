"""Container backend -- runs workers inside Docker containers with host-mounted workspaces."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from mission_control.backends.base import WorkerBackend, WorkerHandle
from mission_control.backends.local import HealthCheckResult
from mission_control.config import ContainerConfig, MissionConfig, claude_subprocess_env
from mission_control.workspace import WorkspacePool

logger = logging.getLogger(__name__)


_MB = 1024 * 1024
_OUTPUT_WARNING_THRESHOLDS_MB = (10, 25, 50)


class ContainerBackend(WorkerBackend):
	"""Execute workers inside Docker containers with volume-mounted workspaces.

	WorkspacePool provisions clones on the host filesystem. Each container
	mounts the host workspace path, so GreenBranchManager's git remote add
	pattern works unchanged.
	"""

	def __init__(
		self,
		source_repo: str,
		pool_dir: str,
		container_config: ContainerConfig,
		max_clones: int = 10,
		base_branch: str = "main",
		max_output_mb: int = 50,
		config: MissionConfig | None = None,
		container_health_timeout: int = 10,
	) -> None:
		self._config = config
		self._pool = WorkspacePool(
			source_repo=source_repo,
			pool_dir=pool_dir,
			max_clones=max_clones,
			base_branch=base_branch,
		)
		self._container_config = container_config
		self._container_health_timeout = container_health_timeout
		self._processes: dict[str, asyncio.subprocess.Process] = {}
		self._container_names: dict[str, str] = {}
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

		# Fetch latest refs
		fetch_proc = await asyncio.create_subprocess_exec(
			"git", "fetch", "origin",
			cwd=str(workspace),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		await fetch_proc.communicate()

		# Checkout base branch
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

		# Create unit branch
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

		# No .venv symlink -- meaningless inside container
		return str(workspace)

	def _build_docker_command(
		self, worker_id: str, host_workspace: str, command: list[str]
	) -> list[str]:
		"""Build the docker run command list."""
		cc = self._container_config
		container_name = f"mc-worker-{worker_id}"
		self._container_names[worker_id] = container_name

		cmd = [
			cc.docker_executable, "run", "--rm",
			"--name", container_name,
			"--network", cc.network,
			"-v", f"{host_workspace}:{cc.workspace_mount}",
		]

		# Mount Claude config dir (read-only) if configured
		claude_config = cc.claude_config_dir or os.environ.get("CLAUDE_CONFIG_DIR", "")
		if claude_config:
			claude_config = os.path.expanduser(claude_config)
			cmd.extend(["-v", f"{claude_config}:/home/mcworker/.config/claude:ro"])

		# Extra volumes
		for vol in cc.extra_volumes:
			cmd.extend(["-v", vol])

		cmd.extend(["--workdir", cc.workspace_mount])

		# Security: drop capabilities
		for cap in cc.cap_drop:
			cmd.extend(["--cap-drop", cap])

		# Security: security options
		for opt in cc.security_opt:
			cmd.extend(["--security-opt", opt])

		# Run as non-root
		cmd.extend(["--user", cc.run_as_user])

		# Environment variables
		cmd.extend(["-e", "HOME=/home/mcworker"])
		if claude_config:
			cmd.extend(["-e", "CLAUDE_CONFIG_DIR=/home/mcworker/.config/claude"])

		# Pass through allowlisted env vars
		for key, value in claude_subprocess_env(self._config).items():
			cmd.extend(["-e", f"{key}={value}"])

		cmd.append(cc.image)
		cmd.extend(command)

		return cmd

	async def _verify_container_workspace(
		self, worker_id: str, workspace_path: str,
	) -> HealthCheckResult:
		"""Run pre-flight health checks on a container workspace.

		The workspace lives on the host filesystem and is volume-mounted into
		containers, so checks run on the host path directly. Additionally
		checks for stale containers that would conflict with a new spawn.

		Checks:
		1. No stale container with the same name (docker inspect)
		2. .git/HEAD exists in the host workspace
		3. git status --porcelain exits 0 on the host workspace
		"""
		issues: list[str] = []
		cc = self._container_config
		container_name = self._container_names.get(worker_id, f"mc-worker-{worker_id}")
		timeout = self._container_health_timeout
		workspace = Path(workspace_path)

		# If the host workspace directory doesn't exist, nothing to verify.
		if not workspace.is_dir():
			logger.info(
				"Container health check skipped for %s: workspace %s does not exist",
				container_name, workspace_path,
			)
			return HealthCheckResult(passed=True)

		# 1. Check for stale container via docker inspect
		try:
			inspect_proc = await asyncio.create_subprocess_exec(
				cc.docker_executable, "inspect", "--format", "{{.State.Running}}", container_name,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			await asyncio.wait_for(inspect_proc.wait(), timeout=timeout)
			if inspect_proc.returncode == 0:
				# Container exists -- stale leftover from a previous run
				issues.append(f"Stale container {container_name} found")
		except asyncio.TimeoutError:
			issues.append(f"docker inspect timed out after {timeout}s for {container_name}")
		except OSError as exc:
			issues.append(f"docker inspect failed with OS error for {container_name}: {exc}")

		# 2. Check .git/HEAD exists in the host workspace
		git_head = workspace / ".git" / "HEAD"
		try:
			git_head.read_text()
		except (OSError, FileNotFoundError):
			issues.append(f".git/HEAD missing or unreadable at {workspace_path}")

		# 3. Run git status --porcelain on the host workspace
		try:
			status_proc = await asyncio.create_subprocess_exec(
				"git", "status", "--porcelain",
				cwd=workspace_path,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			await asyncio.wait_for(status_proc.wait(), timeout=timeout)
			if status_proc.returncode != 0:
				issues.append(
					f"git status failed (rc={status_proc.returncode}) at {workspace_path}"
				)
		except asyncio.TimeoutError:
			issues.append(f"git status timed out after {timeout}s at {workspace_path}")
		except OSError as exc:
			issues.append(f"git status failed with OS error: {exc}")

		result = HealthCheckResult(passed=len(issues) == 0, issues=issues)
		logger.info(
			"Container health check for %s (workspace=%s): passed=%s issues=%s",
			container_name, workspace_path, result.passed, result.issues,
		)
		return result

	async def _repair_container(self, worker_id: str) -> None:
		"""Stop and remove a container to allow re-provisioning."""
		cc = self._container_config
		container_name = self._container_names.get(worker_id, f"mc-worker-{worker_id}")
		timeout = self._container_health_timeout

		# Stop the container
		try:
			stop_proc = await asyncio.create_subprocess_exec(
				cc.docker_executable, "stop", "--time", "5", container_name,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			await asyncio.wait_for(stop_proc.communicate(), timeout=timeout)
		except (asyncio.TimeoutError, OSError) as exc:
			logger.warning("docker stop failed for %s during repair: %s", container_name, exc)

		# Remove the container (--force in case stop didn't work)
		try:
			rm_proc = await asyncio.create_subprocess_exec(
				cc.docker_executable, "rm", "--force", container_name,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			await asyncio.wait_for(rm_proc.communicate(), timeout=timeout)
		except (asyncio.TimeoutError, OSError) as exc:
			logger.warning("docker rm failed for %s during repair: %s", container_name, exc)

	async def spawn(
		self, worker_id: str, workspace_path: str, command: list[str], timeout: int
	) -> WorkerHandle:
		# Pre-flight health check: verify container workspace integrity.
		health = await self._verify_container_workspace(worker_id, workspace_path)
		if not health.passed:
			for issue in health.issues:
				logger.warning("Container health check failed: %s", issue)
			# Attempt one repair cycle: stop, remove, re-provision the container.
			logger.info("Attempting container repair for worker %s", worker_id)
			await self._repair_container(worker_id)
			# Re-check after repair
			health = await self._verify_container_workspace(worker_id, workspace_path)
			if not health.passed:
				for issue in health.issues:
					logger.warning("Container health check failed after repair: %s", issue)
				raise RuntimeError(
					f"Container health check failed after repair for {worker_id}: "
					+ "; ".join(health.issues)
				)

		docker_cmd = self._build_docker_command(worker_id, workspace_path, command)

		proc = await asyncio.create_subprocess_exec(
			*docker_cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		self._processes[worker_id] = proc
		self._stdout_bufs[worker_id] = b""
		self._stdout_collected.discard(worker_id)
		self._output_warnings_fired[worker_id] = set()

		# Return host workspace path -- controller and GreenBranchManager
		# only ever see host paths.
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
			if proc.stdout and wid not in self._stdout_collected:
				remaining = await proc.stdout.read()
				self._stdout_bufs[wid] = self._stdout_bufs.get(wid, b"") + remaining
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

		# Fallback: docker stop the container directly
		container_name = self._container_names.get(handle.worker_id)
		if container_name:
			try:
				stop_proc = await asyncio.create_subprocess_exec(
					self._container_config.docker_executable, "stop", "--time", "5", container_name,
					stdout=asyncio.subprocess.PIPE,
					stderr=asyncio.subprocess.STDOUT,
				)
				await asyncio.wait_for(stop_proc.communicate(), timeout=10)
			except (asyncio.TimeoutError, OSError) as e:
				logger.warning("docker stop timed out for %s: %s", container_name, e)

	async def release_workspace(self, workspace_path: str) -> None:
		await self._pool.release(Path(workspace_path))

	async def cleanup(self) -> None:
		for wid, proc in self._processes.items():
			if proc.returncode is None:
				proc.kill()
				await proc.wait()
		self._processes.clear()
		self._container_names.clear()
		self._stdout_bufs.clear()
		self._stdout_collected.clear()
		self._output_warnings_fired.clear()
		await self._pool.cleanup()
