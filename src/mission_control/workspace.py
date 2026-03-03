"""Workspace pool -- manage git clone --shared repos for worker isolation."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceState:
	"""Per-workspace metadata for warm-start optimisation."""

	last_branch: str = ""
	last_fetch_time: float = 0.0
	is_dirty: bool = True
	last_green_sha: str = ""


class WorkspacePool:
	"""Pool of shared git clones for parallel worker isolation.

	Uses `git clone --shared` to create lightweight clones that hardlink
	to the source repo's .git/objects, making them instant and minimal on disk.
	"""

	def __init__(
		self,
		source_repo: str | Path,
		pool_dir: str | Path,
		max_clones: int = 10,
		base_branch: str = "main",
		green_branch: str | None = None,
		fetch_ttl: float = 60.0,
	) -> None:
		self.source_repo = Path(source_repo)
		self.pool_dir = Path(pool_dir)
		self.max_clones = max_clones
		self.base_branch = base_branch
		self.green_branch = green_branch
		self.fetch_ttl = fetch_ttl
		self._available: list[Path] = []
		self._in_use: set[Path] = set()
		self._lock = asyncio.Lock()
		self._states: dict[Path, WorkspaceState] = {}
		self._claim_times: list[float] = []

	@property
	def total_clones(self) -> int:
		"""Total number of clones (available + in use)."""
		return len(self._available) + len(self._in_use)

	@property
	def available_slots(self) -> int:
		"""Number of pool slots not currently in use (including ones not yet created)."""
		return self.max_clones - len(self._in_use)

	async def initialize(self, warm_count: int = 0) -> None:
		"""Create pool directory and optionally pre-warm clones.

		Removes leftover worker directories from prior runs that are not
		tracked in this process's in-memory sets (e.g. after a crash or kill).
		"""
		if self.pool_dir.exists():
			stale = [
				p for p in self.pool_dir.iterdir()
				if p.is_dir() and p.name.startswith("worker-")
				and p not in self._available and p not in self._in_use
			]
			for p in stale:
				logger.info("Removing stale worker directory: %s", p.name)
				shutil.rmtree(p)
		self.pool_dir.mkdir(parents=True, exist_ok=True)
		for _ in range(warm_count):
			clone = await self._create_clone()
			if clone is None:
				break
			self._available.append(clone)

	async def acquire(self) -> Path | None:
		"""Get a clone from the pool, creating one if needed.

		Returns None if at max_clones limit.  Serialized via asyncio.Lock
		to prevent concurrent callers from racing past the size check.
		"""
		async with self._lock:
			if self._available:
				workspace = self._available.pop()
				self._in_use.add(workspace)
				return workspace

			clone = await self._create_clone()
			if clone is None:
				return None
			self._in_use.add(clone)
			return clone

	async def claim(
		self,
		branch: str | None = None,
		sha: str | None = None,
	) -> Path | None:
		"""Warm-start-aware acquire: skip redundant git ops when possible.

		If the workspace was recently released (within fetch_ttl), skip fetch.
		If already on the target branch at the correct SHA, skip checkout.
		Falls back to full fetch+checkout for cold workspaces.

		Returns None if no workspace is available.
		"""
		t0 = time.monotonic()

		async with self._lock:
			workspace: Path | None = None
			if self._available:
				workspace = self._available.pop()
				self._in_use.add(workspace)
			else:
				workspace = await self._create_clone()
				if workspace is None:
					return None
				self._in_use.add(workspace)

		state = self._states.get(workspace)
		target_branch = branch or self.green_branch or self.base_branch
		now = time.monotonic()

		needs_fetch = True
		needs_checkout = True

		if state and not state.is_dirty:
			# Skip fetch if last fetch was within TTL
			if (now - state.last_fetch_time) < self.fetch_ttl:
				needs_fetch = False

			# Skip checkout if already on the target branch at the right SHA
			if state.last_branch == target_branch:
				if sha is None or state.last_green_sha == sha:
					needs_checkout = False

		cwd = str(workspace)

		if needs_fetch:
			fetch = await asyncio.create_subprocess_exec(
				"git", "fetch", "origin",
				cwd=cwd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			await fetch.communicate()
			# Update fetch time in state
			if workspace not in self._states:
				self._states[workspace] = WorkspaceState()
			self._states[workspace].last_fetch_time = time.monotonic()

		if needs_checkout:
			checkout = await asyncio.create_subprocess_exec(
				"git", "checkout", target_branch,
				cwd=cwd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			await checkout.communicate()

			if sha:
				reset = await asyncio.create_subprocess_exec(
					"git", "reset", "--hard", sha,
					cwd=cwd,
					stdout=asyncio.subprocess.PIPE,
					stderr=asyncio.subprocess.STDOUT,
				)
				await reset.communicate()

			# Update branch state
			if workspace not in self._states:
				self._states[workspace] = WorkspaceState()
			self._states[workspace].last_branch = target_branch
			if sha:
				self._states[workspace].last_green_sha = sha

		elapsed_ms = (time.monotonic() - t0) * 1000
		self._claim_times.append(elapsed_ms)

		return workspace

	async def release(self, workspace: Path) -> None:
		"""Return a clone to the pool after resetting it and warming state."""
		if workspace not in self._in_use:
			return
		self._in_use.discard(workspace)
		await self._reset_clone(workspace)

		# Record warm state after reset
		state = self._states.get(workspace)
		if state is None:
			state = WorkspaceState()
			self._states[workspace] = state

		state.last_fetch_time = time.monotonic()
		state.is_dirty = False

		# Determine which branch/sha the workspace is on after reset
		reset_branch = self.green_branch or self.base_branch
		state.last_branch = reset_branch

		cwd = str(workspace)
		rev = await asyncio.create_subprocess_exec(
			"git", "rev-parse", "HEAD",
			cwd=cwd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await rev.communicate()
		if rev.returncode == 0 and stdout:
			state.last_green_sha = stdout.decode().strip()

		self._available.append(workspace)

	async def cleanup(self) -> None:
		"""Delete all clones and the pool directory."""
		all_clones = list(self._available) + list(self._in_use)
		for clone in all_clones:
			if clone.exists():
				shutil.rmtree(clone)
		self._available.clear()
		self._in_use.clear()
		self._states.clear()
		self._claim_times.clear()
		if self.pool_dir.exists():
			shutil.rmtree(self.pool_dir)

	def get_pool_stats(self) -> dict[str, int | float]:
		"""Return pool statistics for observability."""
		warm_count = sum(
			1 for ws in self._available
			if ws in self._states and not self._states[ws].is_dirty
		)
		cold_count = len(self._available) - warm_count

		avg_claim_ms = 0.0
		if self._claim_times:
			avg_claim_ms = sum(self._claim_times) / len(self._claim_times)

		return {
			"total_workspaces": self.total_clones,
			"warm_count": warm_count,
			"cold_count": cold_count,
			"avg_claim_time_ms": round(avg_claim_ms, 2),
		}

	def get_workspace_state(self, workspace: Path) -> WorkspaceState | None:
		"""Return the tracked state for a workspace, or None if untracked."""
		return self._states.get(workspace)

	async def _create_clone(self) -> Path | None:
		"""Create a git clone --shared of source_repo.

		Returns None if at max_clones limit.
		"""
		if self.total_clones >= self.max_clones:
			return None

		name = f"worker-{uuid.uuid4().hex[:8]}"
		clone_path = self.pool_dir / name

		proc = await asyncio.create_subprocess_exec(
			"git", "clone", "--shared", str(self.source_repo), str(clone_path),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await proc.communicate()

		if proc.returncode != 0:
			output = stdout.decode(errors="replace") if stdout else ""
			logger.error("Failed to create shared clone at %s: %s", clone_path, output)
			return None

		return clone_path

	async def _reset_clone(self, clone_path: Path) -> None:
		"""Reset a clone to clean state: checkout base, fetch, reset --hard, clean -fdx.

		IMPORTANT: checkout base_branch first so that `git reset --hard` only
		moves the base branch ref, not the current (unit) branch.  Without
		this, resetting while on mc/unit-X moves that branch ref to
		origin/main, destroying the worker's commit before the green-branch
		merge processor can fetch it.

		When a green_branch is configured, try resetting to origin/{green_branch}
		first (latest merged state). Falls back to origin/{base_branch} if the
		green branch ref does not exist yet.
		"""
		cwd = str(clone_path)

		# Detach from unit branch so reset doesn't destroy its ref
		checkout = await asyncio.create_subprocess_exec(
			"git", "checkout", self.base_branch,
			cwd=cwd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		await checkout.communicate()

		fetch = await asyncio.create_subprocess_exec(
			"git", "fetch", "origin",
			cwd=cwd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		await fetch.communicate()

		# Prefer green branch (latest merged state) over base branch
		reset_ref = f"origin/{self.base_branch}"
		if self.green_branch:
			verify = await asyncio.create_subprocess_exec(
				"git", "rev-parse", "--verify", f"origin/{self.green_branch}",
				cwd=cwd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			await verify.communicate()
			if verify.returncode == 0:
				reset_ref = f"origin/{self.green_branch}"

		reset = await asyncio.create_subprocess_exec(
			"git", "reset", "--hard", reset_ref,
			cwd=cwd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		await reset.communicate()

		clean = await asyncio.create_subprocess_exec(
			"git", "clean", "-fdx",
			cwd=cwd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		await clean.communicate()
