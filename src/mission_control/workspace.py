"""Workspace pool -- manage git clone --shared repos for worker isolation."""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


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
	) -> None:
		self.source_repo = Path(source_repo)
		self.pool_dir = Path(pool_dir)
		self.max_clones = max_clones
		self.base_branch = base_branch
		self.green_branch = green_branch
		self._available: list[Path] = []
		self._in_use: set[Path] = set()
		self._lock = asyncio.Lock()

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

	async def release(self, workspace: Path) -> None:
		"""Return a clone to the pool after resetting it."""
		if workspace not in self._in_use:
			return
		self._in_use.discard(workspace)
		await self._reset_clone(workspace)
		self._available.append(workspace)

	async def cleanup(self) -> None:
		"""Delete all clones and the pool directory."""
		all_clones = list(self._available) + list(self._in_use)
		for clone in all_clones:
			if clone.exists():
				shutil.rmtree(clone)
		self._available.clear()
		self._in_use.clear()
		if self.pool_dir.exists():
			shutil.rmtree(self.pool_dir)

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
