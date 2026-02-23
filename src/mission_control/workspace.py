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
	) -> None:
		self.source_repo = Path(source_repo)
		self.pool_dir = Path(pool_dir)
		self.max_clones = max_clones
		self.base_branch = base_branch
		self._available: list[Path] = []
		self._in_use: set[Path] = set()

	@property
	def total_clones(self) -> int:
		"""Total number of clones (available + in use)."""
		return len(self._available) + len(self._in_use)

	@property
	def available_slots(self) -> int:
		"""Number of pool slots not currently in use (including ones not yet created)."""
		return self.max_clones - len(self._in_use)

	async def initialize(self, warm_count: int = 0) -> None:
		"""Create pool directory and optionally pre-warm clones."""
		self.pool_dir.mkdir(parents=True, exist_ok=True)
		for _ in range(warm_count):
			clone = await self._create_clone()
			if clone is None:
				break
			self._available.append(clone)

	async def acquire(self) -> Path | None:
		"""Get a clone from the pool, creating one if needed.

		Returns None if at max_clones limit.
		"""
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
		"""Reset a clone to clean state: fetch, reset --hard, clean -fdx."""
		cwd = str(clone_path)

		fetch = await asyncio.create_subprocess_exec(
			"git", "fetch", "origin",
			cwd=cwd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		await fetch.communicate()

		reset = await asyncio.create_subprocess_exec(
			"git", "reset", "--hard", f"origin/{self.base_branch}",
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
