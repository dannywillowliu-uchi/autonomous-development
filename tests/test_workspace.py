"""Tests for workspace pool -- shared git clone management."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mission_control.workspace import WorkspacePool


@pytest.fixture()
def source_repo(tmp_path: Path) -> Path:
	"""Create a real git repo to use as the clone source."""
	import subprocess

	repo = tmp_path / "source"
	repo.mkdir()
	subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
	subprocess.run(["git", "checkout", "-b", "main"], cwd=str(repo), check=True, capture_output=True)

	# Create an initial commit so the branch exists
	readme = repo / "README.md"
	readme.write_text("# Test repo\n")
	subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
	subprocess.run(
		["git", "commit", "-m", "Initial commit"],
		cwd=str(repo), check=True, capture_output=True,
		env={"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test.com",
			"GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test.com",
			"PATH": subprocess.check_output(["bash", "-c", "echo $PATH"]).decode().strip()},
	)
	return repo


@pytest.fixture()
def pool_dir(tmp_path: Path) -> Path:
	"""Directory for workspace clones."""
	return tmp_path / "pool"


class TestWorkspacePool:
	async def test_create_clone_has_git_dir(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Create a clone and verify .git exists."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		workspace = await pool.acquire()
		assert workspace is not None
		assert (workspace / ".git").exists()

		await pool.cleanup()

	async def test_acquire_release_lifecycle(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Acquire returns a path, release makes it available again."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		workspace = await pool.acquire()
		assert workspace is not None
		assert pool.total_clones == 1
		assert len(pool._in_use) == 1
		assert len(pool._available) == 0

		await pool.release(workspace)
		assert len(pool._in_use) == 0
		assert len(pool._available) == 1

		# Acquiring again should return the same clone (from stack)
		workspace2 = await pool.acquire()
		assert workspace2 == workspace

		await pool.cleanup()

	async def test_reset_clone_removes_dirty_files(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Release resets the clone to clean state, removing untracked files."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		workspace = await pool.acquire()
		assert workspace is not None

		# Create a dirty file
		dirty_file = workspace / "dirty.txt"
		dirty_file.write_text("this should be cleaned up")
		assert dirty_file.exists()

		# Release should reset the clone
		await pool.release(workspace)

		# The dirty file should be gone after reset
		assert not dirty_file.exists()

		await pool.cleanup()

	async def test_max_clones_enforced(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Acquire returns None when max_clones limit is reached."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=2)
		await pool.initialize()

		w1 = await pool.acquire()
		w2 = await pool.acquire()
		assert w1 is not None
		assert w2 is not None
		assert pool.total_clones == 2

		# Third acquire should fail
		w3 = await pool.acquire()
		assert w3 is None

		await pool.cleanup()

	async def test_cleanup_removes_all_clones(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Cleanup removes all clones and the pool directory."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		w1 = await pool.acquire()
		w2 = await pool.acquire()
		assert w1 is not None
		assert w2 is not None

		# Release one so we have both available and in-use
		await pool.release(w1)

		await pool.cleanup()
		assert pool.total_clones == 0
		assert not pool_dir.exists()

	async def test_initialize_with_warm_clones(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Initialize with warm_count pre-creates clones."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=5)
		await pool.initialize(warm_count=3)

		assert pool.total_clones == 3
		assert len(pool._available) == 3
		assert len(pool._in_use) == 0

		# All pre-warmed clones should have .git
		for clone in pool._available:
			assert (clone / ".git").exists()

		await pool.cleanup()


	async def test_available_slots(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""available_slots returns max_clones minus in-use count."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=4)
		await pool.initialize()

		assert pool.available_slots == 4

		w1 = await pool.acquire()
		assert w1 is not None
		assert pool.available_slots == 3

		w2 = await pool.acquire()
		assert w2 is not None
		assert pool.available_slots == 2

		await pool.release(w1)
		assert pool.available_slots == 3

		await pool.cleanup()


class TestWorkspacePoolConcurrency:
	"""Integration tests for concurrent workspace access."""

	async def test_concurrent_acquire(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Multiple concurrent acquires don't exceed max_clones."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		# Acquire all 3 concurrently
		results = await asyncio.gather(
			pool.acquire(),
			pool.acquire(),
			pool.acquire(),
		)
		acquired = [r for r in results if r is not None]
		assert len(acquired) == 3
		assert pool.total_clones == 3

		# All should be unique paths
		assert len(set(acquired)) == 3

		# Next acquire should return None
		extra = await pool.acquire()
		assert extra is None

		await pool.cleanup()

	async def test_concurrent_acquire_release_cycles(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Rapid acquire/release cycles don't leak resources."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=2)
		await pool.initialize()

		for _ in range(5):
			w1 = await pool.acquire()
			w2 = await pool.acquire()
			assert w1 is not None
			assert w2 is not None

			# Both in use
			assert len(pool._in_use) == 2
			assert len(pool._available) == 0

			await pool.release(w1)
			await pool.release(w2)

			# Both back to available
			assert len(pool._in_use) == 0
			assert len(pool._available) == 2

		# Total clones should not exceed max
		assert pool.total_clones <= 2
		await pool.cleanup()

	async def test_cleanup_during_active_use(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Cleanup removes clones even if some are still in use."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		w1 = await pool.acquire()
		w2 = await pool.acquire()
		assert w1 is not None
		assert w2 is not None

		# Don't release -- cleanup should still work
		await pool.cleanup()

		assert pool.total_clones == 0
		assert len(pool._in_use) == 0
		assert len(pool._available) == 0
		assert not pool_dir.exists()

	async def test_release_unknown_workspace_is_noop(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Releasing a path not in _in_use is silently ignored."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		# Release a path that was never acquired
		fake_path = pool_dir / "fake-workspace"
		await pool.release(fake_path)

		assert pool.total_clones == 0
		await pool.cleanup()

	async def test_workspace_isolation(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Changes in one workspace don't affect another."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		w1 = await pool.acquire()
		w2 = await pool.acquire()
		assert w1 is not None
		assert w2 is not None

		# Create a file in w1
		(w1 / "w1_only.txt").write_text("w1")

		# w2 should not have it
		assert not (w2 / "w1_only.txt").exists()

		await pool.cleanup()
