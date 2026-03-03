"""Tests for workspace warm-start and state tracking."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from mission_control.workspace import WorkspacePool, WorkspaceState


def _git_env() -> dict[str, str]:
	"""Git environment with author/committer set for test commits."""
	return {
		"GIT_AUTHOR_NAME": "test",
		"GIT_AUTHOR_EMAIL": "test@test.com",
		"GIT_COMMITTER_NAME": "test",
		"GIT_COMMITTER_EMAIL": "test@test.com",
		"PATH": subprocess.check_output(["bash", "-c", "echo $PATH"]).decode().strip(),
	}


@pytest.fixture()
def source_repo(tmp_path: Path) -> Path:
	"""Create a real git repo to use as the clone source."""
	repo = tmp_path / "source"
	repo.mkdir()
	env = _git_env()
	subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
	subprocess.run(["git", "checkout", "-b", "main"], cwd=str(repo), check=True, capture_output=True)
	readme = repo / "README.md"
	readme.write_text("# Test repo\n")
	subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
	subprocess.run(
		["git", "commit", "-m", "Initial commit"],
		cwd=str(repo), check=True, capture_output=True, env=env,
	)
	return repo


@pytest.fixture()
def pool_dir(tmp_path: Path) -> Path:
	"""Directory for workspace clones."""
	return tmp_path / "pool"


class TestWorkspaceState:
	def test_defaults(self) -> None:
		state = WorkspaceState()
		assert state.last_branch == ""
		assert state.last_fetch_time == 0.0
		assert state.is_dirty is True
		assert state.last_green_sha == ""

	def test_custom_values(self) -> None:
		state = WorkspaceState(
			last_branch="main",
			last_fetch_time=123.456,
			is_dirty=False,
			last_green_sha="abc123",
		)
		assert state.last_branch == "main"
		assert state.last_fetch_time == 123.456
		assert state.is_dirty is False
		assert state.last_green_sha == "abc123"


class TestWarmClaim:
	async def test_warm_claim_skips_fetch(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""A warm workspace within TTL skips git fetch on claim."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3, fetch_ttl=60.0)
		await pool.initialize()

		# Acquire + release to warm the workspace
		workspace = await pool.acquire()
		assert workspace is not None
		await pool.release(workspace)

		# State should be warm
		state = pool.get_workspace_state(workspace)
		assert state is not None
		assert state.is_dirty is False

		# Claim should reuse the warm workspace.
		# The state's fetch_ttl hasn't expired, so fetch should be skipped.
		claimed = await pool.claim(branch="main")
		assert claimed is not None

		# After claim, workspace should still have recent fetch time
		state_after = pool.get_workspace_state(claimed)
		assert state_after is not None

		await pool.cleanup()

	async def test_warm_claim_skips_fetch_via_subprocess_spy(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Verify warm claim actually skips the fetch subprocess call."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3, fetch_ttl=120.0)
		await pool.initialize()

		# Acquire + release to warm the workspace
		ws = await pool.acquire()
		assert ws is not None
		await pool.release(ws)

		# The workspace is now warm with a recent fetch time.
		# Claim for the same branch should skip fetch entirely.
		state = pool.get_workspace_state(ws)
		assert state is not None
		assert state.is_dirty is False
		target_branch = pool.base_branch
		assert state.last_branch == target_branch

		calls: list[tuple[str, ...]] = []
		real_exec = __import__("asyncio").create_subprocess_exec

		async def spy_exec(*args: str, **kwargs: object) -> object:
			calls.append(args)
			return await real_exec(*args, **kwargs)

		with patch("asyncio.create_subprocess_exec", side_effect=spy_exec):
			claimed = await pool.claim(branch="main")

		assert claimed is not None

		# fetch should NOT appear in subprocess calls (warm path)
		fetch_calls = [c for c in calls if "fetch" in c]
		assert len(fetch_calls) == 0, f"Expected no fetch calls, got {fetch_calls}"

		# checkout should also be skipped since we're already on main
		checkout_calls = [c for c in calls if "checkout" in c]
		assert len(checkout_calls) == 0, f"Expected no checkout calls, got {checkout_calls}"

		await pool.cleanup()

	async def test_cold_claim_does_full_fetch(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""A freshly created (cold) workspace does full fetch on claim."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3, fetch_ttl=60.0)
		await pool.initialize()

		calls: list[tuple[str, ...]] = []
		real_exec = __import__("asyncio").create_subprocess_exec

		async def spy_exec(*args: str, **kwargs: object) -> object:
			calls.append(args)
			return await real_exec(*args, **kwargs)

		with patch("asyncio.create_subprocess_exec", side_effect=spy_exec):
			claimed = await pool.claim(branch="main")

		assert claimed is not None

		# Cold workspace has no state, so fetch SHOULD be called
		fetch_calls = [c for c in calls if "fetch" in c]
		assert len(fetch_calls) >= 1, f"Expected fetch calls for cold workspace, got {calls}"

		await pool.cleanup()

	async def test_expired_ttl_triggers_fetch(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""When TTL has expired, claim performs a fresh fetch."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3, fetch_ttl=0.0)
		await pool.initialize()

		# Acquire + release to warm
		ws = await pool.acquire()
		assert ws is not None
		await pool.release(ws)

		calls: list[tuple[str, ...]] = []
		real_exec = __import__("asyncio").create_subprocess_exec

		async def spy_exec(*args: str, **kwargs: object) -> object:
			calls.append(args)
			return await real_exec(*args, **kwargs)

		# TTL=0 means fetch is always needed
		with patch("asyncio.create_subprocess_exec", side_effect=spy_exec):
			claimed = await pool.claim(branch="main")

		assert claimed is not None
		fetch_calls = [c for c in calls if "fetch" in c]
		assert len(fetch_calls) >= 1, "Expected fetch when TTL expired"

		await pool.cleanup()


class TestDirtyWorkspace:
	async def test_dirty_workspace_gets_cleaned_on_release(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""A dirty workspace is cleaned during release and marked not dirty."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		ws = await pool.acquire()
		assert ws is not None

		# Make the workspace dirty
		dirty_file = ws / "dirty.txt"
		dirty_file.write_text("dirty content")
		assert dirty_file.exists()

		await pool.release(ws)

		# Dirty file removed by _reset_clone
		assert not dirty_file.exists()

		# State should reflect clean workspace
		state = pool.get_workspace_state(ws)
		assert state is not None
		assert state.is_dirty is False

		await pool.cleanup()

	async def test_dirty_flag_forces_full_ops_on_claim(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""A workspace manually marked dirty gets full git ops on claim."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3, fetch_ttl=120.0)
		await pool.initialize()

		# Acquire + release to warm
		ws = await pool.acquire()
		assert ws is not None
		await pool.release(ws)

		# Manually mark as dirty
		state = pool.get_workspace_state(ws)
		assert state is not None
		state.is_dirty = True

		calls: list[tuple[str, ...]] = []
		real_exec = __import__("asyncio").create_subprocess_exec

		async def spy_exec(*args: str, **kwargs: object) -> object:
			calls.append(args)
			return await real_exec(*args, **kwargs)

		with patch("asyncio.create_subprocess_exec", side_effect=spy_exec):
			claimed = await pool.claim(branch="main")

		assert claimed is not None
		# Dirty workspace should trigger fetch
		fetch_calls = [c for c in calls if "fetch" in c]
		assert len(fetch_calls) >= 1, "Expected fetch for dirty workspace"

		await pool.cleanup()


class TestPoolStats:
	async def test_stats_initial(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Empty pool stats."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=5)
		await pool.initialize()

		stats = pool.get_pool_stats()
		assert stats["total_workspaces"] == 0
		assert stats["warm_count"] == 0
		assert stats["cold_count"] == 0
		assert stats["avg_claim_time_ms"] == 0.0

		await pool.cleanup()

	async def test_stats_after_warm_cycle(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Stats reflect warm workspaces after acquire+release cycle."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=5)
		await pool.initialize()

		ws1 = await pool.acquire()
		ws2 = await pool.acquire()
		assert ws1 is not None
		assert ws2 is not None

		# Release one to warm it
		await pool.release(ws1)

		stats = pool.get_pool_stats()
		assert stats["total_workspaces"] == 2
		assert stats["warm_count"] == 1
		assert stats["cold_count"] == 0  # cold_count = available - warm

		await pool.cleanup()

	async def test_stats_avg_claim_time(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""avg_claim_time_ms is populated after claims."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=5, fetch_ttl=120.0)
		await pool.initialize()

		# Do a claim to record timing
		ws = await pool.claim(branch="main")
		assert ws is not None

		stats = pool.get_pool_stats()
		assert stats["avg_claim_time_ms"] > 0
		assert stats["total_workspaces"] == 1

		await pool.cleanup()

	async def test_stats_reset_on_cleanup(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Cleanup resets all stats."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		ws = await pool.claim(branch="main")
		assert ws is not None
		await pool.release(ws)

		await pool.cleanup()

		stats = pool.get_pool_stats()
		assert stats["total_workspaces"] == 0
		assert stats["warm_count"] == 0
		assert stats["avg_claim_time_ms"] == 0.0


class TestReleaseWarmsState:
	async def test_release_records_sha(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Release records the HEAD sha in workspace state."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		ws = await pool.acquire()
		assert ws is not None
		await pool.release(ws)

		state = pool.get_workspace_state(ws)
		assert state is not None
		assert len(state.last_green_sha) == 40  # full sha1

		await pool.cleanup()

	async def test_release_records_branch(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Release records the branch the workspace was reset to."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		await pool.initialize()

		ws = await pool.acquire()
		assert ws is not None
		await pool.release(ws)

		state = pool.get_workspace_state(ws)
		assert state is not None
		assert state.last_branch == "main"

		await pool.cleanup()

	async def test_release_records_green_branch(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Release records green branch when configured."""
		env = _git_env()
		subprocess.run(
			["git", "checkout", "-b", "mc/green"],
			cwd=str(source_repo), check=True, capture_output=True,
		)
		green_file = source_repo / "green.txt"
		green_file.write_text("from green")
		subprocess.run(["git", "add", "."], cwd=str(source_repo), check=True, capture_output=True)
		subprocess.run(
			["git", "commit", "-m", "green commit"],
			cwd=str(source_repo), check=True, capture_output=True, env=env,
		)
		subprocess.run(
			["git", "checkout", "main"],
			cwd=str(source_repo), check=True, capture_output=True,
		)

		pool = WorkspacePool(
			source_repo, pool_dir, max_clones=3, green_branch="mc/green",
		)
		await pool.initialize()

		ws = await pool.acquire()
		assert ws is not None
		await pool.release(ws)

		state = pool.get_workspace_state(ws)
		assert state is not None
		assert state.last_branch == "mc/green"

		await pool.cleanup()


class TestFetchTTLConfig:
	async def test_custom_fetch_ttl(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Pool respects custom fetch_ttl parameter."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3, fetch_ttl=300.0)
		assert pool.fetch_ttl == 300.0

		await pool.cleanup()

	async def test_default_fetch_ttl(
		self, source_repo: Path, pool_dir: Path,
	) -> None:
		"""Default fetch_ttl is 60 seconds."""
		pool = WorkspacePool(source_repo, pool_dir, max_clones=3)
		assert pool.fetch_ttl == 60.0

		await pool.cleanup()
