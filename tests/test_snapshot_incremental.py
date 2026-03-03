"""Tests for incremental snapshot building and cache invalidation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from mission_control.snapshot import (
	_snapshot_cache,
	_snapshot_entries_cache,
	build_incremental_snapshot,
	build_project_snapshot,
	clear_snapshot_cache,
	get_project_snapshot,
	invalidate_snapshot,
)


def _write(path: Path, content: str) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(dedent(content))
	return path


class TestInvalidateSnapshot:
	def test_removes_specific_root(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "a.py", "def a(): pass")
		get_project_snapshot(tmp_path)
		resolved = tmp_path.resolve()
		assert resolved in _snapshot_cache
		assert resolved in _snapshot_entries_cache

		invalidate_snapshot(tmp_path)
		assert resolved not in _snapshot_cache
		assert resolved not in _snapshot_entries_cache

	def test_noop_for_unknown_root(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		# Should not raise even for a path not in the cache
		invalidate_snapshot(tmp_path / "nonexistent")

	def test_invalidate_then_rebuild(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "mod.py", "def old(): pass")
		r1 = get_project_snapshot(tmp_path)
		assert "def old" in r1

		invalidate_snapshot(tmp_path)
		_write(tmp_path / "mod.py", "def new(): pass")
		r2 = get_project_snapshot(tmp_path)
		assert "def new" in r2
		assert "def old" not in r2


class TestBuildIncrementalSnapshot:
	def test_falls_back_to_full_build_without_cache(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "app.py", "def app(): pass")
		result = build_incremental_snapshot(tmp_path, ["app.py"])
		assert "app.py" in result
		assert "def app" in result

	def test_incremental_updates_changed_file(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "core.py", "def core_v1(): pass")
		_write(tmp_path / "utils.py", "def helper(): pass")
		build_project_snapshot(tmp_path)

		# Modify core.py
		_write(tmp_path / "core.py", "def core_v2(): pass")
		result = build_incremental_snapshot(tmp_path, ["core.py"])

		assert "def core_v2" in result
		assert "def core_v1" not in result
		# utils.py should still be present from cache
		assert "def helper" in result

	def test_empty_changed_files_returns_cached(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "mod.py", "def original(): pass")
		build_project_snapshot(tmp_path)

		# Modify file on disk but pass empty changed_files
		_write(tmp_path / "mod.py", "def changed(): pass")
		result = build_incremental_snapshot(tmp_path, [])

		# Should return cached version (original)
		assert "def original" in result

	def test_handles_deleted_file(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "keep.py", "def keep(): pass")
		_write(tmp_path / "remove.py", "def remove(): pass")
		build_project_snapshot(tmp_path)

		(tmp_path / "remove.py").unlink()
		result = build_incremental_snapshot(tmp_path, ["remove.py"])

		assert "def keep" in result
		assert "remove.py" not in result

	def test_adds_new_file(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "existing.py", "def existing(): pass")
		build_project_snapshot(tmp_path)

		_write(tmp_path / "brand_new.py", "def brand_new(): pass")
		result = build_incremental_snapshot(tmp_path, ["brand_new.py"])

		assert "def existing" in result
		assert "def brand_new" in result

	def test_non_python_files_ignored(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "code.py", "def code(): pass")
		build_project_snapshot(tmp_path)

		_write(tmp_path / "readme.md", "# Hello")
		result = build_incremental_snapshot(tmp_path, ["readme.md"])

		assert "def code" in result
		assert "readme" not in result

	def test_updates_both_caches(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "mod.py", "def v1(): pass")
		build_project_snapshot(tmp_path)

		_write(tmp_path / "mod.py", "def v2(): pass")
		build_incremental_snapshot(tmp_path, ["mod.py"])

		resolved = tmp_path.resolve()
		assert "def v2" in _snapshot_cache[resolved]
		assert any("def v2" in sig for _, sigs in _snapshot_entries_cache[resolved] for sig in sigs)


class TestClearSnapshotCache:
	def test_clears_both_caches(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "a.py", "def a(): pass")
		build_project_snapshot(tmp_path)

		resolved = tmp_path.resolve()
		assert resolved in _snapshot_cache
		assert resolved in _snapshot_entries_cache

		clear_snapshot_cache()
		assert resolved not in _snapshot_cache
		assert resolved not in _snapshot_entries_cache


class TestWorkerPromptIntegration:
	def test_render_includes_snapshot_when_project_root_given(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "src" / "api.py", """\
			def handle_request(data: dict) -> str:
				return "ok"
		""")

		from mission_control.models import WorkUnit
		from mission_control.worker import render_mission_worker_prompt

		unit = WorkUnit(
			title="Fix API handler",
			description="Fix the handler",
			files_hint="src/api.py",
		)

		# Minimal config mock
		class _VerifCfg:
			command = "pytest"

		class _TargetCfg:
			name = "test-project"
			verification = _VerifCfg()

		class _Cfg:
			target = _TargetCfg()

		prompt = render_mission_worker_prompt(
			unit=unit,
			config=_Cfg(),  # type: ignore[arg-type]
			workspace_path=str(tmp_path),
			branch_name="mc/test",
			project_root=tmp_path,
		)
		assert "Project Snapshot" in prompt
		assert "def handle_request" in prompt

	def test_render_no_snapshot_without_project_root(self) -> None:
		from mission_control.models import WorkUnit
		from mission_control.worker import render_mission_worker_prompt

		unit = WorkUnit(
			title="Some task",
			description="Description",
		)

		class _VerifCfg:
			command = "pytest"

		class _TargetCfg:
			name = "test-project"
			verification = _VerifCfg()

		class _Cfg:
			target = _TargetCfg()

		prompt = render_mission_worker_prompt(
			unit=unit,
			config=_Cfg(),  # type: ignore[arg-type]
			workspace_path="/tmp/ws",
			branch_name="mc/test",
		)
		assert "Project Snapshot" not in prompt
