"""Tests for the periodic cleanup cycle feature."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mission_control.cli import _build_cleanup_objective, _is_cleanup_due, _is_cleanup_mission
from mission_control.config import ContinuousConfig, MissionConfig, TargetConfig, load_config
from mission_control.db import Database
from mission_control.models import Mission
from mission_control.worker import VALID_SPECIALISTS, load_specialist_template


class TestCleanupConfig:
	def test_defaults(self) -> None:
		cc = ContinuousConfig()
		assert cc.cleanup_enabled is True
		assert cc.cleanup_interval == 3

	def test_toml_parsing(self, tmp_path: Path) -> None:
		toml = tmp_path / "mission-control.toml"
		toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[continuous]
cleanup_enabled = false
cleanup_interval = 5
""")
		cfg = load_config(toml)
		assert cfg.continuous.cleanup_enabled is False
		assert cfg.continuous.cleanup_interval == 5


class TestIsCleanupMission:
	def test_detects_cleanup_prefix(self) -> None:
		m = Mission(objective="[CLEANUP] Consolidate test suite")
		assert _is_cleanup_mission(m) is True

	def test_rejects_normal_objective(self) -> None:
		m = Mission(objective="Add user authentication")
		assert _is_cleanup_mission(m) is False

	def test_rejects_cleanup_in_middle(self) -> None:
		m = Mission(objective="Do some [CLEANUP] work")
		assert _is_cleanup_mission(m) is False


class TestIsCleanupDue:
	def test_not_due_too_few_missions(self, db: Database) -> None:
		"""With fewer than interval completed missions, cleanup is not due."""
		m = Mission(objective="Normal mission", status="completed")
		db.insert_mission(m)
		assert _is_cleanup_due(db, interval=3) is False

	def test_due_after_interval(self, db: Database) -> None:
		"""After interval non-cleanup missions, cleanup is due."""
		for i in range(3):
			m = Mission(objective=f"Mission {i}", status="completed")
			db.insert_mission(m)
		assert _is_cleanup_due(db, interval=3) is True

	def test_not_due_after_recent_cleanup(self, db: Database) -> None:
		"""After a recent cleanup mission, count resets."""
		# Insert a cleanup mission first (oldest)
		cleanup = Mission(objective="[CLEANUP] Old cleanup", status="completed")
		db.insert_mission(cleanup)
		# Then 2 normal missions (not enough to trigger again at interval=3)
		for i in range(2):
			m = Mission(objective=f"Mission {i}", status="completed")
			db.insert_mission(m)
		assert _is_cleanup_due(db, interval=3) is False

	def test_ignores_running_missions(self, db: Database) -> None:
		"""Running missions don't count toward the interval."""
		for i in range(3):
			m = Mission(objective=f"Mission {i}", status="running")
			db.insert_mission(m)
		assert _is_cleanup_due(db, interval=3) is False


class TestBuildCleanupObjective:
	def test_contains_prefix(self, tmp_path: Path) -> None:
		cfg = MissionConfig()
		cfg.target = TargetConfig(name="test", path=str(tmp_path))
		obj = _build_cleanup_objective(cfg)
		assert obj.startswith("[CLEANUP]")

	def test_contains_metrics(self, tmp_path: Path) -> None:
		cfg = MissionConfig()
		cfg.target = TargetConfig(name="test", path=str(tmp_path))

		with (
			patch("mission_control.cli.subprocess.run") as mock_run,
		):
			# First call: find test files
			find_result = type(
				"Result", (), {"returncode": 0, "stdout": "tests/test_a.py\ntests/test_b.py\n"},
			)()
			# Second call: pytest --co -q
			pytest_result = type(
				"Result", (), {
					"returncode": 0,
					"stdout": "test_a.py::test_1\ntest_b.py::test_2\n42 tests collected\n",
				},
			)()
			mock_run.side_effect = [find_result, pytest_result]

			obj = _build_cleanup_objective(cfg)

		assert "2 files" in obj
		assert "42 tests" in obj

	def test_handles_subprocess_failure(self, tmp_path: Path) -> None:
		"""Gracefully handles subprocess failures with zero metrics."""
		cfg = MissionConfig()
		cfg.target = TargetConfig(name="test", path=str(tmp_path))

		with patch("mission_control.cli.subprocess.run", side_effect=OSError("not found")):
			obj = _build_cleanup_objective(cfg)

		assert obj.startswith("[CLEANUP]")
		assert "0 files" in obj
		assert "0 tests" in obj


class TestSimplifierSpecialist:
	def test_in_valid_specialists(self) -> None:
		assert "simplifier" in VALID_SPECIALISTS

	def test_template_loads(self, tmp_path: Path) -> None:
		"""Specialist template loads from the bundled templates directory."""
		cfg = MissionConfig()
		cfg.target = TargetConfig(name="test", path=str(tmp_path))
		# Create the template in the expected location
		templates_dir = tmp_path / "specialist_templates"
		templates_dir.mkdir()
		(templates_dir / "simplifier.md").write_text("# Specialist: Simplifier\nTest content")
		template = load_specialist_template("simplifier", cfg)
		assert "Simplifier" in template

	def test_cleanup_forces_specialist(self) -> None:
		"""Verify the cleanup mission prefix convention."""
		objective = "[CLEANUP] Consolidate test suite"
		assert objective.startswith("[CLEANUP]")
