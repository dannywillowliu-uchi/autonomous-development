"""Tests for project registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_control.registry import ProjectRegistry


@pytest.fixture
def registry(tmp_path):
	db_path = tmp_path / "test_registry.db"
	reg = ProjectRegistry(db_path=db_path, allowed_bases=[tmp_path, Path.home()])
	yield reg
	reg.close()


@pytest.fixture
def sample_config(tmp_path):
	"""Create a minimal config file."""
	config = tmp_path / "mission-control.toml"
	config.write_text(
		'[target]\nname = "test"\npath = "."\nbranch = "main"\nobjective = ""\n'
		'[target.verification]\ncommand = "echo ok"\ntimeout = 60\n'
		"[scheduler]\nmodel = \"sonnet\"\n"
		"[scheduler.git]\nstrategy = \"branch-per-session\"\n"
		"[scheduler.budget]\nmax_per_session_usd = 1.0\nmax_per_run_usd = 10.0\n"
		"[scheduler.parallel]\nnum_workers = 2\n"
		"[rounds]\nmax_rounds = 5\nstall_threshold = 3\n"
		"[planner]\nmax_depth = 2\n"
		"[green_branch]\nworking_branch = \"mc/working\"\ngreen_branch = \"mc/green\"\n"
		'[backend]\ntype = "local"\n'
	)
	return config


class TestProjectRegistry:
	def test_register_project(self, registry, sample_config):
		project = registry.register(
			name="test-project",
			config_path=str(sample_config),
			description="A test project",
		)
		assert project.name == "test-project"
		assert project.description == "A test project"
		assert str(sample_config.resolve()) in project.config_path

	def test_register_derives_db_path(self, registry, sample_config):
		project = registry.register(name="test", config_path=str(sample_config))
		assert project.db_path.endswith("mission-control.db")

	def test_register_duplicate_name_fails(self, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))
		with pytest.raises(ValueError, match="already registered"):
			registry.register(name="test", config_path=str(sample_config))

	def test_unregister(self, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))
		assert registry.unregister("test") is True
		assert registry.get_project("test") is None

	def test_unregister_nonexistent(self, registry):
		assert registry.unregister("nope") is False

	def test_list_projects(self, registry, sample_config):
		registry.register(name="alpha", config_path=str(sample_config))
		registry.register(name="beta", config_path=str(sample_config))

		projects = registry.list_projects()
		names = [p.name for p in projects]
		assert "alpha" in names
		assert "beta" in names
		assert len(projects) == 2

	def test_list_projects_sorted(self, registry, sample_config):
		registry.register(name="zebra", config_path=str(sample_config))
		registry.register(name="alpha", config_path=str(sample_config))

		projects = registry.list_projects()
		assert projects[0].name == "alpha"
		assert projects[1].name == "zebra"

	def test_get_project(self, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))
		project = registry.get_project("test")
		assert project is not None
		assert project.name == "test"

	def test_get_project_nonexistent(self, registry):
		assert registry.get_project("nope") is None

	def test_update_pid(self, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))
		registry.update_pid("test", 12345)

		project = registry.get_project("test")
		assert project.active_pid == 12345

		registry.update_pid("test", None)
		project = registry.get_project("test")
		assert project.active_pid is None

	def test_get_project_status_no_db(self, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))
		status = registry.get_project_status("test")
		assert status is not None
		assert status.mission_status == "idle"

	def test_register_rejects_traversal_path(self, registry):
		with pytest.raises(ValueError, match="path outside allowed directories"):
			registry.register(
				name="evil",
				config_path="../../../../etc/passwd",
			)

	def test_register_rejects_null_byte_path(self, registry):
		with pytest.raises(ValueError, match="invalid path"):
			registry.register(
				name="evil",
				config_path="/home/user/config\x00.toml",
			)

	def test_get_project_status_with_mission(self, registry, sample_config, tmp_path):
		# Create a project DB with a mission
		from mission_control.db import Database
		from mission_control.models import Mission

		db_path = tmp_path / "mission-control.db"
		db = Database(db_path)
		mission = Mission(objective="build stuff", status="running", total_rounds=3, final_score=0.7)
		db.insert_mission(mission)
		db.close()

		registry.register(
			name="test",
			config_path=str(sample_config),
			db_path=str(db_path),
		)

		status = registry.get_project_status("test")
		assert status.mission_status == "running"
		assert status.mission_objective == "build stuff"
		assert status.mission_score == 0.7
		assert status.mission_rounds == 3
