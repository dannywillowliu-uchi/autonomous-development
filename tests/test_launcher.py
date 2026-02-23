"""Tests for mission launcher."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mission_control.db import Database
from mission_control.launcher import MissionLauncher, _pid_alive
from mission_control.models import Mission
from mission_control.registry import ProjectRegistry


@pytest.fixture
def registry(tmp_path):
	db_path = tmp_path / "test_registry.db"
	reg = ProjectRegistry(db_path=db_path, allowed_bases=[tmp_path, Path.home()])
	yield reg
	reg.close()


@pytest.fixture
def sample_config(tmp_path):
	config = tmp_path / "mission-control.toml"
	config.write_text(
		'[target]\nname = "test"\npath = "."\nbranch = "main"\nobjective = "test"\n'
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


@pytest.fixture
def launcher(registry):
	return MissionLauncher(registry)


class TestMissionLauncher:
	def test_launch_unregistered_project(self, launcher):
		with pytest.raises(ValueError, match="not registered"):
			launcher.launch("nonexistent")

	def test_launch_missing_config(self, launcher, registry, tmp_path):
		registry.register(
			name="test",
			config_path=str(tmp_path / "nonexistent.toml"),
		)
		with pytest.raises(FileNotFoundError):
			launcher.launch("test")

	@patch("mission_control.launcher.subprocess.Popen")
	def test_launch_spawns_subprocess(self, mock_popen, launcher, registry, sample_config):
		mock_proc = MagicMock()
		mock_proc.pid = 99999
		mock_popen.return_value = mock_proc

		registry.register(name="test", config_path=str(sample_config))
		pid = launcher.launch("test")

		assert pid == 99999
		mock_popen.assert_called_once()
		# Verify PID stored in registry
		project = registry.get_project("test")
		assert project.active_pid == 99999

	@patch("mission_control.launcher.subprocess.Popen")
	@patch("mission_control.launcher._pid_alive", return_value=True)
	def test_launch_fails_if_already_running(self, mock_alive, mock_popen, launcher, registry, sample_config):
		mock_proc = MagicMock()
		mock_proc.pid = 99999
		mock_popen.return_value = mock_proc

		registry.register(name="test", config_path=str(sample_config))
		launcher.launch("test")

		with pytest.raises(RuntimeError, match="already has a running mission"):
			launcher.launch("test")

	def test_is_running_no_pid(self, launcher, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))
		assert launcher.is_running("test") is False

	@patch("mission_control.launcher._pid_alive", return_value=False)
	def test_is_running_stale_pid(self, mock_alive, launcher, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))
		registry.update_pid("test", 99999)
		assert launcher.is_running("test") is False
		# Stale PID should be cleaned up
		project = registry.get_project("test")
		assert project.active_pid is None

	def test_stop_inserts_signal(self, launcher, registry, sample_config, tmp_path):
		db_path = tmp_path / "mission-control.db"
		registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))

		# Create DB with running mission
		db = Database(db_path)
		mission = Mission(objective="test", status="running")
		db.insert_mission(mission)

		result = launcher.stop("test")
		assert result is True

		signals = db.get_pending_signals(mission.id)
		assert len(signals) == 1
		assert signals[0].signal_type == "stop"
		db.close()

	def test_stop_no_running_mission(self, launcher, registry, sample_config, tmp_path):
		db_path = tmp_path / "mission-control.db"
		registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))

		db = Database(db_path)
		mission = Mission(objective="test", status="completed")
		db.insert_mission(mission)
		db.close()

		result = launcher.stop("test")
		assert result is False

	def test_retry_unit_inserts_signal(self, launcher, registry, sample_config, tmp_path):
		db_path = tmp_path / "mission-control.db"
		registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))

		db = Database(db_path)
		mission = Mission(objective="test", status="running")
		db.insert_mission(mission)

		result = launcher.retry_unit("test", "unit-abc")
		assert result is True

		signals = db.get_pending_signals(mission.id)
		assert len(signals) == 1
		assert signals[0].signal_type == "retry_unit"
		assert signals[0].payload == "unit-abc"
		db.close()

	def test_adjust_inserts_signal(self, launcher, registry, sample_config, tmp_path):
		import json

		db_path = tmp_path / "mission-control.db"
		registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))

		db = Database(db_path)
		mission = Mission(objective="test", status="running")
		db.insert_mission(mission)

		result = launcher.adjust("test", {"max_rounds": 50})
		assert result is True

		signals = db.get_pending_signals(mission.id)
		assert len(signals) == 1
		assert signals[0].signal_type == "adjust"
		assert json.loads(signals[0].payload) == {"max_rounds": 50}
		db.close()


class TestPidAlive:
	def test_current_process(self):
		assert _pid_alive(os.getpid()) is True

	def test_nonexistent_pid(self):
		assert _pid_alive(999999999) is False
