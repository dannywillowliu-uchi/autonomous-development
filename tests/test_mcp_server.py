"""Tests for MCP server tool handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mcp", reason="mcp package not installed")

from mission_control.db import Database
from mission_control.models import Mission


@pytest.fixture
def registry(tmp_path):
	from mission_control.registry import ProjectRegistry
	db_path = tmp_path / "test_registry.db"
	reg = ProjectRegistry(db_path=db_path)
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


class TestMCPToolHandlers:
	"""Test the _dispatch function directly (no MCP transport)."""

	def test_list_projects_empty(self, registry):
		from mission_control.mcp_server import _dispatch
		result = _dispatch("list_projects", {}, registry)
		assert result["count"] == 0
		assert result["projects"] == []

	def test_list_projects_with_data(self, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))
		from mission_control.mcp_server import _dispatch
		result = _dispatch("list_projects", {}, registry)
		assert result["count"] == 1
		assert result["projects"][0]["name"] == "test"

	def test_get_project_status_not_found(self, registry):
		from mission_control.mcp_server import _dispatch
		result = _dispatch("get_project_status", {"project_name": "nope"}, registry)
		assert "error" in result

	def test_get_project_status_found(self, registry, sample_config, tmp_path):
		db_path = tmp_path / "mission-control.db"
		registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))

		db = Database(db_path)
		mission = Mission(objective="build", status="running", total_rounds=2, final_score=0.5)
		db.insert_mission(mission)
		db.close()

		from mission_control.mcp_server import _dispatch
		result = _dispatch("get_project_status", {"project_name": "test"}, registry)
		assert result["mission_status"] == "running"
		assert result["mission_score"] == 0.5

	def test_stop_mission_no_project(self, registry):
		from mission_control.mcp_server import _dispatch
		result = _dispatch("stop_mission", {"project_name": "nope"}, registry)
		assert "error" in result

	def test_stop_mission(self, registry, sample_config, tmp_path):
		db_path = tmp_path / "mission-control.db"
		registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))

		db = Database(db_path)
		mission = Mission(objective="test", status="running")
		db.insert_mission(mission)
		db.close()

		from mission_control.mcp_server import _dispatch
		result = _dispatch("stop_mission", {"project_name": "test"}, registry)
		assert result["status"] == "stop_signal_sent"

	def test_retry_unit(self, registry, sample_config, tmp_path):
		db_path = tmp_path / "mission-control.db"
		registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))

		db = Database(db_path)
		mission = Mission(objective="test", status="running")
		db.insert_mission(mission)
		db.close()

		from mission_control.mcp_server import _dispatch
		result = _dispatch("retry_unit", {"project_name": "test", "unit_id": "abc123"}, registry)
		assert result["status"] == "retry_signal_sent"

	def test_adjust_mission(self, registry, sample_config, tmp_path):
		db_path = tmp_path / "mission-control.db"
		registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))

		db = Database(db_path)
		mission = Mission(objective="test", status="running")
		db.insert_mission(mission)
		db.close()

		from mission_control.mcp_server import _dispatch
		result = _dispatch(
			"adjust_mission",
			{"project_name": "test", "max_rounds": 50},
			registry,
		)
		assert result["status"] == "adjusted"
		assert result["params"]["max_rounds"] == 50

	def test_adjust_no_params(self, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))
		from mission_control.mcp_server import _dispatch
		result = _dispatch("adjust_mission", {"project_name": "test"}, registry)
		assert "error" in result

	def test_register_project(self, registry, sample_config):
		from mission_control.mcp_server import _dispatch
		result = _dispatch(
			"register_project",
			{"config_path": str(sample_config), "name": "new-proj"},
			registry,
		)
		assert result["status"] == "registered"
		assert result["name"] == "new-proj"

	def test_register_duplicate(self, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))
		from mission_control.mcp_server import _dispatch
		result = _dispatch(
			"register_project",
			{"config_path": str(sample_config), "name": "test"},
			registry,
		)
		assert "error" in result

	def test_get_round_details_no_project(self, registry):
		from mission_control.mcp_server import _dispatch
		result = _dispatch("get_round_details", {"project_name": "nope"}, registry)
		assert "error" in result

	@patch("mission_control.mcp_server.subprocess", create=True)
	def test_launch_mission(self, mock_sub, registry, sample_config):
		registry.register(name="test", config_path=str(sample_config))

		from mission_control.mcp_server import _dispatch
		with patch("mission_control.launcher.subprocess.Popen") as mock_popen:
			mock_proc = MagicMock()
			mock_proc.pid = 12345
			mock_popen.return_value = mock_proc

			result = _dispatch(
				"launch_mission",
				{"project_name": "test", "max_rounds": 10},
				registry,
			)
			assert result["status"] == "launched"
			assert result["pid"] == 12345

	def test_unknown_tool(self, registry):
		from mission_control.mcp_server import _dispatch
		result = _dispatch("nonexistent", {}, registry)
		assert "error" in result
