"""Tests for MCP server tool handlers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mcp", reason="mcp package not installed")

from mission_control.db import Database
from mission_control.models import Mission


@pytest.fixture
def registry(tmp_path):
	from pathlib import Path

	from mission_control.registry import ProjectRegistry

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

	def test_register_rejects_traversal_path(self, registry):
		from mission_control.mcp_server import _dispatch
		result = _dispatch(
			"register_project",
			{"config_path": "../../../../etc/passwd", "name": "evil"},
			registry,
		)
		assert "error" in result
		assert "path outside allowed directories" in result["error"]

	def test_register_rejects_null_byte_path(self, registry):
		from mission_control.mcp_server import _dispatch
		result = _dispatch(
			"register_project",
			{"config_path": "/home/user/config\x00.toml", "name": "evil"},
			registry,
		)
		assert "error" in result
		assert "invalid path" in result["error"]

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

	def test_web_research_dispatch(self, registry):
		from mission_control.mcp_server import _dispatch
		with patch("mission_control.mcp_server._tool_web_research") as mock_wr:
			mock_wr.return_value = {"query": "test", "results": []}
			result = _dispatch("web_research", {"query": "test"}, registry)
			mock_wr.assert_called_once_with({"query": "test"})
			assert result["query"] == "test"


def _mock_urlopen(data: dict | str, content_type: str = "application/json"):
	"""Create a mock urllib response context manager."""
	if isinstance(data, dict):
		body = json.dumps(data).encode()
	else:
		body = data.encode()
	resp = MagicMock()
	resp.read.return_value = body
	resp.headers = {"Content-Type": content_type}
	resp.__enter__ = lambda s: s
	resp.__exit__ = MagicMock(return_value=False)
	return resp


class TestWebResearch:
	"""Tests for web_research tool and its search backends."""

	def test_general_search_with_abstract(self):
		from mission_control.mcp_server import _tool_web_research
		ddg_response = {
			"Heading": "Python",
			"Abstract": "Python is a programming language.",
			"AbstractURL": "https://python.org",
			"AbstractSource": "Wikipedia",
			"RelatedTopics": [],
			"Answer": "",
		}
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=_mock_urlopen(ddg_response)):
			result = _tool_web_research({"query": "python programming"})
		assert result["search_type"] == "general"
		assert len(result["results"]) == 1
		assert result["results"][0]["title"] == "Python"
		assert result["results"][0]["snippet"] == "Python is a programming language."

	def test_general_search_with_related_topics(self):
		from mission_control.mcp_server import _tool_web_research
		ddg_response = {
			"Heading": "",
			"Abstract": "",
			"RelatedTopics": [
				{"Text": "Topic one about Python", "FirstURL": "https://example.com/1"},
				{"Text": "Topic two about asyncio", "FirstURL": "https://example.com/2"},
			],
			"Answer": "",
		}
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=_mock_urlopen(ddg_response)):
			result = _tool_web_research({"query": "python asyncio"})
		assert len(result["results"]) == 2

	def test_general_search_with_subtopics(self):
		from mission_control.mcp_server import _tool_web_research
		ddg_response = {
			"Heading": "",
			"Abstract": "",
			"RelatedTopics": [
				{"Topics": [
					{"Text": "Sub topic A", "FirstURL": "https://example.com/a"},
				]},
			],
			"Answer": "",
		}
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=_mock_urlopen(ddg_response)):
			result = _tool_web_research({"query": "something"})
		assert len(result["results"]) == 1
		assert result["results"][0]["snippet"] == "Sub topic A"

	def test_general_search_answer_fallback(self):
		from mission_control.mcp_server import _tool_web_research
		ddg_response = {
			"Heading": "",
			"Abstract": "",
			"RelatedTopics": [],
			"Answer": "42",
		}
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=_mock_urlopen(ddg_response)):
			result = _tool_web_research({"query": "answer to everything"})
		assert len(result["results"]) == 1
		assert result["results"][0]["snippet"] == "42"

	def test_pypi_search(self):
		from mission_control.mcp_server import _tool_web_research
		pypi_response = {
			"info": {
				"name": "requests",
				"version": "2.31.0",
				"summary": "HTTP library",
				"home_page": "https://requests.readthedocs.io",
				"requires_python": ">=3.7",
				"license": "Apache 2.0",
				"project_urls": {"Homepage": "https://requests.readthedocs.io"},
			},
		}
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=_mock_urlopen(pypi_response)):
			result = _tool_web_research({"query": "requests", "search_type": "pypi"})
		assert result["search_type"] == "pypi"
		assert result["results"][0]["name"] == "requests"
		assert result["results"][0]["version"] == "2.31.0"

	def test_github_search(self):
		from mission_control.mcp_server import _tool_web_research
		gh_response = {
			"items": [
				{
					"full_name": "pallets/flask",
					"description": "Micro web framework",
					"html_url": "https://github.com/pallets/flask",
					"stargazers_count": 60000,
					"language": "Python",
					"updated_at": "2024-01-01T00:00:00Z",
				},
			],
		}
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=_mock_urlopen(gh_response)):
			result = _tool_web_research({"query": "flask", "search_type": "github", "max_results": 3})
		assert result["search_type"] == "github"
		assert len(result["results"]) == 1
		assert result["results"][0]["name"] == "pallets/flask"
		assert result["results"][0]["stars"] == 60000

	def test_url_fetch_html(self):
		from mission_control.mcp_server import _tool_web_research
		html = "<html><head><style>body{}</style></head><body><p>Hello world</p></body></html>"
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=_mock_urlopen(html, "text/html")):
			result = _tool_web_research({"query": "https://example.com", "search_type": "url_fetch"})
		assert result["search_type"] == "url_fetch"
		assert "Hello world" in result["results"][0]["content"]
		assert "<style>" not in result["results"][0]["content"]

	def test_url_fetch_json(self):
		from mission_control.mcp_server import _tool_web_research
		data = {"key": "value"}
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=_mock_urlopen(data)):
			result = _tool_web_research({"query": "https://api.example.com/data", "search_type": "url_fetch"})
		assert '"key"' in result["results"][0]["content"]

	def test_url_fetch_truncation(self):
		from mission_control.mcp_server import _tool_web_research
		long_text = "x" * 10000
		mock_resp = _mock_urlopen(long_text, "text/plain")
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=mock_resp):
			result = _tool_web_research({"query": "https://example.com/big", "search_type": "url_fetch"})
		assert result["results"][0]["truncated"] is True
		assert len(result["results"][0]["content"]) == 8000

	def test_unit_id_passthrough(self):
		from mission_control.mcp_server import _tool_web_research
		ddg_response = {"Heading": "", "Abstract": "", "RelatedTopics": [], "Answer": ""}
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=_mock_urlopen(ddg_response)):
			result = _tool_web_research({"query": "test", "unit_id": "abc123"})
		assert result["unit_id"] == "abc123"

	def test_unit_id_absent_when_not_provided(self):
		from mission_control.mcp_server import _tool_web_research
		ddg_response = {"Heading": "", "Abstract": "", "RelatedTopics": [], "Answer": ""}
		with patch("mission_control.mcp_server.urllib.request.urlopen", return_value=_mock_urlopen(ddg_response)):
			result = _tool_web_research({"query": "test"})
		assert "unit_id" not in result

	def test_network_error(self):
		import urllib.error

		from mission_control.mcp_server import _tool_web_research
		with patch("mission_control.mcp_server.urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
			result = _tool_web_research({"query": "fail"})
		assert "error" in result
		assert "Network error" in result["error"]

	def test_tool_in_tools_list(self):
		from mission_control.mcp_server import TOOLS
		names = [t.name for t in TOOLS]
		assert "web_research" in names
