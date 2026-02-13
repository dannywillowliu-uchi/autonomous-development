"""Tests for web dashboard endpoints (multi-project, control, wizard)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mission_control.db import Database
from mission_control.models import Mission
from mission_control.registry import ProjectRegistry


@pytest.fixture
def registry(tmp_path):
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
		'[scheduler]\nmodel = "sonnet"\n'
		'[scheduler.git]\nstrategy = "branch-per-session"\n'
		'[scheduler.budget]\nmax_per_session_usd = 1.0\nmax_per_run_usd = 10.0\n'
		'[scheduler.parallel]\nnum_workers = 2\n'
		'[rounds]\nmax_rounds = 5\nstall_threshold = 3\n'
		'[planner]\nmax_depth = 2\n'
		'[green_branch]\nworking_branch = "mc/working"\ngreen_branch = "mc/green"\n'
		'[backend]\ntype = "local"\n'
	)
	return config


@pytest.fixture
def project_with_db(registry, sample_config, tmp_path):
	"""Register a project with a DB that has a running mission."""
	db_path = tmp_path / "mission-control.db"
	registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))

	db = Database(db_path)
	mission = Mission(objective="build", status="running", total_rounds=2, final_score=0.5)
	db.insert_mission(mission)
	db.close()

	return {"name": "test", "db_path": db_path}


@pytest.fixture
def client(registry):
	from mission_control.dashboard.web.server import create_app
	app = create_app(registry=registry)
	with TestClient(app) as c:
		yield c


@pytest.fixture
def client_with_project(registry, project_with_db):
	from mission_control.dashboard.web.server import create_app
	app = create_app(registry=registry)
	with TestClient(app) as c:
		yield c


class TestLayoutAndOverview:
	def test_index_returns_html(self, client):
		resp = client.get("/")
		assert resp.status_code == 200
		assert "Mission Control" in resp.text

	def test_overview_empty(self, client):
		resp = client.get("/overview")
		assert resp.status_code == 200
		assert "No projects registered" in resp.text

	def test_overview_with_project(self, client_with_project):
		resp = client_with_project.get("/overview")
		assert resp.status_code == 200
		assert "test" in resp.text

	def test_tab_bar(self, client):
		resp = client.get("/partials/tab-bar")
		assert resp.status_code == 200


class TestProjectDashboard:
	def test_project_dashboard(self, client_with_project):
		resp = client_with_project.get("/project/test/dashboard")
		assert resp.status_code == 200
		assert "test" in resp.text

	def test_project_mission_header(self, client_with_project):
		resp = client_with_project.get("/project/test/partials/mission-header")
		assert resp.status_code == 200

	def test_project_score_history_api(self, client_with_project):
		resp = client_with_project.get("/project/test/api/score-history")
		assert resp.status_code == 200
		data = resp.json()
		assert "labels" in data
		assert "data" in data


class TestControlEndpoints:
	def test_stop_mission(self, client_with_project):
		resp = client_with_project.post("/project/test/stop")
		assert resp.status_code == 200
		assert "toast" in resp.text

	def test_stop_nonexistent_project(self, client):
		resp = client.post("/project/nope/stop")
		assert resp.status_code == 200
		# Returns a toast with error

	def test_retry_unit(self, client_with_project):
		resp = client_with_project.post("/project/test/retry/unit123")
		assert resp.status_code == 200

	def test_adjust_mission(self, client_with_project):
		resp = client_with_project.post(
			"/project/test/adjust",
			data={"max_rounds": "50"},
		)
		assert resp.status_code == 200


class TestWizardEndpoints:
	def test_wizard_step1(self, client_with_project):
		resp = client_with_project.get("/launch/step/1")
		assert resp.status_code == 200
		assert "test" in resp.text

	def test_wizard_step2(self, client_with_project):
		resp = client_with_project.get("/launch/step/2?project=test")
		assert resp.status_code == 200

	def test_wizard_step3(self, client_with_project):
		resp = client_with_project.get("/launch/step/3?project=test&objective=build")
		assert resp.status_code == 200

	def test_wizard_step4(self, client_with_project):
		resp = client_with_project.get(
			"/launch/step/4?project=test&objective=build&model=sonnet&workers=4&max_rounds=20&budget=5.0"
		)
		assert resp.status_code == 200
		assert "Review" in resp.text

	@patch("mission_control.launcher.subprocess.Popen")
	def test_wizard_execute(self, mock_popen, client_with_project):
		mock_proc = MagicMock()
		mock_proc.pid = 12345
		mock_popen.return_value = mock_proc

		resp = client_with_project.post(
			"/launch/execute",
			data={"project": "test", "max_rounds": "10"},
		)
		assert resp.status_code == 200


class TestLegacyCompat:
	def test_legacy_mission_header(self, client):
		resp = client.get("/partials/mission-header")
		assert resp.status_code == 200

	def test_legacy_score_history(self, client):
		resp = client.get("/api/score-history")
		assert resp.status_code == 200
		data = resp.json()
		assert "labels" in data
