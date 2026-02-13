"""Tests for observability dashboard endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from mission_control.db import Database
from mission_control.models import (
	Handoff,
	Mission,
	Plan,
	PlanNode,
	Reflection,
	Round,
	WorkUnit,
)
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


def _populate_db(db_path):
	"""Create a DB with mission, rounds, handoffs, reflections, and plan nodes."""
	db = Database(db_path)

	mission = Mission(id="m1", objective="build feature", status="running", total_cost_usd=2.0)
	db.insert_mission(mission)

	plan1 = Plan(id="plan1", objective="round 1 plan")
	plan2 = Plan(id="plan2", objective="round 2 plan")
	db.insert_plan(plan1)
	db.insert_plan(plan2)

	r1 = Round(
		id="r1", mission_id="m1", number=1, status="completed",
		plan_id="plan1", objective_score=40.0, cost_usd=0.80,
		total_units=3, completed_units=2, failed_units=1,
	)
	r2 = Round(
		id="r2", mission_id="m1", number=2, status="executing",
		plan_id="plan2", objective_score=75.0, cost_usd=1.20,
		total_units=4, completed_units=1, failed_units=0,
	)
	db.insert_round(r1)
	db.insert_round(r2)

	# Work units (needed for handoff FK constraints)
	wu1 = WorkUnit(id="wu1", plan_id="plan1", title="Auth module", status="completed")
	wu2 = WorkUnit(id="wu2", plan_id="plan2", title="API endpoint", status="completed")
	db.insert_work_unit(wu1)
	db.insert_work_unit(wu2)

	# Plan nodes for plan2
	root = PlanNode(
		id="pn1", plan_id="plan2", depth=0,
		scope="Implement feature X", strategy="subdivide",
		node_type="branch", children_ids="pn2,pn3",
	)
	child1 = PlanNode(
		id="pn2", plan_id="plan2", parent_id="pn1", depth=1,
		scope="Backend API", strategy="leaves", node_type="leaf",
	)
	child2 = PlanNode(
		id="pn3", plan_id="plan2", parent_id="pn1", depth=1,
		scope="Frontend UI", strategy="leaves", node_type="leaf",
	)
	db.insert_plan_node(root)
	db.insert_plan_node(child1)
	db.insert_plan_node(child2)

	# Handoffs for round 1
	h1 = Handoff(
		id="h1", work_unit_id="wu1", round_id="r1", status="completed",
		summary="Implemented auth module",
		discoveries=json.dumps(["Found unused config file", "API rate limit is 100/min"]),
		concerns=json.dumps(["Test coverage is low"]),
	)
	db.insert_handoff(h1)

	# Handoffs for round 2
	h2 = Handoff(
		id="h2", work_unit_id="wu2", round_id="r2", status="completed",
		summary="Built the API endpoint",
		discoveries=json.dumps(["DB schema needs migration"]),
		concerns=json.dumps([]),
	)
	db.insert_handoff(h2)

	# Reflection for round 1
	ref1 = Reflection(
		id="ref1", mission_id="m1", round_id="r1", round_number=1,
		tests_before=80, tests_after=85, tests_delta=5,
		lint_delta=-2, objective_score=40.0, score_delta=0.0,
		units_planned=3, units_completed=2, units_failed=1,
		completion_rate=66.7, plan_depth=1, plan_strategy="leaves",
		fixup_promoted=True, fixup_attempts=1, merge_conflicts=0,
		discoveries_count=2,
	)
	db.insert_reflection(ref1)

	# Reflection for round 2
	ref2 = Reflection(
		id="ref2", mission_id="m1", round_id="r2", round_number=2,
		tests_before=85, tests_after=92, tests_delta=7,
		lint_delta=-1, objective_score=75.0, score_delta=35.0,
		units_planned=4, units_completed=1, units_failed=0,
		completion_rate=25.0, plan_depth=2, plan_strategy="subdivide",
		fixup_promoted=False, fixup_attempts=2, merge_conflicts=1,
		discoveries_count=1,
	)
	db.insert_reflection(ref2)

	db.close()


@pytest.fixture
def project_with_observability_data(registry, sample_config, tmp_path):
	"""Register a project with rich observability data."""
	db_path = tmp_path / "mission-control.db"
	registry.register(name="test", config_path=str(sample_config), db_path=str(db_path))
	_populate_db(db_path)
	return {"name": "test", "db_path": db_path, "tmp_path": tmp_path}


@pytest.fixture
def client_with_data(registry, project_with_observability_data):
	from mission_control.dashboard.web.server import create_app
	app = create_app(registry=registry)
	with TestClient(app) as c:
		yield c


@pytest.fixture
def client_empty(registry):
	from mission_control.dashboard.web.server import create_app
	app = create_app(registry=registry)
	with TestClient(app) as c:
		yield c


class TestLiveStatus:
	def test_live_status_returns_html(self, client_with_data):
		resp = client_with_data.get("/project/test/partials/live-status")
		assert resp.status_code == 200
		assert "live-status" in resp.text

	def test_live_status_discovery_via_helper(self, project_with_observability_data, registry):
		"""Test the discovery extraction directly (avoids provider polling race)."""
		from mission_control.dashboard.web.server import create_app
		app = create_app(registry=registry)
		with TestClient(app):
			from mission_control.dashboard.web.server import _get_latest_discovery
			discovery = _get_latest_discovery("test", "r2")
			assert discovery == "DB schema needs migration"

	def test_live_status_empty(self, client_empty):
		resp = client_empty.get("/project/nonexistent/partials/live-status")
		assert resp.status_code == 200
		assert "No active round" in resp.text


class TestRoundTimeline:
	def test_round_timeline_with_data(self, client_with_data):
		resp = client_with_data.get("/project/test/partials/round-timeline")
		assert resp.status_code == 200
		# Should show both rounds (newest first)
		assert "R2" in resp.text
		assert "R1" in resp.text

	def test_round_timeline_shows_scores(self, client_with_data):
		resp = client_with_data.get("/project/test/partials/round-timeline")
		assert resp.status_code == 200
		assert "75.0" in resp.text  # R2 score
		assert "40.0" in resp.text  # R1 score

	def test_round_timeline_empty(self, client_empty):
		resp = client_empty.get("/project/nonexistent/partials/round-timeline")
		assert resp.status_code == 200
		assert "No rounds yet" in resp.text


class TestRoundDetail:
	def test_round_detail_with_reflection(self, client_with_data):
		resp = client_with_data.get("/project/test/partials/round-detail/r1")
		assert resp.status_code == 200
		# Reflection metrics
		assert "80" in resp.text  # tests_before
		assert "85" in resp.text  # tests_after

	def test_round_detail_plan_nodes(self, client_with_data):
		resp = client_with_data.get("/project/test/partials/round-detail/r2")
		assert resp.status_code == 200
		# Plan structure from plan2
		assert "Implement feature X" in resp.text
		assert "Backend API" in resp.text
		assert "Frontend UI" in resp.text

	def test_round_detail_handoffs(self, client_with_data):
		resp = client_with_data.get("/project/test/partials/round-detail/r1")
		assert resp.status_code == 200
		# Handoff summary and discoveries
		assert "Implemented auth module" in resp.text
		assert "Found unused config file" in resp.text
		assert "Test coverage is low" in resp.text

	def test_round_detail_nonexistent(self, client_with_data):
		resp = client_with_data.get("/project/test/partials/round-detail/nope")
		assert resp.status_code == 200
		assert "No reflection data" in resp.text


class TestDiscoveriesFeed:
	def test_discoveries_feed_with_data(self, client_with_data):
		resp = client_with_data.get("/project/test/partials/discoveries-feed")
		assert resp.status_code == 200
		# Should contain discoveries from both rounds
		assert "Found unused config file" in resp.text
		assert "DB schema needs migration" in resp.text

	def test_discoveries_feed_shows_concerns(self, client_with_data):
		resp = client_with_data.get("/project/test/partials/discoveries-feed")
		assert resp.status_code == 200
		assert "Test coverage is low" in resp.text

	def test_discoveries_feed_empty(self, client_empty):
		resp = client_empty.get("/project/nonexistent/partials/discoveries-feed")
		assert resp.status_code == 200
		assert "No discoveries yet" in resp.text


class TestLogTail:
	def test_log_tail_with_log_file(self, client_with_data, project_with_observability_data):
		# Create a log file
		logs_dir = project_with_observability_data["tmp_path"] / "logs"
		logs_dir.mkdir()
		log_file = logs_dir / "mission-2025-06-01.log"
		log_file.write_text("Line 1: Starting mission\nLine 2: Round 1 started\nLine 3: Done\n")

		resp = client_with_data.get("/project/test/partials/log-tail")
		assert resp.status_code == 200
		assert "Starting mission" in resp.text
		assert "Round 1 started" in resp.text

	def test_log_tail_no_log(self, client_with_data):
		resp = client_with_data.get("/project/test/partials/log-tail")
		assert resp.status_code == 200
		assert "No log file found" in resp.text

	def test_log_tail_truncates_to_100_lines(self, client_with_data, project_with_observability_data):
		logs_dir = project_with_observability_data["tmp_path"] / "logs"
		logs_dir.mkdir(exist_ok=True)
		log_file = logs_dir / "mission-2025-06-01.log"
		lines = [f"Line {i}: log entry" for i in range(200)]
		log_file.write_text("\n".join(lines))

		resp = client_with_data.get("/project/test/partials/log-tail")
		assert resp.status_code == 200
		# Should have the last 100 lines (100-199)
		assert "Line 100" in resp.text
		assert "Line 199" in resp.text
		# Should NOT have the first lines
		assert "Line 0:" not in resp.text


class TestHelperFunctions:
	def test_get_all_rounds(self, project_with_observability_data, registry):
		"""Direct test of _get_all_rounds helper."""
		# Need to init state for the helper to work
		from mission_control.dashboard.web.server import create_app
		app = create_app(registry=registry)
		with TestClient(app):
			from mission_control.dashboard.web.server import _get_all_rounds
			rounds = _get_all_rounds("test")
			assert len(rounds) == 2
			assert rounds[0].number == 1
			assert rounds[1].number == 2

	def test_get_latest_discovery(self, project_with_observability_data, registry):
		"""Direct test of _get_latest_discovery helper."""
		from mission_control.dashboard.web.server import create_app
		app = create_app(registry=registry)
		with TestClient(app):
			from mission_control.dashboard.web.server import _get_latest_discovery
			discovery = _get_latest_discovery("test", "r2")
			assert discovery == "DB schema needs migration"

	def test_get_all_discoveries(self, project_with_observability_data, registry):
		"""Direct test of _get_all_discoveries helper."""
		from mission_control.dashboard.web.server import create_app
		app = create_app(registry=registry)
		with TestClient(app):
			from mission_control.dashboard.web.server import _get_all_discoveries
			items = _get_all_discoveries("test")
			# Should have discoveries + concerns from both rounds
			assert len(items) == 4
			# Newest first
			assert items[0]["round_number"] == 2
			types = {i["type"] for i in items}
			assert "discovery" in types
			assert "concern" in types

	def test_get_round_detail(self, project_with_observability_data, registry):
		"""Direct test of _get_round_detail helper."""
		from mission_control.dashboard.web.server import create_app
		app = create_app(registry=registry)
		with TestClient(app):
			from mission_control.dashboard.web.server import _get_round_detail
			reflection, plan_nodes, handoffs = _get_round_detail("test", "r2")
			assert reflection is not None
			assert reflection.round_number == 2
			assert len(plan_nodes) == 3  # root + 2 children
			assert len(handoffs) == 1
			assert hasattr(handoffs[0], "_discoveries")
			assert handoffs[0]._discoveries == ["DB schema needs migration"]
