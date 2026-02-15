"""Tests for the live mission control dashboard."""

from __future__ import annotations

import json
import sqlite3

from fastapi.testclient import TestClient

from mission_control.dashboard.live import LiveDashboard, _serialize_event, _serialize_mission, _serialize_unit
from mission_control.db import Database
from mission_control.models import Epoch, Mission, Plan, UnitEvent, WorkUnit


def _setup_db(*, check_same_thread: bool = True) -> Database:
	"""Create an in-memory DB with test data."""
	db = Database.__new__(Database)
	import asyncio
	db.conn = sqlite3.connect(":memory:", check_same_thread=check_same_thread)
	db.conn.row_factory = sqlite3.Row
	db.conn.execute("PRAGMA foreign_keys=ON")
	db._lock = asyncio.Lock()
	db._create_tables()

	mission = Mission(id="m1", objective="Improve tests", status="running")
	db.insert_mission(mission)

	epoch = Epoch(id="ep1", mission_id="m1", number=1, units_planned=3)
	db.insert_epoch(epoch)

	plan = Plan(id="p1", objective="Improve tests")
	db.insert_plan(plan)

	units = [
		WorkUnit(
			id="wu1", plan_id="p1", title="Add unit tests",
			status="running", epoch_id="ep1", attempt=0,
			files_hint="tests/test_foo.py",
		),
		WorkUnit(
			id="wu2", plan_id="p1", title="Fix linting",
			status="completed", epoch_id="ep1", attempt=1,
			output_summary="Fixed 3 lint errors",
		),
		WorkUnit(
			id="wu3", plan_id="p1", title="Add type hints",
			status="failed", epoch_id="ep1", attempt=3, max_attempts=3,
			output_summary="Timeout after 300s",
		),
	]
	for u in units:
		db.insert_work_unit(u)

	events = [
		UnitEvent(
			id="ev1", mission_id="m1", epoch_id="ep1",
			work_unit_id="wu1", event_type="dispatched",
		),
		UnitEvent(
			id="ev2", mission_id="m1", epoch_id="ep1",
			work_unit_id="wu2", event_type="merged",
		),
		UnitEvent(
			id="ev3", mission_id="m1", epoch_id="ep1",
			work_unit_id="wu3", event_type="failed",
		),
	]
	for e in events:
		db.insert_unit_event(e)

	return db


class TestSnapshotBuilding:
	def test_build_snapshot_with_data(self) -> None:
		db = _setup_db()
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db

		snapshot = dashboard._build_snapshot()

		assert snapshot["mission"] is not None
		assert snapshot["mission"]["id"] == "m1"
		assert snapshot["mission"]["status"] == "running"
		assert len(snapshot["units"]) == 3
		assert len(snapshot["events"]) == 3
		assert len(snapshot["plan_tree"]) == 1  # 1 epoch

	def test_build_snapshot_no_mission(self) -> None:
		db = Database(":memory:")
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db

		snapshot = dashboard._build_snapshot()
		assert snapshot["mission"] is None
		assert snapshot["units"] == []

	def test_plan_tree_structure(self) -> None:
		db = _setup_db()
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db

		tree = dashboard._build_plan_tree("m1")
		assert len(tree) == 1
		epoch_node = tree[0]
		assert epoch_node["type"] == "epoch"
		assert epoch_node["number"] == 1
		assert len(epoch_node["children"]) == 3

		# Check children have correct statuses
		statuses = {c["id"]: c["status"] for c in epoch_node["children"]}
		assert statuses["wu1"] == "running"
		assert statuses["wu2"] == "completed"
		assert statuses["wu3"] == "failed"


class TestSerialization:
	def test_serialize_mission(self) -> None:
		m = Mission(id="m1", objective="Test", status="running", total_cost_usd=5.5)
		result = _serialize_mission(m)
		assert result["id"] == "m1"
		assert result["objective"] == "Test"
		assert result["total_cost_usd"] == 5.5

	def test_serialize_unit(self) -> None:
		u = WorkUnit(
			id="wu1", title="Task", status="running",
			description="A" * 300, output_summary="B" * 300,
		)
		result = _serialize_unit(u)
		assert result["id"] == "wu1"
		assert len(result["description"]) <= 200
		assert len(result["output_summary"]) <= 200

	def test_serialize_event(self) -> None:
		e = UnitEvent(
			id="ev1", mission_id="m1", epoch_id="ep1",
			work_unit_id="wu1", event_type="merged",
		)
		result = _serialize_event(e)
		assert result["event_type"] == "merged"
		assert result["work_unit_id"] == "wu1"


class TestRESTEndpoints:
	def _make_client(self) -> TestClient:
		db = _setup_db(check_same_thread=False)
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db
		dashboard._connections = set()
		dashboard._broadcast_task = None
		from fastapi import FastAPI
		dashboard.app = FastAPI()
		dashboard._setup_routes()
		return TestClient(dashboard.app)

	def test_get_mission(self) -> None:
		client = self._make_client()
		resp = client.get("/api/mission")
		assert resp.status_code == 200
		data = resp.json()
		assert data["id"] == "m1"
		assert data["status"] == "running"

	def test_get_units(self) -> None:
		client = self._make_client()
		resp = client.get("/api/units")
		assert resp.status_code == 200
		data = resp.json()
		assert len(data) == 3

	def test_get_events(self) -> None:
		client = self._make_client()
		resp = client.get("/api/events")
		assert resp.status_code == 200
		data = resp.json()
		assert len(data) == 3

	def test_get_plan_tree(self) -> None:
		client = self._make_client()
		resp = client.get("/api/plan-tree")
		assert resp.status_code == 200
		data = resp.json()
		assert len(data) == 1
		assert data[0]["type"] == "epoch"

	def test_send_signal(self) -> None:
		client = self._make_client()
		resp = client.post("/api/signal", json={
			"signal_type": "pause",
			"payload": {},
		})
		assert resp.status_code == 200
		data = resp.json()
		assert data["status"] == "ok"
		assert "signal_id" in data

	def test_send_signal_with_payload(self) -> None:
		client = self._make_client()
		resp = client.post("/api/signal", json={
			"signal_type": "cancel_unit",
			"payload": {"unit_id": "wu1"},
		})
		assert resp.status_code == 200
		data = resp.json()
		assert data["status"] == "ok"

	def test_index_returns_html(self) -> None:
		client = self._make_client()
		resp = client.get("/")
		assert resp.status_code == 200
		assert "Mission Control" in resp.text


class TestSignalInsertion:
	def test_signal_inserted_into_db(self) -> None:
		db = _setup_db(check_same_thread=False)
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db
		dashboard._connections = set()
		dashboard._broadcast_task = None
		from fastapi import FastAPI
		dashboard.app = FastAPI()
		dashboard._setup_routes()

		client = TestClient(dashboard.app)
		client.post("/api/signal", json={
			"signal_type": "stop",
			"payload": {},
		})

		signals = db.get_pending_signals("m1")
		assert len(signals) == 1
		assert signals[0].signal_type == "stop"

	def test_cancel_unit_signal_has_payload(self) -> None:
		db = _setup_db(check_same_thread=False)
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db
		dashboard._connections = set()
		dashboard._broadcast_task = None
		from fastapi import FastAPI
		dashboard.app = FastAPI()
		dashboard._setup_routes()

		client = TestClient(dashboard.app)
		client.post("/api/signal", json={
			"signal_type": "cancel_unit",
			"payload": {"unit_id": "wu1"},
		})

		signals = db.get_pending_signals("m1")
		assert len(signals) == 1
		payload = json.loads(signals[0].payload)
		assert payload["unit_id"] == "wu1"


class TestDBQueries:
	def test_get_active_mission(self) -> None:
		db = _setup_db()
		mission = db.get_active_mission()
		assert mission is not None
		assert mission.id == "m1"
		assert mission.status == "running"

	def test_get_active_mission_none(self) -> None:
		db = Database(":memory:")
		assert db.get_active_mission() is None

	def test_get_work_units_for_mission(self) -> None:
		db = _setup_db()
		units = db.get_work_units_for_mission("m1")
		assert len(units) == 3

	def test_get_work_units_for_nonexistent_mission(self) -> None:
		db = _setup_db()
		units = db.get_work_units_for_mission("nonexistent")
		assert units == []
