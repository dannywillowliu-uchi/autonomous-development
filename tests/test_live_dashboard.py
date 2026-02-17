"""Tests for the live mission control dashboard."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any

from fastapi.testclient import TestClient

from mission_control.dashboard.live import (
	VALID_SIGNAL_TYPES,
	LiveDashboard,
	_serialize_event,
	_serialize_mission,
	_serialize_unit,
	_serialize_worker,
)
from mission_control.db import Database
from mission_control.models import Epoch, Mission, Plan, UnitEvent, Worker, WorkUnit


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

	def test_build_snapshot_includes_summary(self) -> None:
		db = _setup_db()
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db

		snapshot = dashboard._build_snapshot()
		assert "summary" in snapshot
		summary = snapshot["summary"]
		assert summary["units_merged"] == 1  # wu2 completed
		assert summary["units_failed"] == 1  # wu3 failed
		assert summary["units_running"] == 1  # wu1 running
		assert summary["current_epoch"] == 1

	def test_build_snapshot_no_mission(self) -> None:
		db = Database(":memory:")
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db

		snapshot = dashboard._build_snapshot()
		assert snapshot["mission"] is None
		assert snapshot["units"] == []
		assert snapshot["summary"] is None

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
	def _make_client(self, auth_token: str = "") -> TestClient:
		db = _setup_db(check_same_thread=False)
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db
		dashboard.auth_token = auth_token
		dashboard._connections = set()
		dashboard._broadcast_task = None
		dashboard._signal_timestamps = defaultdict(list)
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

	def test_get_summary(self) -> None:
		client = self._make_client()
		resp = client.get("/api/summary")
		assert resp.status_code == 200
		data = resp.json()
		assert "units_merged" in data
		assert "units_failed" in data
		assert "current_epoch" in data

	def test_get_summary_no_mission(self) -> None:
		db = Database.__new__(Database)
		import asyncio
		db.conn = sqlite3.connect(":memory:", check_same_thread=False)
		db.conn.row_factory = sqlite3.Row
		db.conn.execute("PRAGMA foreign_keys=ON")
		db._lock = asyncio.Lock()
		db._create_tables()

		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db
		dashboard.auth_token = ""
		dashboard._connections = set()
		dashboard._broadcast_task = None
		dashboard._signal_timestamps = defaultdict(list)
		from fastapi import FastAPI
		dashboard.app = FastAPI()
		dashboard._setup_routes()

		client = TestClient(dashboard.app)
		resp = client.get("/api/summary")
		assert resp.status_code == 200
		data = resp.json()
		assert data["status"] == "no_mission"

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
		dashboard.auth_token = ""
		dashboard._connections = set()
		dashboard._broadcast_task = None
		dashboard._signal_timestamps = defaultdict(list)
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
		dashboard.auth_token = ""
		dashboard._connections = set()
		dashboard._broadcast_task = None
		dashboard._signal_timestamps = defaultdict(list)
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


class TestHotReload:
	def test_snapshot_does_not_include_reload(self) -> None:
		"""_build_snapshot() itself never sets reload -- that's done by the broadcast loop."""
		db = _setup_db()
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db

		snapshot = dashboard._build_snapshot()
		assert "reload" not in snapshot

	def test_ui_mtime_initialized(self, tmp_path: object) -> None:
		"""__init__ records the UI file's mtime for change detection."""
		import tempfile
		from pathlib import Path
		with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
			f.write(b"<html></html>")
			f.flush()
			import mission_control.dashboard.live as live_mod
			original_path = live_mod._UI_PATH
			try:
				live_mod._UI_PATH = Path(f.name)
				db = _setup_db(check_same_thread=False)
				dashboard = LiveDashboard.__new__(LiveDashboard)
				dashboard.db = db
				dashboard._ui_mtime = live_mod._UI_PATH.stat().st_mtime
				assert dashboard._ui_mtime > 0
			finally:
				live_mod._UI_PATH = original_path
				Path(f.name).unlink(missing_ok=True)


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


class TestWorkerVisibility:
	def _setup_with_workers(self, *, check_same_thread: bool = True) -> Database:
		db = _setup_db(check_same_thread=check_same_thread)
		w1 = Worker(
			id="wu1", workspace_path="/tmp/ws/wu1", status="working",
			current_unit_id="wu1", pid=1234, backend_type="local",
			backend_metadata='{"output_excerpt": "Running tests..."}',
		)
		w2 = Worker(
			id="wu2", workspace_path="/tmp/ws/wu2", status="idle",
			units_completed=1, backend_type="local",
		)
		db.insert_worker(w1)
		db.insert_worker(w2)
		return db

	def test_snapshot_includes_workers(self) -> None:
		db = self._setup_with_workers()
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db

		snapshot = dashboard._build_snapshot()
		assert "workers" in snapshot
		assert len(snapshot["workers"]) == 2

	def test_snapshot_no_mission_has_empty_workers(self) -> None:
		db = Database(":memory:")
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db

		snapshot = dashboard._build_snapshot()
		assert snapshot["workers"] == []

	def test_serialize_worker_extracts_output_excerpt(self) -> None:
		w = Worker(
			id="w1", workspace_path="/tmp/ws",
			backend_metadata='{"output_excerpt": "hello world"}',
		)
		result = _serialize_worker(w)
		assert result["output_excerpt"] == "hello world"
		assert result["workspace_path"] == "/tmp/ws"

	def test_serialize_worker_empty_metadata(self) -> None:
		w = Worker(id="w1")
		result = _serialize_worker(w)
		assert result["output_excerpt"] == ""

	def test_serialize_worker_invalid_json_metadata(self) -> None:
		w = Worker(id="w1", backend_metadata="not-json")
		result = _serialize_worker(w)
		assert result["output_excerpt"] == ""

	def test_api_workers_endpoint(self) -> None:
		db = self._setup_with_workers(check_same_thread=False)
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db
		dashboard.auth_token = ""
		dashboard._connections = set()
		dashboard._broadcast_task = None
		dashboard._signal_timestamps = defaultdict(list)
		from fastapi import FastAPI
		dashboard.app = FastAPI()
		dashboard._setup_routes()

		client = TestClient(dashboard.app)
		resp = client.get("/api/workers")
		assert resp.status_code == 200
		data = resp.json()
		assert len(data) == 2
		working = [w for w in data if w["status"] == "working"]
		assert len(working) == 1
		assert working[0]["pid"] == 1234
		assert working[0]["output_excerpt"] == "Running tests..."


def _make_authed_client(auth_token: str = "test-secret") -> TestClient:
	"""Create a test client with auth enabled."""
	db = _setup_db(check_same_thread=False)
	dashboard = LiveDashboard.__new__(LiveDashboard)
	dashboard.db = db
	dashboard.auth_token = auth_token
	dashboard._connections = set()
	dashboard._broadcast_task = None
	dashboard._signal_timestamps = defaultdict(list)
	from fastapi import FastAPI
	dashboard.app = FastAPI()
	dashboard._setup_routes()
	return TestClient(dashboard.app)


class TestBearerTokenAuth:
	def test_missing_token_returns_401(self) -> None:
		client = _make_authed_client()
		resp = client.get("/api/mission")
		assert resp.status_code == 401

	def test_wrong_token_returns_401(self) -> None:
		client = _make_authed_client()
		resp = client.get("/api/mission", headers={"Authorization": "Bearer wrong-token"})
		assert resp.status_code == 401

	def test_correct_token_returns_200(self) -> None:
		client = _make_authed_client("my-token")
		resp = client.get("/api/mission", headers={"Authorization": "Bearer my-token"})
		assert resp.status_code == 200

	def test_auth_on_all_api_endpoints(self) -> None:
		client = _make_authed_client()
		paths = [
			"/api/mission", "/api/units", "/api/events",
			"/api/plan-tree", "/api/workers", "/api/summary", "/api/history",
		]
		for path in paths:
			resp = client.get(path)
			assert resp.status_code == 401, f"{path} should require auth"

	def test_signal_endpoint_requires_auth(self) -> None:
		client = _make_authed_client()
		resp = client.post("/api/signal", json={"signal_type": "stop", "payload": {}})
		assert resp.status_code == 401

	def test_index_does_not_require_auth(self) -> None:
		client = _make_authed_client()
		resp = client.get("/")
		assert resp.status_code == 200

	def test_no_token_mode_allows_all(self) -> None:
		client = _make_authed_client(auth_token="")
		resp = client.get("/api/mission")
		assert resp.status_code == 200


class TestWebSocketAuth:
	def test_ws_rejects_missing_token(self) -> None:
		from starlette.websockets import WebSocketDisconnect as StarletteWSDisconnect
		client = _make_authed_client()
		try:
			with client.websocket_connect("/ws"):
				pass
			assert False, "Expected WebSocketDisconnect"
		except StarletteWSDisconnect as e:
			assert e.code == 4401

	def test_ws_rejects_wrong_token(self) -> None:
		from starlette.websockets import WebSocketDisconnect as StarletteWSDisconnect
		client = _make_authed_client("secret")
		try:
			with client.websocket_connect("/ws?token=wrong"):
				pass
			assert False, "Expected WebSocketDisconnect"
		except StarletteWSDisconnect as e:
			assert e.code == 4401

	def test_ws_accepts_correct_token(self) -> None:
		client = _make_authed_client("secret")
		with client.websocket_connect("/ws?token=secret") as ws:
			data = ws.receive_json()
			assert "mission" in data


class TestSignalValidation:
	def test_unknown_signal_type_returns_422(self) -> None:
		client = _make_authed_client("")
		resp = client.post("/api/signal", json={"signal_type": "nuke_everything", "payload": {}})
		assert resp.status_code == 422
		assert "Unknown signal_type" in resp.json()["detail"]

	def test_all_valid_signal_types_accepted(self) -> None:
		for sig_type in VALID_SIGNAL_TYPES:
			client = _make_authed_client("")
			payload: dict[str, Any] = {}
			if sig_type == "adjust_workers":
				payload = {"num_workers": 2}
			elif sig_type == "cancel_unit":
				payload = {"unit_id": "wu1"}
			resp = client.post("/api/signal", json={"signal_type": sig_type, "payload": payload})
			assert resp.status_code == 200, f"{sig_type} should be accepted"

	def test_adjust_workers_requires_num_workers(self) -> None:
		client = _make_authed_client("")
		resp = client.post("/api/signal", json={"signal_type": "adjust_workers", "payload": {}})
		assert resp.status_code == 422
		assert "num_workers" in resp.json()["detail"]

	def test_cancel_unit_requires_unit_id(self) -> None:
		client = _make_authed_client("")
		resp = client.post("/api/signal", json={"signal_type": "cancel_unit", "payload": {}})
		assert resp.status_code == 422
		assert "unit_id" in resp.json()["detail"]

	def test_adjust_workers_maps_to_adjust_in_db(self) -> None:
		db = _setup_db(check_same_thread=False)
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db
		dashboard.auth_token = ""
		dashboard._connections = set()
		dashboard._broadcast_task = None
		dashboard._signal_timestamps = defaultdict(list)
		from fastapi import FastAPI
		dashboard.app = FastAPI()
		dashboard._setup_routes()

		client = TestClient(dashboard.app)
		client.post("/api/signal", json={"signal_type": "adjust_workers", "payload": {"num_workers": 3}})

		signals = db.get_pending_signals("m1")
		assert len(signals) == 1
		assert signals[0].signal_type == "adjust"


class TestRateLimiting:
	def test_rate_limit_exceeded(self) -> None:
		client = _make_authed_client("")
		for i in range(10):
			resp = client.post("/api/signal", json={"signal_type": "pause", "payload": {}})
			assert resp.status_code == 200, f"Request {i+1} should succeed"

		resp = client.post("/api/signal", json={"signal_type": "pause", "payload": {}})
		assert resp.status_code == 429
		assert "Rate limit" in resp.json()["detail"]


def _setup_db_with_history(*, check_same_thread: bool = True) -> Database:
	"""Create an in-memory DB with current + completed missions for history tests."""
	db = Database.__new__(Database)
	import asyncio
	db.conn = sqlite3.connect(":memory:", check_same_thread=check_same_thread)
	db.conn.row_factory = sqlite3.Row
	db.conn.execute("PRAGMA foreign_keys=ON")
	db._lock = asyncio.Lock()
	db._create_tables()

	# Completed mission with timestamps for duration
	m_old = Mission(
		id="m-old", objective="Old mission", status="completed",
		started_at="2025-01-01T00:00:00", finished_at="2025-01-01T01:30:00",
		total_cost_usd=2.5, stopped_reason="objective_met",
	)
	db.insert_mission(m_old)
	epoch_old = Epoch(id="ep-old", mission_id="m-old", number=1, units_planned=2, units_completed=2)
	db.insert_epoch(epoch_old)
	plan_old = Plan(id="p-old", objective="Old mission")
	db.insert_plan(plan_old)
	for uid in ("wu-old1", "wu-old2"):
		db.insert_work_unit(WorkUnit(
			id=uid, plan_id="p-old", title=f"Task {uid}",
			status="completed", epoch_id="ep-old",
		))

	# Currently running mission
	m_cur = Mission(id="m-cur", objective="Current mission", status="running", total_cost_usd=1.0)
	db.insert_mission(m_cur)
	epoch_cur = Epoch(id="ep-cur", mission_id="m-cur", number=1, units_planned=1)
	db.insert_epoch(epoch_cur)
	plan_cur = Plan(id="p-cur", objective="Current mission")
	db.insert_plan(plan_cur)
	db.insert_work_unit(WorkUnit(
		id="wu-cur1", plan_id="p-cur", title="Running task",
		status="running", epoch_id="ep-cur",
	))

	return db


class TestHistoryEndpoint:
	def _make_client(self) -> TestClient:
		db = _setup_db_with_history(check_same_thread=False)
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db
		dashboard.auth_token = ""
		dashboard._connections = set()
		dashboard._broadcast_task = None
		dashboard._signal_timestamps = defaultdict(list)
		from fastapi import FastAPI
		dashboard.app = FastAPI()
		dashboard._setup_routes()
		return TestClient(dashboard.app)

	def test_history_returns_all_missions(self) -> None:
		client = self._make_client()
		resp = client.get("/api/history")
		assert resp.status_code == 200
		data = resp.json()
		assert len(data) == 2
		ids = {m["id"] for m in data}
		assert ids == {"m-old", "m-cur"}

	def test_history_entry_fields(self) -> None:
		client = self._make_client()
		resp = client.get("/api/history")
		data = resp.json()
		expected_keys = {
			"id", "objective", "status", "started_at", "finished_at",
			"duration", "units_merged", "units_failed", "total_cost_usd",
			"stopped_reason", "trajectory_rating",
		}
		for entry in data:
			assert set(entry.keys()) == expected_keys

	def test_history_completed_mission_has_duration(self) -> None:
		client = self._make_client()
		resp = client.get("/api/history")
		data = resp.json()
		completed = [m for m in data if m["id"] == "m-old"][0]
		assert completed["duration"] == 5400.0  # 1.5 hours
		assert completed["status"] == "completed"
		assert completed["units_merged"] == 2
		assert completed["total_cost_usd"] == 2.5
		assert completed["stopped_reason"] == "objective_met"

	def test_history_running_mission_has_null_duration(self) -> None:
		client = self._make_client()
		resp = client.get("/api/history")
		data = resp.json()
		running = [m for m in data if m["id"] == "m-cur"][0]
		assert running["duration"] is None
		assert running["status"] == "running"

	def test_history_empty_when_no_missions(self) -> None:
		db = Database.__new__(Database)
		import asyncio
		db.conn = sqlite3.connect(":memory:", check_same_thread=False)
		db.conn.row_factory = sqlite3.Row
		db.conn.execute("PRAGMA foreign_keys=ON")
		db._lock = asyncio.Lock()
		db._create_tables()

		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db
		dashboard.auth_token = ""
		dashboard._connections = set()
		dashboard._broadcast_task = None
		dashboard._signal_timestamps = defaultdict(list)
		from fastapi import FastAPI
		dashboard.app = FastAPI()
		dashboard._setup_routes()

		client = TestClient(dashboard.app)
		resp = client.get("/api/history")
		assert resp.status_code == 200
		assert resp.json() == []


class TestSnapshotHistory:
	def test_snapshot_includes_history(self) -> None:
		db = _setup_db_with_history()
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db

		snapshot = dashboard._build_snapshot()
		assert "history" in snapshot
		assert len(snapshot["history"]) == 2

	def test_snapshot_no_mission_includes_empty_history(self) -> None:
		db = Database(":memory:")
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db

		snapshot = dashboard._build_snapshot()
		assert "history" in snapshot
		assert snapshot["history"] == []

	def test_snapshot_history_matches_api(self) -> None:
		db = _setup_db_with_history(check_same_thread=False)
		dashboard = LiveDashboard.__new__(LiveDashboard)
		dashboard.db = db
		dashboard.auth_token = ""
		dashboard._connections = set()
		dashboard._broadcast_task = None
		dashboard._signal_timestamps = defaultdict(list)
		from fastapi import FastAPI
		dashboard.app = FastAPI()
		dashboard._setup_routes()

		snapshot = dashboard._build_snapshot()
		client = TestClient(dashboard.app)
		resp = client.get("/api/history")

		assert snapshot["history"] == resp.json()
