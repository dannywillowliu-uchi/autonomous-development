"""Live mission control dashboard -- real-time WebSocket + REST API."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from mission_control.db import Database
from mission_control.models import Signal

logger = logging.getLogger(__name__)

_UI_PATH = Path(__file__).parent / "live_ui.html"


class SignalRequest(BaseModel):
	"""Request body for sending a signal to the controller."""

	signal_type: str
	payload: dict[str, Any] = {}


class LiveDashboard:
	"""Real-time mission control dashboard.

	Serves a WebSocket endpoint that pushes mission snapshots every second,
	plus REST endpoints for sending control signals and querying state.
	"""

	def __init__(self, db_path: str) -> None:
		self.db = Database(db_path)
		self.app = FastAPI(title="Mission Control Live")
		self._connections: set[WebSocket] = set()
		self._broadcast_task: asyncio.Task[None] | None = None
		self._ui_mtime: float = _UI_PATH.stat().st_mtime if _UI_PATH.exists() else 0.0
		self._setup_routes()

	def _setup_routes(self) -> None:
		@self.app.on_event("startup")
		async def startup() -> None:
			self._broadcast_task = asyncio.create_task(self._broadcast_loop())

		@self.app.on_event("shutdown")
		async def shutdown() -> None:
			if self._broadcast_task:
				self._broadcast_task.cancel()
				try:
					await self._broadcast_task
				except asyncio.CancelledError:
					pass
			self.db.close()

		@self.app.websocket("/ws")
		async def ws_endpoint(websocket: WebSocket) -> None:
			await websocket.accept()
			self._connections.add(websocket)
			try:
				# Send initial snapshot immediately
				snapshot = self._build_snapshot()
				await websocket.send_json(snapshot)
				# Keep connection alive, listen for client messages
				while True:
					await websocket.receive_text()
			except WebSocketDisconnect:
				pass
			finally:
				self._connections.discard(websocket)

		@self.app.get("/", response_class=HTMLResponse)
		async def index() -> HTMLResponse:
			if _UI_PATH.exists():
				return HTMLResponse(_UI_PATH.read_text())
			return HTMLResponse("<h1>Mission Control Live</h1><p>UI file not found.</p>")

		@self.app.post("/api/signal")
		async def send_signal(body: SignalRequest) -> dict[str, str]:
			mission = self.db.get_active_mission() or self.db.get_latest_mission()
			if mission is None:
				return {"status": "error", "message": "No mission found"}

			signal = Signal(
				mission_id=mission.id,
				signal_type=body.signal_type,
				payload=json.dumps(body.payload) if body.payload else "",
			)
			self.db.insert_signal(signal)
			return {"status": "ok", "signal_id": signal.id}

		@self.app.get("/api/mission")
		async def get_mission() -> dict[str, Any]:
			mission = self.db.get_active_mission() or self.db.get_latest_mission()
			if mission is None:
				return {"status": "no_mission"}
			return _serialize_mission(mission)

		@self.app.get("/api/units")
		async def get_units() -> list[dict[str, Any]]:
			mission = self.db.get_active_mission() or self.db.get_latest_mission()
			if mission is None:
				return []
			units = self.db.get_work_units_for_mission(mission.id)
			return [_serialize_unit(u) for u in units]

		@self.app.get("/api/events")
		async def get_events() -> list[dict[str, Any]]:
			mission = self.db.get_active_mission() or self.db.get_latest_mission()
			if mission is None:
				return []
			events = self.db.get_unit_events_for_mission(mission.id, limit=200)
			return [_serialize_event(e) for e in events]

		@self.app.get("/api/plan-tree")
		async def get_plan_tree() -> list[dict[str, Any]]:
			mission = self.db.get_active_mission() or self.db.get_latest_mission()
			if mission is None:
				return []
			return self._build_plan_tree(mission.id)

	async def _broadcast_loop(self) -> None:
		"""Poll DB every 1s, push snapshot to all connected clients."""
		while True:
			if self._connections:
				snapshot = self._build_snapshot()
				# Check if the UI file has changed on disk (e.g. from auto-push)
				if _UI_PATH.exists():
					current_mtime = _UI_PATH.stat().st_mtime
					if current_mtime != self._ui_mtime:
						self._ui_mtime = current_mtime
						snapshot["reload"] = True
				for ws in list(self._connections):
					try:
						await ws.send_json(snapshot)
					except Exception:
						self._connections.discard(ws)
			await asyncio.sleep(1)

	def _build_snapshot(self) -> dict[str, Any]:
		"""Build current mission state from DB."""
		mission = self.db.get_active_mission() or self.db.get_latest_mission()
		if mission is None:
			return {"mission": None, "units": [], "events": [], "plan_tree": []}

		units = self.db.get_work_units_for_mission(mission.id)
		events = self.db.get_unit_events_for_mission(mission.id, limit=100)

		return {
			"mission": _serialize_mission(mission),
			"units": [_serialize_unit(u) for u in units],
			"events": [_serialize_event(e) for e in events],
			"plan_tree": self._build_plan_tree(mission.id),
		}

	def _build_plan_tree(self, mission_id: str) -> list[dict[str, Any]]:
		"""Build plan tree from epochs and their plan nodes."""
		epochs = self.db.get_epochs_for_mission(mission_id)
		tree: list[dict[str, Any]] = []

		for epoch in epochs:
			# Get units for this epoch
			units = [
				u for u in self.db.get_work_units_for_mission(mission_id)
				if u.epoch_id == epoch.id
			]
			tree.append({
				"type": "epoch",
				"id": epoch.id,
				"number": epoch.number,
				"units_planned": epoch.units_planned,
				"units_completed": epoch.units_completed,
				"units_failed": epoch.units_failed,
				"children": [
					{
						"type": "unit",
						"id": u.id,
						"title": u.title,
						"status": u.status,
						"attempt": u.attempt,
						"max_attempts": u.max_attempts,
						"files_hint": u.files_hint,
					}
					for u in units
				],
			})

		return tree


def _serialize_mission(m: Any) -> dict[str, Any]:
	return {
		"id": m.id,
		"objective": m.objective,
		"status": m.status,
		"started_at": m.started_at,
		"finished_at": m.finished_at,
		"total_cost_usd": m.total_cost_usd,
		"stopped_reason": m.stopped_reason,
	}


def _serialize_unit(u: Any) -> dict[str, Any]:
	return {
		"id": u.id,
		"title": u.title,
		"description": u.description[:200],
		"status": u.status,
		"attempt": u.attempt,
		"max_attempts": u.max_attempts,
		"files_hint": u.files_hint,
		"started_at": u.started_at,
		"finished_at": u.finished_at,
		"output_summary": u.output_summary[:200],
		"cost_usd": u.cost_usd,
		"input_tokens": u.input_tokens,
		"output_tokens": u.output_tokens,
		"unit_type": u.unit_type,
		"epoch_id": u.epoch_id,
	}


def _serialize_event(e: Any) -> dict[str, Any]:
	return {
		"id": e.id,
		"work_unit_id": e.work_unit_id,
		"event_type": e.event_type,
		"timestamp": e.timestamp,
		"details": e.details,
	}
