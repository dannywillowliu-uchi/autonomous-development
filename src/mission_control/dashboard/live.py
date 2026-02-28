"""Live mission control dashboard -- real-time WebSocket + REST API."""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from mission_control.db import Database
from mission_control.models import Signal, TrajectoryRating

logger = logging.getLogger(__name__)

_UI_PATH = importlib.resources.files("mission_control.dashboard").joinpath("live_ui.html")

VALID_SIGNAL_TYPES = {"stop", "pause", "resume", "cancel_unit", "adjust_workers"}

_SIGNAL_PAYLOAD_REQUIREMENTS: dict[str, set[str]] = {
	"adjust_workers": {"num_workers"},
	"cancel_unit": {"unit_id"},
}

_RATE_LIMIT_MAX = 10
_RATE_LIMIT_WINDOW = 60  # seconds


class RatingRequest(BaseModel):
	"""Request body for submitting a trajectory rating."""

	mission_id: str
	rating: int
	feedback: str = ""


class SignalRequest(BaseModel):
	"""Request body for sending a signal to the controller."""

	signal_type: str
	payload: dict[str, Any] = {}


class LiveDashboard:
	"""Real-time mission control dashboard.

	Serves a WebSocket endpoint that pushes mission snapshots every second,
	plus REST endpoints for sending control signals and querying state.
	"""

	def __init__(self, db_path: str, auth_token: str = "") -> None:
		self.db = Database(db_path)
		self.auth_token = auth_token
		self._connections: set[WebSocket] = set()
		self._broadcast_task: asyncio.Task[None] | None = None
		# Resolve to a concrete Path for mtime tracking (hot-reload support)
		self._ui_concrete = Path(str(_UI_PATH)) if _UI_PATH.is_file() else None
		self._ui_mtime: float = self._ui_concrete.stat().st_mtime if self._ui_concrete else 0.0
		self._signal_timestamps: dict[str, list[float]] = defaultdict(list)

		@asynccontextmanager
		async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
			self._broadcast_task = asyncio.create_task(self._broadcast_loop())
			yield
			if self._broadcast_task:
				self._broadcast_task.cancel()
				try:
					await self._broadcast_task
				except asyncio.CancelledError:
					pass
			self.db.close()

		self.app = FastAPI(title="Mission Control Live", lifespan=_lifespan)
		self.app.add_middleware(
			CORSMiddleware,
			allow_origins=["http://127.0.0.1", "http://localhost"],
			allow_methods=["GET", "POST", "PATCH", "DELETE"],
			allow_headers=["Authorization", "Content-Type"],
		)
		self._setup_routes()

	def _current_mission(self) -> Any | None:
		"""Return the active mission, falling back to the latest."""
		return self.db.get_active_mission() or self.db.get_latest_mission()

	def _get_chain_mission_ids(self, mission: Any) -> list[str]:
		"""Return mission IDs for the chain, or just [mission.id] if not chained."""
		if mission.chain_id:
			chain_missions = self.db.get_missions_for_chain(mission.chain_id)
			if chain_missions:
				return [m.id for m in chain_missions]
		return [mission.id]

	def _verify_token(self, credentials: HTTPAuthorizationCredentials) -> None:
		if self.auth_token and credentials.credentials != self.auth_token:
			raise HTTPException(status_code=401, detail="Invalid token")

	def _check_rate_limit(self, client_ip: str) -> None:
		now = time.monotonic()
		cutoff = now - _RATE_LIMIT_WINDOW
		timestamps = self._signal_timestamps[client_ip]
		self._signal_timestamps[client_ip] = [t for t in timestamps if t > cutoff]
		if len(self._signal_timestamps[client_ip]) >= _RATE_LIMIT_MAX:
			raise HTTPException(status_code=429, detail="Rate limit exceeded (max 10 signals/min)")
		self._signal_timestamps[client_ip].append(now)

	def _setup_routes(self) -> None:
		_security = HTTPBearer(auto_error=False)

		async def verify_token(
			credentials: HTTPAuthorizationCredentials | None = Depends(_security),
		) -> None:
			if not self.auth_token:
				return
			if credentials is None:
				raise HTTPException(status_code=401, detail="Missing authorization header")
			self._verify_token(credentials)

		@self.app.websocket("/ws")
		async def ws_endpoint(websocket: WebSocket, token: str = Query(default="")) -> None:
			if self.auth_token and token != self.auth_token:
				await websocket.close(code=4401, reason="Invalid token")
				return
			await websocket.accept()
			self._connections.add(websocket)
			try:
				snapshot = self._build_snapshot()
				await websocket.send_json(snapshot)
				while True:
					await websocket.receive_text()
			except WebSocketDisconnect:
				pass
			finally:
				self._connections.discard(websocket)

		@self.app.get("/", response_class=HTMLResponse)
		async def index() -> HTMLResponse:
			if _UI_PATH.is_file():
				return HTMLResponse(_UI_PATH.read_text())
			return HTMLResponse("<h1>Mission Control Live</h1><p>UI file not found.</p>")

		@self.app.post("/api/signal", dependencies=[Depends(verify_token)])
		async def send_signal(body: SignalRequest, request: Request) -> dict[str, str]:
			if body.signal_type not in VALID_SIGNAL_TYPES:
				raise HTTPException(
					status_code=422,
					detail=f"Unknown signal_type '{body.signal_type}'. Valid: {sorted(VALID_SIGNAL_TYPES)}",
				)

			required_keys = _SIGNAL_PAYLOAD_REQUIREMENTS.get(body.signal_type, set())
			missing = required_keys - set(body.payload.keys())
			if missing:
				raise HTTPException(
					status_code=422,
					detail=f"signal_type '{body.signal_type}' requires payload keys: {sorted(missing)}",
				)

			client_ip = request.client.host if request.client else "unknown"
			self._check_rate_limit(client_ip)

			mission = self._current_mission()
			if mission is None:
				return {"status": "error", "message": "No mission found"}

			# Map adjust_workers -> adjust for controller compatibility
			db_signal_type = "adjust" if body.signal_type == "adjust_workers" else body.signal_type
			signal = Signal(
				mission_id=mission.id,
				signal_type=db_signal_type,
				payload=json.dumps(body.payload) if body.payload else "",
			)
			self.db.insert_signal(signal)
			return {"status": "ok", "signal_id": signal.id}

		@self.app.get("/api/mission", dependencies=[Depends(verify_token)])
		async def get_mission() -> dict[str, Any]:
			mission = self._current_mission()
			if mission is None:
				return {"status": "no_mission"}
			return _serialize_mission(mission)

		@self.app.get("/api/units", dependencies=[Depends(verify_token)])
		async def get_units() -> list[dict[str, Any]]:
			mission = self._current_mission()
			if mission is None:
				return []
			chain_mids = self._get_chain_mission_ids(mission)
			units: list[Any] = []
			for mid in chain_mids:
				units.extend(self.db.get_work_units_for_mission(mid))
			return [_serialize_unit(u) for u in units]

		@self.app.get("/api/events", dependencies=[Depends(verify_token)])
		async def get_events() -> list[dict[str, Any]]:
			mission = self._current_mission()
			if mission is None:
				return []
			chain_mids = self._get_chain_mission_ids(mission)
			events: list[Any] = []
			for mid in chain_mids:
				events.extend(self.db.get_unit_events_for_mission(mid, limit=200))
			events.sort(key=lambda e: e.timestamp)
			return [_serialize_event(e) for e in events[-200:]]

		@self.app.get("/api/plan-tree", dependencies=[Depends(verify_token)])
		async def get_plan_tree() -> list[dict[str, Any]]:
			mission = self._current_mission()
			if mission is None:
				return []
			chain_mids = self._get_chain_mission_ids(mission)
			return self._build_plan_tree(chain_mids)

		@self.app.get("/api/timeline", dependencies=[Depends(verify_token)])
		async def get_timeline() -> list[dict[str, Any]]:
			mission = self._current_mission()
			if mission is None:
				return []
			chain_mids = self._get_chain_mission_ids(mission)
			return self._build_timeline(chain_mids)

		@self.app.get("/api/workers", dependencies=[Depends(verify_token)])
		async def get_workers() -> list[dict[str, Any]]:
			workers = self.db.get_all_workers()
			return [_serialize_worker(w) for w in workers]

		@self.app.get("/api/summary", dependencies=[Depends(verify_token)])
		async def get_summary() -> dict[str, Any]:
			mission = self._current_mission()
			if mission is None:
				return {"status": "no_mission"}
			chain_mids = self._get_chain_mission_ids(mission)
			return self._build_summary(mission, chain_mids)

		@self.app.get("/api/history", dependencies=[Depends(verify_token)])
		async def get_history() -> list[dict[str, Any]]:
			return self._build_history()

		@self.app.post("/api/rating", dependencies=[Depends(verify_token)])
		async def submit_rating(body: RatingRequest) -> dict[str, str]:
			if not 1 <= body.rating <= 10:
				raise HTTPException(
					status_code=422,
					detail="Rating must be between 1 and 10",
				)
			mission = self.db.get_mission(body.mission_id)
			if mission is None:
				raise HTTPException(status_code=404, detail="Mission not found")
			rating = TrajectoryRating(
				mission_id=body.mission_id,
				rating=body.rating,
				feedback=body.feedback,
			)
			self.db.insert_trajectory_rating(rating)
			return {"status": "ok", "rating_id": rating.id}

		@self.app.get(
			"/api/ratings/{mission_id}",
			dependencies=[Depends(verify_token)],
		)
		async def get_ratings(mission_id: str) -> list[dict[str, Any]]:
			ratings = self.db.get_trajectory_ratings_for_mission(mission_id)
			return [
				{
					"id": r.id,
					"mission_id": r.mission_id,
					"rating": r.rating,
					"feedback": r.feedback,
					"timestamp": r.timestamp,
				}
				for r in ratings
			]

	async def _broadcast_loop(self) -> None:
		"""Poll DB every 1s, push snapshot to all connected clients."""
		while True:
			if self._connections:
				snapshot = self._build_snapshot()
				# Check if the UI file has changed on disk (e.g. from auto-push)
				if self._ui_concrete and self._ui_concrete.exists():
					current_mtime = self._ui_concrete.stat().st_mtime
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
		mission = self._current_mission()
		if mission is None:
			return {
				"mission": None, "units": [], "events": [],
				"plan_tree": [], "workers": [], "summary": None,
				"history": [],
			}

		chain_mids = self._get_chain_mission_ids(mission)

		units: list[Any] = []
		events: list[Any] = []
		rating_data: list[dict[str, Any]] = []
		review_by_unit: dict[str, dict[str, Any]] = {}
		for mid in chain_mids:
			units.extend(self.db.get_work_units_for_mission(mid))
			events.extend(self.db.get_unit_events_for_mission(mid, limit=100))
			for r in self.db.get_trajectory_ratings_for_mission(mid):
				rating_data.append({"rating": r.rating, "feedback": r.feedback, "timestamp": r.timestamp})
			for r in self.db.get_unit_reviews_for_mission(mid):
				review_by_unit[r.work_unit_id] = {
					"alignment": r.alignment_score,
					"approach": r.approach_score,
					"tests": r.test_score,
					"avg": r.avg_score,
					"rationale": r.rationale[:200],
				}
		events.sort(key=lambda e: e.timestamp)
		events = events[-100:]

		workers = self.db.get_all_workers()

		plan_tree = self._build_plan_tree(chain_mids)

		# Extract current (last) epoch's units for the mission statement section
		current_epoch_units: list[dict[str, Any]] = []
		if plan_tree:
			last_epoch = plan_tree[-1]
			current_epoch_units = last_epoch.get("children", [])

		return {
			"mission": _serialize_mission(mission),
			"units": [_serialize_unit(u) for u in units],
			"events": [_serialize_event(e) for e in events],
			"plan_tree": plan_tree,
			"workers": [_serialize_worker(w) for w in workers],
			"summary": self._build_summary(mission, chain_mids),
			"history": self._build_history(),
			"ratings": rating_data,
			"unit_reviews": review_by_unit,
			"current_epoch_units": current_epoch_units,
			"timeline": self._build_timeline(chain_mids),
		}

	def _build_plan_tree(self, mission_ids: list[str]) -> list[dict[str, Any]]:
		"""Build plan tree from epochs and their plan nodes across chain missions."""
		all_epochs: list[Any] = []
		all_units: list[Any] = []
		for mid in mission_ids:
			all_epochs.extend(self.db.get_epochs_for_mission(mid))
			all_units.extend(self.db.get_work_units_for_mission(mid))

		# Index units by epoch for O(1) lookup instead of repeated DB queries
		units_by_epoch: dict[str, list[Any]] = defaultdict(list)
		for u in all_units:
			if u.epoch_id:
				units_by_epoch[u.epoch_id].append(u)

		tree: list[dict[str, Any]] = []
		for seq, epoch in enumerate(all_epochs, 1):
			tree.append({
				"type": "epoch",
				"id": epoch.id,
				"number": seq,
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
					for u in units_by_epoch.get(epoch.id, [])
				],
			})

		return tree

	def _build_summary(self, mission: Any, chain_mids: list[str]) -> dict[str, Any]:
		"""Build lightweight summary stats for the dashboard stats bar."""
		total_cost = 0.0
		merged = 0
		failed = 0
		running = 0
		pending = 0
		all_epochs: list[dict[str, Any]] = []
		for mid in chain_mids:
			agg = self.db.get_mission_summary(mid)
			units = agg["units_by_status"]
			merged += units.get("completed", 0)
			failed += units.get("failed", 0)
			running += units.get("running", 0) + units.get("claimed", 0)
			pending += units.get("pending", 0)
			total_cost += agg["total_cost_usd"]
			all_epochs.extend(agg["epochs"])
		# Use first mission's started_at (chain_mids ordered by started_at)
		first_mission = self.db.get_mission(chain_mids[0]) if chain_mids else mission
		return {
			"total_cost_usd": total_cost,
			"units_merged": merged,
			"units_failed": failed,
			"units_running": running,
			"units_pending": pending,
			"current_epoch": len(all_epochs),
			"started_at": first_mission.started_at if first_mission else mission.started_at,
		}

	def _build_timeline(self, chain_mids: list[str]) -> list[dict[str, Any]]:
		"""Build timeline entries from work units for Gantt visualization."""
		all_units: list[Any] = []
		for mid in chain_mids:
			all_units.extend(self.db.get_work_units_for_mission(mid))

		timeline: list[dict[str, Any]] = []
		for u in all_units:
			if not u.started_at:
				continue
			files_changed: list[str] = []
			if u.files_hint:
				files_changed = [f.strip() for f in u.files_hint.split(",") if f.strip()]
			timeline.append({
				"unit_id": u.id,
				"title": u.title,
				"worker_id": u.worker_id or "",
				"start_time": u.started_at,
				"end_time": u.finished_at,
				"status": u.status,
				"files_changed": files_changed,
			})
		timeline.sort(key=lambda t: t["start_time"])
		return timeline

	def _build_history(self) -> list[dict[str, Any]]:
		"""Build mission history from all past missions."""
		missions = self.db.get_all_missions(limit=50)
		result: list[dict[str, Any]] = []
		for m in missions:
			summary = self.db.get_mission_summary(m.id)
			units = summary["units_by_status"]
			duration: float | None = None
			if m.started_at and m.finished_at:
				try:
					start = datetime.fromisoformat(m.started_at)
					end = datetime.fromisoformat(m.finished_at)
					duration = (end - start).total_seconds()
				except (ValueError, TypeError):
					pass
			ratings = self.db.get_trajectory_ratings_for_mission(m.id)
			latest_rating = ratings[0].rating if ratings else None
			result.append({
				"id": m.id,
				"objective": m.objective,
				"status": m.status,
				"started_at": m.started_at,
				"finished_at": m.finished_at,
				"duration": duration,
				"units_merged": units.get("completed", 0),
				"units_failed": units.get("failed", 0),
				"total_cost_usd": m.total_cost_usd,
				"stopped_reason": m.stopped_reason,
				"trajectory_rating": latest_rating,
			})
		return result


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


def _serialize_worker(w: Any) -> dict[str, Any]:
	output_excerpt = ""
	if w.backend_metadata:
		try:
			meta = json.loads(w.backend_metadata)
			output_excerpt = meta.get("output_excerpt", "")
		except (json.JSONDecodeError, TypeError):
			pass
	return {
		"id": w.id,
		"workspace_path": w.workspace_path,
		"status": w.status,
		"current_unit_id": w.current_unit_id,
		"pid": w.pid,
		"started_at": w.started_at,
		"last_heartbeat": w.last_heartbeat,
		"units_completed": w.units_completed,
		"units_failed": w.units_failed,
		"total_cost_usd": w.total_cost_usd,
		"backend_type": w.backend_type,
		"output_excerpt": output_excerpt,
	}
