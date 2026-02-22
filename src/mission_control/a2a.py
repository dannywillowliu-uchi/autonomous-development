"""Agent-to-Agent (A2A) protocol v0.3 support for mission-control.

Implements the A2A protocol directly with FastAPI -- no external SDK needed.
The protocol is simple JSON-RPC over HTTPS: AgentCard discovery + task CRUD.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel

if TYPE_CHECKING:
	from mission_control.config import A2AConfig
	from mission_control.db import Database

logger = logging.getLogger(__name__)


# -- Status mapping --

class A2ATaskState(str, Enum):
	"""A2A task lifecycle states."""

	SUBMITTED = "submitted"
	WORKING = "working"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELED = "canceled"


_WU_STATUS_MAP: dict[str, str] = {
	"pending": A2ATaskState.SUBMITTED,
	"queued": A2ATaskState.SUBMITTED,
	"dispatched": A2ATaskState.WORKING,
	"running": A2ATaskState.WORKING,
	"completed": A2ATaskState.COMPLETED,
	"merged": A2ATaskState.COMPLETED,
	"failed": A2ATaskState.FAILED,
	"cancelled": A2ATaskState.CANCELED,
}


def _wu_status_to_a2a(wu_status: str) -> str:
	"""Map a WorkUnit status string to an A2A task state."""
	return _WU_STATUS_MAP.get(wu_status, A2ATaskState.SUBMITTED)


# -- Pydantic models --

class AgentCard(BaseModel):
	"""A2A agent capability advertisement (/.well-known/agent.json)."""

	name: str
	version: str
	description: str = "mission-control autonomous development agent"
	capabilities: list[str] = []
	supported_task_types: list[str] = ["code_edit", "test_write", "refactor"]
	endpoint: str = ""
	auth: dict[str, Any] = {}


class A2ATaskCreate(BaseModel):
	"""Inbound task creation request."""

	title: str
	description: str = ""
	task_type: str = "code_edit"
	metadata: dict[str, Any] = {}


class A2ATaskStatus(BaseModel):
	"""Task status response."""

	id: str
	state: str
	summary: str = ""
	created_at: str = ""
	updated_at: str = ""


# -- Server --

class A2AServer:
	"""A2A protocol server backed by FastAPI."""

	def __init__(self, config: A2AConfig, db: Database) -> None:
		self._config = config
		self._db = db
		self._server_thread: threading.Thread | None = None
		self._server: Any = None
		self._app = self._build_app()

	def _build_app(self) -> Any:
		try:
			from fastapi import FastAPI, HTTPException
		except ImportError:
			raise ImportError(
				"FastAPI is required for A2A server. "
				"Install with: pip install mission-control[dashboard]"
			)

		app = FastAPI(title="mission-control A2A", version="0.3")
		config = self._config
		db = self._db

		card = AgentCard(
			name=config.agent_name,
			version=config.agent_version,
			capabilities=config.agent_capabilities,
			endpoint=f"http://{config.host}:{config.port}",
		)

		@app.get("/.well-known/agent.json")
		def get_agent_card() -> dict[str, Any]:
			return card.model_dump()

		@app.post("/a2a/tasks", status_code=201)
		def create_task(req: A2ATaskCreate) -> dict[str, Any]:
			from mission_control.models import Plan, WorkUnit, _now_iso

			now = _now_iso()
			plan = Plan(objective=req.title, status="active")
			db.insert_plan(plan)

			unit = WorkUnit(
				plan_id=plan.id,
				title=req.title,
				description=req.description,
				status="pending",
			)
			db.insert_work_unit(unit)

			status = A2ATaskStatus(
				id=unit.id,
				state=_wu_status_to_a2a(unit.status),
				summary=req.title,
				created_at=now,
				updated_at=now,
			)
			return status.model_dump()

		@app.get("/a2a/tasks/{task_id}")
		def get_task(task_id: str) -> dict[str, Any]:
			unit = db.get_work_unit(task_id)
			if unit is None:
				raise HTTPException(status_code=404, detail="Task not found")
			return A2ATaskStatus(
				id=unit.id,
				state=_wu_status_to_a2a(unit.status),
				summary=unit.title,
				created_at=unit.started_at or "",
				updated_at=unit.finished_at or unit.started_at or "",
			).model_dump()

		@app.post("/a2a/tasks/{task_id}/cancel")
		def cancel_task(task_id: str) -> dict[str, Any]:
			unit = db.get_work_unit(task_id)
			if unit is None:
				raise HTTPException(status_code=404, detail="Task not found")
			unit.status = "cancelled"
			db.update_work_unit(unit)
			return A2ATaskStatus(
				id=unit.id,
				state=_wu_status_to_a2a(unit.status),
				summary=unit.title,
			).model_dump()

		return app

	@property
	def app(self) -> Any:
		"""Expose the FastAPI app (useful for testing with TestClient)."""
		return self._app

	def start(self) -> None:
		"""Launch uvicorn in a background thread (non-blocking)."""
		try:
			import uvicorn
		except ImportError:
			raise ImportError(
				"uvicorn is required for A2A server. "
				"Install with: pip install mission-control[dashboard]"
			)

		uvi_config = uvicorn.Config(
			app=self._app,
			host=self._config.host,
			port=self._config.port,
			log_level="warning",
		)
		self._server = uvicorn.Server(uvi_config)

		self._server_thread = threading.Thread(
			target=self._server.run,
			daemon=True,
			name="a2a-server",
		)
		self._server_thread.start()
		logger.info("A2A server started on %s:%d", self._config.host, self._config.port)

	def stop(self) -> None:
		"""Shut down the server gracefully."""
		if self._server is not None:
			self._server.should_exit = True
			if self._server_thread is not None:
				self._server_thread.join(timeout=5)
			logger.info("A2A server stopped")


# -- Client --

class A2AClient:
	"""A2A protocol client using httpx."""

	def __init__(self) -> None:
		self._client = httpx.AsyncClient(timeout=30.0)

	async def discover(self, endpoint: str) -> AgentCard:
		"""Fetch the agent card from a remote A2A endpoint."""
		url = f"{endpoint.rstrip('/')}/.well-known/agent.json"
		resp = await self._client.get(url)
		resp.raise_for_status()
		return AgentCard(**resp.json())

	async def delegate(self, endpoint: str, task: A2ATaskCreate) -> A2ATaskStatus:
		"""Create a task on a remote A2A agent."""
		url = f"{endpoint.rstrip('/')}/a2a/tasks"
		resp = await self._client.post(url, json=task.model_dump())
		resp.raise_for_status()
		return A2ATaskStatus(**resp.json())

	async def get_status(self, endpoint: str, task_id: str) -> A2ATaskStatus:
		"""Get task status from a remote A2A agent."""
		url = f"{endpoint.rstrip('/')}/a2a/tasks/{task_id}"
		resp = await self._client.get(url)
		resp.raise_for_status()
		return A2ATaskStatus(**resp.json())

	async def cancel(self, endpoint: str, task_id: str) -> A2ATaskStatus:
		"""Cancel a task on a remote A2A agent."""
		url = f"{endpoint.rstrip('/')}/a2a/tasks/{task_id}/cancel"
		resp = await self._client.post(url)
		resp.raise_for_status()
		return A2ATaskStatus(**resp.json())

	async def close(self) -> None:
		"""Close the underlying HTTP client."""
		await self._client.aclose()
