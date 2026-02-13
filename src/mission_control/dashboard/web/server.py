"""FastAPI web dashboard for mission-control monitoring and control."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mission_control.dashboard.provider import DashboardProvider
from mission_control.launcher import MissionLauncher
from mission_control.registry import ProjectRegistry

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"


class DashboardState:
	"""Shared state for the dashboard app."""

	def __init__(self, registry: ProjectRegistry) -> None:
		self.registry = registry
		self.launcher = MissionLauncher(registry)
		self._providers: dict[str, DashboardProvider] = {}

	def get_provider(self, project_name: str) -> DashboardProvider | None:
		"""Lazy-create a DashboardProvider for the given project."""
		if project_name in self._providers:
			return self._providers[project_name]

		project = self.registry.get_project(project_name)
		if project is None:
			return None

		db_path = Path(project.db_path)
		if not db_path.exists():
			return None

		provider = DashboardProvider(str(db_path))
		provider.start_polling(interval=2.0)
		self._providers[project_name] = provider
		return provider

	def stop_all(self) -> None:
		"""Stop all providers."""
		for p in self._providers.values():
			p.stop()
		self._providers.clear()


_state: DashboardState | None = None


def create_app(db_path: str | None = None, registry: ProjectRegistry | None = None) -> FastAPI:
	"""Factory: build the dashboard FastAPI app.

	Accepts either a single db_path (legacy single-project mode) or
	a ProjectRegistry (multi-project mode).
	"""

	@asynccontextmanager
	async def lifespan(app: FastAPI):
		global _state
		if registry is not None:
			_state = DashboardState(registry)
		else:
			# Legacy single-project mode: create temporary registry
			reg = ProjectRegistry()
			if db_path:
				config_dir = Path(db_path).parent
				config_path = config_dir / "mission-control.toml"
				name = config_dir.name or "default"
				try:
					reg.register(name=name, config_path=str(config_path), db_path=db_path)
				except ValueError:
					pass  # Already registered
			_state = DashboardState(reg)
		yield
		if _state:
			_state.stop_all()
		_state = None

	app = FastAPI(title="Mission Control Dashboard", lifespan=lifespan)

	templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

	app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

	# -- Full page --

	@app.get("/", response_class=HTMLResponse)
	async def index(request: Request):
		return templates.TemplateResponse("layout.html", {"request": request})

	@app.get("/project/{name}", response_class=HTMLResponse)
	async def project_page(request: Request, name: str):
		"""Full-page load for a project tab (for direct URL access)."""
		return templates.TemplateResponse("layout.html", {"request": request})

	# -- Overview --

	@app.get("/overview", response_class=HTMLResponse)
	async def overview(request: Request):
		assert _state is not None
		projects = []
		for p in _state.registry.list_projects():
			status = _state.registry.get_project_status(p.name)
			if status:
				projects.append(status)
		return templates.TemplateResponse(
			"overview.html",
			{"request": request, "projects": projects},
		)

	# -- Tab bar --

	@app.get("/partials/tab-bar", response_class=HTMLResponse)
	async def tab_bar(request: Request, active: str = "overview"):
		assert _state is not None
		project_names = [p.name for p in _state.registry.list_projects()]
		return templates.TemplateResponse(
			"partials/tab_bar.html",
			{"request": request, "project_names": project_names, "active_tab": active},
		)

	# -- Per-project dashboard --

	@app.get("/project/{name}/dashboard", response_class=HTMLResponse)
	async def project_dashboard(request: Request, name: str):
		return templates.TemplateResponse(
			"project_dashboard.html",
			{"request": request, "project_name": name},
		)

	# -- Per-project HTMX partials --

	@app.get("/project/{name}/partials/mission-header", response_class=HTMLResponse)
	async def project_mission_header(request: Request, name: str):
		assert _state is not None
		provider = _state.get_provider(name)
		snap = provider.get_snapshot() if provider else None
		return templates.TemplateResponse(
			"partials/mission_header.html",
			{"request": request, "snap": snap, "project_name": name},
		)

	@app.get("/project/{name}/partials/worker-pool", response_class=HTMLResponse)
	async def project_worker_pool(request: Request, name: str):
		assert _state is not None
		provider = _state.get_provider(name)
		snap = provider.get_snapshot() if provider else None
		return templates.TemplateResponse(
			"partials/worker_pool.html",
			{"request": request, "snap": snap},
		)

	@app.get("/project/{name}/partials/work-units", response_class=HTMLResponse)
	async def project_work_units(request: Request, name: str):
		assert _state is not None
		provider = _state.get_provider(name)
		snap = provider.get_snapshot() if provider else None
		units = _get_current_units(name) if snap else []
		return templates.TemplateResponse(
			"partials/work_units.html",
			{"request": request, "snap": snap, "units": units, "project_name": name},
		)

	@app.get("/project/{name}/partials/merge-queue", response_class=HTMLResponse)
	async def project_merge_queue(request: Request, name: str):
		assert _state is not None
		provider = _state.get_provider(name)
		snap = provider.get_snapshot() if provider else None
		return templates.TemplateResponse(
			"partials/merge_queue.html",
			{"request": request, "snap": snap},
		)

	@app.get("/project/{name}/partials/activity-log", response_class=HTMLResponse)
	async def project_activity_log(request: Request, name: str):
		assert _state is not None
		provider = _state.get_provider(name)
		snap = provider.get_snapshot() if provider else None
		return templates.TemplateResponse(
			"partials/activity_log.html",
			{"request": request, "snap": snap},
		)

	@app.get("/project/{name}/partials/unit/{unit_id}", response_class=HTMLResponse)
	async def project_unit_detail(request: Request, name: str, unit_id: str):
		assert _state is not None
		unit = _get_work_unit(name, unit_id)
		return templates.TemplateResponse(
			"partials/unit_detail.html",
			{"request": request, "unit": unit, "project_name": name},
		)

	# -- Per-project JSON APIs --

	@app.get("/project/{name}/api/score-history")
	async def project_score_history(name: str):
		assert _state is not None
		provider = _state.get_provider(name)
		snap = provider.get_snapshot() if provider else None
		if not snap or not snap.score_history:
			return JSONResponse({"labels": [], "data": []})
		labels = [f"R{r}" for r, _ in snap.score_history]
		data = [s for _, s in snap.score_history]
		return JSONResponse({"labels": labels, "data": data})

	@app.get("/project/{name}/api/test-trend")
	async def project_test_trend(name: str):
		assert _state is not None
		provider = _state.get_provider(name)
		snap = provider.get_snapshot() if provider else None
		if not snap or not snap.test_trend:
			return JSONResponse({"labels": [], "passed": [], "failed": []})
		labels = [f"R{r}" for r, _, _ in snap.test_trend]
		passed = [p for _, p, _ in snap.test_trend]
		failed = [f for _, _, f in snap.test_trend]
		return JSONResponse({"labels": labels, "passed": passed, "failed": failed})

	@app.get("/project/{name}/api/cost-per-round")
	async def project_cost_per_round(name: str):
		assert _state is not None
		provider = _state.get_provider(name)
		snap = provider.get_snapshot() if provider else None
		if not snap or not snap.cost_per_round:
			return JSONResponse({"labels": [], "data": []})
		labels = [f"R{r}" for r, _ in snap.cost_per_round]
		data = [c for _, c in snap.cost_per_round]
		return JSONResponse({"labels": labels, "data": data})

	# -- Legacy single-project partials (backwards compat) --

	@app.get("/partials/mission-header", response_class=HTMLResponse)
	async def legacy_mission_header(request: Request):
		snap = _get_first_provider_snap()
		return templates.TemplateResponse(
			"partials/mission_header.html",
			{"request": request, "snap": snap, "project_name": None},
		)

	@app.get("/partials/worker-pool", response_class=HTMLResponse)
	async def legacy_worker_pool(request: Request):
		snap = _get_first_provider_snap()
		return templates.TemplateResponse("partials/worker_pool.html", {"request": request, "snap": snap})

	@app.get("/partials/work-units", response_class=HTMLResponse)
	async def legacy_work_units(request: Request):
		snap = _get_first_provider_snap()
		return templates.TemplateResponse(
			"partials/work_units.html",
			{"request": request, "snap": snap, "units": [], "project_name": None},
		)

	@app.get("/partials/merge-queue", response_class=HTMLResponse)
	async def legacy_merge_queue(request: Request):
		snap = _get_first_provider_snap()
		return templates.TemplateResponse("partials/merge_queue.html", {"request": request, "snap": snap})

	@app.get("/partials/activity-log", response_class=HTMLResponse)
	async def legacy_activity_log(request: Request):
		snap = _get_first_provider_snap()
		return templates.TemplateResponse("partials/activity_log.html", {"request": request, "snap": snap})

	@app.get("/api/score-history")
	async def legacy_score_history():
		snap = _get_first_provider_snap()
		if not snap or not snap.score_history:
			return JSONResponse({"labels": [], "data": []})
		labels = [f"R{r}" for r, _ in snap.score_history]
		data = [s for _, s in snap.score_history]
		return JSONResponse({"labels": labels, "data": data})

	@app.get("/api/test-trend")
	async def legacy_test_trend():
		snap = _get_first_provider_snap()
		if not snap or not snap.test_trend:
			return JSONResponse({"labels": [], "passed": [], "failed": []})
		labels = [f"R{r}" for r, _, _ in snap.test_trend]
		passed = [p for _, p, _ in snap.test_trend]
		failed = [f for _, _, f in snap.test_trend]
		return JSONResponse({"labels": labels, "passed": passed, "failed": failed})

	@app.get("/api/cost-per-round")
	async def legacy_cost_per_round():
		snap = _get_first_provider_snap()
		if not snap or not snap.cost_per_round:
			return JSONResponse({"labels": [], "data": []})
		labels = [f"R{r}" for r, _ in snap.cost_per_round]
		data = [c for _, c in snap.cost_per_round]
		return JSONResponse({"labels": labels, "data": data})

	# -- Launch wizard --

	@app.get("/launch/step/1", response_class=HTMLResponse)
	async def launch_step1(request: Request):
		assert _state is not None
		projects = _state.registry.list_projects()
		return templates.TemplateResponse(
			"wizard/step1_project.html",
			{"request": request, "projects": projects},
		)

	@app.get("/launch/step/2", response_class=HTMLResponse)
	async def launch_step2(request: Request, project: str = ""):
		assert _state is not None
		default_objective = ""
		proj = _state.registry.get_project(project)
		if proj:
			# Try to load objective from config
			try:
				from mission_control.config import load_config
				config = load_config(proj.config_path)
				default_objective = config.target.objective
			except Exception:
				pass
		return templates.TemplateResponse(
			"wizard/step2_objective.html",
			{"request": request, "project_name": project, "default_objective": default_objective},
		)

	@app.get("/launch/step/3", response_class=HTMLResponse)
	async def launch_step3(request: Request, project: str = "", objective: str = ""):
		assert _state is not None
		defaults = {"model": "sonnet", "workers": 4, "max_rounds": 20, "budget": 5.0}
		proj = _state.registry.get_project(project)
		if proj:
			try:
				from mission_control.config import load_config
				config = load_config(proj.config_path)
				defaults["model"] = config.scheduler.model
				defaults["workers"] = config.scheduler.parallel.num_workers
				defaults["max_rounds"] = config.rounds.max_rounds
				defaults["budget"] = config.scheduler.budget.max_per_session_usd
			except Exception:
				pass
		return templates.TemplateResponse(
			"wizard/step3_configure.html",
			{"request": request, "project_name": project, "objective": objective, "defaults": defaults},
		)

	@app.get("/launch/step/4", response_class=HTMLResponse)
	async def launch_step4(
		request: Request,
		project: str = "",
		objective: str = "",
		model: str = "sonnet",
		workers: int = 4,
		max_rounds: int = 20,
		budget: float = 5.0,
	):
		return templates.TemplateResponse(
			"wizard/step4_review.html",
			{
				"request": request,
				"project_name": project,
				"objective": objective,
				"model": model,
				"workers": workers,
				"max_rounds": max_rounds,
				"budget": budget,
			},
		)

	@app.post("/launch/execute", response_class=HTMLResponse)
	async def launch_execute(request: Request):
		assert _state is not None
		form = await request.form()
		project_name = str(form.get("project", ""))
		overrides: dict[str, str] = {}
		if form.get("max_rounds"):
			overrides["max_rounds"] = str(form["max_rounds"])
		if form.get("workers"):
			overrides["workers"] = str(form["workers"])

		try:
			_state.launcher.launch(project_name, config_overrides=overrides)
			# Redirect to project dashboard
			return templates.TemplateResponse(
				"project_dashboard.html",
				{"request": request, "project_name": project_name},
				headers={"HX-Push-Url": f"/project/{project_name}"},
			)
		except Exception as e:
			return templates.TemplateResponse(
				"partials/toast.html",
				{"request": request, "message": f"Launch failed: {e}", "level": "error"},
			)

	# -- Control endpoints --

	@app.post("/project/{name}/stop", response_class=HTMLResponse)
	async def stop_mission(request: Request, name: str):
		assert _state is not None
		try:
			result = _state.launcher.stop(name)
			if result:
				msg, level = "Stop signal sent", "success"
			else:
				msg, level = "No running mission to stop", "warning"
		except Exception as e:
			msg, level = f"Error: {e}", "error"
		return templates.TemplateResponse(
			"partials/toast.html",
			{"request": request, "message": msg, "level": level},
		)

	@app.post("/project/{name}/retry/{unit_id}", response_class=HTMLResponse)
	async def retry_unit(request: Request, name: str, unit_id: str):
		assert _state is not None
		try:
			result = _state.launcher.retry_unit(name, unit_id)
			if result:
				msg, level = f"Retry signal sent for {unit_id[:8]}", "success"
			else:
				msg, level = "Could not send retry signal", "warning"
		except Exception as e:
			msg, level = f"Error: {e}", "error"
		return templates.TemplateResponse(
			"partials/toast.html",
			{"request": request, "message": msg, "level": level},
		)

	@app.post("/project/{name}/adjust", response_class=HTMLResponse)
	async def adjust_mission(request: Request, name: str):
		assert _state is not None
		form = await request.form()
		params: dict[str, int | float | str] = {}
		if form.get("max_rounds"):
			params["max_rounds"] = int(form["max_rounds"])
		if form.get("num_workers"):
			params["num_workers"] = int(form["num_workers"])

		try:
			result = _state.launcher.adjust(name, params)
			if result:
				msg, level = f"Adjusted: {params}", "success"
			else:
				msg, level = "No running mission to adjust", "warning"
		except Exception as e:
			msg, level = f"Error: {e}", "error"
		return templates.TemplateResponse(
			"partials/toast.html",
			{"request": request, "message": msg, "level": level},
		)

	return app


def _get_first_provider_snap():
	"""Get snapshot from first available provider (legacy compat)."""
	if not _state:
		return None
	for p in _state.registry.list_projects():
		provider = _state.get_provider(p.name)
		if provider:
			return provider.get_snapshot()
	return None


def _get_current_units(project_name: str):
	"""Get work units for the current round of a project."""
	if not _state:
		return []
	project = _state.registry.get_project(project_name)
	if not project:
		return []
	db_path = Path(project.db_path)
	if not db_path.exists():
		return []
	try:
		from mission_control.db import Database
		db = Database(db_path)
		try:
			mission = db.get_latest_mission()
			if not mission:
				return []
			rounds = db.get_rounds_for_mission(mission.id)
			if not rounds:
				return []
			current = rounds[-1]
			if current.plan_id:
				return db.get_work_units_for_plan(current.plan_id)
		finally:
			db.close()
	except Exception:
		return []
	return []


def _get_work_unit(project_name: str, unit_id: str):
	"""Get a single work unit by ID."""
	if not _state:
		return None
	project = _state.registry.get_project(project_name)
	if not project:
		return None
	db_path = Path(project.db_path)
	if not db_path.exists():
		return None
	try:
		from mission_control.db import Database
		db = Database(db_path)
		try:
			return db.get_work_unit(unit_id)
		finally:
			db.close()
	except Exception:
		return None
