"""FastAPI web dashboard for mission-control monitoring."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mission_control.dashboard.provider import DashboardProvider

_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

_provider: DashboardProvider | None = None


def create_app(db_path: str) -> FastAPI:
	"""Factory: build the dashboard FastAPI app."""

	@asynccontextmanager
	async def lifespan(app: FastAPI):
		global _provider
		_provider = DashboardProvider(db_path)
		_provider.start_polling(interval=2.0)
		yield
		_provider.stop()
		_provider = None

	app = FastAPI(title="Mission Control Dashboard", lifespan=lifespan)

	templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

	app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

	# -- Full page --

	@app.get("/", response_class=HTMLResponse)
	async def index(request: Request):
		return templates.TemplateResponse("index.html", {"request": request})

	# -- HTMX partials --

	@app.get("/partials/mission-header", response_class=HTMLResponse)
	async def mission_header(request: Request):
		snap = _provider.get_snapshot() if _provider else None
		return templates.TemplateResponse(
			"partials/mission_header.html",
			{"request": request, "snap": snap},
		)

	@app.get("/partials/worker-pool", response_class=HTMLResponse)
	async def worker_pool(request: Request):
		snap = _provider.get_snapshot() if _provider else None
		return templates.TemplateResponse(
			"partials/worker_pool.html",
			{"request": request, "snap": snap},
		)

	@app.get("/partials/work-units", response_class=HTMLResponse)
	async def work_units(request: Request):
		snap = _provider.get_snapshot() if _provider else None
		return templates.TemplateResponse(
			"partials/work_units.html",
			{"request": request, "snap": snap},
		)

	@app.get("/partials/merge-queue", response_class=HTMLResponse)
	async def merge_queue(request: Request):
		snap = _provider.get_snapshot() if _provider else None
		return templates.TemplateResponse(
			"partials/merge_queue.html",
			{"request": request, "snap": snap},
		)

	@app.get("/partials/activity-log", response_class=HTMLResponse)
	async def activity_log(request: Request):
		snap = _provider.get_snapshot() if _provider else None
		return templates.TemplateResponse(
			"partials/activity_log.html",
			{"request": request, "snap": snap},
		)

	# -- JSON API for charts --

	@app.get("/api/score-history")
	async def api_score_history():
		snap = _provider.get_snapshot() if _provider else None
		if not snap or not snap.score_history:
			return JSONResponse({"labels": [], "data": []})
		labels = [f"R{r}" for r, _ in snap.score_history]
		data = [s for _, s in snap.score_history]
		return JSONResponse({"labels": labels, "data": data})

	@app.get("/api/test-trend")
	async def api_test_trend():
		snap = _provider.get_snapshot() if _provider else None
		if not snap or not snap.test_trend:
			return JSONResponse({"labels": [], "passed": [], "failed": []})
		labels = [f"R{r}" for r, _, _ in snap.test_trend]
		passed = [p for _, p, _ in snap.test_trend]
		failed = [f for _, _, f in snap.test_trend]
		return JSONResponse({"labels": labels, "passed": passed, "failed": failed})

	@app.get("/api/cost-per-round")
	async def api_cost_per_round():
		snap = _provider.get_snapshot() if _provider else None
		if not snap or not snap.cost_per_round:
			return JSONResponse({"labels": [], "data": []})
		labels = [f"R{r}" for r, _ in snap.cost_per_round]
		data = [c for _, c in snap.cost_per_round]
		return JSONResponse({"labels": labels, "data": data})

	return app
