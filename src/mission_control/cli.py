"""CLI interface for mission-control."""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from mission_control.config import MissionConfig, load_config, validate_config
from mission_control.db import Database
from mission_control.models import Mission

DEFAULT_CONFIG = "mission-control.toml"
DEFAULT_DB = "mission-control.db"


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="mc",
		description="Mission Control - autonomous development scheduler",
	)
	sub = parser.add_subparsers(dest="command")

	# mc status
	status = sub.add_parser("status", help="Show current status")
	status.add_argument("--config", default=DEFAULT_CONFIG)

	# mc history
	history = sub.add_parser("history", help="Show session history")
	history.add_argument("--config", default=DEFAULT_CONFIG)
	history.add_argument("--limit", type=int, default=10)

	# mc mission
	mission = sub.add_parser("mission", help="Run continuous mission mode")
	mission.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	mission.add_argument("--workers", type=int, default=None, help="Number of workers")
	mission.add_argument("--dry-run", action="store_true", help="Show mission plan without executing")
	mission.add_argument(
		"--approve-all", action="store_true",
		help="Auto-approve proposed objective (skip checkpoint)",
	)
	mission.add_argument(
		"--experiment", action="store_true",
		help="Run in experiment mode: units produce comparison reports instead of merged commits",
	)
	mission.add_argument(
		"--chain", action="store_true",
		help="Automatically chain missions when next_objective is set",
	)
	mission.add_argument(
		"--max-chain-depth", type=int, default=3,
		help="Maximum number of chained missions (default: 3)",
	)
	mission.add_argument(
		"--no-dashboard", action="store_true",
		help="Disable the live dashboard (enabled by default)",
	)
	mission.add_argument(
		"--dashboard-port", type=int, default=8080,
		help="Port for the live dashboard (default: 8080)",
	)

	# mc init
	init_cmd = sub.add_parser("init", help="Initialize a mission-control config")
	init_cmd.add_argument("path", nargs="?", default=".")

	# mc dashboard
	dash = sub.add_parser("dashboard", help="Launch TUI dashboard")
	dash.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

	# mc live
	live = sub.add_parser("live", help="Launch live mission control dashboard")
	live.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	live.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
	live.add_argument("--port", type=int, default=8080, help="Port to serve on")

	# mc summary
	summary = sub.add_parser("summary", help="Show mission summary")
	summary.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	summary.add_argument("--all", action="store_true", dest="show_all", help="Show table of all missions")
	summary.add_argument("--mission-id", default=None, help="Specific mission ID")
	summary.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

	# mc register
	reg = sub.add_parser("register", help="Register a project in the central registry")
	reg.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	reg.add_argument("--name", default=None, help="Project alias (defaults to target.name)")
	reg.add_argument("--description", default="", help="Project description")

	# mc unregister
	unreg = sub.add_parser("unregister", help="Remove a project from the registry")
	unreg.add_argument("name", help="Project name to unregister")

	# mc projects
	sub.add_parser("projects", help="List all registered projects")

	# mc mcp
	sub.add_parser("mcp", help="Start the MCP server (stdio)")

	# mc a2a
	a2a = sub.add_parser("a2a", help="Start the A2A protocol server")
	a2a.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

	# mc validate-config
	vc = sub.add_parser("validate-config", help="Validate config file semantically")
	vc.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

	# mc intel
	intel = sub.add_parser("intel", help="Scan external sources for AI/agent ecosystem intelligence")
	intel.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
	intel.add_argument("--threshold", type=float, default=0.3, help="Relevance threshold for proposals (default: 0.3)")

	return parser


def _find_config_path(config_path: str) -> Path:
	"""Resolve config path to absolute, searching upward if not found in CWD."""
	p = Path(config_path)
	if p.is_absolute():
		return p.resolve()
	# Check CWD first
	candidate = Path.cwd() / p
	if candidate.exists():
		return candidate.resolve()
	# Search upward for the config file
	current = Path.cwd()
	while True:
		candidate = current / p.name
		if candidate.exists():
			return candidate.resolve()
		parent = current.parent
		if parent == current:
			break
		current = parent
	# Fallback: resolve relative to CWD even if it doesn't exist yet
	return (Path.cwd() / p).resolve()


def _get_db_path(config_path: str) -> Path:
	"""Derive database path from config path location."""
	resolved = _find_config_path(config_path)
	return resolved.parent / DEFAULT_DB


def cmd_status(args: argparse.Namespace) -> int:
	"""Show current project status."""
	load_config(args.config)  # validate config exists
	db_path = _get_db_path(args.config)

	if not db_path.exists():
		print("No database found. Run 'mc start' first.")
		return 1

	with Database(db_path) as db:
		latest = db.get_latest_snapshot()
		if latest:
			print(f"Tests: {latest.test_passed}/{latest.test_total} passing")
			print(f"Lint errors: {latest.lint_errors}")
			print(f"Type errors: {latest.type_errors}")
			print(f"Security findings: {latest.security_findings}")
		else:
			print("No snapshots yet.")

		recent = db.get_recent_sessions(3)
		if recent:
			print(f"\nRecent sessions ({len(recent)}):")
			for s in recent:
				print(f"  [{s.status}] {s.id}: {s.task_description}")
		return 0


def cmd_history(args: argparse.Namespace) -> int:
	"""Show session history."""
	db_path = _get_db_path(args.config)

	if not db_path.exists():
		print("No database found. Run 'mc start' first.")
		return 1

	with Database(db_path) as db:
		sessions = db.get_recent_sessions(args.limit)
		if not sessions:
			print("No sessions yet.")
			return 0

		for s in sessions:
			status_icon = {"completed": "+", "failed": "x", "reverted": "!"}.get(s.status, "?")
			print(f"[{status_icon}] {s.id} | {s.task_description} | {s.status}")
			if s.output_summary:
				print(f"    {s.output_summary[:100]}")
		return 0


def _is_cleanup_mission(mission: Mission) -> bool:
	"""Check if a mission is a cleanup mission by convention prefix."""
	return mission.objective.startswith("[CLEANUP]")


def _is_cleanup_due(db: Database, interval: int) -> bool:
	"""Check if a cleanup mission is due based on completed non-cleanup missions since last cleanup."""
	missions = db.get_all_missions(limit=interval + 5)
	count_since_cleanup = 0
	for m in missions:
		if m.status not in ("completed", "stopped"):
			continue
		if _is_cleanup_mission(m):
			break
		count_since_cleanup += 1
	return count_since_cleanup >= interval


def _build_cleanup_objective(config: MissionConfig) -> str:
	"""Build a cleanup mission objective with current test suite metrics."""
	target_path = str(config.target.resolved_path)
	num_files = 0
	num_tests = 0

	try:
		result = subprocess.run(
			["find", "tests", "-name", "test_*.py"],
			capture_output=True, text=True, cwd=target_path, timeout=10,
		)
		if result.returncode == 0:
			num_files = len([line for line in result.stdout.strip().splitlines() if line])
	except (subprocess.TimeoutExpired, OSError):
		pass

	try:
		result = subprocess.run(
			["pytest", "--co", "-q"],
			capture_output=True, text=True, cwd=target_path, timeout=30,
		)
		if result.returncode == 0:
			# Last line of pytest --co -q is "N tests collected"
			for line in reversed(result.stdout.strip().splitlines()):
				if "test" in line:
					parts = line.split()
					if parts and parts[0].isdigit():
						num_tests = int(parts[0])
					break
	except (subprocess.TimeoutExpired, OSError):
		pass

	return (
		f"[CLEANUP] Consolidate test suite: {num_files} files, {num_tests} tests. "
		"Merge test files for the same module, consolidate small test classes, "
		"remove dead code. All tests must pass."
	)


def _generate_dashboard_token() -> str:
	"""Generate a random auth token for the dashboard."""
	import secrets
	return secrets.token_urlsafe(32)


def _start_dashboard_background(
	db_path: Path, port: int,
) -> tuple[object, object]:
	"""Start the live dashboard in a background thread and open the browser.

	The LiveDashboard (and its Database connection) must be created inside
	the background thread because SQLite objects are not thread-safe.
	"""
	import threading
	import webbrowser

	import uvicorn

	# Ensure the DB file exists before the dashboard tries to read it
	if not db_path.exists():
		db = Database(db_path)
		db.close()

	auth_token = _generate_dashboard_token()

	# We need the server ref to signal shutdown, but it's created in the thread.
	# Use a list as a mutable container to pass it back.
	server_ref: list[uvicorn.Server | None] = [None]
	ready = threading.Event()

	def _run_dashboard() -> None:
		from mission_control.dashboard.live import LiveDashboard

		dashboard = LiveDashboard(str(db_path), auth_token=auth_token)
		config = uvicorn.Config(
			dashboard.app, host="127.0.0.1", port=port, log_level="warning",
		)
		server = uvicorn.Server(config)
		server_ref[0] = server
		ready.set()
		server.run()

	thread = threading.Thread(target=_run_dashboard, daemon=True)
	thread.start()
	ready.wait(timeout=5)

	url = f"http://127.0.0.1:{port}?token={auth_token}"
	print(f"Live dashboard: {url}")

	# Write token URL to well-known file so it can be retrieved when stdout is unavailable
	url_file = db_path.parent / ".mc-dashboard-url"
	try:
		url_file.write_text(url + "\n")
	except OSError:
		pass

	# Give the server a moment to bind, then open browser
	def _open_browser() -> None:
		import time
		time.sleep(1.5)
		webbrowser.open(url)

	threading.Thread(target=_open_browser, daemon=True).start()

	return thread, server_ref[0]


def cmd_mission(args: argparse.Namespace) -> int:
	"""Run the continuous mission mode (outer loop)."""
	config = load_config(args.config)
	db_path = _get_db_path(args.config)

	if args.workers is not None:
		config.scheduler.parallel.num_workers = args.workers

	if args.dry_run:
		print(f"Target: {config.target.name} ({config.target.resolved_path})")
		print(f"Objective: {config.target.objective}")
		print(f"Model: {config.scheduler.model}")
		print(f"Workers: {config.scheduler.parallel.num_workers}")
		cont = config.continuous
		print(f"Max wall time: {cont.max_wall_time_seconds}s")
		print(f"Planner budget: ${config.planner.budget_per_call_usd}/call")
		gb = config.green_branch
		print(f"Green branch: {gb.green_branch}")
		print(f"Backend: {config.backend.type}")
		print(f"Session timeout: {config.scheduler.session_timeout}s")
		per_session = config.scheduler.budget.max_per_session_usd
		per_run = config.scheduler.budget.max_per_run_usd
		print(f"Budget: ${per_session}/session, ${per_run}/run")
		print(f"Database: {db_path}")

		if config.target.objective:
			with Database(db_path) as db:
				from mission_control.continuous_controller import ContinuousController
				controller = ContinuousController(config, db)
				asyncio.run(controller.run(dry_run=True))
		return 0

	# Experiment mode: prepend instruction to objective for planner
	if args.experiment and config.target.objective:
		config.target.objective = (
			"[EXPERIMENT MODE] "
			"All units in this mission are experiments. "
			"Each unit should try multiple approaches (default 2), benchmark each, "
			"and report which is better with data. "
			"Produce comparison reports, not merged commits. "
			"Set experiment_mode=True on all work units. "
			+ config.target.objective
		)

	if not config.target.objective:
		print(
			"Error: target.objective must be set in config."
		)
		return 1

	from mission_control.continuous_controller import ContinuousController

	# Start live dashboard in background thread (default: on)
	dashboard_thread = None
	dashboard_server = None
	if not args.no_dashboard:
		dashboard_thread, dashboard_server = _start_dashboard_background(
			db_path, args.dashboard_port,
		)

	from mission_control.models import _new_id
	chain_id = _new_id() if args.chain else ""

	chain_depth = 0
	max_chain_depth = getattr(args, "max_chain_depth", 3)
	last_result = None

	try:
		while True:
			# Check if a cleanup mission is due (only when chaining is enabled)
			if config.continuous.cleanup_enabled and args.chain:
				with Database(db_path) as db:
					if _is_cleanup_due(db, config.continuous.cleanup_interval):
						cleanup_obj = _build_cleanup_objective(config)
						config.target.objective = cleanup_obj
						interval = config.continuous.cleanup_interval
						print(f"\n--- Cleanup mission triggered (every {interval} missions) ---")
						print(f"Objective: {cleanup_obj}")

			with Database(db_path) as db:
				controller = ContinuousController(config, db, chain_id=chain_id)
				result = asyncio.run(controller.run())

			print(f"Mission: {result.mission_id}")
			print(f"Objective met: {result.objective_met}")
			disp = result.total_units_dispatched
			merged = result.total_units_merged
			failed = result.total_units_failed
			print(f"Units: {disp} dispatched, {merged} merged, {failed} failed")
			if result.final_verification_passed is not None:
				fv_status = "PASS" if result.final_verification_passed else "FAIL"
				print(f"Final verification: {fv_status}")
			print(f"Wall time: {result.wall_time_seconds:.1f}s")
			print(f"Stopped: {result.stopped_reason}")

			last_result = result
			chain_depth += 1

			if not args.chain:
				break
			if chain_depth >= max_chain_depth:
				print(f"Chain depth limit reached ({max_chain_depth}). Stopping.")
				break
			# Use deliberative planner's critic to propose next objective
			from mission_control.deliberative_planner import DeliberativePlanner
			with Database(db_path) as chain_db:
				chain_planner = DeliberativePlanner(config, chain_db)
				obj, rationale = asyncio.run(
					chain_planner.propose_next_objective(
						Mission(objective=config.target.objective),
						last_result,
					)
				)
			if not obj:
				print("Critic did not propose a next objective. Stopping chain.")
				break
			print(f"\n--- Chaining mission {chain_depth + 1}/{max_chain_depth} ---")
			print(f"Next objective: {obj}")
			if rationale:
				print(f"Rationale: {rationale[:200]}")
			if not args.approve_all:
				answer = input("Approve this objective? [y/N] ").strip().lower()
				if answer != "y":
					print("Objective rejected. Stopping chain.")
					break
			config.target.objective = obj
	finally:
		if dashboard_server is not None:
			dashboard_server.should_exit = True
		if dashboard_thread is not None:
			dashboard_thread.join(timeout=5)

	return 0 if last_result.objective_met else 1


def cmd_summary(args: argparse.Namespace) -> int:
	"""Show mission summary."""
	import json as json_mod

	db_path = _get_db_path(args.config)
	if not db_path.exists():
		print("No database found. Run 'mc mission' first.")
		return 1

	with Database(db_path) as db:
		if args.show_all:
			missions = db.get_all_missions()
			if not missions:
				print("No missions found.")
				return 0
			if args.json_output:
				data = [
					{
						"id": m.id, "objective": m.objective[:60],
						"status": m.status, "started_at": m.started_at,
						"total_cost_usd": m.total_cost_usd,
					}
					for m in missions
				]
				print(json_mod.dumps(data, indent=2))
			else:
				print(f"\n{'ID':<12} {'Status':<10} {'Cost':>8} {'Objective':<40}")
				print("-" * 72)
				for m in missions:
					obj = m.objective[:38] + ".." if len(m.objective) > 40 else m.objective
					print(f"{m.id[:10]:<12} {m.status:<10} ${m.total_cost_usd:>6.2f} {obj:<40}")
			return 0

		# Single mission
		if args.mission_id:
			mission = db.get_mission(args.mission_id)
		else:
			mission = db.get_latest_mission()

		if mission is None:
			print("No mission found.")
			return 1

		summary = db.get_mission_summary(mission.id)

		if args.json_output:
			data = {
				"id": mission.id,
				"objective": mission.objective,
				"status": mission.status,
				"started_at": mission.started_at,
				"finished_at": mission.finished_at,
				"total_cost_usd": mission.total_cost_usd,
				"stopped_reason": mission.stopped_reason,
				**summary,
			}
			print(json_mod.dumps(data, indent=2))
			return 0

		# Narrative output
		print(f"\nMission: {mission.id}")
		print(f"Status: {mission.status}")
		print(f"Objective: {mission.objective}")
		print(f"Cost: ${mission.total_cost_usd:.2f}")
		if mission.started_at and mission.finished_at:
			print(f"Started: {mission.started_at}")
			print(f"Finished: {mission.finished_at}")
		if mission.stopped_reason:
			print(f"Stopped: {mission.stopped_reason}")

		# Unit counts
		units = summary["units_by_status"]
		if units:
			print("\nUnits by status:")
			for status, count in sorted(units.items()):
				print(f"  {status}: {count}")

		# Epoch breakdown
		epochs = summary["epochs"]
		if epochs:
			print(f"\n{'Epoch':>5} {'Planned':>8} {'Done':>8} {'Failed':>8}")
			print("-" * 31)
			for ep in epochs:
				print(
					f"{ep['number']:>5} {ep['units_planned']:>8} "
					f"{ep['units_completed']:>8} {ep['units_failed']:>8}"
				)

		# Event distribution
		events = summary["events_by_type"]
		if events:
			print("\nEvents:")
			for etype, count in sorted(events.items()):
				print(f"  {etype}: {count}")

		# Token usage
		if summary["total_input_tokens"] or summary["total_output_tokens"]:
			inp = summary["total_input_tokens"]
			out = summary["total_output_tokens"]
			print(f"\nTokens: {inp:,} input, {out:,} output")

		return 0


INIT_TEMPLATE = """\
[target]
name = "{name}"
path = "{path}"
branch = "main"
objective = ""

[target.verification]
command = "uv run pytest -q && uv run ruff check src/"
timeout = 300

[scheduler]
session_timeout = 2700
cooldown = 60
max_sessions_per_run = 10
model = "sonnet"

[scheduler.git]
strategy = "branch-per-session"
auto_merge = false

[scheduler.budget]
max_per_session_usd = 5.0
max_per_run_usd = 200.0

[scheduler.parallel]
num_workers = 8

[rounds]
max_rounds = 20
stall_threshold = 3
cooldown_between_rounds = 30

[planner]

[green_branch]
working_branch = "mc/working"
green_branch = "mc/green"
fixup_max_attempts = 3

[continuous]
max_wall_time_seconds = 7200
timeout_multiplier = 1.2

[heartbeat]
interval = 300
idle_threshold = 3

[notifications.telegram]
# bot_token and chat_id read from env vars TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
on_heartbeat = true
on_merge_fail = true
on_mission_end = true

[backend]
type = "local"
"""


def cmd_init(args: argparse.Namespace) -> int:
	"""Initialize a mission-control config."""
	target = Path(args.path).resolve()
	config_path = target / DEFAULT_CONFIG

	if config_path.exists():
		print(f"Config already exists: {config_path}")
		return 1

	config_path.write_text(INIT_TEMPLATE.format(
		name=target.name,
		path=str(target),
	))
	print(f"Created {config_path}")
	return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
	"""Launch the TUI dashboard."""
	try:
		from mission_control.dashboard.tui import DashboardApp
	except ImportError:
		print("Dashboard dependencies not installed. Run: pip install -e '.[dashboard]'")
		return 1

	db_path = _get_db_path(args.config)
	if not db_path.exists():
		print("No database found. Run 'mc start' or 'mc mission' first.")
		return 1

	app = DashboardApp(db_path=str(db_path))
	app.run()
	return 0


def cmd_live(args: argparse.Namespace) -> int:
	"""Launch the live mission control dashboard."""
	try:
		import uvicorn

		from mission_control.dashboard.live import LiveDashboard
	except ImportError:
		print("Dashboard dependencies not installed. Run: pip install -e '.[dashboard]'")
		return 1

	db_path = str(_get_db_path(args.config))
	if not Path(db_path).exists():
		print("No existing database -- creating empty one")
		Database(db_path)

	auth_token = _generate_dashboard_token()
	dashboard = LiveDashboard(db_path, auth_token=auth_token)
	url = f"http://{args.host}:{args.port}?token={auth_token}"
	print(f"Starting live dashboard at {url}")

	# Write token URL to well-known file so it can be retrieved when stdout is unavailable
	url_file = Path(db_path).parent / ".mc-dashboard-url"
	try:
		url_file.write_text(url + "\n")
	except OSError:
		pass

	uvicorn.run(dashboard.app, host=args.host, port=args.port, log_level="warning")
	return 0


def cmd_register(args: argparse.Namespace) -> int:
	"""Register a project in the central registry."""
	from mission_control.registry import ProjectRegistry

	config = load_config(args.config)
	name = args.name or config.target.name

	registry = ProjectRegistry()
	try:
		db_path = str(_get_db_path(args.config))
		project = registry.register(
			name=name,
			config_path=str(Path(args.config).resolve()),
			db_path=db_path,
			description=args.description,
		)
		print(f"Registered '{project.name}' -> {project.config_path}")
		return 0
	except ValueError as e:
		print(f"Error: {e}")
		return 1
	finally:
		registry.close()


def cmd_unregister(args: argparse.Namespace) -> int:
	"""Remove a project from the registry."""
	from mission_control.registry import ProjectRegistry

	registry = ProjectRegistry()
	try:
		if registry.unregister(args.name):
			print(f"Unregistered '{args.name}'")
			return 0
		else:
			print(f"Project '{args.name}' not found")
			return 1
	finally:
		registry.close()


def cmd_projects(args: argparse.Namespace) -> int:
	"""List all registered projects with status."""
	from mission_control.registry import ProjectRegistry

	registry = ProjectRegistry()
	try:
		projects = registry.list_projects()
		if not projects:
			print("No projects registered. Use 'mc register' to add one.")
			return 0

		for p in projects:
			status = registry.get_project_status(p.name)
			mission_info = ""
			if status and status.mission_status != "idle":
				mission_info = (
					f" | {status.mission_status}"
					f" | score={status.mission_score:.1f}"
					f" | rounds={status.mission_rounds}"
					f" | ${status.mission_cost:.2f}"
				)
			pid_info = f" (PID {p.active_pid})" if p.active_pid else ""
			print(f"  {p.name}{pid_info}: {p.config_path}{mission_info}")
		return 0
	finally:
		registry.close()


def cmd_validate_config(args: argparse.Namespace) -> int:
	"""Validate config file semantically."""
	config = load_config(args.config)
	issues = validate_config(config)

	errors = [(lvl, msg) for lvl, msg in issues if lvl == "error"]
	warnings = [(lvl, msg) for lvl, msg in issues if lvl == "warning"]

	for level, msg in issues:
		print(f"[{level.upper()}] {msg}")

	if not issues:
		print("Config OK")

	print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
	return 1 if errors else 0


def cmd_mcp(args: argparse.Namespace) -> int:
	"""Start the MCP server."""
	try:
		from mission_control.mcp_server import run_mcp_server
	except ImportError:
		print("MCP dependencies not installed. Run: pip install -e '.[mcp]'")
		return 1

	run_mcp_server()
	return 0


def cmd_a2a(args: argparse.Namespace) -> int:
	"""Start the A2A protocol server."""
	import signal

	try:
		from mission_control.a2a import A2AServer
	except ImportError:
		print("FastAPI/uvicorn not installed. Run: pip install mission-control[dashboard]")
		return 1

	config = load_config(args.config)
	db_path = _get_db_path(args.config)

	with Database(db_path) as db:
		server = A2AServer(config.a2a, db)
		server.start()
		print(f"A2A server running on {config.a2a.host}:{config.a2a.port}")
		print("Press Ctrl-C to stop.")

		stop_event = threading.Event()
		original_handler = signal.getsignal(signal.SIGINT)

		def _handle_sigint(signum: int, frame: Any) -> None:
			stop_event.set()

		signal.signal(signal.SIGINT, _handle_sigint)
		try:
			stop_event.wait()
		finally:
			signal.signal(signal.SIGINT, original_handler)
			server.stop()

	return 0




def cmd_intel(args: argparse.Namespace) -> int:
	"""Scan external sources for AI/agent ecosystem intelligence."""
	import dataclasses
	import json as json_mod

	from mission_control.intelligence import run_scan

	try:
		report = asyncio.run(run_scan(threshold=args.threshold))
	except Exception as exc:
		print(f"Intel scan failed: {exc}")
		return 1

	if args.json_output:
		print(json_mod.dumps(dataclasses.asdict(report), indent=2))
	else:
		print(f"Intel Report ({report.timestamp})")
		print(f"Sources: {', '.join(report.sources_scanned)}")
		print(f"Scan duration: {report.scan_duration_seconds}s")
		print(f"Findings: {len(report.findings)}, Proposals: {len(report.proposals)}")
		if report.findings:
			print(f"\n{'Score':>6}  {'Source':<12} {'Title'}")
			print("-" * 72)
			for f in report.findings[:20]:
				title = f.title[:50] + ".." if len(f.title) > 52 else f.title
				print(f"{f.relevance_score:6.2f}  {f.source:<12} {title}")
		if report.proposals:
			print(f"\n{'Pri':>3}  {'Type':<14} {'Effort':<8} {'Title'}")
			print("-" * 72)
			for p in report.proposals:
				title = p.title[:45] + ".." if len(p.title) > 47 else p.title
				print(f"{p.priority:>3}  {p.proposal_type:<14} {p.effort_estimate:<8} {title}")

	return 0


COMMANDS = {
	"status": cmd_status,
	"history": cmd_history,
	"mission": cmd_mission,
	"init": cmd_init,
	"dashboard": cmd_dashboard,
	"live": cmd_live,
	"summary": cmd_summary,
	"intel": cmd_intel,

	"register": cmd_register,
	"unregister": cmd_unregister,
	"projects": cmd_projects,
	"mcp": cmd_mcp,
	"a2a": cmd_a2a,
	"validate-config": cmd_validate_config,
}


def main(argv: list[str] | None = None) -> int:
	# Force line-buffered stderr for nohup/redirect scenarios
	import sys
	if hasattr(sys.stderr, "reconfigure"):
		sys.stderr.reconfigure(line_buffering=True)  # type: ignore[union-attr]
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
		datefmt="%H:%M:%S",
		force=True,
	)
	parser = build_parser()
	args = parser.parse_args(argv)

	if args.command is None:
		parser.print_help()
		return 0

	if hasattr(args, "config"):
		from mission_control.path_security import validate_config_path

		try:
			validate_config_path(args.config, [Path.home()])
		except ValueError as e:
			print(f"Error: {e}")
			return 1

	handler = COMMANDS.get(args.command)
	if handler is None:
		print(f"Unknown command: {args.command}")
		return 1

	return handler(args)


if __name__ == "__main__":
	sys.exit(main())
