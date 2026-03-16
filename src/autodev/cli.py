"""CLI interface for autodev."""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from autodev.config import MissionConfig, load_config, validate_config
from autodev.db import Database
from autodev.diagnose import run_diagnose
from autodev.models import Mission

DEFAULT_CONFIG = "autodev.toml"
DEFAULT_DB = "autodev.db"


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="autodev",
		description="Automatic Development - autonomous development scheduler",
	)
	sub = parser.add_subparsers(dest="command")

	# autodev status
	status = sub.add_parser("status", help="Show current status")
	status.add_argument("--config", default=DEFAULT_CONFIG)

	# autodev history
	history = sub.add_parser("history", help="Show session history")
	history.add_argument("--config", default=DEFAULT_CONFIG)
	history.add_argument("--limit", type=int, default=10)

	# autodev mission
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

	# autodev swarm
	swarm = sub.add_parser("swarm", help="Run swarm mode with driving planner")
	swarm.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	swarm.add_argument("--max-agents", type=int, default=None, help="Override max agents (0=unbounded)")
	swarm.add_argument("--planner-model", default=None, help="Override planner model")
	swarm.add_argument(
		"--no-dashboard", action="store_true",
		help="Disable the live dashboard",
	)
	swarm.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard port")

	# autodev swarm-tui
	swarm_tui = sub.add_parser("swarm-tui", help="Live TUI dashboard for swarm monitoring")
	swarm_tui.add_argument("path", nargs="?", default=".", help="Project path (default: cwd)")

	# autodev swarm-inject
	swarm_inject = sub.add_parser("swarm-inject", help="Send a directive to a running swarm planner")
	swarm_inject.add_argument("message", help="Directive message for the planner")
	swarm_inject.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

	# autodev init
	init_cmd = sub.add_parser("init", help="Initialize a autodev config")
	init_cmd.add_argument("path", nargs="?", default=".")

	# autodev dashboard
	dash = sub.add_parser("dashboard", help="Launch TUI dashboard")
	dash.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

	# autodev live
	live = sub.add_parser("live", help="Launch live automatic development dashboard")
	live.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	live.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
	live.add_argument("--port", type=int, default=8080, help="Port to serve on")

	# autodev summary
	summary = sub.add_parser("summary", help="Show mission summary")
	summary.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	summary.add_argument("--all", action="store_true", dest="show_all", help="Show table of all missions")
	summary.add_argument("--mission-id", default=None, help="Specific mission ID")
	summary.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

	# autodev register
	reg = sub.add_parser("register", help="Register a project in the central registry")
	reg.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	reg.add_argument("--name", default=None, help="Project alias (defaults to target.name)")
	reg.add_argument("--description", default="", help="Project description")

	# autodev unregister
	unreg = sub.add_parser("unregister", help="Remove a project from the registry")
	unreg.add_argument("name", help="Project name to unregister")

	# autodev projects
	sub.add_parser("projects", help="List all registered projects")

	# autodev mcp
	sub.add_parser("mcp", help="Start the MCP server (stdio)")

	# autodev a2a
	a2a = sub.add_parser("a2a", help="Start the A2A protocol server")
	a2a.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

	# autodev trace
	trace = sub.add_parser("trace", help="Read trace file and print human-readable timeline")
	trace.add_argument("file", nargs="?", default="trace.jsonl", help="Trace JSONL file (default: trace.jsonl)")
	trace.add_argument("--worker", default=None, help="Filter by worker ID")
	trace.add_argument("--type", default=None, dest="event_type", help="Filter by event type")
	trace.add_argument("--json", action="store_true", dest="json_output", help="Output raw JSON instead of formatted")

	# autodev diagnose
	diag = sub.add_parser("diagnose", help="Run operational health checks")
	diag.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	diag.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

	# autodev validate-config
	vc = sub.add_parser("validate-config", help="Validate config file semantically")
	vc.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

	# autodev intel
	intel = sub.add_parser("intel", help="Scan external sources for AI/agent ecosystem intelligence")
	intel.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
	intel.add_argument("--threshold", type=float, default=0.3, help="Relevance threshold for proposals (default: 0.3)")

	# autodev auto-update
	auto_update = sub.add_parser("auto-update", help="Scan for improvements and auto-apply")
	auto_update.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	auto_update.add_argument("--dry-run", action="store_true", help="Show proposals without launching")
	auto_update.add_argument("--approve-all", action="store_true", help="Skip approval for high-risk")
	auto_update.add_argument("--threshold", type=float, default=0.3, help="Relevance threshold (default: 0.3)")
	auto_update.add_argument("--daemon", action="store_true", help="Run as daemon with recurring update cycles")
	auto_update.add_argument("--interval", type=float, default=24.0, help="Hours between daemon cycles (default: 24.0)")

	# autodev trace-review
	trace_review = sub.add_parser("trace-review", help="Review agent traces from swarm runs")
	trace_review.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	trace_review.add_argument("--last", action="store_true", help="Review most recent run")
	trace_review.add_argument("--run-id", default=None, help="Review a specific run by ID")
	trace_review.add_argument("--history", action="store_true", help="Review across multiple runs")
	trace_review.add_argument("--runs", type=int, default=5, help="Number of runs for history (default: 5)")
	trace_review.add_argument("--deep", action="store_true", help="Enable LLM-assisted review")

	# autodev contrib
	contrib = sub.add_parser("contrib", help="Multi-contributor coordination")
	contrib.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	contrib.add_argument("--username", default=None, help="Contributor username (default: git user.name)")
	contrib_sub = contrib.add_subparsers(dest="contrib_command")

	contrib_sub.add_parser("register", help="Register as contributor")

	contrib_sub.add_parser("proposals", help="List available proposals")

	contrib_claim = contrib_sub.add_parser("claim", help="Claim a proposal")
	contrib_claim.add_argument("proposal_id", help="Proposal ID to claim")

	contrib_publish = contrib_sub.add_parser("publish", help="Publish results for a proposal")
	contrib_publish.add_argument("proposal_id", help="Proposal ID")
	contrib_publish.add_argument("--commit", default="", help="Commit hash")
	contrib_publish.add_argument("--tests-before", type=int, default=0, help="Tests passing before")
	contrib_publish.add_argument("--tests-after", type=int, default=0, help="Tests passing after")
	contrib_publish.add_argument("--outcome", default="", help="Outcome: keep/discard/crash")

	contrib_sub.add_parser("sync", help="Sync shared learnings")

	# autodev metrics
	metrics = sub.add_parser("metrics", help="Show swarm performance metrics and trends")
	metrics.add_argument("--trend", action="store_true", help="Show trend analysis across recent runs")
	metrics.add_argument("--correlate", action="store_true", help="Correlate metrics with self-modifications")
	metrics.add_argument(
		"--last-n", type=int, default=10,
		help="Number of recent runs for trend analysis (default: 10)",
	)
	metrics.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

	# autodev tool-usage
	tool_usage = sub.add_parser("tool-usage", help="Show tool usage summary")
	tool_usage.add_argument("--config", default=DEFAULT_CONFIG)
	tool_usage.add_argument("--run-id", default=None, help="Filter by run ID")
	tool_usage.add_argument("--failures-only", action="store_true", help="Show only failed tool calls")
	tool_usage.add_argument("--top", type=int, default=20, help="Show top N most-used tools")

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
		print("No database found. Run 'autodev start' first.")
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
		print("No database found. Run 'autodev start' first.")
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
		from autodev.dashboard.live import LiveDashboard

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
	url_file = db_path.parent / ".autodev-dashboard-url"
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
				from autodev.continuous_controller import ContinuousController
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

	from autodev.continuous_controller import ContinuousController

	# Start live dashboard in background thread (default: on)
	dashboard_thread = None
	dashboard_server = None
	if not args.no_dashboard:
		dashboard_thread, dashboard_server = _start_dashboard_background(
			db_path, args.dashboard_port,
		)

	from autodev.models import _new_id
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
			from autodev.deliberative_planner import DeliberativePlanner
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
		print("No database found. Run 'autodev mission' first.")
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
working_branch = "autodev/working"
green_branch = "autodev/green"
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

[swarm]
enabled = false
planner_model = "opus"
planner_cooldown = 10
max_agents = 0
min_agents = 2
stagnation_threshold = 3
inherit_global_mcps = true
"""


def cmd_init(args: argparse.Namespace) -> int:
	"""Initialize a autodev config."""
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
		from autodev.dashboard.tui import DashboardApp
	except ImportError:
		print("Dashboard dependencies not installed. Run: pip install -e '.[dashboard]'")
		return 1

	db_path = _get_db_path(args.config)
	if not db_path.exists():
		print("No database found. Run 'autodev start' or 'autodev mission' first.")
		return 1

	app = DashboardApp(db_path=str(db_path))
	app.run()
	return 0


def cmd_live(args: argparse.Namespace) -> int:
	"""Launch the live automatic development dashboard."""
	try:
		import uvicorn

		from autodev.dashboard.live import LiveDashboard
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
	url_file = Path(db_path).parent / ".autodev-dashboard-url"
	try:
		url_file.write_text(url + "\n")
	except OSError:
		pass

	uvicorn.run(dashboard.app, host=args.host, port=args.port, log_level="warning")
	return 0


def cmd_register(args: argparse.Namespace) -> int:
	"""Register a project in the central registry."""
	from autodev.registry import ProjectRegistry

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
	from autodev.registry import ProjectRegistry

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
	from autodev.registry import ProjectRegistry

	registry = ProjectRegistry()
	try:
		projects = registry.list_projects()
		if not projects:
			print("No projects registered. Use 'autodev register' to add one.")
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
		from autodev.mcp_server import run_mcp_server
	except ImportError:
		print("MCP dependencies not installed. Run: pip install -e '.[mcp]'")
		return 1

	run_mcp_server()
	return 0


def cmd_a2a(args: argparse.Namespace) -> int:
	"""Start the A2A protocol server."""
	import signal

	try:
		from autodev.a2a import A2AServer
	except ImportError:
		print("FastAPI/uvicorn not installed. Run: pip install autodev[dashboard]")
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




def cmd_trace(args: argparse.Namespace) -> int:
	"""Read a trace JSONL file and print a human-readable timeline."""
	import json as json_mod
	from collections import defaultdict
	from datetime import datetime, timezone

	from autodev.trace_log import TraceEvent

	trace_path = Path(args.file)
	if not trace_path.exists():
		print(f"Error: trace file not found: {trace_path}")
		return 1

	# Parse all events
	events: list[TraceEvent] = []
	with open(trace_path) as f:
		for line_num, line in enumerate(f, 1):
			line = line.strip()
			if not line:
				continue
			try:
				data = json_mod.loads(line)
				events.append(TraceEvent.from_dict(data))
			except (json_mod.JSONDecodeError, TypeError) as exc:
				print(f"Warning: skipping invalid line {line_num}: {exc}")

	if not events:
		print("No trace events found.")
		return 0

	# Apply filters
	if args.worker:
		events = [e for e in events if e.worker_id == args.worker]
	if args.event_type:
		events = [e for e in events if e.event_type == args.event_type]

	if not events:
		print("No events match the given filters.")
		return 0

	# JSON passthrough mode
	if args.json_output:
		for e in events:
			print(json_mod.dumps(e.to_dict()))
		return 0

	# Parse the earliest timestamp as the baseline for relative times
	def _parse_ts(ts: str) -> datetime:
		try:
			return datetime.fromisoformat(ts)
		except ValueError:
			return datetime.now(timezone.utc)

	t0 = _parse_ts(events[0].timestamp)

	# Group events by unit_id
	groups: dict[str, list[TraceEvent]] = defaultdict(list)
	for e in events:
		key = e.unit_id or "(no unit)"
		groups[key].append(e)

	for unit_id, unit_events in groups.items():
		print(f"\n== Unit: {unit_id} ==")
		for e in unit_events:
			dt = _parse_ts(e.timestamp)
			rel = (dt - t0).total_seconds()
			details_str = ""
			if e.details:
				parts = [f"{k}={v}" for k, v in e.details.items()]
				details_str = " " + " ".join(parts)
			print(f"  +{rel:>8.1f}s  {e.event_type:<20} worker={e.worker_id}{details_str}")

	return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
	"""Run operational health checks on the autodev installation."""
	import json as json_mod

	config_path = _find_config_path(args.config)
	db_path = _get_db_path(args.config)
	report = run_diagnose(config_path, db_path)

	if args.json_output:
		print(json_mod.dumps(report, indent=2))
		return 0

	for section in report["checks"]:
		status = section["status"]
		tag = {"OK": "OK", "WARN": "WARN", "ERROR": "ERROR"}.get(status, status)
		print(f"[{tag:>5}] {section['name']}")
		for detail in section.get("details", []):
			print(f"         {detail}")
		if section.get("remediation"):
			print(f"         -> {section['remediation']}")

	print()
	statuses = [c["status"] for c in report["checks"]]
	if "ERROR" in statuses:
		print("Overall: ERROR")
		return 1
	elif "WARN" in statuses:
		print("Overall: WARN")
	else:
		print("Overall: OK")
	return 0


def cmd_intel(args: argparse.Namespace) -> int:
	"""Scan external sources for AI/agent ecosystem intelligence."""
	import dataclasses
	import json as json_mod

	from autodev.intelligence import run_scan

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


def cmd_swarm(args: argparse.Namespace) -> int:
	"""Run swarm mode with driving planner."""
	config = load_config(args.config)
	db_path = _get_db_path(args.config)

	if not config.target.objective:
		print("Error: target.objective must be set in config.")
		return 1

	# Enable swarm and apply overrides
	config.swarm.enabled = True
	if args.max_agents is not None:
		config.swarm.max_agents = args.max_agents
	if args.planner_model is not None:
		config.swarm.planner_model = args.planner_model

	print(f"Swarm mode: {config.target.name}")
	print(f"Objective: {config.target.objective}")
	print(f"Planner model: {config.swarm.planner_model}")
	max_display = config.swarm.max_agents if config.swarm.max_agents > 0 else "unbounded"
	print(f"Max agents: {max_display}")

	# Start dashboard
	dashboard_thread = None
	dashboard_server = None
	if not args.no_dashboard:
		dashboard_thread, dashboard_server = _start_dashboard_background(
			db_path, args.dashboard_port,
		)

	try:
		with Database(db_path) as db:
			from autodev.swarm.controller import SwarmController
			from autodev.swarm.planner import DrivingPlanner

			controller = SwarmController(config, config.swarm, db)
			planner = DrivingPlanner(controller, config.swarm)

			async def _run() -> None:
				await controller.initialize()

				# Core test runner if configured
				core_test_runner = None
				if config.core_tests.enabled:
					from autodev.core_tests import _parse_results

					ct = config.core_tests
					proj = config.target.resolved_path

					def _core_test_runner():
						import subprocess as _sp
						try:
							_sp.run(
								ct.runner_command,
								shell=True,  # nosec B602
								cwd=proj,
								capture_output=True,
								text=True,
								timeout=ct.timeout,
							)
							# Find results.json
							import json as _json
							for candidate in [
								Path(proj) / "results.json",
								Path(proj) / "tests" / "torture" / "results.json",
							]:
								if candidate.exists():
									data = _json.loads(candidate.read_text())
									r = _parse_results(data)
									return {
										"pass": r.passed,
										"fail": r.failed,
										"skip": r.skipped,
										"total": r.total,
									}
						except Exception:
							pass
						return None

					core_test_runner = _core_test_runner

				await planner.run(core_test_runner=core_test_runner)

			asyncio.run(_run())
	finally:
		if dashboard_server is not None:
			dashboard_server.should_exit = True
		if dashboard_thread is not None:
			dashboard_thread.join(timeout=5)

	print("Swarm completed.")
	return 0


def cmd_swarm_inject(args: argparse.Namespace) -> int:
	"""Send a directive to a running swarm planner via the team-lead inbox."""
	import fcntl
	import json
	import tempfile
	from datetime import datetime, timezone

	config = load_config(args.config)
	team_name = f"autodev-{config.target.name}"
	inbox_path = Path.home() / ".claude" / "teams" / team_name / "inboxes" / "team-lead.json"

	if not inbox_path.parent.exists():
		print(f"Error: no swarm inbox found at {inbox_path.parent}")
		print("Is the swarm running?")
		return 1

	message = {
		"type": "directive",
		"from": "human",
		"text": args.message,
		"timestamp": datetime.now(timezone.utc).isoformat(),
	}

	lock_path = inbox_path.with_suffix(".lock")
	try:
		with open(lock_path, "w") as lock_file:
			fcntl.flock(lock_file, fcntl.LOCK_EX)
			try:
				messages = json.loads(inbox_path.read_text()) if inbox_path.exists() else []
				if not isinstance(messages, list):
					messages = []
			except (json.JSONDecodeError, OSError):
				messages = []
			messages.append(message)
			tmp_fd, tmp_path = tempfile.mkstemp(dir=str(inbox_path.parent), suffix=".tmp")
			try:
				import os
				os.write(tmp_fd, json.dumps(messages, indent=2).encode())
				os.close(tmp_fd)
				tmp_fd = None
				os.rename(tmp_path, str(inbox_path))
			except BaseException:
				if tmp_fd is not None:
					os.close(tmp_fd)
				if Path(tmp_path).exists():
					os.unlink(tmp_path)
				raise
	except OSError as e:
		print(f"Error writing to inbox: {e}")
		return 1

	print(f"Directive sent to swarm planner: \"{args.message}\"")
	print("The planner will see this on its next cycle.")
	return 0


def cmd_auto_update(args: argparse.Namespace) -> int:
	"""Scan for improvements and auto-apply via swarm missions."""
	config = load_config(args.config)
	db_path = _get_db_path(args.config)

	if args.daemon:
		from autodev.scheduler import AutoUpdateScheduler

		with Database(db_path) as db:
			scheduler = AutoUpdateScheduler(config, db, interval_hours=args.interval)
			try:
				asyncio.run(scheduler.run_forever())
			except KeyboardInterrupt:
				scheduler.stop()
		return 0

	with Database(db_path) as db:
		from autodev.auto_update import AutoUpdatePipeline

		pipeline = AutoUpdatePipeline(config, db)
		results = asyncio.run(pipeline.run(
			dry_run=args.dry_run,
			approve_all=args.approve_all,
			threshold=args.threshold,
		))

	if not results:
		print("No new proposals to process.")
		return 0

	print(f"\n{'Action':<10} {'Risk':<6} {'Title'}")
	print("-" * 72)
	for r in results:
		title = r.title[:50] + ".." if len(r.title) > 52 else r.title
		print(f"{r.action:<10} {r.risk_level:<6} {title}")
		if r.mission_id:
			print(f"           Mission: {r.mission_id}")

	launched = sum(1 for r in results if r.action in ("launched", "approved"))
	skipped = sum(1 for r in results if r.action in ("rejected", "skipped"))
	print(f"\nLaunched: {launched}, Skipped: {skipped}")
	return 0


def cmd_swarm_tui(args: argparse.Namespace) -> int:
	"""Launch the swarm TUI dashboard."""
	from autodev.swarm.tui import main as tui_main
	tui_main(project_path=args.path)
	return 0


def cmd_trace_review(args: argparse.Namespace) -> int:
	"""Review agent traces from swarm runs."""
	config = load_config(args.config)
	db_path = _get_db_path(args.config)

	if not db_path.exists():
		print("No database found. Run a swarm first.")
		return 1

	from autodev.trace_review import TraceAnalyzer

	with Database(db_path) as db:
		analyzer = TraceAnalyzer(db, config.target.resolved_path)

		if args.history:
			analysis = asyncio.run(analyzer.analyze_history(args.runs))
			report = asyncio.run(analyzer.generate_report(analysis))
			print(report)
			return 0

		run_id = args.run_id
		if args.last:
			traces = db.get_agent_traces(limit=1)
			if not traces:
				print("No traces found.")
				return 1
			run_id = traces[0]["run_id"]

		if not run_id:
			print("Specify --last, --run-id RUN_ID, or --history")
			return 1

		analysis = asyncio.run(analyzer.analyze_run(run_id))
		report = asyncio.run(analyzer.generate_report(analysis))
		print(report)

		if args.deep:
			prompt = asyncio.run(analyzer.llm_review_traces(run_id))
			print("\n## LLM Review Prompt\n")
			print(prompt)

		return 0


def _resolve_contrib_username(args: argparse.Namespace, repo_path: Path) -> str:
	"""Resolve contributor username from args or git config."""
	if args.username:
		return args.username
	result = subprocess.run(
		["git", "config", "user.name"],
		capture_output=True, text=True, cwd=str(repo_path),
	)
	name = result.stdout.strip()
	if not name:
		print("Error: no --username provided and git user.name is not set")
		sys.exit(1)
	return name


def cmd_contrib(args: argparse.Namespace) -> int:
	"""Multi-contributor coordination commands."""
	if not args.contrib_command:
		print("Usage: autodev contrib {register,proposals,claim,publish,sync}")
		return 1

	config = load_config(args.config)
	repo_path = config.target.resolved_path

	from autodev.contrib import ContributorProtocol

	username = _resolve_contrib_username(args, repo_path)
	protocol = ContributorProtocol(repo_path, username)

	if args.contrib_command == "register":
		info = asyncio.run(protocol.register())
		print(f"Registered: {info.username} (joined: {info.joined_at})")
		return 0

	if args.contrib_command == "proposals":
		proposals = asyncio.run(protocol.list_proposals())
		if not proposals:
			print("No available proposals.")
			return 0
		for p in proposals:
			pid = p.get("proposal_id", "?")
			status = p.get("status", "available")
			print(f"  {pid}: {status}")
		return 0

	if args.contrib_command == "claim":
		success = asyncio.run(protocol.claim_proposal(args.proposal_id))
		if success:
			print(f"Claimed: {args.proposal_id}")
			return 0
		print(f"Failed to claim: {args.proposal_id} (already claimed?)")
		return 1

	if args.contrib_command == "publish":
		from datetime import datetime, timezone

		result = {
			"commit": args.commit,
			"tests_before": args.tests_before,
			"tests_after": args.tests_after,
			"outcome": args.outcome,
			"proposal_title": args.proposal_id,
			"duration_s": "",
			"cost_usd": "",
			"timestamp": datetime.now(timezone.utc).isoformat(),
		}
		asyncio.run(protocol.publish_result(args.proposal_id, result))
		print(f"Published result for: {args.proposal_id}")
		return 0

	if args.contrib_command == "sync":
		learnings = asyncio.run(protocol.sync_learnings())
		if learnings:
			print(learnings)
		else:
			print("No shared learnings found.")
		return 0

	return 1


def cmd_metrics(args: argparse.Namespace) -> int:
	"""Show swarm performance metrics and trends."""
	import json as json_mod

	from autodev.metrics import MetricsTracker

	config = load_config(args.config)
	tracker = MetricsTracker(config.target.resolved_path)

	if args.trend:
		result = tracker.get_trend(args.last_n)
		if result.get("error") == "insufficient_data":
			print(f"Insufficient data: only {result['rows']} run(s) recorded (need at least 2).")
			return 1
		# Format SwarmMetrics objects for JSON serialization
		for key in ("best_run", "worst_run"):
			if key in result and hasattr(result[key], "__dict__"):
				from dataclasses import asdict
				result[key] = asdict(result[key])
		print(json_mod.dumps(result, indent=2))
		return 0

	if args.correlate:
		results = tracker.correlate_with_modifications()
		if not results:
			print("No correlation data available.")
			return 0
		print(json_mod.dumps(results, indent=2))
		return 0

	# Default: print latest metrics from TSV
	rows = tracker._read_rows()
	if not rows:
		print("No metrics recorded yet. Run a swarm first.")
		return 0

	latest = rows[-1]
	print(f"Run: {latest.run_id} ({latest.timestamp})")
	print(f"Tests: {latest.test_count} (pass rate: {latest.test_pass_rate:.1%})")
	print(f"Cost: ${latest.total_cost_usd:.2f} (${latest.cost_per_task:.2f}/task)")
	print(f"Agent success rate: {latest.agent_success_rate:.1%}")
	print(f"Duration: {latest.total_duration_s:.0f}s")
	print(f"Tasks: {latest.tasks_completed} completed, {latest.tasks_failed} failed")
	return 0


def cmd_tool_usage(args: argparse.Namespace) -> int:
	"""Show tool usage summary from swarm runs."""
	db_path = _get_db_path(args.config)
	db = Database(str(db_path))

	if args.failures_only:
		failures = db.get_tool_failure_summary(run_id=args.run_id)
		if not failures:
			print("No tool failures found.")
			return 0
		print(f"{'Tool':<40} {'Failures':>8}  Last Error")
		print("-" * 80)
		for f in failures:
			print(f"{f['tool_name']:<40} {f['failure_count']:>8}  {f['last_error'][:60]}")
		return 0

	usage = db.get_tool_usage(run_id=args.run_id, limit=args.top)
	if not usage:
		print("No tool usage data found.")
		return 0
	print(f"{'Tool':<40} {'Server':<15} {'Status':>7} {'Duration':>10}")
	print("-" * 80)
	for u in usage:
		status = "OK" if u["success"] else "FAIL"
		print(f"{u['tool_name']:<40} {u.get('mcp_server', ''):<15} {status:>7} {u.get('duration_ms', 0):>8.0f}ms")
	return 0


COMMANDS = {
	"status": cmd_status,
	"history": cmd_history,
	"mission": cmd_mission,
	"swarm": cmd_swarm,
	"swarm-inject": cmd_swarm_inject,
	"swarm-tui": cmd_swarm_tui,
	"init": cmd_init,
	"dashboard": cmd_dashboard,
	"live": cmd_live,
	"summary": cmd_summary,
	"trace": cmd_trace,
	"intel": cmd_intel,
	"auto-update": cmd_auto_update,
	"diagnose": cmd_diagnose,
	"trace-review": cmd_trace_review,
	"contrib": cmd_contrib,

	"register": cmd_register,
	"unregister": cmd_unregister,
	"projects": cmd_projects,
	"mcp": cmd_mcp,
	"a2a": cmd_a2a,
	"validate-config": cmd_validate_config,
	"metrics": cmd_metrics,
	"tool-usage": cmd_tool_usage,
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
		from autodev.path_security import validate_config_path

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
