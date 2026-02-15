"""CLI interface for mission-control."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from mission_control.config import load_config, validate_config
from mission_control.db import Database
from mission_control.scheduler import Scheduler

DEFAULT_CONFIG = "mission-control.toml"
DEFAULT_DB = "mission-control.db"


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="mc",
		description="Mission Control - autonomous development scheduler",
	)
	sub = parser.add_subparsers(dest="command")

	# mc start
	start = sub.add_parser("start", help="Start the scheduler")
	start.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	start.add_argument("--max-sessions", type=int, default=None, help="Max sessions to run")
	start.add_argument("--dry-run", action="store_true", help="Show plan without executing")

	# mc status
	status = sub.add_parser("status", help="Show current status")
	status.add_argument("--config", default=DEFAULT_CONFIG)

	# mc history
	history = sub.add_parser("history", help="Show session history")
	history.add_argument("--config", default=DEFAULT_CONFIG)
	history.add_argument("--limit", type=int, default=10)

	# mc snapshot
	snap = sub.add_parser("snapshot", help="Take a project health snapshot")
	snap.add_argument("--config", default=DEFAULT_CONFIG)

	# mc parallel
	par = sub.add_parser("parallel", help="Run parallel execution mode")
	par.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	par.add_argument("--workers", type=int, default=None, help="Number of workers")
	par.add_argument("--dry-run", action="store_true", help="Show plan without executing")

	# mc mission
	mission = sub.add_parser("mission", help="Run continuous mission mode")
	mission.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	mission.add_argument("--workers", type=int, default=None, help="Number of workers")
	mission.add_argument("--dry-run", action="store_true", help="Show mission plan without executing")
	mission.add_argument(
		"--auto-discover", action="store_true",
		help="Run discovery before mission, use results as objective",
	)
	mission.add_argument(
		"--approve-all", action="store_true",
		help="Auto-approve all discovery items (skip checkpoint)",
	)

	# mc discover
	discover = sub.add_parser("discover", help="Run codebase discovery analysis")
	discover.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
	discover.add_argument("--dry-run", action="store_true", help="Show what would be analyzed")
	discover.add_argument("--json", action="store_true", dest="json_output", help="Output raw JSON")
	discover.add_argument("--latest", action="store_true", help="Show latest discovery from DB")

	# mc init
	init_cmd = sub.add_parser("init", help="Initialize a mission-control config")
	init_cmd.add_argument("path", nargs="?", default=".")

	# mc dashboard
	dash = sub.add_parser("dashboard", help="Launch TUI dashboard")
	dash.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

	# mc live
	live = sub.add_parser("live", help="Launch live mission control dashboard")
	live.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
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

	# mc validate-config
	vc = sub.add_parser("validate-config", help="Validate config file semantically")
	vc.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

	return parser


def _get_db_path(config_path: str) -> Path:
	"""Derive database path from config path location."""
	return Path(config_path).parent / DEFAULT_DB


def cmd_start(args: argparse.Namespace) -> int:
	"""Run the scheduler."""
	config = load_config(args.config)
	db_path = _get_db_path(args.config)

	if args.dry_run:
		print(f"Target: {config.target.name} ({config.target.resolved_path})")
		print(f"Model: {config.scheduler.model}")
		print(f"Max sessions: {args.max_sessions or config.scheduler.max_sessions_per_run}")
		print(f"Session timeout: {config.scheduler.session_timeout}s")
		print(f"Budget: ${config.scheduler.budget.max_per_run_usd}/run")
		print(f"Database: {db_path}")
		return 0

	db = Database(db_path)
	try:
		scheduler = Scheduler(config, db)
		report = asyncio.run(scheduler.run(max_sessions=args.max_sessions))
		print(f"Sessions run: {report.sessions_run}")
		print(f"Helped: {report.sessions_helped}, Hurt: {report.sessions_hurt}, Neutral: {report.sessions_neutral}")
		print(f"Stopped: {report.stopped_reason}")
		return 0
	finally:
		db.close()


def cmd_status(args: argparse.Namespace) -> int:
	"""Show current project status."""
	load_config(args.config)  # validate config exists
	db_path = _get_db_path(args.config)

	if not db_path.exists():
		print("No database found. Run 'mc start' first.")
		return 1

	db = Database(db_path)
	try:
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
	finally:
		db.close()


def cmd_history(args: argparse.Namespace) -> int:
	"""Show session history."""
	db_path = _get_db_path(args.config)

	if not db_path.exists():
		print("No database found. Run 'mc start' first.")
		return 1

	db = Database(db_path)
	try:
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
	finally:
		db.close()


def cmd_snapshot(args: argparse.Namespace) -> int:
	"""Take a project health snapshot."""
	from mission_control.state import snapshot_project_health

	config = load_config(args.config)
	snapshot = asyncio.run(snapshot_project_health(config))
	print(f"Tests: {snapshot.test_passed}/{snapshot.test_total} passing ({snapshot.test_failed} failed)")
	print(f"Lint errors: {snapshot.lint_errors}")
	print(f"Type errors: {snapshot.type_errors}")
	print(f"Security findings: {snapshot.security_findings}")
	return 0


def cmd_parallel(args: argparse.Namespace) -> int:
	"""Run the parallel coordinator."""
	from mission_control.coordinator import Coordinator

	config = load_config(args.config)
	db_path = _get_db_path(args.config)

	num_workers = args.workers or config.scheduler.parallel.num_workers

	if args.dry_run:
		print(f"Target: {config.target.name} ({config.target.resolved_path})")
		print(f"Objective: {config.target.objective}")
		print(f"Model: {config.scheduler.model}")
		print(f"Workers: {num_workers}")
		print(f"Pool dir: {config.scheduler.parallel.pool_dir or '(auto)'}")
		print(f"Session timeout: {config.scheduler.session_timeout}s")
		print(f"Budget: ${config.scheduler.budget.max_per_session_usd}/session")
		print(f"Database: {db_path}")
		return 0

	db = Database(db_path)
	try:
		coordinator = Coordinator(config, db, num_workers=num_workers)
		report = asyncio.run(coordinator.run())
		print(f"Plan: {report.plan_id}")
		print(f"Units: {report.total_units} total, {report.units_completed} completed, {report.units_failed} failed")
		print(f"Merged: {report.units_merged}, Rejected: {report.units_rejected}")
		print(f"Workers: {report.workers_spawned}")
		print(f"Wall time: {report.wall_time_seconds:.1f}s")
		print(f"Stopped: {report.stopped_reason}")
		return 0
	finally:
		db.close()


def cmd_discover(args: argparse.Namespace) -> int:
	"""Run codebase discovery analysis."""
	import json as json_mod

	from mission_control.auto_discovery import DiscoveryEngine

	config = load_config(args.config)
	db_path = _get_db_path(args.config)

	if args.dry_run:
		dc = config.discovery
		print(f"Target: {config.target.name} ({config.target.resolved_path})")
		print(f"Tracks: {', '.join(dc.tracks)}")
		print(f"Model: {dc.model}")
		print(f"Budget: ${dc.budget_per_call_usd}/call")
		print(f"Min priority: {dc.min_priority_score}")
		print(f"Max items/track: {dc.max_items_per_track}")
		return 0

	db = Database(db_path)
	try:
		if args.latest:
			result, items = db.get_latest_discovery()
			if result is None:
				print("No discovery results found.")
				return 1
		else:
			engine = DiscoveryEngine(config, db)
			result, items = asyncio.run(engine.discover())

		if args.json_output:
			data = {
				"id": result.id,
				"timestamp": result.timestamp,
				"model": result.model,
				"item_count": result.item_count,
				"items": [
					{
						"track": i.track,
						"title": i.title,
						"description": i.description,
						"rationale": i.rationale,
						"files_hint": i.files_hint,
						"impact": i.impact,
						"effort": i.effort,
						"priority_score": i.priority_score,
						"status": i.status,
					}
					for i in items
				],
			}
			print(json_mod.dumps(data, indent=2))
		else:
			_print_discovery_table(items)

		return 0
	finally:
		db.close()


def _print_discovery_table(items: list) -> None:
	"""Print discovery items in a formatted table."""
	if not items:
		print("No items discovered.")
		return

	print(f"\n{'Track':<10} {'Title':<40} {'Priority':>8} {'Impact':>7} {'Effort':>7}")
	print("-" * 75)
	for i in items:
		title = i.title[:38] + ".." if len(i.title) > 40 else i.title
		print(
			f"{i.track:<10} {title:<40} {i.priority_score:>8.1f} "
			f"{i.impact:>7} {i.effort:>7}"
		)
	print(f"\nTotal: {len(items)} items")


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
		print(f"Backlog min size: {cont.backlog_min_size}")
		print(f"Planner max depth: {config.planner.max_depth}")
		gb = config.green_branch
		print(f"Green branch: {gb.green_branch}")
		print(f"Backend: {config.backend.type}")
		print(f"Session timeout: {config.scheduler.session_timeout}s")
		per_session = config.scheduler.budget.max_per_session_usd
		per_run = config.scheduler.budget.max_per_run_usd
		print(f"Budget: ${per_session}/session, ${per_run}/run")
		print(f"Database: {db_path}")

		if config.target.objective:
			db = Database(db_path)
			try:
				from mission_control.continuous_controller import ContinuousController
				controller = ContinuousController(config, db)
				asyncio.run(controller.run(dry_run=True))
			finally:
				db.close()
		return 0

	# Auto-discover mode: run discovery, then use results as objective
	if args.auto_discover:
		from mission_control.auto_discovery import DiscoveryEngine

		db = Database(db_path)
		try:
			engine = DiscoveryEngine(config, db)
			result, items = asyncio.run(engine.discover())

			if not items:
				print("Discovery found no actionable items.")
				return 0

			_print_discovery_table(items)

			if not args.approve_all:
				print("\nReview the items above.")
				answer = input(
					"Approve all and start mission? [y/N] "
				).strip().lower()
				if answer != "y":
					print("Mission cancelled.")
					return 0

			# Mark all as approved
			for item in items:
				db.update_discovery_item_status(item.id, "approved")

			# Compose objective from approved items
			config.target.objective = engine.compose_objective(items)
			print(f"\nComposed objective from {len(items)} items.")
		finally:
			db.close()

	if not config.target.objective:
		print(
			"Error: target.objective must be set in config "
			"for mission mode (or use --auto-discover)."
		)
		return 1

	db = Database(db_path)
	try:
		from mission_control.continuous_controller import ContinuousController
		controller = ContinuousController(config, db)
		result = asyncio.run(controller.run())
		print(f"Mission: {result.mission_id}")
		print(f"Objective met: {result.objective_met}")
		disp = result.total_units_dispatched
		merged = result.total_units_merged
		failed = result.total_units_failed
		print(f"Units: {disp} dispatched, {merged} merged, {failed} failed")
		if result.final_verification_passed is not None:
			status = "PASS" if result.final_verification_passed else "FAIL"
			print(f"Final verification: {status}")
		print(f"Wall time: {result.wall_time_seconds:.1f}s")
		print(f"Stopped: {result.stopped_reason}")
		return 0 if result.objective_met else 1
	finally:
		db.close()


def cmd_summary(args: argparse.Namespace) -> int:
	"""Show mission summary."""
	import json as json_mod

	db_path = _get_db_path(args.config)
	if not db_path.exists():
		print("No database found. Run 'mc mission' first.")
		return 1

	db = Database(db_path)
	try:
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
	finally:
		db.close()


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
max_depth = 3
max_children_per_node = 5

[green_branch]
working_branch = "mc/working"
green_branch = "mc/green"
fixup_max_attempts = 3

[continuous]
max_wall_time_seconds = 7200
backlog_min_size = 2
timeout_multiplier = 1.2

[discovery]
enabled = true
tracks = ["feature", "quality", "security"]
max_items_per_track = 3
min_priority_score = 3.0
model = "opus"
budget_per_call_usd = 2.0

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
		print("No database found. Run 'mc mission' first.")
		return 1

	dashboard = LiveDashboard(db_path)
	print(f"Starting live dashboard at http://localhost:{args.port}")
	uvicorn.run(dashboard.app, host="0.0.0.0", port=args.port, log_level="warning")
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


COMMANDS = {
	"start": cmd_start,
	"status": cmd_status,
	"history": cmd_history,
	"snapshot": cmd_snapshot,
	"parallel": cmd_parallel,
	"discover": cmd_discover,
	"mission": cmd_mission,
	"init": cmd_init,
	"dashboard": cmd_dashboard,
	"live": cmd_live,
	"summary": cmd_summary,

	"register": cmd_register,
	"unregister": cmd_unregister,
	"projects": cmd_projects,
	"mcp": cmd_mcp,
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

	handler = COMMANDS.get(args.command)
	if handler is None:
		print(f"Unknown command: {args.command}")
		return 1

	return handler(args)


if __name__ == "__main__":
	sys.exit(main())
