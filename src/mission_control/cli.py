"""CLI interface for mission-control."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mission_control.config import load_config
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

	# mc init
	init_cmd = sub.add_parser("init", help="Initialize a mission-control config")
	init_cmd.add_argument("path", nargs="?", default=".")

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
max_per_run_usd = 50.0
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


COMMANDS = {
	"start": cmd_start,
	"status": cmd_status,
	"history": cmd_history,
	"snapshot": cmd_snapshot,
	"init": cmd_init,
}


def main(argv: list[str] | None = None) -> int:
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
