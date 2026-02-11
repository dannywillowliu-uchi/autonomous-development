"""CLI interface for mission-control."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="mc",
		description="Mission Control - autonomous development scheduler",
	)
	sub = parser.add_subparsers(dest="command")

	# mc start
	start = sub.add_parser("start", help="Start the scheduler")
	start.add_argument("--config", default="mission-control.toml", help="Config file path")
	start.add_argument("--max-sessions", type=int, default=None, help="Max sessions to run")
	start.add_argument("--dry-run", action="store_true", help="Show plan without executing")

	# mc status
	status = sub.add_parser("status", help="Show current status")
	status.add_argument("--config", default="mission-control.toml")

	# mc history
	history = sub.add_parser("history", help="Show session history")
	history.add_argument("--config", default="mission-control.toml")
	history.add_argument("--limit", type=int, default=10)

	# mc snapshot
	snap = sub.add_parser("snapshot", help="Take a project health snapshot")
	snap.add_argument("--config", default="mission-control.toml")

	# mc init
	init = sub.add_parser("init", help="Initialize a mission-control config")
	init.add_argument("path", nargs="?", default=".")

	return parser


def main(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	if args.command is None:
		parser.print_help()
		return 0

	# Commands implemented in later phases
	print(f"Command '{args.command}' not yet implemented.")
	return 1


if __name__ == "__main__":
	sys.exit(main())
