"""Tests for CLI argument parsing and commands."""

from __future__ import annotations

from pathlib import Path

from mission_control.cli import build_parser, cmd_init, main


class TestArgParsing:
	def test_start_defaults(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["start"])
		assert args.command == "start"
		assert args.config == "mission-control.toml"
		assert args.max_sessions is None
		assert args.dry_run is False

	def test_start_with_flags(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["start", "--config", "custom.toml", "--max-sessions", "5", "--dry-run"])
		assert args.config == "custom.toml"
		assert args.max_sessions == 5
		assert args.dry_run is True

	def test_history_limit(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["history", "--limit", "20"])
		assert args.command == "history"
		assert args.limit == 20

	def test_init_default_path(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["init"])
		assert args.command == "init"
		assert args.path == "."

	def test_init_custom_path(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["init", "/tmp/myproject"])
		assert args.path == "/tmp/myproject"

	def test_no_command_returns_0(self) -> None:
		result = main([])
		assert result == 0


class TestCmdInit:
	def test_creates_config(self, tmp_path: Path) -> None:
		parser = build_parser()
		args = parser.parse_args(["init", str(tmp_path)])
		result = cmd_init(args)
		assert result == 0
		config_file = tmp_path / "mission-control.toml"
		assert config_file.exists()
		content = config_file.read_text()
		assert tmp_path.name in content

	def test_refuses_overwrite(self, tmp_path: Path) -> None:
		(tmp_path / "mission-control.toml").write_text("existing")
		parser = build_parser()
		args = parser.parse_args(["init", str(tmp_path)])
		result = cmd_init(args)
		assert result == 1
