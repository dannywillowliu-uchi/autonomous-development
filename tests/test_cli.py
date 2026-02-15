"""Tests for CLI argument parsing and commands."""

from __future__ import annotations

from pathlib import Path

from mission_control.cli import build_parser, cmd_discover, cmd_init, cmd_parallel, main


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


class TestParallelArgs:
	def test_parallel_defaults(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["parallel"])
		assert args.command == "parallel"
		assert args.config == "mission-control.toml"
		assert args.workers is None
		assert args.dry_run is False

	def test_parallel_with_flags(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["parallel", "--workers", "8", "--dry-run", "--config", "c.toml"])
		assert args.workers == 8
		assert args.dry_run is True
		assert args.config == "c.toml"

	def test_parallel_dry_run(self, tmp_path: Path) -> None:
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text("""\
[target]
name = "test"
path = "/tmp/test"
objective = "fix things"

[target.verification]
command = "pytest"

[scheduler]
model = "sonnet"
""")
		parser = build_parser()
		args = parser.parse_args(["parallel", "--config", str(config_file), "--dry-run"])
		result = cmd_parallel(args)
		assert result == 0


class TestDiscoverArgs:
	def test_discover_defaults(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["discover"])
		assert args.command == "discover"
		assert args.config == "mission-control.toml"
		assert args.dry_run is False
		assert args.json_output is False
		assert args.latest is False

	def test_discover_with_flags(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["discover", "--json", "--config", "c.toml"])
		assert args.json_output is True
		assert args.config == "c.toml"

	def test_discover_dry_run(self, tmp_path: Path) -> None:
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[discovery]
model = "sonnet"
tracks = ["feature", "quality"]
""")
		parser = build_parser()
		args = parser.parse_args(["discover", "--config", str(config_file), "--dry-run"])
		result = cmd_discover(args)
		assert result == 0

	def test_discover_latest_empty(self, tmp_path: Path) -> None:
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text("""\
[target]
name = "test"
path = "/tmp/test"
""")
		parser = build_parser()
		args = parser.parse_args(["discover", "--config", str(config_file), "--latest"])
		result = cmd_discover(args)
		assert result == 1  # No discoveries yet


class TestMissionAutoDiscoverArgs:
	def test_mission_auto_discover_flag(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--auto-discover"])
		assert args.auto_discover is True
		assert args.approve_all is False

	def test_mission_approve_all_flag(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--auto-discover", "--approve-all"])
		assert args.auto_discover is True
		assert args.approve_all is True
