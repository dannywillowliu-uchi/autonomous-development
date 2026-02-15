"""Tests for CLI argument parsing and commands."""

from __future__ import annotations

from pathlib import Path

from mission_control.cli import build_parser, cmd_discover, cmd_init, cmd_parallel, cmd_summary, main
from mission_control.db import Database
from mission_control.models import Epoch, Mission, Plan, WorkUnit


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


class TestSummaryArgs:
	def test_summary_defaults(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["summary"])
		assert args.command == "summary"
		assert args.show_all is False
		assert args.mission_id is None
		assert args.json_output is False

	def test_summary_all(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["summary", "--all"])
		assert args.show_all is True

	def test_summary_json(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["summary", "--json"])
		assert args.json_output is True

	def test_summary_mission_id(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["summary", "--mission-id", "abc123"])
		assert args.mission_id == "abc123"


class TestCmdSummary:
	def _setup_db(self, tmp_path: Path) -> tuple[Path, Path]:
		"""Create config and DB with test data. Returns (config_path, db_path)."""
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text('[target]\nname = "test"\npath = "/tmp/test"\n')
		db_path = tmp_path / "mission-control.db"
		db = Database(db_path)
		mission = Mission(id="m1", objective="Build API", status="completed", total_cost_usd=12.50)
		db.insert_mission(mission)
		epoch = Epoch(id="ep1", mission_id="m1", number=1, units_planned=2, units_completed=2)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="Build API")
		db.insert_plan(plan)
		wu = WorkUnit(id="wu1", plan_id="p1", title="Unit 1", status="completed", epoch_id="ep1")
		db.insert_work_unit(wu)
		db.close()
		return config_file, db_path

	def test_summary_latest(self, tmp_path: Path, capsys: object) -> None:
		config_file, _ = self._setup_db(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["summary", "--config", str(config_file)])
		result = cmd_summary(args)
		assert result == 0

	def test_summary_no_db(self, tmp_path: Path) -> None:
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text('[target]\nname = "test"\npath = "/tmp/test"\n')
		parser = build_parser()
		args = parser.parse_args(["summary", "--config", str(config_file)])
		result = cmd_summary(args)
		assert result == 1

	def test_summary_all_missions(self, tmp_path: Path) -> None:
		config_file, _ = self._setup_db(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["summary", "--config", str(config_file), "--all"])
		result = cmd_summary(args)
		assert result == 0

	def test_summary_json_output(self, tmp_path: Path) -> None:
		config_file, _ = self._setup_db(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["summary", "--config", str(config_file), "--json"])
		result = cmd_summary(args)
		assert result == 0


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
