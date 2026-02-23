"""Tests for CLI argument parsing and commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mission_control.cli import (
	_render_backlog_markdown,
	build_parser,
	cmd_discover,
	cmd_init,
	cmd_live,
	cmd_mission,
	cmd_priority_defer,
	cmd_priority_export,
	cmd_priority_import,
	cmd_priority_list,
	cmd_priority_recalc,
	cmd_priority_set,
	cmd_summary,
	main,
	parse_backlog_md,
)
from mission_control.db import Database
from mission_control.models import BacklogItem, Epoch, Mission, Plan, WorkUnit, _now_iso


class TestArgParsing:
	def test_init_default_path(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["init"])
		assert args.command == "init"
		assert args.path == "."

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


class TestMissionChainArgs:
	def test_chain_flag_default_false(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission"])
		assert args.chain is False

	def test_chain_flag_set(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--chain"])
		assert args.chain is True


@dataclass
class _FakeResult:
	mission_id: str = "m1"
	objective: str = "test"
	objective_met: bool = True
	total_units_dispatched: int = 1
	total_units_merged: int = 1
	total_units_failed: int = 0
	wall_time_seconds: float = 10.0
	stopped_reason: str = "objective_met"
	final_verification_passed: bool | None = None
	next_objective: str = ""
	proposed_by_strategist: bool = False


@patch("mission_control.cli._start_dashboard_background", return_value=(None, None))
class TestMissionChainLoop:
	"""Tests that the chaining loop in cmd_mission respects max depth and next_objective."""

	def _mock_mission_run(self, results: list[_FakeResult]) -> tuple[MagicMock, MagicMock, list]:
		"""Setup mocks for cmd_mission. Returns (mock_load_config, mock_controller_cls, call_log)."""
		call_log: list[_FakeResult] = []
		result_iter = iter(results)

		mock_controller_cls = MagicMock()

		def make_controller(config: object, db: object) -> MagicMock:
			controller = MagicMock()
			res = next(result_iter)
			call_log.append(res)

			async def fake_run(**kwargs: object) -> _FakeResult:
				return res

			controller.run = fake_run
			controller.proposed_by_strategist = False
			return controller

		mock_controller_cls.side_effect = make_controller
		return mock_controller_cls, call_log

	@patch("mission_control.cli.Database")
	@patch("mission_control.cli.load_config")
	def test_chain_runs_multiple_missions(
		self, mock_load_config: MagicMock, mock_db_cls: MagicMock, _mock_dashboard: MagicMock,
	) -> None:
		"""Chain 2 missions then stop (next_objective empty on 2nd)."""
		config = MagicMock()
		config.target.objective = "first objective"
		config.scheduler.parallel.num_workers = 2
		config.continuous.cleanup_enabled = False
		mock_load_config.return_value = config

		results = [
			_FakeResult(mission_id="m1", next_objective="second objective"),
			_FakeResult(mission_id="m2", next_objective=""),
		]
		mock_ctrl, call_log = self._mock_mission_run(results)

		with patch("mission_control.continuous_controller.ContinuousController", mock_ctrl):
			parser = build_parser()
			args = parser.parse_args(["mission", "--chain", "--max-chain-depth", "5", "--config", "fake.toml"])
			result = cmd_mission(args)

		assert result == 0
		assert len(call_log) == 2
		assert config.target.objective == "second objective"

	@patch("mission_control.cli.Database")
	@patch("mission_control.cli.load_config")
	def test_chain_respects_max_depth(
		self, mock_load_config: MagicMock, mock_db_cls: MagicMock, _mock_dashboard: MagicMock,
	) -> None:
		"""Even if next_objective keeps being set, stop at max_chain_depth."""
		config = MagicMock()
		config.target.objective = "start"
		config.scheduler.parallel.num_workers = 2
		config.continuous.cleanup_enabled = False
		mock_load_config.return_value = config

		results = [
			_FakeResult(mission_id=f"m{i}", next_objective=f"objective {i+1}")
			for i in range(5)
		]
		mock_ctrl, call_log = self._mock_mission_run(results)

		with patch("mission_control.continuous_controller.ContinuousController", mock_ctrl):
			parser = build_parser()
			args = parser.parse_args(["mission", "--chain", "--max-chain-depth", "2", "--config", "fake.toml"])
			cmd_mission(args)

		assert len(call_log) == 2

	@patch("mission_control.cli.Database")
	@patch("mission_control.cli.load_config")
	def test_no_chain_runs_once(
		self, mock_load_config: MagicMock, mock_db_cls: MagicMock, _mock_dashboard: MagicMock,
	) -> None:
		"""Without --chain, only one mission runs even if next_objective is set."""
		config = MagicMock()
		config.target.objective = "start"
		config.scheduler.parallel.num_workers = 2
		mock_load_config.return_value = config

		results = [_FakeResult(mission_id="m1", next_objective="should be ignored")]
		mock_ctrl, call_log = self._mock_mission_run(results)

		with patch("mission_control.continuous_controller.ContinuousController", mock_ctrl):
			parser = build_parser()
			args = parser.parse_args(["mission", "--config", "fake.toml"])
			result = cmd_mission(args)

		assert result == 0
		assert len(call_log) == 1

	@patch("mission_control.cli.Database")
	@patch("mission_control.cli.load_config")
	def test_chain_updates_objective_between_missions(
		self, mock_load_config: MagicMock, mock_db_cls: MagicMock, _mock_dashboard: MagicMock,
	) -> None:
		"""Verify config.target.objective is updated to next_objective between chains."""
		config = MagicMock()
		config.target.objective = "original"
		config.scheduler.parallel.num_workers = 2
		config.continuous.cleanup_enabled = False
		mock_load_config.return_value = config

		objectives_seen: list[str] = []

		mock_controller_cls = MagicMock()
		call_count = [0]

		def make_controller(cfg: object, db: object) -> MagicMock:
			objectives_seen.append(config.target.objective)
			controller = MagicMock()
			call_count[0] += 1
			if call_count[0] == 1:
				res = _FakeResult(mission_id="m1", next_objective="chained objective")
			else:
				res = _FakeResult(mission_id="m2", next_objective="")

			async def fake_run(**kwargs: object) -> _FakeResult:
				return res

			controller.run = fake_run
			controller.proposed_by_strategist = False
			return controller

		mock_controller_cls.side_effect = make_controller

		with patch("mission_control.continuous_controller.ContinuousController", mock_controller_cls):
			parser = build_parser()
			args = parser.parse_args(["mission", "--chain", "--config", "fake.toml"])
			cmd_mission(args)

		assert objectives_seen == ["original", "chained objective"]

	@patch("mission_control.cli.Database")
	@patch("mission_control.cli.load_config")
	def test_chain_returns_failure_from_last_mission(
		self, mock_load_config: MagicMock, mock_db_cls: MagicMock, _mock_dashboard: MagicMock,
	) -> None:
		"""Return code reflects the last mission result."""
		config = MagicMock()
		config.target.objective = "start"
		config.scheduler.parallel.num_workers = 2
		config.continuous.cleanup_enabled = False
		mock_load_config.return_value = config

		results = [
			_FakeResult(mission_id="m1", objective_met=True, next_objective="next"),
			_FakeResult(mission_id="m2", objective_met=False, next_objective=""),
		]
		mock_ctrl, call_log = self._mock_mission_run(results)

		with patch("mission_control.continuous_controller.ContinuousController", mock_ctrl):
			parser = build_parser()
			args = parser.parse_args(["mission", "--chain", "--config", "fake.toml"])
			result = cmd_mission(args)

		assert result == 1  # last mission failed


class TestCmdLive:
	def test_creates_db_when_missing(self, tmp_path: Path) -> None:
		"""cmd_live creates DB if it doesn't exist instead of returning error."""
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text('[target]\nname = "test"\npath = "/tmp/test"\n')
		db_path = tmp_path / "mission-control.db"
		assert not db_path.exists()

		parser = build_parser()
		args = parser.parse_args(["live", "--config", str(config_file)])

		# Patch LiveDashboard and uvicorn at the module level so the lazy
		# import inside cmd_live picks them up.
		mock_dash_cls = MagicMock()
		mock_uvicorn = MagicMock()

		with patch.dict("sys.modules", {
			"mission_control.dashboard.live": MagicMock(LiveDashboard=mock_dash_cls),
			"uvicorn": mock_uvicorn,
		}):
			result = cmd_live(args)

		assert result == 0
		assert db_path.exists()


# --- Priority subcommand tests (merged from test_cli_priority.py) ---

SAMPLE_BACKLOG_MD = """\
# Backlog

## P0: Replace LLM Evaluator with Objective Signals

**Problem**: The evaluator is expensive and noisy.

**Files**: evaluator.py

---

## P1: N-of-M Candidate Selection for Fixup

**Problem**: Fixup agent makes one attempt.

---

## P2: Architect/Editor Model Split

**Problem**: Workers use the same model.
"""


class TestPriorityArgParsing:
	def test_priority_set_args(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["priority", "set", "abc123", "9.5"])
		assert args.command == "priority"
		assert args.priority_command == "set"
		assert args.item_id == "abc123"
		assert args.score == 9.5

	def test_priority_import_args(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["priority", "import", "--file", "/tmp/BACKLOG.md"])
		assert args.command == "priority"
		assert args.priority_command == "import"
		assert args.file == "/tmp/BACKLOG.md"


class TestParseBacklogMd:
	def test_parses_items(self) -> None:
		items = parse_backlog_md(SAMPLE_BACKLOG_MD)
		assert len(items) == 3

	def test_extracts_titles(self) -> None:
		items = parse_backlog_md(SAMPLE_BACKLOG_MD)
		assert items[0].title == "Replace LLM Evaluator with Objective Signals"
		assert items[1].title == "N-of-M Candidate Selection for Fixup"
		assert items[2].title == "Architect/Editor Model Split"

	def test_calculates_impact_from_priority(self) -> None:
		items = parse_backlog_md(SAMPLE_BACKLOG_MD)
		assert items[0].impact == 10  # P0 -> 10-0 = 10
		assert items[1].impact == 9   # P1 -> 10-1 = 9
		assert items[2].impact == 8   # P2 -> 10-2 = 8

	def test_extracts_description(self) -> None:
		items = parse_backlog_md(SAMPLE_BACKLOG_MD)
		assert "expensive and noisy" in items[0].description


def _setup_db_with_config(tmp_path: Path) -> tuple[Path, Path]:
	"""Create config and DB for priority tests."""
	config_file = tmp_path / "mission-control.toml"
	config_file.write_text('[target]\nname = "test"\npath = "/tmp/test"\n')
	db_path = tmp_path / "mission-control.db"
	Database(db_path).close()
	return config_file, db_path


def _insert_sample_items(db_path: Path) -> list[str]:
	"""Insert sample backlog items, return their IDs."""
	now = _now_iso()
	ids = []
	with Database(db_path) as db:
		for i, (title, impact, effort) in enumerate([
			("Fix auth bug", 9, 3),
			("Add caching", 7, 5),
			("Update docs", 4, 2),
		]):
			item = BacklogItem(
				id=f"item{i}",
				title=title,
				impact=impact,
				effort=effort,
				priority_score=impact * (11 - effort) / 10.0,
				track="feature",
				status="pending",
				created_at=now,
				updated_at=now,
			)
			db.insert_backlog_item(item)
			ids.append(item.id)
	return ids


class TestCmdPriorityList:
	def test_list_with_items(self, tmp_path: Path, capsys: object) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		_insert_sample_items(db_path)
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "list"])
		result = cmd_priority_list(args)
		assert result == 0


class TestCmdPrioritySet:
	def test_set_score(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		ids = _insert_sample_items(db_path)
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "set", ids[0], "99.0"])
		result = cmd_priority_set(args)
		assert result == 0
		with Database(db_path) as db:
			item = db.get_backlog_item(ids[0])
			assert item is not None
			assert item.pinned_score == 99.0

	def test_set_not_found(self, tmp_path: Path) -> None:
		config_file, _ = _setup_db_with_config(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "set", "nonexistent", "5.0"])
		result = cmd_priority_set(args)
		assert result == 1


class TestCmdPriorityDefer:
	def test_defer_item(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		ids = _insert_sample_items(db_path)
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "defer", ids[1]])
		result = cmd_priority_defer(args)
		assert result == 0
		with Database(db_path) as db:
			item = db.get_backlog_item(ids[1])
			assert item is not None
			assert item.status == "deferred"


class TestCmdPriorityImport:
	def test_import_from_file(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		backlog_file = tmp_path / "BACKLOG.md"
		backlog_file.write_text(SAMPLE_BACKLOG_MD)
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "import", "--file", str(backlog_file),
		])
		result = cmd_priority_import(args)
		assert result == 0
		with Database(db_path) as db:
			items = db.list_backlog_items()
			assert len(items) == 3

	def test_import_skips_duplicates(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		backlog_file = tmp_path / "BACKLOG.md"
		backlog_file.write_text(SAMPLE_BACKLOG_MD)
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "import", "--file", str(backlog_file),
		])
		cmd_priority_import(args)
		# Import again -- should skip duplicates
		args2 = parser.parse_args([
			"priority", "--config", str(config_file), "import", "--file", str(backlog_file),
		])
		result = cmd_priority_import(args2)
		assert result == 0
		with Database(db_path) as db:
			items = db.list_backlog_items()
			assert len(items) == 3


class TestCmdPriorityRecalc:
	def test_recalc_arg_parsing(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["priority", "recalc"])
		assert args.command == "priority"
		assert args.priority_command == "recalc"

	def test_recalc_updates_scores(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		now = _now_iso()
		with Database(db_path) as db:
			db.insert_backlog_item(BacklogItem(
				id="item1", title="Test item", impact=10, effort=1,
				priority_score=0.0, status="pending",
				created_at=now, updated_at=now,
			))

		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "recalc",
		])
		result = cmd_priority_recalc(args)
		assert result == 0

		with Database(db_path) as db:
			item = db.get_backlog_item("item1")
			assert item is not None
			assert item.priority_score == pytest.approx(10.0)

	def test_recalc_no_changes(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "recalc",
		])
		result = cmd_priority_recalc(args)
		assert result == 0

	def test_recalc_no_db(self, tmp_path: Path) -> None:
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text('[target]\nname = "test"\npath = "/tmp/test"\n')
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "recalc",
		])
		result = cmd_priority_recalc(args)
		assert result == 1


class TestArgParsingExport:
	def test_export_default_args(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["priority", "export"])
		assert args.command == "priority"
		assert args.priority_command == "export"
		assert args.file is None
		assert args.status is None

	def test_export_with_file(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["priority", "export", "--file", "/tmp/backlog.md"])
		assert args.file == "/tmp/backlog.md"

	def test_export_with_status(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["priority", "export", "--status", "pending"])
		assert args.status == "pending"


class TestCmdPriorityExport:
	def test_export_stdout(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		_insert_sample_items(db_path)
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "export"])
		result = cmd_priority_export(args)
		assert result == 0
		captured = capsys.readouterr()
		assert "# Backlog Queue" in captured.out
		assert "Fix auth bug" in captured.out
		assert "Add caching" in captured.out

	def test_export_to_file(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		_insert_sample_items(db_path)
		outfile = tmp_path / "export.md"
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "export", "--file", str(outfile),
		])
		result = cmd_priority_export(args)
		assert result == 0
		assert outfile.exists()
		content = outfile.read_text()
		assert "# Backlog Queue" in content
		assert "Fix auth bug" in content

	def test_export_with_status_filter(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		now = _now_iso()
		with Database(db_path) as db:
			db.insert_backlog_item(BacklogItem(
				id="done1", title="Done task", impact=5, effort=3,
				priority_score=4.0, status="completed",
				created_at=now, updated_at=now,
			))
			db.insert_backlog_item(BacklogItem(
				id="pend1", title="Pending task", impact=8, effort=2,
				priority_score=7.2, status="pending",
				created_at=now, updated_at=now,
			))
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "export", "--status", "pending",
		])
		result = cmd_priority_export(args)
		assert result == 0
		captured = capsys.readouterr()
		assert "Pending task" in captured.out
		assert "Done task" not in captured.out
		assert "Filtered by status: pending" in captured.out

	def test_export_empty_db(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
		config_file, _ = _setup_db_with_config(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "export"])
		result = cmd_priority_export(args)
		assert result == 0
		captured = capsys.readouterr()
		assert "No backlog items" in captured.out

	def test_export_shows_attempt_count(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		now = _now_iso()
		with Database(db_path) as db:
			db.insert_backlog_item(BacklogItem(
				id="retry1", title="Retried task", impact=7, effort=3,
				priority_score=5.6, status="pending", attempt_count=3,
				created_at=now, updated_at=now,
			))
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "export"])
		result = cmd_priority_export(args)
		assert result == 0
		captured = capsys.readouterr()
		assert "**Attempts**: 3" in captured.out

	def test_export_no_db(self, tmp_path: Path) -> None:
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text('[target]\nname = "test"\npath = "/tmp/test"\n')
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "export",
		])
		result = cmd_priority_export(args)
		assert result == 1


class TestRenderBacklogMarkdown:
	def test_groups_by_status(self) -> None:
		now = _now_iso()
		items = [
			BacklogItem(
				id="a", title="Active", status="in_progress",
				impact=8, effort=3, priority_score=6.4,
				created_at=now, updated_at=now,
			),
			BacklogItem(
				id="b", title="Waiting", status="pending",
				impact=6, effort=4, priority_score=4.2,
				created_at=now, updated_at=now,
			),
			BacklogItem(
				id="c", title="Done", status="completed",
				impact=5, effort=2, priority_score=4.5,
				created_at=now, updated_at=now,
			),
		]
		md = _render_backlog_markdown(items)
		# in_progress should appear before pending which should appear before completed
		pos_active = md.index("In Progress")
		pos_pending = md.index("Pending")
		pos_completed = md.index("Completed")
		assert pos_active < pos_pending < pos_completed

	def test_sorts_by_score_within_group(self) -> None:
		now = _now_iso()
		items = [
			BacklogItem(
				id="lo", title="Low score", status="pending",
				priority_score=1.0, created_at=now, updated_at=now,
			),
			BacklogItem(
				id="hi", title="High score", status="pending",
				priority_score=9.0, created_at=now, updated_at=now,
			),
		]
		md = _render_backlog_markdown(items)
		assert md.index("High score") < md.index("Low score")
