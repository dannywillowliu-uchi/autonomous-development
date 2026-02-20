"""Tests for CLI argument parsing and commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from mission_control.cli import build_parser, cmd_discover, cmd_init, cmd_live, cmd_mission, cmd_summary, main
from mission_control.db import Database
from mission_control.models import Epoch, Mission, Plan, WorkUnit


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
