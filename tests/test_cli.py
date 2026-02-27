"""Tests for CLI argument parsing and commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from mission_control.cli import (
	build_parser,
	cmd_init,
	cmd_live,
	cmd_mission,
	cmd_summary,
	main,
)
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


class TestConfigPathValidation:
	def test_traversal_path_rejected(self) -> None:
		result = main(["status", "--config", "../../../../etc/passwd"])
		assert result == 1

	def test_null_byte_path_rejected(self) -> None:
		result = main(["status", "--config", "/home/user/config\x00.toml"])
		assert result == 1

	def test_traversal_in_register_rejected(self) -> None:
		result = main(["register", "--config", "../../../../etc/passwd"])
		assert result == 1

	def test_traversal_in_mission_rejected(self) -> None:
		result = main(["mission", "--config", "../../../../etc/passwd"])
		assert result == 1


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


@patch("mission_control.cli._start_dashboard_background", return_value=(None, None))
class TestMissionChainLoop:
	"""Tests that the chaining loop in cmd_mission respects max depth and deliberative planner."""

	def _mock_mission_run(self, results: list[_FakeResult]) -> tuple[MagicMock, list]:
		"""Setup mocks for cmd_mission. Returns (mock_controller_cls, call_log)."""
		call_log: list[_FakeResult] = []
		result_iter = iter(results)

		mock_controller_cls = MagicMock()

		def make_controller(config: object, db: object, **kwargs: object) -> MagicMock:
			controller = MagicMock()
			res = next(result_iter)
			call_log.append(res)

			async def fake_run(**kwargs: object) -> _FakeResult:
				return res

			controller.run = fake_run
			return controller

		mock_controller_cls.side_effect = make_controller
		return mock_controller_cls, call_log

	@patch("mission_control.cli.Database")
	@patch("mission_control.cli.load_config")
	def test_no_chain_runs_once(
		self, mock_load_config: MagicMock, mock_db_cls: MagicMock, _mock_dashboard: MagicMock,
	) -> None:
		"""Without --chain, only one mission runs."""
		config = MagicMock()
		config.target.objective = "start"
		config.scheduler.parallel.num_workers = 2
		mock_load_config.return_value = config

		results = [_FakeResult(mission_id="m1")]
		mock_ctrl, call_log = self._mock_mission_run(results)

		with patch("mission_control.continuous_controller.ContinuousController", mock_ctrl):
			parser = build_parser()
			args = parser.parse_args(["mission", "--config", "fake.toml"])
			result = cmd_mission(args)

		assert result == 0
		assert len(call_log) == 1

	@patch("mission_control.cli.Database")
	@patch("mission_control.cli.load_config")
	def test_chain_proposes_next_objective(
		self, mock_load_config: MagicMock, mock_db_cls: MagicMock, _mock_dashboard: MagicMock,
	) -> None:
		"""With --chain, deliberative planner proposes next objective."""
		config = MagicMock()
		config.target.objective = "start"
		config.scheduler.parallel.num_workers = 2
		config.continuous.cleanup_enabled = False
		mock_load_config.return_value = config

		results = [_FakeResult(mission_id="m1")]
		mock_ctrl, call_log = self._mock_mission_run(results)

		# Mock the deliberative planner to return empty objective (stops chain)
		mock_delib = MagicMock()
		mock_delib.return_value.propose_next_objective = MagicMock(
			return_value=("", ""),
		)

		with (
			patch("mission_control.continuous_controller.ContinuousController", mock_ctrl),
			patch("mission_control.deliberative_planner.DeliberativePlanner", mock_delib),
			patch("mission_control.cli.asyncio.run", side_effect=[results[0], ("", "")]),
		):
			parser = build_parser()
			args = parser.parse_args(["mission", "--chain", "--config", "fake.toml"])
			result = cmd_mission(args)

		assert result == 0
		assert len(call_log) == 1


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


