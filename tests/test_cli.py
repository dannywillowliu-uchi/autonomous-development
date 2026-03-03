"""Tests for CLI argument parsing and commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from mission_control.cli import (
	build_parser,
	cmd_diagnose,
	cmd_init,
	cmd_live,
	cmd_mission,
	cmd_summary,
	cmd_trace,
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


SAMPLE_EVENTS = [
	'{"timestamp":"2026-01-15T10:00:00+00:00","worker_id":"w1","unit_id":"u1","event_type":"spawn","details":{"branch":"mc/unit-1"}}',
	'{"timestamp":"2026-01-15T10:00:05+00:00","worker_id":"w1","unit_id":"u1","event_type":"session_start","details":{}}',
	'{"timestamp":"2026-01-15T10:00:30+00:00","worker_id":"w2","unit_id":"u2","event_type":"spawn","details":{"branch":"mc/unit-2"}}',
	'{"timestamp":"2026-01-15T10:01:00+00:00","worker_id":"w1","unit_id":"u1","event_type":"merge","details":{"result":"ok"}}',
	'{"timestamp":"2026-01-15T10:01:10+00:00","worker_id":"w2","unit_id":"u2","event_type":"merge","details":{"result":"conflict"}}',
]


class TestCmdTrace:
	def _write_trace(self, tmp_path: Path, lines: list[str] | None = None) -> Path:
		trace_file = tmp_path / "trace.jsonl"
		trace_file.write_text("\n".join(lines or SAMPLE_EVENTS) + "\n")
		return trace_file

	def test_timeline_output_format(self, tmp_path: Path, capsys: object) -> None:
		"""Timeline groups by unit_id and shows relative timestamps."""
		trace_file = self._write_trace(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["trace", str(trace_file)])
		result = cmd_trace(args)
		assert result == 0
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "== Unit: u1 ==" in out
		assert "== Unit: u2 ==" in out
		assert "spawn" in out
		assert "worker=w1" in out
		# First event should be +0.0s relative
		assert "+     0.0s" in out

	def test_worker_filter(self, tmp_path: Path, capsys: object) -> None:
		"""--worker filters to a single worker's events."""
		trace_file = self._write_trace(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["trace", str(trace_file), "--worker", "w2"])
		result = cmd_trace(args)
		assert result == 0
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "worker=w2" in out
		assert "worker=w1" not in out

	def test_type_filter(self, tmp_path: Path, capsys: object) -> None:
		"""--type filters to a specific event type."""
		trace_file = self._write_trace(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["trace", str(trace_file), "--type", "merge"])
		result = cmd_trace(args)
		assert result == 0
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "merge" in out
		assert "spawn" not in out

	def test_json_passthrough(self, tmp_path: Path, capsys: object) -> None:
		"""--json outputs raw JSON lines."""
		import json

		trace_file = self._write_trace(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["trace", str(trace_file), "--json"])
		result = cmd_trace(args)
		assert result == 0
		out = capsys.readouterr().out  # type: ignore[union-attr]
		lines = [line for line in out.strip().splitlines() if line]
		assert len(lines) == 5
		# Each line should be valid JSON
		for line in lines:
			parsed = json.loads(line)
			assert "event_type" in parsed

	def test_missing_file(self, tmp_path: Path, capsys: object) -> None:
		"""Missing trace file returns error code 1."""
		parser = build_parser()
		args = parser.parse_args(["trace", str(tmp_path / "nonexistent.jsonl")])
		result = cmd_trace(args)
		assert result == 1
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "not found" in out

	def test_empty_file(self, tmp_path: Path, capsys: object) -> None:
		"""Empty trace file prints message and returns 0."""
		trace_file = tmp_path / "trace.jsonl"
		trace_file.write_text("")
		parser = build_parser()
		args = parser.parse_args(["trace", str(trace_file)])
		result = cmd_trace(args)
		assert result == 0
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "No trace events" in out


# Minimal valid TOML that load_config can parse
_DIAG_TOML = '[target]\nname = "test"\npath = "/tmp/test"\n'


class TestCmdDiagnose:
	def _setup_healthy(self, tmp_path: Path) -> Path:
		"""Create a config + DB with a completed mission and no problems."""
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text(_DIAG_TOML)
		db_path = tmp_path / "mission-control.db"
		db = Database(db_path)
		mission = Mission(id="m1", objective="Test", status="completed", total_cost_usd=3.50)
		db.insert_mission(mission)
		db.close()
		return config_file

	def test_healthy_state_all_ok(self, tmp_path: Path, capsys: object) -> None:
		"""Healthy installation reports all OK."""
		config_file = self._setup_healthy(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["diagnose", "--config", str(config_file)])
		result = cmd_diagnose(args)
		assert result == 0
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "Overall: OK" in out
		assert "[   OK] Database" in out
		assert "[   OK] Cost Summary" in out

	def test_stale_wal_detected(self, tmp_path: Path, capsys: object) -> None:
		"""Stale WAL file triggers WARN."""
		config_file = self._setup_healthy(tmp_path)
		# Create a non-empty WAL file to simulate stale state
		wal_file = tmp_path / "mission-control.db-wal"
		wal_file.write_bytes(b"\x00" * 100)
		parser = build_parser()
		args = parser.parse_args(["diagnose", "--config", str(config_file)])
		result = cmd_diagnose(args)
		assert result == 0  # WARN doesn't cause exit 1
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "WARN" in out
		assert "WAL/SHM" in out

	def test_orphaned_running_units(self, tmp_path: Path, capsys: object) -> None:
		"""Orphaned running work units trigger WARN."""
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text(_DIAG_TOML)
		db_path = tmp_path / "mission-control.db"
		db = Database(db_path)
		mission = Mission(id="m1", objective="Test", status="completed")
		db.insert_mission(mission)
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="Test")
		db.insert_plan(plan)
		# Create a work unit stuck in 'running' status
		wu = WorkUnit(id="wu1", plan_id="p1", title="Stuck unit", status="running", epoch_id="ep1")
		db.insert_work_unit(wu)
		db.close()
		parser = build_parser()
		args = parser.parse_args(["diagnose", "--config", str(config_file)])
		result = cmd_diagnose(args)
		assert result == 0
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "Orphaned running work units: 1" in out
		assert "WARN" in out

	def test_missing_config_handled(self, tmp_path: Path, capsys: object) -> None:
		"""Missing config file reports ERROR gracefully."""
		parser = build_parser()
		missing = str(tmp_path / "nonexistent.toml")
		args = parser.parse_args(["diagnose", "--config", missing])
		result = cmd_diagnose(args)
		assert result == 1
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "ERROR" in out
		assert "Config" in out

	def test_cost_summary_formatting(self, tmp_path: Path, capsys: object) -> None:
		"""Cost summary shows mission count and total cost."""
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text(_DIAG_TOML)
		db_path = tmp_path / "mission-control.db"
		db = Database(db_path)
		db.insert_mission(Mission(id="m1", objective="A", status="completed", total_cost_usd=5.25))
		db.insert_mission(Mission(id="m2", objective="B", status="completed", total_cost_usd=3.75))
		db.close()
		parser = build_parser()
		args = parser.parse_args(["diagnose", "--config", str(config_file)])
		result = cmd_diagnose(args)
		assert result == 0
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "Recent missions: 2" in out
		assert "$9.00" in out

	def test_json_output(self, tmp_path: Path, capsys: object) -> None:
		"""--json flag produces valid JSON output."""
		import json

		config_file = self._setup_healthy(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["diagnose", "--config", str(config_file), "--json"])
		result = cmd_diagnose(args)
		assert result == 0
		out = capsys.readouterr().out  # type: ignore[union-attr]
		data = json.loads(out)
		assert "checks" in data
		assert len(data["checks"]) == 4
		names = {c["name"] for c in data["checks"]}
		assert "Database" in names
		assert "Config" in names
		assert "Cost Summary" in names

	def test_no_db_still_reports(self, tmp_path: Path, capsys: object) -> None:
		"""When DB doesn't exist, diagnose still runs and reports WARN for DB."""
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text(_DIAG_TOML)
		parser = build_parser()
		args = parser.parse_args(["diagnose", "--config", str(config_file)])
		result = cmd_diagnose(args)
		assert result == 0
		out = capsys.readouterr().out  # type: ignore[union-attr]
		assert "Database file not found" in out

	def test_diagnose_arg_parsing(self) -> None:
		"""Diagnose subcommand parses correctly."""
		parser = build_parser()
		args = parser.parse_args(["diagnose"])
		assert args.command == "diagnose"
		assert args.json_output is False
