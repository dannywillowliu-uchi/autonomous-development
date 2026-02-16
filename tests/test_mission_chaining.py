"""Tests for mission chaining: --chain CLI flag, suggest_followup, ContinuousMissionResult.next_objective."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mission_control.cli import build_parser, cmd_mission
from mission_control.config import ContinuousConfig, MissionConfig, _build_continuous
from mission_control.continuous_controller import ContinuousMissionResult
from mission_control.models import BacklogItem
from mission_control.strategist import Strategist


class TestContinuousConfigChainMaxDepth:
	"""Test chain_max_depth field on ContinuousConfig."""

	def test_default_value(self) -> None:
		cc = ContinuousConfig()
		assert cc.chain_max_depth == 3

	def test_custom_value(self) -> None:
		cc = ContinuousConfig(chain_max_depth=5)
		assert cc.chain_max_depth == 5

	def test_build_continuous_with_chain_max_depth(self) -> None:
		cc = _build_continuous({"chain_max_depth": 7})
		assert cc.chain_max_depth == 7

	def test_build_continuous_without_chain_max_depth(self) -> None:
		cc = _build_continuous({})
		assert cc.chain_max_depth == 3


class TestContinuousMissionResultNextObjective:
	"""Test next_objective field on ContinuousMissionResult."""

	def test_default_empty(self) -> None:
		result = ContinuousMissionResult()
		assert result.next_objective == ""

	def test_set_next_objective(self) -> None:
		result = ContinuousMissionResult(next_objective="Build feature X")
		assert result.next_objective == "Build feature X"

	def test_all_fields_present(self) -> None:
		result = ContinuousMissionResult(
			mission_id="abc",
			objective="Build A",
			objective_met=True,
			total_units_dispatched=5,
			total_units_merged=4,
			total_units_failed=1,
			wall_time_seconds=120.0,
			stopped_reason="planner_completed",
			next_objective="Build B",
			ambition_score=7,
			proposed_by_strategist=True,
		)
		assert result.next_objective == "Build B"
		assert result.ambition_score == 7
		assert result.proposed_by_strategist is True


class TestChainCLIArgs:
	"""Test --chain and --max-chain-depth argument parsing."""

	def test_chain_flag_default_false(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission"])
		assert args.chain is False

	def test_chain_flag_set(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--chain"])
		assert args.chain is True

	def test_max_chain_depth_default(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission"])
		assert args.max_chain_depth == 3

	def test_max_chain_depth_custom(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--max-chain-depth", "5"])
		assert args.max_chain_depth == 5

	def test_chain_with_max_depth(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--chain", "--max-chain-depth", "2"])
		assert args.chain is True
		assert args.max_chain_depth == 2


class TestChainLoopInCLI:
	"""Test the chaining loop in cmd_mission with mocked controller."""

	def _make_config(self, tmp_path: Path) -> tuple[Path, MissionConfig]:
		config_path = tmp_path / "mission-control.toml"
		config_path.write_text(
			f'[target]\nname = "test"\npath = "{tmp_path}"\nobjective = "Do something"\n'
		)
		config = MissionConfig()
		config.target.name = "test"
		config.target.path = str(tmp_path)
		config.target.objective = "Do something"
		return config_path, config

	@patch("mission_control.continuous_controller.ContinuousController")
	@patch("mission_control.cli.load_config")
	@patch("mission_control.cli.Database")
	def test_no_chain_runs_once(
		self, mock_db_cls: MagicMock, mock_load: MagicMock, mock_ctrl_cls: MagicMock, tmp_path: Path,
	) -> None:
		_, config = self._make_config(tmp_path)
		mock_load.return_value = config

		result = ContinuousMissionResult(
			mission_id="m1", objective_met=True, next_objective="Follow up",
		)
		mock_ctrl = MagicMock()
		mock_ctrl.run = MagicMock(return_value=result)
		mock_ctrl_cls.return_value = mock_ctrl

		parser = build_parser()
		args = parser.parse_args(["mission", "--config", str(tmp_path / "mission-control.toml")])

		with patch("asyncio.run", side_effect=lambda coro: result):
			ret = cmd_mission(args)

		assert ret == 0

	@patch("mission_control.continuous_controller.ContinuousController")
	@patch("mission_control.cli.load_config")
	@patch("mission_control.cli.Database")
	def test_chain_runs_multiple(
		self, mock_db_cls: MagicMock, mock_load: MagicMock, mock_ctrl_cls: MagicMock, tmp_path: Path,
	) -> None:
		_, config = self._make_config(tmp_path)
		mock_load.return_value = config

		result1 = ContinuousMissionResult(
			mission_id="m1", objective_met=False, next_objective="Continue X",
		)
		result2 = ContinuousMissionResult(
			mission_id="m2", objective_met=True, next_objective="",
		)

		call_count = [0]

		def run_side_effect(coro):
			call_count[0] += 1
			if call_count[0] == 1:
				return result1
			return result2

		mock_ctrl = MagicMock()
		mock_ctrl_cls.return_value = mock_ctrl

		parser = build_parser()
		args = parser.parse_args([
			"mission", "--chain", "--max-chain-depth", "3",
			"--config", str(tmp_path / "mission-control.toml"),
		])

		with patch("asyncio.run", side_effect=run_side_effect):
			ret = cmd_mission(args)

		assert ret == 0
		assert call_count[0] == 2
		# Objective should have been updated to the chained one
		assert config.target.objective == "Continue X"

	@patch("mission_control.continuous_controller.ContinuousController")
	@patch("mission_control.cli.load_config")
	@patch("mission_control.cli.Database")
	def test_chain_respects_max_depth(
		self, mock_db_cls: MagicMock, mock_load: MagicMock, mock_ctrl_cls: MagicMock, tmp_path: Path,
	) -> None:
		_, config = self._make_config(tmp_path)
		mock_load.return_value = config

		# Always return a next_objective to force hitting the depth limit
		result = ContinuousMissionResult(
			mission_id="m1", objective_met=False, next_objective="Keep going",
		)

		mock_ctrl = MagicMock()
		mock_ctrl_cls.return_value = mock_ctrl

		call_count = [0]

		def run_side_effect(coro):
			call_count[0] += 1
			return result

		parser = build_parser()
		args = parser.parse_args([
			"mission", "--chain", "--max-chain-depth", "2",
			"--config", str(tmp_path / "mission-control.toml"),
		])

		with patch("asyncio.run", side_effect=run_side_effect):
			ret = cmd_mission(args)

		# Should stop at max_chain_depth=2
		assert call_count[0] == 2
		assert ret == 1  # objective_met=False -> return 1

	@patch("mission_control.continuous_controller.ContinuousController")
	@patch("mission_control.cli.load_config")
	@patch("mission_control.cli.Database")
	def test_chain_stops_on_empty_next_objective(
		self, mock_db_cls: MagicMock, mock_load: MagicMock, mock_ctrl_cls: MagicMock, tmp_path: Path,
	) -> None:
		_, config = self._make_config(tmp_path)
		mock_load.return_value = config

		result = ContinuousMissionResult(
			mission_id="m1", objective_met=False, next_objective="",
		)
		mock_ctrl = MagicMock()
		mock_ctrl_cls.return_value = mock_ctrl

		parser = build_parser()
		args = parser.parse_args([
			"mission", "--chain",
			"--config", str(tmp_path / "mission-control.toml"),
		])

		with patch("asyncio.run", return_value=result):
			cmd_mission(args)


class TestStrategistSuggestFollowup:
	"""Test Strategist.suggest_followup method."""

	def _make_strategist(self) -> tuple[Strategist, MagicMock]:
		config = MissionConfig()
		config.target.path = "/tmp/test"
		db = MagicMock()
		strategist = Strategist(config, db)
		return strategist, db

	def test_no_followup_when_objective_met_no_failures(self) -> None:
		strategist, db = self._make_strategist()
		db.get_pending_backlog.return_value = []

		result = MagicMock()
		result.objective_met = True
		result.total_units_merged = 5
		result.total_units_failed = 0
		result.stopped_reason = "planner_completed"

		mission = MagicMock()
		mission.objective = "Build feature"

		assert strategist.suggest_followup(result, mission) == ""

	def test_followup_when_objective_not_met(self) -> None:
		strategist, db = self._make_strategist()
		db.get_pending_backlog.return_value = [
			BacklogItem(title="Fix bug", track="quality", priority_score=8.0),
		]

		result = MagicMock()
		result.objective_met = False
		result.total_units_merged = 2
		result.total_units_failed = 3
		result.stopped_reason = "wall_time_exceeded"

		mission = MagicMock()
		mission.objective = "Build feature"

		followup = strategist.suggest_followup(result, mission)
		assert followup != ""
		assert "wall_time_exceeded" in followup
		assert "Fix bug" in followup

	def test_followup_when_units_failed(self) -> None:
		strategist, db = self._make_strategist()
		db.get_pending_backlog.return_value = [
			BacklogItem(title="Add auth", track="feature", priority_score=9.0),
			BacklogItem(title="Add tests", track="quality", priority_score=7.0),
		]

		result = MagicMock()
		result.objective_met = True
		result.total_units_merged = 3
		result.total_units_failed = 2
		result.stopped_reason = "planner_completed"

		mission = MagicMock()
		mission.objective = "Build stuff"

		followup = strategist.suggest_followup(result, mission)
		assert followup != ""
		assert "2 units failed" in followup
		assert "Add auth" in followup

	def test_no_followup_when_no_backlog(self) -> None:
		strategist, db = self._make_strategist()
		db.get_pending_backlog.return_value = []

		result = MagicMock()
		result.objective_met = False
		result.total_units_merged = 0
		result.total_units_failed = 5
		result.stopped_reason = "error"

		mission = MagicMock()
		mission.objective = "Build feature"

		assert strategist.suggest_followup(result, mission) == ""

	def test_followup_limits_to_3_items(self) -> None:
		strategist, db = self._make_strategist()
		db.get_pending_backlog.return_value = [
			BacklogItem(title=f"Item {i}", track="feature", priority_score=float(10 - i))
			for i in range(5)
		]

		result = MagicMock()
		result.objective_met = False
		result.total_units_merged = 1
		result.total_units_failed = 1
		result.stopped_reason = "stall"

		mission = MagicMock()
		mission.objective = "Build"

		followup = strategist.suggest_followup(result, mission)
		# Should mention 5 remaining items but only describe top 3
		assert "5 remaining backlog items" in followup
		assert "Item 0" in followup
		assert "Item 1" in followup
		assert "Item 2" in followup
		# Item 3 and 4 should not be in the descriptions
		assert "Item 3" not in followup
