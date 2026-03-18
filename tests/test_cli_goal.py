"""Tests for CLI --goal flag and goal-status subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from autodev.cli import _resolve_goal_path, build_parser, cmd_goal_status
from autodev.config import GoalConfig, MissionConfig, TargetConfig

MINIMAL_GOAL_MD = """\
# Goal: test coverage

Maximize test coverage.

## Fitness

echo 0.75
"""

COMPONENTS_GOAL_MD = """\
# Goal: quality

Code quality metrics.

## Components

- lint (weight: 0.6): echo 0.9
- tests (weight: 0.4): echo 0.8

## Target

0.95
"""

MINIMAL_CONFIG_TOML = '[target]\nname = "test"\npath = "{path}"\nobjective = "test"\n'


class TestGoalArgParsing:
	def test_mission_goal_flag(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--goal", "/tmp/GOAL.md"])
		assert args.goal == "/tmp/GOAL.md"

	def test_mission_goal_default_none(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission"])
		assert args.goal is None

	def test_swarm_goal_flag(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["swarm", "--goal", "my/GOAL.md"])
		assert args.goal == "my/GOAL.md"

	def test_swarm_goal_default_none(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["swarm"])
		assert args.goal is None

	def test_goal_status_subcommand(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["goal-status"])
		assert args.command == "goal-status"

	def test_goal_status_with_goal_path(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["goal-status", "--goal", "/tmp/GOAL.md"])
		assert args.goal == "/tmp/GOAL.md"

	def test_goal_status_with_timeout(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["goal-status", "--timeout", "30"])
		assert args.timeout == 30


class TestResolveGoalPath:
	def test_explicit_absolute_path(self, tmp_path: Path) -> None:
		goal_file = tmp_path / "GOAL.md"
		goal_file.write_text(MINIMAL_GOAL_MD)
		config = MissionConfig(
			target=TargetConfig(path=str(tmp_path)),
			goal=GoalConfig(),
		)
		args = MagicMock(goal=str(goal_file))
		result = _resolve_goal_path(args, config)
		assert result == goal_file

	def test_auto_detect_in_target(self, tmp_path: Path) -> None:
		goal_file = tmp_path / "GOAL.md"
		goal_file.write_text(MINIMAL_GOAL_MD)
		config = MissionConfig(
			target=TargetConfig(path=str(tmp_path)),
			goal=GoalConfig(),
		)
		args = MagicMock(goal=None)
		result = _resolve_goal_path(args, config)
		assert result == goal_file

	def test_returns_none_when_missing(self, tmp_path: Path) -> None:
		config = MissionConfig(
			target=TargetConfig(path=str(tmp_path)),
			goal=GoalConfig(),
		)
		args = MagicMock(goal=None)
		result = _resolve_goal_path(args, config)
		assert result is None


class TestCmdGoalStatus:
	def _make_args(self, tmp_path: Path, goal_path: str | None = None, timeout: int = 60) -> tuple:
		config_file = tmp_path / "autodev.toml"
		config_file.write_text(MINIMAL_CONFIG_TOML.format(path=str(tmp_path)))
		parser = build_parser()
		argv = ["goal-status", "--config", str(config_file)]
		if goal_path:
			argv += ["--goal", goal_path]
		argv += ["--timeout", str(timeout)]
		return parser.parse_args(argv)

	def test_no_goal_file_returns_error(self, tmp_path: Path, capsys) -> None:
		args = self._make_args(tmp_path)
		result = cmd_goal_status(args)
		assert result == 1
		assert "no GOAL.md found" in capsys.readouterr().out

	def test_displays_composite_score(self, tmp_path: Path, capsys) -> None:
		goal_file = tmp_path / "GOAL.md"
		goal_file.write_text(MINIMAL_GOAL_MD)
		args = self._make_args(tmp_path, goal_path=str(goal_file))
		result = cmd_goal_status(args)
		assert result == 0
		out = capsys.readouterr().out
		assert "Goal: test coverage" in out
		assert "0.750" in out
		assert "Composite score" in out

	def test_displays_components(self, tmp_path: Path, capsys) -> None:
		goal_file = tmp_path / "GOAL.md"
		goal_file.write_text(COMPONENTS_GOAL_MD)
		args = self._make_args(tmp_path, goal_path=str(goal_file))
		result = cmd_goal_status(args)
		assert result == 0
		out = capsys.readouterr().out
		assert "Components:" in out
		assert "lint" in out
		assert "tests" in out
		assert "0.95" in out  # target score shown

	def test_target_met_status(self, tmp_path: Path, capsys) -> None:
		goal_md = "# Goal: easy\n\n## Fitness\n\necho 1.0\n\n## Target\n\n1.0\n"
		goal_file = tmp_path / "GOAL.md"
		goal_file.write_text(goal_md)
		args = self._make_args(tmp_path, goal_path=str(goal_file))
		result = cmd_goal_status(args)
		assert result == 0
		out = capsys.readouterr().out
		assert "TARGET MET" in out

	def test_in_progress_status(self, tmp_path: Path, capsys) -> None:
		goal_file = tmp_path / "GOAL.md"
		goal_file.write_text(MINIMAL_GOAL_MD)
		args = self._make_args(tmp_path, goal_path=str(goal_file))
		result = cmd_goal_status(args)
		assert result == 0
		out = capsys.readouterr().out
		assert "IN PROGRESS" in out

	def test_auto_detects_goal_in_target(self, tmp_path: Path, capsys) -> None:
		goal_file = tmp_path / "GOAL.md"
		goal_file.write_text(MINIMAL_GOAL_MD)
		args = self._make_args(tmp_path)
		result = cmd_goal_status(args)
		assert result == 0
		out = capsys.readouterr().out
		assert "Goal: test coverage" in out

	def test_invalid_goal_file(self, tmp_path: Path, capsys) -> None:
		goal_file = tmp_path / "GOAL.md"
		goal_file.write_text("# Not a valid goal file\n\nNo fitness section.\n")
		args = self._make_args(tmp_path, goal_path=str(goal_file))
		result = cmd_goal_status(args)
		assert result == 1
		out = capsys.readouterr().out
		assert "Error parsing" in out

	def test_fitness_command_failure(self, tmp_path: Path, capsys) -> None:
		goal_md = "# Goal: failing\n\n## Fitness\n\nexit 1\n"
		goal_file = tmp_path / "GOAL.md"
		goal_file.write_text(goal_md)
		args = self._make_args(tmp_path, goal_path=str(goal_file))
		result = cmd_goal_status(args)
		assert result == 1
		assert "failed" in capsys.readouterr().out


class TestGoalFlagWiring:
	def test_mission_goal_sets_config(self, tmp_path: Path) -> None:
		"""Verify cmd_mission sets goal config before running."""
		config_file = tmp_path / "autodev.toml"
		config_file.write_text(MINIMAL_CONFIG_TOML.format(path=str(tmp_path)))

		mock_config = MissionConfig(
			target=TargetConfig(path=str(tmp_path), objective="test"),
			goal=GoalConfig(),
		)

		captured_config = {}

		def fake_controller_init(cfg, db, **kwargs):
			captured_config["enabled"] = cfg.goal.enabled
			captured_config["goal_file"] = cfg.goal.goal_file
			raise RuntimeError("stop here")

		with (
			patch("autodev.cli.load_config", return_value=mock_config),
			patch("autodev.cli._get_db_path", return_value=tmp_path / "autodev.db"),
			patch("autodev.cli.Database"),
			patch("autodev.continuous_controller.ContinuousController", side_effect=fake_controller_init),
		):
			parser = build_parser()
			args = parser.parse_args([
				"mission", "--config", str(config_file),
				"--goal", "/tmp/GOAL.md", "--no-dashboard",
			])

			from autodev.cli import cmd_mission
			try:
				cmd_mission(args)
			except RuntimeError:
				pass

			assert captured_config["enabled"] is True
			assert captured_config["goal_file"] == "/tmp/GOAL.md"

	def test_swarm_goal_sets_config(self, tmp_path: Path) -> None:
		"""Verify cmd_swarm sets goal config before attempting swarm startup."""
		config_file = tmp_path / "autodev.toml"
		config_file.write_text(MINIMAL_CONFIG_TOML.format(path=str(tmp_path)))

		mock_config = MissionConfig(
			target=TargetConfig(path=str(tmp_path), objective="test"),
			goal=GoalConfig(),
		)

		with (
			patch("autodev.cli.load_config", return_value=mock_config),
			patch("autodev.cli._get_db_path", return_value=tmp_path / "autodev.db"),
			patch("autodev.swarm.controller.SwarmController") as mock_ctrl_cls,
			patch("autodev.swarm.planner.DrivingPlanner") as mock_planner_cls,
		):
			# Make the planner run() immediately return
			mock_planner = MagicMock()
			mock_planner.run = MagicMock(return_value=MagicMock())
			mock_planner_cls.return_value = mock_planner

			mock_ctrl = MagicMock()
			mock_ctrl.initialize = MagicMock(return_value=MagicMock())
			mock_ctrl_cls.return_value = mock_ctrl

			parser = build_parser()
			args = parser.parse_args([
				"swarm", "--config", str(config_file),
				"--goal", "/tmp/GOAL.md", "--no-dashboard",
			])

			from autodev.cli import cmd_swarm
			try:
				cmd_swarm(args)
			except Exception:
				pass

			assert mock_config.goal.enabled is True
			assert mock_config.goal.goal_file == "/tmp/GOAL.md"
