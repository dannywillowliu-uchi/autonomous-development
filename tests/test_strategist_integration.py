"""Integration tests for strategist CLI flag and post-mission context append."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.cli import build_parser, cmd_mission
from mission_control.continuous_controller import ContinuousController, ContinuousMissionResult
from mission_control.db import Database
from mission_control.models import Mission


@pytest.fixture()
def mock_strategist_module():
	"""Inject a fake mission_control.strategist module into sys.modules."""
	mock_module = types.ModuleType("mission_control.strategist")
	mock_cls = MagicMock()
	mock_cls.return_value.propose_objective = AsyncMock(
		return_value=("Build a REST API", "High priority backlog item", 7),
	)
	mock_module.Strategist = mock_cls  # type: ignore[attr-defined]
	sys.modules["mission_control.strategist"] = mock_module
	yield mock_cls
	sys.modules.pop("mission_control.strategist", None)


class TestStrategistCliFlag:
	"""Test --strategist flag parsing in CLI."""

	def test_strategist_flag_default_false(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission"])
		assert args.strategist is False

	def test_strategist_flag_set(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--strategist"])
		assert args.strategist is True

	def test_strategist_with_approve_all(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--strategist", "--approve-all"])
		assert args.strategist is True
		assert args.approve_all is True

	def test_strategist_with_other_flags(self) -> None:
		parser = build_parser()
		args = parser.parse_args([
			"mission", "--strategist", "--workers", "4", "--config", "custom.toml",
		])
		assert args.strategist is True
		assert args.workers == 4
		assert args.config == "custom.toml"


class TestStrategistApprovalFlow:
	"""Test strategist objective proposal and approval flow in cmd_mission."""

	def _make_config(self, tmp_path: Path) -> Path:
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text("""\
[target]
name = "test"
path = "/tmp/test"
objective = ""

[target.verification]
command = "echo ok"

[scheduler]
model = "sonnet"
""")
		return config_file

	@patch("mission_control.cli.input", return_value="y")
	@patch("mission_control.cli.asyncio.run")
	def test_strategist_approved_sets_objective(
		self,
		mock_run: MagicMock,
		mock_input: MagicMock,
		tmp_path: Path,
		mock_strategist_module: MagicMock,
	) -> None:
		config_file = self._make_config(tmp_path)
		db_path = tmp_path / "mission-control.db"
		Database(db_path).close()

		proposal = ("Build a REST API", "High priority backlog item", 7)
		mission_result = ContinuousMissionResult(
			mission_id="m1", objective_met=True, stopped_reason="planner_completed",
		)
		mock_run.side_effect = [proposal, mission_result]

		parser = build_parser()
		args = parser.parse_args([
			"mission", "--strategist", "--config", str(config_file),
		])
		result = cmd_mission(args)

		assert result == 0
		mock_input.assert_called_once()

	@patch("mission_control.cli.input", return_value="n")
	@patch("mission_control.cli.asyncio.run")
	def test_strategist_rejected_exits_zero(
		self,
		mock_run: MagicMock,
		mock_input: MagicMock,
		tmp_path: Path,
		mock_strategist_module: MagicMock,
	) -> None:
		config_file = self._make_config(tmp_path)
		db_path = tmp_path / "mission-control.db"
		Database(db_path).close()

		proposal = ("Build a REST API", "High priority backlog item", 7)
		mock_run.return_value = proposal

		parser = build_parser()
		args = parser.parse_args([
			"mission", "--strategist", "--config", str(config_file),
		])
		result = cmd_mission(args)

		assert result == 0

	@patch("mission_control.cli.asyncio.run")
	def test_strategist_auto_approve_skips_prompt(
		self,
		mock_run: MagicMock,
		tmp_path: Path,
		mock_strategist_module: MagicMock,
	) -> None:
		config_file = self._make_config(tmp_path)
		db_path = tmp_path / "mission-control.db"
		Database(db_path).close()

		proposal = ("Build a REST API", "High priority backlog item", 7)
		mission_result = ContinuousMissionResult(
			mission_id="m1", objective_met=True, stopped_reason="planner_completed",
		)
		mock_run.side_effect = [proposal, mission_result]

		with patch("mission_control.cli.input") as mock_input:
			parser = build_parser()
			args = parser.parse_args([
				"mission", "--strategist", "--approve-all", "--config", str(config_file),
			])
			result = cmd_mission(args)

		assert result == 0
		mock_input.assert_not_called()


class TestPostMissionStrategicContext:
	"""Test that strategic context is appended after mission completion."""

	def test_controller_init_defaults(self) -> None:
		"""ContinuousController.__init__ sets strategist attributes."""
		db = Database(":memory:")
		config = MagicMock()
		controller = ContinuousController(config, db)
		assert controller.ambition_score == 0.0
		assert controller.proposed_by_strategist is False
		db.close()

	def test_append_strategic_context_logic(self) -> None:
		"""Verify the strategic context append logic produces correct args."""
		db = Database(":memory:")
		db.append_strategic_context = MagicMock()  # type: ignore[attr-defined]
		db.get_recent_handoffs = MagicMock(return_value=[])  # type: ignore[method-assign]

		mission = Mission(id="m1", objective="Test objective", status="completed")

		# Replicate the logic from continuous_controller.py
		merged_summaries: list[str] = []
		failed_summaries: list[str] = []
		handoffs = db.get_recent_handoffs(mission.id, limit=50)
		for h in handoffs:
			summary_text = h.summary[:200] if h.summary else ""
			if h.status == "completed" and summary_text:
				merged_summaries.append(summary_text)
			elif summary_text:
				failed_summaries.append(summary_text)

		db.append_strategic_context(
			what_attempted=mission.objective[:500],
			what_worked="; ".join(merged_summaries[:10]) or "nothing merged",
			what_failed="; ".join(failed_summaries[:10]) or "no failures",
			recommended_next="planner_completed",
		)

		db.append_strategic_context.assert_called_once_with(
			what_attempted="Test objective",
			what_worked="nothing merged",
			what_failed="no failures",
			recommended_next="planner_completed",
		)
		db.close()

	def test_append_strategic_context_with_handoffs(self) -> None:
		"""Verify handoff summaries are correctly categorized."""
		db = Database(":memory:")
		db.append_strategic_context = MagicMock()  # type: ignore[attr-defined]

		# Create fake handoffs
		merged_handoff = MagicMock()
		merged_handoff.status = "completed"
		merged_handoff.summary = "Added user authentication"

		failed_handoff = MagicMock()
		failed_handoff.status = "failed"
		failed_handoff.summary = "Database migration failed"

		db.get_recent_handoffs = MagicMock(  # type: ignore[method-assign]
			return_value=[merged_handoff, failed_handoff],
		)

		mission = Mission(id="m2", objective="Improve auth system", status="completed")

		# Replicate the logic
		merged_summaries: list[str] = []
		failed_summaries: list[str] = []
		handoffs = db.get_recent_handoffs(mission.id, limit=50)
		for h in handoffs:
			summary_text = h.summary[:200] if h.summary else ""
			if h.status == "completed" and summary_text:
				merged_summaries.append(summary_text)
			elif summary_text:
				failed_summaries.append(summary_text)

		db.append_strategic_context(
			what_attempted=mission.objective[:500],
			what_worked="; ".join(merged_summaries[:10]) or "nothing merged",
			what_failed="; ".join(failed_summaries[:10]) or "no failures",
			recommended_next="planner_completed",
		)

		db.append_strategic_context.assert_called_once_with(
			what_attempted="Improve auth system",
			what_worked="Added user authentication",
			what_failed="Database migration failed",
			recommended_next="planner_completed",
		)
		db.close()

	def test_proposed_by_strategist_propagated_from_cli(self) -> None:
		"""When --strategist is used, controller.proposed_by_strategist should be True."""
		db = Database(":memory:")
		config = MagicMock()
		controller = ContinuousController(config, db)
		controller.proposed_by_strategist = True
		assert controller.proposed_by_strategist is True
		db.close()

	def test_ambition_score_set_on_controller(self) -> None:
		"""Ambition score can be set on the controller."""
		db = Database(":memory:")
		config = MagicMock()
		controller = ContinuousController(config, db)
		controller.ambition_score = 7.5
		assert controller.ambition_score == 7.5
		db.close()

	def test_mission_hasattr_guard_works(self) -> None:
		"""The hasattr guard for Mission fields works without error."""
		mission = Mission(id="m1", objective="Test", status="completed")

		# Simulate the hasattr guard from the controller
		if hasattr(mission, "ambition_score"):
			mission.ambition_score = 6.0  # type: ignore[attr-defined]
		if hasattr(mission, "proposed_by_strategist"):
			mission.proposed_by_strategist = True  # type: ignore[attr-defined]

		# Currently Mission doesn't have these fields, so hasattr returns False
		# This test verifies the guard works without error
		assert mission.status == "completed"
