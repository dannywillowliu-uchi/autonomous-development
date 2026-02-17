"""Tests for planner_context module (extracted from continuous_controller)."""

from __future__ import annotations

import json
from pathlib import Path

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import Epoch, Handoff, Mission, Plan, WorkUnit
from mission_control.planner_context import build_planner_context, update_mission_state


class TestBuildPlannerContext:
	def test_no_handoffs_returns_empty(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		result = build_planner_context(db, "m1")
		assert result == ""

	def test_single_completed_handoff(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)
		handoff = Handoff(
			id="h1", work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="completed", summary="Did the thing",
		)
		db.insert_handoff(handoff)

		result = build_planner_context(db, "m1")
		assert "## Recent Handoff Summaries" in result
		assert "wu1" in result[:200]
		assert "Did the thing" in result
		assert "1 merged, 0 failed" in result

	def test_mixed_statuses(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		for i, status in enumerate(["completed", "failed", "completed"]):
			uid = f"wu{i}"
			unit = WorkUnit(id=uid, plan_id="p1", title=f"Task {i}")
			db.insert_work_unit(unit)
			handoff = Handoff(
				id=f"h{i}", work_unit_id=uid, round_id="", epoch_id="ep1",
				status=status, summary=f"Summary {i}",
			)
			db.insert_handoff(handoff)

		result = build_planner_context(db, "m1")
		assert "2 merged, 1 failed" in result

	def test_discoveries_and_concerns_included(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)
		handoff = Handoff(
			id="h1", work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="completed", summary="Done",
			discoveries=json.dumps(["Found pattern X"]),
			concerns=json.dumps(["Watch out for Y"]),
		)
		db.insert_handoff(handoff)

		result = build_planner_context(db, "m1")
		assert "Found pattern X" in result
		assert "Watch out for Y" in result

	def test_nonexistent_mission_returns_empty(self, db: Database) -> None:
		result = build_planner_context(db, "nonexistent")
		assert result == ""

	def test_db_error_returns_empty(self, config: MissionConfig) -> None:
		"""If db.get_recent_handoffs raises, returns empty string."""
		from unittest.mock import MagicMock
		mock_db = MagicMock()
		mock_db.get_recent_handoffs.side_effect = RuntimeError("DB down")
		result = build_planner_context(mock_db, "m1")
		assert result == ""


class TestUpdateMissionState:
	def test_writes_mission_state_file(self, config: MissionConfig, db: Database, tmp_path: Path) -> None:
		config.target.path = str(tmp_path)
		db.insert_mission(Mission(id="m1", objective="Build the thing"))

		mission = Mission(id="m1", objective="Build the thing")
		update_mission_state(db, mission, config)

		state_path = tmp_path / "MISSION_STATE.md"
		assert state_path.exists()
		content = state_path.read_text()
		assert "# Mission State" in content
		assert "Build the thing" in content
		assert "## Remaining" in content

	def test_includes_completed_handoffs(self, config: MissionConfig, db: Database, tmp_path: Path) -> None:
		config.target.path = str(tmp_path)
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task", finished_at="2025-01-01T12:00:00")
		db.insert_work_unit(unit)
		handoff = Handoff(
			id="h1", work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="completed", summary="Done with it",
			files_changed=json.dumps(["src/main.py"]),
		)
		db.insert_handoff(handoff)

		mission = Mission(id="m1", objective="test")
		update_mission_state(db, mission, config)

		content = (tmp_path / "MISSION_STATE.md").read_text()
		assert "## Completed" in content
		assert "wu1" in content[:500]
		assert "Done with it" in content
		assert "src/main.py" in content

	def test_includes_failed_handoffs(self, config: MissionConfig, db: Database, tmp_path: Path) -> None:
		config.target.path = str(tmp_path)
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)
		handoff = Handoff(
			id="h1", work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="failed", summary="Broke",
			concerns=json.dumps(["Something went wrong"]),
		)
		db.insert_handoff(handoff)

		mission = Mission(id="m1", objective="test")
		update_mission_state(db, mission, config)

		content = (tmp_path / "MISSION_STATE.md").read_text()
		assert "## Failed" in content
		assert "Something went wrong" in content

	def test_includes_changelog(self, config: MissionConfig, db: Database, tmp_path: Path) -> None:
		config.target.path = str(tmp_path)
		db.insert_mission(Mission(id="m1", objective="test"))

		mission = Mission(id="m1", objective="test")
		changelog = ["- 2025-01-01 | abc12345 merged -- did stuff"]
		update_mission_state(db, mission, config, state_changelog=changelog)

		content = (tmp_path / "MISSION_STATE.md").read_text()
		assert "## Changelog" in content
		assert "abc12345 merged" in content

	def test_no_changelog_when_empty(self, config: MissionConfig, db: Database, tmp_path: Path) -> None:
		config.target.path = str(tmp_path)
		db.insert_mission(Mission(id="m1", objective="test"))

		mission = Mission(id="m1", objective="test")
		update_mission_state(db, mission, config, state_changelog=[])

		content = (tmp_path / "MISSION_STATE.md").read_text()
		assert "## Changelog" not in content

	def test_files_modified_section(self, config: MissionConfig, db: Database, tmp_path: Path) -> None:
		config.target.path = str(tmp_path)
		db.insert_mission(Mission(id="m1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
		db.insert_work_unit(unit)
		handoff = Handoff(
			id="h1", work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="completed", summary="Done",
			files_changed=json.dumps(["src/a.py", "src/b.py"]),
		)
		db.insert_handoff(handoff)

		mission = Mission(id="m1", objective="test")
		update_mission_state(db, mission, config)

		content = (tmp_path / "MISSION_STATE.md").read_text()
		assert "## Files Modified" in content
		assert "src/a.py" in content
		assert "src/b.py" in content
