"""Tests for MISSION_STATE.md generation in ContinuousController.

Covers:
- Initial generation produces valid markdown with objective and empty sections
- Update after completion adds timestamped entry and changelog
- Update after failure adds timestamped entry
- Changelog accumulates entries across multiple updates
"""

from __future__ import annotations

import json

from mission_control.config import MissionConfig
from mission_control.continuous_controller import ContinuousController
from mission_control.db import Database
from mission_control.models import Epoch, Handoff, Mission, Plan, WorkUnit


def _insert_unit_with_handoff(
	db: Database,
	*,
	unit_id: str,
	plan_id: str = "p1",
	epoch_id: str = "e1",
	mission_id: str = "m1",
	title: str = "",
	unit_status: str = "completed",
	commit_hash: str | None = None,
	finished_at: str | None = None,
	handoff_status: str = "completed",
	summary: str = "",
	files_changed: list[str] | None = None,
	concerns: list[str] | None = None,
) -> None:
	"""Helper to insert a Plan, WorkUnit, and Handoff with proper FK relationships."""
	# Ensure plan exists (ignore if already inserted)
	try:
		db.insert_plan(Plan(id=plan_id, objective="test"))
	except Exception:
		pass

	unit = WorkUnit(
		id=unit_id, plan_id=plan_id, title=title,
		status=unit_status, commit_hash=commit_hash,
		finished_at=finished_at, epoch_id=epoch_id,
	)
	db.insert_work_unit(unit)

	handoff = Handoff(
		work_unit_id=unit_id, epoch_id=epoch_id,
		status=handoff_status, summary=summary,
		files_changed=json.dumps(files_changed or []),
		concerns=json.dumps(concerns or []),
	)
	db.insert_handoff(handoff)


class TestInitialGeneration:
	"""Test that initial _update_mission_state produces valid markdown."""

	def test_initial_state_has_objective(self, config: MissionConfig, db: Database) -> None:
		"""Initial MISSION_STATE.md contains the objective."""
		mission = Mission(id="m1", objective="Build the widget", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "# Mission State" in content
		assert "Build the widget" in content

	def test_initial_state_has_remaining_section(self, config: MissionConfig, db: Database) -> None:
		"""Initial state includes the Remaining section for the planner."""
		mission = Mission(id="m1", objective="Do stuff", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Remaining" in content
		assert "planner should focus" in content

	def test_initial_state_no_completed_section(self, config: MissionConfig, db: Database) -> None:
		"""Initial state has no Completed section when nothing has been done."""
		mission = Mission(id="m1", objective="Do stuff", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Completed" not in content

	def test_initial_state_no_changelog(self, config: MissionConfig, db: Database) -> None:
		"""Initial state has no Changelog section when nothing has happened."""
		mission = Mission(id="m1", objective="Do stuff", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Changelog" not in content


class TestCompletionWithTimestamp:
	"""Test that completed units get timestamped entries."""

	def test_completed_entry_has_timestamp(self, config: MissionConfig, db: Database) -> None:
		"""Completed handoff entry includes the work unit's finished_at timestamp."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		_insert_unit_with_handoff(
			db,
			unit_id="unit1234abcd",
			title="Add feature",
			unit_status="completed",
			commit_hash="abc123",
			finished_at="2024-01-15T10:30:00+00:00",
			summary="Added the feature",
			files_changed=["src/foo.py"],
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Completed" in content
		assert "unit1234" in content
		assert "2024-01-15T10:30:00" in content
		assert "Added the feature" in content

	def test_completed_entry_has_files(self, config: MissionConfig, db: Database) -> None:
		"""Completed entry includes file list."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		_insert_unit_with_handoff(
			db,
			unit_id="unitabcd1234",
			title="Fix bug",
			unit_status="completed",
			commit_hash="def456",
			finished_at="2024-02-20T14:00:00+00:00",
			summary="Fixed the bug",
			files_changed=["src/bar.py", "tests/test_bar.py"],
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "src/bar.py" in content
		assert "tests/test_bar.py" in content

	def test_changelog_entry_for_merged_unit(self, config: MissionConfig, db: Database) -> None:
		"""Changelog section appears when entries have been appended."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._state_changelog.append(
			"- 2024-01-15T10:30:00 | unit1234 merged (commit: abc123) -- Added feature"
		)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Changelog" in content
		assert "unit1234 merged (commit: abc123)" in content
		assert "2024-01-15T10:30:00" in content


class TestFailureWithTimestamp:
	"""Test that failed units get timestamped entries."""

	def test_failed_entry_has_timestamp(self, config: MissionConfig, db: Database) -> None:
		"""Failed handoff entry includes the work unit's finished_at timestamp."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		_insert_unit_with_handoff(
			db,
			unit_id="failunit1234",
			title="Broken thing",
			unit_status="failed",
			finished_at="2024-03-10T08:45:00+00:00",
			handoff_status="failed",
			summary="It broke",
			concerns=["Merge conflict on models.py"],
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Failed" in content
		assert "failunit" in content
		assert "2024-03-10T08:45:00" in content
		assert "Merge conflict on models.py" in content

	def test_changelog_entry_for_failed_unit(self, config: MissionConfig, db: Database) -> None:
		"""Changelog includes failed unit entries."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._state_changelog.append(
			"- 2024-03-10T08:45:00 | failunit failed -- It broke"
		)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Changelog" in content
		assert "failunit failed" in content


class TestChangelogAccumulation:
	"""Test that changelog accumulates entries across multiple updates."""

	def test_changelog_accumulates_multiple_entries(self, config: MissionConfig, db: Database) -> None:
		"""Multiple changelog entries persist across _update_mission_state calls."""
		mission = Mission(id="m1", objective="Test accumulation", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)

		# First update
		ctrl._state_changelog.append(
			"- 2024-01-15T10:00:00 | unit0001 merged (commit: aaa111) -- First feature"
		)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Changelog" in content
		assert "unit0001 merged" in content

		# Second update
		ctrl._state_changelog.append(
			"- 2024-01-15T11:00:00 | unit0002 merged (commit: bbb222) -- Second feature"
		)
		ctrl._update_mission_state(mission)

		content = state_path.read_text()
		assert "unit0001 merged" in content
		assert "unit0002 merged" in content

		# Third update (failure)
		ctrl._state_changelog.append(
			"- 2024-01-15T12:00:00 | unit0003 failed -- Merge conflict"
		)
		ctrl._update_mission_state(mission)

		content = state_path.read_text()
		assert "unit0001 merged" in content
		assert "unit0002 merged" in content
		assert "unit0003 failed" in content

	def test_changelog_preserves_order(self, config: MissionConfig, db: Database) -> None:
		"""Changelog entries appear in chronological order (insertion order)."""
		mission = Mission(id="m1", objective="Test order", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._state_changelog.append("- 2024-01-15T10:00:00 | first entry")
		ctrl._state_changelog.append("- 2024-01-15T11:00:00 | second entry")
		ctrl._state_changelog.append("- 2024-01-15T12:00:00 | third entry")
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		first_pos = content.index("first entry")
		second_pos = content.index("second entry")
		third_pos = content.index("third entry")
		assert first_pos < second_pos < third_pos

	def test_no_changelog_when_empty(self, config: MissionConfig, db: Database) -> None:
		"""No Changelog section when _state_changelog is empty."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Changelog" not in content


class TestTimestampFallback:
	"""Test timestamp handling when finished_at is missing."""

	def test_no_timestamp_when_unit_not_found(self, config: MissionConfig, db: Database) -> None:
		"""Entry is still created when work unit is not found in DB (no timestamp)."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		# Insert dummy work unit, then handoff, then remove unit (FK off temporarily)
		db.insert_plan(Plan(id="p1", objective="test"))
		dummy = WorkUnit(id="ghost12345ab", plan_id="p1", epoch_id="e1")
		db.insert_work_unit(dummy)

		handoff = Handoff(
			work_unit_id="ghost12345ab", epoch_id="e1",
			status="completed", summary="Ghost unit",
			files_changed=json.dumps(["src/ghost.py"]),
		)
		db.insert_handoff(handoff)

		# Disable FK checks to delete the work unit, then re-enable
		db.conn.execute("PRAGMA foreign_keys = OFF")
		db.conn.execute("DELETE FROM work_units WHERE id = 'ghost12345ab'")
		db.conn.execute("PRAGMA foreign_keys = ON")

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "## Completed" in content
		assert "ghost123" in content
		assert "Ghost unit" in content

	def test_no_timestamp_when_finished_at_is_none(self, config: MissionConfig, db: Database) -> None:
		"""Entry omits timestamp when work unit's finished_at is None."""
		mission = Mission(id="m1", objective="Test", status="running")
		db.insert_mission(mission)
		epoch = Epoch(id="e1", mission_id="m1")
		db.insert_epoch(epoch)

		_insert_unit_with_handoff(
			db,
			unit_id="notime123456",
			title="No time",
			unit_status="completed",
			commit_hash="xyz",
			finished_at=None,
			summary="Done without timestamp",
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_mission_state(mission)

		state_path = config.target.resolved_path / "MISSION_STATE.md"
		content = state_path.read_text()
		assert "notime12" in content
		assert "Done without timestamp" in content
		# No parenthesized timestamp
		assert "()" not in content
