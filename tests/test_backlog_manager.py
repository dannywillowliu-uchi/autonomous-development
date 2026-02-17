"""Tests for backlog_manager module (extracted from continuous_controller)."""

from __future__ import annotations

import json

from mission_control.backlog_manager import BacklogManager
from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import BacklogItem, Handoff, Mission, WorkUnit


class TestLoadBacklogObjective:
	def test_no_pending_items_returns_none(self, config: MissionConfig, db: Database) -> None:
		mgr = BacklogManager(db, config)
		result = mgr.load_backlog_objective()
		assert result is None
		assert mgr.backlog_item_ids == []

	def test_loads_pending_items(self, config: MissionConfig, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(
			id="bl1", title="Add auth", description="Implement auth module",
			priority_score=8.0, track="feature", status="pending",
		))
		db.insert_backlog_item(BacklogItem(
			id="bl2", title="Fix tests", description="Fix broken tests",
			priority_score=6.0, track="quality", status="pending",
		))

		mgr = BacklogManager(db, config)
		result = mgr.load_backlog_objective()

		assert result is not None
		assert "Add auth" in result
		assert "Fix tests" in result
		assert "backlog_item_id=bl1" in result
		assert mgr.backlog_item_ids == ["bl1", "bl2"]

		# Items should be marked in_progress
		item1 = db.get_backlog_item("bl1")
		assert item1 is not None
		assert item1.status == "in_progress"
		item2 = db.get_backlog_item("bl2")
		assert item2 is not None
		assert item2.status == "in_progress"

	def test_respects_limit(self, config: MissionConfig, db: Database) -> None:
		for i in range(5):
			db.insert_backlog_item(BacklogItem(
				id=f"bl{i}", title=f"Item {i}", description=f"Desc {i}",
				priority_score=float(10 - i), track="feature", status="pending",
			))

		mgr = BacklogManager(db, config)
		result = mgr.load_backlog_objective(limit=2)

		assert result is not None
		assert len(mgr.backlog_item_ids) == 2

	def test_skips_non_pending(self, config: MissionConfig, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(
			id="bl1", title="Completed", description="Already done",
			priority_score=8.0, track="feature", status="completed",
		))
		db.insert_backlog_item(BacklogItem(
			id="bl2", title="Pending", description="To do",
			priority_score=6.0, track="quality", status="pending",
		))

		mgr = BacklogManager(db, config)
		result = mgr.load_backlog_objective()

		assert result is not None
		assert "Pending" in result
		assert "Completed" not in result
		assert mgr.backlog_item_ids == ["bl2"]


class TestUpdateBacklogOnCompletion:
	def test_marks_completed_on_success(self, config: MissionConfig, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(
			id="bl1", title="Task A", priority_score=5.0,
			track="feature", status="in_progress",
		))

		mgr = BacklogManager(db, config)
		mgr.backlog_item_ids = ["bl1"]
		mgr.update_backlog_on_completion(objective_met=True, handoffs=[])

		item = db.get_backlog_item("bl1")
		assert item is not None
		assert item.status == "completed"

	def test_resets_to_pending_on_failure(self, config: MissionConfig, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(
			id="bl1", title="Task A", priority_score=5.0,
			track="feature", status="in_progress",
		))

		mgr = BacklogManager(db, config)
		mgr.backlog_item_ids = ["bl1"]
		mgr.update_backlog_on_completion(objective_met=False, handoffs=[])

		item = db.get_backlog_item("bl1")
		assert item is not None
		assert item.status == "pending"
		assert item.attempt_count == 1

	def test_stores_failure_reasons(self, config: MissionConfig, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(
			id="bl1", title="Task A", priority_score=5.0,
			track="feature", status="in_progress",
		))

		handoff = Handoff(
			id="h1", work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="failed", summary="Import error",
			concerns=json.dumps(["Could not import module X"]),
		)

		mgr = BacklogManager(db, config)
		mgr.backlog_item_ids = ["bl1"]
		mgr.update_backlog_on_completion(objective_met=False, handoffs=[handoff])

		item = db.get_backlog_item("bl1")
		assert item is not None
		assert "Could not import module X" in item.last_failure_reason

	def test_no_items_does_nothing(self, config: MissionConfig, db: Database) -> None:
		mgr = BacklogManager(db, config)
		mgr.backlog_item_ids = []
		mgr.update_backlog_on_completion(objective_met=True, handoffs=[])

	def test_missing_item_skipped(self, config: MissionConfig, db: Database) -> None:
		mgr = BacklogManager(db, config)
		mgr.backlog_item_ids = ["nonexistent"]
		mgr.update_backlog_on_completion(objective_met=True, handoffs=[])


class TestUpdateBacklogFromCompletion:
	def _setup_backlog(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(
			id="bl1", title="Add authentication module",
			description="Implement JWT auth",
			priority_score=8.0, track="feature", status="in_progress",
		))

	def test_merged_unit_marks_completed(self, config: MissionConfig, db: Database) -> None:
		self._setup_backlog(db)
		db.insert_mission(Mission(id="m1", objective="test"))

		unit = WorkUnit(id="wu1", plan_id="p1", title="Add authentication module")
		mgr = BacklogManager(db, config)
		mgr.update_backlog_from_completion(unit, merged=True, handoff=None, mission_id="m1")

		item = db.get_backlog_item("bl1")
		assert item is not None
		assert item.status == "completed"
		assert item.source_mission_id == "m1"

	def test_failed_max_attempts_records_failure(self, config: MissionConfig, db: Database) -> None:
		self._setup_backlog(db)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Add authentication module",
			attempt=3, max_attempts=3, output_summary="Tests failed",
		)
		mgr = BacklogManager(db, config)
		mgr.update_backlog_from_completion(unit, merged=False, handoff=None, mission_id="m1")

		item = db.get_backlog_item("bl1")
		assert item is not None
		assert item.attempt_count == 1
		assert "Tests failed" in item.last_failure_reason

	def test_partial_completion_appends_context(self, config: MissionConfig, db: Database) -> None:
		self._setup_backlog(db)

		handoff = Handoff(
			id="h1", work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="failed",
			discoveries=json.dumps(["Found pattern X"]),
			concerns=json.dumps(["Watch out for Y"]),
		)
		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Add authentication module",
			attempt=1, max_attempts=3,
		)
		mgr = BacklogManager(db, config)
		mgr.update_backlog_from_completion(unit, merged=False, handoff=handoff, mission_id="m1")

		item = db.get_backlog_item("bl1")
		assert item is not None
		assert item.status == "in_progress"
		assert "Found pattern X" in item.description

	def test_no_matching_item_does_nothing(self, config: MissionConfig, db: Database) -> None:
		unit = WorkUnit(id="wu1", plan_id="p1", title="Completely unrelated task")
		mgr = BacklogManager(db, config)
		mgr.update_backlog_from_completion(unit, merged=True, handoff=None, mission_id="m1")

	def test_short_title_words_skipped(self, config: MissionConfig, db: Database) -> None:
		"""Unit titles with only short words (<=2 chars) should be skipped."""
		self._setup_backlog(db)
		unit = WorkUnit(id="wu1", plan_id="p1", title="do it")
		mgr = BacklogManager(db, config)
		mgr.update_backlog_from_completion(unit, merged=True, handoff=None, mission_id="m1")

		# "do" and "it" are both <= 2 chars, so no matching happens
		item = db.get_backlog_item("bl1")
		assert item is not None
		assert item.status == "in_progress"  # unchanged
