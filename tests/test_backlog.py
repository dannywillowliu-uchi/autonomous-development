"""Tests for backlog item CRUD and query operations."""

from __future__ import annotations

from mission_control.db import Database
from mission_control.models import BacklogItem


class TestBacklogCRUD:
	def test_insert_and_get_backlog_item(self, db: Database) -> None:
		item = BacklogItem(
			id="bl1",
			title="Add auth module",
			description="Implement JWT authentication",
			priority_score=8.5,
			impact=9,
			effort=3,
			track="feature",
			status="pending",
			source_mission_id="m1",
			attempt_count=1,
			last_failure_reason="timeout",
			pinned_score=10.0,
			depends_on="bl0",
			tags="auth,security",
		)
		db.insert_backlog_item(item)
		result = db.get_backlog_item("bl1")
		assert result is not None
		assert result.id == "bl1"
		assert result.title == "Add auth module"
		assert result.description == "Implement JWT authentication"
		assert result.priority_score == 8.5
		assert result.impact == 9
		assert result.effort == 3
		assert result.track == "feature"
		assert result.status == "pending"
		assert result.source_mission_id == "m1"
		assert result.attempt_count == 1
		assert result.last_failure_reason == "timeout"
		assert result.pinned_score == 10.0
		assert result.depends_on == "bl0"
		assert result.tags == "auth,security"
		assert result.created_at == item.created_at
		assert result.updated_at == item.updated_at

	def test_update_backlog_item(self, db: Database) -> None:
		item = BacklogItem(id="bl2", title="Original", track="quality")
		db.insert_backlog_item(item)
		item.title = "Updated title"
		item.description = "New description"
		item.priority_score = 7.0
		item.impact = 8
		item.effort = 2
		item.track = "security"
		item.status = "in_progress"
		item.tags = "refactor"
		db.update_backlog_item(item)
		result = db.get_backlog_item("bl2")
		assert result is not None
		assert result.title == "Updated title"
		assert result.description == "New description"
		assert result.priority_score == 7.0
		assert result.impact == 8
		assert result.effort == 2
		assert result.track == "security"
		assert result.status == "in_progress"
		assert result.tags == "refactor"

	def test_get_nonexistent_backlog_item(self, db: Database) -> None:
		assert db.get_backlog_item("does_not_exist") is None

	def test_backlog_item_defaults(self, db: Database) -> None:
		item = BacklogItem(id="bl_def")
		db.insert_backlog_item(item)
		result = db.get_backlog_item("bl_def")
		assert result is not None
		assert result.title == ""
		assert result.description == ""
		assert result.priority_score == 0.0
		assert result.impact == 5
		assert result.effort == 5
		assert result.track == ""
		assert result.status == "pending"
		assert result.source_mission_id == ""
		assert result.attempt_count == 0
		assert result.last_failure_reason == ""
		assert result.pinned_score is None
		assert result.depends_on == ""
		assert result.tags == ""


class TestBacklogQueries:
	def test_list_backlog_items_all(self, db: Database) -> None:
		for i in range(5):
			db.insert_backlog_item(BacklogItem(id=f"bl{i}", title=f"Item {i}", priority_score=float(i)))
		items = db.list_backlog_items()
		assert len(items) == 5
		# Ordered by priority_score DESC
		scores = [it.priority_score for it in items]
		assert scores == sorted(scores, reverse=True)

	def test_list_backlog_items_by_status(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(id="bl1", title="Pending", status="pending"))
		db.insert_backlog_item(BacklogItem(id="bl2", title="Done", status="completed"))
		db.insert_backlog_item(BacklogItem(id="bl3", title="Also pending", status="pending"))
		results = db.list_backlog_items(status="pending")
		assert len(results) == 2
		assert all(r.status == "pending" for r in results)

	def test_list_backlog_items_by_track(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(id="bl1", title="Feat", track="feature"))
		db.insert_backlog_item(BacklogItem(id="bl2", title="Sec", track="security"))
		db.insert_backlog_item(BacklogItem(id="bl3", title="Feat2", track="feature"))
		results = db.list_backlog_items(track="feature")
		assert len(results) == 2
		assert all(r.track == "feature" for r in results)

	def test_list_backlog_items_limit(self, db: Database) -> None:
		for i in range(10):
			db.insert_backlog_item(BacklogItem(id=f"bl{i}", title=f"Item {i}"))
		results = db.list_backlog_items(limit=3)
		assert len(results) == 3

	def test_get_pending_backlog_ordering(self, db: Database) -> None:
		"""Pinned items use pinned_score for ordering; unpinned use priority_score."""
		db.insert_backlog_item(BacklogItem(id="low", title="Low", priority_score=1.0, status="pending"))
		db.insert_backlog_item(BacklogItem(id="high", title="High", priority_score=9.0, status="pending"))
		db.insert_backlog_item(BacklogItem(
			id="pinned", title="Pinned", priority_score=2.0, pinned_score=20.0, status="pending",
		))
		results = db.get_pending_backlog()
		assert results[0].id == "pinned"  # pinned_score=20 sorts highest
		assert results[1].id == "high"  # priority_score=9
		assert results[2].id == "low"  # priority_score=1

	def test_get_backlog_items_for_mission(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(id="bl1", title="M1 item", source_mission_id="m1"))
		db.insert_backlog_item(BacklogItem(id="bl2", title="M2 item", source_mission_id="m2"))
		db.insert_backlog_item(BacklogItem(id="bl3", title="M1 other", source_mission_id="m1"))
		results = db.get_backlog_items_for_mission("m1")
		assert len(results) == 2
		assert all(r.source_mission_id == "m1" for r in results)

	def test_search_backlog_items(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(id="bl1", title="Fix auth bug", tags="auth,security"))
		db.insert_backlog_item(BacklogItem(id="bl2", title="Add logging", description="structured logging for auth"))
		db.insert_backlog_item(BacklogItem(id="bl3", title="Update docs"))
		results = db.search_backlog_items(["auth"])
		assert len(results) == 2
		ids = {r.id for r in results}
		assert "bl1" in ids
		assert "bl2" in ids

	def test_search_backlog_items_empty_keywords(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(id="bl1", title="Anything"))
		assert db.search_backlog_items([]) == []


class TestBacklogMutations:
	def test_update_attempt_count(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(id="bl1", title="Flaky", attempt_count=0))
		db.update_attempt_count("bl1", failure_reason="merge conflict")
		result = db.get_backlog_item("bl1")
		assert result is not None
		assert result.attempt_count == 1
		assert result.last_failure_reason == "merge conflict"
		# Increment again
		db.update_attempt_count("bl1", failure_reason="timeout")
		result = db.get_backlog_item("bl1")
		assert result is not None
		assert result.attempt_count == 2
		assert result.last_failure_reason == "timeout"

	def test_update_attempt_count_no_reason(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(id="bl1", title="Test"))
		db.update_attempt_count("bl1")
		result = db.get_backlog_item("bl1")
		assert result is not None
		assert result.attempt_count == 1
		assert result.last_failure_reason is None

	def test_defer_backlog_item(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(id="bl1", title="Defer me", status="pending"))
		db.defer_backlog_item("bl1")
		result = db.get_backlog_item("bl1")
		assert result is not None
		assert result.status == "deferred"

	def test_pin_backlog_score(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(id="bl1", title="Pin me", priority_score=5.0))
		db.pin_backlog_score("bl1", 99.0)
		result = db.get_backlog_item("bl1")
		assert result is not None
		assert result.pinned_score == 99.0
		assert result.priority_score == 5.0  # unchanged

	def test_pin_backlog_score_affects_ordering(self, db: Database) -> None:
		db.insert_backlog_item(BacklogItem(id="bl1", title="Low", priority_score=1.0, status="pending"))
		db.insert_backlog_item(BacklogItem(id="bl2", title="High", priority_score=9.0, status="pending"))
		# Pin the low-priority item above the high one
		db.pin_backlog_score("bl1", 50.0)
		results = db.get_pending_backlog()
		assert results[0].id == "bl1"
		assert results[1].id == "bl2"
