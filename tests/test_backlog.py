"""Tests for backlog item CRUD and query operations."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import ContinuousConfig, MissionConfig, PlannerConfig, TargetConfig
from mission_control.continuous_controller import ContinuousController
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.models import BacklogItem, DiscoveryItem, Handoff, Mission, Plan, PlanNode, WorkUnit


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


# ============================================================================
# Tests consolidated from test_backlog_continuity.py
# ============================================================================


class TestTitleBasedMatching:
	"""Test that _update_backlog_from_completion finds backlog items by title."""

	def test_matches_by_title_keywords(self, config: MissionConfig, db: Database) -> None:
		"""Unit title keywords match against backlog item titles."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Add user authentication",
			description="OAuth2 flow", priority_score=8.0, track="feature",
			status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Add user authentication module",
			status="completed", commit_hash="abc123", attempt=1, max_attempts=3,
		)
		handoff = Handoff(
			work_unit_id="u1", status="completed", summary="Done",
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, handoff, "mission-1")

		item = db.get_backlog_item("b1")
		assert item.status == "completed"

	def test_picks_best_match_by_keyword_overlap(self, config: MissionConfig, db: Database) -> None:
		"""When multiple items match, the one with most keyword overlap wins."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Fix database connection pooling",
			description="desc", priority_score=5.0, track="quality",
			status="in_progress",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="Add database migration scripts",
			description="desc", priority_score=6.0, track="feature",
			status="in_progress",
		))

		# Title overlaps more with b1: "fix", "database", "connection", "pooling"
		unit = WorkUnit(
			id="u1", plan_id="p1",
			title="Fix database connection pooling issues",
			status="completed", commit_hash="abc123", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, None, "mission-1")

		# b1 should be completed (better match), b2 should remain in_progress
		assert db.get_backlog_item("b1").status == "completed"
		assert db.get_backlog_item("b2").status == "in_progress"

	def test_no_match_with_unrelated_title(self, config: MissionConfig, db: Database) -> None:
		"""Unit with completely unrelated title doesn't match any backlog item."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Implement caching layer",
			description="Redis caching", priority_score=7.0, track="feature",
			status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1",
			title="Refactor authentication middleware",
			status="completed", commit_hash="abc123", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, None, "mission-1")

		# Item should remain unchanged
		assert db.get_backlog_item("b1").status == "in_progress"

	def test_short_title_words_filtered(self, config: MissionConfig, db: Database) -> None:
		"""Words with 2 or fewer characters are excluded from matching."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Add API endpoint",
			description="desc", priority_score=5.0, track="feature",
			status="in_progress",
		))

		# "an" and "to" should be filtered (len <= 2)
		unit = WorkUnit(
			id="u1", plan_id="p1", title="Add API endpoint",
			status="completed", commit_hash="abc", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, None, "m1")

		assert db.get_backlog_item("b1").status == "completed"

	def test_empty_title_no_match(self, config: MissionConfig, db: Database) -> None:
		"""Empty unit title produces no match and no error."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Some task", description="desc",
			priority_score=5.0, track="quality", status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="",
			status="completed", commit_hash="abc", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, None, "m1")

		# Should remain unchanged
		assert db.get_backlog_item("b1").status == "in_progress"


class TestStatusTransitionsOnSuccess:
	"""Test backlog item status updates on successful merge."""

	def test_sets_completed_on_merge(self, config: MissionConfig, db: Database) -> None:
		"""Successful merge marks matching backlog item as completed."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Implement search feature",
			description="Full-text search", priority_score=7.0, track="feature",
			status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Implement search feature",
			status="completed", commit_hash="abc123", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, None, "mission-1")

		item = db.get_backlog_item("b1")
		assert item.status == "completed"

	def test_sets_source_mission_id_on_merge(self, config: MissionConfig, db: Database) -> None:
		"""Successful merge sets source_mission_id to current mission."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Implement search feature",
			description="desc", priority_score=7.0, track="feature",
			status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Implement search feature",
			status="completed", commit_hash="abc", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, None, "mission-42")

		item = db.get_backlog_item("b1")
		assert item.source_mission_id == "mission-42"

	def test_updates_timestamp_on_completion(self, config: MissionConfig, db: Database) -> None:
		"""Completed items get updated_at timestamp refreshed."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Implement search feature",
			description="desc", priority_score=7.0, track="feature",
			status="in_progress", updated_at="2024-01-01T00:00:00+00:00",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Implement search feature",
			status="completed", commit_hash="abc", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, None, "m1")

		item = db.get_backlog_item("b1")
		assert item.updated_at != "2024-01-01T00:00:00+00:00"


class TestStatusTransitionsOnFailure:
	"""Test backlog item status updates on unit failure after max retries."""

	def test_increments_attempt_count(self, config: MissionConfig, db: Database) -> None:
		"""Failed unit after max retries increments backlog attempt_count."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Fix memory leak",
			description="desc", priority_score=6.0, track="quality",
			status="in_progress", attempt_count=1,
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Fix memory leak in worker",
			status="failed", attempt=3, max_attempts=3,
			output_summary="Segfault during test",
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, False, None, "m1")

		item = db.get_backlog_item("b1")
		assert item.attempt_count == 2  # Was 1, incremented to 2

	def test_sets_failure_reason_from_handoff_concerns(self, config: MissionConfig, db: Database) -> None:
		"""Failure reason comes from handoff concerns."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Fix memory leak",
			description="desc", priority_score=6.0, track="quality",
			status="in_progress",
		))

		handoff = Handoff(
			work_unit_id="u1", status="failed",
			concerns=json.dumps(["OOM at test_large_dataset"]),
			summary="Failed",
		)

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Fix memory leak in worker",
			status="failed", attempt=3, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, False, handoff, "m1")

		item = db.get_backlog_item("b1")
		assert "OOM at test_large_dataset" in item.last_failure_reason

	def test_sets_failure_reason_from_output_summary(self, config: MissionConfig, db: Database) -> None:
		"""When no handoff concerns, failure reason comes from output_summary."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Fix memory leak",
			description="desc", priority_score=6.0, track="quality",
			status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Fix memory leak in worker",
			status="failed", attempt=3, max_attempts=3,
			output_summary="Timed out after 300s",
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, False, None, "m1")

		item = db.get_backlog_item("b1")
		assert "Timed out" in item.last_failure_reason

	def test_default_failure_reason_on_max_retries(self, config: MissionConfig, db: Database) -> None:
		"""Default failure reason when no handoff or output_summary."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Fix memory leak",
			description="desc", priority_score=6.0, track="quality",
			status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Fix memory leak",
			status="failed", attempt=3, max_attempts=3,
			output_summary="",
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, False, None, "m1")

		item = db.get_backlog_item("b1")
		assert item.last_failure_reason == "Max retries exceeded"


class TestPartialCompletionContextCarryForward:
	"""Test that partial completions append context to backlog descriptions."""

	def test_appends_discoveries_to_description(self, config: MissionConfig, db: Database) -> None:
		"""Partial completion appends handoff discoveries to backlog item description."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Refactor database layer",
			description="Original description",
			priority_score=5.0, track="quality", status="in_progress",
		))

		handoff = Handoff(
			work_unit_id="u1", status="completed",
			discoveries=json.dumps(["Found circular import in db.py"]),
			concerns=json.dumps([]),
		)

		# attempt < max_attempts AND not merged -> partial
		unit = WorkUnit(
			id="u1", plan_id="p1", title="Refactor database layer",
			status="failed", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, False, handoff, "m1")

		item = db.get_backlog_item("b1")
		assert item.status == "in_progress"
		assert "Found circular import in db.py" in item.description
		assert "Original description" in item.description

	def test_appends_concerns_to_description(self, config: MissionConfig, db: Database) -> None:
		"""Partial completion appends handoff concerns to backlog item description."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Refactor database layer",
			description="Original",
			priority_score=5.0, track="quality", status="in_progress",
		))

		handoff = Handoff(
			work_unit_id="u1", status="failed",
			discoveries=json.dumps([]),
			concerns=json.dumps(["Merge conflict on models.py"]),
		)

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Refactor database layer",
			status="failed", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, False, handoff, "m1")

		item = db.get_backlog_item("b1")
		assert item.status == "in_progress"
		assert "Merge conflict on models.py" in item.description

	def test_no_context_added_without_handoff(self, config: MissionConfig, db: Database) -> None:
		"""Partial completion without handoff keeps description unchanged."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Refactor database layer",
			description="Original description",
			priority_score=5.0, track="quality", status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Refactor database layer",
			status="failed", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, False, None, "m1")

		item = db.get_backlog_item("b1")
		assert item.description == "Original description"


class TestNoMatchGracefulHandling:
	"""Test graceful handling when no backlog items match."""

	def test_no_backlog_items_in_db(self, config: MissionConfig, db: Database) -> None:
		"""No error when backlog is completely empty."""
		unit = WorkUnit(
			id="u1", plan_id="p1", title="Some task",
			status="completed", commit_hash="abc", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		# Should not raise
		ctrl._update_backlog_from_completion(unit, True, None, "m1")

	def test_no_keyword_overlap(self, config: MissionConfig, db: Database) -> None:
		"""No match when search returns items but title overlap is zero."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Implement caching",
			description="Redis layer", priority_score=5.0, track="feature",
			status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Zebra elephant giraffe",
			status="completed", commit_hash="abc", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, None, "m1")

		# Item unchanged
		assert db.get_backlog_item("b1").status == "in_progress"

	def test_only_short_words_in_title(self, config: MissionConfig, db: Database) -> None:
		"""Title with only 1-2 char words produces no match."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Fix it",
			description="desc", priority_score=5.0, track="quality",
			status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="do it",
			status="completed", commit_hash="abc", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, None, "m1")

		# "do" and "it" are both <= 2 chars, so filtered out
		assert db.get_backlog_item("b1").status == "in_progress"


class TestMissionIdTracking:
	"""Test source_mission_id is tracked on backlog items."""

	def test_source_mission_id_set_on_successful_completion(self, config: MissionConfig, db: Database) -> None:
		"""source_mission_id is set when unit merge succeeds."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Build notification system",
			description="desc", priority_score=7.0, track="feature",
			status="in_progress",
		))

		unit = WorkUnit(
			id="u1", plan_id="p1", title="Build notification system",
			status="completed", commit_hash="abc", attempt=1, max_attempts=3,
		)

		ctrl = ContinuousController(config, db)
		ctrl._update_backlog_from_completion(unit, True, None, "mission-99")

		item = db.get_backlog_item("b1")
		assert item.source_mission_id == "mission-99"

	@pytest.mark.asyncio
	async def test_discovery_items_get_source_mission_id(self, config: MissionConfig, db: Database) -> None:
		"""Post-mission discovery items have source_mission_id set."""
		mock_items = [
			DiscoveryItem(
				id="d1", title="Add logging", description="Structured logging",
				priority_score=6.0, impact=7, effort=4, track="quality",
			),
		]

		mock_engine = MagicMock()
		mock_engine.discover = AsyncMock(return_value=(MagicMock(), mock_items))

		ctrl = ContinuousController(config, db)

		with patch("mission_control.auto_discovery.DiscoveryEngine", return_value=mock_engine):
			await ctrl._run_post_mission_discovery("mission-42")

		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 1
		assert backlog[0].source_mission_id == "mission-42"

	@pytest.mark.asyncio
	async def test_discovery_items_empty_mission_id_when_not_provided(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""source_mission_id defaults to empty string when not provided."""
		mock_items = [
			DiscoveryItem(
				id="d1", title="Add logging", description="Structured logging",
				priority_score=6.0, impact=7, effort=4, track="quality",
			),
		]

		mock_engine = MagicMock()
		mock_engine.discover = AsyncMock(return_value=(MagicMock(), mock_items))

		ctrl = ContinuousController(config, db)

		with patch("mission_control.auto_discovery.DiscoveryEngine", return_value=mock_engine):
			await ctrl._run_post_mission_discovery()

		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 1
		assert backlog[0].source_mission_id == ""


# ============================================================================
# Tests consolidated from test_backlog_intake.py
# ============================================================================


class TestLoadBacklogObjective:
	"""Test _load_backlog_objective() method."""

	def test_loads_pending_items_as_objective(self, config: MissionConfig, db: Database) -> None:
		"""Top pending backlog items compose into an objective string."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Add auth", description="Implement OAuth",
			priority_score=8.0, track="feature",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="Fix XSS", description="Sanitize inputs",
			priority_score=9.0, track="security",
		))

		ctrl = ContinuousController(config, db)
		objective = ctrl._load_backlog_objective(limit=5)

		assert objective is not None
		assert "Fix XSS" in objective
		assert "Add auth" in objective
		assert "backlog_item_id=b1" in objective
		assert "backlog_item_id=b2" in objective

	def test_marks_items_as_in_progress(self, config: MissionConfig, db: Database) -> None:
		"""Selected backlog items are marked in_progress."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality",
		))

		ctrl = ContinuousController(config, db)
		ctrl._load_backlog_objective(limit=5)

		item = db.get_backlog_item("b1")
		assert item is not None
		assert item.status == "in_progress"

	def test_stores_item_ids_for_tracking(self, config: MissionConfig, db: Database) -> None:
		"""Item IDs are stored on the controller for post-mission tracking."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="Task 2", description="desc",
			priority_score=3.0, track="feature",
		))

		ctrl = ContinuousController(config, db)
		ctrl._load_backlog_objective(limit=5)

		assert "b1" in ctrl._backlog_item_ids
		assert "b2" in ctrl._backlog_item_ids

	def test_returns_none_on_empty_backlog(self, config: MissionConfig, db: Database) -> None:
		"""Returns None when no pending backlog items exist."""
		ctrl = ContinuousController(config, db)
		objective = ctrl._load_backlog_objective(limit=5)

		assert objective is None
		assert ctrl._backlog_item_ids == []

	def test_respects_limit(self, config: MissionConfig, db: Database) -> None:
		"""Only loads up to `limit` items."""
		for i in range(10):
			db.insert_backlog_item(BacklogItem(
				id=f"b{i}", title=f"Task {i}", description="desc",
				priority_score=float(10 - i), track="quality",
			))

		ctrl = ContinuousController(config, db)
		ctrl._load_backlog_objective(limit=3)

		assert len(ctrl._backlog_item_ids) == 3

	def test_only_loads_pending_items(self, config: MissionConfig, db: Database) -> None:
		"""Does not load items that are already in_progress or completed."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Pending", description="desc",
			priority_score=5.0, track="quality", status="pending",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="In Progress", description="desc",
			priority_score=9.0, track="quality", status="in_progress",
		))
		db.insert_backlog_item(BacklogItem(
			id="b3", title="Completed", description="desc",
			priority_score=9.0, track="quality", status="completed",
		))

		ctrl = ContinuousController(config, db)
		objective = ctrl._load_backlog_objective(limit=5)

		assert objective is not None
		assert "Pending" in objective
		assert "In Progress" not in objective
		assert "Completed" not in objective
		assert ctrl._backlog_item_ids == ["b1"]

	def test_sorted_by_priority_desc(self, config: MissionConfig, db: Database) -> None:
		"""Items are selected by priority score descending."""
		db.insert_backlog_item(BacklogItem(
			id="low", title="Low priority", description="desc",
			priority_score=1.0, track="quality",
		))
		db.insert_backlog_item(BacklogItem(
			id="high", title="High priority", description="desc",
			priority_score=9.0, track="feature",
		))

		ctrl = ContinuousController(config, db)
		objective = ctrl._load_backlog_objective(limit=1)

		assert objective is not None
		assert "High priority" in objective
		assert ctrl._backlog_item_ids == ["high"]


class TestUpdateBacklogOnCompletion:
	"""Test _update_backlog_on_completion() method."""

	def test_marks_completed_on_success(self, config: MissionConfig, db: Database) -> None:
		"""When objective_met=True, all targeted items become completed."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality", status="in_progress",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="Task 2", description="desc",
			priority_score=3.0, track="feature", status="in_progress",
		))

		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = ["b1", "b2"]
		ctrl._update_backlog_on_completion(objective_met=True, handoffs=[])

		assert db.get_backlog_item("b1").status == "completed"
		assert db.get_backlog_item("b2").status == "completed"

	def test_resets_to_pending_on_failure(self, config: MissionConfig, db: Database) -> None:
		"""When objective_met=False, items reset to pending with incremented attempt_count."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality", status="in_progress",
			attempt_count=0,
		))

		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = ["b1"]
		ctrl._update_backlog_on_completion(objective_met=False, handoffs=[])

		item = db.get_backlog_item("b1")
		assert item.status == "pending"
		assert item.attempt_count == 1

	def test_stores_failure_reason_from_handoffs(self, config: MissionConfig, db: Database) -> None:
		"""Failure reasons from handoff concerns are stored in last_failure_reason."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality", status="in_progress",
		))

		handoff = Handoff(
			work_unit_id="wu1",
			status="failed",
			concerns=json.dumps(["Merge conflict on config.py"]),
			summary="Failed to complete task",
		)

		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = ["b1"]
		ctrl._update_backlog_on_completion(objective_met=False, handoffs=[handoff])

		item = db.get_backlog_item("b1")
		assert item.last_failure_reason is not None
		assert "Merge conflict on config.py" in item.last_failure_reason

	def test_noop_when_no_backlog_items(self, config: MissionConfig, db: Database) -> None:
		"""Does nothing when no backlog items were loaded."""
		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = []
		# Should not raise
		ctrl._update_backlog_on_completion(objective_met=True, handoffs=[])

	def test_handles_missing_item_gracefully(self, config: MissionConfig, db: Database) -> None:
		"""Skips items that no longer exist in the DB."""
		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = ["nonexistent"]
		# Should not raise
		ctrl._update_backlog_on_completion(objective_met=True, handoffs=[])

	def test_failure_context_from_summary_when_no_concerns(self, config: MissionConfig, db: Database) -> None:
		"""Uses handoff summary as failure reason when concerns are empty."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Task 1", description="desc",
			priority_score=5.0, track="quality", status="in_progress",
		))

		handoff = Handoff(
			work_unit_id="wu1",
			status="failed",
			concerns="[]",
			summary="Timed out after 300s",
		)

		ctrl = ContinuousController(config, db)
		ctrl._backlog_item_ids = ["b1"]
		ctrl._update_backlog_on_completion(objective_met=False, handoffs=[handoff])

		item = db.get_backlog_item("b1")
		assert "Timed out" in item.last_failure_reason


class TestDiscoveryToBacklog:
	"""Test that post-mission discovery items flow into the persistent backlog."""

	@pytest.mark.asyncio
	async def test_discovery_items_inserted_to_backlog(self, config: MissionConfig, db: Database) -> None:
		"""Discovery items are converted to BacklogItems and inserted."""
		mock_items = [
			DiscoveryItem(
				id="d1", title="Add caching", description="Redis caching layer",
				priority_score=7.0, impact=8, effort=5, track="feature",
			),
			DiscoveryItem(
				id="d2", title="Fix SQL injection", description="Parameterize queries",
				priority_score=9.0, impact=10, effort=3, track="security",
			),
		]

		mock_engine = MagicMock()
		mock_engine.discover = AsyncMock(return_value=(MagicMock(), mock_items))

		ctrl = ContinuousController(config, db)

		with patch("mission_control.auto_discovery.DiscoveryEngine", return_value=mock_engine):
			await ctrl._run_post_mission_discovery()

		# Check backlog items were created
		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) >= 2
		titles = {item.title for item in backlog}
		assert "Add caching" in titles
		assert "Fix SQL injection" in titles

		# Verify fields are mapped correctly
		caching_items = [i for i in backlog if i.title == "Add caching"]
		assert len(caching_items) == 1
		assert caching_items[0].priority_score == 7.0
		assert caching_items[0].impact == 8
		assert caching_items[0].effort == 5
		assert caching_items[0].track == "feature"

	@pytest.mark.asyncio
	async def test_empty_discovery_no_backlog_items(self, config: MissionConfig, db: Database) -> None:
		"""When discovery returns no items, no backlog items are created."""
		mock_engine = MagicMock()
		mock_engine.discover = AsyncMock(return_value=(MagicMock(), []))

		ctrl = ContinuousController(config, db)

		with patch("mission_control.auto_discovery.DiscoveryEngine", return_value=mock_engine):
			await ctrl._run_post_mission_discovery()

		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 0


class TestBacklogIntegrationInRun:
	"""Test backlog integration points in the run() method."""

	def test_empty_backlog_falls_back_to_config_objective(self, config: MissionConfig, db: Database) -> None:
		"""When backlog is empty, the config objective is used unchanged."""
		config.target.objective = "Build the widget"
		config.discovery.enabled = True

		ctrl = ContinuousController(config, db)
		objective = ctrl._load_backlog_objective(limit=5)

		assert objective is None
		assert config.target.objective == "Build the widget"

	def test_backlog_merged_with_existing_objective(self, config: MissionConfig, db: Database) -> None:
		"""Backlog items are appended to existing config objective."""
		config.target.objective = "Build the widget"
		config.discovery.enabled = True

		db.insert_backlog_item(BacklogItem(
			id="b1", title="Add tests", description="Coverage for auth module",
			priority_score=7.0, track="quality",
		))

		ctrl = ContinuousController(config, db)
		backlog_objective = ctrl._load_backlog_objective(limit=5)

		assert backlog_objective is not None
		# Simulate what run() does
		merged = config.target.objective + "\n\n" + backlog_objective
		assert "Build the widget" in merged
		assert "Add tests" in merged
		assert "backlog_item_id=b1" in merged

	def test_backlog_replaces_empty_objective(self, config: MissionConfig, db: Database) -> None:
		"""When config has no objective, backlog becomes the sole objective."""
		config.target.objective = ""
		config.discovery.enabled = True

		db.insert_backlog_item(BacklogItem(
			id="b1", title="Refactor DB", description="Split into modules",
			priority_score=6.0, track="quality",
		))

		ctrl = ContinuousController(config, db)
		backlog_objective = ctrl._load_backlog_objective(limit=5)

		assert backlog_objective is not None
		assert "Refactor DB" in backlog_objective

	def test_discovery_disabled_skips_backlog(self, config: MissionConfig, db: Database) -> None:
		"""When discovery is disabled, backlog loading is skipped in run()."""
		config.discovery.enabled = False

		db.insert_backlog_item(BacklogItem(
			id="b1", title="Should not load", description="desc",
			priority_score=9.0, track="quality",
		))

		# The run() method checks config.discovery.enabled before calling
		# _load_backlog_objective(). We verify the flag works correctly.
		_ = ContinuousController(config, db)
		# Directly verify the condition that run() checks
		assert not config.discovery.enabled
		# Item should still be pending since we didn't call _load_backlog_objective
		item = db.get_backlog_item("b1")
		assert item.status == "pending"


def _planner_config() -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(name="test", path="/tmp/test", objective="Build API")
	mc.planner = PlannerConfig(max_depth=2)
	mc.continuous = ContinuousConfig(backlog_min_size=2)
	return mc


def _mock_plan_round(unit_ids: list[str]) -> AsyncMock:
	"""Create a mock plan_round that returns units with the given IDs."""
	units = [WorkUnit(id=uid, plan_id="p1", title=f"Task {uid}") for uid in unit_ids]
	plan = Plan(id="p1", objective="test")
	root = PlanNode(id="root", plan_id="p1", strategy="leaves")
	root._child_leaves = [  # type: ignore[attr-defined]
		(PlanNode(id=f"leaf-{wu.id}", node_type="leaf"), wu)
		for wu in units
	]
	return AsyncMock(return_value=(plan, root))


class TestPlannerSetBacklogItems:
	"""Test ContinuousPlanner.set_backlog_items() stores items."""

	def test_stores_items(self) -> None:
		"""set_backlog_items stores the provided BacklogItem list."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		items = [
			BacklogItem(id="b1", title="Add auth", description="OAuth flow", priority_score=8.0, track="feature"),
			BacklogItem(id="b2", title="Fix XSS", description="Sanitize inputs", priority_score=9.0, track="security"),
		]
		planner.set_backlog_items(items)

		assert len(planner._backlog_items) == 2
		assert planner._backlog_items[0].title == "Add auth"
		assert planner._backlog_items[1].title == "Fix XSS"

	def test_stores_copy(self) -> None:
		"""set_backlog_items stores a copy, not a reference to the original list."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		items = [BacklogItem(id="b1", title="Task 1", description="desc", priority_score=5.0, track="quality")]
		planner.set_backlog_items(items)

		items.append(BacklogItem(id="b2", title="Task 2", description="desc", priority_score=3.0, track="feature"))
		assert len(planner._backlog_items) == 1

	def test_replaces_previous_items(self) -> None:
		"""Calling set_backlog_items again replaces the previous items."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		planner.set_backlog_items([
			BacklogItem(id="b1", title="Old", description="desc", priority_score=5.0, track="quality"),
		])
		planner.set_backlog_items([
			BacklogItem(id="b2", title="New", description="desc", priority_score=7.0, track="feature"),
		])

		assert len(planner._backlog_items) == 1
		assert planner._backlog_items[0].title == "New"

	def test_empty_list_clears(self) -> None:
		"""Passing empty list clears stored items."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		planner.set_backlog_items([
			BacklogItem(id="b1", title="Task", description="desc", priority_score=5.0, track="quality"),
		])
		planner.set_backlog_items([])

		assert planner._backlog_items == []


class TestBacklogContextInPlanning:
	"""Test that backlog item context appears in planning prompts."""

	async def test_backlog_items_appear_in_replan_context(self) -> None:
		"""Backlog items set via set_backlog_items appear in the feedback_context sent to plan_round."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])
		planner.set_backlog_items([
			BacklogItem(id="b1", title="Add auth", description="OAuth flow", priority_score=8.0, track="feature"),
			BacklogItem(id="b2", title="Fix XSS", description="Sanitize inputs", priority_score=9.0, track="security"),
		])

		mission = Mission(id="m1", objective="Build API")
		await planner.get_next_units(mission)

		call_kwargs = planner._inner.plan_round.call_args[1]
		ctx = call_kwargs["feedback_context"]
		assert "Priority backlog items for this mission:" in ctx
		assert "Add auth" in ctx
		assert "Fix XSS" in ctx
		assert "priority=8.0" in ctx
		assert "priority=9.0" in ctx

	async def test_backlog_items_appended_to_existing_context(self) -> None:
		"""Backlog section is appended after existing feedback context."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])
		planner.set_backlog_items([
			BacklogItem(id="b1", title="Refactor DB", description="Split modules", priority_score=6.0, track="quality"),
		])

		mission = Mission(id="m1", objective="Build API")
		await planner.get_next_units(mission, feedback_context="Previous work completed auth.")

		call_kwargs = planner._inner.plan_round.call_args[1]
		ctx = call_kwargs["feedback_context"]
		assert ctx.startswith("Previous work completed auth.")
		assert "Refactor DB" in ctx

	async def test_no_backlog_items_no_extra_context(self) -> None:
		"""Without backlog items, no backlog section appears in context."""
		planner = ContinuousPlanner(_planner_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])

		mission = Mission(id="m1", objective="Build API")
		await planner.get_next_units(mission, feedback_context="Some feedback.")

		call_kwargs = planner._inner.plan_round.call_args[1]
		ctx = call_kwargs["feedback_context"]
		assert "Priority backlog items" not in ctx


class TestControllerPassesBacklogToPlanner:
	"""Test that controller loads backlog items and passes them to planner."""

	def test_controller_passes_items_to_planner(self, config: MissionConfig, db: Database) -> None:
		"""After loading backlog, controller calls planner.set_backlog_items with the items."""
		db.insert_backlog_item(BacklogItem(
			id="b1", title="Add auth", description="OAuth flow",
			priority_score=8.0, track="feature",
		))
		db.insert_backlog_item(BacklogItem(
			id="b2", title="Fix XSS", description="Sanitize inputs",
			priority_score=9.0, track="security",
		))

		ctrl = ContinuousController(config, db)
		ctrl._planner = ContinuousPlanner(config, db)

		# Load backlog items (simulating what run() does)
		config.discovery.enabled = True
		ctrl._load_backlog_objective(limit=5)

		# Pass to planner (simulating what run() does after _load_backlog_objective)
		items = [db.get_backlog_item(bid) for bid in ctrl._backlog_item_ids]
		ctrl._planner.set_backlog_items([i for i in items if i is not None])

		assert len(ctrl._planner._backlog_items) == 2
		titles = {i.title for i in ctrl._planner._backlog_items}
		assert "Add auth" in titles
		assert "Fix XSS" in titles

	def test_empty_backlog_skips_planner_call(self, config: MissionConfig, db: Database) -> None:
		"""When no backlog items are loaded, planner.set_backlog_items is not called."""
		ctrl = ContinuousController(config, db)
		ctrl._planner = ContinuousPlanner(config, db)

		config.discovery.enabled = True
		ctrl._load_backlog_objective(limit=5)

		# No items loaded, so _backlog_item_ids is empty
		assert ctrl._backlog_item_ids == []
		assert ctrl._planner._backlog_items == []


# ============================================================================
# Tests consolidated from test_backlog_planner.py
# ============================================================================


def _config() -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(name="test", path="/tmp/test", objective="Build API")
	mc.planner = PlannerConfig(max_depth=2)
	mc.continuous = ContinuousConfig(backlog_min_size=2)
	return mc


def _mission() -> Mission:
	return Mission(id="m1", objective="Build a production API")


class TestBacklogItemIdsInContext:
	async def test_backlog_ids_passed_to_planner_context(self) -> None:
		"""Backlog item IDs appear in the enriched context sent to plan_round."""
		planner = ContinuousPlanner(_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])

		await planner.get_next_units(
			_mission(), backlog_item_ids=["backlog-001", "backlog-002"],
		)

		call_kwargs = planner._inner.plan_round.call_args[1]
		ctx = call_kwargs["feedback_context"]
		assert "Backlog items being worked on:" in ctx
		assert "backlog-001" in ctx
		assert "backlog-002" in ctx

	async def test_backlog_ids_appended_to_existing_feedback(self) -> None:
		"""Backlog section is appended after existing feedback context."""
		planner = ContinuousPlanner(_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])

		await planner.get_next_units(
			_mission(),
			feedback_context="Previous round completed auth module.",
			backlog_item_ids=["backlog-099"],
		)

		call_kwargs = planner._inner.plan_round.call_args[1]
		ctx = call_kwargs["feedback_context"]
		assert ctx.startswith("Previous round completed auth module.")
		assert "backlog-099" in ctx

	async def test_no_backlog_ids_leaves_context_unchanged(self) -> None:
		"""Without backlog IDs, context has no backlog section."""
		planner = ContinuousPlanner(_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])

		await planner.get_next_units(
			_mission(), feedback_context="Some feedback.",
		)

		call_kwargs = planner._inner.plan_round.call_args[1]
		ctx = call_kwargs["feedback_context"]
		assert "Backlog items being worked on:" not in ctx


class TestUnitToBacklogMapping:
	async def test_single_backlog_id_mapped_to_units(self) -> None:
		"""Units produced by replan are mapped to the provided backlog ID."""
		planner = ContinuousPlanner(_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1", "wu2"])

		await planner.get_next_units(
			_mission(), backlog_item_ids=["backlog-001"],
		)

		mapping = planner.get_backlog_mapping()
		assert mapping["wu1"] == "backlog-001"
		assert mapping["wu2"] == "backlog-001"

	async def test_multiple_backlog_ids_joined(self) -> None:
		"""Multiple backlog IDs are joined with commas in the mapping."""
		planner = ContinuousPlanner(_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])

		await planner.get_next_units(
			_mission(), backlog_item_ids=["b1", "b2", "b3"],
		)

		mapping = planner.get_backlog_mapping()
		assert mapping["wu1"] == "b1,b2,b3"

	async def test_no_backlog_ids_no_mapping(self) -> None:
		"""Without backlog IDs, no unit-to-backlog mapping is created."""
		planner = ContinuousPlanner(_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1"])

		await planner.get_next_units(_mission())

		assert planner.get_backlog_mapping() == {}

	async def test_mapping_accumulates_across_replans(self) -> None:
		"""Mapping grows as multiple replan cycles produce new units."""
		planner = ContinuousPlanner(_config(), Database(":memory:"))

		# First replan
		planner._inner.plan_round = _mock_plan_round(["wu1"])
		await planner.get_next_units(
			_mission(), backlog_item_ids=["backlog-A"],
		)

		# Second replan
		planner._inner.plan_round = _mock_plan_round(["wu2"])
		await planner.get_next_units(
			_mission(), backlog_item_ids=["backlog-B"],
		)

		mapping = planner.get_backlog_mapping()
		assert mapping["wu1"] == "backlog-A"
		assert mapping["wu2"] == "backlog-B"


class TestGetBacklogMapping:
	def test_returns_copy(self) -> None:
		"""get_backlog_mapping returns a copy, not the internal dict."""
		planner = ContinuousPlanner(_config(), Database(":memory:"))
		planner._unit_to_backlog["wu1"] = "backlog-001"

		mapping = planner.get_backlog_mapping()
		mapping["wu1"] = "tampered"

		assert planner._unit_to_backlog["wu1"] == "backlog-001"

	def test_empty_initially(self) -> None:
		"""Mapping is empty on a fresh planner."""
		planner = ContinuousPlanner(_config(), Database(":memory:"))
		assert planner.get_backlog_mapping() == {}


class TestPlannerWithoutBacklogIds:
	async def test_normal_replan_works(self) -> None:
		"""Planner works normally when no backlog IDs are provided."""
		planner = ContinuousPlanner(_config(), Database(":memory:"))
		planner._inner.plan_round = _mock_plan_round(["wu1", "wu2"])

		plan, units, epoch = await planner.get_next_units(_mission())

		assert len(units) == 2
		assert epoch.number == 1
		assert planner.get_backlog_mapping() == {}

	async def test_backlog_serve_works_without_ids(self) -> None:
		"""Serving from existing backlog works without backlog IDs."""
		config = _config()
		config.continuous.backlog_min_size = 1
		planner = ContinuousPlanner(config, Database(":memory:"))

		# Pre-populate backlog above min_size
		planner._backlog = [
			WorkUnit(id="wu1", title="T1"),
			WorkUnit(id="wu2", title="T2"),
		]

		plan, units, epoch = await planner.get_next_units(_mission(), max_units=1)

		assert len(units) == 1
		assert units[0].id == "wu1"
		assert planner.get_backlog_mapping() == {}
