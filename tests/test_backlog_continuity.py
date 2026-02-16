"""Tests for backlog continuity in ContinuousController.

Covers:
- Title-based matching between work units and backlog items
- Status transitions on successful merge
- Status transitions on failure after max retries
- attempt_count incrementing on failure
- Partial completion context carry-forward
- No-match graceful handling
- Mission ID tracking on newly discovered items
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig
from mission_control.continuous_controller import ContinuousController
from mission_control.db import Database
from mission_control.models import BacklogItem, DiscoveryItem, Handoff, WorkUnit


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
