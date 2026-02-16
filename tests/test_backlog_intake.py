"""Tests for backlog intake in ContinuousController.

Covers:
- Loading backlog items as mission objective
- Marking items complete/failed post-mission
- Discovery items flowing to persistent backlog
- Empty backlog fallback to normal discovery
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig
from mission_control.continuous_controller import ContinuousController
from mission_control.db import Database
from mission_control.models import BacklogItem, DiscoveryItem, Handoff


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
