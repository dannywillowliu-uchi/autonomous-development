"""Tests for discovery items flowing into the persistent backlog table."""

from __future__ import annotations

from mission_control.auto_discovery import DiscoveryEngine, _compute_priority
from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import BacklogItem, DiscoveryItem


class TestDiscoveryToBacklog:
	"""Test that DiscoveryEngine._insert_items_to_backlog inserts into backlog_items."""

	def test_discovery_items_inserted_into_backlog(self, config: MissionConfig, db: Database) -> None:
		"""Discovery items are converted to BacklogItems and inserted."""
		engine = DiscoveryEngine(config, db)
		items = [
			DiscoveryItem(
				id="d1", title="Add caching", description="Redis layer",
				priority_score=7.0, impact=8, effort=5, track="feature",
			),
			DiscoveryItem(
				id="d2", title="Fix XSS", description="Sanitize inputs",
				priority_score=9.0, impact=10, effort=3, track="security",
			),
		]

		engine._insert_items_to_backlog(items)

		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 2
		titles = {item.title for item in backlog}
		assert "Add caching" in titles
		assert "Fix XSS" in titles

	def test_duplicate_titles_not_reinserted(self, config: MissionConfig, db: Database) -> None:
		"""Items with titles already in backlog_items are skipped."""
		# Pre-insert a backlog item with the same title
		db.insert_backlog_item(BacklogItem(
			id="existing", title="Add caching", description="Already exists",
			priority_score=5.0, impact=5, effort=5, track="quality",
		))

		engine = DiscoveryEngine(config, db)
		items = [
			DiscoveryItem(
				id="d1", title="Add caching", description="Redis layer",
				priority_score=7.0, impact=8, effort=5, track="feature",
			),
			DiscoveryItem(
				id="d2", title="New item", description="Something new",
				priority_score=6.0, impact=7, effort=4, track="quality",
			),
		]

		engine._insert_items_to_backlog(items)

		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 2
		# The existing one should still have the original description
		existing = [i for i in backlog if i.title == "Add caching"]
		assert len(existing) == 1
		assert existing[0].description == "Already exists"
		# The new one should be inserted
		new = [i for i in backlog if i.title == "New item"]
		assert len(new) == 1

	def test_priority_score_carried_over(self, config: MissionConfig, db: Database) -> None:
		"""Priority score, impact, effort, and track are mapped correctly."""
		engine = DiscoveryEngine(config, db)
		items = [
			DiscoveryItem(
				id="d1", title="Optimize queries", description="Add indexes",
				priority_score=_compute_priority(9, 2), impact=9, effort=2, track="quality",
			),
		]

		engine._insert_items_to_backlog(items)

		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 1
		item = backlog[0]
		assert item.title == "Optimize queries"
		assert item.description == "Add indexes"
		assert item.priority_score == _compute_priority(9, 2)
		assert item.impact == 9
		assert item.effort == 2
		assert item.track == "quality"
		assert item.status == "pending"

	def test_duplicate_within_same_batch(self, config: MissionConfig, db: Database) -> None:
		"""If the same title appears twice in one batch, only the first is inserted."""
		engine = DiscoveryEngine(config, db)
		items = [
			DiscoveryItem(
				id="d1", title="Same title", description="First",
				priority_score=7.0, impact=8, effort=5, track="feature",
			),
			DiscoveryItem(
				id="d2", title="Same title", description="Second",
				priority_score=6.0, impact=7, effort=4, track="feature",
			),
		]

		engine._insert_items_to_backlog(items)

		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 1
		assert backlog[0].description == "First"

	def test_empty_items_noop(self, config: MissionConfig, db: Database) -> None:
		"""Passing an empty list does nothing."""
		engine = DiscoveryEngine(config, db)
		engine._insert_items_to_backlog([])
		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 0
