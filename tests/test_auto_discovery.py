"""Tests for auto_discovery module."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from mission_control.auto_discovery import DiscoveryEngine, _compute_priority
from mission_control.config import (
	DiscoveryConfig,
	MissionConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.db import Database
from mission_control.models import DiscoveryItem, DiscoveryResult


def _config() -> MissionConfig:
	return MissionConfig(
		target=TargetConfig(
			name="test-proj",
			path="/tmp/test",
			branch="main",
			objective="",
			verification=VerificationConfig(command="pytest -q"),
		),
		discovery=DiscoveryConfig(
			enabled=True,
			tracks=["feature", "quality", "security"],
			max_items_per_track=3,
			min_priority_score=3.0,
			model="sonnet",
			budget_per_call_usd=1.0,
		),
	)


def _discovery_json(items: list[dict]) -> str:
	return f"DISCOVERY_RESULT:\n```json\n{json.dumps({'items': items})}\n```"


def _sample_item(
	track: str = "quality",
	title: str = "Fix test coverage",
	impact: int = 7,
	effort: int = 3,
) -> dict:
	return {
		"track": track,
		"title": title,
		"description": "Add missing tests for utils module.",
		"rationale": "Low coverage in critical paths.",
		"files_hint": "src/utils.py, tests/test_utils.py",
		"impact": impact,
		"effort": effort,
	}


class TestComputePriority:
	def test_basic(self) -> None:
		assert _compute_priority(7, 3) == 5.6

	def test_high_impact_low_effort(self) -> None:
		assert _compute_priority(10, 1) == 10.0

	def test_low_impact_high_effort(self) -> None:
		assert _compute_priority(1, 10) == 0.1

	def test_clamps_values(self) -> None:
		assert _compute_priority(0, 0) == 1.0  # clamped to 1,1
		assert _compute_priority(15, 15) == 1.0  # clamped to 10,10


class TestParseDiscoveryOutput:
	def setup_method(self) -> None:
		self.config = _config()
		self.db = Database(":memory:")
		self.engine = DiscoveryEngine(self.config, self.db)

	def test_parse_valid_output(self) -> None:
		output = _discovery_json([_sample_item()])
		result, items = self.engine._parse_discovery_output(output)
		assert result.target_path == "/tmp/test"
		assert result.model == "sonnet"
		assert result.item_count == 1
		assert items[0].title == "Fix test coverage"
		assert items[0].track == "quality"
		assert items[0].priority_score == 5.6

	def test_parse_multiple_tracks(self) -> None:
		output = _discovery_json([
			_sample_item(track="feature", title="Add API endpoint"),
			_sample_item(track="quality", title="Refactor utils"),
			_sample_item(track="security", title="Fix SQL injection"),
		])
		result, items = self.engine._parse_discovery_output(output)
		assert result.item_count == 3
		tracks = {i.track for i in items}
		assert tracks == {"feature", "quality", "security"}

	def test_filters_low_priority(self) -> None:
		output = _discovery_json([
			_sample_item(impact=1, effort=10),  # priority 0.1
			_sample_item(impact=8, effort=2, title="High priority"),
		])
		result, items = self.engine._parse_discovery_output(output)
		assert result.item_count == 1
		assert items[0].title == "High priority"

	def test_filters_invalid_track(self) -> None:
		output = _discovery_json([
			_sample_item(track="invalid"),
			_sample_item(track="quality"),
		])
		result, items = self.engine._parse_discovery_output(output)
		assert result.item_count == 1
		assert items[0].track == "quality"

	def test_sorted_by_priority(self) -> None:
		output = _discovery_json([
			_sample_item(impact=5, effort=4, title="Low"),  # 3.5
			_sample_item(impact=9, effort=2, title="High"),  # 8.1
			_sample_item(impact=7, effort=3, title="Mid"),  # 5.6
		])
		result, items = self.engine._parse_discovery_output(output)
		assert items[0].title == "High"
		assert items[-1].title == "Low"

	def test_empty_output(self) -> None:
		result, items = self.engine._parse_discovery_output("")
		assert result.item_count == 0
		assert items == []

	def test_no_marker(self) -> None:
		output = "Some random text without JSON"
		result, items = self.engine._parse_discovery_output(output)
		assert items == []

	def test_fallback_json_without_marker(self) -> None:
		raw = json.dumps({"items": [_sample_item()]})
		result, items = self.engine._parse_discovery_output(raw)
		assert result.item_count == 1


class TestBuildPrompt:
	def setup_method(self) -> None:
		self.config = _config()
		self.db = Database(":memory:")
		self.engine = DiscoveryEngine(self.config, self.db)

	def test_includes_tracks(self) -> None:
		prompt = self.engine._build_discovery_prompt()
		assert "Track A - Features" in prompt
		assert "Track B - Code Quality" in prompt
		assert "Track C - Security" in prompt

	def test_includes_max_per_track(self) -> None:
		prompt = self.engine._build_discovery_prompt()
		assert "up to 3 improvements" in prompt

	def test_includes_past_discoveries(self) -> None:
		# Insert a past discovery
		dr = DiscoveryResult(target_path="/tmp/test", model="sonnet")
		item = DiscoveryItem(
			track="quality",
			title="Old improvement",
			priority_score=5.0,
		)
		self.db.insert_discovery_result(dr, [item])

		prompt = self.engine._build_discovery_prompt()
		assert "Old improvement" in prompt
		assert "Previously Discovered" in prompt

	def test_subset_tracks(self) -> None:
		self.config.discovery.tracks = ["security"]
		prompt = self.engine._build_discovery_prompt()
		assert "Track C - Security" in prompt
		assert "Track A" not in prompt
		assert "Track B" not in prompt


class TestComposeObjective:
	def setup_method(self) -> None:
		self.config = _config()
		self.db = Database(":memory:")
		self.engine = DiscoveryEngine(self.config, self.db)

	def test_basic(self) -> None:
		items = [
			DiscoveryItem(
				track="feature",
				title="Add endpoint",
				description="Add /users endpoint",
				files_hint="src/api.py",
				priority_score=8.0,
			),
		]
		obj = self.engine.compose_objective(items)
		assert "Add endpoint" in obj
		assert "Add /users endpoint" in obj
		assert "Features" in obj

	def test_multiple_tracks(self) -> None:
		items = [
			DiscoveryItem(track="feature", title="T1", description="D1", priority_score=8.0),
			DiscoveryItem(track="quality", title="T2", description="D2", priority_score=7.0),
		]
		obj = self.engine.compose_objective(items)
		assert "Features" in obj
		assert "Code Quality" in obj

	def test_empty(self) -> None:
		assert self.engine.compose_objective([]) == ""


class TestDBIntegration:
	def test_insert_and_retrieve(self) -> None:
		db = Database(":memory:")
		dr = DiscoveryResult(target_path="/tmp/test", model="sonnet", item_count=2)
		items = [
			DiscoveryItem(track="feature", title="Item 1", priority_score=8.0),
			DiscoveryItem(track="quality", title="Item 2", priority_score=5.0),
		]
		db.insert_discovery_result(dr, items)

		latest, retrieved_items = db.get_latest_discovery()
		assert latest is not None
		assert latest.id == dr.id
		assert latest.item_count == 2
		assert len(retrieved_items) == 2
		# Should be sorted by priority_score DESC
		assert retrieved_items[0].title == "Item 1"
		assert retrieved_items[1].title == "Item 2"

	def test_past_titles(self) -> None:
		db = Database(":memory:")
		dr = DiscoveryResult(target_path="/tmp/test", model="sonnet", item_count=1)
		items = [DiscoveryItem(track="feature", title="My Title", priority_score=5.0)]
		db.insert_discovery_result(dr, items)

		titles = db.get_past_discovery_titles()
		assert "My Title" in titles

	def test_update_item_status(self) -> None:
		db = Database(":memory:")
		dr = DiscoveryResult(target_path="/tmp/test", model="sonnet", item_count=1)
		item = DiscoveryItem(track="feature", title="T", priority_score=5.0)
		db.insert_discovery_result(dr, [item])

		db.update_discovery_item_status(item.id, "approved")
		_, items = db.get_latest_discovery()
		assert items[0].status == "approved"


class TestDiscoverSubprocess:
	@pytest.mark.asyncio
	async def test_discover_calls_subprocess(self) -> None:
		config = _config()
		db = Database(":memory:")
		engine = DiscoveryEngine(config, db)

		mock_output = _discovery_json([_sample_item()])

		mock_proc = AsyncMock()
		mock_proc.communicate = AsyncMock(return_value=(mock_output.encode(), b""))
		mock_proc.returncode = 0

		with patch("mission_control.auto_discovery.asyncio.create_subprocess_exec", return_value=mock_proc):
			result, items = await engine.discover()

		assert result.item_count == 1
		assert items[0].title == "Fix test coverage"

		# Verify persisted to DB
		latest, db_items = db.get_latest_discovery()
		assert latest is not None
		assert latest.id == result.id
		assert len(db_items) == 1

	@pytest.mark.asyncio
	async def test_discover_timeout(self) -> None:
		config = _config()
		db = Database(":memory:")
		engine = DiscoveryEngine(config, db)

		with patch(
			"mission_control.auto_discovery.asyncio.create_subprocess_exec",
			side_effect=asyncio.TimeoutError,
		):
			result, items = await engine.discover()

		assert result.item_count == 0
		assert items == []
