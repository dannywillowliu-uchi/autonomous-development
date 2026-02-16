"""Tests for priority recalculation engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mission_control.db import Database
from mission_control.models import BacklogItem
from mission_control.priority import (
	_compute_base_score,
	_compute_failure_penalty,
	_is_stale,
	parse_backlog_md,
	recalculate_priorities,
)


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


def _make_item(**overrides: object) -> BacklogItem:
	defaults = {
		"title": "Test item",
		"description": "A test backlog item",
		"impact": 8,
		"effort": 3,
		"status": "pending",
	}
	defaults.update(overrides)
	return BacklogItem(**defaults)


class TestBaseScore:
	def test_formula(self) -> None:
		assert _compute_base_score(10, 1) == 10.0
		assert _compute_base_score(10, 10) == 1.0
		assert _compute_base_score(5, 5) == 3.0

	def test_high_impact_low_effort(self) -> None:
		score = _compute_base_score(10, 1)
		assert score == 10.0

	def test_low_impact_high_effort(self) -> None:
		score = _compute_base_score(1, 10)
		assert score == 0.1

	def test_mid_range(self) -> None:
		score = _compute_base_score(8, 3)
		assert score == pytest.approx(6.4)


class TestFailurePenalty:
	def test_no_failures(self) -> None:
		assert _compute_failure_penalty(0, None) == 0.0

	def test_one_failure(self) -> None:
		assert _compute_failure_penalty(1, "test failed") == pytest.approx(0.2)

	def test_two_failures(self) -> None:
		assert _compute_failure_penalty(2, "test failed") == pytest.approx(0.4)

	def test_cap_at_60_percent(self) -> None:
		assert _compute_failure_penalty(5, "test failed") == pytest.approx(0.6)
		assert _compute_failure_penalty(10, "some error") == pytest.approx(0.6)

	def test_infrastructure_exception(self) -> None:
		assert _compute_failure_penalty(3, "infrastructure error on deploy") == 0.0

	def test_timeout_exception(self) -> None:
		assert _compute_failure_penalty(2, "timeout waiting for response") == 0.0

	def test_network_exception(self) -> None:
		assert _compute_failure_penalty(4, "network connection refused") == 0.0

	def test_infrastructure_case_insensitive(self) -> None:
		assert _compute_failure_penalty(2, "INFRASTRUCTURE failure") == 0.0
		assert _compute_failure_penalty(2, "Network Timeout Error") == 0.0

	def test_none_reason_still_penalizes(self) -> None:
		assert _compute_failure_penalty(2, None) == pytest.approx(0.4)


class TestStaleness:
	def test_not_stale(self) -> None:
		now = datetime.now(timezone.utc)
		recent = (now - timedelta(hours=1)).isoformat()
		assert _is_stale(recent, now) is False

	def test_stale(self) -> None:
		now = datetime.now(timezone.utc)
		old = (now - timedelta(hours=100)).isoformat()
		assert _is_stale(old, now) is True

	def test_boundary(self) -> None:
		now = datetime.now(timezone.utc)
		at_boundary = (now - timedelta(hours=72)).isoformat()
		assert _is_stale(at_boundary, now) is False

	def test_invalid_date(self) -> None:
		now = datetime.now(timezone.utc)
		assert _is_stale("not-a-date", now) is False


class TestRecalculatePriorities:
	def test_basic_scoring(self, db: Database) -> None:
		item = _make_item(impact=10, effort=1)
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 1
		assert updated[0].priority_score == pytest.approx(10.0)

	def test_failure_penalty_applied(self, db: Database) -> None:
		item = _make_item(impact=10, effort=1, attempt_count=2, last_failure_reason="test failed")
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 1
		expected = 10.0 * (1.0 - 0.4)
		assert updated[0].priority_score == pytest.approx(expected)

	def test_infrastructure_failure_no_penalty(self, db: Database) -> None:
		item = _make_item(
			impact=10, effort=1,
			attempt_count=3, last_failure_reason="infrastructure error",
		)
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 1
		assert updated[0].priority_score == pytest.approx(10.0)

	def test_staleness_penalty(self, db: Database) -> None:
		old_time = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
		item = _make_item(impact=10, effort=1, updated_at=old_time)
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 1
		expected = 10.0 * 0.9
		assert updated[0].priority_score == pytest.approx(expected)

	def test_pinned_score_override(self, db: Database) -> None:
		item = _make_item(impact=10, effort=1, pinned_score=42.0)
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 1
		assert updated[0].priority_score == 42.0

	def test_pinned_score_ignores_penalties(self, db: Database) -> None:
		old_time = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
		item = _make_item(
			impact=10, effort=1,
			attempt_count=3, last_failure_reason="test failed",
			updated_at=old_time,
			pinned_score=99.0,
		)
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 1
		assert updated[0].priority_score == 99.0

	def test_completed_items_skipped(self, db: Database) -> None:
		item = _make_item(impact=10, effort=1, status="completed", priority_score=5.0)
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 0

	def test_deferred_items_skipped(self, db: Database) -> None:
		item = _make_item(impact=10, effort=1, status="deferred", priority_score=5.0)
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 0

	def test_in_progress_items_recalculated(self, db: Database) -> None:
		item = _make_item(impact=10, effort=1, status="in_progress")
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 1
		assert updated[0].priority_score == pytest.approx(10.0)

	def test_combined_failure_and_staleness(self, db: Database) -> None:
		old_time = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
		item = _make_item(
			impact=10, effort=1,
			attempt_count=1, last_failure_reason="test failed",
			updated_at=old_time,
		)
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 1
		expected = 10.0 * (1.0 - 0.2) * 0.9
		assert updated[0].priority_score == pytest.approx(expected)

	def test_no_update_when_score_unchanged(self, db: Database) -> None:
		item = _make_item(impact=8, effort=3)
		expected_score = round(_compute_base_score(8, 3), 4)
		item.priority_score = expected_score
		db.insert_backlog_item(item)
		updated = recalculate_priorities(db)
		assert len(updated) == 0


class TestParseBacklogMd:
	def test_parse_basic(self, tmp_path: object) -> None:
		from pathlib import Path
		md = tmp_path / "BACKLOG.md"  # type: ignore[operator]
		md.write_text(
			"# Backlog\n\n"
			"## P0: Critical task\n"
			"This is the description.\n"
			"More details here.\n"
			"\n"
			"## P3: Medium task\n"
			"Medium priority description.\n"
		)
		items = parse_backlog_md(Path(str(md)))
		assert len(items) == 2
		assert items[0].title == "Critical task"
		assert items[0].impact == 10
		assert items[0].priority_score == pytest.approx(6.0)
		assert items[1].title == "Medium task"
		assert items[1].impact == 7
		assert items[1].priority_score == pytest.approx(4.2)

	def test_parse_real_format(self, tmp_path: object) -> None:
		from pathlib import Path
		md = tmp_path / "BACKLOG.md"  # type: ignore[operator]
		md.write_text(
			"# Backlog\n\n"
			"## P0: Replace LLM Evaluator\n\n"
			"**Problem**: The evaluator is expensive.\n\n"
			"**Files**: evaluator.py\n\n"
			"---\n\n"
			"## P1: N-of-M Candidate Selection\n\n"
			"**Problem**: Fixup makes one attempt.\n"
		)
		items = parse_backlog_md(Path(str(md)))
		assert len(items) == 2
		assert items[0].title == "Replace LLM Evaluator"
		assert items[0].impact == 10
		assert "expensive" in items[0].description
		assert items[1].title == "N-of-M Candidate Selection"
		assert items[1].impact == 9

	def test_parse_p9(self, tmp_path: object) -> None:
		from pathlib import Path
		md = tmp_path / "BACKLOG.md"  # type: ignore[operator]
		md.write_text("## P9: Low priority\nDescription.\n")
		items = parse_backlog_md(Path(str(md)))
		assert len(items) == 1
		assert items[0].impact == 1
		assert items[0].priority_score == pytest.approx(0.6)

	def test_parse_empty_file(self, tmp_path: object) -> None:
		from pathlib import Path
		md = tmp_path / "BACKLOG.md"  # type: ignore[operator]
		md.write_text("# Backlog\n\nNo items here.\n")
		items = parse_backlog_md(Path(str(md)))
		assert len(items) == 0

	def test_parse_description_multiline(self, tmp_path: object) -> None:
		from pathlib import Path
		md = tmp_path / "BACKLOG.md"  # type: ignore[operator]
		md.write_text(
			"## P2: Multi-line task\n"
			"Line one.\n"
			"Line two.\n"
			"Line three.\n"
		)
		items = parse_backlog_md(Path(str(md)))
		assert len(items) == 1
		assert "Line one." in items[0].description
		assert "Line three." in items[0].description
