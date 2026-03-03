"""Tests for event stream analytics and aggregation functions."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mission_control.event_stream import (
	compute_cost_summary,
	compute_failure_rate,
	compute_merge_velocity,
	compute_unit_duration,
	read_events,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(minutes_offset: float = 0) -> str:
	"""ISO timestamp offset from a fixed base time."""
	base = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
	return (base + timedelta(minutes=minutes_offset)).isoformat()


def _event(
	event_type: str = "dispatched",
	*,
	mission_id: str = "m1",
	unit_id: str = "u1",
	cost_usd: float = 0.0,
	timestamp: str | None = None,
) -> dict:
	return {
		"timestamp": timestamp or _ts(),
		"event_type": event_type,
		"mission_id": mission_id,
		"unit_id": unit_id,
		"cost_usd": cost_usd,
	}


def _write_jsonl(path: Path, events: list[dict]) -> None:
	with path.open("w", encoding="utf-8") as f:
		for e in events:
			f.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# read_events
# ---------------------------------------------------------------------------

class TestReadEvents:
	def test_reads_all_events(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		events = [_event("dispatched"), _event("merged"), _event("merge_failed")]
		_write_jsonl(p, events)
		result = read_events(p)
		assert len(result) == 3

	def test_filter_by_event_type(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		events = [_event("dispatched"), _event("merged"), _event("dispatched")]
		_write_jsonl(p, events)
		result = read_events(p, event_type="dispatched")
		assert len(result) == 2
		assert all(e["event_type"] == "dispatched" for e in result)

	def test_filter_by_unit_id(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		events = [
			_event(unit_id="u1"),
			_event(unit_id="u2"),
			_event(unit_id="u1"),
		]
		_write_jsonl(p, events)
		result = read_events(p, unit_id="u2")
		assert len(result) == 1
		assert result[0]["unit_id"] == "u2"

	def test_filter_by_mission_id(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		events = [
			_event(mission_id="m1"),
			_event(mission_id="m2"),
			_event(mission_id="m1"),
		]
		_write_jsonl(p, events)
		result = read_events(p, mission_id="m2")
		assert len(result) == 1

	def test_combined_filters(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		events = [
			_event("dispatched", mission_id="m1", unit_id="u1"),
			_event("dispatched", mission_id="m1", unit_id="u2"),
			_event("merged", mission_id="m1", unit_id="u1"),
			_event("dispatched", mission_id="m2", unit_id="u1"),
		]
		_write_jsonl(p, events)
		result = read_events(p, event_type="dispatched", mission_id="m1", unit_id="u1")
		assert len(result) == 1

	def test_nonexistent_file(self, tmp_path: Path) -> None:
		p = tmp_path / "nope.jsonl"
		assert read_events(p) == []

	def test_empty_file(self, tmp_path: Path) -> None:
		p = tmp_path / "empty.jsonl"
		p.write_text("")
		assert read_events(p) == []

	def test_skips_blank_lines(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		p.write_text(json.dumps(_event()) + "\n\n" + json.dumps(_event("merged")) + "\n")
		assert len(read_events(p)) == 2


# ---------------------------------------------------------------------------
# compute_failure_rate
# ---------------------------------------------------------------------------

class TestComputeFailureRate:
	def test_zero_percent(self) -> None:
		events = [_event("dispatched"), _event("merged")]
		assert compute_failure_rate(events) == 0.0

	def test_fifty_percent(self) -> None:
		events = [
			_event("dispatched", unit_id="u1"),
			_event("dispatched", unit_id="u2"),
			_event("merge_failed", unit_id="u1"),
			_event("merged", unit_id="u2"),
		]
		assert compute_failure_rate(events) == 0.5

	def test_hundred_percent(self) -> None:
		events = [
			_event("dispatched"),
			_event("merge_failed"),
		]
		assert compute_failure_rate(events) == 1.0

	def test_no_dispatched_events(self) -> None:
		events = [_event("merged"), _event("merge_failed")]
		assert compute_failure_rate(events) == 0.0

	def test_empty_events(self) -> None:
		assert compute_failure_rate([]) == 0.0

	def test_multiple_failures(self) -> None:
		events = [
			_event("dispatched", unit_id="u1"),
			_event("dispatched", unit_id="u2"),
			_event("dispatched", unit_id="u3"),
			_event("merge_failed", unit_id="u1"),
			_event("merge_failed", unit_id="u2"),
		]
		assert abs(compute_failure_rate(events) - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# compute_cost_summary
# ---------------------------------------------------------------------------

class TestComputeCostSummary:
	def test_basic_costs(self) -> None:
		events = [
			_event(cost_usd=0.10),
			_event(cost_usd=0.20),
			_event(cost_usd=0.30),
		]
		summary = compute_cost_summary(events)
		assert abs(summary["total_cost"] - 0.60) < 1e-9
		assert abs(summary["mean_cost"] - 0.20) < 1e-9
		assert abs(summary["max_cost"] - 0.30) < 1e-9

	def test_empty_events(self) -> None:
		summary = compute_cost_summary([])
		assert summary == {"total_cost": 0.0, "mean_cost": 0.0, "max_cost": 0.0}

	def test_all_zero_costs(self) -> None:
		events = [_event(cost_usd=0.0), _event(cost_usd=0.0)]
		summary = compute_cost_summary(events)
		assert summary == {"total_cost": 0.0, "mean_cost": 0.0, "max_cost": 0.0}

	def test_single_event(self) -> None:
		events = [_event(cost_usd=0.42)]
		summary = compute_cost_summary(events)
		assert abs(summary["total_cost"] - 0.42) < 1e-9
		assert abs(summary["mean_cost"] - 0.42) < 1e-9
		assert abs(summary["max_cost"] - 0.42) < 1e-9

	def test_mixed_zero_and_nonzero(self) -> None:
		events = [
			_event(cost_usd=0.0),
			_event(cost_usd=0.50),
			_event(cost_usd=0.0),
			_event(cost_usd=1.00),
		]
		summary = compute_cost_summary(events)
		assert abs(summary["total_cost"] - 1.50) < 1e-9
		# mean only over events with cost > 0
		assert abs(summary["mean_cost"] - 0.75) < 1e-9
		assert abs(summary["max_cost"] - 1.00) < 1e-9

	def test_missing_cost_field(self) -> None:
		events = [{"event_type": "dispatched"}, {"event_type": "merged"}]
		summary = compute_cost_summary(events)
		assert summary == {"total_cost": 0.0, "mean_cost": 0.0, "max_cost": 0.0}


# ---------------------------------------------------------------------------
# compute_merge_velocity
# ---------------------------------------------------------------------------

class TestComputeMergeVelocity:
	def test_basic_velocity(self) -> None:
		# 2 merges in a 1-hour span -> 2.0 merges/hr
		events = [
			_event("dispatched", timestamp=_ts(0)),
			_event("merged", timestamp=_ts(30)),
			_event("merged", timestamp=_ts(60)),
		]
		assert abs(compute_merge_velocity(events) - 2.0) < 1e-9

	def test_no_merged_events(self) -> None:
		events = [_event("dispatched", timestamp=_ts(0)), _event("dispatched", timestamp=_ts(60))]
		assert compute_merge_velocity(events) == 0.0

	def test_empty_events(self) -> None:
		assert compute_merge_velocity([]) == 0.0

	def test_single_event(self) -> None:
		events = [_event("merged", timestamp=_ts(0))]
		# Only 1 timestamp, can't compute span
		assert compute_merge_velocity(events) == 0.0

	def test_all_same_timestamp(self) -> None:
		ts = _ts(0)
		events = [
			_event("dispatched", timestamp=ts),
			_event("merged", timestamp=ts),
		]
		# Zero time span -> 0.0
		assert compute_merge_velocity(events) == 0.0

	def test_high_velocity(self) -> None:
		# 10 merges in 30 minutes -> 20 merges/hr
		events = [_event("dispatched", timestamp=_ts(0))]
		for i in range(10):
			events.append(_event("merged", timestamp=_ts(3 * (i + 1))))
		assert abs(compute_merge_velocity(events) - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_unit_duration
# ---------------------------------------------------------------------------

class TestComputeUnitDuration:
	def test_basic_duration(self) -> None:
		events = [
			_event("dispatched", unit_id="u1", timestamp=_ts(0)),
			_event("merged", unit_id="u1", timestamp=_ts(10)),
		]
		# 10 minutes = 600 seconds
		assert compute_unit_duration(events, "u1") == 600.0

	def test_duration_with_retry(self) -> None:
		# First dispatch, merge_failed, re-dispatch, merged
		events = [
			_event("dispatched", unit_id="u1", timestamp=_ts(0)),
			_event("merge_failed", unit_id="u1", timestamp=_ts(5)),
			_event("dispatched", unit_id="u1", timestamp=_ts(6)),
			_event("merged", unit_id="u1", timestamp=_ts(15)),
		]
		# First dispatch (t=0) to last terminal event (t=15min) = 900s
		assert compute_unit_duration(events, "u1") == 900.0

	def test_duration_ended_by_failure(self) -> None:
		events = [
			_event("dispatched", unit_id="u1", timestamp=_ts(0)),
			_event("merge_failed", unit_id="u1", timestamp=_ts(20)),
		]
		assert compute_unit_duration(events, "u1") == 1200.0

	def test_unknown_unit(self) -> None:
		events = [_event("dispatched", unit_id="u1", timestamp=_ts(0))]
		assert compute_unit_duration(events, "u99") is None

	def test_no_terminal_event(self) -> None:
		events = [_event("dispatched", unit_id="u1", timestamp=_ts(0))]
		assert compute_unit_duration(events, "u1") is None

	def test_no_dispatch_event(self) -> None:
		events = [_event("merged", unit_id="u1", timestamp=_ts(10))]
		assert compute_unit_duration(events, "u1") is None

	def test_ignores_other_units(self) -> None:
		events = [
			_event("dispatched", unit_id="u1", timestamp=_ts(0)),
			_event("dispatched", unit_id="u2", timestamp=_ts(5)),
			_event("merged", unit_id="u1", timestamp=_ts(10)),
			_event("merged", unit_id="u2", timestamp=_ts(20)),
		]
		assert compute_unit_duration(events, "u1") == 600.0
		assert compute_unit_duration(events, "u2") == 900.0

	def test_empty_events(self) -> None:
		assert compute_unit_duration([], "u1") is None
