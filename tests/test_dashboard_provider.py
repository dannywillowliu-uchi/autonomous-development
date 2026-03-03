"""Tests for MissionMetricsProvider live event stream aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mission_control.dashboard.provider import MissionMetricsProvider


def _write_events(path: Path, events: list[dict]) -> None:
	"""Write a list of event dicts as JSONL."""
	with path.open("w", encoding="utf-8") as f:
		for ev in events:
			f.write(json.dumps(ev) + "\n")


def _append_events(path: Path, events: list[dict]) -> None:
	"""Append event dicts to an existing JSONL file."""
	with path.open("a", encoding="utf-8") as f:
		for ev in events:
			f.write(json.dumps(ev) + "\n")


class TestMissionMetricsProvider:
	def test_empty_file(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		p.write_text("")
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert m["epoch"] == 0
		assert m["active_workers"] == 0
		assert m["units_dispatched"] == 0
		assert m["units_merged"] == 0
		assert m["units_failed"] == 0
		assert m["success_rate"] == 0.0
		assert m["total_cost_usd"] == 0.0
		assert m["recent_events"] == []

	def test_nonexistent_file(self, tmp_path: Path) -> None:
		p = tmp_path / "missing.jsonl"
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert m["units_dispatched"] == 0

	def test_dispatched_count(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		_write_events(p, [
			{"event_type": "dispatched", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
			{"event_type": "dispatched", "unit_id": "u2", "worker_id": "w2", "cost_usd": 0.0},
			{"event_type": "dispatched", "unit_id": "u3", "worker_id": "w3", "cost_usd": 0.0},
		])
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert m["units_dispatched"] == 3
		assert m["active_workers"] == 3

	def test_merged_count(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		_write_events(p, [
			{"event_type": "dispatched", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
			{"event_type": "dispatched", "unit_id": "u2", "worker_id": "w2", "cost_usd": 0.0},
			{"event_type": "merged", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.5},
			{"event_type": "merged", "unit_id": "u2", "worker_id": "w2", "cost_usd": 0.3},
		])
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert m["units_dispatched"] == 2
		assert m["units_merged"] == 2
		assert m["units_failed"] == 0
		assert m["active_workers"] == 0
		assert m["success_rate"] == 1.0

	def test_failed_count(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		_write_events(p, [
			{"event_type": "dispatched", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
			{"event_type": "merge_failed", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
		])
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert m["units_dispatched"] == 1
		assert m["units_failed"] == 1
		assert m["units_merged"] == 0
		assert m["success_rate"] == 0.0

	def test_mixed_success_rate(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		_write_events(p, [
			{"event_type": "dispatched", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
			{"event_type": "dispatched", "unit_id": "u2", "worker_id": "w2", "cost_usd": 0.0},
			{"event_type": "dispatched", "unit_id": "u3", "worker_id": "w3", "cost_usd": 0.0},
			{"event_type": "merged", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
			{"event_type": "merged", "unit_id": "u2", "worker_id": "w2", "cost_usd": 0.0},
			{"event_type": "merge_failed", "unit_id": "u3", "worker_id": "w3", "cost_usd": 0.0},
		])
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert m["units_merged"] == 2
		assert m["units_failed"] == 1
		assert m["success_rate"] == pytest.approx(2 / 3)

	def test_cost_tracking(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		_write_events(p, [
			{"event_type": "dispatched", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
			{"event_type": "merged", "unit_id": "u1", "worker_id": "w1", "cost_usd": 1.25},
			{"event_type": "dispatched", "unit_id": "u2", "worker_id": "w2", "cost_usd": 0.0},
			{"event_type": "merge_failed", "unit_id": "u2", "worker_id": "w2", "cost_usd": 0.75},
		])
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert m["total_cost_usd"] == pytest.approx(2.0)

	def test_epoch_from_mission_started(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		_write_events(p, [
			{"event_type": "mission_started", "mission_id": "m1", "cost_usd": 0.0},
			{"event_type": "dispatched", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
			{"event_type": "mission_ended", "mission_id": "m1", "cost_usd": 0.0},
			{"event_type": "mission_started", "mission_id": "m2", "cost_usd": 0.0},
		])
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert m["epoch"] == 2

	def test_incremental_tail(self, tmp_path: Path) -> None:
		"""Provider reads only new lines on subsequent calls."""
		p = tmp_path / "events.jsonl"
		_write_events(p, [
			{"event_type": "dispatched", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
		])
		provider = MissionMetricsProvider(p)
		m1 = provider.get_current_metrics()
		assert m1["units_dispatched"] == 1

		_append_events(p, [
			{"event_type": "merged", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.5},
			{"event_type": "dispatched", "unit_id": "u2", "worker_id": "w2", "cost_usd": 0.0},
		])
		m2 = provider.get_current_metrics()
		assert m2["units_dispatched"] == 2
		assert m2["units_merged"] == 1
		assert m2["total_cost_usd"] == pytest.approx(0.5)

	def test_recent_events_returned(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		events = [
			{"event_type": "mission_started", "mission_id": "m1", "cost_usd": 0.0},
			{"event_type": "dispatched", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
			{"event_type": "merged", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.5},
		]
		_write_events(p, events)
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert len(m["recent_events"]) == 3
		assert m["recent_events"][0]["event_type"] == "mission_started"
		assert m["recent_events"][-1]["event_type"] == "merged"

	def test_recent_events_capped(self, tmp_path: Path) -> None:
		"""Recent events list is capped at max_recent (50)."""
		p = tmp_path / "events.jsonl"
		events = [
			{"event_type": "dispatched", "unit_id": f"u{i}", "worker_id": f"w{i}", "cost_usd": 0.0}
			for i in range(60)
		]
		_write_events(p, events)
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert len(m["recent_events"]) == 50

	def test_malformed_line_skipped(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		ev1 = {"event_type": "dispatched", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0}
		ev2 = {"event_type": "merged", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.5}
		with p.open("w", encoding="utf-8") as f:
			f.write(json.dumps(ev1) + "\n")
			f.write("not valid json\n")
			f.write(json.dumps(ev2) + "\n")
		provider = MissionMetricsProvider(p)
		m = provider.get_current_metrics()
		assert m["units_dispatched"] == 1
		assert m["units_merged"] == 1

	def test_active_workers_tracks_dispatch_and_completion(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		_write_events(p, [
			{"event_type": "dispatched", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
			{"event_type": "dispatched", "unit_id": "u2", "worker_id": "w2", "cost_usd": 0.0},
			{"event_type": "dispatched", "unit_id": "u3", "worker_id": "w3", "cost_usd": 0.0},
		])
		provider = MissionMetricsProvider(p)
		m1 = provider.get_current_metrics()
		assert m1["active_workers"] == 3

		_append_events(p, [
			{"event_type": "merged", "unit_id": "u1", "worker_id": "w1", "cost_usd": 0.0},
			{"event_type": "merge_failed", "unit_id": "u2", "worker_id": "w2", "cost_usd": 0.0},
		])
		m2 = provider.get_current_metrics()
		assert m2["active_workers"] == 1
