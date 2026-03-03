"""JSONL event stream for structured post-mission analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from mission_control.tracing import get_current_trace_context

# ---------------------------------------------------------------------------
# Analytics / aggregation helpers (pure functions over event dicts)
# ---------------------------------------------------------------------------


def read_events(
	path: Path,
	event_type: str | None = None,
	unit_id: str | None = None,
	mission_id: str | None = None,
) -> list[dict]:
	"""Read and optionally filter events from a JSONL event stream file."""
	if not path.exists():
		return []
	events: list[dict] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			record = json.loads(line)
			if event_type is not None and record.get("event_type") != event_type:
				continue
			if unit_id is not None and record.get("unit_id") != unit_id:
				continue
			if mission_id is not None and record.get("mission_id") != mission_id:
				continue
			events.append(record)
	return events


def compute_failure_rate(events: list[dict]) -> float:
	"""Ratio of 'merge_failed' events to total 'dispatched' events.

	Returns 0.0 when there are no dispatched events.
	"""
	dispatched = sum(1 for e in events if e.get("event_type") == "dispatched")
	if dispatched == 0:
		return 0.0
	failed = sum(1 for e in events if e.get("event_type") == "merge_failed")
	return failed / dispatched


def compute_cost_summary(events: list[dict]) -> dict:
	"""Aggregate cost metrics across events.

	Returns dict with keys: total_cost, mean_cost, max_cost.
	All values are 0.0 when events list is empty or no costs recorded.
	"""
	costs = [e.get("cost_usd", 0.0) for e in events if e.get("cost_usd", 0.0) > 0]
	if not costs:
		return {"total_cost": 0.0, "mean_cost": 0.0, "max_cost": 0.0}
	return {
		"total_cost": sum(costs),
		"mean_cost": sum(costs) / len(costs),
		"max_cost": max(costs),
	}


def compute_merge_velocity(events: list[dict]) -> float:
	"""Successful merges per hour based on event timestamps.

	Uses the time span from the earliest to the latest event as the window.
	Returns 0.0 when there are fewer than 2 events or no merged events.
	"""
	merged_count = sum(1 for e in events if e.get("event_type") == "merged")
	if merged_count == 0:
		return 0.0
	timestamps = []
	for e in events:
		ts = e.get("timestamp")
		if ts:
			timestamps.append(datetime.fromisoformat(ts))
	if len(timestamps) < 2:
		return 0.0
	span_seconds = (max(timestamps) - min(timestamps)).total_seconds()
	if span_seconds <= 0:
		return 0.0
	return merged_count / (span_seconds / 3600)


def compute_unit_duration(events: list[dict], unit_id: str) -> float | None:
	"""Seconds between first dispatch and final merged/failed event for a unit.

	Returns None if the unit has no dispatched event or no terminal event.
	"""
	unit_events = [e for e in events if e.get("unit_id") == unit_id]
	dispatch_ts: datetime | None = None
	terminal_ts: datetime | None = None
	for e in unit_events:
		ts_str = e.get("timestamp")
		if not ts_str:
			continue
		ts = datetime.fromisoformat(ts_str)
		if e.get("event_type") == "dispatched" and dispatch_ts is None:
			dispatch_ts = ts
		if e.get("event_type") in ("merged", "merge_failed"):
			terminal_ts = ts  # keep last terminal event
	if dispatch_ts is None or terminal_ts is None:
		return None
	return (terminal_ts - dispatch_ts).total_seconds()


class EventStream:
	"""Append-only JSONL writer for mission events.

	Complements the DB event store with a portable, jq-friendly format.
	"""

	def __init__(self, path: Path) -> None:
		self._path = path
		self._file: IO[str] | None = None

	def open(self) -> None:
		self._path.parent.mkdir(parents=True, exist_ok=True)
		self._file = self._path.open("a", encoding="utf-8")

	def close(self) -> None:
		if self._file is not None:
			self._file.close()
			self._file = None

	def emit(
		self,
		event_type: str,
		*,
		mission_id: str = "",
		epoch_id: str = "",
		unit_id: str = "",
		worker_id: str = "",
		details: dict[str, Any] | None = None,
		input_tokens: int = 0,
		output_tokens: int = 0,
		cost_usd: float = 0.0,
		trace_id: str = "",
		span_id: str = "",
	) -> None:
		if self._file is None:
			return
		# Auto-extract trace context from OTEL if not explicitly provided
		if not trace_id:
			trace_id, span_id = get_current_trace_context()
		record: dict[str, Any] = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"event_type": event_type,
			"mission_id": mission_id,
			"epoch_id": epoch_id,
			"unit_id": unit_id,
			"worker_id": worker_id,
			"details": details or {},
			"input_tokens": input_tokens,
			"output_tokens": output_tokens,
			"cost_usd": cost_usd,
			"trace_id": trace_id,
			"span_id": span_id,
		}
		self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
		self._file.flush()
