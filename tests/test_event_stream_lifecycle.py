"""Tests for EventStream write-path lifecycle (open/emit/close)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

from mission_control.event_stream import EventStream


class TestOpen:
	def test_open_creates_jsonl_file(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		try:
			assert p.exists()
		finally:
			stream.close()

	def test_open_creates_parent_directories(self, tmp_path: Path) -> None:
		p = tmp_path / "deep" / "nested" / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		try:
			assert p.exists()
		finally:
			stream.close()

	def test_open_appends_to_existing_file(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		p.write_text('{"existing":"line"}\n')
		stream = EventStream(p)
		stream.open()
		stream.emit("test_event")
		stream.close()
		lines = p.read_text().strip().split("\n")
		assert len(lines) == 2
		assert json.loads(lines[0]) == {"existing": "line"}


class TestEmit:
	def test_emit_writes_valid_json_line(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("dispatched", mission_id="m1", unit_id="u1")
		stream.close()
		line = p.read_text().strip()
		record = json.loads(line)
		assert record["event_type"] == "dispatched"
		assert record["mission_id"] == "m1"
		assert record["unit_id"] == "u1"

	def test_emit_includes_all_schema_fields(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit(
			"dispatched",
			mission_id="m1",
			epoch_id="e1",
			unit_id="u1",
			worker_id="w1",
			details={"key": "val"},
			input_tokens=100,
			output_tokens=50,
			cost_usd=0.42,
			trace_id="t1",
			span_id="s1",
		)
		stream.close()
		record = json.loads(p.read_text().strip())
		expected_keys = {
			"timestamp", "event_type", "mission_id", "epoch_id",
			"unit_id", "worker_id", "details", "input_tokens",
			"output_tokens", "cost_usd", "trace_id", "span_id",
		}
		assert set(record.keys()) == expected_keys
		assert record["epoch_id"] == "e1"
		assert record["worker_id"] == "w1"
		assert record["details"] == {"key": "val"}
		assert record["input_tokens"] == 100
		assert record["output_tokens"] == 50
		assert record["cost_usd"] == 0.42

	def test_emit_timestamp_is_iso_utc(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("test")
		stream.close()
		record = json.loads(p.read_text().strip())
		ts = record["timestamp"]
		assert "+00:00" in ts or ts.endswith("Z")

	def test_emit_defaults_details_to_empty_dict(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("test")
		stream.close()
		record = json.loads(p.read_text().strip())
		assert record["details"] == {}

	def test_multiple_emits_produce_valid_jsonl(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		for i in range(5):
			stream.emit("event", unit_id=f"u{i}")
		stream.close()
		lines = [ln for ln in p.read_text().split("\n") if ln.strip()]
		assert len(lines) == 5
		for i, line in enumerate(lines):
			record = json.loads(line)
			assert record["unit_id"] == f"u{i}"


class TestClose:
	def test_close_flushes_data(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("dispatched")
		stream.close()
		# After close, data must be fully written
		content = p.read_text().strip()
		assert len(content) > 0
		json.loads(content)  # valid JSON

	def test_close_sets_file_to_none(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		assert stream._file is not None
		stream.close()
		assert stream._file is None

	def test_close_idempotent(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.close()
		stream.close()  # should not raise
		assert stream._file is None


class TestEmitAfterClose:
	def test_emit_after_close_is_noop(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("before_close")
		stream.close()
		stream.emit("after_close")
		lines = [ln for ln in p.read_text().split("\n") if ln.strip()]
		assert len(lines) == 1
		assert json.loads(lines[0])["event_type"] == "before_close"

	def test_emit_without_open_is_noop(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.emit("no_open")
		assert not p.exists()


class TestTraceContextInjection:
	def test_explicit_trace_context_used(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		stream.emit("test", trace_id="explicit-trace", span_id="explicit-span")
		stream.close()
		record = json.loads(p.read_text().strip())
		assert record["trace_id"] == "explicit-trace"
		assert record["span_id"] == "explicit-span"

	def test_auto_trace_context_from_otel(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		with patch(
			"mission_control.event_stream.get_current_trace_context",
			return_value=("auto-trace-123", "auto-span-456"),
		):
			stream.emit("test")
		stream.close()
		record = json.loads(p.read_text().strip())
		assert record["trace_id"] == "auto-trace-123"
		assert record["span_id"] == "auto-span-456"

	def test_no_trace_context_available(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		with patch(
			"mission_control.event_stream.get_current_trace_context",
			return_value=("", ""),
		):
			stream.emit("test")
		stream.close()
		record = json.loads(p.read_text().strip())
		assert record["trace_id"] == ""
		assert record["span_id"] == ""


class TestConcurrentEmit:
	def test_concurrent_emits_produce_valid_jsonl(self, tmp_path: Path) -> None:
		p = tmp_path / "events.jsonl"
		stream = EventStream(p)
		stream.open()
		errors: list[Exception] = []

		def emit_batch(start: int) -> None:
			try:
				for i in range(20):
					stream.emit("concurrent", unit_id=f"u{start + i}")
			except Exception as exc:
				errors.append(exc)

		threads = [threading.Thread(target=emit_batch, args=(i * 20,)) for i in range(5)]
		for t in threads:
			t.start()
		for t in threads:
			t.join()
		stream.close()

		assert not errors
		lines = [ln for ln in p.read_text().split("\n") if ln.strip()]
		assert len(lines) == 100
		# Every line must be valid JSON
		for line in lines:
			record = json.loads(line)
			assert record["event_type"] == "concurrent"
