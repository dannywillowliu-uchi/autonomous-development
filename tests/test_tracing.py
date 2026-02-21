"""Tests for OpenTelemetry tracing integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mission_control.config import TracingConfig
from mission_control.event_stream import EventStream
from mission_control.tracing import OTEL_AVAILABLE, MissionTracer, NoOpSpan, get_current_trace_context


class TestNoOpSpan:
	def test_set_attribute_is_noop(self) -> None:
		span = NoOpSpan()
		span.set_attribute("key", "value")  # should not raise

	def test_set_status_is_noop(self) -> None:
		span = NoOpSpan()
		span.set_status("ok")  # should not raise

	def test_record_exception_is_noop(self) -> None:
		span = NoOpSpan()
		span.record_exception(ValueError("test"))  # should not raise

	def test_end_is_noop(self) -> None:
		span = NoOpSpan()
		span.end()  # should not raise

	def test_context_is_none(self) -> None:
		span = NoOpSpan()
		assert span.context is None

	def test_span_context_has_zero_ids(self) -> None:
		span = NoOpSpan()
		ctx = span.get_span_context()
		assert ctx.trace_id == 0
		assert ctx.span_id == 0


class TestNoOpFallback:
	def test_tracer_works_when_otel_not_installed(self) -> None:
		"""MissionTracer returns NoOpSpan when OTEL is unavailable."""
		with patch("mission_control.tracing.OTEL_AVAILABLE", False):
			config = TracingConfig(enabled=True)
			tracer = MissionTracer(config)
			assert not tracer.active
			with tracer.start_mission_span("m1") as span:
				assert isinstance(span, NoOpSpan)


class TestTracingDisabled:
	def test_returns_noop_when_disabled(self) -> None:
		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		assert not tracer.active
		with tracer.start_mission_span("m1") as span:
			assert isinstance(span, NoOpSpan)

	def test_epoch_span_noop(self) -> None:
		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		with tracer.start_epoch_span("e1") as span:
			assert isinstance(span, NoOpSpan)

	def test_unit_span_noop(self) -> None:
		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		with tracer.start_unit_span("u1") as span:
			assert isinstance(span, NoOpSpan)

	def test_verification_span_noop(self) -> None:
		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		with tracer.start_verification_span() as span:
			assert isinstance(span, NoOpSpan)


class TestSpanHierarchy:
	def test_noop_spans_nest_correctly(self) -> None:
		"""NoOp spans can nest without error."""
		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		with tracer.start_mission_span("m1") as mission_span:
			with tracer.start_epoch_span("e1") as epoch_span:
				with tracer.start_unit_span("u1") as unit_span:
					with tracer.start_verification_span() as verify_span:
						assert isinstance(mission_span, NoOpSpan)
						assert isinstance(epoch_span, NoOpSpan)
						assert isinstance(unit_span, NoOpSpan)
						assert isinstance(verify_span, NoOpSpan)


class TestGetCurrentTraceContext:
	def test_returns_empty_when_otel_unavailable(self) -> None:
		with patch("mission_control.tracing.OTEL_AVAILABLE", False):
			trace_id, span_id = get_current_trace_context()
			assert trace_id == ""
			assert span_id == ""

	def test_returns_empty_when_no_active_span(self) -> None:
		if not OTEL_AVAILABLE:
			trace_id, span_id = get_current_trace_context()
			assert trace_id == ""
			assert span_id == ""


class TestEventStreamTraceContext:
	def test_emit_includes_trace_fields(self, tmp_path: Path) -> None:
		"""emit() includes trace_id and span_id fields in JSONL output."""
		stream = EventStream(tmp_path / "events.jsonl")
		stream.open()
		stream.emit(
			"test_event",
			mission_id="m1",
			trace_id="abc123",
			span_id="def456",
		)
		stream.close()

		line = (tmp_path / "events.jsonl").read_text().strip()
		record = json.loads(line)
		assert record["trace_id"] == "abc123"
		assert record["span_id"] == "def456"

	def test_emit_auto_extracts_trace_context(self, tmp_path: Path) -> None:
		"""When no trace_id provided, emit() calls get_current_trace_context."""
		stream = EventStream(tmp_path / "events.jsonl")
		stream.open()
		with patch("mission_control.event_stream.get_current_trace_context", return_value=("auto-trace", "auto-span")):
			stream.emit("test_event", mission_id="m1")
		stream.close()

		line = (tmp_path / "events.jsonl").read_text().strip()
		record = json.loads(line)
		assert record["trace_id"] == "auto-trace"
		assert record["span_id"] == "auto-span"

	def test_emit_without_trace_has_empty_fields(self, tmp_path: Path) -> None:
		"""Without OTEL, trace_id and span_id are empty strings."""
		stream = EventStream(tmp_path / "events.jsonl")
		stream.open()
		with patch("mission_control.event_stream.get_current_trace_context", return_value=("", "")):
			stream.emit("test_event", mission_id="m1")
		stream.close()

		line = (tmp_path / "events.jsonl").read_text().strip()
		record = json.loads(line)
		assert record["trace_id"] == ""
		assert record["span_id"] == ""
