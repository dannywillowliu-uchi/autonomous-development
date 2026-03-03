"""Tests for exception capture, error status, and custom events on tracing spans."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mission_control.config import TracingConfig
from mission_control.tracing import (
	MissionSpan,
	MissionTracer,
	NoOpSpan,
	traced_operation,
)

# ---------------------------------------------------------------------------
# NoOpSpan no-op methods
# ---------------------------------------------------------------------------


class TestNoOpSpanErrorMethods:
	def test_record_exception_is_noop(self) -> None:
		span = NoOpSpan()
		span.record_exception(RuntimeError("boom"))

	def test_set_error_status_is_noop(self) -> None:
		span = NoOpSpan()
		span.set_error_status("something went wrong")

	def test_add_event_is_noop(self) -> None:
		span = NoOpSpan()
		span.add_event("my.event")

	def test_add_event_with_attributes_is_noop(self) -> None:
		span = NoOpSpan()
		span.add_event("my.event", attributes={"count": 42, "ratio": 0.5})


# ---------------------------------------------------------------------------
# MissionSpan delegation
# ---------------------------------------------------------------------------


class TestMissionSpanRecordException:
	def test_delegates_to_underlying_span(self) -> None:
		mock_span = MagicMock()
		ms = MissionSpan(mock_span)
		exc = ValueError("test error")
		ms.record_exception(exc)
		mock_span.record_exception.assert_called_once_with(exc)


class TestMissionSpanSetErrorStatus:
	def test_delegates_set_status_with_error_code(self) -> None:
		mock_span = MagicMock()
		mock_status_code = MagicMock()
		ms = MissionSpan(mock_span)
		with patch("mission_control.tracing._StatusCode", mock_status_code):
			ms.set_error_status("bad things happened")
			mock_span.set_status.assert_called_once_with(mock_status_code.ERROR, "bad things happened")

	def test_noop_when_otel_unavailable(self) -> None:
		mock_span = MagicMock()
		ms = MissionSpan(mock_span)
		with patch("mission_control.tracing._StatusCode", None):
			ms.set_error_status("should not call set_status")
			mock_span.set_status.assert_not_called()


class TestMissionSpanAddEvent:
	def test_add_event_without_attributes(self) -> None:
		mock_span = MagicMock()
		ms = MissionSpan(mock_span)
		ms.add_event("checkpoint.reached")
		mock_span.add_event.assert_called_once_with("checkpoint.reached")

	def test_add_event_with_attributes(self) -> None:
		mock_span = MagicMock()
		ms = MissionSpan(mock_span)
		attrs = {"items": 5, "ratio": 0.95}
		ms.add_event("batch.complete", attributes=attrs)
		mock_span.add_event.assert_called_once_with("batch.complete", attributes=attrs)


# ---------------------------------------------------------------------------
# traced_operation context manager
# ---------------------------------------------------------------------------


class TestTracedOperationSuccess:
	def test_noop_path_completes_without_error(self) -> None:
		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		with traced_operation(tracer, "test_op") as span:
			assert isinstance(span, NoOpSpan)

	def test_noop_path_sets_attributes_on_span(self) -> None:
		"""NoOpSpan silently accepts attributes set by traced_operation."""
		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		with traced_operation(tracer, "test_op", key="value") as span:
			assert isinstance(span, NoOpSpan)


class TestTracedOperationFailure:
	def test_noop_path_reraises_exception(self) -> None:
		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		with pytest.raises(RuntimeError, match="boom"):
			with traced_operation(tracer, "failing_op"):
				raise RuntimeError("boom")

	def test_active_path_reraises_exception(self) -> None:
		"""With a mock tracer, exceptions propagate after being recorded."""
		mock_raw_span = MagicMock()
		mock_tracer_obj = MagicMock()
		mock_tracer_obj.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_raw_span)
		mock_tracer_obj.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		tracer._tracer = mock_tracer_obj  # force active

		with pytest.raises(ValueError, match="oops"):
			with traced_operation(tracer, "failing_op", unit_id="u1") as span:
				assert isinstance(span, MissionSpan)
				raise ValueError("oops")

		mock_raw_span.record_exception.assert_called_once()
		recorded_exc = mock_raw_span.record_exception.call_args[0][0]
		assert isinstance(recorded_exc, ValueError)
		assert str(recorded_exc) == "oops"

	def test_active_path_sets_error_status_on_exception(self) -> None:
		mock_raw_span = MagicMock()
		mock_tracer_obj = MagicMock()
		mock_tracer_obj.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_raw_span)
		mock_tracer_obj.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
		mock_status_code = MagicMock()

		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		tracer._tracer = mock_tracer_obj

		with patch("mission_control.tracing._StatusCode", mock_status_code):
			with pytest.raises(TypeError):
				with traced_operation(tracer, "err_op"):
					raise TypeError("type issue")

		mock_raw_span.set_status.assert_called_once_with(mock_status_code.ERROR, "type issue")

	def test_active_path_sets_attributes(self) -> None:
		mock_raw_span = MagicMock()
		mock_tracer_obj = MagicMock()
		mock_tracer_obj.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_raw_span)
		mock_tracer_obj.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		tracer._tracer = mock_tracer_obj

		with traced_operation(tracer, "op", foo="bar", count=3):
			pass

		calls = mock_raw_span.set_attribute.call_args_list
		assert any(c[0] == ("foo", "bar") for c in calls)
		assert any(c[0] == ("count", 3) for c in calls)

	def test_active_path_no_error_on_success(self) -> None:
		mock_raw_span = MagicMock()
		mock_tracer_obj = MagicMock()
		mock_tracer_obj.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_raw_span)
		mock_tracer_obj.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

		config = TracingConfig(enabled=False)
		tracer = MissionTracer(config)
		tracer._tracer = mock_tracer_obj

		with traced_operation(tracer, "ok_op") as span:
			span.add_event("progress")

		mock_raw_span.record_exception.assert_not_called()
		mock_raw_span.set_status.assert_not_called()
