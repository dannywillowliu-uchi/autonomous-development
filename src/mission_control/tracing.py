"""OpenTelemetry tracing integration with no-op fallback."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from mission_control.config import TracingConfig

logger = logging.getLogger(__name__)

try:
	from opentelemetry import trace
	from opentelemetry.sdk.resources import Resource
	from opentelemetry.sdk.trace import TracerProvider
	from opentelemetry.sdk.trace.export import (
		ConsoleSpanExporter,
		SimpleSpanProcessor,
	)

	OTEL_AVAILABLE = True
except ImportError:
	OTEL_AVAILABLE = False


class NoOpSpan:
	"""A no-op span that acts as a context manager and attribute sink."""

	def set_attribute(self, key: str, value: Any) -> None:
		pass

	def set_status(self, status: Any, description: str | None = None) -> None:
		pass

	def record_exception(self, exception: BaseException) -> None:
		pass

	def end(self) -> None:
		pass

	@property
	def context(self) -> None:
		return None

	def get_span_context(self) -> _NoOpSpanContext:
		return _NoOpSpanContext()


class _NoOpSpanContext:
	"""Minimal span context for NoOpSpan."""

	trace_id: int = 0
	span_id: int = 0


class MissionTracer:
	"""Manages OpenTelemetry tracing with graceful no-op fallback.

	When OTEL is not installed or tracing is disabled, all methods return
	NoOpSpan instances that silently discard attributes and events.
	"""

	def __init__(self, config: TracingConfig) -> None:
		self._config = config
		self._tracer: Any = None

		if not config.enabled or not OTEL_AVAILABLE:
			if config.enabled and not OTEL_AVAILABLE:
				logger.warning(
					"Tracing enabled but opentelemetry not installed. "
					"Install with: pip install mission-control[tracing]"
				)
			return

		resource = Resource.create({"service.name": config.service_name})
		provider = TracerProvider(resource=resource)

		if config.exporter == "console":
			provider.add_span_processor(
				SimpleSpanProcessor(ConsoleSpanExporter())
			)
		elif config.exporter == "otlp":
			try:
				from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
				provider.add_span_processor(
					SimpleSpanProcessor(OTLPSpanExporter(endpoint=config.otlp_endpoint))
				)
			except ImportError:
				logger.warning(
					"OTLP exporter not available. Install opentelemetry-exporter-otlp-proto-grpc"
				)
				provider.add_span_processor(
					SimpleSpanProcessor(ConsoleSpanExporter())
				)

		trace.set_tracer_provider(provider)
		self._tracer = trace.get_tracer("mission-control")

	@property
	def active(self) -> bool:
		return self._tracer is not None

	@contextmanager
	def start_mission_span(self, mission_id: str) -> Generator[Any, None, None]:
		if not self.active:
			yield NoOpSpan()
			return
		with self._tracer.start_as_current_span("mission") as span:
			span.set_attribute("mission.id", mission_id)
			yield span

	@contextmanager
	def start_epoch_span(self, epoch_id: str) -> Generator[Any, None, None]:
		if not self.active:
			yield NoOpSpan()
			return
		with self._tracer.start_as_current_span("epoch") as span:
			span.set_attribute("epoch.id", epoch_id)
			yield span

	@contextmanager
	def start_unit_span(self, unit_id: str) -> Generator[Any, None, None]:
		if not self.active:
			yield NoOpSpan()
			return
		with self._tracer.start_as_current_span("unit") as span:
			span.set_attribute("unit.id", unit_id)
			yield span

	@contextmanager
	def start_verification_span(self) -> Generator[Any, None, None]:
		if not self.active:
			yield NoOpSpan()
			return
		with self._tracer.start_as_current_span("verification") as span:
			yield span


def get_current_trace_context() -> tuple[str, str]:
	"""Extract trace_id and span_id from the current OTEL context.

	Returns ("", "") if OTEL is not available or no active span.
	"""
	if not OTEL_AVAILABLE:
		return ("", "")
	span = trace.get_current_span()
	ctx = span.get_span_context()
	if ctx is None or ctx.trace_id == 0:
		return ("", "")
	return (
		format(ctx.trace_id, "032x"),
		format(ctx.span_id, "016x"),
	)
