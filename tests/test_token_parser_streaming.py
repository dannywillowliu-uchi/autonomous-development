"""Tests for streaming token parser with backpressure."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from mission_control.token_parser import (
	StreamingTokenParser,
	TokenEvent,
	compute_token_cost_incremental,
	parse_stream_json_chunked,
)

# --- Helpers ---


async def _bytes_iter(lines: list[str]) -> AsyncIterator[bytes]:
	"""Convert a list of strings to an async iterator of byte chunks."""
	for line in lines:
		yield (line + "\n").encode("utf-8")


async def _collect(stream: AsyncIterator[TokenEvent]) -> list[TokenEvent]:
	"""Collect all TokenEvents from an async iterator."""
	events = []
	async for event in stream:
		events.append(event)
	return events


def _make_result_line(
	input_tokens: int = 0,
	output_tokens: int = 0,
	text: str = "",
) -> str:
	event = {
		"type": "result",
		"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
		"content": [{"type": "text", "text": text}] if text else [],
	}
	return json.dumps(event)


def _make_assistant_line(
	input_tokens: int = 0,
	output_tokens: int = 0,
	text: str = "",
) -> str:
	event = {
		"type": "assistant",
		"message": {
			"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
			"content": [{"type": "text", "text": text}] if text else [],
		},
	}
	return json.dumps(event)


def _make_delta_line(text: str) -> str:
	return json.dumps({
		"type": "content_block_delta",
		"delta": {"type": "text_delta", "text": text},
	})


# --- StreamingTokenParser unit tests ---


class TestStreamingTokenParser:
	def test_feed_empty_line(self) -> None:
		parser = StreamingTokenParser()
		assert parser.feed_line("") is None
		assert parser.feed_line("  ") is None

	def test_feed_result_event(self) -> None:
		parser = StreamingTokenParser()
		event = parser.feed_line(_make_result_line(100, 50, "hello"))
		assert event is not None
		assert event.input_tokens == 100
		assert event.output_tokens == 50
		assert event.content_text == "hello"
		assert event.is_truncated is False

	def test_feed_assistant_event(self) -> None:
		parser = StreamingTokenParser()
		event = parser.feed_line(_make_assistant_line(200, 80, "world"))
		assert event is not None
		assert event.input_tokens == 200
		assert event.output_tokens == 80
		assert event.content_text == "world"

	def test_feed_delta_event(self) -> None:
		parser = StreamingTokenParser()
		event = parser.feed_line(_make_delta_line("chunk"))
		assert event is not None
		assert event.content_text == "chunk"
		assert event.input_tokens == 0
		assert event.output_tokens == 0

	def test_feed_malformed_json(self) -> None:
		parser = StreamingTokenParser()
		assert parser.feed_line("not json") is None
		assert parser.feed_line("{broken") is None

	def test_feed_unknown_type(self) -> None:
		parser = StreamingTokenParser()
		assert parser.feed_line(json.dumps({"type": "ping"})) is None

	def test_backpressure_truncation(self) -> None:
		parser = StreamingTokenParser(max_buffer_bytes=10)
		# First event: 5 bytes of content
		e1 = parser.feed_line(_make_result_line(100, 50, "12345"))
		assert e1 is not None
		assert e1.content_text == "12345"
		assert e1.is_truncated is False

		# Second event: would push past 10 bytes
		e2 = parser.feed_line(_make_result_line(200, 60, "ABCDEFGHIJ"))
		assert e2 is not None
		assert e2.input_tokens == 200
		assert e2.output_tokens == 60
		assert len(e2.content_text.encode("utf-8")) <= 5
		assert e2.is_truncated is True


# --- parse_stream_json_chunked tests ---


class TestParseStreamJsonChunked:
	def test_normal_stream(self) -> None:
		lines = [
			_make_assistant_line(500, 200, "Part 1. "),
			_make_result_line(600, 300, "Part 2."),
		]
		events = asyncio.run(_collect(parse_stream_json_chunked(_bytes_iter(lines))))
		assert len(events) == 2
		assert events[0].input_tokens == 500
		assert events[0].output_tokens == 200
		assert events[0].content_text == "Part 1. "
		assert events[1].input_tokens == 600
		assert events[1].output_tokens == 300
		assert events[1].content_text == "Part 2."
		assert all(not e.is_truncated for e in events)

	def test_oversized_stream_truncation(self) -> None:
		# 20 byte limit; first event uses 10 bytes
		lines = [
			_make_assistant_line(100, 50, "A" * 10),
			_make_result_line(200, 80, "B" * 30),  # exceeds limit
		]
		events = asyncio.run(
			_collect(parse_stream_json_chunked(_bytes_iter(lines), max_buffer_bytes=20))
		)
		assert len(events) == 2
		# Token counts must be accurate regardless of truncation
		assert events[0].input_tokens == 100
		assert events[0].output_tokens == 50
		assert events[1].input_tokens == 200
		assert events[1].output_tokens == 80
		# First event fits, second is truncated
		assert events[0].is_truncated is False
		assert events[1].is_truncated is True
		assert len(events[1].content_text.encode("utf-8")) <= 10

	def test_empty_stream(self) -> None:
		async def empty_stream() -> AsyncIterator[bytes]:
			return
			yield  # noqa: RET504 - makes this an async generator

		events = asyncio.run(_collect(parse_stream_json_chunked(empty_stream())))
		assert events == []

	def test_malformed_lines_skipped(self) -> None:
		async def mixed_stream() -> AsyncIterator[bytes]:
			yield b"not json\n"
			yield b"{broken json}\n"
			yield (_make_result_line(42, 7, "ok") + "\n").encode()

		events = asyncio.run(_collect(parse_stream_json_chunked(mixed_stream())))
		assert len(events) == 1
		assert events[0].input_tokens == 42
		assert events[0].output_tokens == 7
		assert events[0].content_text == "ok"

	def test_multi_line_chunk(self) -> None:
		"""Multiple NDJSON lines in a single byte chunk."""
		combined = (
			_make_delta_line("hello") + "\n" + _make_delta_line(" world") + "\n"
		)

		async def single_chunk() -> AsyncIterator[bytes]:
			yield combined.encode()

		events = asyncio.run(_collect(parse_stream_json_chunked(single_chunk())))
		assert len(events) == 2
		assert events[0].content_text == "hello"
		assert events[1].content_text == " world"

	def test_split_across_chunks(self) -> None:
		"""A single NDJSON line split across two byte chunks."""
		full_line = _make_result_line(10, 5, "split")
		mid = len(full_line) // 2

		async def split_stream() -> AsyncIterator[bytes]:
			yield full_line[:mid].encode()
			yield (full_line[mid:] + "\n").encode()

		events = asyncio.run(_collect(parse_stream_json_chunked(split_stream())))
		assert len(events) == 1
		assert events[0].input_tokens == 10
		assert events[0].content_text == "split"


# --- compute_token_cost_incremental tests ---


class TestComputeTokenCostIncremental:
	def test_basic_cost(self) -> None:
		events = [
			TokenEvent(input_tokens=1_000_000, output_tokens=0),
			TokenEvent(input_tokens=0, output_tokens=1_000_000),
		]
		inp, out, cost = compute_token_cost_incremental(events)
		assert inp == 1_000_000
		assert out == 1_000_000
		# Default: 3.0 input + 15.0 output
		assert abs(cost - 18.0) < 0.001

	def test_empty_events(self) -> None:
		inp, out, cost = compute_token_cost_incremental([])
		assert inp == 0
		assert out == 0
		assert cost == 0.0

	def test_with_pricing(self) -> None:
		from mission_control.config import PricingConfig

		events = [TokenEvent(input_tokens=500_000, output_tokens=100_000)]
		pricing = PricingConfig(input_per_million=6.0, output_per_million=30.0)
		inp, out, cost = compute_token_cost_incremental(events, pricing=pricing)
		assert inp == 500_000
		assert out == 100_000
		expected = (500_000 * 6.0 / 1_000_000) + (100_000 * 30.0 / 1_000_000)
		assert abs(cost - expected) < 0.001

	def test_truncated_events_preserve_tokens(self) -> None:
		"""Token counts are accurate even when content is truncated."""
		events = [
			TokenEvent(input_tokens=100, output_tokens=50, content_text="ok", is_truncated=False),
			TokenEvent(input_tokens=200, output_tokens=80, content_text="tr", is_truncated=True),
		]
		inp, out, cost = compute_token_cost_incremental(events)
		assert inp == 300
		assert out == 130


# --- TokenEvent dataclass tests ---


class TestTokenEvent:
	def test_defaults(self) -> None:
		e = TokenEvent()
		assert e.input_tokens == 0
		assert e.output_tokens == 0
		assert e.content_text == ""
		assert e.is_truncated is False

	def test_fields(self) -> None:
		e = TokenEvent(input_tokens=10, output_tokens=5, content_text="hi", is_truncated=True)
		assert e.input_tokens == 10
		assert e.output_tokens == 5
		assert e.content_text == "hi"
		assert e.is_truncated is True
