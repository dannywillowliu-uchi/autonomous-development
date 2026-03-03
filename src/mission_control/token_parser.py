"""Parse Claude's stream-json output for token usage and MC_RESULT."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field


@dataclass
class TokenUsage:
	"""Accumulated token counts from a Claude session."""

	input_tokens: int = 0
	output_tokens: int = 0
	cache_creation_tokens: int = 0
	cache_read_tokens: int = 0


@dataclass
class StreamJsonResult:
	"""Parsed result from Claude's stream-json output format."""

	usage: TokenUsage = field(default_factory=TokenUsage)
	text_content: str = ""
	mc_result: dict[str, object] | None = None


def parse_stream_json(output: str) -> StreamJsonResult:
	"""Parse NDJSON stream-json output from Claude CLI.

	Each line is a JSON object. For type=="assistant", accumulates
	message.usage tokens and concatenates content[].text.
	Then extracts MC_RESULT from the concatenated text.

	Args:
		output: Raw stdout from `claude -p --output-format stream-json`.

	Returns:
		StreamJsonResult with accumulated tokens, text, and parsed MC_RESULT.
	"""
	result = StreamJsonResult()
	if not output or not output.strip():
		return result

	texts: list[str] = []
	usage = TokenUsage()

	for line in output.splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			event = json.loads(line)
		except (json.JSONDecodeError, ValueError):
			continue

		if not isinstance(event, dict):
			continue

		event_type = event.get("type", "")

		if event_type == "result":
			# Final result message has top-level usage
			msg_usage = event.get("usage", {})
			if isinstance(msg_usage, dict):
				usage.input_tokens += int(msg_usage.get("input_tokens", 0))
				usage.output_tokens += int(msg_usage.get("output_tokens", 0))
				usage.cache_creation_tokens += int(
					msg_usage.get("cache_creation_input_tokens", 0)
				)
				usage.cache_read_tokens += int(
					msg_usage.get("cache_read_input_tokens", 0)
				)
			# Extract text from result content
			for block in event.get("content", []):
				if isinstance(block, dict) and block.get("type") == "text":
					texts.append(str(block.get("text", "")))

		elif event_type == "assistant":
			# Assistant message with usage and content
			msg = event.get("message", event)
			msg_usage = msg.get("usage", {})
			if isinstance(msg_usage, dict):
				usage.input_tokens += int(msg_usage.get("input_tokens", 0))
				usage.output_tokens += int(msg_usage.get("output_tokens", 0))
				usage.cache_creation_tokens += int(
					msg_usage.get("cache_creation_input_tokens", 0)
				)
				usage.cache_read_tokens += int(
					msg_usage.get("cache_read_input_tokens", 0)
				)
			for block in msg.get("content", []):
				if isinstance(block, dict) and block.get("type") == "text":
					texts.append(str(block.get("text", "")))

		elif event_type == "content_block_delta":
			delta = event.get("delta", {})
			if isinstance(delta, dict) and delta.get("type") == "text_delta":
				texts.append(str(delta.get("text", "")))

	result.usage = usage
	result.text_content = "".join(texts)

	# Extract MC_RESULT from concatenated text
	from mission_control.session import parse_mc_result

	mc = parse_mc_result(result.text_content)
	result.mc_result = mc

	return result


DEFAULT_MAX_BUFFER_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass
class TokenEvent:
	"""A single token usage event extracted from a stream chunk."""

	input_tokens: int = 0
	output_tokens: int = 0
	content_text: str = ""
	is_truncated: bool = False


class StreamingTokenParser:
	"""Processes stream-json output in chunks with backpressure.

	Instead of buffering the entire stream, extracts token/cost data
	as each JSON line arrives and discards raw content after extraction.
	"""

	def __init__(self, max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES) -> None:
		self.max_buffer_bytes = max_buffer_bytes
		self._buffer = ""
		self._content_bytes = 0

	def feed_line(self, line: str) -> TokenEvent | None:
		"""Parse a single NDJSON line and return a TokenEvent if relevant.

		Returns None for non-content lines (malformed, unknown types, etc.).
		"""
		line = line.strip()
		if not line:
			return None

		try:
			event = json.loads(line)
		except (json.JSONDecodeError, ValueError):
			return None

		if not isinstance(event, dict):
			return None

		return self._extract_event(event)

	def _extract_event(self, event: dict) -> TokenEvent | None:
		event_type = event.get("type", "")
		input_tokens = 0
		output_tokens = 0
		texts: list[str] = []

		if event_type == "result":
			msg_usage = event.get("usage", {})
			if isinstance(msg_usage, dict):
				input_tokens = int(msg_usage.get("input_tokens", 0))
				output_tokens = int(msg_usage.get("output_tokens", 0))
			for block in event.get("content", []):
				if isinstance(block, dict) and block.get("type") == "text":
					texts.append(str(block.get("text", "")))

		elif event_type == "assistant":
			msg = event.get("message", event)
			msg_usage = msg.get("usage", {})
			if isinstance(msg_usage, dict):
				input_tokens = int(msg_usage.get("input_tokens", 0))
				output_tokens = int(msg_usage.get("output_tokens", 0))
			for block in msg.get("content", []):
				if isinstance(block, dict) and block.get("type") == "text":
					texts.append(str(block.get("text", "")))

		elif event_type == "content_block_delta":
			delta = event.get("delta", {})
			if isinstance(delta, dict) and delta.get("type") == "text_delta":
				texts.append(str(delta.get("text", "")))

		else:
			return None

		content_text = "".join(texts)
		is_truncated = False

		# Backpressure: truncate content if buffer would exceed limit
		new_bytes = len(content_text.encode("utf-8"))
		if self._content_bytes + new_bytes > self.max_buffer_bytes:
			remaining = max(0, self.max_buffer_bytes - self._content_bytes)
			content_text = content_text.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
			is_truncated = True

		self._content_bytes += len(content_text.encode("utf-8"))

		if input_tokens == 0 and output_tokens == 0 and not content_text and not is_truncated:
			# content_block_delta with empty text after truncation - still yield for token accuracy
			if texts:
				return TokenEvent(is_truncated=is_truncated)
			return None

		return TokenEvent(
			input_tokens=input_tokens,
			output_tokens=output_tokens,
			content_text=content_text,
			is_truncated=is_truncated,
		)


async def parse_stream_json_chunked(
	stream: AsyncIterator[bytes],
	max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
) -> AsyncIterator[TokenEvent]:
	"""Async generator that yields TokenEvents from a stream-json byte stream.

	Reads chunks incrementally, splits on newlines, and yields TokenEvent
	objects as each JSON line arrives. If accumulated content exceeds
	max_buffer_bytes, truncates content_text but keeps token counts accurate.

	Args:
		stream: Async iterator of raw bytes (e.g., from subprocess stdout).
		max_buffer_bytes: Maximum content buffer size before truncation.

	Yields:
		TokenEvent for each relevant NDJSON line.
	"""
	parser = StreamingTokenParser(max_buffer_bytes=max_buffer_bytes)
	leftover = ""

	async for chunk in stream:
		text = leftover + chunk.decode("utf-8", errors="replace")
		lines = text.split("\n")
		# Last element is incomplete line (or empty if chunk ended with \n)
		leftover = lines[-1]

		for line in lines[:-1]:
			event = parser.feed_line(line)
			if event is not None:
				yield event

	# Process any remaining data
	if leftover.strip():
		event = parser.feed_line(leftover)
		if event is not None:
			yield event


def compute_token_cost_incremental(
	events: Iterable[TokenEvent],
	pricing: object | None = None,
) -> tuple[int, int, float]:
	"""Compute token cost from an iterable of TokenEvents without buffering.

	Args:
		events: Iterable of TokenEvent objects.
		pricing: PricingConfig with per-million rates. Uses defaults if None.

	Returns:
		Tuple of (input_tokens, output_tokens, cost_usd).
	"""
	total_input = 0
	total_output = 0

	for event in events:
		total_input += event.input_tokens
		total_output += event.output_tokens

	input_rate = getattr(pricing, "input_per_million", 3.0) if pricing else 3.0
	output_rate = getattr(pricing, "output_per_million", 15.0) if pricing else 15.0

	cost = (total_input * input_rate / 1_000_000) + (total_output * output_rate / 1_000_000)

	return (total_input, total_output, cost)


def compute_token_cost(usage: TokenUsage, pricing: object) -> float:
	"""Compute USD cost from token usage and pricing config.

	Args:
		usage: Token counts from a session.
		pricing: PricingConfig with per-million rates.

	Returns:
		Total cost in USD.
	"""
	cost = 0.0
	cost += usage.input_tokens * getattr(pricing, "input_per_million", 3.0) / 1_000_000
	cost += usage.output_tokens * getattr(pricing, "output_per_million", 15.0) / 1_000_000
	cost += usage.cache_creation_tokens * getattr(pricing, "cache_write_per_million", 3.75) / 1_000_000
	cost += usage.cache_read_tokens * getattr(pricing, "cache_read_per_million", 0.30) / 1_000_000
	return cost
