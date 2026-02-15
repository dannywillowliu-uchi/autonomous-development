"""Tests for TelegramNotifier."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from mission_control.notifier import (
	MAX_QUEUE_SIZE,
	MAX_RETRIES,
	RETRY_BASE_DELAY,
	TELEGRAM_MAX_LEN,
	NotificationPriority,
	TelegramNotifier,
)


@pytest.fixture
def notifier() -> TelegramNotifier:
	return TelegramNotifier(bot_token="test-token", chat_id="12345")


class TestTelegramNotifier:
	@pytest.mark.asyncio
	async def test_flush_batch_sends_via_httpx(self, notifier: TelegramNotifier) -> None:
		mock_response = httpx.Response(200)
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.return_value = mock_response
			await notifier._flush_batch(["test message"])

			mock_post.assert_called_once()
			_, kwargs = mock_post.call_args
			assert "test-token" in mock_post.call_args[0][0]
			assert kwargs["json"]["chat_id"] == "12345"
			assert kwargs["json"]["text"] == "test message"

		await notifier.close()

	@pytest.mark.asyncio
	async def test_flush_batch_concatenates_messages(self, notifier: TelegramNotifier) -> None:
		mock_response = httpx.Response(200)
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.return_value = mock_response
			await notifier._flush_batch(["msg1", "msg2", "msg3"])

			_, kwargs = mock_post.call_args
			assert kwargs["json"]["text"] == "msg1\n---\nmsg2\n---\nmsg3"

		await notifier.close()

	@pytest.mark.asyncio
	async def test_flush_batch_empty_is_noop(self, notifier: TelegramNotifier) -> None:
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			await notifier._flush_batch([])
			mock_post.assert_not_called()

	@pytest.mark.asyncio
	async def test_send_queues_message(self, notifier: TelegramNotifier) -> None:
		with patch.object(notifier, "_flush_batch", new_callable=AsyncMock):
			await notifier.send("hello")
			assert len(notifier._priority_queue) == 1
			assert notifier._priority_queue[0] == (NotificationPriority.LOW, "hello")
			assert notifier._batch_task is not None

		notifier._batch_task.cancel()
		try:
			await notifier._batch_task
		except asyncio.CancelledError:
			pass

	@pytest.mark.asyncio
	async def test_send_graceful_on_error(self, notifier: TelegramNotifier) -> None:
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.side_effect = httpx.HTTPError("network error")
			await notifier._flush_batch(["test message"])
			# Should not raise

	@pytest.mark.asyncio
	async def test_batching_collects_messages(self, notifier: TelegramNotifier) -> None:
		"""Messages sent within the batch window are combined into one flush."""
		flushed: list[list[str]] = []

		async def capture_flush(messages: list[str]) -> None:
			flushed.append(list(messages))

		with patch.object(notifier, "_flush_batch", side_effect=capture_flush), \
			patch("mission_control.notifier.BATCH_WINDOW", 0.05):
			notifier._priority_queue.append((NotificationPriority.LOW, "msg1"))
			notifier._priority_queue.append((NotificationPriority.LOW, "msg2"))
			notifier._priority_queue.append((NotificationPriority.LOW, "msg3"))
			notifier._queue_event.set()

			task = asyncio.create_task(notifier._batch_loop())
			await asyncio.sleep(0.15)
			task.cancel()
			try:
				await task
			except asyncio.CancelledError:
				pass

		assert len(flushed) >= 1
		all_msgs = [m for batch in flushed for m in batch]
		assert "msg1" in all_msgs
		assert "msg2" in all_msgs
		assert "msg3" in all_msgs

	@pytest.mark.asyncio
	async def test_close_flushes_remaining(self, notifier: TelegramNotifier) -> None:
		mock_response = httpx.Response(200)
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.return_value = mock_response
			notifier._priority_queue.append((NotificationPriority.LOW, "leftover"))
			await notifier.close()

			mock_post.assert_called_once()
			_, kwargs = mock_post.call_args
			assert kwargs["json"]["text"] == "leftover"

	@pytest.mark.asyncio
	async def test_send_mission_start(self, notifier: TelegramNotifier) -> None:
		with patch.object(notifier, "send", new_callable=AsyncMock) as mock_send:
			await notifier.send_mission_start("fix all bugs", 4)
			mock_send.assert_called_once()
			msg = mock_send.call_args[0][0]
			assert "Mission started" in msg
			assert "fix all bugs" in msg
			assert "4" in msg
			assert mock_send.call_args[1]["priority"] == NotificationPriority.HIGH

	@pytest.mark.asyncio
	async def test_send_mission_end(self, notifier: TelegramNotifier) -> None:
		with patch.object(notifier, "send", new_callable=AsyncMock) as mock_send:
			await notifier.send_mission_end(
				objective_met=True,
				merged=10,
				failed=2,
				wall_time=3600.0,
				stopped_reason="planner_completed",
				verification_passed=True,
			)
			mock_send.assert_called_once()
			msg = mock_send.call_args[0][0]
			assert "COMPLETE" in msg
			assert "10" in msg
			assert "PASS" in msg

	@pytest.mark.asyncio
	async def test_send_mission_end_stopped(self, notifier: TelegramNotifier) -> None:
		with patch.object(notifier, "send", new_callable=AsyncMock) as mock_send:
			await notifier.send_mission_end(
				objective_met=False,
				merged=5,
				failed=3,
				wall_time=1800.0,
				stopped_reason="heartbeat_stalled",
			)
			msg = mock_send.call_args[0][0]
			assert "STOPPED" in msg

	@pytest.mark.asyncio
	async def test_send_merge_conflict(self, notifier: TelegramNotifier) -> None:
		with patch.object(notifier, "send", new_callable=AsyncMock) as mock_send:
			await notifier.send_merge_conflict("fix auth module", "CONFLICT in auth.py")
			msg = mock_send.call_args[0][0]
			assert "Merge conflict" in msg
			assert "fix auth module" in msg
			assert mock_send.call_args[1]["priority"] == NotificationPriority.HIGH

	@pytest.mark.asyncio
	async def test_backpressure_drops_low_priority(self, notifier: TelegramNotifier) -> None:
		"""When queue exceeds MAX_QUEUE_SIZE, low-priority messages are dropped."""
		with patch.object(notifier, "_ensure_batch_task"):
			for i in range(MAX_QUEUE_SIZE):
				await notifier.send(f"low-{i}", priority=NotificationPriority.LOW)
			assert len(notifier._priority_queue) == MAX_QUEUE_SIZE

			for i in range(20):
				await notifier.send(f"high-{i}", priority=NotificationPriority.HIGH)

			assert len(notifier._priority_queue) <= MAX_QUEUE_SIZE

			messages = [msg for _, msg in notifier._priority_queue]
			for i in range(20):
				assert f"high-{i}" in messages

			low_count = sum(1 for p, _ in notifier._priority_queue if p == NotificationPriority.LOW)
			assert low_count < MAX_QUEUE_SIZE

	@pytest.mark.asyncio
	async def test_batching_combines_messages(self, notifier: TelegramNotifier) -> None:
		"""Verify 5s window batching behavior: messages within window are combined."""
		flushed: list[list[str]] = []

		async def capture_flush(messages: list[str]) -> None:
			flushed.append(list(messages))

		with patch.object(notifier, "_flush_batch", side_effect=capture_flush), \
			patch("mission_control.notifier.BATCH_WINDOW", 0.05):
			notifier._priority_queue.append((NotificationPriority.LOW, "batch-a"))
			notifier._priority_queue.append((NotificationPriority.HIGH, "batch-b"))
			notifier._priority_queue.append((NotificationPriority.LOW, "batch-c"))
			notifier._queue_event.set()

			task = asyncio.create_task(notifier._batch_loop())
			await asyncio.sleep(0.15)
			task.cancel()
			try:
				await task
			except asyncio.CancelledError:
				pass

		assert len(flushed) >= 1
		first_batch = flushed[0]
		assert "batch-a" in first_batch
		assert "batch-b" in first_batch
		assert "batch-c" in first_batch


class TestSplitMessage:
	def test_short_message_not_split(self) -> None:
		"""A message under max_len is returned as a single-element list."""
		msg = "short message"
		result = TelegramNotifier._split_message(msg)
		assert result == [msg]

	def test_split_at_separator_boundaries(self) -> None:
		"""A message over max_len is split at \\n---\\n boundaries."""
		chunk_a = "a" * 2000
		chunk_b = "b" * 2000
		chunk_c = "c" * 2000
		combined = f"{chunk_a}\n---\n{chunk_b}\n---\n{chunk_c}"
		assert len(combined) > TELEGRAM_MAX_LEN

		result = TelegramNotifier._split_message(combined)
		assert len(result) == 2
		assert result[0] == f"{chunk_a}\n---\n{chunk_b}"
		assert result[1] == chunk_c

	def test_oversized_single_chunk_hard_split(self) -> None:
		"""A single chunk exceeding max_len is hard-split at max_len boundaries."""
		big = "x" * (TELEGRAM_MAX_LEN * 2 + 100)
		result = TelegramNotifier._split_message(big)
		assert len(result) == 3
		assert result[0] == "x" * TELEGRAM_MAX_LEN
		assert result[1] == "x" * TELEGRAM_MAX_LEN
		assert result[2] == "x" * 100

	def test_exact_boundary_not_split(self) -> None:
		"""A message exactly at max_len is returned as a single-element list."""
		msg = "z" * TELEGRAM_MAX_LEN
		result = TelegramNotifier._split_message(msg)
		assert result == [msg]

	def test_exact_boundary_with_separator(self) -> None:
		"""Two chunks that together with separator equal max_len stay in one message."""
		sep = "\n---\n"
		half = (TELEGRAM_MAX_LEN - len(sep)) // 2
		chunk_a = "a" * half
		chunk_b = "b" * (TELEGRAM_MAX_LEN - len(sep) - half)
		combined = f"{chunk_a}{sep}{chunk_b}"
		assert len(combined) == TELEGRAM_MAX_LEN

		result = TelegramNotifier._split_message(combined)
		assert result == [combined]

	@pytest.mark.asyncio
	async def test_flush_batch_sends_multiple_posts_for_long_batch(self, notifier: TelegramNotifier) -> None:
		"""_flush_batch sends multiple posts when combined message exceeds max_len."""
		mock_response = httpx.Response(200)
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.return_value = mock_response
			# Create messages that exceed 4096 when combined
			messages = ["m" * 2000, "n" * 2000, "o" * 2000]
			await notifier._flush_batch(messages)

			assert mock_post.call_count >= 2
			sent_texts = [call.kwargs["json"]["text"] for call in mock_post.call_args_list]
			rejoined = "\n---\n".join(sent_texts)
			assert "m" * 2000 in rejoined
			assert "n" * 2000 in rejoined
			assert "o" * 2000 in rejoined

		await notifier.close()


class TestRetryLogic:
	"""Tests for transient network failure retry logic in _send_with_retry."""

	@pytest.mark.asyncio
	async def test_http_429_retries_with_backoff(self, notifier: TelegramNotifier) -> None:
		"""HTTP 429 rate limit triggers retry; succeeds on second attempt."""
		resp_429 = httpx.Response(429, headers={"Retry-After": "0.01"})
		resp_200 = httpx.Response(200)
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.side_effect = [resp_429, resp_200]
			await notifier._flush_batch(["rate limited msg"])

			assert mock_post.call_count == 2

		await notifier.close()

	@pytest.mark.asyncio
	async def test_http_429_respects_retry_after_header(self, notifier: TelegramNotifier) -> None:
		"""When 429 includes Retry-After, that value is used for the delay."""
		resp_429 = httpx.Response(429, headers={"Retry-After": "0.01"})
		resp_200 = httpx.Response(200)

		sleep_delays: list[float] = []
		original_sleep = asyncio.sleep

		async def track_sleep(delay: float) -> None:
			sleep_delays.append(delay)
			await original_sleep(min(delay, 0.01))

		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("mission_control.notifier.asyncio.sleep", side_effect=track_sleep):
			mock_post.side_effect = [resp_429, resp_200]
			await notifier._flush_batch(["msg"])

		assert len(sleep_delays) >= 1
		assert sleep_delays[0] == pytest.approx(0.01)

		await notifier.close()

	@pytest.mark.asyncio
	async def test_http_429_exhausts_retries(self, notifier: TelegramNotifier) -> None:
		"""HTTP 429 on every attempt exhausts retries and drops the batch."""
		resp_429 = httpx.Response(429, headers={"Retry-After": "0.01"})
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("mission_control.notifier.asyncio.sleep", new_callable=AsyncMock):
			mock_post.return_value = resp_429
			await notifier._flush_batch(["doomed msg"])

			assert mock_post.call_count == MAX_RETRIES

		await notifier.close()

	@pytest.mark.asyncio
	async def test_http_500_retries_then_succeeds(self, notifier: TelegramNotifier) -> None:
		"""HTTP 500 triggers retry; succeeds on second attempt."""
		resp_500 = httpx.Response(500)
		resp_200 = httpx.Response(200)
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("mission_control.notifier.asyncio.sleep", new_callable=AsyncMock):
			mock_post.side_effect = [resp_500, resp_200]
			await notifier._flush_batch(["server error msg"])

			assert mock_post.call_count == 2

		await notifier.close()

	@pytest.mark.asyncio
	async def test_http_5xx_exhausts_retries(self, notifier: TelegramNotifier) -> None:
		"""HTTP 5xx on every attempt exhausts retries and drops the batch."""
		resp_502 = httpx.Response(502)
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("mission_control.notifier.asyncio.sleep", new_callable=AsyncMock):
			mock_post.return_value = resp_502
			await notifier._flush_batch(["repeated 502 msg"])

			assert mock_post.call_count == MAX_RETRIES

		await notifier.close()

	@pytest.mark.asyncio
	async def test_connection_timeout_retries_then_succeeds(self, notifier: TelegramNotifier) -> None:
		"""ConnectTimeout triggers retry; succeeds on second attempt without hanging."""
		resp_200 = httpx.Response(200)
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("mission_control.notifier.asyncio.sleep", new_callable=AsyncMock):
			mock_post.side_effect = [httpx.ConnectTimeout("timed out"), resp_200]

			async def run_with_timeout() -> None:
				await notifier._flush_batch(["timeout msg"])

			await asyncio.wait_for(run_with_timeout(), timeout=5.0)

			assert mock_post.call_count == 2

		await notifier.close()

	@pytest.mark.asyncio
	async def test_connection_timeout_exhausts_retries(self, notifier: TelegramNotifier) -> None:
		"""ConnectTimeout on every attempt exhausts retries without hanging."""
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("mission_control.notifier.asyncio.sleep", new_callable=AsyncMock):
			mock_post.side_effect = httpx.ConnectTimeout("timed out")

			async def run_with_timeout() -> None:
				await notifier._flush_batch(["doomed timeout msg"])

			await asyncio.wait_for(run_with_timeout(), timeout=5.0)

			assert mock_post.call_count == MAX_RETRIES

		await notifier.close()

	@pytest.mark.asyncio
	async def test_exponential_backoff_delays(self, notifier: TelegramNotifier) -> None:
		"""Verify retry delays follow exponential backoff pattern (1s, 2s, 4s)."""
		resp_500 = httpx.Response(500)
		sleep_delays: list[float] = []

		async def capture_sleep(delay: float) -> None:
			sleep_delays.append(delay)

		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("mission_control.notifier.asyncio.sleep", side_effect=capture_sleep):
			mock_post.return_value = resp_500
			await notifier._flush_batch(["backoff msg"])

		# MAX_RETRIES=3 means attempts 0,1,2; sleeps between 0->1 and 1->2
		assert len(sleep_delays) == MAX_RETRIES - 1
		for i, delay in enumerate(sleep_delays):
			assert delay == pytest.approx(RETRY_BASE_DELAY * (2 ** i))

		await notifier.close()

	@pytest.mark.asyncio
	async def test_queue_overflow_drops_low_priority(self, notifier: TelegramNotifier) -> None:
		"""Rapidly enqueuing > MAX_QUEUE_SIZE messages drops low-priority ones."""
		with patch.object(notifier, "_ensure_batch_task"):
			# Fill queue with low-priority messages
			for i in range(MAX_QUEUE_SIZE + 50):
				await notifier.send(f"overflow-low-{i}", priority=NotificationPriority.LOW)

			assert len(notifier._priority_queue) <= MAX_QUEUE_SIZE

			# Now add high-priority messages on top
			for i in range(30):
				await notifier.send(f"overflow-high-{i}", priority=NotificationPriority.HIGH)

			assert len(notifier._priority_queue) <= MAX_QUEUE_SIZE

			# All high-priority messages are preserved
			messages = [msg for _, msg in notifier._priority_queue]
			for i in range(30):
				assert f"overflow-high-{i}" in messages

			# Low-priority messages were dropped to make room
			high_count = sum(1 for p, _ in notifier._priority_queue if p == NotificationPriority.HIGH)
			low_count = sum(1 for p, _ in notifier._priority_queue if p == NotificationPriority.LOW)
			assert high_count == 30
			assert low_count == MAX_QUEUE_SIZE - 30

	@pytest.mark.asyncio
	async def test_4xx_non_429_not_retried(self, notifier: TelegramNotifier) -> None:
		"""Non-retryable 4xx errors (e.g. 400) are not retried."""
		resp_400 = httpx.Response(400)
		with patch("mission_control.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.return_value = resp_400
			await notifier._flush_batch(["bad request msg"])

			assert mock_post.call_count == 1

		await notifier.close()
