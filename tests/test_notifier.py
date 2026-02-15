"""Tests for TelegramNotifier."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from mission_control.notifier import MAX_QUEUE_SIZE, NotificationPriority, TelegramNotifier


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
