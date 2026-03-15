"""Tests for TelegramNotifier."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from autodev.notifier import (
	MAX_QUEUE_SIZE,
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
		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
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
		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.return_value = mock_response
			await notifier._flush_batch(["msg1", "msg2", "msg3"])

			_, kwargs = mock_post.call_args
			assert kwargs["json"]["text"] == "msg1\n---\nmsg2\n---\nmsg3"

		await notifier.close()

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
		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
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
			patch("autodev.notifier.BATCH_WINDOW", 0.05):
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
		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
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

	@pytest.mark.asyncio
	async def test_flush_batch_sends_multiple_posts_for_long_batch(self, notifier: TelegramNotifier) -> None:
		"""_flush_batch sends multiple posts when combined message exceeds max_len."""
		mock_response = httpx.Response(200)
		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
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


class TestEpochSummary:
	@pytest.mark.asyncio
	async def test_send_epoch_summary_formats_table(self, notifier: TelegramNotifier) -> None:
		"""send_epoch_summary formats a MarkdownV2 table with all metrics."""
		with patch.object(notifier, "_send_message", new_callable=AsyncMock) as mock_send:
			await notifier.send_epoch_summary(
				epoch_num=5,
				dispatched=10,
				merged=8,
				failed=2,
				cost_this_epoch=1.23,
				cumulative_cost=5.67,
				conflict_rate=0.15,
			)
			mock_send.assert_called_once()
			msg = mock_send.call_args[0][0]
			assert "Epoch 5 Summary" in msg
			assert "```" in msg
			assert "Dispatched" in msg
			assert "10" in msg
			assert "Merged" in msg
			assert "Failed" in msg
			assert "Conflict Rate" in msg
			assert "15.0%" in msg
			assert "$1.23" in msg
			assert "$5.67" in msg

	@pytest.mark.asyncio
	async def test_send_epoch_summary_calls_send_message(self, notifier: TelegramNotifier) -> None:
		"""send_epoch_summary delegates to _send_message."""
		with patch.object(notifier, "_send_message", new_callable=AsyncMock) as mock_send:
			await notifier.send_epoch_summary(
				epoch_num=1, dispatched=5, merged=3, failed=1,
				cost_this_epoch=0.50, cumulative_cost=2.00, conflict_rate=0.0,
			)
			mock_send.assert_called_once()
			assert mock_send.call_args[0][0].startswith("*Epoch 1 Summary*")


class TestCostAlert:
	@pytest.mark.asyncio
	async def test_send_cost_alert_includes_percentage(self, notifier: TelegramNotifier) -> None:
		"""send_cost_alert shows projected cost as percentage of budget."""
		with patch.object(notifier, "_send_message", new_callable=AsyncMock) as mock_send:
			await notifier.send_cost_alert(
				current_cost=5.00,
				budget_limit=10.00,
				projected_cost=8.00,
			)
			mock_send.assert_called_once()
			msg = mock_send.call_args[0][0]
			assert "Cost Alert" in msg
			assert "80%" in msg

	@pytest.mark.asyncio
	async def test_send_cost_alert_includes_overage(self, notifier: TelegramNotifier) -> None:
		"""send_cost_alert shows projected overage when over budget."""
		with patch.object(notifier, "_send_message", new_callable=AsyncMock) as mock_send:
			await notifier.send_cost_alert(
				current_cost=6.00,
				budget_limit=10.00,
				projected_cost=12.00,
			)
			mock_send.assert_called_once()
			msg = mock_send.call_args[0][0]
			assert "120%" in msg
			assert "overage" in msg.lower()

	@pytest.mark.asyncio
	async def test_send_cost_alert_no_overage_when_under_budget(self, notifier: TelegramNotifier) -> None:
		"""send_cost_alert omits overage line when projected is under budget."""
		with patch.object(notifier, "_send_message", new_callable=AsyncMock) as mock_send:
			await notifier.send_cost_alert(
				current_cost=3.00,
				budget_limit=10.00,
				projected_cost=6.00,
			)
			msg = mock_send.call_args[0][0]
			assert "overage" not in msg.lower()


class TestBackwardCompatibility:
	"""Verify existing method signatures and behavior are unchanged."""

	@pytest.mark.asyncio
	async def test_send_still_accepts_priority(self, notifier: TelegramNotifier) -> None:
		with patch.object(notifier, "_ensure_batch_task"):
			await notifier.send("hello", priority=NotificationPriority.HIGH)
			assert notifier._priority_queue[0] == (NotificationPriority.HIGH, "hello")

	@pytest.mark.asyncio
	async def test_send_mission_start_unchanged(self, notifier: TelegramNotifier) -> None:
		with patch.object(notifier, "send", new_callable=AsyncMock) as mock_send:
			await notifier.send_mission_start("test objective", 3)
			mock_send.assert_called_once()
			msg = mock_send.call_args[0][0]
			assert "Mission started" in msg
			assert "test objective" in msg

	@pytest.mark.asyncio
	async def test_send_mission_end_unchanged(self, notifier: TelegramNotifier) -> None:
		with patch.object(notifier, "send", new_callable=AsyncMock) as mock_send:
			await notifier.send_mission_end(
				objective_met=False, merged=5, failed=3,
				wall_time=1800.0, stopped_reason="budget",
			)
			mock_send.assert_called_once()
			msg = mock_send.call_args[0][0]
			assert "STOPPED" in msg


class TestRequestApproval:
	@pytest.mark.asyncio
	async def test_approved_returns_true(self, notifier: TelegramNotifier) -> None:
		"""request_approval returns True when user replies 'approve'."""
		mock_post_resp = httpx.Response(200)
		approve_update = {
			"result": [{
				"update_id": 1,
				"message": {
					"text": "approve",
					"chat": {"id": 12345},
				},
			}],
		}
		mock_get_resp = httpx.Response(200, json=approve_update)

		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("autodev.notifier.httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
			mock_post.return_value = mock_post_resp
			mock_get.return_value = mock_get_resp

			result = await notifier.request_approval("Deploy to prod?", timeout_seconds=5.0, poll_interval=0.01)
			assert result is True
			mock_post.assert_called_once()

		await notifier.close()

	@pytest.mark.asyncio
	async def test_rejected_returns_false(self, notifier: TelegramNotifier) -> None:
		"""request_approval returns False when user replies 'reject'."""
		mock_post_resp = httpx.Response(200)
		reject_update = {
			"result": [{
				"update_id": 1,
				"message": {
					"text": "no",
					"chat": {"id": 12345},
				},
			}],
		}
		mock_get_resp = httpx.Response(200, json=reject_update)

		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("autodev.notifier.httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
			mock_post.return_value = mock_post_resp
			mock_get.return_value = mock_get_resp

			result = await notifier.request_approval("Deploy to prod?", timeout_seconds=5.0, poll_interval=0.01)
			assert result is False

		await notifier.close()

	@pytest.mark.asyncio
	async def test_timeout_returns_false(self, notifier: TelegramNotifier) -> None:
		"""request_approval returns False when no response before timeout."""
		mock_post_resp = httpx.Response(200)
		empty_updates = {"result": []}
		mock_get_resp = httpx.Response(200, json=empty_updates)

		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("autodev.notifier.httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
			mock_post.return_value = mock_post_resp
			mock_get.return_value = mock_get_resp

			result = await notifier.request_approval("Deploy?", timeout_seconds=0.05, poll_interval=0.01)
			assert result is False

		await notifier.close()

	@pytest.mark.asyncio
	async def test_ignores_other_chat_ids(self, notifier: TelegramNotifier) -> None:
		"""request_approval ignores messages from other chats."""
		mock_post_resp = httpx.Response(200)
		wrong_chat = {
			"result": [{
				"update_id": 1,
				"message": {
					"text": "approve",
					"chat": {"id": 99999},
				},
			}],
		}
		mock_get_resp = httpx.Response(200, json=wrong_chat)

		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("autodev.notifier.httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
			mock_post.return_value = mock_post_resp
			mock_get.return_value = mock_get_resp

			result = await notifier.request_approval("Deploy?", timeout_seconds=0.05, poll_interval=0.01)
			assert result is False

		await notifier.close()

	@pytest.mark.asyncio
	async def test_poll_error_continues(self, notifier: TelegramNotifier) -> None:
		"""request_approval continues polling after transient errors."""
		mock_post_resp = httpx.Response(200)
		approve_update = {
			"result": [{
				"update_id": 1,
				"message": {
					"text": "y",
					"chat": {"id": 12345},
				},
			}],
		}
		mock_get_resp = httpx.Response(200, json=approve_update)

		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("autodev.notifier.httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
			mock_post.return_value = mock_post_resp
			mock_get.side_effect = [httpx.ConnectError("net error"), mock_get_resp]

			result = await notifier.request_approval("Deploy?", timeout_seconds=5.0, poll_interval=0.01)
			assert result is True

		await notifier.close()


class TestRetryLogic:
	"""Tests for transient network failure retry logic in _send_with_retry."""

	@pytest.mark.asyncio
	async def test_http_429_retries_with_backoff(self, notifier: TelegramNotifier) -> None:
		"""HTTP 429 rate limit triggers retry; succeeds on second attempt."""
		resp_429 = httpx.Response(429, headers={"Retry-After": "0.01"})
		resp_200 = httpx.Response(200)
		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.side_effect = [resp_429, resp_200]
			await notifier._flush_batch(["rate limited msg"])

			assert mock_post.call_count == 2

		await notifier.close()

	@pytest.mark.asyncio
	async def test_http_500_retries_then_succeeds(self, notifier: TelegramNotifier) -> None:
		"""HTTP 500 triggers retry; succeeds on second attempt."""
		resp_500 = httpx.Response(500)
		resp_200 = httpx.Response(200)
		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("autodev.notifier.asyncio.sleep", new_callable=AsyncMock):
			mock_post.side_effect = [resp_500, resp_200]
			await notifier._flush_batch(["server error msg"])

			assert mock_post.call_count == 2

		await notifier.close()

	@pytest.mark.asyncio
	async def test_connection_timeout_retries_then_succeeds(self, notifier: TelegramNotifier) -> None:
		"""ConnectTimeout triggers retry; succeeds on second attempt without hanging."""
		resp_200 = httpx.Response(200)
		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
			patch("autodev.notifier.asyncio.sleep", new_callable=AsyncMock):
			mock_post.side_effect = [httpx.ConnectTimeout("timed out"), resp_200]

			async def run_with_timeout() -> None:
				await notifier._flush_batch(["timeout msg"])

			await asyncio.wait_for(run_with_timeout(), timeout=5.0)

			assert mock_post.call_count == 2

		await notifier.close()



class TestAuthNotifications:
	"""Tests for auth-related notification methods."""

	@pytest.mark.asyncio
	async def test_send_auth_request(self, notifier: TelegramNotifier) -> None:
		"""send_auth_request sends a HIGH priority message."""
		with patch.object(notifier, "send", new_callable=AsyncMock) as mock_send:
			await notifier.send_auth_request("github", "CLI access", "https://github.com/login")
			mock_send.assert_called_once()
			msg = mock_send.call_args[0][0]
			assert "Auth Required" in msg
			assert "github" in msg
			assert "CLI access" in msg
			assert "https://github.com/login" in msg
			assert mock_send.call_args[1]["priority"] == NotificationPriority.HIGH

	@pytest.mark.asyncio
	async def test_send_auth_help_uploads_photo(self, notifier: TelegramNotifier, tmp_path) -> None:
		"""send_auth_help sends screenshot via sendPhoto."""
		screenshot = tmp_path / "screenshot.png"
		screenshot.write_bytes(b"fake-png-data")
		mock_response = httpx.Response(200)
		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.return_value = mock_response
			await notifier.send_auth_help("google", str(screenshot))
			mock_post.assert_called_once()
			call_args = mock_post.call_args
			assert "sendPhoto" in call_args[0][0]
			assert call_args[1]["data"]["chat_id"] == "12345"
			assert "google" in call_args[1]["data"]["caption"]
			assert "photo" in call_args[1]["files"]
		await notifier.close()

	@pytest.mark.asyncio
	async def test_send_auth_help_handles_error(self, notifier: TelegramNotifier, tmp_path) -> None:
		"""send_auth_help handles errors gracefully."""
		screenshot = tmp_path / "screenshot.png"
		screenshot.write_bytes(b"fake-png-data")
		with patch("autodev.notifier.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
			mock_post.side_effect = httpx.HTTPError("network error")
			await notifier.send_auth_help("google", str(screenshot))

	@pytest.mark.asyncio
	async def test_send_auth_help_missing_file(self, notifier: TelegramNotifier) -> None:
		"""send_auth_help handles missing file gracefully."""
		await notifier.send_auth_help("google", "/nonexistent/screenshot.png")

	@pytest.mark.asyncio
	async def test_send_signup_request_approved(self, notifier: TelegramNotifier) -> None:
		"""send_signup_request returns True when approved."""
		with patch.object(notifier, "request_approval", new_callable=AsyncMock) as mock_approval:
			mock_approval.return_value = True
			result = await notifier.send_signup_request("newservice", "API integration")
			assert result is True
			mock_approval.assert_called_once()
			msg = mock_approval.call_args[0][0]
			assert "newservice" in msg
			assert "API integration" in msg
			assert "create an account" in msg

	@pytest.mark.asyncio
	async def test_send_signup_request_rejected(self, notifier: TelegramNotifier) -> None:
		"""send_signup_request returns False when rejected."""
		with patch.object(notifier, "request_approval", new_callable=AsyncMock) as mock_approval:
			mock_approval.return_value = False
			result = await notifier.send_signup_request("newservice", "API integration")
			assert result is False

	@pytest.mark.asyncio
	async def test_send_spend_request_approved(self, notifier: TelegramNotifier) -> None:
		"""send_spend_request returns True when approved."""
		with patch.object(notifier, "request_approval", new_callable=AsyncMock) as mock_approval:
			mock_approval.return_value = True
			result = await notifier.send_spend_request("openai", "$20/month", "GPT-4 API access")
			assert result is True
			mock_approval.assert_called_once()
			msg = mock_approval.call_args[0][0]
			assert "openai" in msg
			assert "$20/month" in msg
			assert "GPT-4 API access" in msg

	@pytest.mark.asyncio
	async def test_send_spend_request_rejected(self, notifier: TelegramNotifier) -> None:
		"""send_spend_request returns False when rejected."""
		with patch.object(notifier, "request_approval", new_callable=AsyncMock) as mock_approval:
			mock_approval.return_value = False
			result = await notifier.send_spend_request("openai", "$20/month", "GPT-4 API access")
			assert result is False
