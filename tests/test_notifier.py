"""Tests for TelegramNotifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mission_control.notifier import TelegramNotifier


@pytest.fixture
def notifier() -> TelegramNotifier:
	return TelegramNotifier(bot_token="test-token", chat_id="12345")


class TestTelegramNotifier:
	@pytest.mark.asyncio
	async def test_send_calls_urlopen(self, notifier: TelegramNotifier) -> None:
		with patch("mission_control.notifier.urllib.request.urlopen") as mock_urlopen:
			mock_resp = MagicMock()
			mock_resp.__enter__ = MagicMock(return_value=mock_resp)
			mock_resp.__exit__ = MagicMock(return_value=False)
			mock_urlopen.return_value = mock_resp

			await notifier.send("test message")

			mock_urlopen.assert_called_once()
			req = mock_urlopen.call_args[0][0]
			assert "test-token" in req.full_url
			body = json.loads(req.data)
			assert body["chat_id"] == "12345"
			assert body["text"] == "test message"

	@pytest.mark.asyncio
	async def test_send_graceful_on_error(self, notifier: TelegramNotifier) -> None:
		with patch("mission_control.notifier.urllib.request.urlopen") as mock_urlopen:
			mock_urlopen.side_effect = Exception("network error")
			# Should not raise
			await notifier.send("test message")

	@pytest.mark.asyncio
	async def test_send_mission_start(self, notifier: TelegramNotifier) -> None:
		with patch.object(notifier, "send") as mock_send:
			mock_send.return_value = None
			await notifier.send_mission_start("fix all bugs", 4)
			mock_send.assert_called_once()
			msg = mock_send.call_args[0][0]
			assert "Mission started" in msg
			assert "fix all bugs" in msg
			assert "4" in msg

	@pytest.mark.asyncio
	async def test_send_mission_end(self, notifier: TelegramNotifier) -> None:
		with patch.object(notifier, "send") as mock_send:
			mock_send.return_value = None
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
		with patch.object(notifier, "send") as mock_send:
			mock_send.return_value = None
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
		with patch.object(notifier, "send") as mock_send:
			mock_send.return_value = None
			await notifier.send_merge_conflict("fix auth module", "CONFLICT in auth.py")
			msg = mock_send.call_args[0][0]
			assert "Merge conflict" in msg
			assert "fix auth module" in msg
