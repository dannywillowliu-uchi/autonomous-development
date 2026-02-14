"""Tests for the heartbeat progress monitor."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from mission_control.heartbeat import Heartbeat


@pytest.fixture
def notifier() -> AsyncMock:
	return AsyncMock()


class TestHeartbeat:
	@pytest.mark.asyncio
	async def test_no_check_before_interval(self) -> None:
		hb = Heartbeat(interval=300, idle_threshold=3)
		# Force a first check to set baseline
		hb._last_check_time = time.monotonic()
		result = await hb.check(total_merged=0, total_failed=0)
		assert result == ""

	@pytest.mark.asyncio
	async def test_progress_resets_idle(self, notifier: AsyncMock) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3, notifier=notifier)
		hb._last_merged_count = 5
		hb._consecutive_idle = 2
		result = await hb.check(total_merged=8, total_failed=0)
		assert result == ""
		assert hb.consecutive_idle == 0
		notifier.send.assert_called_once()
		assert "3 merged" in notifier.send.call_args[0][0]

	@pytest.mark.asyncio
	async def test_idle_increments(self, notifier: AsyncMock) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3, notifier=notifier)
		hb._last_merged_count = 5
		result = await hb.check(total_merged=5, total_failed=1)
		assert result == ""
		assert hb.consecutive_idle == 1

	@pytest.mark.asyncio
	async def test_stall_after_threshold(self, notifier: AsyncMock) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3, notifier=notifier)
		hb._last_merged_count = 5
		hb._consecutive_idle = 2  # Already 2 idle checks
		result = await hb.check(total_merged=5, total_failed=0)
		assert result == "heartbeat_stalled"
		assert hb.consecutive_idle == 3

	@pytest.mark.asyncio
	async def test_no_stall_with_merges(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=1)
		hb._last_merged_count = 0
		result = await hb.check(total_merged=1, total_failed=0)
		assert result == ""

	@pytest.mark.asyncio
	async def test_first_check_baseline(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3)
		# Both 0 -- first check just sets baseline
		result = await hb.check(total_merged=0, total_failed=0)
		assert result == ""
		assert hb.consecutive_idle == 0

	@pytest.mark.asyncio
	async def test_no_notifier(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3, notifier=None)
		hb._last_merged_count = 5
		result = await hb.check(total_merged=5, total_failed=0)
		assert result == ""
		assert hb.consecutive_idle == 1

	@pytest.mark.asyncio
	async def test_custom_idle_threshold(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=5)
		hb._last_merged_count = 5
		for i in range(4):
			result = await hb.check(total_merged=5, total_failed=0)
			assert result == ""
		result = await hb.check(total_merged=5, total_failed=0)
		assert result == "heartbeat_stalled"
