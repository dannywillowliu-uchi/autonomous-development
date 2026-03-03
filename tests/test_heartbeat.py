"""Tests for the heartbeat progress monitor."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.heartbeat import Heartbeat


@pytest.fixture
def notifier() -> AsyncMock:
	return AsyncMock()


def _make_mock_db(running_rows: list[tuple] | None = None) -> MagicMock:
	"""Create a mock DB with conn.execute returning rows for running units.

	Each row is a tuple of (id, title, started_at).
	"""
	db = MagicMock()
	if running_rows is None:
		running_rows = []

	cursor = MagicMock()
	cursor.fetchall.return_value = running_rows
	db.conn.execute.return_value = cursor
	return db


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
		hb._baseline_set = True
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
		hb._baseline_set = True
		hb._last_merged_count = 5
		result = await hb.check(total_merged=5, total_failed=1)
		assert result == ""
		assert hb.consecutive_idle == 1

	@pytest.mark.asyncio
	async def test_stall_after_threshold(self, notifier: AsyncMock) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3, notifier=notifier, enable_recovery=False)
		hb._baseline_set = True
		hb._last_merged_count = 5
		hb._consecutive_idle = 2  # Already 2 idle checks
		result = await hb.check(total_merged=5, total_failed=0)
		assert result == "heartbeat_stalled"
		assert hb.consecutive_idle == 3

	@pytest.mark.asyncio
	async def test_no_stall_with_merges(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=1)
		hb._baseline_set = True
		hb._last_merged_count = 0
		result = await hb.check(total_merged=1, total_failed=0)
		assert result == ""

	@pytest.mark.asyncio
	async def test_first_check_baseline(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3)
		# First check just sets baseline
		result = await hb.check(total_merged=0, total_failed=0)
		assert result == ""
		assert hb.consecutive_idle == 0
		assert hb._baseline_set is True
		assert hb._last_merged_count == 0

	@pytest.mark.asyncio
	async def test_zero_merge_stall(self) -> None:
		"""Heartbeat detects stuck missions when total_merged stays at 0."""
		hb = Heartbeat(interval=0, idle_threshold=3, enable_recovery=False)
		# First check sets baseline
		result = await hb.check(total_merged=0, total_failed=0)
		assert result == ""
		assert hb.consecutive_idle == 0

		# Subsequent checks with no merges should increment idle
		result = await hb.check(total_merged=0, total_failed=0)
		assert result == ""
		assert hb.consecutive_idle == 1

		result = await hb.check(total_merged=0, total_failed=0)
		assert result == ""
		assert hb.consecutive_idle == 2

		# Third idle check hits threshold
		result = await hb.check(total_merged=0, total_failed=0)
		assert result == "heartbeat_stalled"
		assert hb.consecutive_idle == 3



class TestHeartbeatRecovery:
	@pytest.mark.asyncio
	async def test_recovery_called_on_stall(self, notifier: AsyncMock) -> None:
		"""Verify recover() is called when idle threshold is hit with recovery enabled."""
		mock_db = _make_mock_db([
			("unit-abc123", "Fix the widget", "2026-01-01T00:00:00Z"),
		])
		hb = Heartbeat(
			interval=0, idle_threshold=3, notifier=notifier,
			db=mock_db, enable_recovery=True,
		)
		hb._baseline_set = True
		hb._last_merged_count = 5
		hb._consecutive_idle = 2

		result = await hb.check(total_merged=5, total_failed=0)

		assert result == "heartbeat_recovered"
		# DB was queried for running units
		mock_db.conn.execute.assert_called_with(
			"SELECT id, title, started_at FROM work_units WHERE status = 'running'"
		)
		# Notifier was called with stuck unit info (idle warning + recovery diagnostic)
		notify_calls = notifier.send.call_args_list
		assert len(notify_calls) >= 2
		recovery_msg = notify_calls[-1][0][0]
		assert "stuck units" in recovery_msg
		assert "unit-abc123" in recovery_msg

	@pytest.mark.asyncio
	async def test_recovery_disabled(self, notifier: AsyncMock) -> None:
		"""Verify recover() is NOT called when enable_recovery=False."""
		mock_db = _make_mock_db([
			("unit-xyz789", "Deploy service", "2026-01-01T00:00:00Z"),
		])
		hb = Heartbeat(
			interval=0, idle_threshold=3, notifier=notifier,
			db=mock_db, enable_recovery=False,
		)
		hb._baseline_set = True
		hb._last_merged_count = 5
		hb._consecutive_idle = 2

		result = await hb.check(total_merged=5, total_failed=0)

		assert result == "heartbeat_stalled"
		# DB should NOT have been queried for recovery
		mock_db.conn.execute.assert_not_called()

	@pytest.mark.asyncio
	async def test_recovery_sends_diagnostic(self, notifier: AsyncMock) -> None:
		"""Verify Telegram notification includes stuck unit info."""
		mock_db = _make_mock_db([
			("unit-aaa111", "Refactor auth module", "2026-01-01T00:00:00Z"),
			("unit-bbb222", "Add caching layer", "2026-01-01T12:00:00Z"),
		])
		hb = Heartbeat(
			interval=0, idle_threshold=1, notifier=notifier,
			db=mock_db, enable_recovery=True,
		)
		hb._baseline_set = True
		hb._last_merged_count = 3

		result = await hb.check(total_merged=3, total_failed=0)

		assert result == "heartbeat_recovered"

		# Find the recovery diagnostic message (last send call)
		notify_calls = notifier.send.call_args_list
		recovery_msg = notify_calls[-1][0][0]
		assert "Heartbeat recovery" in recovery_msg
		assert "unit-aaa111" in recovery_msg
		assert "Refactor auth module" in recovery_msg
		assert "unit-bbb222" in recovery_msg
		assert "Add caching layer" in recovery_msg

	@pytest.mark.asyncio
	async def test_recovery_returns_stuck_ids(self) -> None:
		"""Verify recover() returns the list of stuck unit IDs."""
		mock_db = _make_mock_db([
			("unit-111", "Task A", None),
			("unit-222", "Task B", "2026-01-01T00:00:00Z"),
		])
		hb = Heartbeat(
			interval=0, idle_threshold=1,
			db=mock_db, enable_recovery=True,
		)

		stuck_ids = await hb.recover()

		assert stuck_ids == ["unit-111", "unit-222"]


class TestCostMonitoring:
	def test_record_cost_stores_entries(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3)
		hb.record_cost(0.50)
		hb.record_cost(0.25)
		assert len(hb._cost_entries) == 2

	def test_get_cost_rate_zero_with_no_entries(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3)
		assert hb.get_cost_rate() == 0.0

	def test_get_cost_rate_zero_with_single_entry(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3)
		hb.record_cost(1.0)
		assert hb.get_cost_rate() == 0.0

	def test_get_cost_rate_calculation(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3, cost_window_seconds=600.0)
		base = time.monotonic()
		# Simulate entries spread over 60 seconds: $0.50 + $0.50 = $1.00 over 1 min
		hb._cost_entries.append((base - 60, 0.50))
		hb._cost_entries.append((base - 30, 0.50))
		with patch("mission_control.heartbeat.time.monotonic", return_value=base):
			rate = hb.get_cost_rate()
		# $1.00 total over 60s span = $1.00/min
		assert abs(rate - 1.0) < 0.01

	def test_cost_rate_window_rollover(self) -> None:
		"""Old entries outside the window are pruned."""
		hb = Heartbeat(interval=0, idle_threshold=3, cost_window_seconds=60.0)
		base = time.monotonic()
		# Entry well outside the 60s window
		hb._cost_entries.append((base - 200, 5.00))
		# Entries inside the window
		hb._cost_entries.append((base - 30, 0.30))
		hb._cost_entries.append((base - 10, 0.20))
		with patch("mission_control.heartbeat.time.monotonic", return_value=base):
			rate = hb.get_cost_rate()
		# After pruning: $0.30 + $0.20 = $0.50 over 30s span = $1.00/min
		assert abs(rate - 1.0) < 0.05

	def test_project_budget_exhaustion_returns_none_at_zero_rate(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3)
		assert hb.project_budget_exhaustion(10.0) is None

	def test_project_budget_exhaustion_calculation(self) -> None:
		hb = Heartbeat(interval=0, idle_threshold=3, cost_window_seconds=600.0)
		base = time.monotonic()
		# $2.00 over 120s = $1.00/min
		hb._cost_entries.append((base - 120, 1.00))
		hb._cost_entries.append((base - 60, 1.00))
		with patch("mission_control.heartbeat.time.monotonic", return_value=base):
			minutes = hb.project_budget_exhaustion(5.0)
		assert minutes is not None
		assert abs(minutes - 5.0) < 0.1

	@pytest.mark.asyncio
	async def test_check_cost_health_alerts_on_high_rate(self, notifier: AsyncMock) -> None:
		hb = Heartbeat(
			interval=0, idle_threshold=3, notifier=notifier,
			cost_rate_threshold=0.50,
		)
		base = time.monotonic()
		# $2.00 over 60s = $2.00/min, well above 0.50 threshold
		hb._cost_entries.append((base - 60, 1.00))
		hb._cost_entries.append((base - 30, 1.00))
		with patch("mission_control.heartbeat.time.monotonic", return_value=base):
			await hb.check_cost_health(remaining_budget=100.0)
		notifier.send.assert_called_once()
		msg = notifier.send.call_args[0][0]
		assert "Cost alert" in msg
		assert "exceeds threshold" in msg

	@pytest.mark.asyncio
	async def test_check_cost_health_alerts_on_exhaustion(self, notifier: AsyncMock) -> None:
		hb = Heartbeat(
			interval=0, idle_threshold=3, notifier=notifier,
			cost_rate_threshold=100.0,  # High threshold so rate alert doesn't fire
			exhaustion_warning_minutes=10.0,
		)
		base = time.monotonic()
		# $2.00 over 60s = $2.00/min; remaining $5 => ~2.5 min left (< 10 min warning)
		hb._cost_entries.append((base - 60, 1.00))
		hb._cost_entries.append((base - 30, 1.00))
		with patch("mission_control.heartbeat.time.monotonic", return_value=base):
			await hb.check_cost_health(remaining_budget=5.0)
		notifier.send.assert_called_once()
		msg = notifier.send.call_args[0][0]
		assert "Budget exhaustion" in msg

	@pytest.mark.asyncio
	async def test_check_cost_health_no_alert_when_healthy(self, notifier: AsyncMock) -> None:
		hb = Heartbeat(
			interval=0, idle_threshold=3, notifier=notifier,
			cost_rate_threshold=10.0,
			exhaustion_warning_minutes=10.0,
		)
		base = time.monotonic()
		# $0.10 over 60s = $0.10/min, well below threshold; budget lasts 1000 min
		hb._cost_entries.append((base - 60, 0.05))
		hb._cost_entries.append((base - 30, 0.05))
		with patch("mission_control.heartbeat.time.monotonic", return_value=base):
			await hb.check_cost_health(remaining_budget=100.0)
		notifier.send.assert_not_called()

	@pytest.mark.asyncio
	async def test_check_wires_cost_health(self, notifier: AsyncMock) -> None:
		"""Verify check() calls check_cost_health with remaining_budget."""
		hb = Heartbeat(
			interval=0, idle_threshold=3, notifier=notifier,
			cost_rate_threshold=0.01,
		)
		base = time.monotonic()
		hb._cost_entries.append((base - 60, 1.00))
		hb._cost_entries.append((base - 30, 1.00))
		with patch("mission_control.heartbeat.time.monotonic", return_value=base):
			await hb.check(total_merged=0, total_failed=0, remaining_budget=50.0)
		# Should have sent a cost alert (first call) since rate exceeds 0.01
		cost_alert_sent = any(
			"Cost alert" in str(call) for call in notifier.send.call_args_list
		)
		assert cost_alert_sent
