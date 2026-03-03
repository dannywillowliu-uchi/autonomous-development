"""Time-based progress monitor inspired by OpenClaw's heartbeat pattern.

Checks at regular intervals whether the mission is making progress
(new merges since last check). Sends Telegram alerts and can trigger
a stall stop after consecutive idle heartbeats.

Includes cost-aware monitoring: tracks spending rate over a sliding
window and projects budget exhaustion time.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mission_control.db import Database
	from mission_control.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class Heartbeat:
	"""Time-based progress monitor with cost awareness.

	Every `interval` seconds, checks if the mission is making progress
	by comparing recent merge activity. If no progress for `idle_threshold`
	consecutive checks, returns a stall signal.

	Tracks spending via a sliding window and alerts when the cost rate
	exceeds a threshold or projected budget exhaustion is imminent.
	"""

	def __init__(
		self,
		interval: int = 300,
		idle_threshold: int = 3,
		notifier: TelegramNotifier | None = None,
		db: Database | None = None,
		enable_recovery: bool = True,
		cost_window_seconds: float = 600.0,
		cost_rate_threshold: float = 1.0,
		exhaustion_warning_minutes: float = 10.0,
	) -> None:
		self._interval = interval
		self._idle_threshold = idle_threshold
		self._notifier = notifier
		self._db = db
		self._enable_recovery = enable_recovery
		self._last_check_time: float = 0.0
		self._last_merged_count: int = 0
		self._consecutive_idle: int = 0
		self._baseline_set: bool = False
		# Cost tracking
		self._cost_window_seconds = cost_window_seconds
		self._cost_rate_threshold = cost_rate_threshold
		self._exhaustion_warning_minutes = exhaustion_warning_minutes
		self._cost_entries: deque[tuple[float, float]] = deque()

	def _prune_cost_window(self, now: float | None = None) -> None:
		"""Remove cost entries older than the sliding window."""
		if now is None:
			now = time.monotonic()
		cutoff = now - self._cost_window_seconds
		while self._cost_entries and self._cost_entries[0][0] < cutoff:
			self._cost_entries.popleft()

	def record_cost(self, amount_usd: float) -> None:
		"""Record a cost event with the current timestamp."""
		now = time.monotonic()
		self._cost_entries.append((now, amount_usd))
		self._prune_cost_window(now)

	def get_cost_rate(self) -> float:
		"""Return current spending rate in $/minute over the sliding window.

		Returns 0.0 if there are fewer than 2 entries or no time span.
		"""
		now = time.monotonic()
		self._prune_cost_window(now)
		if len(self._cost_entries) < 2:
			return 0.0
		earliest_ts = self._cost_entries[0][0]
		span_seconds = now - earliest_ts
		if span_seconds <= 0:
			return 0.0
		total = sum(amount for _, amount in self._cost_entries)
		return total / (span_seconds / 60.0)

	def project_budget_exhaustion(self, remaining_budget: float) -> float | None:
		"""Estimate minutes until budget is exhausted at the current rate.

		Returns None if the rate is zero or cannot be calculated.
		"""
		rate = self.get_cost_rate()
		if rate <= 0:
			return None
		return remaining_budget / rate

	async def check_cost_health(self, remaining_budget: float | None = None) -> None:
		"""Check cost rate and projected exhaustion, alert if unhealthy."""
		rate = self.get_cost_rate()
		if rate <= 0:
			return

		alerts: list[str] = []

		if rate > self._cost_rate_threshold:
			alerts.append(
				f"Cost rate ${rate:.3f}/min exceeds threshold "
				f"${self._cost_rate_threshold:.3f}/min"
			)

		if remaining_budget is not None:
			minutes_left = self.project_budget_exhaustion(remaining_budget)
			if minutes_left is not None and minutes_left < self._exhaustion_warning_minutes:
				alerts.append(
					f"Budget exhaustion in ~{minutes_left:.1f} min "
					f"(${remaining_budget:.2f} remaining at ${rate:.3f}/min)"
				)

		if alerts and self._notifier:
			msg = "Cost alert: " + "; ".join(alerts)
			logger.warning(msg)
			try:
				await self._notifier.send(msg)
			except Exception as exc:
				logger.error("Failed to send cost alert: %s", exc)

	async def recover(self) -> list[str]:
		"""Attempt recovery when heartbeat detects idle stall.

		Returns list of stuck work unit IDs for the controller to handle.
		"""
		stuck_unit_ids: list[str] = []

		if not self._db:
			logger.warning("Recovery requested but no DB available")
			return stuck_unit_ids

		# Query only the fields we need from running work units
		try:
			rows = self._db.conn.execute(
				"SELECT id, title, started_at FROM work_units WHERE status = 'running'"
			).fetchall()
		except Exception as exc:
			logger.error("Failed to query running units for recovery: %s", exc)
			return stuck_unit_ids

		if not rows:
			logger.info("Recovery: no running units found")
			return stuck_unit_ids

		stuck_info: list[tuple[str, str, str]] = []
		for row in rows:
			unit_id = str(row[0])
			title = str(row[1])
			started_at = str(row[2]) if row[2] else ""
			stuck_unit_ids.append(unit_id)
			stuck_info.append((unit_id, title, started_at))

		# Dump states to logs
		for unit_id, title, started_at in stuck_info:
			elapsed = ""
			if started_at:
				try:
					from datetime import datetime, timezone
					started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
					elapsed_secs = (datetime.now(timezone.utc) - started).total_seconds()
					elapsed = f" (running for {int(elapsed_secs)}s)"
				except (ValueError, TypeError):
					pass
			logger.warning(
				"Recovery: stuck unit %s: %s%s",
				unit_id[:12], title[:60], elapsed,
			)

		# Send diagnostic Telegram message
		if self._notifier and stuck_info:
			lines = ["Heartbeat recovery: detected stuck units:"]
			for unit_id, title, started_at in stuck_info:
				elapsed_str = ""
				if started_at:
					try:
						from datetime import datetime, timezone
						started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
						elapsed_secs = (datetime.now(timezone.utc) - started).total_seconds()
						elapsed_str = f" ({int(elapsed_secs)}s)"
					except (ValueError, TypeError):
						pass
				lines.append(f"  - {unit_id[:12]}: {title[:60]}{elapsed_str}")
			try:
				await self._notifier.send("\n".join(lines))
			except Exception as exc:
				logger.error("Failed to send recovery notification: %s", exc)

		return stuck_unit_ids

	async def check(
		self,
		total_merged: int,
		total_failed: int,
		remaining_budget: float | None = None,
	) -> str:
		"""Called from dispatch loop. Returns stop reason or empty string."""
		now = time.monotonic()
		if self._last_check_time > 0 and now - self._last_check_time < self._interval:
			return ""
		self._last_check_time = now

		# Cost health check on every interval tick
		await self.check_cost_health(remaining_budget)

		# First check: just record baseline
		if not self._baseline_set:
			self._baseline_set = True
			self._last_merged_count = total_merged
			return ""

		new_merges = total_merged - self._last_merged_count
		self._last_merged_count = total_merged

		if new_merges == 0:
			self._consecutive_idle += 1
			msg = (
				f"Heartbeat: no merges in last {self._interval}s "
				f"(idle count: {self._consecutive_idle}/{self._idle_threshold}). "
				f"Total: {total_merged} merged, {total_failed} failed."
			)
			logger.warning(msg)
			if self._notifier:
				await self._notifier.send(msg)

			if self._consecutive_idle >= self._idle_threshold:
				if self._enable_recovery:
					await self.recover()
					return "heartbeat_recovered"
				return "heartbeat_stalled"
		else:
			self._consecutive_idle = 0
			msg = (
				f"Heartbeat OK: {new_merges} merged since last check. "
				f"Total: {total_merged} merged, {total_failed} failed."
			)
			logger.info(msg)
			if self._notifier:
				await self._notifier.send(msg)

		return ""

	@property
	def consecutive_idle(self) -> int:
		return self._consecutive_idle
