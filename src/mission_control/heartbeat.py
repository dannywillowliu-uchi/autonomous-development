"""Time-based progress monitor inspired by OpenClaw's heartbeat pattern.

Checks at regular intervals whether the mission is making progress
(new merges since last check). Sends Telegram alerts and can trigger
a stall stop after consecutive idle heartbeats.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mission_control.db import Database
	from mission_control.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class Heartbeat:
	"""Time-based progress monitor.

	Every `interval` seconds, checks if the mission is making progress
	by comparing recent merge activity. If no progress for `idle_threshold`
	consecutive checks, returns a stall signal.
	"""

	def __init__(
		self,
		interval: int = 300,
		idle_threshold: int = 3,
		notifier: TelegramNotifier | None = None,
		db: Database | None = None,
		enable_recovery: bool = True,
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
	) -> str:
		"""Called from dispatch loop. Returns stop reason or empty string."""
		now = time.monotonic()
		if self._last_check_time > 0 and now - self._last_check_time < self._interval:
			return ""
		self._last_check_time = now

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
