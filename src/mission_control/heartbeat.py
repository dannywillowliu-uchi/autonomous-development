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
	) -> None:
		self._interval = interval
		self._idle_threshold = idle_threshold
		self._notifier = notifier
		self._last_check_time: float = 0.0
		self._last_merged_count: int = 0
		self._consecutive_idle: int = 0

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
		if self._last_merged_count == 0 and total_merged == 0:
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
