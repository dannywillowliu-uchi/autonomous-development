"""Telegram notifications for mission events.

Uses async httpx client. Notifications are batched over a 5-second window
and sent as a single concatenated message to reduce API calls. Includes
priority-based backpressure: when the internal queue exceeds MAX_QUEUE_SIZE,
low-priority notifications are dropped.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from collections import deque

import httpx

logger = logging.getLogger(__name__)

BATCH_WINDOW = 5.0
MAX_QUEUE_SIZE = 100


class NotificationPriority(enum.Enum):
	HIGH = "high"
	LOW = "low"


class TelegramNotifier:
	"""Sends mission updates to Telegram via Bot API with message batching."""

	def __init__(self, bot_token: str, chat_id: str) -> None:
		self._bot_token = bot_token
		self._chat_id = chat_id
		self._client: httpx.AsyncClient | None = None
		self._priority_queue: deque[tuple[NotificationPriority, str]] = deque()
		self._batch_task: asyncio.Task[None] | None = None
		self._queue_event: asyncio.Event = asyncio.Event()

	async def _ensure_client(self) -> httpx.AsyncClient:
		if self._client is None:
			self._client = httpx.AsyncClient(timeout=10.0)
		return self._client

	def _ensure_batch_task(self) -> None:
		if self._batch_task is None or self._batch_task.done():
			self._batch_task = asyncio.create_task(self._batch_loop())

	async def _batch_loop(self) -> None:
		"""Collect messages for BATCH_WINDOW seconds, then flush."""
		while True:
			try:
				await self._queue_event.wait()
				self._queue_event.clear()
			except asyncio.CancelledError:
				return

			await asyncio.sleep(BATCH_WINDOW)

			messages: list[str] = []
			while self._priority_queue:
				_, msg = self._priority_queue.popleft()
				messages.append(msg)

			await self._flush_batch(messages)

	async def _flush_batch(self, messages: list[str]) -> None:
		"""Send a batch of messages as a single concatenated Telegram message."""
		if not messages:
			return
		combined = "\n---\n".join(messages)
		try:
			client = await self._ensure_client()
			url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
			await client.post(url, json={
				"chat_id": self._chat_id,
				"text": combined,
				"parse_mode": "Markdown",
				"disable_web_page_preview": True,
			})
		except Exception as exc:
			logger.warning("Telegram send failed: %s", exc)

	def _apply_backpressure(self) -> None:
		"""Drop low-priority messages when queue exceeds MAX_QUEUE_SIZE."""
		if len(self._priority_queue) <= MAX_QUEUE_SIZE:
			return
		high: list[tuple[NotificationPriority, str]] = []
		low: list[tuple[NotificationPriority, str]] = []
		for item in self._priority_queue:
			if item[0] == NotificationPriority.HIGH:
				high.append(item)
			else:
				low.append(item)
		low_slots = max(0, MAX_QUEUE_SIZE - len(high))
		dropped = len(low) - low_slots
		if dropped > 0:
			low = low[:low_slots]
			logger.warning("Backpressure: dropped %d low-priority notifications", dropped)
		self._priority_queue.clear()
		self._priority_queue.extend(high)
		self._priority_queue.extend(low)

	async def send(self, message: str, priority: NotificationPriority = NotificationPriority.LOW) -> None:
		"""Queue a message for batched delivery with optional priority."""
		self._ensure_batch_task()
		self._priority_queue.append((priority, message))
		self._apply_backpressure()
		self._queue_event.set()

	async def close(self) -> None:
		"""Flush remaining messages and close the HTTP client."""
		if self._batch_task is not None and not self._batch_task.done():
			self._batch_task.cancel()
			try:
				await self._batch_task
			except asyncio.CancelledError:
				pass

		remaining: list[str] = []
		while self._priority_queue:
			_, msg = self._priority_queue.popleft()
			remaining.append(msg)
		if remaining:
			await self._flush_batch(remaining)

		if self._client is not None:
			await self._client.aclose()
			self._client = None

	async def send_mission_start(self, objective: str, workers: int) -> None:
		await self.send(
			f"*Mission started*\n"
			f"Objective: {objective[:200]}\n"
			f"Workers: {workers}",
			priority=NotificationPriority.HIGH,
		)

	async def send_mission_end(
		self,
		objective_met: bool,
		merged: int,
		failed: int,
		wall_time: float,
		stopped_reason: str,
		verification_passed: bool | None = None,
	) -> None:
		status = "COMPLETE" if objective_met else "STOPPED"
		lines = [
			f"*Mission {status}*",
			f"Merged: {merged}, Failed: {failed}",
			f"Wall time: {wall_time:.0f}s",
			f"Reason: {stopped_reason}",
		]
		if verification_passed is not None:
			lines.append(
				f"Final verification: {'PASS' if verification_passed else 'FAIL'}"
			)
		await self.send("\n".join(lines), priority=NotificationPriority.HIGH)

	async def send_merge_conflict(self, unit_title: str, failure: str) -> None:
		await self.send(
			f"Merge conflict: {unit_title[:100]}\n"
			f"```\n{failure[:300]}\n```",
			priority=NotificationPriority.HIGH,
		)
