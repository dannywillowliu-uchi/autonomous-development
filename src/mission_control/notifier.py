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
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


class NotificationPriority(enum.Enum):
	HIGH = "high"
	LOW = "low"


TELEGRAM_MAX_LEN = 4096


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

	@staticmethod
	def _split_message(combined: str, max_len: int = TELEGRAM_MAX_LEN) -> list[str]:
		"""Split a combined message at separator boundaries to fit within max_len."""
		if len(combined) <= max_len:
			return [combined]

		separator = "\n---\n"
		chunks = combined.split(separator)
		result: list[str] = []
		current = ""

		for chunk in chunks:
			if len(chunk) > max_len:
				# Flush current buffer first
				if current:
					result.append(current)
					current = ""
				# Hard-split oversized chunk
				for i in range(0, len(chunk), max_len):
					result.append(chunk[i:i + max_len])
			elif not current:
				current = chunk
			elif len(current) + len(separator) + len(chunk) <= max_len:
				current = current + separator + chunk
			else:
				result.append(current)
				current = chunk

		if current:
			result.append(current)

		return result

	@staticmethod
	def _escape_md2(text: str) -> str:
		"""Escape special characters for Telegram MarkdownV2 format."""
		special = set("_*[]()~`>#+-=|{}.!")
		return "".join(f"\\{ch}" if ch in special else ch for ch in str(text))

	async def _send_with_retry(self, client: httpx.AsyncClient, url: str, payload: dict) -> None:
		"""Send a single Telegram API request with exponential backoff on transient failures."""
		for attempt in range(MAX_RETRIES):
			try:
				resp = await client.post(url, json=payload)
				if resp.status_code == 429:
					default_delay = RETRY_BASE_DELAY * (2 ** attempt)
					retry_after = float(resp.headers.get("Retry-After", default_delay))
					logger.warning(
						"Telegram rate limited (429), retry after %.1fs (%d/%d)",
						retry_after, attempt + 1, MAX_RETRIES,
					)
					if attempt < MAX_RETRIES - 1:
						await asyncio.sleep(retry_after)
						continue
					logger.error("Telegram send failed after %d retries: HTTP 429", MAX_RETRIES)
					return
				if resp.status_code >= 500:
					delay = RETRY_BASE_DELAY * (2 ** attempt)
					logger.warning(
						"Telegram server error (%d), retry after %.1fs (%d/%d)",
						resp.status_code, delay, attempt + 1, MAX_RETRIES,
					)
					if attempt < MAX_RETRIES - 1:
						await asyncio.sleep(delay)
						continue
					logger.error(
						"Telegram send failed after %d retries: HTTP %d",
						MAX_RETRIES, resp.status_code,
					)
					return
				return
			except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as exc:
				delay = RETRY_BASE_DELAY * (2 ** attempt)
				logger.warning(
					"Telegram connection error: %s, retry after %.1fs (%d/%d)",
					exc, delay, attempt + 1, MAX_RETRIES,
				)
				if attempt < MAX_RETRIES - 1:
					await asyncio.sleep(delay)
					continue
				logger.error("Telegram send failed after %d retries: %s", MAX_RETRIES, exc)
				return

	async def _flush_batch(self, messages: list[str]) -> None:
		"""Send a batch of messages, splitting across multiple Telegram messages if needed."""
		if not messages:
			return
		combined = "\n---\n".join(messages)
		parts = self._split_message(combined)
		try:
			client = await self._ensure_client()
			url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
			for part in parts:
				await self._send_with_retry(client, url, {
					"chat_id": self._chat_id,
					"text": part,
					"parse_mode": "Markdown",
					"disable_web_page_preview": True,
				})
		except Exception as exc:
			logger.error("Telegram send failed: %s", exc)

	async def _send_message(self, text: str, parse_mode: str = "MarkdownV2") -> None:
		"""Send a single message immediately with specified parse mode."""
		try:
			client = await self._ensure_client()
			url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
			parts = self._split_message(text)
			for part in parts:
				await self._send_with_retry(client, url, {
					"chat_id": self._chat_id,
					"text": part,
					"parse_mode": parse_mode,
					"disable_web_page_preview": True,
				})
		except Exception as exc:
			logger.error("Telegram send failed: %s", exc)

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

	async def send_epoch_summary(
		self,
		epoch_num: int,
		dispatched: int,
		merged: int,
		failed: int,
		cost_this_epoch: float,
		cumulative_cost: float,
		conflict_rate: float,
	) -> None:
		"""Send a structured epoch summary with formatted metrics table."""
		header = f"*Epoch {epoch_num} Summary*"
		table = (
			"```\n"
			f"{'Metric':<20} {'Value':>10}\n"
			f"{'─' * 20} {'─' * 10}\n"
			f"{'Dispatched':<20} {dispatched:>10}\n"
			f"{'Merged':<20} {merged:>10}\n"
			f"{'Failed':<20} {failed:>10}\n"
			f"{'Conflict Rate':<20} {f'{conflict_rate:.1%}':>10}\n"
			f"{'Cost (epoch)':<20} {f'${cost_this_epoch:.2f}':>10}\n"
			f"{'Cost (cumulative)':<20} {f'${cumulative_cost:.2f}':>10}\n"
			"```"
		)
		await self._send_message(f"{header}\n\n{table}")

	async def send_cost_alert(
		self,
		current_cost: float,
		budget_limit: float,
		projected_cost: float,
	) -> None:
		"""Send a cost alert when projected cost approaches or exceeds budget."""
		e = self._escape_md2
		pct = (projected_cost / budget_limit * 100) if budget_limit > 0 else 0.0
		overage = projected_cost - budget_limit
		lines = [
			"*Cost Alert*",
			"",
			f"Current: {e(f'${current_cost:.2f}')}",
			f"Budget: {e(f'${budget_limit:.2f}')}",
			f"Projected: {e(f'${projected_cost:.2f}')} \\({e(f'{pct:.0f}%')} of budget\\)",
		]
		if overage > 0:
			lines.append(f"Projected overage: {e(f'${overage:.2f}')}")
		await self._send_message("\n".join(lines))
