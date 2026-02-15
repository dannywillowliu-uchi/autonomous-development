"""Telegram notifications for mission events.

Uses async httpx client. Notifications are batched over a 5-second window
and sent as a single concatenated message to reduce API calls.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

BATCH_WINDOW = 5.0


class TelegramNotifier:
	"""Sends mission updates to Telegram via Bot API with message batching."""

	def __init__(self, bot_token: str, chat_id: str) -> None:
		self._bot_token = bot_token
		self._chat_id = chat_id
		self._client: httpx.AsyncClient | None = None
		self._batch_queue: asyncio.Queue[str] = asyncio.Queue()
		self._batch_task: asyncio.Task[None] | None = None

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
			messages: list[str] = []
			try:
				msg = await self._batch_queue.get()
				messages.append(msg)
			except asyncio.CancelledError:
				return

			await asyncio.sleep(BATCH_WINDOW)

			while not self._batch_queue.empty():
				try:
					messages.append(self._batch_queue.get_nowait())
				except asyncio.QueueEmpty:
					break

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

	async def send(self, message: str) -> None:
		"""Queue a message for batched delivery."""
		self._ensure_batch_task()
		self._batch_queue.put_nowait(message)

	async def close(self) -> None:
		"""Flush remaining messages and close the HTTP client."""
		if self._batch_task is not None and not self._batch_task.done():
			self._batch_task.cancel()
			try:
				await self._batch_task
			except asyncio.CancelledError:
				pass

		remaining: list[str] = []
		while not self._batch_queue.empty():
			try:
				remaining.append(self._batch_queue.get_nowait())
			except asyncio.QueueEmpty:
				break
		if remaining:
			await self._flush_batch(remaining)

		if self._client is not None:
			await self._client.aclose()
			self._client = None

	async def send_mission_start(self, objective: str, workers: int) -> None:
		await self.send(
			f"*Mission started*\n"
			f"Objective: {objective[:200]}\n"
			f"Workers: {workers}"
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
		await self.send("\n".join(lines))

	async def send_merge_conflict(self, unit_title: str, failure: str) -> None:
		await self.send(
			f"Merge conflict: {unit_title[:100]}\n"
			f"```\n{failure[:300]}\n```"
		)
