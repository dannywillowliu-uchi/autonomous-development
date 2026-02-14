"""Telegram notifications for mission events.

Uses stdlib urllib to avoid adding aiohttp dependency. Runs blocking
urlopen in asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class TelegramNotifier:
	"""Sends mission updates to Telegram via Bot API."""

	def __init__(self, bot_token: str, chat_id: str) -> None:
		self._bot_token = bot_token
		self._chat_id = chat_id

	async def send(self, message: str) -> None:
		"""Send a text message to the configured chat."""
		try:
			await asyncio.to_thread(self._send_sync, message)
		except Exception as exc:
			logger.warning("Telegram send failed: %s", exc)

	def _send_sync(self, message: str) -> None:
		"""Blocking send via urllib."""
		url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
		payload = json.dumps({
			"chat_id": self._chat_id,
			"text": message,
			"parse_mode": "Markdown",
			"disable_web_page_preview": True,
		}).encode()
		req = urllib.request.Request(
			url,
			data=payload,
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		try:
			with urllib.request.urlopen(req, timeout=10) as resp:
				resp.read()
		except urllib.error.URLError as exc:
			logger.warning("Telegram API error: %s", exc)

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
