"""Auto-update scheduler: runs the update pipeline on a recurring interval."""

from __future__ import annotations

import asyncio
import logging

from autodev.auto_update import AutoUpdatePipeline
from autodev.config import MissionConfig
from autodev.db import Database

logger = logging.getLogger(__name__)


class AutoUpdateScheduler:
	"""Daemon that runs AutoUpdatePipeline on a configurable interval."""

	def __init__(self, config: MissionConfig, db: Database, interval_hours: float = 24.0) -> None:
		self._config = config
		self._db = db
		self._interval_hours = interval_hours
		self._running = False

	async def run_forever(self) -> None:
		"""Main loop: run pipeline, sleep, repeat."""
		self._running = True
		while self._running:
			try:
				pipeline = AutoUpdatePipeline(self._config, self._db)
				results = await pipeline.run()
				for r in results:
					logger.info("Processed: %s -> %s", r.title, r.action)
			except Exception:
				logger.exception("Auto-update cycle failed")
			await asyncio.sleep(self._interval_hours * 3600)

	def stop(self) -> None:
		"""Signal the scheduler to stop after the current cycle."""
		self._running = False
