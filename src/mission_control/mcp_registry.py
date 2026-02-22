"""MCP tool registry -- persistent DB-backed registry for synthesized tools.

Tools are discovered and reused across missions/projects. Quality gating
ensures only high-quality tools get promoted to the registry.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mission_control.models import _now_iso
from mission_control.tool_synthesis import ToolRegistry

if TYPE_CHECKING:
	from mission_control.config import MCPRegistryConfig
	from mission_control.db import Database

logger = logging.getLogger(__name__)


@dataclass
class MCPToolEntry:
	"""A registered tool in the MCP registry."""

	id: str = ""
	name: str = ""
	description: str = ""
	input_schema: str = "{}"
	handler_path: str = ""
	quality_score: float = 0.5
	usage_count: int = 0
	created_by_mission: str = ""
	created_at: str = ""
	updated_at: str = ""


class MCPToolRegistry:
	"""Persistent tool registry backed by SQLite via the Database class."""

	def __init__(self, db: Database, config: MCPRegistryConfig) -> None:
		self._db = db
		self._config = config

	def register(
		self,
		name: str,
		description: str,
		script_content: str,
		quality_score: float,
		mission_id: str,
	) -> MCPToolEntry | None:
		"""Register a tool if quality_score meets the promotion threshold.

		Returns the MCPToolEntry if registered, None if below threshold.
		Validates script content using ToolRegistry's blocked imports check.
		"""
		if quality_score < self._config.promotion_threshold:
			logger.debug(
				"Tool %r quality %.2f below threshold %.2f, skipping",
				name, quality_score, self._config.promotion_threshold,
			)
			return None

		# Validate script safety
		ToolRegistry._validate_script_content_static(script_content)

		now = _now_iso()
		entry = MCPToolEntry(
			id=str(uuid.uuid4()),
			name=name,
			description=description,
			input_schema="{}",
			handler_path="",
			quality_score=quality_score,
			usage_count=0,
			created_by_mission=mission_id,
			created_at=now,
			updated_at=now,
		)
		self._db.insert_tool_registry_entry(entry)
		logger.info("Registered tool %r (score=%.2f) from mission %s", name, quality_score, mission_id)
		return entry

	def discover(self, intent: str, limit: int = 5) -> list[MCPToolEntry]:
		"""Search for tools matching the given intent keywords."""
		return self._db.search_tool_registry(intent, limit=limit)

	def get_handler(self, name: str) -> str | None:
		"""Return the handler_path for a registered tool, or None."""
		entry = self._db.get_tool_registry_entry(name)
		if entry is None:
			return None
		return entry.handler_path or None

	def record_usage(self, name: str) -> None:
		"""Increment usage_count and boost quality score via EMA."""
		entry = self._db.get_tool_registry_entry(name)
		if entry is None:
			return
		entry.usage_count += 1
		alpha = self._config.ema_alpha
		entry.quality_score = alpha * 1.0 + (1 - alpha) * entry.quality_score
		entry.updated_at = _now_iso()
		self._db.update_tool_registry_entry(entry)

	def update_quality(self, name: str, score: float) -> None:
		"""Update quality score via EMA: new = alpha * score + (1 - alpha) * old."""
		entry = self._db.get_tool_registry_entry(name)
		if entry is None:
			return
		alpha = self._config.ema_alpha
		entry.quality_score = alpha * score + (1 - alpha) * entry.quality_score
		entry.updated_at = _now_iso()
		self._db.update_tool_registry_entry(entry)

	def prune(self, min_quality: float = 0.0) -> int:
		"""Remove tools below the given quality threshold. Returns count removed."""
		return self._db.delete_tool_registry_entries_below(min_quality)
