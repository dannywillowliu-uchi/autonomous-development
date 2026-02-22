"""MCP tool registry -- persistent DB-backed registry for synthesized tools.

Tools are discovered and reused across missions/projects. Quality gating
ensures only high-quality tools get promoted to the registry.
"""

from __future__ import annotations

from dataclasses import dataclass


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
