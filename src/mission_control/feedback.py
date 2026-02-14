"""Feedback utilities -- worker context from past experiences."""

from __future__ import annotations

import json
import logging
import re

from mission_control.db import Database
from mission_control.models import WorkUnit

log = logging.getLogger(__name__)


def _extract_keywords(text: str) -> list[str]:
	"""Extract meaningful keywords from text for experience search."""
	stop_words = {
		"the", "and", "for", "that", "this", "with", "from", "are", "was",
		"will", "have", "has", "been", "not", "but", "can", "all", "its",
		"add", "fix", "update", "implement", "create", "make", "use",
	}
	words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
	return [w for w in words if len(w) > 2 and w not in stop_words]


def get_worker_context(
	db: Database,
	unit: WorkUnit,
) -> str:
	"""Find relevant past experiences for a work unit."""
	keywords = _extract_keywords(f"{unit.title} {unit.description} {unit.files_hint}")
	if not keywords:
		return ""

	search_keywords = keywords[:8]
	experiences = db.search_experiences(search_keywords, limit=3)
	if not experiences:
		return ""

	lines: list[str] = []
	for exp in experiences:
		if exp.status == "completed":
			lines.append(f"- [{exp.title}] succeeded (reward={exp.reward:.2f}): {exp.summary}")
			if exp.discoveries:
				try:
					discoveries = json.loads(exp.discoveries)
					if discoveries:
						lines.append(f"  Insights: {', '.join(discoveries[:2])}")
				except (json.JSONDecodeError, TypeError):
					pass
			if exp.concerns:
				try:
					concerns = json.loads(exp.concerns)
					if concerns:
						lines.append(f"  Pitfalls: {', '.join(concerns[:2])}")
				except (json.JSONDecodeError, TypeError):
					pass
		else:
			lines.append(f"- [{exp.title}] FAILED: {exp.summary}")

	return "\n".join(lines)
