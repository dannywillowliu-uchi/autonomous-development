"""Priority recalculation engine for persistent backlog items."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from mission_control.db import Database
from mission_control.models import BacklogItem, _now_iso

logger = logging.getLogger(__name__)

INFRASTRUCTURE_KEYWORDS = ("infrastructure", "timeout", "network")
MAX_FAILURE_PENALTY = 0.6
FAILURE_PENALTY_PER_ATTEMPT = 0.2
STALENESS_PENALTY = 0.1
STALENESS_THRESHOLD_HOURS = 72


def _compute_base_score(impact: int, effort: int) -> float:
	"""Base score: impact * (11 - effort) / 10."""
	return impact * (11 - effort) / 10


def _compute_failure_penalty(attempt_count: int, last_failure_reason: str | None) -> float:
	"""Reduce score by 20% per failed attempt, cap at 60%.

	Skip penalty if last failure reason contains infrastructure/timeout/network.
	"""
	if attempt_count <= 0:
		return 0.0
	if last_failure_reason:
		reason_lower = last_failure_reason.lower()
		if any(kw in reason_lower for kw in INFRASTRUCTURE_KEYWORDS):
			return 0.0
	raw_penalty = attempt_count * FAILURE_PENALTY_PER_ATTEMPT
	return min(raw_penalty, MAX_FAILURE_PENALTY)


def _is_stale(updated_at: str, now: datetime) -> bool:
	"""Check if item hasn't been updated in the staleness threshold period."""
	try:
		updated = datetime.fromisoformat(updated_at)
		if updated.tzinfo is None:
			updated = updated.replace(tzinfo=timezone.utc)
		delta = now - updated
		return delta.total_seconds() > STALENESS_THRESHOLD_HOURS * 3600
	except (ValueError, TypeError):
		return False


def recalculate_priorities(db: Database) -> list[BacklogItem]:
	"""Recalculate priority scores for all pending/in_progress backlog items.

	Scoring:
	- Base: impact * (11 - effort) / 10
	- Failure penalty: -20% per failed attempt (cap 60%), skip if infrastructure
	- Staleness: -10% if not updated in last 72 hours
	- Pinned override: if pinned_score is set, use it directly
	"""
	items = db.list_backlog_items(limit=1000)
	now = datetime.now(timezone.utc)
	updated: list[BacklogItem] = []

	for item in items:
		if item.status not in ("pending", "in_progress"):
			continue

		if item.pinned_score is not None:
			if item.priority_score != item.pinned_score:
				item.priority_score = item.pinned_score
				item.updated_at = _now_iso()
				db.update_backlog_item(item)
				updated.append(item)
			continue

		base = _compute_base_score(item.impact, item.effort)
		penalty = _compute_failure_penalty(item.attempt_count, item.last_failure_reason)
		score = base * (1.0 - penalty)

		if _is_stale(item.updated_at, now):
			score *= (1.0 - STALENESS_PENALTY)

		score = round(score, 4)
		if item.priority_score != score:
			item.priority_score = score
			db.update_backlog_item(item)
			updated.append(item)

	logger.info("Recalculated priorities for %d items (%d updated)", len(items), len(updated))
	return updated


_PRIORITY_RE = re.compile(r"^##\s+P(\d)\b")


def parse_backlog_md(path: Path) -> list[BacklogItem]:
	"""Parse a BACKLOG.md file into BacklogItem instances.

	Expected format:
	  ## P0: Title here
	  description text...
	  **Files**: file1.py, file2.py

	Priority headers map to impact (P0=10, P1=9, ..., P9=1).
	Effort defaults to 5. Items get a base priority_score computed from impact/effort.
	"""
	text = path.read_text(encoding="utf-8")
	items: list[BacklogItem] = []
	current_title = ""
	current_priority = 0
	current_lines: list[str] = []

	def _flush() -> None:
		nonlocal current_title, current_lines
		if not current_title:
			return
		description = "\n".join(current_lines).strip()
		impact = 10 - current_priority
		effort = 5
		score = _compute_base_score(impact, effort)
		items.append(BacklogItem(
			title=current_title,
			description=description,
			impact=impact,
			effort=effort,
			priority_score=round(score, 4),
		))
		current_title = ""
		current_lines = []

	for line in text.splitlines():
		match = _PRIORITY_RE.match(line)
		if match:
			_flush()
			current_priority = int(match.group(1))
			header_rest = line[match.end():].strip()
			if header_rest.startswith(":"):
				header_rest = header_rest[1:].strip()
			current_title = header_rest or f"P{current_priority} item"
		elif current_title:
			current_lines.append(line)

	_flush()
	return items
