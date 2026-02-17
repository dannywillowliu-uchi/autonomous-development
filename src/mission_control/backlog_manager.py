"""Backlog management -- extracted from continuous_controller."""

from __future__ import annotations

import json
import logging

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import BacklogItem, Handoff, WorkUnit, _now_iso

logger = logging.getLogger(__name__)


class BacklogManager:
	"""Manages backlog item lifecycle during a mission."""

	def __init__(self, db: Database, config: MissionConfig) -> None:
		self.db = db
		self.config = config
		self.backlog_item_ids: list[str] = []

	def load_backlog_objective(self, limit: int = 5) -> str | None:
		"""Load top pending backlog items and compose an objective string.

		Marks selected items as in_progress and stores their IDs for
		post-mission completion tracking.

		Returns the composed objective string, or None if no backlog items found.
		"""
		items = self.db.get_pending_backlog(limit=limit)
		if not items:
			return None

		self.backlog_item_ids = [item.id for item in items]

		# Mark selected items as in_progress
		for item in items:
			item.status = "in_progress"
			item.updated_at = _now_iso()
			self.db.update_backlog_item(item)

		# Compose objective from backlog items
		lines = ["Priority backlog items to address:"]
		for i, item in enumerate(items, 1):
			lines.append(
				f"{i}. [{item.track}] {item.title} "
				f"(backlog_item_id={item.id}, priority={item.priority_score:.1f}): "
				f"{item.description}"
			)

		logger.info(
			"Loaded %d backlog items as mission objective (IDs: %s)",
			len(items),
			", ".join(item.id[:8] for item in items),
		)
		return "\n".join(lines)

	def update_backlog_on_completion(
		self, objective_met: bool, handoffs: list[Handoff],
	) -> None:
		"""Update backlog items after mission ends.

		If objective_met: mark all targeted items as completed.
		If not: reset to pending, store failure context, increment attempt_count.
		"""
		if not self.backlog_item_ids:
			return

		# Build failure context from handoffs
		failure_reasons: list[str] = []
		for h in handoffs:
			if h.status != "completed":
				try:
					concerns = json.loads(h.concerns) if h.concerns else []
				except (json.JSONDecodeError, TypeError):
					concerns = []
				if concerns:
					failure_reasons.append(concerns[-1][:200])
				elif h.summary:
					failure_reasons.append(h.summary[:200])

		for item_id in self.backlog_item_ids:
			item = self.db.get_backlog_item(item_id)
			if item is None:
				continue

			if objective_met:
				item.status = "completed"
				item.updated_at = _now_iso()
				self.db.update_backlog_item(item)
			else:
				item.status = "pending"
				item.attempt_count += 1
				if failure_reasons:
					item.last_failure_reason = "; ".join(failure_reasons[:3])
				item.updated_at = _now_iso()
				self.db.update_backlog_item(item)

		logger.info(
			"Updated %d backlog items: %s",
			len(self.backlog_item_ids),
			"completed" if objective_met else "reset to pending",
		)

	def update_backlog_from_completion(
		self,
		unit: WorkUnit,
		merged: bool,
		handoff: Handoff | None,
		mission_id: str,
	) -> None:
		"""Update backlog items based on individual unit completion.

		Searches backlog by title matching against the unit title, then:
		- On successful merge: marks matching item 'completed', sets source_mission_id
		- On failure after max retries: increments attempt_count, sets last_failure_reason
		- On partial completion (retryable): keeps 'in_progress', appends context
		"""
		# Extract keywords from unit title for matching
		title_words = [w for w in unit.title.lower().split() if len(w) > 2]
		if not title_words:
			return

		matching_items = self.db.search_backlog_items(title_words, limit=5)
		if not matching_items:
			return

		# Score matches: count how many title words appear in the backlog item title
		best_item: BacklogItem | None = None
		best_score = 0
		for item in matching_items:
			item_title_lower = item.title.lower()
			score = sum(1 for w in title_words if w in item_title_lower)
			if score > best_score:
				best_score = score
				best_item = item

		if best_item is None or best_score == 0:
			return

		# Extract failure context from handoff
		failure_reason = ""
		context_additions: list[str] = []
		if handoff:
			try:
				concerns = json.loads(handoff.concerns) if handoff.concerns else []
			except (json.JSONDecodeError, TypeError):
				concerns = []
			if concerns:
				failure_reason = "; ".join(str(c)[:200] for c in concerns[:3])

			try:
				discoveries = json.loads(handoff.discoveries) if handoff.discoveries else []
			except (json.JSONDecodeError, TypeError):
				discoveries = []
			if discoveries:
				context_additions.extend(str(d)[:200] for d in discoveries[:3])
			if concerns:
				context_additions.extend(str(c)[:200] for c in concerns[:3])

		if not failure_reason and unit.output_summary:
			failure_reason = unit.output_summary[:300]

		if merged:
			# Successful merge: mark completed
			best_item.status = "completed"
			best_item.source_mission_id = mission_id
			best_item.updated_at = _now_iso()
			self.db.update_backlog_item(best_item)
			logger.info(
				"Backlog item '%s' marked completed (matched unit '%s')",
				best_item.title[:40], unit.title[:40],
			)
		elif unit.attempt >= unit.max_attempts:
			# Failed after max retries
			best_item.attempt_count += 1
			best_item.last_failure_reason = failure_reason or "Max retries exceeded"
			best_item.updated_at = _now_iso()
			self.db.update_backlog_item(best_item)
			logger.info(
				"Backlog item '%s' failure recorded (attempt %d, unit '%s')",
				best_item.title[:40], best_item.attempt_count, unit.title[:40],
			)
		else:
			# Partial completion: keep in_progress, append context
			best_item.status = "in_progress"
			if context_additions:
				separator = "\n\n--- Context from unit " + unit.id[:8] + " ---\n"
				best_item.description += separator + "\n".join(context_additions)
			best_item.updated_at = _now_iso()
			self.db.update_backlog_item(best_item)
			logger.info(
				"Backlog item '%s' updated with partial context (unit '%s')",
				best_item.title[:40], unit.title[:40],
			)
