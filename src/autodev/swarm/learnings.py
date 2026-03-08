"""Swarm learnings -- persistent memory that accumulates across runs.

Writes to .autodev-swarm-learnings.md in the target project root.
The planner reads this file each cycle to avoid repeating mistakes
and build on what worked.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LEARNINGS_FILE = ".autodev-swarm-learnings.md"

HEADER = """\
# Swarm Learnings

Accumulated knowledge from swarm runs. The planner reads this file each cycle
to inform decisions. Entries are added automatically when agents complete tasks,
discover insights, or when approaches fail.

"""


class SwarmLearnings:
	"""Manages persistent learnings for a swarm project."""

	def __init__(self, project_path: Path) -> None:
		self._path = project_path / LEARNINGS_FILE
		self._ensure_file()
		self._existing = self._read()

	def _ensure_file(self) -> None:
		try:
			if not self._path.exists():
				self._path.parent.mkdir(parents=True, exist_ok=True)
				self._path.write_text(HEADER)
		except OSError:
			pass

	def _read(self) -> str:
		try:
			return self._path.read_text()
		except OSError:
			return HEADER

	def _write(self, content: str) -> None:
		try:
			self._path.write_text(content)
			self._existing = content
		except OSError as e:
			logger.warning("Failed to write learnings file: %s", e)

	def _is_duplicate(self, text: str) -> bool:
		"""Check if a learning is already recorded (fuzzy match)."""
		# Normalize for comparison
		normalized = re.sub(r"\s+", " ", text.strip().lower())
		existing_normalized = re.sub(r"\s+", " ", self._existing.lower())
		# Check if the core content (first 80 chars) already exists
		check = normalized[:80]
		return check in existing_normalized

	def add_discovery(self, source: str, text: str) -> bool:
		"""Add a discovery from an agent or task."""
		if self._is_duplicate(text):
			return False
		ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
		entry = f"\n## Discovery ({ts})\n**Source:** {source}\n\n{text.strip()}\n"
		self._append(entry)
		return True

	def add_successful_approach(self, task_title: str, summary: str) -> bool:
		"""Record what worked when a task completed successfully."""
		if self._is_duplicate(summary):
			return False
		ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
		entry = f"\n## What Worked ({ts})\n**Task:** {task_title}\n\n{summary.strip()}\n"
		self._append(entry)
		return True

	def add_failed_approach(self, task_title: str, error: str, attempt: int) -> bool:
		"""Record what didn't work to avoid repeating mistakes."""
		if self._is_duplicate(error):
			return False
		ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
		entry = (
			f"\n## What Failed ({ts})\n"
			f"**Task:** {task_title} (attempt {attempt})\n\n"
			f"{error.strip()}\n"
		)
		self._append(entry)
		return True

	def add_stagnation_insight(self, metric: str, pivot: str) -> bool:
		"""Record a stagnation detection and the pivot taken."""
		if self._is_duplicate(pivot):
			return False
		ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
		entry = (
			f"\n## Stagnation Pivot ({ts})\n"
			f"**Metric:** {metric}\n\n"
			f"{pivot.strip()}\n"
		)
		self._append(entry)
		return True

	def _append(self, entry: str) -> None:
		content = self._read() + entry
		# Keep file bounded -- trim oldest entries if over ~200 lines
		lines = content.split("\n")
		if len(lines) > 200:
			# Keep header + most recent entries
			header_end = content.find("\n## ")
			if header_end == -1:
				header_end = len(HEADER)
			header = content[:header_end]
			entries = content[header_end:]
			# Keep last ~150 lines of entries
			entry_lines = entries.split("\n")
			trimmed = "\n".join(entry_lines[-150:])
			content = header + trimmed
		self._write(content)
		logger.info("Added learning to %s", self._path)

	def get_for_planner(self, max_lines: int = 50) -> str:
		"""Get a condensed version of learnings for the planner prompt."""
		content = self._read()
		if content.strip() == HEADER.strip():
			return ""

		# Extract just the entries, skip header
		header_end = content.find("\n## ")
		if header_end == -1:
			return ""

		entries = content[header_end:]
		lines = entries.strip().split("\n")

		# Return last N lines (most recent learnings)
		if len(lines) > max_lines:
			lines = lines[-max_lines:]

		return "## Accumulated Learnings (from previous runs)\n\n" + "\n".join(lines)
