"""Swarm learnings -- persistent memory that accumulates across runs.

Writes to .autodev-swarm-learnings.md in the target project root.
The planner reads this file each cycle to avoid repeating mistakes
and build on what worked.
"""

from __future__ import annotations

import hashlib
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


_FILE_PATH_RE = re.compile(r"(?:\./|[\w-]+/)+[\w.-]+\.py\b")
_FUNC_NAME_RE = re.compile(r"\b\w+\(\)|(?<!\w)[a-z]+(?:_[a-z]+){2,}\b")
_ACTIONABLE_RE = re.compile(r"\b(?:bug|fix|race|gotcha|must|never)\b", re.IGNORECASE)
_GENERIC_RE = re.compile(r"\b(?:all pass|completed|successfully)\b", re.IGNORECASE)


def _score_learning(text: str) -> float:
	"""Score learning text quality. Returns >= 0.5 for useful learnings."""
	score = 0.0
	if _FILE_PATH_RE.search(text):
		score += 1
	if _FUNC_NAME_RE.search(text):
		score += 1
	if _ACTIONABLE_RE.search(text):
		score += 1
	if len(text.strip()) < 50:
		score -= 1
	if _GENERIC_RE.search(text):
		score -= 0.5
	return score


def _hash_text(text: str) -> str:
	"""Normalize text and return sha256 hex digest."""
	normalized = re.sub(r"\s+", " ", text.strip().lower())
	return hashlib.sha256(normalized.encode()).hexdigest()


class SwarmLearnings:
	"""Manages persistent learnings for a swarm project."""

	def __init__(self, project_path: Path) -> None:
		self._path = project_path / LEARNINGS_FILE
		self._seen_hashes: set[str] = set()
		self._ensure_file()
		self._existing = self._read()
		self._populate_hashes()

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

	def _populate_hashes(self) -> None:
		"""Build hash set from existing file content."""
		content = self._existing
		# Extract text blocks between section headers
		sections = re.split(r"\n## ", content)
		for section in sections[1:]:  # skip header
			# Body is after the first blank line
			parts = section.split("\n\n", 1)
			if len(parts) == 2:
				body = parts[1].strip()
				if body:
					self._seen_hashes.add(_hash_text(body))

	def _write(self, content: str) -> None:
		try:
			self._path.write_text(content)
			self._existing = content
		except OSError as e:
			logger.warning("Failed to write learnings file: %s", e)

	def _is_duplicate(self, text: str) -> bool:
		"""Check if a learning is already recorded (fuzzy match + hash)."""
		if _hash_text(text) in self._seen_hashes:
			return True
		# Normalize for comparison
		normalized = re.sub(r"\s+", " ", text.strip().lower())
		existing_normalized = re.sub(r"\s+", " ", self._existing.lower())
		# Check if the core content (first 80 chars) already exists
		check = normalized[:80]
		return check in existing_normalized

	def add_discovery(self, source: str, text: str) -> bool:
		"""Add a discovery from an agent or task."""
		if _score_learning(text) < 0.5:
			logger.debug("Rejected low-quality discovery: %.50s...", text)
			return False
		if self._is_duplicate(text):
			return False
		ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
		entry = f"\n## Discovery ({ts})\n**Source:** {source}\n\n{text.strip()}\n"
		self._append(entry)
		return True

	def add_successful_approach(self, task_title: str, summary: str) -> bool:
		"""Record what worked when a task completed successfully."""
		if _score_learning(summary) < 0.5:
			logger.debug("Rejected low-quality approach: %.50s...", summary)
			return False
		if self._is_duplicate(summary):
			return False
		ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
		entry = f"\n## What Worked ({ts})\n**Task:** {task_title}\n\n{summary.strip()}\n"
		self._append(entry)
		return True

	def add_failed_approach(self, task_title: str, error: str, attempt: int) -> bool:
		"""Record what didn't work to avoid repeating mistakes."""
		if _score_learning(error) < 0.5:
			logger.debug("Rejected low-quality failure: %.50s...", error)
			return False
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
		# Extract body text from entry for hashing
		parts = entry.split("\n\n", 1)
		if len(parts) == 2:
			body = parts[1].strip()
			if body:
				self._seen_hashes.add(_hash_text(body))
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
