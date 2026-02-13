"""Post-planning file overlap detection.

Scans work units for overlapping files_hint entries and injects
depends_on edges so that units touching the same file execute
sequentially rather than in parallel.
"""

from __future__ import annotations

import logging

from mission_control.models import WorkUnit

logger = logging.getLogger(__name__)


def _parse_files_hint(hint: str) -> set[str]:
	"""Parse a comma-separated files_hint into a set of stripped paths."""
	if not hint or not hint.strip():
		return set()
	return {f.strip() for f in hint.split(",") if f.strip()}


def resolve_file_overlaps(units: list[WorkUnit]) -> list[WorkUnit]:
	"""Detect file overlaps between units and add depends_on edges.

	For each pair of units with overlapping files_hint entries, the
	lower-priority unit (higher priority number) gets a depends_on edge
	to the higher-priority unit.  Ties are broken by list position
	(earlier = higher priority).

	Existing depends_on entries are preserved.

	Returns the same list (mutated in place) for convenience.
	"""
	if len(units) <= 1:
		return units

	# Parse files_hint for each unit
	unit_files: list[tuple[int, set[str]]] = []
	for i, unit in enumerate(units):
		files = _parse_files_hint(unit.files_hint)
		unit_files.append((i, files))

	# Check all pairs for overlaps
	for i in range(len(units)):
		for j in range(i + 1, len(units)):
			files_i = unit_files[i][1]
			files_j = unit_files[j][1]
			if not files_i or not files_j:
				continue

			overlap = files_i & files_j
			if not overlap:
				continue

			# Determine dependency direction: lower priority number = higher priority
			# Ties broken by list position (earlier index = higher priority)
			unit_i = units[i]
			unit_j = units[j]

			if unit_j.priority < unit_i.priority:
				# j has higher priority (lower number), so i depends on j
				dependent, dependency = unit_i, unit_j
			else:
				# i has higher priority or tie (same priority, earlier position)
				dependent, dependency = unit_j, unit_i

			# Parse existing depends_on
			existing_deps = set()
			if dependent.depends_on:
				existing_deps = {d.strip() for d in dependent.depends_on.split(",") if d.strip()}

			if dependency.id not in existing_deps:
				existing_deps.add(dependency.id)
				dependent.depends_on = ",".join(sorted(existing_deps))
				logger.info(
					"Overlap detected: %s and %s share files %s; "
					"%s now depends on %s",
					unit_i.title[:40], unit_j.title[:40],
					overlap, dependent.title[:40], dependency.title[:40],
				)

	return units
