"""Post-planning file overlap detection.

Scans work units for overlapping files_hint entries and injects
depends_on edges so that units touching the same file execute
sequentially rather than in parallel.
"""

from __future__ import annotations

import logging
from collections import deque

from mission_control.models import WorkUnit

logger = logging.getLogger(__name__)


def _parse_files_hint(hint: str) -> set[str]:
	"""Parse a comma-separated files_hint into a set of stripped paths."""
	if not hint or not hint.strip():
		return set()
	return {f.strip() for f in hint.split(",") if f.strip()}


def _add_dependency_edge(dependent: WorkUnit, dependency: WorkUnit, reason: str = "") -> bool:
	"""Add a depends_on edge from dependent to dependency.

	Returns True if the edge was newly added, False if it already existed.
	"""
	existing_deps = set()
	if dependent.depends_on:
		existing_deps = {d.strip() for d in dependent.depends_on.split(",") if d.strip()}

	if dependency.id in existing_deps:
		return False

	existing_deps.add(dependency.id)
	dependent.depends_on = ",".join(sorted(existing_deps))
	if reason:
		logger.info(
			"%s; %s now depends on %s",
			reason, dependent.title[:40], dependency.title[:40],
		)
	return True


def resolve_file_overlaps(units: list[WorkUnit]) -> list[WorkUnit]:
	"""Detect file overlaps between units and add depends_on edges.

	For each pair of units with overlapping files_hint entries, the
	lower-priority unit (higher priority number) gets a depends_on edge
	to the higher-priority unit.  Ties are broken by list position
	(earlier = higher priority).

	When both units in a pair have empty files_hint (unknown scope),
	they are conservatively serialized (later unit depends on earlier).

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

			if not files_i and not files_j:
				# Both unknown scope -- serialize conservatively
				_add_dependency_edge(
					units[j], units[i],
					reason=f"Both empty files_hint: {units[i].title[:40]} and {units[j].title[:40]}",
				)
				continue
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

			_add_dependency_edge(
				dependent, dependency,
				reason=f"Overlap detected: {unit_i.title[:40]} and {unit_j.title[:40]} share files {overlap}",
			)

	_break_cycles(units)
	return units


def _parse_depends_on(dep_str: str) -> set[str]:
	"""Parse a comma-separated depends_on string into a set of IDs."""
	if not dep_str or not dep_str.strip():
		return set()
	return {d.strip() for d in dep_str.split(",") if d.strip()}


def topological_layers(units: list[WorkUnit]) -> list[list[WorkUnit]]:
	"""Group work units into parallelizable layers via topological sort.

	Each inner list contains units that can execute concurrently (all
	dependencies satisfied by prior layers). Uses Kahn's algorithm.

	Edge cases:
	  - No dependencies: single layer with all units
	  - All serial: one unit per layer
	  - Orphan deps (IDs not in `units`): silently ignored
	  - Empty input: returns []
	  - Cycles (shouldn't happen after _break_cycles): forced into final layer
	"""
	if not units:
		return []

	unit_map: dict[str, WorkUnit] = {u.id: u for u in units}
	unit_ids = set(unit_map.keys())

	# Build in-degree and adjacency (dependency -> dependents)
	in_degree: dict[str, int] = {uid: 0 for uid in unit_ids}
	adjacency: dict[str, list[str]] = {uid: [] for uid in unit_ids}

	for u in units:
		for dep_id in _parse_depends_on(u.depends_on):
			if dep_id in unit_ids:
				adjacency[dep_id].append(u.id)
				in_degree[u.id] += 1

	layers: list[list[WorkUnit]] = []
	remaining = dict(in_degree)

	while remaining:
		# Collect nodes with no unresolved dependencies
		layer_ids = [uid for uid, deg in remaining.items() if deg == 0]
		if not layer_ids:
			# Cycle remnant -- force remaining into one layer
			layer_ids = list(remaining.keys())

		layers.append([unit_map[uid] for uid in layer_ids])

		# Remove this layer and update in-degrees
		for uid in layer_ids:
			del remaining[uid]
			for neighbor in adjacency.get(uid, []):
				if neighbor in remaining:
					remaining[neighbor] -= 1

	return layers


def _break_cycles(units: list[WorkUnit]) -> None:
	"""Detect and break circular dependencies using topological sort.

	Uses Kahn's algorithm to find cycles. When a cycle is detected,
	breaks it by removing the edge from the lowest-priority unit
	(highest priority number) in the cycle.
	"""
	if len(units) <= 1:
		return

	unit_map: dict[str, WorkUnit] = {u.id: u for u in units}
	unit_ids = set(unit_map.keys())

	while True:
		# Build adjacency and in-degree from current depends_on state
		in_degree: dict[str, int] = {uid: 0 for uid in unit_ids}
		adjacency: dict[str, list[str]] = {uid: [] for uid in unit_ids}

		for u in units:
			for dep_id in _parse_depends_on(u.depends_on):
				if dep_id in unit_ids:
					adjacency[dep_id].append(u.id)
					in_degree[u.id] += 1

		# Kahn's algorithm
		queue: deque[str] = deque()
		for uid, deg in in_degree.items():
			if deg == 0:
				queue.append(uid)

		visited = 0
		while queue:
			node = queue.popleft()
			visited += 1
			for neighbor in adjacency[node]:
				in_degree[neighbor] -= 1
				if in_degree[neighbor] == 0:
					queue.append(neighbor)

		if visited == len(unit_ids):
			return

		# Cycle exists -- find units still in the cycle (in_degree > 0)
		cycle_ids = {uid for uid, deg in in_degree.items() if deg > 0}

		# Find the lowest-priority unit in the cycle (highest priority number).
		# Ties broken by later position in the original list.
		worst_unit: WorkUnit | None = None
		worst_idx = -1
		for i, u in enumerate(units):
			if u.id not in cycle_ids:
				continue
			if worst_unit is None or u.priority > worst_unit.priority or (
				u.priority == worst_unit.priority and i > worst_idx
			):
				worst_unit = u
				worst_idx = i

		assert worst_unit is not None

		# Remove one incoming edge from worst_unit that is part of the cycle
		deps = _parse_depends_on(worst_unit.depends_on)
		cycle_dep = None
		for dep_id in deps:
			if dep_id in cycle_ids:
				cycle_dep = dep_id
				break

		assert cycle_dep is not None
		deps.discard(cycle_dep)
		worst_unit.depends_on = ",".join(sorted(deps)) if deps else ""
		logger.info(
			"Cycle broken: removed dependency %s -> %s from %s (priority=%d)",
			worst_unit.id, cycle_dep, worst_unit.title[:40], worst_unit.priority,
		)
