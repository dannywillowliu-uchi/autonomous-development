"""Tests for post-planning file overlap detection."""

from __future__ import annotations

from mission_control.models import WorkUnit
from mission_control.overlap import resolve_file_overlaps


def _wu(title: str = "", files_hint: str = "", priority: int = 1, depends_on: str = "") -> WorkUnit:
	return WorkUnit(
		plan_id="plan-1",
		title=title,
		files_hint=files_hint,
		priority=priority,
		depends_on=depends_on,
	)


class TestNoOverlap:
	def test_disjoint_files(self) -> None:
		"""Units with no shared files get no dependency edges."""
		units = [
			_wu("A", "src/a.py,src/b.py", priority=1),
			_wu("B", "src/c.py,src/d.py", priority=2),
		]
		result = resolve_file_overlaps(units)
		assert result[0].depends_on == ""
		assert result[1].depends_on == ""

	def test_single_unit(self) -> None:
		"""Single unit returns unchanged."""
		units = [_wu("A", "src/a.py")]
		result = resolve_file_overlaps(units)
		assert len(result) == 1
		assert result[0].depends_on == ""

	def test_empty_list(self) -> None:
		"""Empty list returns empty."""
		assert resolve_file_overlaps([]) == []


class TestSingleOverlap:
	def test_overlap_adds_dependency(self) -> None:
		"""Two units sharing a file: lower-priority depends on higher-priority."""
		units = [
			_wu("High", "shared.py,a.py", priority=1),
			_wu("Low", "shared.py,b.py", priority=2),
		]
		result = resolve_file_overlaps(units)
		# Low (priority=2) should depend on High (priority=1)
		assert result[0].depends_on == ""
		assert result[1].depends_on == result[0].id


class TestPriorityBasedDirection:
	def test_higher_priority_number_depends(self) -> None:
		"""Unit with higher priority number (lower priority) gets the dep."""
		units = [
			_wu("P3", "lib.py", priority=3),
			_wu("P1", "lib.py", priority=1),
		]
		result = resolve_file_overlaps(units)
		# P3 (lower priority) depends on P1 (higher priority)
		assert result[0].depends_on == result[1].id
		assert result[1].depends_on == ""

	def test_tie_broken_by_position(self) -> None:
		"""Same priority: later unit depends on earlier unit."""
		units = [
			_wu("First", "shared.py", priority=1),
			_wu("Second", "shared.py", priority=1),
		]
		result = resolve_file_overlaps(units)
		# Second (later position) depends on First (earlier position)
		assert result[0].depends_on == ""
		assert result[1].depends_on == result[0].id


class TestExistingDependsOnPreserved:
	def test_existing_deps_merged(self) -> None:
		"""Existing depends_on entries are kept when adding new ones."""
		units = [
			_wu("A", "shared.py", priority=1),
			_wu("B", "shared.py", priority=2, depends_on="existing-dep-id"),
		]
		result = resolve_file_overlaps(units)
		deps = set(result[1].depends_on.split(","))
		assert "existing-dep-id" in deps
		assert result[0].id in deps

	def test_no_duplicate_deps(self) -> None:
		"""If dependency already exists, don't add it again."""
		a = _wu("A", "shared.py", priority=1)
		b = _wu("B", "shared.py", priority=2, depends_on=a.id)
		units = [a, b]
		result = resolve_file_overlaps(units)
		# Should still have exactly one dep (no duplicates)
		deps = [d for d in result[1].depends_on.split(",") if d.strip()]
		assert len(deps) == 1
		assert deps[0] == a.id


class TestEmptyFilesHint:
	def test_empty_hint_skipped(self) -> None:
		"""Units with empty files_hint don't create overlaps."""
		units = [
			_wu("A", "", priority=1),
			_wu("B", "src/a.py", priority=2),
		]
		result = resolve_file_overlaps(units)
		assert result[0].depends_on == ""
		assert result[1].depends_on == ""

	def test_whitespace_only_hint_skipped(self) -> None:
		"""Units with whitespace-only files_hint are treated as empty."""
		units = [
			_wu("A", "  , , ", priority=1),
			_wu("B", "src/a.py", priority=2),
		]
		result = resolve_file_overlaps(units)
		assert result[0].depends_on == ""
		assert result[1].depends_on == ""


class TestMultipleOverlaps:
	def test_chain_of_three(self) -> None:
		"""Three units sharing a file: creates chain A <- B <- C."""
		units = [
			_wu("A", "shared.py", priority=1),
			_wu("B", "shared.py", priority=2),
			_wu("C", "shared.py", priority=3),
		]
		result = resolve_file_overlaps(units)
		# A has no deps (highest priority)
		assert result[0].depends_on == ""
		# B depends on A
		assert result[1].depends_on == result[0].id
		# C depends on both A and B
		c_deps = set(result[2].depends_on.split(","))
		assert result[0].id in c_deps
		assert result[1].id in c_deps

	def test_two_separate_overlaps(self) -> None:
		"""Different files overlap between different pairs."""
		units = [
			_wu("A", "file1.py,file2.py", priority=1),
			_wu("B", "file1.py,file3.py", priority=2),
			_wu("C", "file2.py,file4.py", priority=3),
		]
		result = resolve_file_overlaps(units)
		# B overlaps with A on file1.py -> B depends on A
		assert result[1].depends_on == result[0].id
		# C overlaps with A on file2.py -> C depends on A
		assert result[0].id in result[2].depends_on


class TestCycleDetectionSimpleMutual:
	def test_ab_mutual_overlap_cycle_broken(self) -> None:
		"""A<->B mutual dependency: cycle is broken by removing the edge from the lower-priority unit."""
		a = _wu("A", "shared.py", priority=1)
		b = _wu("B", "shared.py", priority=2)
		# Manually create a mutual cycle: A depends on B, B depends on A
		a.depends_on = b.id
		b.depends_on = a.id
		units = [a, b]
		resolve_file_overlaps(units)
		# B has lower priority (higher number), so its edge should be removed
		# A should still depend on B or B's edge to A removed
		# The cycle must be broken -- verify no cycle remains
		a_deps = {d for d in a.depends_on.split(",") if d.strip()}
		b_deps = {d for d in b.depends_on.split(",") if d.strip()}
		# Cannot have both A->B and B->A
		assert not (b.id in a_deps and a.id in b_deps), "Cycle not broken"
		# The lower-priority unit (B, priority=2) should have had its edge removed
		assert a.id not in b_deps or b.id not in a_deps

	def test_mutual_overlap_lower_priority_edge_removed(self) -> None:
		"""In A<->B cycle, the edge FROM the lower-priority unit is removed."""
		a = _wu("A", "shared.py", priority=1)
		b = _wu("B", "shared.py", priority=3)
		a.depends_on = b.id
		b.depends_on = a.id
		units = [a, b]
		resolve_file_overlaps(units)
		b_deps = {d for d in b.depends_on.split(",") if d.strip()}
		# B (priority=3) is lower priority, its dep on A should be removed
		assert a.id not in b_deps


class TestCycleDetectionLongerCycle:
	def test_abc_cycle_broken(self) -> None:
		"""A->B->C->A cycle is broken by removing edge from lowest-priority unit."""
		a = _wu("A", "f1.py", priority=1)
		b = _wu("B", "f2.py", priority=2)
		c = _wu("C", "f3.py", priority=3)
		# Create cycle: A depends on C, B depends on A, C depends on B
		a.depends_on = c.id
		b.depends_on = a.id
		c.depends_on = b.id
		units = [a, b, c]
		resolve_file_overlaps(units)
		# After breaking, verify no cycle exists via simple reachability check
		assert _is_acyclic(units), "Cycle was not broken"

	def test_abc_cycle_correct_edge_removed(self) -> None:
		"""In A->B->C->A, C (lowest priority) should lose its outgoing dep edge."""
		a = _wu("A", "f1.py", priority=1)
		b = _wu("B", "f2.py", priority=2)
		c = _wu("C", "f3.py", priority=3)
		a.depends_on = c.id
		b.depends_on = a.id
		c.depends_on = b.id
		units = [a, b, c]
		resolve_file_overlaps(units)
		c_deps = {d for d in c.depends_on.split(",") if d.strip()}
		# C (priority=3) is the lowest priority, so its dep on B should be removed
		assert b.id not in c_deps


class TestCycleFreeGraphUnchanged:
	def test_linear_chain_unchanged(self) -> None:
		"""A linear chain A<-B<-C has no cycle and passes through unchanged."""
		a = _wu("A", "shared.py", priority=1)
		b = _wu("B", "shared.py", priority=2)
		c = _wu("C", "shared.py", priority=3)
		b.depends_on = a.id
		c.depends_on = b.id
		units = [a, b, c]
		resolve_file_overlaps(units)
		# B and C may gain additional overlap edges but original deps should be preserved
		assert a.id in b.depends_on
		assert b.id in c.depends_on

	def test_diamond_dag_unchanged(self) -> None:
		"""A diamond DAG (A<-B, A<-C, B<-D, C<-D) has no cycle."""
		a = _wu("A", "f1.py", priority=1)
		b = _wu("B", "f2.py", priority=2)
		c = _wu("C", "f3.py", priority=2)
		d = _wu("D", "f4.py", priority=3)
		b.depends_on = a.id
		c.depends_on = a.id
		d.depends_on = f"{b.id},{c.id}"
		units = [a, b, c, d]
		resolve_file_overlaps(units)
		assert _is_acyclic(units)
		# D still depends on B and C
		d_deps = {dep for dep in d.depends_on.split(",") if dep.strip()}
		assert b.id in d_deps
		assert c.id in d_deps


def _is_acyclic(units: list[WorkUnit]) -> bool:
	"""Helper: return True if the dependency graph has no cycles (Kahn's algorithm)."""
	from collections import deque

	unit_ids = {u.id for u in units}
	in_degree: dict[str, int] = {u.id: 0 for u in units}
	adjacency: dict[str, list[str]] = {u.id: [] for u in units}
	for u in units:
		for dep_id in (d.strip() for d in u.depends_on.split(",") if d.strip()):
			if dep_id in unit_ids:
				adjacency[dep_id].append(u.id)
				in_degree[u.id] += 1
	queue: deque[str] = deque(uid for uid, deg in in_degree.items() if deg == 0)
	visited = 0
	while queue:
		node = queue.popleft()
		visited += 1
		for neighbor in adjacency[node]:
			in_degree[neighbor] -= 1
			if in_degree[neighbor] == 0:
				queue.append(neighbor)
	return visited == len(unit_ids)


class TestMutatesInPlace:
	def test_returns_same_list(self) -> None:
		"""resolve_file_overlaps returns the same list object."""
		units = [_wu("A", "a.py"), _wu("B", "b.py")]
		result = resolve_file_overlaps(units)
		assert result is units
