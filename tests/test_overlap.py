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


class TestMutatesInPlace:
	def test_returns_same_list(self) -> None:
		"""resolve_file_overlaps returns the same list object."""
		units = [_wu("A", "a.py"), _wu("B", "b.py")]
		result = resolve_file_overlaps(units)
		assert result is units
