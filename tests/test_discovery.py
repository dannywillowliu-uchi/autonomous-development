"""Tests for task discovery."""

from __future__ import annotations

from mission_control.config import MissionConfig, TargetConfig
from mission_control.discovery import discover_from_snapshot, discover_todos_from_output
from mission_control.models import Session, Snapshot


def _config(objective: str = "") -> MissionConfig:
	return MissionConfig(target=TargetConfig(name="test", path="/tmp/test", objective=objective))


class TestDiscoverFromSnapshot:
	def test_failing_tests(self) -> None:
		snap = Snapshot(test_total=10, test_passed=8, test_failed=2)
		tasks = discover_from_snapshot(snap, _config())
		assert len(tasks) == 1
		assert tasks[0].source == "test_failure"
		assert tasks[0].priority == 2

	def test_lint_errors(self) -> None:
		snap = Snapshot(test_total=10, test_passed=10, test_failed=0, lint_errors=5)
		tasks = discover_from_snapshot(snap, _config())
		assert len(tasks) == 1
		assert tasks[0].source == "lint"
		assert tasks[0].priority == 3

	def test_type_errors(self) -> None:
		snap = Snapshot(test_total=10, test_passed=10, test_failed=0, type_errors=3)
		tasks = discover_from_snapshot(snap, _config())
		assert len(tasks) == 1
		assert tasks[0].source == "type_error"
		assert tasks[0].priority == 4

	def test_regression_highest_priority(self) -> None:
		before = Snapshot(test_total=10, test_passed=10, test_failed=0)
		snap = Snapshot(test_total=10, test_passed=8, test_failed=2, lint_errors=3)
		tasks = discover_from_snapshot(snap, _config(), previous_snapshot=before)
		assert tasks[0].source == "regression"
		assert tasks[0].priority == 1

	def test_multiple_issues_sorted(self) -> None:
		snap = Snapshot(test_total=10, test_passed=8, test_failed=2, lint_errors=5, type_errors=3)
		tasks = discover_from_snapshot(snap, _config())
		priorities = [t.priority for t in tasks]
		assert priorities == sorted(priorities)

	def test_clean_project_with_objective(self) -> None:
		snap = Snapshot(test_total=10, test_passed=10, test_failed=0)
		tasks = discover_from_snapshot(snap, _config(objective="Build API"))
		assert len(tasks) == 1
		assert tasks[0].source == "objective"
		assert tasks[0].priority == 7

	def test_clean_project_no_objective(self) -> None:
		snap = Snapshot(test_total=10, test_passed=10, test_failed=0)
		tasks = discover_from_snapshot(snap, _config())
		assert len(tasks) == 0

	def test_objective_not_duplicated_from_recent(self) -> None:
		snap = Snapshot(test_total=10, test_passed=10, test_failed=0)
		recent = [Session(task_description="Work toward objective: Build API")]
		tasks = discover_from_snapshot(snap, _config(objective="Build API"), recent_sessions=recent)
		assert len(tasks) == 0

	def test_security_findings(self) -> None:
		snap = Snapshot(test_total=10, test_passed=10, test_failed=0, security_findings=2)
		tasks = discover_from_snapshot(snap, _config())
		assert len(tasks) == 1
		assert tasks[0].source == "security"

	def test_test_failure_suppressed_when_regression_exists(self) -> None:
		"""Line 42 branch: test_failure task is skipped when regression already covers failures."""
		before = Snapshot(test_total=10, test_passed=10, test_failed=0)
		snap = Snapshot(test_total=10, test_passed=7, test_failed=3)
		tasks = discover_from_snapshot(snap, _config(), previous_snapshot=before)
		sources = [t.source for t in tasks]
		assert "regression" in sources
		assert "test_failure" not in sources

	def test_objective_reappears_outside_lookback_window(self) -> None:
		"""Objective should re-appear when matching session is outside the lookback window."""
		snap = Snapshot(test_total=10, test_passed=10, test_failed=0)
		# Default lookback is 5. Put the objective session first, then 5 unrelated sessions
		# so the objective falls outside the window.
		old = Session(task_description="Work toward objective: Build API")
		filler = [Session(task_description=f"Unrelated task {i}") for i in range(5)]
		recent = [old] + filler  # old is at index 0, window is last 5 => excludes old
		tasks = discover_from_snapshot(snap, _config(objective="Build API"), recent_sessions=recent)
		assert len(tasks) == 1
		assert tasks[0].source == "objective"

	def test_regression_from_nonzero_baseline(self) -> None:
		"""Regression detected when previous had failures and current has MORE failures."""
		before = Snapshot(test_total=10, test_passed=8, test_failed=2)
		snap = Snapshot(test_total=10, test_passed=5, test_failed=5)
		tasks = discover_from_snapshot(snap, _config(), previous_snapshot=before)
		sources = [t.source for t in tasks]
		assert "regression" in sources
		regression = next(t for t in tasks if t.source == "regression")
		assert "3" in regression.description  # 5 - 2 = 3 new regressions
		# test_failure should be suppressed since regression covers it
		assert "test_failure" not in sources


class TestDiscoverTodos:
	def test_parse_todos(self) -> None:
		output = """\
src/foo.py:10: # TODO: Implement caching
src/bar.py:20: # FIXME: Handle edge case
src/baz.py:30: # TODO: Add logging
"""
		tasks = discover_todos_from_output(output)
		assert len(tasks) == 3
		assert all(t.source == "todo" for t in tasks)
		assert all(t.priority == 5 for t in tasks)

	def test_empty_output(self) -> None:
		tasks = discover_todos_from_output("")
		assert len(tasks) == 0

	def test_deduplicates(self) -> None:
		output = """\
src/a.py:1: # TODO: Fix this
src/b.py:2: # TODO: Fix this
"""
		tasks = discover_todos_from_output(output)
		assert len(tasks) == 1
