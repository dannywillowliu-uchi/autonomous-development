"""Tests for project health snapshots."""

from __future__ import annotations

from mission_control.models import Snapshot
from mission_control.state import (
	_parse_mypy,
	_parse_pytest,
	_parse_ruff,
	compare_snapshots,
)


class TestParsePytest:
	def test_all_passing(self) -> None:
		output = "10 passed in 0.05s"
		result = _parse_pytest(output)
		assert result["test_total"] == 10
		assert result["test_passed"] == 10
		assert result["test_failed"] == 0

	def test_mixed_results(self) -> None:
		output = "8 passed, 2 failed in 0.10s"
		result = _parse_pytest(output)
		assert result["test_total"] == 10
		assert result["test_passed"] == 8
		assert result["test_failed"] == 2

	def test_all_failing(self) -> None:
		output = "5 failed in 0.03s"
		result = _parse_pytest(output)
		assert result["test_total"] == 5
		assert result["test_passed"] == 0
		assert result["test_failed"] == 5

	def test_with_errors(self) -> None:
		output = "3 passed, 1 failed, 1 error in 0.08s"
		result = _parse_pytest(output)
		assert result["test_total"] == 5
		assert result["test_passed"] == 3
		# errors are included in test_failed count
		assert result["test_failed"] == 2

	def test_errors_only(self) -> None:
		"""Errors with no failures should still result in nonzero test_failed."""
		output = "2 passed, 1 error in 0.05s"
		result = _parse_pytest(output)
		assert result["test_total"] == 3
		assert result["test_passed"] == 2
		assert result["test_failed"] == 1

	def test_no_tests(self) -> None:
		output = "no tests ran"
		result = _parse_pytest(output)
		assert result["test_total"] == 0
		assert result["test_passed"] == 0

	def test_empty_output(self) -> None:
		result = _parse_pytest("")
		assert result["test_total"] == 0


class TestParseRuff:
	def test_clean(self) -> None:
		result = _parse_ruff("All checks passed!")
		assert result["lint_errors"] == 0

	def test_empty_output(self) -> None:
		result = _parse_ruff("")
		assert result["lint_errors"] == 0

	def test_errors(self) -> None:
		output = """\
src/foo.py:10:1: E501 Line too long (130 > 120)
src/foo.py:20:1: F401 'os' imported but unused
Found 2 errors."""
		result = _parse_ruff(output)
		assert result["lint_errors"] == 2

	def test_single_error(self) -> None:
		output = "src/bar.py:5:8: F401 unused import\nFound 1 error."
		result = _parse_ruff(output)
		assert result["lint_errors"] == 1


class TestParseMypy:
	def test_success(self) -> None:
		result = _parse_mypy("Success: no issues found in 5 source files")
		assert result["type_errors"] == 0

	def test_errors(self) -> None:
		output = """\
src/foo.py:10: error: Incompatible types
src/foo.py:20: error: Missing return statement
Found 2 errors in 1 file"""
		result = _parse_mypy(output)
		assert result["type_errors"] == 2

	def test_empty(self) -> None:
		result = _parse_mypy("")
		assert result["type_errors"] == 0

	def test_no_false_positives_from_pytest_output(self) -> None:
		"""Pytest tracebacks with 'error:' should not be counted as mypy errors."""
		output = """\
FAILED tests/test_foo.py::test_bar - ValueError: error: something broke
E   ValueError: error: unexpected
tests/test_baz.py::test_qux PASSED
3 passed, 1 failed in 0.5s"""
		result = _parse_mypy(output)
		assert result["type_errors"] == 0

	def test_combined_pytest_and_mypy_output(self) -> None:
		"""Only mypy-format lines should be counted, not pytest tracebacks."""
		output = """\
10 passed, 1 failed in 0.5s
FAILED tests/test_foo.py::test_bar - AssertionError: error: mismatch
src/mission_control/db.py:10: error: Incompatible types
src/mission_control/worker.py:20: error: Missing return
Found 2 errors in 2 files"""
		result = _parse_mypy(output)
		assert result["type_errors"] == 2


class TestCompareSnapshots:
	def test_improvement(self) -> None:
		before = Snapshot(test_total=10, test_passed=8, test_failed=2, lint_errors=5)
		after = Snapshot(test_total=10, test_passed=10, test_failed=0, lint_errors=3)
		delta = compare_snapshots(before, after)
		assert delta.tests_fixed == 2
		assert delta.tests_broken == 0
		assert delta.lint_delta == -2
		assert delta.improved is True
		assert delta.regressed is False

	def test_regression(self) -> None:
		before = Snapshot(test_total=10, test_passed=10, test_failed=0)
		after = Snapshot(test_total=10, test_passed=8, test_failed=2)
		delta = compare_snapshots(before, after)
		assert delta.tests_broken == 2
		assert delta.regressed is True

	def test_neutral(self) -> None:
		before = Snapshot(test_total=10, test_passed=10, test_failed=0)
		after = Snapshot(test_total=10, test_passed=10, test_failed=0)
		delta = compare_snapshots(before, after)
		assert delta.improved is False
		assert delta.regressed is False

	def test_security_regression(self) -> None:
		before = Snapshot(security_findings=0)
		after = Snapshot(security_findings=2)
		delta = compare_snapshots(before, after)
		assert delta.regressed is True

	def test_mixed_lint_improvement_with_test_break(self) -> None:
		before = Snapshot(test_total=10, test_passed=10, test_failed=0, lint_errors=10)
		after = Snapshot(test_total=10, test_passed=9, test_failed=1, lint_errors=0)
		delta = compare_snapshots(before, after)
		assert delta.regressed is True
		assert delta.improved is False


class TestRunCommandTimeout:
	async def test_timeout_kills_subprocess(self) -> None:
		"""Timed-out subprocesses should be killed, not left as zombies."""
		from unittest.mock import AsyncMock, MagicMock, patch

		from mission_control.state import _run_command

		mock_proc = AsyncMock()
		mock_proc.kill = MagicMock()  # kill() is synchronous
		mock_proc.wait = AsyncMock()

		import asyncio

		with patch("mission_control.state.asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch("mission_control.state.asyncio.wait_for", side_effect=asyncio.TimeoutError):
				result = await _run_command("sleep 999", "/tmp", timeout=1)

		assert result["returncode"] == -1
		assert "timed out" in result["output"]
		# Verify process was killed
		mock_proc.kill.assert_called_once()
		mock_proc.wait.assert_called_once()
