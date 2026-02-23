"""Tests for project health snapshots."""

from __future__ import annotations

from mission_control.models import Snapshot, VerificationNodeKind
from mission_control.state import (
	_build_result_from_single_command,
	_parse_mypy,
	_parse_pytest,
	_parse_ruff,
	_tool_detected,
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


class TestToolDetected:
	def test_pytest_detected(self) -> None:
		assert _tool_detected("pytest", "10 passed in 0.5s")
		assert _tool_detected("pytest", "3 failed, 2 passed in 1.0s")
		assert _tool_detected("pytest", "1 error in 0.1s")
		assert _tool_detected("pytest", "no tests ran")

	def test_pytest_not_detected(self) -> None:
		assert not _tool_detected("pytest", "")
		assert not _tool_detected("pytest", "All checks passed!")

	def test_ruff_detected(self) -> None:
		assert _tool_detected("ruff", "All checks passed!")
		assert _tool_detected("ruff", "Found 2 errors.")
		assert _tool_detected("ruff", "src/foo.py:10:1: E501 Line too long")

	def test_ruff_not_detected(self) -> None:
		assert not _tool_detected("ruff", "")
		assert not _tool_detected("ruff", "10 passed in 0.5s")

	def test_mypy_detected(self) -> None:
		assert _tool_detected("mypy", "Success: no issues found in 5 source files")
		assert _tool_detected("mypy", "src/foo.py:10: error: Incompatible types")

	def test_mypy_not_detected(self) -> None:
		assert not _tool_detected("mypy", "")
		assert not _tool_detected("mypy", "10 passed in 0.5s")

	def test_bandit_detected(self) -> None:
		assert _tool_detected("bandit", "No issues identified.")
		assert _tool_detected("bandit", ">> Issue: [B101:assert_used]")
		assert _tool_detected("bandit", "Run started:2024-01-01")

	def test_bandit_not_detected(self) -> None:
		assert not _tool_detected("bandit", "")
		assert not _tool_detected("bandit", "10 passed in 0.5s")

	def test_unknown_kind(self) -> None:
		assert not _tool_detected("unknown", "anything")


class TestBuildResultSkipped:
	"""Tests for skipped node detection in _build_result_from_single_command."""

	PYTEST_RUFF_OUTPUT = "10 passed in 0.5s\nAll checks passed!"

	def test_pytest_ruff_marks_those_two_passed(self) -> None:
		"""(a) Combined pytest+ruff output marks those two as passed."""
		report = _build_result_from_single_command(self.PYTEST_RUFF_OUTPUT, returncode=0)
		by_kind = {r.kind: r for r in report.results}

		assert by_kind[VerificationNodeKind.PYTEST].passed is True
		assert by_kind[VerificationNodeKind.PYTEST].skipped is False
		assert by_kind[VerificationNodeKind.RUFF].passed is True
		assert by_kind[VerificationNodeKind.RUFF].skipped is False

	def test_mypy_bandit_marked_skipped(self) -> None:
		"""(b) mypy/bandit nodes are marked skipped when only pytest+ruff ran."""
		report = _build_result_from_single_command(self.PYTEST_RUFF_OUTPUT, returncode=0)
		by_kind = {r.kind: r for r in report.results}

		assert by_kind[VerificationNodeKind.MYPY].skipped is True
		assert by_kind[VerificationNodeKind.MYPY].passed is False
		assert by_kind[VerificationNodeKind.BANDIT].skipped is True
		assert by_kind[VerificationNodeKind.BANDIT].passed is False

	def test_overall_passed_with_skipped_nodes(self) -> None:
		"""(c) overall_passed ignores skipped nodes -- only considers nodes that ran."""
		report = _build_result_from_single_command(self.PYTEST_RUFF_OUTPUT, returncode=0)
		assert report.overall_passed is True

	def test_overall_passed_fails_when_ran_node_fails(self) -> None:
		"""overall_passed is False when a non-skipped required node fails."""
		output = "3 passed, 1 failed in 0.5s\nAll checks passed!"
		report = _build_result_from_single_command(output, returncode=1)
		assert report.overall_passed is False

	def test_weighted_score_excludes_skipped(self) -> None:
		"""weighted_score only counts nodes that actually ran."""
		report = _build_result_from_single_command(self.PYTEST_RUFF_OUTPUT, returncode=0)
		# 2 nodes ran and passed, each weight=1.0 → score=2.0
		assert report.weighted_score == 2.0

	def test_failed_kinds_excludes_skipped(self) -> None:
		"""failed_kinds should not include skipped nodes."""
		report = _build_result_from_single_command(self.PYTEST_RUFF_OUTPUT, returncode=0)
		assert VerificationNodeKind.MYPY not in report.failed_kinds()
		assert VerificationNodeKind.BANDIT not in report.failed_kinds()

	def test_all_four_tools_detected(self) -> None:
		"""When all four tools produce output, none are skipped."""
		output = (
			"10 passed in 0.5s\n"
			"All checks passed!\n"
			"Success: no issues found in 5 source files\n"
			"No issues identified.\n"
		)
		report = _build_result_from_single_command(output, returncode=0)
		for r in report.results:
			assert r.skipped is False, f"{r.kind} should not be skipped"
			assert r.passed is True, f"{r.kind} should be passed"

	def test_nonzero_returncode_skipped_nodes(self) -> None:
		"""When command fails partway, undetected tools are still skipped."""
		# pytest failed, ruff never ran
		output = "3 passed, 2 failed in 0.5s"
		report = _build_result_from_single_command(output, returncode=1)
		by_kind = {r.kind: r for r in report.results}

		assert by_kind[VerificationNodeKind.PYTEST].skipped is False
		assert by_kind[VerificationNodeKind.RUFF].skipped is True
		assert by_kind[VerificationNodeKind.MYPY].skipped is True
		assert by_kind[VerificationNodeKind.BANDIT].skipped is True

	def test_no_tools_detected_falls_back_to_returncode(self) -> None:
		"""When no tool markers are found (custom script), nothing is skipped."""
		report = _build_result_from_single_command("", returncode=0)
		for r in report.results:
			assert r.skipped is False, f"{r.kind} should not be skipped with empty output"
			assert r.passed is True, f"{r.kind} should be passed when returncode=0"
