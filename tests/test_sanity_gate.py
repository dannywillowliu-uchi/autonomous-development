"""Tests for the pre-merge diff sanity gate."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mission_control.config import GreenBranchConfig, MissionConfig, TargetConfig, VerificationConfig
from mission_control.db import Database
from mission_control.green_branch import (
	MAX_DIFF_LINES,
	GreenBranchManager,
	_is_test_file,
	_sanity_check_diff,
)
from mission_control.models import WorkUnit

# -- Helpers --

def _make_diff(*files: str, lines_per_file: int = 10) -> str:
	"""Build a fake unified diff touching the given file paths."""
	parts: list[str] = []
	for f in files:
		parts.append(f"diff --git a/{f} b/{f}")
		parts.append(f"--- a/{f}")
		parts.append(f"+++ b/{f}")
		parts.append("@@ -1,3 +1,5 @@")
		for i in range(lines_per_file):
			parts.append(f"+added line {i}")
	return "\n".join(parts)


def _unit(
	files_hint: str = "",
	unit_type: str = "implementation",
) -> WorkUnit:
	return WorkUnit(files_hint=files_hint, unit_type=unit_type)


# -- _sanity_check_diff unit tests --

class TestSanityCheckDiffEmpty:
	def test_empty_string_rejected(self) -> None:
		result = _sanity_check_diff("", None)
		assert not result.passed
		assert "empty" in result.rejection_reason.lower()

	def test_whitespace_only_rejected(self) -> None:
		result = _sanity_check_diff("   \n\n  ", None)
		assert not result.passed
		assert "empty" in result.rejection_reason.lower()


class TestSanityCheckDiffNonEmpty:
	def test_valid_diff_passes(self) -> None:
		diff = _make_diff("src/foo.py")
		result = _sanity_check_diff(diff, None)
		assert result.passed
		assert result.rejection_reason == ""

	def test_valid_diff_with_unit_passes(self) -> None:
		diff = _make_diff("src/mission_control/green_branch.py")
		unit = _unit(files_hint="src/mission_control/green_branch.py")
		result = _sanity_check_diff(diff, unit)
		assert result.passed
		assert not result.warnings


class TestSanityCheckDiffFilesHintOverlap:
	def test_no_overlap_warns(self) -> None:
		diff = _make_diff("src/other_module.py")
		unit = _unit(files_hint="src/mission_control/green_branch.py")
		result = _sanity_check_diff(diff, unit)
		assert result.passed  # warning, not rejection
		assert len(result.warnings) == 1
		assert "overlap" in result.warnings[0].lower()

	def test_directory_prefix_overlap_ok(self) -> None:
		diff = _make_diff("src/mission_control/green_branch.py")
		unit = _unit(files_hint="src/mission_control/")
		result = _sanity_check_diff(diff, unit)
		assert result.passed
		assert not result.warnings

	def test_multiple_hints_one_matches(self) -> None:
		diff = _make_diff("src/worker.py")
		unit = _unit(files_hint="src/planner.py, src/worker.py")
		result = _sanity_check_diff(diff, unit)
		assert result.passed
		assert not result.warnings

	def test_empty_files_hint_no_warning(self) -> None:
		diff = _make_diff("src/foo.py")
		unit = _unit(files_hint="")
		result = _sanity_check_diff(diff, unit)
		assert result.passed
		assert not result.warnings


class TestSanityCheckDiffTestOnly:
	def test_test_only_diff_rejected_for_implementation(self) -> None:
		diff = _make_diff("tests/test_foo.py", "tests/test_bar.py")
		unit = _unit(unit_type="implementation")
		result = _sanity_check_diff(diff, unit)
		assert not result.passed
		assert "test files" in result.rejection_reason.lower()

	def test_test_only_diff_allowed_for_test_writer(self) -> None:
		diff = _make_diff("tests/test_foo.py")
		unit = _unit(unit_type="test-writer")
		result = _sanity_check_diff(diff, unit)
		assert result.passed

	def test_mixed_diff_passes_for_implementation(self) -> None:
		diff = _make_diff("src/foo.py", "tests/test_foo.py")
		unit = _unit(unit_type="implementation")
		result = _sanity_check_diff(diff, unit)
		assert result.passed

	def test_test_only_rejected_for_research_type(self) -> None:
		diff = _make_diff("tests/test_analysis.py")
		unit = _unit(unit_type="research")
		result = _sanity_check_diff(diff, unit)
		assert not result.passed


class TestSanityCheckDiffSize:
	def test_oversized_diff_rejected(self) -> None:
		# Each file gets ~14 lines in _make_diff with lines_per_file=10.
		# We need >5000 total lines.
		big_diff = "\n".join([f"+line {i}" for i in range(MAX_DIFF_LINES + 100)])
		diff = f"diff --git a/src/big.py b/src/big.py\n{big_diff}"
		result = _sanity_check_diff(diff, None)
		assert not result.passed
		assert "too large" in result.rejection_reason.lower()

	def test_diff_at_limit_passes(self) -> None:
		# Build a diff with exactly MAX_DIFF_LINES lines
		lines = ["diff --git a/src/ok.py b/src/ok.py"]
		lines.extend([f"+line {i}" for i in range(MAX_DIFF_LINES - 1)])
		diff = "\n".join(lines)
		assert len(diff.splitlines()) == MAX_DIFF_LINES
		result = _sanity_check_diff(diff, None)
		assert result.passed


class TestIsTestFile:
	@pytest.mark.parametrize("path", [
		"tests/test_foo.py",
		"test_bar.py",
		"src/tests/test_util.py",
		"something_test.py",
	])
	def test_detected_as_test(self, path: str) -> None:
		assert _is_test_file(path)

	@pytest.mark.parametrize("path", [
		"src/foo.py",
		"src/mission_control/worker.py",
		"lib/utils.py",
	])
	def test_not_detected_as_test(self, path: str) -> None:
		assert not _is_test_file(path)


class TestSanityCheckNoUnit:
	def test_no_unit_skips_overlap_and_type_checks(self) -> None:
		diff = _make_diff("tests/test_only.py")
		result = _sanity_check_diff(diff, None)
		assert result.passed
		assert not result.warnings


# -- Integration: merge_unit with sanity gate --

def _config() -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(
		name="test",
		path="/tmp/test",
		branch="main",
		verification=VerificationConfig(command="pytest -q"),
	)
	mc.green_branch = GreenBranchConfig(
		working_branch="mc/working",
		green_branch="mc/green",
	)
	mc.continuous.verify_before_merge = False
	return mc


def _manager() -> GreenBranchManager:
	config = _config()
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


class TestMergeUnitSanityGate:
	async def test_empty_diff_blocks_merge(self) -> None:
		mgr = _manager()

		async def fake_run_git(*args: str) -> tuple[bool, str]:
			if args[0] == "remote":
				return (True, "")
			if args[0] == "fetch":
				return (True, "")
			if args[0] == "diff":
				return (True, "")  # empty diff
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=fake_run_git)

		result = await mgr.merge_unit("/tmp/ws", "mc/unit-abc", unit=_unit())
		assert not result.merged
		assert result.failure_stage == "sanity_check"
		assert "empty" in result.failure_output.lower()

	async def test_oversized_diff_blocks_merge(self) -> None:
		mgr = _manager()
		big_diff = "diff --git a/src/x.py b/src/x.py\n" + "\n".join(
			[f"+line {i}" for i in range(MAX_DIFF_LINES + 100)]
		)

		async def fake_run_git(*args: str) -> tuple[bool, str]:
			if args[0] == "remote":
				return (True, "")
			if args[0] == "fetch":
				return (True, "")
			if args[0] == "diff":
				return (True, big_diff)
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=fake_run_git)

		result = await mgr.merge_unit("/tmp/ws", "mc/unit-big", unit=_unit())
		assert not result.merged
		assert result.failure_stage == "sanity_check"
		assert "too large" in result.failure_output.lower()

	async def test_passing_sanity_proceeds_to_merge(self) -> None:
		mgr = _manager()
		valid_diff = _make_diff("src/foo.py")

		async def fake_run_git(*args: str) -> tuple[bool, str]:
			if args[0] == "remote":
				return (True, "")
			if args[0] == "fetch":
				return (True, "")
			if args[0] == "diff" and "..." in args[1]:
				return (True, valid_diff)
			if args[0] == "checkout":
				return (True, "")
			if args[0] == "reset":
				return (True, "")
			if args[0] == "clean":
				return (True, "")
			if args[0] == "branch":
				return (True, "")
			if args[0] == "rebase":
				return (True, "")
			if args[0] == "merge":
				return (True, "")
			if args[0] == "rev-parse":
				return (True, "abc123\n")
			if args[0] == "diff":
				return (True, "src/foo.py\n")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=fake_run_git)
		mgr._sync_to_source = AsyncMock(return_value=True)

		result = await mgr.merge_unit(
			"/tmp/ws", "mc/unit-ok", unit=_unit(files_hint="src/foo.py"),
		)
		assert result.merged
