"""Tests for green branch conflict handling: fixup candidates and file conflict resolution."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import (
	GreenBranchConfig,
	MissionConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.continuous_controller import (
	ContinuousController,
	ContinuousMissionResult,
	WorkerCompletion,
)
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.green_branch import (
	FIXUP_PROMPTS,
	FixupCandidate,
	FixupResult,
	GreenBranchManager,
	UnitMergeResult,
)
from mission_control.models import Epoch, Handoff, Mission, Plan, WorkUnit

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fixup_config(fixup_candidates: int = 3) -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(
		name="test",
		path="/tmp/test",
		branch="main",
		verification=VerificationConfig(command="pytest -q && ruff check src/"),
	)
	mc.green_branch = GreenBranchConfig(
		working_branch="mc/working",
		green_branch="mc/green",
		fixup_candidates=fixup_candidates,
	)
	return mc


def _fixup_manager(fixup_candidates: int = 3) -> GreenBranchManager:
	config = _fixup_config(fixup_candidates)
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


def _candidate(
	index: int,
	passed: bool = False,
	tests: int = 0,
	lint: int = 0,
	diff: int = 0,
) -> FixupCandidate:
	return FixupCandidate(
		branch=f"mc/fixup-candidate-{index}",
		verification_passed=passed,
		tests_passed=tests,
		lint_errors=lint,
		diff_lines=diff,
	)


def _conflict_config() -> MissionConfig:
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
	return mc


def _conflict_manager() -> GreenBranchManager:
	config = _conflict_config()
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


# ---------------------------------------------------------------------------
# Fixup candidate dataclasses
# ---------------------------------------------------------------------------


class TestFixupCandidateDataclass:
	def test_defaults(self) -> None:
		c = FixupCandidate()
		assert c.branch == ""
		assert c.verification_passed is False
		assert c.tests_passed == 0
		assert c.lint_errors == 0
		assert c.diff_lines == 0

	def test_fields_set(self) -> None:
		c = FixupCandidate(
			branch="mc/fixup-candidate-0",
			verification_passed=True,
			tests_passed=42,
			lint_errors=2,
			diff_lines=15,
		)
		assert c.branch == "mc/fixup-candidate-0"
		assert c.verification_passed is True
		assert c.tests_passed == 42
		assert c.lint_errors == 2
		assert c.diff_lines == 15


class TestFixupResultDataclass:
	def test_defaults(self) -> None:
		r = FixupResult()
		assert r.success is False
		assert r.winner is None
		assert r.candidates == []

	def test_with_winner(self) -> None:
		winner = FixupCandidate(
			branch="mc/fixup-candidate-1", verification_passed=True,
		)
		r = FixupResult(success=True, winner=winner, candidates=[winner])
		assert r.success is True
		assert r.winner is winner


# ---------------------------------------------------------------------------
# Diff line counting
# ---------------------------------------------------------------------------


class TestCountDiffLines:
	def test_normal_stat(self) -> None:
		output = " 3 files changed, 10 insertions(+), 5 deletions(-)\n"
		assert GreenBranchManager._count_diff_lines(output) == 15

	def test_insertions_only(self) -> None:
		output = " 1 file changed, 7 insertions(+)\n"
		assert GreenBranchManager._count_diff_lines(output) == 7

	def test_deletions_only(self) -> None:
		output = " 2 files changed, 3 deletions(-)\n"
		assert GreenBranchManager._count_diff_lines(output) == 3

	def test_empty_output(self) -> None:
		assert GreenBranchManager._count_diff_lines("") == 0

	def test_no_changes(self) -> None:
		assert GreenBranchManager._count_diff_lines("no diff\n") == 0

	def test_multiline_stat(self) -> None:
		output = (
			" src/foo.py | 10 +++++++---\n"
			" src/bar.py |  5 ++---\n"
			" 2 files changed, 9 insertions(+), 6 deletions(-)\n"
		)
		assert GreenBranchManager._count_diff_lines(output) == 15


# ---------------------------------------------------------------------------
# N-of-M fixup candidate selection
# ---------------------------------------------------------------------------


class TestRunFixup:
	"""Tests for the N-of-M fixup candidate selection."""

	async def test_best_candidate_selected_by_tests(self) -> None:
		"""Candidate with most tests passing wins."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=True, tests=8, lint=0, diff=10),
			_candidate(1, passed=True, tests=10, lint=0, diff=10),
			_candidate(2, passed=True, tests=5, lint=0, diff=10),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert result.winner is not None
		assert result.winner.tests_passed == 10
		assert result.winner.branch == "mc/fixup-candidate-1"

	async def test_tiebreak_by_lint_errors(self) -> None:
		"""When tests_passed tie, fewer lint errors wins."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=5, diff=10),
			_candidate(1, passed=True, tests=10, lint=2, diff=10),
			_candidate(2, passed=True, tests=10, lint=8, diff=10),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert result.winner is not None
		assert result.winner.lint_errors == 2
		assert result.winner.branch == "mc/fixup-candidate-1"

	async def test_tiebreak_by_diff_size(self) -> None:
		"""When tests and lint tie, smallest diff wins."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=0, diff=50),
			_candidate(1, passed=True, tests=10, lint=0, diff=5),
			_candidate(2, passed=True, tests=10, lint=0, diff=30),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert result.winner is not None
		assert result.winner.diff_lines == 5
		assert result.winner.branch == "mc/fixup-candidate-1"

	async def test_all_candidates_fail(self) -> None:
		"""When all candidates fail verification, result is failure."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=False, tests=3, lint=5, diff=10),
			_candidate(1, passed=False, tests=5, lint=2, diff=8),
			_candidate(2, passed=False, tests=0, lint=10, diff=20),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)

		result = await mgr.run_fixup("some failure")

		assert result.success is False
		assert result.winner is None
		assert len(result.candidates) == 3

	async def test_single_pass(self) -> None:
		"""When only one candidate passes, it wins regardless of metrics."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=False, tests=0, lint=0, diff=0),
			_candidate(1, passed=True, tests=3, lint=5, diff=100),
			_candidate(2, passed=False, tests=0, lint=0, diff=0),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert result.winner is not None
		assert result.winner.branch == "mc/fixup-candidate-1"

	async def test_winner_merged_into_green(self) -> None:
		"""Winning candidate branch is merged into mc/green."""
		mgr = _fixup_manager(fixup_candidates=2)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=0, diff=5),
			_candidate(1, passed=False, tests=3, lint=2, diff=10),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			return (True, "")

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(side_effect=tracking_git)

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		ff_calls = [
			c for c in git_calls
			if c[0] == "merge" and "--ff-only" in c
		]
		assert len(ff_calls) == 1
		assert "mc/fixup-candidate-0" in ff_calls[0]

	async def test_candidate_branches_cleaned_up(self) -> None:
		"""All candidate branches are deleted after fixup."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=0, diff=5),
			_candidate(1, passed=False, tests=3, lint=2, diff=10),
			_candidate(2, passed=False, tests=0, lint=5, diff=20),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			return (True, "")

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(side_effect=tracking_git)

		await mgr.run_fixup("some failure")

		delete_calls = [
			c for c in git_calls if c[0] == "branch" and c[1] == "-D"
		]
		deleted_branches = {c[2] for c in delete_calls}
		assert "mc/fixup-candidate-0" in deleted_branches
		assert "mc/fixup-candidate-1" in deleted_branches
		assert "mc/fixup-candidate-2" in deleted_branches

	async def test_merge_failure_returns_no_winner(self) -> None:
		"""If merging the winner branch fails, result has no winner."""
		mgr = _fixup_manager(fixup_candidates=2)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=0, diff=5),
			_candidate(1, passed=False, tests=3, lint=2, diff=10),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		async def failing_merge(*args: str) -> tuple[bool, str]:
			if args[0] == "merge":
				return (False, "merge failed")
			return (True, "")

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(side_effect=failing_merge)

		result = await mgr.run_fixup("some failure")

		assert result.success is False
		assert result.winner is None

	async def test_configurable_candidate_count(self) -> None:
		"""fixup_candidates config controls how many candidates are spawned."""
		mgr = _fixup_manager(fixup_candidates=5)

		call_count = 0

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			nonlocal call_count
			call_count += 1
			return FixupCandidate(
				branch=f"mc/fixup-candidate-{index}",
				verification_passed=(index == 0),
				tests_passed=10 if index == 0 else 3,
				lint_errors=0,
				diff_lines=5,
			)

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert call_count == 5
		assert len(result.candidates) == 5

	async def test_prompts_cycle_when_n_exceeds_prompts(self) -> None:
		"""When fixup_candidates > len(FIXUP_PROMPTS), prompts padded."""
		mgr = _fixup_manager(fixup_candidates=5)

		prompts_received: list[str] = []

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			prompts_received.append(prompt)
			return FixupCandidate(
				branch=f"mc/fixup-candidate-{index}",
				verification_passed=False,
			)

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)

		await mgr.run_fixup("some failure")

		assert len(prompts_received) == 5
		assert prompts_received[0] == FIXUP_PROMPTS[0]
		assert prompts_received[1] == FIXUP_PROMPTS[1]
		assert prompts_received[2] == FIXUP_PROMPTS[2]
		# Extra candidates get the first prompt
		assert prompts_received[3] == FIXUP_PROMPTS[0]
		assert prompts_received[4] == FIXUP_PROMPTS[0]

	async def test_only_passing_candidates_considered(self) -> None:
		"""Failed candidates never selected even with better metrics."""
		mgr = _fixup_manager(fixup_candidates=3)

		candidates = [
			_candidate(0, passed=False, tests=100, lint=0, diff=1),
			_candidate(1, passed=True, tests=5, lint=3, diff=50),
			_candidate(2, passed=False, tests=50, lint=0, diff=2),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(return_value=(True, ""))

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert result.winner is not None
		assert result.winner.branch == "mc/fixup-candidate-1"
		assert result.winner.tests_passed == 5

	async def test_fallback_to_no_ff_merge(self) -> None:
		"""If ff-only merge fails, falls back to --no-ff merge."""
		mgr = _fixup_manager(fixup_candidates=1)

		candidates = [
			_candidate(0, passed=True, tests=10, lint=0, diff=5),
		]

		async def mock_run_candidate(index, prompt, failure_output, green_ref):
			return candidates[index]

		merge_attempts: list[tuple[str, ...]] = []

		async def selective_git(*args: str) -> tuple[bool, str]:
			if args[0] == "merge":
				merge_attempts.append(args)
				if "--ff-only" in args:
					return (False, "not possible to fast-forward")
				return (True, "")
			return (True, "")

		mgr._run_fixup_candidate = AsyncMock(side_effect=mock_run_candidate)
		mgr._run_git = AsyncMock(side_effect=selective_git)

		result = await mgr.run_fixup("some failure")

		assert result.success is True
		assert len(merge_attempts) == 2
		assert "--ff-only" in merge_attempts[0]
		assert "--no-ff" in merge_attempts[1]


# ---------------------------------------------------------------------------
# Individual candidate execution
# ---------------------------------------------------------------------------


class TestRunFixupCandidate:
	"""Tests for the individual candidate execution."""

	async def test_candidate_creates_branch_from_green(self) -> None:
		"""Candidate creates branch from mc/green and runs verification."""
		mgr = _fixup_manager()

		git_calls: list[tuple[str, ...]] = []

		async def tracking_git(*args: str) -> tuple[bool, str]:
			git_calls.append(args)
			if args[0] == "diff":
				return (True, " 1 file changed, 3 insertions(+)\n")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=tracking_git)
		mgr._run_fixup_session = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(
			return_value=(True, "10 passed\nAll checks passed"),
		)

		await mgr._run_fixup_candidate(0, "fix it", "failure", "mc/green")

		checkout_calls = [c for c in git_calls if c[0] == "checkout"]
		assert any("mc/green" in c for c in checkout_calls)
		assert any("mc/fixup-candidate-0" in c for c in checkout_calls)

	async def test_candidate_parses_verification_results(self) -> None:
		"""Candidate parses pytest and ruff output from verification."""
		mgr = _fixup_manager()

		mgr._run_git = AsyncMock(
			return_value=(True, " 1 file changed, 5 insertions(+)\n"),
		)
		mgr._run_fixup_session = AsyncMock(return_value=(True, ""))
		verify_output = (
			"42 passed, 3 failed\n"
			"src/foo.py:1:1: E501 line too long\n"
			"src/bar.py:2:3: F401 unused import"
		)
		mgr._run_command = AsyncMock(return_value=(True, verify_output))

		candidate = await mgr._run_fixup_candidate(
			0, "fix it", "failure", "mc/green",
		)

		assert candidate.tests_passed == 42
		assert candidate.lint_errors == 2
		assert candidate.diff_lines == 5

	async def test_candidate_handles_branch_creation_failure(self) -> None:
		"""If branch creation fails, returns empty candidate."""
		mgr = _fixup_manager()

		async def failing_checkout(*args: str) -> tuple[bool, str]:
			if args[0] == "checkout" and "-b" in args:
				return (False, "branch already exists")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=failing_checkout)

		candidate = await mgr._run_fixup_candidate(
			0, "fix it", "failure", "mc/green",
		)

		assert candidate.verification_passed is False
		assert candidate.tests_passed == 0

	async def test_candidate_measures_diff(self) -> None:
		"""Candidate uses git diff --stat to measure diff size."""
		mgr = _fixup_manager()

		diff_output = (
			" src/a.py | 10 +++++-----\n"
			" 2 files changed, 15 insertions(+), 8 deletions(-)\n"
		)

		async def selective_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and "--stat" in args:
				return (True, diff_output)
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=selective_git)
		mgr._run_fixup_session = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(
			return_value=(True, "5 passed\nAll checks passed"),
		)

		candidate = await mgr._run_fixup_candidate(
			0, "fix it", "failure", "mc/green",
		)

		assert candidate.diff_lines == 23


# ---------------------------------------------------------------------------
# Claude Code subprocess spawning for fixup
# ---------------------------------------------------------------------------


class TestRunFixupSession:
	"""Tests for the Claude Code subprocess spawning."""

	async def test_spawns_claude_with_correct_args(self) -> None:
		"""_run_fixup_session calls _run_command with claude CLI args."""
		mgr = _fixup_manager()
		mgr._run_command = AsyncMock(return_value=(True, "done"))

		await mgr._run_fixup_session("fix the bug")

		mgr._run_command.assert_awaited_once()
		cmd = mgr._run_command.call_args[0][0]
		assert cmd[0] == "claude"
		assert "--print" in cmd
		assert "--output-format" in cmd
		assert "text" in cmd
		assert "-p" in cmd
		assert "fix the bug" in cmd


# ---------------------------------------------------------------------------
# ZFC fixup prompts
# ---------------------------------------------------------------------------


class TestZFCFixupPrompts:
	def _zfc_manager(self, zfc_fixup: bool = False) -> GreenBranchManager:
		mgr = _fixup_manager()
		mgr.config.zfc.zfc_fixup_prompts = zfc_fixup
		mgr.config.zfc.llm_timeout = 5
		return mgr

	async def test_zfc_disabled_uses_static_prompts(self) -> None:
		"""toggle off -> FIXUP_PROMPTS used."""
		mgr = self._zfc_manager(zfc_fixup=False)

		mgr._run_fixup_candidate = AsyncMock(return_value=MagicMock(
			verification_passed=False, branch="mc/fixup-0",
		))
		mgr._run_git = AsyncMock(return_value=(True, ""))

		with patch.object(mgr, "_zfc_generate_fixup_strategies") as mock_zfc:
			await mgr.run_fixup("test failure")
			mock_zfc.assert_not_called()

	async def test_zfc_enabled_calls_llm(self) -> None:
		"""toggle on -> _zfc_generate_fixup_strategies called."""
		mgr = self._zfc_manager(zfc_fixup=True)

		mgr._run_fixup_candidate = AsyncMock(return_value=MagicMock(
			verification_passed=False, branch="mc/fixup-0",
		))
		mgr._run_git = AsyncMock(return_value=(True, ""))

		with patch.object(
			mgr, "_zfc_generate_fixup_strategies",
			return_value=["Fix A", "Fix B", "Fix C"],
		) as mock_zfc:
			await mgr.run_fixup("test failure")
			mock_zfc.assert_called_once_with("test failure", 3)

	async def test_zfc_llm_failure_falls_back(self) -> None:
		"""LLM returns None -> static prompts used as fallback."""
		mgr = self._zfc_manager(zfc_fixup=True)

		mgr._run_fixup_candidate = AsyncMock(return_value=MagicMock(
			verification_passed=False, branch="mc/fixup-0",
		))
		mgr._run_git = AsyncMock(return_value=(True, ""))

		with patch.object(mgr, "_zfc_generate_fixup_strategies", return_value=None):
			result = await mgr.run_fixup("test failure")
			assert result is not None

	async def test_zfc_parse_strategies_output(self) -> None:
		"""Correct FIXUP_STRATEGIES marker is parsed."""
		mgr = self._zfc_manager(zfc_fixup=True)

		output = (
			'Some reasoning about fixes.\n'
			'FIXUP_STRATEGIES:{"strategies": ["Fix the import", "Update the mock", "Rewrite test"]}'
		)

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (output.encode(), b"")
		mock_proc.returncode = 0

		with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
			strategies = await mgr._zfc_generate_fixup_strategies("test failure", 3)

		assert strategies is not None
		assert len(strategies) == 3
		assert "Fix the import" in strategies[0]

	async def test_zfc_timeout_returns_none(self) -> None:
		"""Subprocess timeout -> returns None."""
		mgr = self._zfc_manager(zfc_fixup=True)
		mgr.config.zfc.llm_timeout = 0

		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = TimeoutError()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch("asyncio.wait_for", side_effect=TimeoutError()):
				strategies = await mgr._zfc_generate_fixup_strategies("test failure", 3)

		assert strategies is None


# ---------------------------------------------------------------------------
# Phase 1: Track actual merged files via git diff
# ---------------------------------------------------------------------------


class TestMergedFilesFromGitDiff:
	"""UnitMergeResult.changed_files is populated from git diff, not files_hint."""

	async def test_changed_files_populated_on_merge(self) -> None:
		"""Successful merge populates changed_files from git diff --name-only."""
		mgr = _conflict_manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and args[1] == "--name-only":
				return (True, "src/app.py\nsrc/utils.py\n")
			if args[0] == "rev-parse" and args[1] == "HEAD":
				return (True, "abc123\n")
			if args[0] == "rev-list" and "--count" in args:
				return (True, "1")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(True, ""))  # type: ignore[method-assign]
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert result.changed_files == ["src/app.py", "src/utils.py"]

	async def test_changed_files_empty_when_diff_empty(self) -> None:
		"""If git diff returns empty output, changed_files is empty list."""
		mgr = _conflict_manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and args[1] == "--name-only":
				return (True, "")
			if args[0] == "rev-list" and "--count" in args:
				return (True, "1")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(True, ""))  # type: ignore[method-assign]
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert result.changed_files == []

	async def test_controller_uses_changed_files_not_files_hint(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Controller tracks actual changed files, not files_hint."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		# Mock green branch manager to return changed_files different from files_hint
		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
				changed_files=["src/app.py"],  # actual diff
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, ""))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
			files_hint="src/app.py, src/index.html",  # declared scope includes index.html
		)
		db.insert_work_unit(unit)

		handoff = Handoff(
			work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="completed", summary="Done",
		)

		completion = WorkerCompletion(
			unit=unit, handoff=handoff, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		# _merged_files should contain only app.py (from git diff), not index.html
		assert "src/app.py" in ctrl._merged_files
		assert "src/index.html" not in ctrl._merged_files

	async def test_controller_falls_back_to_files_hint(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When changed_files is empty, controller falls back to files_hint."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
				changed_files=[],  # empty -- fallback to files_hint
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, ""))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu2", plan_id="p1", title="Task",
			status="completed", commit_hash="def456",
			branch_name="mc/unit-wu2",
			files_hint="src/models.py",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		assert "src/models.py" in ctrl._merged_files


# ---------------------------------------------------------------------------
# Phase 2: Locked files in planner prompt
# ---------------------------------------------------------------------------


class TestGetLockedFiles:
	"""ContinuousController._get_locked_files() returns in-flight + merged files."""

	@pytest.mark.asyncio
	async def test_empty_when_nothing_running(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		locked = await ctrl._get_locked_files()
		assert locked == {}

	@pytest.mark.asyncio
	async def test_includes_merged_files(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		ctrl._merged_files = {"src/app.py", "src/index.html"}
		locked = await ctrl._get_locked_files()
		assert "src/app.py" in locked
		assert "already merged" in locked["src/app.py"]
		assert "src/index.html" in locked
		assert "already merged" in locked["src/index.html"]

	@pytest.mark.asyncio
	async def test_includes_in_flight_files(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(
			id="wu-running", plan_id="p1", title="Build API",
			status="running", files_hint="src/api.py, src/routes.py",
		)
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		locked = await ctrl._get_locked_files()

		assert "src/api.py" in locked
		assert any("in-flight" in r for r in locked["src/api.py"])
		assert "src/routes.py" in locked

	@pytest.mark.asyncio
	async def test_combines_in_flight_and_merged(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(
			id="wu-r", plan_id="p1", title="Build API",
			status="running", files_hint="src/api.py",
		)
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		ctrl._merged_files = {"src/app.py"}
		locked = await ctrl._get_locked_files()

		assert "src/api.py" in locked
		assert "src/app.py" in locked


class TestLockedFilesInPlannerPrompt:
	"""Locked files section is injected into the planner prompt."""

	@pytest.mark.asyncio
	async def test_locked_files_passed_to_planner(self) -> None:
		"""ContinuousPlanner forwards locked_files to RecursivePlanner."""
		config = _conflict_config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		captured_kwargs: dict = {}

		async def mock_plan_round(**kwargs):
			captured_kwargs.update(kwargs)
			plan = Plan(objective="test")
			from mission_control.models import PlanNode
			root = PlanNode(plan_id=plan.id, depth=0, scope="test", node_type="branch")
			root.strategy = "leaves"
			root._child_leaves = []  # type: ignore[attr-defined]
			root.status = "expanded"
			return plan, root

		planner._inner.plan_round = mock_plan_round  # type: ignore[assignment]

		locked = {"src/app.py": ["already merged"], "src/api.py": ["in-flight: Build API"]}
		mission = Mission(id="m1", objective="test")

		await planner.get_next_units(mission, locked_files=locked)

		assert "locked_files" in captured_kwargs
		assert captured_kwargs["locked_files"] == locked

	@pytest.mark.asyncio
	async def test_locked_section_in_prompt(self) -> None:
		"""RecursivePlanner injects ## Locked Files section into the LLM prompt."""
		from mission_control.recursive_planner import PlannerResult, RecursivePlanner

		config = _conflict_config()
		db = Database(":memory:")
		rp = RecursivePlanner(config, db)

		captured_prompt = ""

		async def mock_subprocess(prompt: str, node: object) -> PlannerResult:
			nonlocal captured_prompt
			captured_prompt = prompt
			return PlannerResult(type="leaves", units=[])

		rp._run_planner_subprocess = mock_subprocess  # type: ignore[assignment]

		locked = {"src/app.py": ["already merged"], "src/api.py": ["in-flight: Build API"]}
		await rp.plan_round(
			objective="test",
			snapshot_hash="",
			prior_discoveries=[],
			round_number=1,
			locked_files=locked,
		)

		assert "## Locked Files" in captured_prompt
		assert "src/app.py (already merged)" in captured_prompt
		assert "src/api.py (in-flight: Build API)" in captured_prompt
		assert "AUTOMATICALLY DROPPED" in captured_prompt

	@pytest.mark.asyncio
	async def test_no_locked_section_when_empty(self) -> None:
		"""No ## Locked Files section when locked_files is empty."""
		from mission_control.recursive_planner import PlannerResult, RecursivePlanner

		config = _conflict_config()
		db = Database(":memory:")
		rp = RecursivePlanner(config, db)

		captured_prompt = ""

		async def mock_subprocess(prompt: str, node: object) -> PlannerResult:
			nonlocal captured_prompt
			captured_prompt = prompt
			return PlannerResult(type="leaves", units=[])

		rp._run_planner_subprocess = mock_subprocess  # type: ignore[assignment]

		await rp.plan_round(
			objective="test",
			snapshot_hash="",
			prior_discoveries=[],
			round_number=1,
			locked_files={},
		)

		assert "## Locked Files" not in captured_prompt


# ---------------------------------------------------------------------------
# Real git integration tests
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: str | Path) -> str:
	result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
	return result.stdout.strip()


def _setup_source_repo(tmp_path: Path) -> tuple[Path, Path]:
	source = tmp_path / "source.git"
	source.mkdir()
	_run(["git", "init", "--bare"], source)

	setup_clone = tmp_path / "setup-clone"
	_run(["git", "clone", str(source), str(setup_clone)], tmp_path)
	_run(["git", "config", "user.email", "test@test.com"], setup_clone)
	_run(["git", "config", "user.name", "Test"], setup_clone)

	(setup_clone / "README.md").write_text("# Test\n")
	_run(["git", "add", "README.md"], setup_clone)
	_run(["git", "commit", "-m", "Initial commit"], setup_clone)
	_run(["git", "push", "origin", "main"], setup_clone)

	(setup_clone / "app.py").write_text("print('hello')\n")
	_run(["git", "add", "app.py"], setup_clone)
	_run(["git", "commit", "-m", "Add app.py"], setup_clone)
	_run(["git", "push", "origin", "main"], setup_clone)

	main_hash = _run(["git", "rev-parse", "main"], setup_clone)
	for branch in ("mc/green", "mc/working"):
		_run(["git", "branch", branch, main_hash], setup_clone)
		_run(["git", "push", "origin", branch], setup_clone)

	workspace = tmp_path / "workspace"
	_run(["git", "clone", str(source), str(workspace)], tmp_path)
	_run(["git", "config", "user.email", "test@test.com"], workspace)
	_run(["git", "config", "user.name", "Test"], workspace)
	_run(["git", "branch", "mc/green", "origin/mc/green"], workspace)
	_run(["git", "branch", "mc/working", "origin/mc/working"], workspace)

	return source, workspace


class TestChangedFilesRealGit:
	"""Integration: changed_files populated from actual git diff."""

	async def test_changed_files_from_real_merge(self, tmp_path: Path) -> None:
		"""changed_files reflects actual files changed, not declared scope."""
		source, workspace = _setup_source_repo(tmp_path)
		config = MissionConfig()
		config.target = TargetConfig(
			name="test", path=str(source), branch="main",
			verification=VerificationConfig(command="true"),
		)
		config.green_branch = GreenBranchConfig(
			working_branch="mc/working",
			green_branch="mc/green",
			reset_on_init=False,
		)

		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)
		mgr.workspace = str(workspace)

		# Create worker that only touches feature_a.py
		worker = tmp_path / "worker"
		_run(["git", "clone", str(source), str(worker)], tmp_path)
		_run(["git", "config", "user.email", "w@test.com"], worker)
		_run(["git", "config", "user.name", "Worker"], worker)
		_run(["git", "checkout", "-b", "unit/feature-a"], worker)
		(worker / "feature_a.py").write_text("# Feature A\n")
		_run(["git", "add", "feature_a.py"], worker)
		_run(["git", "commit", "-m", "Add feature A"], worker)

		result = await mgr.merge_unit(str(worker), "unit/feature-a")

		assert result.merged is True
		assert "feature_a.py" in result.changed_files
		# Should NOT include files not actually changed
		assert "app.py" not in result.changed_files
		assert "README.md" not in result.changed_files


# ---------------------------------------------------------------------------
# Phase 4: Reconciler sweep
# ---------------------------------------------------------------------------


class TestReconciliationCheck:
	"""GreenBranchManager.run_reconciliation_check() runs verification on green."""

	async def test_reconciliation_check_passes(self) -> None:
		mgr = _conflict_manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(True, "all tests passed"))  # type: ignore[method-assign]

		ok, output = await mgr.run_reconciliation_check()

		assert ok is True
		assert output == "all tests passed"
		# Should checkout mc/green before running verification
		mgr._run_git.assert_any_call("checkout", "mc/green")

	async def test_reconciliation_check_fails(self) -> None:
		mgr = _conflict_manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(False, "2 tests failed"))  # type: ignore[method-assign]

		ok, output = await mgr.run_reconciliation_check()

		assert ok is False
		assert "failed" in output


class TestReconcilerSweep:
	"""Controller triggers reconciler after merges."""

	@pytest.mark.asyncio
	async def test_reconciler_runs_after_merge(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Reconciler sweep fires after a successful merge."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, "all pass"))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		mock_gbm.run_reconciliation_check.assert_awaited_once()

	@pytest.mark.asyncio
	async def test_reconciler_triggers_fixup_on_failure(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When reconciler verification fails, run_fixup is called."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		from mission_control.green_branch import FixupResult

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(
			return_value=(False, "ImportError: cannot import name 'foo'"),
		)
		mock_gbm.run_fixup = AsyncMock(
			return_value=FixupResult(success=True),
		)
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		mock_gbm.run_fixup.assert_awaited_once()
		# Fixup should receive the failure output
		call_args = mock_gbm.run_fixup.call_args
		assert "ImportError" in call_args[0][0]

	@pytest.mark.asyncio
	async def test_reconciler_skipped_when_no_new_merges(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Reconciler doesn't re-run if no new merges since last check."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, ""))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		# Process first merge
		unit1 = WorkUnit(
			id="wu1", plan_id="p1", title="Task 1",
			status="completed", commit_hash="abc", branch_name="mc/unit-wu1",
		)
		db.insert_work_unit(unit1)
		ctrl._completion_queue.put_nowait(
			WorkerCompletion(unit=unit1, handoff=None, workspace="/tmp/ws", epoch=epoch),
		)

		# Process a failed unit (no merge)
		unit2 = WorkUnit(
			id="wu2", plan_id="p1", title="Task 2",
			status="failed",
		)
		db.insert_work_unit(unit2)
		ctrl._completion_queue.put_nowait(
			WorkerCompletion(unit=unit2, handoff=None, workspace="/tmp/ws", epoch=epoch),
		)

		ctrl.running = False
		await ctrl._process_completions(Mission(id="m1"), result)

		# Reconciler should have run once (after first merge) but not again for the failure
		assert mock_gbm.run_reconciliation_check.await_count == 1


# ---------------------------------------------------------------------------
# Merge conflict auto-resolution via LLM
# ---------------------------------------------------------------------------


class TestResolveMergeConflict:
	"""Tests for _resolve_merge_conflict() LLM-based conflict resolution."""

	async def test_resolve_gets_conflicted_files(self) -> None:
		"""Prompt includes filenames from git diff --diff-filter=U."""
		mgr = _conflict_manager()
		captured_cmd: list[str] = []

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and "--diff-filter=U" in args:
				return (True, "src/app.py\nsrc/utils.py\n")
			if args[0] == "status" and "--porcelain" in args:
				return (True, "")
			return (True, "")

		async def mock_run_command(cmd: str | list[str]) -> tuple[bool, str]:
			if isinstance(cmd, list):
				captured_cmd.extend(cmd)
			return (True, "resolved")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(side_effect=mock_run_command)  # type: ignore[method-assign]
		mgr._run_verification = AsyncMock(  # type: ignore[method-assign]
			return_value=MagicMock(overall_passed=True),
		)

		# Create fake conflicted files in workspace
		with tempfile.TemporaryDirectory() as td:
			mgr.workspace = td
			(Path(td) / "src").mkdir()
			(Path(td) / "src" / "app.py").write_text("<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>>\n")
			(Path(td) / "src" / "utils.py").write_text("<<<<<<< HEAD\nA\n=======\nB\n>>>>>>>\n")

			ok, output = await mgr._resolve_merge_conflict("feat/branch", "Merge feat/branch")

		assert ok is True
		# The prompt (last arg to claude) should mention both files
		prompt = captured_cmd[-1]
		assert "src/app.py" in prompt
		assert "src/utils.py" in prompt

	async def test_resolve_fails_when_no_conflicted_files(self) -> None:
		"""Empty diff --diff-filter=U returns False."""
		mgr = _conflict_manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and "--diff-filter=U" in args:
				return (True, "")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)

		ok, output = await mgr._resolve_merge_conflict("feat/branch", "Merge msg")

		assert ok is False
		assert "No conflicted files" in output

	async def test_resolve_fails_when_session_fails(self) -> None:
		"""Claude session returning failure returns False."""
		mgr = _conflict_manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and "--diff-filter=U" in args:
				return (True, "src/app.py\n")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(False, "session error"))  # type: ignore[method-assign]

		with tempfile.TemporaryDirectory() as td:
			mgr.workspace = td
			(Path(td) / "src").mkdir()
			(Path(td) / "src" / "app.py").write_text("conflict markers")

			ok, output = await mgr._resolve_merge_conflict("feat/branch", "Merge msg")

		assert ok is False

	async def test_resolve_fails_when_working_tree_dirty(self) -> None:
		"""Incomplete resolution (dirty working tree) returns False."""
		mgr = _conflict_manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and "--diff-filter=U" in args:
				return (True, "src/app.py\n")
			if args[0] == "status" and "--porcelain" in args:
				return (True, "UU src/app.py\n")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(True, "attempted resolution"))  # type: ignore[method-assign]

		with tempfile.TemporaryDirectory() as td:
			mgr.workspace = td
			(Path(td) / "src").mkdir()
			(Path(td) / "src" / "app.py").write_text("conflict markers")

			ok, output = await mgr._resolve_merge_conflict("feat/branch", "Merge msg")

		assert ok is False
		assert "dirty" in output.lower()

	async def test_resolve_fails_when_verification_fails(self) -> None:
		"""Bad resolution caught by verification returns False."""
		mgr = _conflict_manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and "--diff-filter=U" in args:
				return (True, "src/app.py\n")
			if args[0] == "status" and "--porcelain" in args:
				return (True, "")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(True, "resolved"))  # type: ignore[method-assign]
		mgr._run_verification = AsyncMock(  # type: ignore[method-assign]
			return_value=MagicMock(overall_passed=False, raw_output="3 tests failed"),
		)

		with tempfile.TemporaryDirectory() as td:
			mgr.workspace = td
			(Path(td) / "src").mkdir()
			(Path(td) / "src" / "app.py").write_text("conflict markers")

			ok, output = await mgr._resolve_merge_conflict("feat/branch", "Merge msg")

		assert ok is False
		assert "Verification failed" in output
