"""Tests for N-of-M fixup candidate selection in green_branch.py."""

from __future__ import annotations

from unittest.mock import AsyncMock

from mission_control.config import (
	GreenBranchConfig,
	MissionConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.db import Database
from mission_control.green_branch import (
	FIXUP_PROMPTS,
	FixupCandidate,
	FixupResult,
	GreenBranchManager,
)


def _config(fixup_candidates: int = 3) -> MissionConfig:
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


def _manager(fixup_candidates: int = 3) -> GreenBranchManager:
	config = _config(fixup_candidates)
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


class TestRunFixup:
	"""Tests for the N-of-M fixup candidate selection."""

	async def test_best_candidate_selected_by_tests(self) -> None:
		"""Candidate with most tests passing wins."""
		mgr = _manager(fixup_candidates=3)

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
		mgr = _manager(fixup_candidates=3)

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
		mgr = _manager(fixup_candidates=3)

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
		mgr = _manager(fixup_candidates=3)

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
		mgr = _manager(fixup_candidates=3)

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
		mgr = _manager(fixup_candidates=2)

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
		mgr = _manager(fixup_candidates=3)

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
		mgr = _manager(fixup_candidates=2)

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
		mgr = _manager(fixup_candidates=5)

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
		mgr = _manager(fixup_candidates=5)

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
		mgr = _manager(fixup_candidates=3)

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
		mgr = _manager(fixup_candidates=1)

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


class TestRunFixupCandidate:
	"""Tests for the individual candidate execution."""

	async def test_candidate_creates_branch_from_green(self) -> None:
		"""Candidate creates branch from mc/green and runs verification."""
		mgr = _manager()

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
		mgr = _manager()

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
		mgr = _manager()

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
		mgr = _manager()

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


class TestRunFixupSession:
	"""Tests for the Claude Code subprocess spawning."""

	async def test_spawns_claude_with_correct_args(self) -> None:
		"""_run_fixup_session calls _run_command with claude CLI args."""
		mgr = _manager()
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
