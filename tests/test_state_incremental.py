"""Tests for change-aware incremental verification in state.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from mission_control.config import (
	MissionConfig,
	TargetConfig,
	VerificationConfig,
	VerificationNodeConfig,
)
from mission_control.models import IncrementalVerificationResult, VerificationNodeKind
from mission_control.state import _build_affected_tests, run_verification_nodes

# -- Helpers --

COLLECT_OUTPUT = (
	"tests/test_state.py::TestParsePytest::test_all_passing\n"
	"tests/test_state.py::TestParsePytest::test_mixed\n"
	"tests/test_state_incremental.py::TestBuildAffectedTests::test_basic\n"
	"tests/test_worker.py::TestWorker::test_run\n"
	"tests/test_config.py::TestConfig::test_load\n"
	"tests/test_green_branch.py::TestGreen::test_merge\n"
	"\n6 tests collected"
)


def _make_config(*, nodes: list[VerificationNodeConfig] | None = None) -> MissionConfig:
	config = MissionConfig()
	config.target = TargetConfig(
		path="/tmp/project",
		verification=VerificationConfig(
			command="python -m pytest -q",
			nodes=nodes or [],
		),
	)
	return config


def _collect_result(output: str = COLLECT_OUTPUT, rc: int = 0) -> dict:
	return {"output": output, "returncode": rc}


def _pytest_result(passed: int = 5, failed: int = 0) -> dict:
	parts = []
	if passed:
		parts.append(f"{passed} passed")
	if failed:
		parts.append(f"{failed} failed")
	summary = ", ".join(parts) + " in 0.1s"
	rc = 0 if failed == 0 else 1
	return {"output": summary, "returncode": rc}


# -- _build_affected_tests --


class TestBuildAffectedTests:
	async def test_maps_source_to_test(self) -> None:
		"""src/mission_control/state.py -> tests/test_state.py + tests/test_state_incremental.py."""
		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=_collect_result()):
			result = await _build_affected_tests(
				["src/mission_control/state.py"], "/tmp/project",
			)

		assert result is not None
		assert "tests/test_state.py" in result
		assert "tests/test_state_incremental.py" in result
		assert "tests/test_worker.py" not in result

	async def test_multiple_changed_files(self) -> None:
		"""Multiple changed source files aggregate their affected tests."""
		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=_collect_result()):
			result = await _build_affected_tests(
				["src/mission_control/state.py", "src/mission_control/worker.py"],
				"/tmp/project",
			)

		assert result is not None
		assert "tests/test_state.py" in result
		assert "tests/test_state_incremental.py" in result
		assert "tests/test_worker.py" in result

	async def test_changed_test_file_included_directly(self) -> None:
		"""A changed test file is included directly if it exists in collection."""
		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=_collect_result()):
			result = await _build_affected_tests(
				["tests/test_config.py"], "/tmp/project",
			)

		assert result is not None
		assert "tests/test_config.py" in result
		assert len(result) == 1

	async def test_no_python_files_returns_none(self) -> None:
		"""Non-Python changes should skip test selection entirely."""
		with patch("mission_control.state._run_command", new_callable=AsyncMock) as mock_cmd:
			result = await _build_affected_tests(
				["README.md", "Makefile", "config.toml"], "/tmp/project",
			)

		assert result is None
		mock_cmd.assert_not_called()

	async def test_empty_changed_files_returns_none(self) -> None:
		"""Empty list of changed files returns None."""
		with patch("mission_control.state._run_command", new_callable=AsyncMock) as mock_cmd:
			result = await _build_affected_tests([], "/tmp/project")

		assert result is None
		mock_cmd.assert_not_called()

	async def test_collection_failure_returns_none(self) -> None:
		"""Pytest collection failure returns None (fallback to full suite)."""
		fail_result = {"output": "", "returncode": 1}
		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=fail_result):
			result = await _build_affected_tests(
				["src/mission_control/state.py"], "/tmp/project",
			)

		assert result is None

	async def test_no_matching_tests_returns_none(self) -> None:
		"""Changed file with no matching tests returns None."""
		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=_collect_result()):
			result = await _build_affected_tests(
				["src/mission_control/totally_new_module.py"], "/tmp/project",
			)

		assert result is None

	async def test_result_is_sorted(self) -> None:
		"""Returned test files should be sorted for deterministic ordering."""
		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=_collect_result()):
			result = await _build_affected_tests(
				["src/mission_control/state.py"], "/tmp/project",
			)

		assert result is not None
		assert result == sorted(result)

	async def test_mixed_python_and_non_python(self) -> None:
		"""Non-Python files are filtered out, Python files are processed."""
		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=_collect_result()):
			result = await _build_affected_tests(
				["README.md", "src/mission_control/config.py", "Dockerfile"],
				"/tmp/project",
			)

		assert result is not None
		assert "tests/test_config.py" in result

	async def test_collection_with_plain_file_lines(self) -> None:
		"""Handle pytest --collect-only output that lists bare file paths."""
		collect = "tests/test_state.py\ntests/test_worker.py\n"
		with patch(
			"mission_control.state._run_command",
			new_callable=AsyncMock,
			return_value={"output": collect, "returncode": 0},
		):
			result = await _build_affected_tests(
				["src/mission_control/state.py"], "/tmp/project",
			)

		assert result is not None
		assert "tests/test_state.py" in result


# -- run_verification_nodes with changed_files --


class TestIncrementalVerificationSingleCommand:
	"""Incremental verification in single-command (backward compat) mode."""

	async def test_no_changed_files_runs_full(self) -> None:
		"""Without changed_files, runs the full command and selection_method='full'."""
		config = _make_config()
		mock_result = _pytest_result(passed=10)

		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=mock_result):
			incremental = await run_verification_nodes(config, "/tmp")

		assert isinstance(incremental, IncrementalVerificationResult)
		assert incremental.selection_method == "full"
		assert incremental.tests_selected == 0
		assert incremental.report.overall_passed is True

	async def test_changed_files_none_runs_full(self) -> None:
		"""Explicitly passing None runs the full suite."""
		config = _make_config()
		mock_result = _pytest_result(passed=10)

		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=mock_result):
			incremental = await run_verification_nodes(config, "/tmp", changed_files=None)

		assert incremental.selection_method == "full"

	async def test_changed_files_triggers_heuristic(self) -> None:
		"""Providing changed_files with matching tests uses heuristic selection."""
		config = _make_config()

		call_log: list[str] = []

		async def mock_run(cmd: str, cwd: str, timeout: int = 300) -> dict:
			call_log.append(cmd)
			if "--collect-only" in cmd:
				return _collect_result()
			return _pytest_result(passed=2)

		with patch("mission_control.state._run_command", side_effect=mock_run):
			incremental = await run_verification_nodes(
				config, "/tmp", changed_files=["src/mission_control/state.py"],
			)

		assert incremental.selection_method == "heuristic"
		# The actual pytest command should reference specific test files
		pytest_cmd = [c for c in call_log if "--collect-only" not in c][0]
		assert "test_state" in pytest_cmd

	async def test_fallback_when_no_matching_tests(self) -> None:
		"""Falls back to full suite when no tests match changed files."""
		config = _make_config()

		async def mock_run(cmd: str, cwd: str, timeout: int = 300) -> dict:
			if "--collect-only" in cmd:
				return _collect_result()
			return _pytest_result(passed=10)

		with patch("mission_control.state._run_command", side_effect=mock_run):
			incremental = await run_verification_nodes(
				config, "/tmp", changed_files=["src/mission_control/unknown_module.py"],
			)

		assert incremental.selection_method == "full"

	async def test_fallback_when_non_python_changes(self) -> None:
		"""Non-Python file changes fall back to full suite."""
		config = _make_config()
		mock_result = _pytest_result(passed=10)

		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=mock_result):
			incremental = await run_verification_nodes(
				config, "/tmp", changed_files=["README.md", "Makefile"],
			)

		assert incremental.selection_method == "full"

	async def test_empty_changed_files_runs_full(self) -> None:
		"""Empty changed_files list falls back to full suite."""
		config = _make_config()

		async def mock_run(cmd: str, cwd: str, timeout: int = 300) -> dict:
			if "--collect-only" in cmd:
				return _collect_result()
			return _pytest_result(passed=10)

		with patch("mission_control.state._run_command", side_effect=mock_run):
			incremental = await run_verification_nodes(
				config, "/tmp", changed_files=[],
			)

		# Empty list is not None, so _build_affected_tests is called,
		# but no python files -> returns None -> full
		assert incremental.selection_method == "full"


class TestIncrementalVerificationMultiNode:
	"""Incremental verification with explicit verification nodes."""

	async def test_heuristic_modifies_pytest_node(self) -> None:
		"""When affected tests found, the pytest node command targets them."""
		config = _make_config(nodes=[
			VerificationNodeConfig(kind="pytest", command="python -m pytest -q", required=True),
			VerificationNodeConfig(kind="ruff", command="ruff check .", required=True),
		])

		call_log: list[str] = []

		async def mock_run(cmd: str, cwd: str, timeout: int = 300) -> dict:
			call_log.append(cmd)
			if "--collect-only" in cmd:
				return _collect_result()
			if "pytest" in cmd:
				return _pytest_result(passed=2)
			return {"output": "All checks passed", "returncode": 0}

		with patch("mission_control.state._run_command", side_effect=mock_run):
			incremental = await run_verification_nodes(
				config, "/tmp", changed_files=["src/mission_control/state.py"],
			)

		assert incremental.selection_method == "heuristic"
		assert incremental.tests_selected == 2
		# Ruff command should be unchanged
		ruff_cmds = [c for c in call_log if "ruff" in c]
		assert len(ruff_cmds) == 1
		assert ruff_cmds[0] == "ruff check ."

	async def test_non_pytest_nodes_unaffected(self) -> None:
		"""Non-pytest nodes are not modified by changed_files."""
		config = _make_config(nodes=[
			VerificationNodeConfig(kind="ruff", command="ruff check .", required=True),
			VerificationNodeConfig(kind="mypy", command="mypy src/", required=False),
		])

		call_log: list[str] = []

		async def mock_run(cmd: str, cwd: str, timeout: int = 300) -> dict:
			call_log.append(cmd)
			if "--collect-only" in cmd:
				return _collect_result()
			return {"output": "All checks passed", "returncode": 0}

		with patch("mission_control.state._run_command", side_effect=mock_run):
			await run_verification_nodes(
				config, "/tmp", changed_files=["src/mission_control/state.py"],
			)

		# Even with changed_files, ruff/mypy commands stay the same
		non_collect = [c for c in call_log if "--collect-only" not in c]
		assert "ruff check ." in non_collect
		assert "mypy src/" in non_collect

	async def test_full_fallback_in_multi_node(self) -> None:
		"""Falls back to full pytest command when no tests match."""
		config = _make_config(nodes=[
			VerificationNodeConfig(kind="pytest", command="python -m pytest -q", required=True),
		])

		call_log: list[str] = []

		async def mock_run(cmd: str, cwd: str, timeout: int = 300) -> dict:
			call_log.append(cmd)
			if "--collect-only" in cmd:
				return _collect_result()
			return _pytest_result(passed=50)

		with patch("mission_control.state._run_command", side_effect=mock_run):
			incremental = await run_verification_nodes(
				config, "/tmp", changed_files=["src/mission_control/brand_new.py"],
			)

		assert incremental.selection_method == "full"
		# Should use original command, not targeted
		pytest_cmds = [c for c in call_log if "pytest" in c and "--collect-only" not in c]
		assert pytest_cmds[0] == "python -m pytest -q"


class TestIncrementalVerificationResultDataclass:
	"""Test the IncrementalVerificationResult dataclass."""

	def test_defaults(self) -> None:
		r = IncrementalVerificationResult()
		assert r.tests_selected == 0
		assert r.tests_skipped == 0
		assert r.selection_method == "full"
		assert r.report.results == []

	def test_heuristic_method(self) -> None:
		r = IncrementalVerificationResult(
			tests_selected=5,
			tests_skipped=495,
			selection_method="heuristic",
		)
		assert r.selection_method == "heuristic"
		assert r.tests_selected == 5
		assert r.tests_skipped == 495

	def test_report_access(self) -> None:
		"""The underlying report is accessible through .report."""
		from mission_control.models import VerificationReport, VerificationResult

		report = VerificationReport(results=[
			VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True, required=True),
		])
		r = IncrementalVerificationResult(report=report, selection_method="heuristic")
		assert r.report.overall_passed is True
