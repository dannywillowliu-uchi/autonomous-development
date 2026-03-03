"""Tests for verification typed nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from mission_control.config import MissionConfig, TargetConfig, VerificationConfig, VerificationNodeConfig
from mission_control.models import VerificationNodeKind, VerificationReport, VerificationResult
from mission_control.state import run_verification_nodes, snapshot_project_health


class TestVerificationResult:
	def test_defaults(self) -> None:
		r = VerificationResult()
		assert r.kind == VerificationNodeKind.CUSTOM
		assert r.passed is False
		assert r.exit_code == 0
		assert r.metrics == {}
		assert r.duration_seconds == 0.0
		assert r.required is True
		assert r.weight == 1.0

	def test_with_metrics(self) -> None:
		r = VerificationResult(
			kind=VerificationNodeKind.PYTEST,
			passed=True,
			metrics={"test_passed": 42, "test_failed": 0, "test_total": 42},
		)
		assert r.metrics["test_passed"] == 42
		assert r.kind == VerificationNodeKind.PYTEST


class TestVerificationReport:
	def test_overall_passed_all_pass(self) -> None:
		report = VerificationReport(results=[
			VerificationResult(passed=True, required=True),
			VerificationResult(passed=True, required=True),
		])
		assert report.overall_passed is True

	def test_overall_passed_required_fails(self) -> None:
		report = VerificationReport(results=[
			VerificationResult(passed=True, required=True),
			VerificationResult(passed=False, required=True),
		])
		assert report.overall_passed is False

	def test_overall_passed_optional_fails(self) -> None:
		report = VerificationReport(results=[
			VerificationResult(passed=True, required=True),
			VerificationResult(passed=False, required=False),
		])
		assert report.overall_passed is True

	def test_weighted_score(self) -> None:
		report = VerificationReport(results=[
			VerificationResult(passed=True, weight=2.0),
			VerificationResult(passed=False, weight=1.0),
			VerificationResult(passed=True, weight=3.0),
		])
		assert report.weighted_score == 5.0

	def test_weighted_score_empty(self) -> None:
		report = VerificationReport(results=[])
		assert report.weighted_score == 0.0

	def test_failed_kinds(self) -> None:
		report = VerificationReport(results=[
			VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True),
			VerificationResult(kind=VerificationNodeKind.RUFF, passed=False),
			VerificationResult(kind=VerificationNodeKind.MYPY, passed=False),
		])
		failed = report.failed_kinds()
		assert VerificationNodeKind.RUFF in failed
		assert VerificationNodeKind.MYPY in failed
		assert VerificationNodeKind.PYTEST not in failed

	def test_failed_kinds_empty_when_all_pass(self) -> None:
		report = VerificationReport(results=[
			VerificationResult(kind=VerificationNodeKind.PYTEST, passed=True),
		])
		assert report.failed_kinds() == []


class TestRunVerificationNodes:
	async def test_single_command_fallback(self) -> None:
		"""When no nodes configured, falls back to single command mode."""
		config = MissionConfig()
		config.target = TargetConfig(
			path="/tmp",
			verification=VerificationConfig(command="echo '10 passed in 0.1s'"),
		)

		mock_result = {"output": "10 passed in 0.1s", "returncode": 0}
		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=mock_result):
			incremental = await run_verification_nodes(config, "/tmp")

		report = incremental.report
		assert report.overall_passed is True
		assert len(report.results) == 4  # pytest, ruff, mypy, bandit
		assert report.raw_output == "10 passed in 0.1s"

	async def test_multi_node_execution(self) -> None:
		"""When nodes are configured, runs each one individually."""
		config = MissionConfig()
		config.target = TargetConfig(
			path="/tmp",
			verification=VerificationConfig(
				command="echo fallback",
				nodes=[
					VerificationNodeConfig(kind="pytest", command="pytest -q", required=True),
					VerificationNodeConfig(kind="ruff", command="ruff check .", required=True),
				],
			),
		)

		async def mock_run(cmd: str, cwd: str, timeout: int = 300) -> dict:
			if "pytest" in cmd:
				return {"output": "5 passed in 0.1s", "returncode": 0}
			else:
				return {"output": "All checks passed", "returncode": 0}

		with patch("mission_control.state._run_command", side_effect=mock_run):
			incremental = await run_verification_nodes(config, "/tmp")

		report = incremental.report
		assert report.overall_passed is True
		assert len(report.results) == 2

	async def test_required_vs_optional(self) -> None:
		"""Optional node failure should not block overall pass."""
		config = MissionConfig()
		config.target = TargetConfig(
			path="/tmp",
			verification=VerificationConfig(
				command="echo fallback",
				nodes=[
					VerificationNodeConfig(kind="pytest", command="pytest -q", required=True),
					VerificationNodeConfig(kind="bandit", command="bandit -r .", required=False),
				],
			),
		)

		async def mock_run(cmd: str, cwd: str, timeout: int = 300) -> dict:
			if "pytest" in cmd:
				return {"output": "3 passed in 0.1s", "returncode": 0}
			else:
				return {"output": ">> Issue: something bad", "returncode": 1}

		with patch("mission_control.state._run_command", side_effect=mock_run):
			incremental = await run_verification_nodes(config, "/tmp")

		report = incremental.report
		assert report.overall_passed is True  # optional failure doesn't block
		assert len(report.failed_kinds()) == 1
		assert VerificationNodeKind.BANDIT in report.failed_kinds()


class TestBackwardCompat:
	async def test_snapshot_project_health_returns_snapshot(self) -> None:
		"""snapshot_project_health still returns Snapshot with correct fields."""
		config = MissionConfig()
		config.target = TargetConfig(
			path="/tmp",
			verification=VerificationConfig(command="echo test"),
		)

		mock_result = {
			"output": "10 passed, 1 failed in 0.5s\nsrc/foo.py:1:1: E501 Line too long",
			"returncode": 1,
		}
		with patch("mission_control.state._run_command", new_callable=AsyncMock, return_value=mock_result):
			snapshot = await snapshot_project_health(config, "/tmp")

		assert snapshot.test_total == 11
		assert snapshot.test_passed == 10
		assert snapshot.test_failed == 1
		assert snapshot.lint_errors == 1
