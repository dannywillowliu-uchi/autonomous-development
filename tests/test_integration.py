"""Integration tests -- end-to-end with a tiny target project."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mission_control.config import (
	BudgetConfig,
	GitConfig,
	MissionConfig,
	SchedulerConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.db import Database
from mission_control.discovery import discover_from_snapshot
from mission_control.models import Session, Snapshot
from mission_control.reviewer import review_session
from mission_control.scheduler import Scheduler
from mission_control.state import _parse_pytest, _parse_ruff, compare_snapshots


def _create_tiny_project(path: Path) -> None:
	"""Create a minimal Python project in tmp_path."""
	src = path / "src"
	src.mkdir()
	(src / "app.py").write_text('def add(a, b):\n\treturn a + b\n\nimport os\n')
	tests = path / "tests"
	tests.mkdir()
	(tests / "test_app.py").write_text(
		"from src.app import add\n\n"
		"def test_add_pass():\n\tassert add(1, 2) == 3\n\n"
		"def test_add_fail():\n\tassert add(1, 2) == 4\n"
	)


def _config_for(path: Path) -> MissionConfig:
	return MissionConfig(
		target=TargetConfig(
			name="tiny",
			path=str(path),
			branch="main",
			objective="Fix all tests",
			verification=VerificationConfig(command="pytest -q", timeout=60),
		),
		scheduler=SchedulerConfig(
			session_timeout=30,
			cooldown=0,
			max_sessions_per_run=3,
			model="sonnet",
			git=GitConfig(strategy="branch-per-session", auto_merge=False),
			budget=BudgetConfig(max_per_session_usd=1.0, max_per_run_usd=5.0),
		),
	)


class TestEndToEnd:
	def test_discovery_finds_failures(self, tmp_path: Path) -> None:
		"""Discovery engine finds test failures from a snapshot."""
		_create_tiny_project(tmp_path)
		config = _config_for(tmp_path)

		snapshot = Snapshot(
			test_total=2, test_passed=1, test_failed=1,
			lint_errors=1, type_errors=0,
		)

		tasks = discover_from_snapshot(snapshot, config)
		assert len(tasks) >= 2
		assert tasks[0].priority <= tasks[1].priority
		sources = {t.source for t in tasks}
		assert "test_failure" in sources
		assert "lint" in sources

	def test_review_detects_improvement(self) -> None:
		"""Reviewer correctly identifies when tests are fixed."""
		before = Snapshot(test_total=2, test_passed=1, test_failed=1, lint_errors=1)
		after = Snapshot(test_total=2, test_passed=2, test_failed=0, lint_errors=0)
		session = Session(id="int1", target_name="tiny", task_description="Fix tests")

		verdict = review_session(before, after, session)
		assert verdict.verdict == "helped"
		assert not verdict.should_revert

	def test_review_detects_regression(self) -> None:
		"""Reviewer correctly identifies when tests break."""
		before = Snapshot(test_total=2, test_passed=2, test_failed=0)
		after = Snapshot(test_total=2, test_passed=1, test_failed=1)
		session = Session(id="int2", target_name="tiny", task_description="Refactor")

		verdict = review_session(before, after, session)
		assert verdict.verdict == "hurt"
		assert verdict.should_revert

	def test_snapshot_delta_drives_verdict(self) -> None:
		"""The full pipeline: snapshot -> delta -> verdict."""
		before = Snapshot(test_total=10, test_passed=8, test_failed=2, lint_errors=5)
		after = Snapshot(test_total=10, test_passed=10, test_failed=0, lint_errors=3)

		delta = compare_snapshots(before, after)
		assert delta.improved
		assert not delta.regressed

	@pytest.mark.asyncio
	async def test_scheduler_with_mock_claude(self, tmp_path: Path) -> None:
		"""Full scheduler loop with mocked Claude subprocess."""
		config = _config_for(tmp_path)
		db = Database(":memory:")

		# Simulate: before has failures, after is clean
		call_count = 0

		async def mock_snapshot(_cfg: MissionConfig) -> Snapshot:
			nonlocal call_count
			call_count += 1
			if call_count % 2 == 1:
				return Snapshot(test_total=2, test_passed=1, test_failed=1, lint_errors=1)
			return Snapshot(test_total=2, test_passed=2, test_failed=0, lint_errors=0)

		async def mock_spawn(*args: object, **kwargs: object) -> Session:
			return Session(
				target_name="tiny",
				task_description="Fix failing test",
				status="completed",
				branch_name="mc/session-test",
				exit_code=0,
				output_summary="Fixed test_add_fail",
			)

		with (
			patch("mission_control.scheduler.snapshot_project_health", side_effect=mock_snapshot),
			patch("mission_control.scheduler.spawn_session", side_effect=mock_spawn),
			patch("mission_control.scheduler.delete_branch", new_callable=AsyncMock),
			patch("mission_control.scheduler.merge_branch", new_callable=AsyncMock),
		):
			scheduler = Scheduler(config, db)
			report = await scheduler.run(max_sessions=1)

		assert report.sessions_run == 1
		assert report.sessions_helped == 1
		assert report.sessions_hurt == 0

		# Verify state was persisted
		sessions = db.get_recent_sessions()
		assert len(sessions) == 1

	def test_parse_real_pytest_output(self) -> None:
		"""Parse realistic pytest output."""
		output = """\
tests/test_app.py::test_add_pass PASSED
tests/test_app.py::test_add_fail FAILED

FAILURES

tests/test_app.py::test_add_fail - assert 3 == 4

1 passed, 1 failed in 0.02s"""
		result = _parse_pytest(output)
		assert result["test_passed"] == 1
		assert result["test_failed"] == 1
		assert result["test_total"] == 2

	def test_parse_real_ruff_output(self) -> None:
		"""Parse realistic ruff output."""
		output = """\
src/app.py:4:8: F401 [*] `os` imported but unused
Found 1 error.
[*] 1 fixable with the `--fix` option."""
		result = _parse_ruff(output)
		assert result["lint_errors"] == 1
