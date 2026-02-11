"""Tests for memory/context loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_control.config import MissionConfig, TargetConfig
from mission_control.db import Database
from mission_control.memory import (
	compress_history,
	load_context,
	summarize_session,
)
from mission_control.models import Decision, Session, TaskRecord
from mission_control.reviewer import ReviewVerdict


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


def _config(path: str = "/tmp/test") -> MissionConfig:
	return MissionConfig(target=TargetConfig(name="test", path=path))


class TestLoadContext:
	def test_empty_db(self, db: Database) -> None:
		task = TaskRecord(source="test_failure", description="Fix tests")
		context = load_context(task, db, _config())
		assert context == ""

	def test_includes_session_history(self, db: Database) -> None:
		db.insert_session(Session(id="s1", target_name="p", task_description="Fixed stuff", status="completed"))
		task = TaskRecord(source="test_failure", description="Fix tests")
		context = load_context(task, db, _config())
		assert "s1" in context
		assert "Fixed stuff" in context

	def test_includes_decisions(self, db: Database) -> None:
		db.insert_session(Session(id="s1", target_name="p"))
		db.insert_decision(Decision(id="d1", session_id="s1", decision="Use async", rationale="Performance"))
		task = TaskRecord(source="lint", description="Fix lint")
		context = load_context(task, db, _config())
		assert "Use async" in context

	def test_includes_claude_md(self, db: Database, tmp_path: Path) -> None:
		(tmp_path / "CLAUDE.md").write_text("# Test Project\nSome instructions")
		db.insert_session(Session(id="s1", target_name="p"))
		task = TaskRecord(source="lint", description="Fix lint")
		context = load_context(task, db, _config(str(tmp_path)))
		assert "Test Project" in context


class TestSummarizeSession:
	def test_basic_summary(self) -> None:
		session = Session(id="abc", task_description="Fix tests", output_summary="All done")
		verdict = ReviewVerdict(verdict="helped", improvements=["2 tests fixed"])
		summary = summarize_session(session, verdict)
		assert "abc" in summary
		assert "helped" in summary
		assert "2 tests fixed" in summary

	def test_regression_summary(self) -> None:
		session = Session(id="xyz", task_description="Refactor")
		verdict = ReviewVerdict(verdict="hurt", regressions=["3 tests broken"])
		summary = summarize_session(session, verdict)
		assert "hurt" in summary
		assert "3 tests broken" in summary


class TestCompressHistory:
	def test_empty(self) -> None:
		assert compress_history([]) == ""

	def test_fits_budget(self) -> None:
		sessions = [Session(id=f"s{i}", task_description=f"Task {i}", status="completed") for i in range(5)]
		result = compress_history(sessions, max_chars=10000)
		assert "s0" in result
		assert "s4" in result

	def test_truncates(self) -> None:
		sessions = [Session(id=f"s{i}", task_description=f"Task {i}", status="completed") for i in range(100)]
		result = compress_history(sessions, max_chars=200)
		assert "... and" in result
		assert "more sessions" in result
