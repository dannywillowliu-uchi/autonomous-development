"""Tests for the feedback system -- worker context and keyword extraction."""

from __future__ import annotations

import pytest

from mission_control.db import Database
from mission_control.feedback import (
	_extract_keywords,
	diagnose_failure,
	get_worker_context,
)
from mission_control.models import (
	Experience,
	Mission,
	Plan,
	Reflection,
	Round,
	WorkUnit,
)


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


class TestGetWorkerContext:
	def test_matching_experience(self, db: Database) -> None:
		"""Keywords match an existing experience."""
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		rnd = Round(id="r1", mission_id="m1", number=1)
		db.insert_round(rnd)
		plan = Plan(id="p1", objective="test", total_units=1)
		db.insert_plan(plan)
		wu_past = WorkUnit(id="wu-past", plan_id="p1", title="API endpoints", files_hint="api.py")
		db.insert_work_unit(wu_past)

		exp = Experience(
			id="exp1", round_id="r1", work_unit_id="wu-past",
			title="API endpoints", scope="REST endpoints for users",
			files_hint="api.py", status="completed",
			summary="Built CRUD endpoints", reward=0.8,
			concerns='["Rate limiting needed"]',
		)
		db.insert_experience(exp)

		unit = WorkUnit(title="API routes", description="REST endpoints", files_hint="api.py")
		ctx = get_worker_context(db, unit)
		assert "API endpoints" in ctx
		assert "succeeded" in ctx

	def test_no_match(self, db: Database) -> None:
		"""No keyword overlap = empty context."""
		unit = WorkUnit(title="Database migration", description="Schema changes", files_hint="schema.sql")
		ctx = get_worker_context(db, unit)
		assert ctx == ""

	def test_includes_discoveries_from_similar_work(self, db: Database) -> None:
		"""Worker context includes discoveries from similar past experiences."""
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		rnd = Round(id="r1", mission_id="m1", number=1)
		db.insert_round(rnd)
		plan = Plan(id="p1", objective="test", total_units=1)
		db.insert_plan(plan)
		wu_past = WorkUnit(id="wu-past", plan_id="p1", title="API tests", files_hint="test_api.py")
		db.insert_work_unit(wu_past)

		exp = Experience(
			id="exp1", round_id="r1", work_unit_id="wu-past",
			title="API tests", scope="Unit tests for REST endpoints",
			files_hint="test_api.py", status="completed",
			summary="Added comprehensive test coverage", reward=0.9,
			discoveries='["Fixtures can be shared across test modules", "Mock external API calls for speed"]',
			concerns='["Test data cleanup needed"]',
		)
		db.insert_experience(exp)

		unit = WorkUnit(title="Add API tests", description="Test REST endpoints", files_hint="test_api.py")
		ctx = get_worker_context(db, unit)
		assert "API tests" in ctx
		assert "Insights: Fixtures can be shared across test modules" in ctx
		assert "Mock external API calls for speed" in ctx
		assert "Pitfalls: Test data cleanup needed" in ctx


class TestExtractKeywords:
	def test_basic(self) -> None:
		keywords = _extract_keywords("Add REST API endpoints for user auth")
		assert "rest" in keywords
		assert "api" in keywords
		assert "endpoints" in keywords
		assert "user" in keywords
		assert "auth" in keywords
		assert "add" not in keywords
		assert "for" not in keywords

	def test_empty(self) -> None:
		assert _extract_keywords("") == []

	def test_underscores(self) -> None:
		keywords = _extract_keywords("test_api_endpoints")
		assert "test_api_endpoints" in keywords

	def test_file_paths_preserved(self) -> None:
		"""File paths like 'src/foo/bar.py' are preserved as whole keywords."""
		keywords = _extract_keywords("fix src/mission_control/worker.py")
		assert "src/mission_control/worker.py" in keywords

	def test_multiple_paths(self) -> None:
		keywords = _extract_keywords("edit src/foo/bar.py and tests/test_bar.py")
		assert "src/foo/bar.py" in keywords
		assert "tests/test_bar.py" in keywords

	def test_tokens_still_extracted_alongside_paths(self) -> None:
		"""Existing token extraction still works for non-path text."""
		keywords = _extract_keywords("fix src/mission_control/worker.py session handling")
		assert "src/mission_control/worker.py" in keywords
		assert "session" in keywords
		assert "handling" in keywords

	def test_deduplication_path_and_tokens(self) -> None:
		"""Dedup works: path tokens don't duplicate the path itself."""
		keywords = _extract_keywords("src/foo/bar.py")
		assert keywords.count("src/foo/bar.py") == 1
		# Tokens from the path (src, foo, bar, py) should not duplicate
		assert len(keywords) == len(set(k.lower() for k in keywords))


class TestExperienceSearchByKeywords:
	def test_like_queries(self, db: Database) -> None:
		"""SQLite LIKE queries work correctly for experience search."""
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		rnd = Round(id="r1", mission_id="m1", number=1)
		db.insert_round(rnd)
		plan = Plan(id="p1", objective="test", total_units=2)
		db.insert_plan(plan)
		wu1 = WorkUnit(id="wu1", plan_id="p1", title="API")
		wu2 = WorkUnit(id="wu2", plan_id="p1", title="DB")
		db.insert_work_unit(wu1)
		db.insert_work_unit(wu2)

		exp1 = Experience(
			id="e1", round_id="r1", work_unit_id="wu1",
			title="REST API setup", scope="endpoints", files_hint="api.py",
			status="completed", summary="Built API", reward=0.9,
		)
		exp2 = Experience(
			id="e2", round_id="r1", work_unit_id="wu2",
			title="Database schema", scope="migrations", files_hint="schema.sql",
			status="completed", summary="Created tables", reward=0.7,
		)
		db.insert_experience(exp1)
		db.insert_experience(exp2)

		results = db.search_experiences(["api", "endpoints"])
		assert len(results) >= 1
		assert results[0].title == "REST API setup"

		results = db.search_experiences(["database", "schema"])
		assert len(results) >= 1
		assert results[0].title == "Database schema"

		results = db.search_experiences(["kubernetes"])
		assert len(results) == 0

		results = db.search_experiences([])
		assert len(results) == 0


class TestDBFeedbackCRUD:
	"""Test DB operations for feedback tables."""

	def test_reflection_round_trip(self, db: Database) -> None:
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		rnd = Round(id="r1", mission_id="m1", number=1)
		db.insert_round(rnd)

		ref = Reflection(
			id="ref1", mission_id="m1", round_id="r1", round_number=1,
			tests_delta=5, completion_rate=0.8, fixup_promoted=True,
		)
		db.insert_reflection(ref)

		results = db.get_recent_reflections("m1")
		assert len(results) == 1
		assert results[0].id == "ref1"
		assert results[0].tests_delta == 5
		assert results[0].completion_rate == 0.8
		assert results[0].fixup_promoted is True

	def test_experience_round_trip(self, db: Database) -> None:
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		rnd = Round(id="r1", mission_id="m1", number=1)
		db.insert_round(rnd)
		plan = Plan(id="p1", objective="test", total_units=1)
		db.insert_plan(plan)
		wu = WorkUnit(id="wu1", plan_id="p1")
		db.insert_work_unit(wu)

		exp = Experience(
			id="exp1", round_id="r1", work_unit_id="wu1",
			title="Test task", reward=0.85, status="completed",
			summary="Did the thing", files_changed='["a.py"]',
		)
		db.insert_experience(exp)

		results = db.get_high_reward_experiences(limit=5)
		assert len(results) == 1
		assert results[0].reward == 0.85
		assert results[0].title == "Test task"

	def test_reflections_ordered_by_round_number(self, db: Database) -> None:
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		for i in range(5):
			rnd = Round(id=f"r{i}", mission_id="m1", number=i + 1)
			db.insert_round(rnd)
			ref = Reflection(
				id=f"ref{i}", mission_id="m1", round_id=f"r{i}",
				round_number=i + 1, objective_score=0.1 * (i + 1),
			)
			db.insert_reflection(ref)

		results = db.get_recent_reflections("m1", limit=3)
		assert len(results) == 3
		assert results[0].round_number == 5
		assert results[1].round_number == 4
		assert results[2].round_number == 3


class TestDiagnoseFailure:
	"""Tests for diagnose_failure() -- pattern matching failure categories."""

	def test_empty_output(self) -> None:
		result = diagnose_failure("")
		assert "No output captured" in result

	def test_none_like_empty(self) -> None:
		"""Empty string triggers the no-output branch."""
		result = diagnose_failure("")
		assert "[" not in result or "No output" in result

	# -- Merge conflicts --

	def test_merge_conflict_markers(self) -> None:
		output = (
			"Auto-merging src/foo.py\n"
			"CONFLICT (content): Merge conflict in src/foo.py\n"
			"<<<<<<< HEAD\n"
			"def old():\n"
			"=======\n"
			"def new():\n"
			">>>>>>> feature\n"
		)
		result = diagnose_failure(output)
		assert "[Merge conflict]" in result
		assert "rebase" in result.lower()

	def test_conflict_keyword_only(self) -> None:
		output = "error: CONFLICT (modify/delete): path/to/file.py"
		result = diagnose_failure(output)
		assert "[Merge conflict]" in result

	# -- Syntax errors --

	def test_syntax_error_basic(self) -> None:
		output = (
			'  File "src/app.py", line 42\n'
			"    def foo(\n"
			"           ^\n"
			"SyntaxError: invalid syntax\n"
		)
		result = diagnose_failure(output)
		assert "[Syntax error]" in result
		assert "parentheses" in result or "syntax" in result.lower()

	def test_syntax_error_indentation(self) -> None:
		output = (
			'  File "src/app.py", line 10\n'
			"    return x\n"
			"IndentationError: unexpected indent\n"
			"SyntaxError\n"
		)
		result = diagnose_failure(output)
		assert "[Syntax error]" in result

	def test_invalid_syntax_without_class_name(self) -> None:
		output = "line 5: invalid syntax near 'def'"
		result = diagnose_failure(output)
		assert "[Syntax error]" in result

	# -- Import errors --

	def test_module_not_found(self) -> None:
		output = (
			"Traceback (most recent call last):\n"
			'  File "src/main.py", line 1, in <module>\n'
			"    from mission_control.nonexistent import Foo\n"
			"ModuleNotFoundError: No module named 'mission_control.nonexistent'\n"
		)
		result = diagnose_failure(output)
		assert "[Import error]" in result
		assert "mission_control.nonexistent" in result
		assert "pip install" in result.lower()

	def test_import_error_relative(self) -> None:
		output = (
			"ImportError: cannot import name 'BadClass' from 'mission_control.models'\n"
		)
		result = diagnose_failure(output)
		assert "[Import error]" in result
		assert "BadClass" in result

	def test_import_error_no_module_name(self) -> None:
		output = "ImportError: DLL load failed"
		result = diagnose_failure(output)
		assert "[Import error]" in result

	# -- Pytest assertion failures --

	def test_pytest_failed_tests(self) -> None:
		output = (
			"tests/test_foo.py::test_add PASSED\n"
			"tests/test_foo.py::test_subtract FAILED\n"
			"tests/test_bar.py::test_multiply FAILED\n"
			"\n"
			"FAILED tests/test_foo.py::test_subtract - AssertionError: assert 3 == 4\n"
			"FAILED tests/test_bar.py::test_multiply - AssertionError: assert 6 == 8\n"
			"2 failed, 1 passed\n"
		)
		result = diagnose_failure(output)
		assert "[Test failure]" in result
		assert "test_subtract" in result or "test_foo" in result

	def test_assertion_error_only(self) -> None:
		output = (
			"E       AssertionError: expected True but got False\n"
			"        assert result is True\n"
		)
		result = diagnose_failure(output)
		assert "[Test failure]" in result

	def test_assert_keyword_in_traceback(self) -> None:
		output = (
			"    assert response.status_code == 200\n"
			"AssertionError\n"
		)
		result = diagnose_failure(output)
		assert "[Test failure]" in result
		assert "do not modify tests" in result.lower()

	# -- Ruff / lint errors --

	def test_ruff_errors(self) -> None:
		output = (
			"src/mission_control/worker.py:10:1: F401 `os` imported but unused\n"
			"src/mission_control/worker.py:25:80: E501 Line too long (130 > 120)\n"
			"Found 2 errors.\n"
		)
		result = diagnose_failure(output)
		assert "[Lint error]" in result
		assert "F401" in result
		assert "E501" in result

	def test_ruff_with_command_output(self) -> None:
		output = (
			"$ ruff check src/ tests/\n"
			"src/foo.py:1:1: F811 Redefinition of unused `bar`\n"
			"Found 1 error.\n"
		)
		result = diagnose_failure(output)
		assert "[Lint error]" in result
		assert "F811" in result

	def test_ruff_warning_codes(self) -> None:
		output = (
			"ruff check output:\n"
			"src/x.py:5:1: W291 trailing whitespace\n"
			"Found 1 error.\n"
		)
		result = diagnose_failure(output)
		assert "[Lint error]" in result
		assert "W291" in result

	# -- Timeout --

	def test_timeout_pattern(self) -> None:
		output = "Process timed out after 300 seconds"
		result = diagnose_failure(output)
		assert "[Timeout]" in result
		assert "scope" in result.lower() or "limit" in result.lower()

	def test_deadline_exceeded(self) -> None:
		output = "error: deadline exceeded waiting for worker response"
		result = diagnose_failure(output)
		assert "[Timeout]" in result

	def test_killed_signal(self) -> None:
		output = "Worker process killed by signal 9 (SIGKILL)"
		result = diagnose_failure(output)
		assert "[Timeout]" in result

	def test_time_out_two_words(self) -> None:
		output = "The operation timed out while running pytest"
		result = diagnose_failure(output)
		assert "[Timeout]" in result

	# -- Generic / unknown failures --

	def test_unknown_failure_with_error_line(self) -> None:
		output = (
			"Running task...\n"
			"Processing files...\n"
			"Fatal error: disk full\n"
			"Aborted.\n"
		)
		result = diagnose_failure(output)
		assert "[Unknown failure]" in result
		assert "disk full" in result.lower()

	def test_unknown_failure_no_error_hints(self) -> None:
		output = "Something happened but no clear error pattern."
		result = diagnose_failure(output)
		assert "[Unknown failure]" in result
		assert "root cause" in result.lower()

	# -- Priority / ordering --

	def test_merge_conflict_takes_priority_over_syntax(self) -> None:
		"""Merge conflicts should be detected before syntax errors."""
		output = (
			"SyntaxError: invalid syntax\n"
			"<<<<<<< HEAD\n"
			"old code\n"
			"=======\n"
			"new code\n"
			">>>>>>> branch\n"
		)
		result = diagnose_failure(output)
		assert "[Merge conflict]" in result

	def test_syntax_error_takes_priority_over_test_failure(self) -> None:
		"""Syntax errors should be detected before test failures."""
		output = (
			"FAILED tests/test_foo.py::test_bar\n"
			"SyntaxError: invalid syntax\n"
		)
		result = diagnose_failure(output)
		assert "[Syntax error]" in result

	def test_import_error_takes_priority_over_test_failure(self) -> None:
		"""Import errors block test execution, so detect first."""
		output = (
			"FAILED tests/test_foo.py::test_bar\n"
			"ModuleNotFoundError: No module named 'foo'\n"
		)
		result = diagnose_failure(output)
		assert "[Import error]" in result
