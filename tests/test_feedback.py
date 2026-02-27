"""Tests for the feedback system -- worker context and keyword extraction."""

from __future__ import annotations

import pytest

from mission_control.db import Database
from mission_control.feedback import (
	_extract_keywords,
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
