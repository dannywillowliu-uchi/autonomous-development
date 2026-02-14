"""Tests for the feedback system -- reward computation, recording, and retrieval."""

from __future__ import annotations

import json

import pytest

from mission_control.db import Database
from mission_control.feedback import (
	_extract_keywords,
	_get_completed_files,
	compute_reward,
	get_planner_context,
	get_worker_context,
	record_round_outcome,
)
from mission_control.models import (
	Experience,
	Handoff,
	Mission,
	Plan,
	PlanNode,
	Reflection,
	Round,
	Snapshot,
	WorkUnit,
)


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


def _make_reflection(**overrides: object) -> Reflection:
	defaults = {
		"mission_id": "m1",
		"round_id": "r1",
		"round_number": 1,
		"tests_before": 10,
		"tests_after": 12,
		"tests_delta": 2,
		"lint_delta": 0,
		"type_delta": 0,
		"objective_score": 0.5,
		"score_delta": 0.2,
		"units_planned": 3,
		"units_completed": 3,
		"units_failed": 0,
		"completion_rate": 1.0,
		"plan_depth": 2,
		"plan_strategy": "subdivide",
		"fixup_promoted": True,
		"fixup_attempts": 1,
		"merge_conflicts": 0,
		"discoveries_count": 2,
	}
	defaults.update(overrides)
	return Reflection(**defaults)  # type: ignore[arg-type]


def _make_snapshots(
	tests_before: int = 10,
	tests_after: int = 12,
	lint_before: int = 5,
	lint_after: int = 3,
	type_before: int = 2,
	type_after: int = 2,
	security_before: int = 0,
	security_after: int = 0,
) -> tuple[Snapshot, Snapshot]:
	before = Snapshot(
		id="snap-before",
		test_total=tests_before + 2,
		test_passed=tests_before,
		test_failed=2,
		lint_errors=lint_before,
		type_errors=type_before,
		security_findings=security_before,
	)
	after = Snapshot(
		id="snap-after",
		test_total=tests_after + 1,
		test_passed=tests_after,
		test_failed=1,
		lint_errors=lint_after,
		type_errors=type_after,
		security_findings=security_after,
	)
	return before, after


class TestComputeReward:
	def test_all_success(self) -> None:
		"""Perfect round: tests improved, full completion, fixup on first try."""
		ref = _make_reflection(
			objective_score=0.5,
			completion_rate=1.0,
			fixup_promoted=True,
			fixup_attempts=1,
		)
		before, after = _make_snapshots(tests_before=10, tests_after=15)
		reward = compute_reward(ref, prev_score=0.2, snapshot_before=before, snapshot_after=after)
		# vi=1.0, cr=1.0, sp=1.0, fe=1.0, nr=1.0
		assert reward.reward == pytest.approx(1.0, abs=0.05)
		assert reward.verification_improvement == 1.0
		assert reward.completion_rate == 1.0
		assert reward.fixup_efficiency == 1.0
		assert reward.no_regression == 1.0

	def test_regression(self) -> None:
		"""Broken tests: low reward."""
		ref = _make_reflection(
			objective_score=0.3,
			completion_rate=0.5,
			fixup_promoted=False,
			fixup_attempts=3,
		)
		# More test failures after
		before = Snapshot(
			id="sb", test_total=20, test_passed=18, test_failed=2,
			lint_errors=0, type_errors=0, security_findings=0,
		)
		after = Snapshot(
			id="sa", test_total=20, test_passed=15, test_failed=5,
			lint_errors=0, type_errors=0, security_findings=0,
		)
		reward = compute_reward(ref, prev_score=0.3, snapshot_before=before, snapshot_after=after)
		# vi=0.0, cr=0.5, sp=0.0, fe=0.0, nr=0.0
		assert reward.reward < 0.2
		assert reward.verification_improvement == 0.0
		assert reward.no_regression == 0.0

	def test_partial_completion(self) -> None:
		"""Partial completion: proportional reward."""
		ref = _make_reflection(
			objective_score=0.4,
			completion_rate=0.6,
			fixup_promoted=True,
			fixup_attempts=2,
		)
		before, after = _make_snapshots()
		reward = compute_reward(ref, prev_score=0.3, snapshot_before=before, snapshot_after=after)
		assert 0.3 < reward.reward < 0.8
		assert reward.completion_rate == 0.6
		assert reward.fixup_efficiency == 0.5  # promoted but >1 attempt

	def test_no_snapshots(self) -> None:
		"""When snapshots are missing, verification scores default to 0.5."""
		ref = _make_reflection(completion_rate=1.0, fixup_promoted=True, fixup_attempts=1)
		reward = compute_reward(ref, prev_score=0.0, snapshot_before=None, snapshot_after=None)
		assert reward.verification_improvement == 0.5
		assert reward.no_regression == 0.5

	def test_score_progress_normalized(self) -> None:
		"""Score progress clamps between 0 and 1."""
		ref = _make_reflection(objective_score=0.8, completion_rate=1.0, fixup_promoted=True)
		before, after = _make_snapshots()
		# Large score jump
		reward = compute_reward(ref, prev_score=0.0, snapshot_before=before, snapshot_after=after)
		assert reward.score_progress == pytest.approx(1.0)
		# Negative score change
		ref2 = _make_reflection(objective_score=0.2, completion_rate=1.0, fixup_promoted=True)
		reward2 = compute_reward(ref2, prev_score=0.5, snapshot_before=before, snapshot_after=after)
		assert reward2.score_progress == 0.0


class TestRecordRoundOutcome:
	def test_full_integration(self, db: Database) -> None:
		"""Record outcome with all data populated."""
		# Setup: mission, round, plan, work units, handoffs
		mission = Mission(id="m1", objective="build feature")
		db.insert_mission(mission)

		rnd = Round(id="r1", mission_id="m1", number=1, objective_score=0.5)
		db.insert_round(rnd)

		plan = Plan(id="p1", objective="build feature", total_units=2)
		plan.root_node_id = "pn-root"
		db.insert_plan(plan)

		root_node = PlanNode(id="pn-root", plan_id="p1", depth=0, strategy="subdivide", node_type="branch")
		db.insert_plan_node(root_node)

		wu1 = WorkUnit(
			id="wu1", plan_id="p1", title="Add API", description="REST endpoints",
			files_hint="api.py", status="completed",
		)
		wu2 = WorkUnit(
			id="wu2", plan_id="p1", title="Add tests", description="Unit tests",
			files_hint="test_api.py", status="completed",
		)
		db.insert_work_unit(wu1)
		db.insert_work_unit(wu2)

		h1 = Handoff(
			id="h1", work_unit_id="wu1", round_id="r1", status="completed",
			summary="Added REST API", files_changed='["api.py"]',
			discoveries='["Found existing endpoint"]', concerns='[]',
		)
		h2 = Handoff(
			id="h2", work_unit_id="wu2", round_id="r1", status="completed",
			summary="Added tests", files_changed='["test_api.py"]',
			discoveries='[]', concerns='["Coverage could be higher"]',
		)
		db.insert_handoff(h1)
		db.insert_handoff(h2)

		from types import SimpleNamespace
		fixup = SimpleNamespace(promoted=True, fixup_attempts=1)
		before, after = _make_snapshots()

		reward = record_round_outcome(
			db=db, mission_id="m1", rnd=rnd, plan=plan,
			handoffs=[h1, h2], fixup_result=fixup,
			snapshot_before=before, snapshot_after=after,
			prev_score=0.3,
		)

		assert reward.reward > 0.0

		# Check reflections were stored
		reflections = db.get_recent_reflections("m1")
		assert len(reflections) == 1
		assert reflections[0].units_completed == 2
		assert reflections[0].completion_rate == 1.0
		assert reflections[0].plan_strategy == "subdivide"

		# Check experiences were stored
		experiences = db.get_high_reward_experiences()
		assert len(experiences) == 2
		assert experiences[0].title in ("Add API", "Add tests")


class TestGetPlannerContext:
	def test_empty_no_history(self, db: Database) -> None:
		"""No reflections = empty context."""
		assert get_planner_context(db, "m1") == ""

	def test_with_data(self, db: Database) -> None:
		"""Reflections produce structured summary."""
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)

		for i in range(3):
			rnd = Round(id=f"r{i}", mission_id="m1", number=i + 1)
			db.insert_round(rnd)
			ref = Reflection(
				id=f"ref{i}", mission_id="m1", round_id=f"r{i}",
				round_number=i + 1, objective_score=0.3 * (i + 1),
				completion_rate=0.8 + i * 0.05, plan_strategy="subdivide",
				fixup_promoted=i > 0, fixup_attempts=1, merge_conflicts=1 if i == 0 else 0,
			)
			db.insert_reflection(ref)

		ctx = get_planner_context(db, "m1")
		assert "Score trajectory" in ctx
		assert "subdivide" in ctx
		assert "Merge conflicts" in ctx
		assert "Fixup promoted" in ctx

	def test_includes_discoveries_from_top_experiences(self, db: Database) -> None:
		"""Planner context includes discoveries from high-reward experiences."""
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)
		rnd = Round(id="r1", mission_id="m1", number=1)
		db.insert_round(rnd)
		ref = Reflection(
			id="ref1", mission_id="m1", round_id="r1", round_number=1,
			objective_score=0.5, completion_rate=0.9, plan_strategy="subdivide",
			fixup_promoted=True, fixup_attempts=1,
		)
		db.insert_reflection(ref)

		# Add some experiences with discoveries
		plan = Plan(id="p1", objective="test", total_units=2)
		db.insert_plan(plan)
		wu1 = WorkUnit(id="wu1", plan_id="p1", title="Task 1")
		wu2 = WorkUnit(id="wu2", plan_id="p1", title="Task 2")
		db.insert_work_unit(wu1)
		db.insert_work_unit(wu2)

		exp1 = Experience(
			id="exp1", round_id="r1", work_unit_id="wu1",
			title="High reward task", status="completed", reward=0.95,
			summary="Did something great",
			discoveries='["Database has existing migration framework", "Use async for better performance"]',
		)
		exp2 = Experience(
			id="exp2", round_id="r1", work_unit_id="wu2",
			title="Lower reward task", status="completed", reward=0.65,
			summary="Did something ok",
			discoveries='["Error handling needs improvement"]',
		)
		db.insert_experience(exp1)
		db.insert_experience(exp2)

		ctx = get_planner_context(db, "m1")
		assert "Key insights from successful past work:" in ctx
		assert "Database has existing migration framework" in ctx
		assert "Use async for better performance" in ctx


	def test_includes_completed_files(self, db: Database) -> None:
		"""Planner context includes files already modified by completed handoffs."""
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)

		rnd = Round(id="r1", mission_id="m1", number=1, plan_id="p1")
		db.insert_round(rnd)

		plan = Plan(id="p1", objective="test", total_units=1)
		db.insert_plan(plan)

		wu = WorkUnit(id="wu1", plan_id="p1", title="Task 1", status="completed")
		db.insert_work_unit(wu)

		h = Handoff(
			id="h1", work_unit_id="wu1", round_id="r1", status="completed",
			summary="Done", files_changed='["src/app.tsx", "src/utils.ts"]',
		)
		db.insert_handoff(h)

		# Need a reflection for get_planner_context to return anything
		ref = Reflection(
			id="ref1", mission_id="m1", round_id="r1", round_number=1,
			objective_score=0.5, completion_rate=1.0, plan_strategy="leaves",
			fixup_promoted=True, fixup_attempts=1,
		)
		db.insert_reflection(ref)

		ctx = get_planner_context(db, "m1")
		assert "Files already modified" in ctx
		assert "src/app.tsx" in ctx
		assert "src/utils.ts" in ctx
		assert "Do NOT create units that re-modify" in ctx

	def test_includes_conflict_files(self, db: Database) -> None:
		"""Planner context includes files that caused merge conflicts."""
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)

		rnd = Round(id="r1", mission_id="m1", number=1, plan_id="p1")
		db.insert_round(rnd)

		plan = Plan(id="p1", objective="test", total_units=1)
		db.insert_plan(plan)

		wu = WorkUnit(
			id="wu1", plan_id="p1", title="Task 1", status="failed",
			output_summary="Merge conflict: conflicting changes in shared files",
			files_hint="src/shared.ts, src/labels.ts",
		)
		db.insert_work_unit(wu)

		ref = Reflection(
			id="ref1", mission_id="m1", round_id="r1", round_number=1,
			objective_score=0.3, completion_rate=0.0, plan_strategy="leaves",
			fixup_promoted=False, fixup_attempts=0,
		)
		db.insert_reflection(ref)

		ctx = get_planner_context(db, "m1")
		assert "Files that caused merge conflicts" in ctx
		assert "src/shared.ts" in ctx
		assert "src/labels.ts" in ctx
		assert "AVOID these files" in ctx

	def test_files_capped(self, db: Database) -> None:
		"""Completed files are capped at 30."""
		mission = Mission(id="m1", objective="test")
		db.insert_mission(mission)

		rnd = Round(id="r1", mission_id="m1", number=1)
		db.insert_round(rnd)

		# Create 40 unique files across multiple handoffs
		all_files = [f"src/file_{i:03d}.py" for i in range(40)]
		for batch_idx in range(4):
			batch = all_files[batch_idx * 10:(batch_idx + 1) * 10]
			plan = Plan(id=f"p{batch_idx}", objective="test", total_units=1)
			db.insert_plan(plan)
			wu = WorkUnit(id=f"wu{batch_idx}", plan_id=f"p{batch_idx}", title=f"Task {batch_idx}", status="completed")
			db.insert_work_unit(wu)
			h = Handoff(
				id=f"h{batch_idx}", work_unit_id=f"wu{batch_idx}", round_id="r1",
				status="completed", summary="Done",
				files_changed=json.dumps(batch),
			)
			db.insert_handoff(h)

		result = _get_completed_files(db, "m1")
		assert len(result) == 30


class TestGetWorkerContext:
	def test_matching_experience(self, db: Database) -> None:
		"""Keywords match an existing experience."""
		# Setup a mission and round so FKs are satisfied
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

		# Search with a matching unit
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

		# Search with a similar unit
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
		# Stop words filtered
		assert "add" not in keywords
		assert "for" not in keywords

	def test_empty(self) -> None:
		assert _extract_keywords("") == []

	def test_underscores(self) -> None:
		keywords = _extract_keywords("test_api_endpoints")
		assert "test_api_endpoints" in keywords


class TestExperienceSearchByKeywords:
	def test_like_queries(self, db: Database) -> None:
		"""SQLite LIKE queries work correctly for experience search."""
		# Setup FKs
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

		# Search for API-related
		results = db.search_experiences(["api", "endpoints"])
		assert len(results) >= 1
		assert results[0].title == "REST API setup"

		# Search for DB-related
		results = db.search_experiences(["database", "schema"])
		assert len(results) >= 1
		assert results[0].title == "Database schema"

		# No match
		results = db.search_experiences(["kubernetes"])
		assert len(results) == 0

		# Empty keywords
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
		# Most recent first
		assert results[0].round_number == 5
		assert results[1].round_number == 4
		assert results[2].round_number == 3
