"""Tests for memory.py context loading and episodic-to-semantic memory system."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mission_control.config import EpisodicMemoryConfig, MissionConfig, TargetConfig, load_config
from mission_control.db import Database
from mission_control.memory import (
	CONTEXT_BUDGET,
	MemoryManager,
	_format_session_history,
	build_file_previews,
	compress_history,
	extract_scope_tokens,
	format_context_items,
	inject_context_items,
	load_context_for_mission_worker,
	load_context_for_work_unit,
)
from mission_control.models import (
	ContextItem,
	ContextScope,
	Decision,
	EpisodicMemory,
	Epoch,
	Handoff,
	Mission,
	Plan,
	SemanticMemory,
	Session,
	WorkUnit,
)
from mission_control.planner_context import build_planner_context


@pytest.fixture
def db() -> Database:
	return Database(":memory:")


@pytest.fixture
def config(tmp_path) -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(path=str(tmp_path))
	return mc


@pytest.fixture()
def manager(db: Database) -> MemoryManager:
	config = EpisodicMemoryConfig(
		enabled=True, default_ttl_days=30, access_boost_days=5,
		min_episodes_for_distill=3,
	)
	return MemoryManager(db, config)


def _make_session(id: str = "s1", status: str = "completed", desc: str = "fix bug", summary: str = "done") -> Session:
	return Session(id=id, task_description=desc, status=status, output_summary=summary)


def _make_decision(decision: str = "use pytest", rationale: str = "standard") -> Decision:
	return Decision(session_id="s1", decision=decision, rationale=rationale)


# -- load_context_for_work_unit --


class TestLoadContextForWorkUnit:
	def test_empty_db_returns_empty(self, db, config):
		unit = WorkUnit(plan_id="", title="do thing")
		result = load_context_for_work_unit(unit, db, config)
		assert result == ""

	def test_includes_plan_objective(self, db, config):
		plan = Plan(id="plan1", objective="Improve test coverage")
		db.insert_plan(plan)
		unit = WorkUnit(plan_id="plan1", title="write tests")
		result = load_context_for_work_unit(unit, db, config)
		assert "### Plan" in result
		assert "Improve test coverage" in result

	def test_includes_sibling_unit_status(self, db, config):
		plan = Plan(id="plan1", objective="Refactor")
		db.insert_plan(plan)
		unit_a = WorkUnit(
			id="unit-a", plan_id="plan1", title="refactor module A",
			status="completed", output_summary="done refactoring A",
		)
		unit_b = WorkUnit(id="unit-b", plan_id="plan1", title="refactor module B", status="running")
		unit_c = WorkUnit(id="unit-c", plan_id="plan1", title="refactor module C", status="pending")
		db.insert_work_unit(unit_a)
		db.insert_work_unit(unit_b)
		db.insert_work_unit(unit_c)

		# Load context for unit_c -- should see siblings a and b
		result = load_context_for_work_unit(unit_c, db, config)
		assert "### Sibling Units" in result
		assert "refactor module A" in result
		assert "refactor module B" in result
		assert "refactor module C" not in result  # self excluded

	def test_includes_claude_md(self, db, config, tmp_path):
		(tmp_path / "CLAUDE.md").write_text("# Instructions\nDo the thing.")
		unit = WorkUnit(plan_id="", title="standalone")
		result = load_context_for_work_unit(unit, db, config)
		assert "### Project Instructions" in result
		assert "# Instructions" in result

	def test_budget_enforcement(self, db, config):
		"""Sibling status respects budget."""
		plan = Plan(id="plan1", objective="Big objective " * 500)
		db.insert_plan(plan)
		for i in range(50):
			db.insert_work_unit(WorkUnit(
				id=f"sib-{i}", plan_id="plan1",
				title=f"sibling task {i} " * 10,
				status="completed", output_summary="summary " * 10,
			))
		unit = WorkUnit(id="target", plan_id="plan1", title="my task")
		db.insert_work_unit(unit)
		result = load_context_for_work_unit(unit, db, config)
		# Result should stay reasonable
		assert len(result) <= CONTEXT_BUDGET + 500


# -- load_context_for_mission_worker --


class TestLoadContextForMissionWorker:
	def test_empty_returns_empty(self, config):
		unit = WorkUnit(title="task")
		result = load_context_for_mission_worker(unit, config)
		assert result == ""

	def test_includes_claude_md(self, config, tmp_path):
		(tmp_path / "CLAUDE.md").write_text("# Mission Instructions\nFollow these rules.")
		unit = WorkUnit(title="task")
		result = load_context_for_mission_worker(unit, config)
		assert "### Project Instructions" in result
		assert "# Mission Instructions" in result

	def test_respects_budget(self, config, tmp_path):
		"""Large CLAUDE.md is truncated to budget."""
		huge_content = "y" * 20000
		(tmp_path / "CLAUDE.md").write_text(huge_content)
		unit = WorkUnit(title="task")
		result = load_context_for_mission_worker(unit, config)
		# Truncated to min(4000, CONTEXT_BUDGET)
		max_content = min(4000, CONTEXT_BUDGET)
		assert result.count("y") <= max_content


# -- _format_session_history --


class TestFormatSessionHistory:
	def test_empty_list(self):
		assert _format_session_history([]) == ""

	def test_completed_session(self):
		s = _make_session(id="abc", status="completed", desc="add tests", summary="added 5 tests")
		result = _format_session_history([s])
		assert "[+] abc: add tests -> completed" in result
		assert "added 5 tests" in result

	def test_failed_session(self):
		s = _make_session(id="def", status="failed", desc="deploy", summary="timeout")
		result = _format_session_history([s])
		assert "[x] def: deploy -> failed (timeout)" in result

	def test_multiple_sessions(self):
		sessions = [
			_make_session(id="s1", status="completed", desc="task1", summary="ok"),
			_make_session(id="s2", status="failed", desc="task2", summary="err"),
		]
		result = _format_session_history(sessions)
		lines = result.strip().split("\n")
		assert len(lines) == 2
		assert "[+] s1" in lines[0]
		assert "[x] s2" in lines[1]


# -- compress_history --


class TestCompressHistory:
	def test_empty_list(self):
		assert compress_history([]) == ""

	def test_single_session(self):
		s = _make_session(id="s1", status="completed", desc="fix bug")
		result = compress_history([s])
		assert "s1: fix bug -> completed" in result

	def test_truncation_with_max_chars(self):
		sessions = [_make_session(id=f"s{i}", status="completed", desc=f"task {i}") for i in range(100)]
		result = compress_history(sessions, max_chars=100)
		assert len(result) <= 200  # generous bound including the "... and N more" line
		assert "... and" in result
		assert "more sessions" in result


# -- Context CRUD --


class TestContextItemCRUD:
	def test_insert_and_get(self, db):
		item = ContextItem(
			id="ctx1", item_type="gotcha", scope="src/db.py",
			content="WAL mode required", source_unit_id="u1",
			round_id="r1", confidence=0.9,
		)
		db.insert_context_item(item)
		fetched = db.get_context_item("ctx1")
		assert fetched is not None
		assert fetched.item_type == "gotcha"
		assert fetched.scope == "src/db.py"
		assert fetched.content == "WAL mode required"
		assert fetched.confidence == 0.9
		assert fetched.source_unit_id == "u1"
		assert fetched.round_id == "r1"

	def test_get_nonexistent_returns_none(self, db):
		assert db.get_context_item("nope") is None

	def test_delete(self, db):
		item = ContextItem(id="ctx2", item_type="convention", content="use tabs")
		db.insert_context_item(item)
		assert db.get_context_item("ctx2") is not None
		db.delete_context_item("ctx2")
		assert db.get_context_item("ctx2") is None

	def test_get_items_for_round(self, db):
		for i in range(3):
			db.insert_context_item(ContextItem(
				id=f"ctx-r1-{i}", round_id="r1",
				item_type="pattern", content=f"pattern {i}",
			))
		db.insert_context_item(ContextItem(
			id="ctx-r2-0", round_id="r2",
			item_type="pattern", content="other round",
		))
		items = db.get_context_items_for_round("r1")
		assert len(items) == 3
		assert all(it.round_id == "r1" for it in items)


# -- Scope-based filtering --


class TestScopeBasedFiltering:
	def test_scope_overlap_matches(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/db.py,src/models.py",
			item_type="gotcha", content="db gotcha", confidence=0.8,
		))
		db.insert_context_item(ContextItem(
			id="ctx-b", scope="src/cli.py",
			item_type="pattern", content="cli pattern", confidence=0.9,
		))
		results = db.get_context_items_by_scope_overlap(["src/db.py"])
		assert len(results) == 1
		assert results[0].id == "ctx-a"

	def test_scope_overlap_multiple_tokens(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/db.py", item_type="gotcha",
			content="db gotcha", confidence=0.8,
		))
		db.insert_context_item(ContextItem(
			id="ctx-b", scope="src/cli.py", item_type="pattern",
			content="cli pattern", confidence=0.9,
		))
		results = db.get_context_items_by_scope_overlap(["src/db.py", "src/cli.py"])
		assert len(results) == 2

	def test_scope_overlap_respects_min_confidence(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-low", scope="src/db.py", item_type="gotcha",
			content="low conf", confidence=0.3,
		))
		db.insert_context_item(ContextItem(
			id="ctx-high", scope="src/db.py", item_type="gotcha",
			content="high conf", confidence=0.8,
		))
		results = db.get_context_items_by_scope_overlap(["src/db.py"], min_confidence=0.5)
		assert len(results) == 1
		assert results[0].id == "ctx-high"

	def test_scope_overlap_empty_tokens(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/db.py", item_type="gotcha", content="stuff",
		))
		results = db.get_context_items_by_scope_overlap([])
		assert results == []

	def test_scope_overlap_ordered_by_confidence(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-low", scope="src/db.py", item_type="gotcha",
			content="low", confidence=0.6,
		))
		db.insert_context_item(ContextItem(
			id="ctx-high", scope="src/db.py", item_type="gotcha",
			content="high", confidence=0.95,
		))
		results = db.get_context_items_by_scope_overlap(["src/db.py"], min_confidence=0.5)
		assert results[0].id == "ctx-high"
		assert results[1].id == "ctx-low"


# -- extract_scope_tokens --


class TestExtractScopeTokens:
	def test_from_files_hint(self):
		unit = WorkUnit(files_hint="src/db.py,src/models.py", title="")
		tokens = extract_scope_tokens(unit)
		assert "src/db.py" in tokens
		assert "db.py" in tokens
		assert "src/models.py" in tokens
		assert "models.py" in tokens

	def test_from_title(self):
		unit = WorkUnit(files_hint="", title="refactor scheduler")
		tokens = extract_scope_tokens(unit)
		assert "refactor scheduler" in tokens

	def test_combined(self):
		unit = WorkUnit(files_hint="src/db.py", title="fix db bug")
		tokens = extract_scope_tokens(unit)
		assert "src/db.py" in tokens
		assert "db.py" in tokens
		assert "fix db bug" in tokens

	def test_empty_unit(self):
		unit = WorkUnit(files_hint="", title="")
		tokens = extract_scope_tokens(unit)
		assert tokens == []


# -- format_context_items --


class TestFormatContextItems:
	def test_empty_list(self):
		assert format_context_items([]) == ""

	def test_single_item_full_confidence(self):
		items = [ContextItem(item_type="gotcha", content="Watch out for WAL")]
		result = format_context_items(items)
		assert "[gotcha] Watch out for WAL" in result
		assert "confidence" not in result

	def test_single_item_low_confidence(self):
		items = [ContextItem(item_type="pattern", content="Use dataclasses", confidence=0.7)]
		result = format_context_items(items)
		assert "[pattern] Use dataclasses (confidence: 0.7)" in result

	def test_multiple_items(self):
		items = [
			ContextItem(item_type="gotcha", content="item1"),
			ContextItem(item_type="convention", content="item2"),
		]
		result = format_context_items(items)
		lines = result.strip().split("\n")
		assert len(lines) == 2


# -- inject_context_items (selective injection) --


class TestInjectContextItems:
	def test_no_scope_tokens_returns_empty(self, db):
		unit = WorkUnit(files_hint="", title="")
		result = inject_context_items(unit, db, budget=5000)
		assert result == ""

	def test_no_matching_items_returns_empty(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/cli.py", item_type="gotcha", content="cli stuff",
		))
		unit = WorkUnit(files_hint="src/db.py", title="db work")
		result = inject_context_items(unit, db, budget=5000)
		assert result == ""

	def test_matching_items_injected(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/db.py", item_type="gotcha",
			content="WAL mode required", confidence=0.9,
		))
		unit = WorkUnit(files_hint="src/db.py", title="db migration")
		result = inject_context_items(unit, db, budget=5000)
		assert "### Context from Prior Workers" in result
		assert "WAL mode required" in result

	def test_budget_truncation(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/db.py", item_type="gotcha",
			content="x" * 500, confidence=0.9,
		))
		unit = WorkUnit(files_hint="src/db.py", title="db work")
		result = inject_context_items(unit, db, budget=100)
		# Result is truncated to fit budget (header + truncated content)
		formatted_part = result.replace("### Context from Prior Workers\n", "")
		assert len(formatted_part) <= 100

	def test_integration_with_load_context_for_work_unit(self, db, config):
		plan = Plan(id="plan1", objective="Fix bugs")
		db.insert_plan(plan)
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/scheduler.py", item_type="architectural",
			content="Scheduler uses single-session mode", confidence=0.85,
		))
		unit = WorkUnit(plan_id="plan1", files_hint="src/scheduler.py", title="fix scheduler")
		result = load_context_for_work_unit(unit, db, config)
		assert "### Context from Prior Workers" in result
		assert "Scheduler uses single-session mode" in result

	def test_integration_with_load_context_for_mission_worker(self, db, config):
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/worker.py", item_type="convention",
			content="Workers must emit MC_RESULT", confidence=0.95,
		))
		unit = WorkUnit(files_hint="src/worker.py", title="update worker")
		result = load_context_for_mission_worker(unit, config, db=db)
		assert "### Context from Prior Workers" in result
		assert "Workers must emit MC_RESULT" in result

	def test_mission_worker_without_db_still_works(self, config, tmp_path):
		(tmp_path / "CLAUDE.md").write_text("# Docs")
		unit = WorkUnit(title="task")
		result = load_context_for_mission_worker(unit, config)
		assert "### Project Instructions" in result


# -- ContextScope constants --


class TestContextScope:
	def test_scope_constants_exist(self):
		assert ContextScope.MISSION == "mission"
		assert ContextScope.ROUND == "round"
		assert ContextScope.UNIT == "unit"


# -- Mission-scoped context items --


class TestMissionScopedContextItems:
	def test_insert_and_retrieve_by_mission(self, db):
		item = ContextItem(
			id="ctx-m1", item_type="architectural", scope="src/db.py",
			content="Use WAL mode", source_unit_id="u1", round_id="r1",
			mission_id="mission-1", confidence=0.9,
			scope_level=ContextScope.MISSION,
		)
		db.insert_context_item(item)
		items = db.get_context_items_for_mission("mission-1")
		assert len(items) == 1
		assert items[0].id == "ctx-m1"
		assert items[0].mission_id == "mission-1"
		assert items[0].scope_level == ContextScope.MISSION

	def test_mission_filter_excludes_other_missions(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-m1", mission_id="mission-1", item_type="gotcha",
			content="gotcha for m1", scope_level=ContextScope.MISSION,
		))
		db.insert_context_item(ContextItem(
			id="ctx-m2", mission_id="mission-2", item_type="gotcha",
			content="gotcha for m2", scope_level=ContextScope.MISSION,
		))
		items = db.get_context_items_for_mission("mission-1")
		assert len(items) == 1
		assert items[0].id == "ctx-m1"

	def test_mission_filter_with_scope_level(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-mission", mission_id="m1", item_type="pattern",
			content="mission-level", scope_level=ContextScope.MISSION,
		))
		db.insert_context_item(ContextItem(
			id="ctx-round", mission_id="m1", item_type="pattern",
			content="round-level", scope_level=ContextScope.ROUND,
		))
		db.insert_context_item(ContextItem(
			id="ctx-unit", mission_id="m1", item_type="pattern",
			content="unit-level", scope_level=ContextScope.UNIT,
		))
		mission_only = db.get_context_items_for_mission("m1", scope_level=ContextScope.MISSION)
		assert len(mission_only) == 1
		assert mission_only[0].id == "ctx-mission"

		round_only = db.get_context_items_for_mission("m1", scope_level=ContextScope.ROUND)
		assert len(round_only) == 1
		assert round_only[0].id == "ctx-round"

		all_items = db.get_context_items_for_mission("m1")
		assert len(all_items) == 3

	def test_mission_filter_with_min_confidence(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-low", mission_id="m1", item_type="gotcha",
			content="low confidence", confidence=0.3,
		))
		db.insert_context_item(ContextItem(
			id="ctx-high", mission_id="m1", item_type="gotcha",
			content="high confidence", confidence=0.9,
		))
		items = db.get_context_items_for_mission("m1", min_confidence=0.5)
		assert len(items) == 1
		assert items[0].id == "ctx-high"

	def test_mission_filter_combined_scope_and_confidence(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", mission_id="m1", item_type="gotcha",
			content="mission high", confidence=0.9, scope_level=ContextScope.MISSION,
		))
		db.insert_context_item(ContextItem(
			id="ctx-b", mission_id="m1", item_type="gotcha",
			content="mission low", confidence=0.3, scope_level=ContextScope.MISSION,
		))
		db.insert_context_item(ContextItem(
			id="ctx-c", mission_id="m1", item_type="gotcha",
			content="round high", confidence=0.9, scope_level=ContextScope.ROUND,
		))
		items = db.get_context_items_for_mission(
			"m1", scope_level=ContextScope.MISSION, min_confidence=0.5,
		)
		assert len(items) == 1
		assert items[0].id == "ctx-a"

	def test_mission_items_ordered_by_confidence_desc(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-low", mission_id="m1", confidence=0.5, item_type="a", content="low",
		))
		db.insert_context_item(ContextItem(
			id="ctx-high", mission_id="m1", confidence=0.95, item_type="a", content="high",
		))
		db.insert_context_item(ContextItem(
			id="ctx-mid", mission_id="m1", confidence=0.7, item_type="a", content="mid",
		))
		items = db.get_context_items_for_mission("m1")
		assert items[0].id == "ctx-high"
		assert items[1].id == "ctx-mid"
		assert items[2].id == "ctx-low"

	def test_empty_mission_returns_empty(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", mission_id="m1", item_type="gotcha", content="stuff",
		))
		items = db.get_context_items_for_mission("nonexistent")
		assert items == []

	def test_get_context_item_preserves_mission_id(self, db):
		item = ContextItem(
			id="ctx-m", mission_id="mission-42", item_type="convention",
			content="tabs only", scope_level=ContextScope.UNIT,
		)
		db.insert_context_item(item)
		fetched = db.get_context_item("ctx-m")
		assert fetched is not None
		assert fetched.mission_id == "mission-42"
		assert fetched.scope_level == ContextScope.UNIT


# -- Scope overlap with mission context --


class TestScopeOverlapWithMission:
	def test_scope_overlap_returns_items_with_mission_id(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/db.py", mission_id="m1",
			item_type="gotcha", content="db gotcha", confidence=0.8,
		))
		results = db.get_context_items_by_scope_overlap(["src/db.py"])
		assert len(results) == 1
		assert results[0].mission_id == "m1"

	def test_scope_overlap_across_missions(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-m1", scope="src/db.py", mission_id="m1",
			item_type="gotcha", content="gotcha from m1", confidence=0.8,
		))
		db.insert_context_item(ContextItem(
			id="ctx-m2", scope="src/db.py", mission_id="m2",
			item_type="pattern", content="pattern from m2", confidence=0.9,
		))
		results = db.get_context_items_by_scope_overlap(["src/db.py"])
		assert len(results) == 2


# -- Additional confidence filtering edge cases --


class TestConfidenceFiltering:
	def test_exact_boundary_confidence(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-exact", scope="src/db.py", item_type="gotcha",
			content="boundary", confidence=0.5,
		))
		results = db.get_context_items_by_scope_overlap(["src/db.py"], min_confidence=0.5)
		assert len(results) == 1

	def test_just_below_boundary(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-below", scope="src/db.py", item_type="gotcha",
			content="below", confidence=0.499,
		))
		results = db.get_context_items_by_scope_overlap(["src/db.py"], min_confidence=0.5)
		assert len(results) == 0

	def test_zero_confidence_items(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-zero", scope="src/db.py", item_type="gotcha",
			content="zero", confidence=0.0,
		))
		results = db.get_context_items_by_scope_overlap(["src/db.py"], min_confidence=0.0)
		assert len(results) == 1

	def test_mission_confidence_zero_threshold(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-low", mission_id="m1", item_type="gotcha",
			content="low conf", confidence=0.1,
		))
		items = db.get_context_items_for_mission("m1", min_confidence=0.0)
		assert len(items) == 1


# -- Budget truncation in inject_context_items --


class TestInjectContextItemsBudget:
	def test_budget_zero_truncates_content(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/db.py", item_type="gotcha",
			content="important stuff", confidence=0.9,
		))
		unit = WorkUnit(files_hint="src/db.py", title="db work")
		result = inject_context_items(unit, db, budget=0)
		formatted_part = result.replace("### Context from Prior Workers\n", "")
		assert len(formatted_part) <= 0

	def test_budget_exactly_fits(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-a", scope="src/db.py", item_type="gotcha",
			content="short", confidence=0.9,
		))
		unit = WorkUnit(files_hint="src/db.py", title="db work")
		result = inject_context_items(unit, db, budget=5000)
		assert "### Context from Prior Workers" in result
		assert "short" in result

	def test_many_items_truncated_to_budget(self, db):
		for i in range(50):
			db.insert_context_item(ContextItem(
				id=f"ctx-{i}", scope="src/db.py", item_type="gotcha",
				content=f"discovery number {i} with extra detail " * 5,
				confidence=0.9,
			))
		unit = WorkUnit(files_hint="src/db.py", title="db work")
		result = inject_context_items(unit, db, budget=200)
		formatted_part = result.replace("### Context from Prior Workers\n", "")
		assert len(formatted_part) <= 200

	def test_confidence_filter_in_inject(self, db):
		db.insert_context_item(ContextItem(
			id="ctx-low", scope="src/db.py", item_type="gotcha",
			content="low conf item", confidence=0.2,
		))
		db.insert_context_item(ContextItem(
			id="ctx-high", scope="src/db.py", item_type="gotcha",
			content="high conf item", confidence=0.9,
		))
		unit = WorkUnit(files_hint="src/db.py", title="db work")
		result = inject_context_items(unit, db, budget=5000, min_confidence=0.5)
		assert "high conf item" in result
		assert "low conf item" not in result


# -- format_context_items edge cases --


class TestFormatContextItemsEdgeCases:
	def test_item_with_empty_content(self):
		items = [ContextItem(item_type="gotcha", content="")]
		result = format_context_items(items)
		assert "[gotcha] " in result

	def test_item_with_special_characters(self):
		items = [ContextItem(item_type="convention", content="Use {braces} & <angle> 'quotes'")]
		result = format_context_items(items)
		assert "{braces}" in result
		assert "<angle>" in result

	def test_confidence_exactly_one_no_annotation(self):
		items = [ContextItem(item_type="pattern", content="some pattern", confidence=1.0)]
		result = format_context_items(items)
		assert "confidence" not in result

	def test_confidence_just_below_one(self):
		items = [ContextItem(item_type="pattern", content="some pattern", confidence=0.99)]
		result = format_context_items(items)
		assert "(confidence: 1.0)" in result

	def test_many_items_formatted(self):
		items = [
			ContextItem(item_type=f"type-{i}", content=f"content-{i}")
			for i in range(20)
		]
		result = format_context_items(items)
		lines = result.strip().split("\n")
		assert len(lines) == 20

	def test_multiline_content_preserved(self):
		items = [ContextItem(item_type="gotcha", content="line1\nline2\nline3")]
		result = format_context_items(items)
		assert "line1\nline2\nline3" in result


# -- Episodic memory model tests --


def test_episodic_memory_defaults() -> None:
	em = EpisodicMemory()
	assert em.event_type == ""
	assert em.content == ""
	assert em.outcome == ""
	assert em.scope_tokens == ""
	assert em.confidence == 1.0
	assert em.access_count == 0
	assert em.ttl_days == 30
	assert em.id
	assert em.created_at
	assert em.last_accessed


def test_episodic_memory_fields() -> None:
	em = EpisodicMemory(
		id="em1", event_type="merge_success", content="Tests passed",
		outcome="pass", scope_tokens="auth.py,models.py",
		confidence=0.9, access_count=2, ttl_days=15,
	)
	assert em.id == "em1"
	assert em.event_type == "merge_success"
	assert em.scope_tokens == "auth.py,models.py"
	assert em.confidence == 0.9
	assert em.ttl_days == 15


def test_semantic_memory_defaults() -> None:
	sm = SemanticMemory()
	assert sm.content == ""
	assert sm.source_episode_ids == ""
	assert sm.confidence == 1.0
	assert sm.application_count == 0
	assert sm.id
	assert sm.created_at


def test_semantic_memory_fields() -> None:
	sm = SemanticMemory(
		id="sm1", content="Always run tests before merging",
		source_episode_ids="em1,em2,em3",
		confidence=0.85, application_count=5,
	)
	assert sm.id == "sm1"
	assert sm.content == "Always run tests before merging"
	assert sm.source_episode_ids == "em1,em2,em3"
	assert sm.confidence == 0.85


# -- Episodic memory config tests --


def test_episodic_memory_config_defaults() -> None:
	cfg = EpisodicMemoryConfig()
	assert cfg.enabled is False
	assert cfg.default_ttl_days == 30
	assert cfg.decay_alpha == 0.1
	assert cfg.access_boost_days == 5
	assert cfg.distill_model == "sonnet"
	assert cfg.distill_budget_usd == 0.30
	assert cfg.min_episodes_for_distill == 3
	assert cfg.max_semantic_per_query == 5


def test_episodic_memory_config_toml_parsing(tmp_path: Path) -> None:
	p = tmp_path / "mission-control.toml"
	p.write_text("""\
[target]
name = "test"
path = "."

[episodic_memory]
enabled = true
default_ttl_days = 60
decay_alpha = 0.2
access_boost_days = 10
distill_model = "opus"
min_episodes_for_distill = 5
max_semantic_per_query = 10
""")
	config = load_config(p)
	assert config.episodic_memory.enabled is True
	assert config.episodic_memory.default_ttl_days == 60
	assert config.episodic_memory.decay_alpha == 0.2
	assert config.episodic_memory.access_boost_days == 10
	assert config.episodic_memory.distill_model == "opus"
	assert config.episodic_memory.min_episodes_for_distill == 5


def test_episodic_memory_absent_in_toml(tmp_path: Path) -> None:
	p = tmp_path / "mission-control.toml"
	p.write_text('[target]\nname = "test"\npath = "."\n')
	config = load_config(p)
	assert config.episodic_memory.enabled is False


# -- Episodic/Semantic DB CRUD tests --


def test_insert_and_get_episodic(db: Database) -> None:
	em = EpisodicMemory(
		id="em1", event_type="merge_success", content="Tests passed",
		scope_tokens="auth.py,models.py", confidence=0.9, ttl_days=20,
	)
	db.insert_episodic_memory(em)
	results = db.get_episodic_memories_by_scope(["auth.py"])
	assert len(results) == 1
	assert results[0].id == "em1"
	assert results[0].event_type == "merge_success"


def test_scope_overlap_ordering(db: Database) -> None:
	db.insert_episodic_memory(EpisodicMemory(
		id="em1", scope_tokens="auth.py", access_count=1, confidence=0.5,
	))
	db.insert_episodic_memory(EpisodicMemory(
		id="em2", scope_tokens="auth.py,views.py", access_count=5, confidence=0.9,
	))
	results = db.get_episodic_memories_by_scope(["auth.py"])
	assert len(results) == 2
	assert results[0].id == "em2"  # higher access_count


def test_expired_excluded(db: Database) -> None:
	db.insert_episodic_memory(EpisodicMemory(
		id="em1", scope_tokens="auth.py", ttl_days=0,
	))
	db.insert_episodic_memory(EpisodicMemory(
		id="em2", scope_tokens="auth.py", ttl_days=10,
	))
	results = db.get_episodic_memories_by_scope(["auth.py"])
	assert len(results) == 1
	assert results[0].id == "em2"


def test_update_episodic(db: Database) -> None:
	em = EpisodicMemory(id="em1", content="original", ttl_days=30)
	db.insert_episodic_memory(em)
	em.content = "updated"
	em.ttl_days = 20
	db.update_episodic_memory(em)
	all_mems = db.get_all_episodic_memories()
	assert len(all_mems) == 1
	assert all_mems[0].content == "updated"
	assert all_mems[0].ttl_days == 20


def test_delete_episodic(db: Database) -> None:
	db.insert_episodic_memory(EpisodicMemory(id="em1"))
	db.delete_episodic_memory("em1")
	assert db.get_all_episodic_memories() == []


def test_insert_and_get_semantic(db: Database) -> None:
	sm = SemanticMemory(
		id="sm1", content="Always test auth", source_episode_ids="em1,em2",
		confidence=0.85, application_count=3,
	)
	db.insert_semantic_memory(sm)
	top = db.get_top_semantic_memories(limit=5)
	assert len(top) == 1
	assert top[0].id == "sm1"
	assert top[0].content == "Always test auth"


def test_semantic_ordering(db: Database) -> None:
	db.insert_semantic_memory(SemanticMemory(
		id="sm1", content="low", confidence=0.5, application_count=10,
	))
	db.insert_semantic_memory(SemanticMemory(
		id="sm2", content="high", confidence=0.9, application_count=1,
	))
	top = db.get_top_semantic_memories()
	assert top[0].id == "sm2"  # higher confidence


def test_update_semantic(db: Database) -> None:
	sm = SemanticMemory(id="sm1", content="original", application_count=0)
	db.insert_semantic_memory(sm)
	sm.application_count = 5
	sm.content = "refined"
	db.update_semantic_memory(sm)
	top = db.get_top_semantic_memories()
	assert top[0].application_count == 5
	assert top[0].content == "refined"


# -- MemoryManager tests --


def test_store_episode(manager: MemoryManager, db: Database) -> None:
	em = manager.store_episode(
		event_type="merge_success",
		content="Tests passed for auth module",
		outcome="pass",
		scope_tokens=["auth.py", "models.py"],
	)
	assert em.event_type == "merge_success"
	assert em.scope_tokens == "auth.py,models.py"
	all_mems = db.get_all_episodic_memories()
	assert len(all_mems) == 1


def test_retrieve_bumps_access(manager: MemoryManager, db: Database) -> None:
	db.insert_episodic_memory(EpisodicMemory(
		id="em1", scope_tokens="auth.py", access_count=0, ttl_days=30,
	))
	results = manager.retrieve_relevant(["auth.py"])
	assert len(results) == 1
	# Check access bumped in DB
	all_mems = db.get_all_episodic_memories()
	assert all_mems[0].access_count == 1


def test_decay_tick_reduces_ttl(manager: MemoryManager, db: Database) -> None:
	db.insert_episodic_memory(EpisodicMemory(id="em1", ttl_days=10, access_count=0))
	evicted, extended = manager.decay_tick()
	assert evicted == 0
	assert extended == 0
	all_mems = db.get_all_episodic_memories()
	assert all_mems[0].ttl_days == 9


def test_decay_tick_evicts_expired(manager: MemoryManager, db: Database) -> None:
	db.insert_episodic_memory(EpisodicMemory(id="em1", ttl_days=1, access_count=0))
	evicted, extended = manager.decay_tick()
	assert evicted == 1
	assert db.get_all_episodic_memories() == []


def test_decay_tick_extends_frequently_accessed(manager: MemoryManager, db: Database) -> None:
	db.insert_episodic_memory(EpisodicMemory(id="em1", ttl_days=5, access_count=3))
	evicted, extended = manager.decay_tick()
	assert extended == 1
	all_mems = db.get_all_episodic_memories()
	# 5 + 5 (boost) - 1 (decay) = 9
	assert all_mems[0].ttl_days == 9
	assert all_mems[0].access_count == 0  # counter reset


def test_get_promote_candidates(manager: MemoryManager, db: Database) -> None:
	db.insert_episodic_memory(EpisodicMemory(id="em1", confidence=0.8, ttl_days=2))
	db.insert_episodic_memory(EpisodicMemory(id="em2", confidence=0.5, ttl_days=2))
	db.insert_episodic_memory(EpisodicMemory(id="em3", confidence=0.9, ttl_days=10))
	candidates = manager.get_promote_candidates()
	assert len(candidates) == 1
	assert candidates[0].id == "em1"


@pytest.mark.asyncio()
async def test_distill_below_min_returns_none(manager: MemoryManager) -> None:
	episodes = [EpisodicMemory(id="em1"), EpisodicMemory(id="em2")]
	result = await manager.distill_to_semantic(episodes)
	assert result is None


@pytest.mark.asyncio()
async def test_distill_success(manager: MemoryManager, db: Database) -> None:
	episodes = [
		EpisodicMemory(
			id="em1", event_type="merge_success",
			content="Auth tests passed", outcome="pass", confidence=0.8,
		),
		EpisodicMemory(
			id="em2", event_type="test_failure",
			content="Auth migration broke", outcome="fail", confidence=0.9,
		),
		EpisodicMemory(
			id="em3", event_type="merge_success",
			content="Auth refactor merged", outcome="pass", confidence=0.7,
		),
	]

	mock_proc = AsyncMock()
	mock_proc.communicate = AsyncMock(return_value=(b"Always run auth tests before merging migrations", b""))
	mock_proc.returncode = 0

	with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
		result = await manager.distill_to_semantic(episodes)

	assert result is not None
	assert result.content == "Always run auth tests before merging migrations"
	assert result.source_episode_ids == "em1,em2,em3"
	# avg confidence: (0.8 + 0.9 + 0.7) / 3 = 0.8
	assert abs(result.confidence - 0.8) < 0.01
	# Verify persisted
	top = db.get_top_semantic_memories()
	assert len(top) == 1


@pytest.mark.asyncio()
async def test_distill_llm_failure(manager: MemoryManager) -> None:
	episodes = [EpisodicMemory(id=f"em{i}") for i in range(3)]
	with patch("asyncio.create_subprocess_exec", side_effect=OSError("no claude")):
		result = await manager.distill_to_semantic(episodes)
	assert result is None


# -- Planner context integration tests --


def test_semantic_memories_injected_in_planner_context(db: Database) -> None:
	db.insert_mission(Mission(id="m1", objective="test"))
	epoch = Epoch(id="ep1", mission_id="m1", number=1)
	db.insert_epoch(epoch)
	plan = Plan(id="p1", objective="test")
	db.insert_plan(plan)
	unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
	db.insert_work_unit(unit)
	handoff = Handoff(
		id="h1", work_unit_id="wu1", round_id="", epoch_id="ep1",
		status="completed", summary="Done",
	)
	db.insert_handoff(handoff)

	db.insert_semantic_memory(SemanticMemory(
		id="sm1", content="Always validate inputs before DB writes",
		confidence=0.85, application_count=3,
	))

	result = build_planner_context(db, "m1")
	assert "## Learned Rules (from past missions)" in result
	assert "Always validate inputs before DB writes" in result
	assert "confidence: 0.8" in result


def test_no_semantic_memories_no_section(db: Database) -> None:
	db.insert_mission(Mission(id="m1", objective="test"))
	epoch = Epoch(id="ep1", mission_id="m1", number=1)
	db.insert_epoch(epoch)
	plan = Plan(id="p1", objective="test")
	db.insert_plan(plan)
	unit = WorkUnit(id="wu1", plan_id="p1", title="Task")
	db.insert_work_unit(unit)
	handoff = Handoff(
		id="h1", work_unit_id="wu1", round_id="", epoch_id="ep1",
		status="completed", summary="Done",
	)
	db.insert_handoff(handoff)

	result = build_planner_context(db, "m1")
	assert "## Learned Rules" not in result


# -- build_file_previews --


class TestBuildFilePreviews:
	def test_returns_content_for_existing_files(self, tmp_path):
		(tmp_path / "src").mkdir()
		(tmp_path / "src" / "foo.py").write_text("def hello():\n\treturn 42\n")
		result = build_file_previews("src/foo.py", tmp_path, budget=3000)
		assert "### File Previews" in result
		assert "src/foo.py" in result
		assert "def hello():" in result

	def test_returns_empty_for_missing_files(self, tmp_path):
		result = build_file_previews("nonexistent.py", tmp_path, budget=3000)
		assert result == ""

	def test_respects_budget(self, tmp_path):
		(tmp_path / "big.py").write_text("x = 1\n" * 200)
		result = build_file_previews("big.py", tmp_path, budget=100)
		assert len(result) <= 200  # generous bound for header + truncation overhead

	def test_multiple_files(self, tmp_path):
		(tmp_path / "a.py").write_text("aaa\n")
		(tmp_path / "b.py").write_text("bbb\n")
		result = build_file_previews("a.py,b.py", tmp_path, budget=3000)
		assert "a.py" in result
		assert "b.py" in result

	def test_skips_missing_reads_existing(self, tmp_path):
		(tmp_path / "exists.py").write_text("found\n")
		result = build_file_previews("missing.py,exists.py", tmp_path, budget=3000)
		assert "exists.py" in result
		assert "found" in result
		assert "missing.py" not in result

	def test_empty_hint_returns_empty(self, tmp_path):
		result = build_file_previews("", tmp_path, budget=3000)
		assert result == ""

	def test_zero_budget_returns_empty(self, tmp_path):
		(tmp_path / "foo.py").write_text("content\n")
		result = build_file_previews("foo.py", tmp_path, budget=0)
		assert result == ""

	def test_max_lines_capped(self, tmp_path):
		(tmp_path / "long.py").write_text("line\n" * 100)
		result = build_file_previews("long.py", tmp_path, budget=5000, max_lines=10)
		assert result.count("line") == 10


class TestMissionWorkerFilePreviewsIntegration:
	def test_includes_previews_when_files_exist(self, config, tmp_path):
		(tmp_path / "src").mkdir()
		(tmp_path / "src" / "worker.py").write_text("class Worker:\n\tpass\n")
		unit = WorkUnit(files_hint="src/worker.py", title="update worker")
		result = load_context_for_mission_worker(unit, config)
		assert "### File Previews" in result
		assert "class Worker:" in result

	def test_no_previews_when_no_files_hint(self, config, tmp_path):
		(tmp_path / "CLAUDE.md").write_text("# Docs")
		unit = WorkUnit(title="task")
		result = load_context_for_mission_worker(unit, config)
		assert "### File Previews" not in result

	def test_no_previews_when_files_missing(self, config, tmp_path):
		(tmp_path / "CLAUDE.md").write_text("# Docs")
		unit = WorkUnit(files_hint="nonexistent.py", title="task")
		result = load_context_for_mission_worker(unit, config)
		assert "### File Previews" not in result
