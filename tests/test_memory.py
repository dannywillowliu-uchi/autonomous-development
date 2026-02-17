"""Tests for memory.py context loading."""

from __future__ import annotations

import pytest

from mission_control.config import MissionConfig, TargetConfig
from mission_control.db import Database
from mission_control.memory import (
	CONTEXT_BUDGET,
	_format_session_history,
	compress_history,
	extract_scope_tokens,
	format_context_items,
	inject_context_items,
	load_context,
	load_context_for_mission_worker,
	load_context_for_work_unit,
	summarize_session,
)
from mission_control.models import ContextItem, Decision, Plan, Session, TaskRecord, WorkUnit
from mission_control.reviewer import ReviewVerdict


@pytest.fixture
def db() -> Database:
	return Database(":memory:")


@pytest.fixture
def config(tmp_path) -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(path=str(tmp_path))
	return mc


def _make_session(id: str = "s1", status: str = "completed", desc: str = "fix bug", summary: str = "done") -> Session:
	return Session(id=id, task_description=desc, status=status, output_summary=summary)


def _make_decision(decision: str = "use pytest", rationale: str = "standard") -> Decision:
	return Decision(session_id="s1", decision=decision, rationale=rationale)


# -- load_context --


class TestLoadContext:
	def test_empty_db_returns_empty(self, db, config):
		task = TaskRecord(description="do something")
		result = load_context(task, db, config)
		assert result == ""

	def test_includes_session_history(self, db, config):
		db.insert_session(_make_session())
		task = TaskRecord(description="do something")
		result = load_context(task, db, config)
		assert "### Recent Sessions" in result
		assert "fix bug" in result

	def test_includes_decisions(self, db, config):
		db.insert_session(_make_session())
		db.insert_decision(_make_decision())
		task = TaskRecord(description="do something")
		result = load_context(task, db, config)
		assert "### Recent Decisions" in result
		assert "use pytest" in result
		assert "standard" in result

	def test_includes_claude_md(self, db, config, tmp_path):
		(tmp_path / "CLAUDE.md").write_text("# Project Docs\nImportant info here.")
		task = TaskRecord(description="do something")
		result = load_context(task, db, config)
		assert "### Project Instructions" in result
		assert "# Project Docs" in result

	def test_budget_enforcement(self, db, config):
		"""Output stays within CONTEXT_BUDGET chars even with lots of data."""
		for i in range(20):
			db.insert_session(_make_session(
				id=f"sess-{i}",
				desc=f"task number {i} with long description " * 10,
				summary="summary " * 50,
			))
		for i in range(20):
			d = _make_decision(decision=f"decision-{i} " * 20, rationale=f"rationale-{i} " * 20)
			d.session_id = "sess-0"  # reference an existing session
			db.insert_decision(d)
		task = TaskRecord(description="do something")
		result = load_context(task, db, config)
		assert len(result) <= CONTEXT_BUDGET + 500  # some overhead for section headers and joining


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


# -- summarize_session --


class TestSummarizeSession:
	def test_basic_format(self):
		session = _make_session(id="abc123", desc="add logging")
		verdict = ReviewVerdict(verdict="helped")
		result = summarize_session(session, verdict)
		assert "Session abc123 (add logging):" in result
		assert "Verdict: helped." in result

	def test_with_improvements(self):
		session = _make_session(id="s1", desc="refactor")
		verdict = ReviewVerdict(verdict="helped", improvements=["reduced complexity", "added types"])
		result = summarize_session(session, verdict)
		assert "Improved: reduced complexity, added types." in result

	def test_with_regressions(self):
		session = _make_session(id="s1", desc="deploy")
		verdict = ReviewVerdict(verdict="hurt", regressions=["broke API", "lost data"])
		result = summarize_session(session, verdict)
		assert "Regressed: broke API, lost data." in result

	def test_full_combo(self):
		session = _make_session(id="s1", desc="big change", summary="lots happened")
		verdict = ReviewVerdict(
			verdict="helped",
			improvements=["test coverage"],
			regressions=["minor lint"],
		)
		result = summarize_session(session, verdict)
		assert "Session s1 (big change):" in result
		assert "Verdict: helped." in result
		assert "Improved: test coverage." in result
		assert "Regressed: minor lint." in result
		assert "Output: lots happened" in result


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
