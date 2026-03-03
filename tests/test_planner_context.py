"""Tests for planner_context: build_planner_context and update_mission_state."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from mission_control.batch_analyzer import EpochCostSummary
from mission_control.config import MissionConfig, TargetConfig, VerificationConfig
from mission_control.models import Epoch, Handoff, KnowledgeItem, Mission, SemanticMemory, WorkUnit
from mission_control.planner_context import build_planner_context, update_mission_state


def _mock_db() -> MagicMock:
	"""Create a mock Database with all methods returning empty defaults."""
	db = MagicMock()
	db.get_recent_handoffs.return_value = []
	db.get_top_semantic_memories.return_value = []
	db.get_work_units_for_mission.return_value = []
	db.get_epochs_for_mission.return_value = []
	db.get_knowledge_for_mission.return_value = []
	db.get_unit_events_for_mission.return_value = []
	return db


def _make_handoff(
	work_unit_id: str = "abcd1234efgh",
	status: str = "failed",
	concerns: list[str] | None = None,
	files_changed: list[str] | None = None,
) -> Handoff:
	return Handoff(
		work_unit_id=work_unit_id,
		status=status,
		concerns=concerns or [],
		files_changed=files_changed or [],
	)


def _make_config(tmp_path: object) -> MissionConfig:
	cfg = MissionConfig()
	cfg.target = TargetConfig(
		name="test-proj",
		path=str(tmp_path),
		branch="main",
		verification=VerificationConfig(command="pytest -q"),
	)
	return cfg


# ---------------------------------------------------------------------------
# build_planner_context
# ---------------------------------------------------------------------------


class TestBuildPlannerContext:
	def test_no_failures_no_memories(self) -> None:
		db = _mock_db()
		result = build_planner_context(db, "m1")
		assert result == ""

	def test_failed_handoffs_shows_last_three(self) -> None:
		db = _mock_db()
		handoffs = [
			_make_handoff(work_unit_id=f"unit{i:04d}xxxxxxxx", concerns=[f"err{i}"])
			for i in range(5)
		]
		# Mix in a completed one that should be excluded
		handoffs.insert(2, _make_handoff(work_unit_id="completedxxxxxxx", status="completed"))
		db.get_recent_handoffs.return_value = handoffs

		result = build_planner_context(db, "m1")
		assert "## Recent Failures" in result
		# Should only show last 3 of the 5 failed (indices 2,3,4)
		assert "err2" in result
		assert "err3" in result
		assert "err4" in result
		# First two failures should not appear
		assert "err0" not in result
		assert "err1" not in result

	def test_failed_handoff_no_concerns_shows_unknown(self) -> None:
		db = _mock_db()
		db.get_recent_handoffs.return_value = [
			_make_handoff(concerns=[]),
		]
		result = build_planner_context(db, "m1")
		assert "unknown" in result

	def test_concern_truncated_at_300(self) -> None:
		db = _mock_db()
		long_concern = "x" * 500
		db.get_recent_handoffs.return_value = [
			_make_handoff(concerns=[long_concern]),
		]
		result = build_planner_context(db, "m1")
		# The detail should be truncated to 300 chars
		assert "x" * 300 in result
		assert "x" * 301 not in result

	def test_semantic_memories_with_full_confidence(self) -> None:
		db = _mock_db()
		db.get_top_semantic_memories.return_value = [
			SemanticMemory(content="Always run tests before merge", confidence=1.0),
		]
		result = build_planner_context(db, "m1")
		assert "## Learned Rules (from past missions)" in result
		assert "- Always run tests before merge" in result
		# Full confidence should NOT show the confidence suffix
		assert "confidence" not in result

	def test_semantic_memories_with_low_confidence(self) -> None:
		db = _mock_db()
		db.get_top_semantic_memories.return_value = [
			SemanticMemory(content="Prefer small PRs", confidence=0.7),
		]
		result = build_planner_context(db, "m1")
		assert "- Prefer small PRs (confidence: 0.7)" in result

	def test_both_failures_and_memories(self) -> None:
		db = _mock_db()
		db.get_recent_handoffs.return_value = [
			_make_handoff(concerns=["timeout on CI"]),
		]
		db.get_top_semantic_memories.return_value = [
			SemanticMemory(content="Rule A", confidence=0.9),
		]
		result = build_planner_context(db, "m1")
		assert "## Recent Failures" in result
		assert "timeout on CI" in result
		assert "## Learned Rules (from past missions)" in result
		assert "Rule A (confidence: 0.9)" in result

	def test_resilient_when_handoffs_raises(self) -> None:
		db = _mock_db()
		db.get_recent_handoffs.side_effect = RuntimeError("db locked")
		db.get_top_semantic_memories.return_value = [
			SemanticMemory(content="still works", confidence=1.0),
		]
		result = build_planner_context(db, "m1")
		# Handoff failure is swallowed; memories still appear
		assert "## Learned Rules" in result
		assert "still works" in result

	def test_resilient_when_memories_raises(self) -> None:
		db = _mock_db()
		db.get_recent_handoffs.return_value = [
			_make_handoff(concerns=["fail reason"]),
		]
		db.get_top_semantic_memories.side_effect = RuntimeError("corrupt")
		result = build_planner_context(db, "m1")
		# Memory failure is swallowed; failures still appear
		assert "## Recent Failures" in result
		assert "fail reason" in result

	def test_resilient_when_both_raise(self) -> None:
		db = _mock_db()
		db.get_recent_handoffs.side_effect = RuntimeError("boom")
		db.get_top_semantic_memories.side_effect = RuntimeError("bang")
		result = build_planner_context(db, "m1")
		assert result == ""

	def test_cost_trend_appears_when_epochs_have_data(self) -> None:
		db = _mock_db()
		db.get_epochs_for_mission.return_value = [
			Epoch(id="ep1", mission_id="m1", number=1),
		]
		db.get_work_units_for_mission.return_value = [
			WorkUnit(
				id="wu1", plan_id="p1", title="T",
				epoch_id="ep1", status="completed",
				cost_usd=2.0, input_tokens=100, output_tokens=50,
			),
		]
		from mission_control.models import UnitEvent
		db.get_unit_events_for_mission.return_value = [
			UnitEvent(
				mission_id="m1", epoch_id="ep1", work_unit_id="wu1",
				event_type="dispatched",
			),
			UnitEvent(
				mission_id="m1", epoch_id="ep1", work_unit_id="wu1",
				event_type="merged",
			),
		]
		result = build_planner_context(db, "m1")
		assert "## Cost Trend (recent epochs)" in result
		assert "| Epoch |" in result
		assert "ep1" in result

	def test_cost_trend_absent_when_no_epochs(self) -> None:
		db = _mock_db()
		result = build_planner_context(db, "m1")
		assert "Cost Trend" not in result

	def test_cost_trend_resilient_on_error(self) -> None:
		db = _mock_db()
		db.get_epochs_for_mission.side_effect = RuntimeError("fail")
		db.get_recent_handoffs.return_value = [
			_make_handoff(concerns=["err"]),
		]
		result = build_planner_context(db, "m1")
		assert "## Recent Failures" in result
		assert "Cost Trend" not in result


# ---------------------------------------------------------------------------
# update_mission_state
# ---------------------------------------------------------------------------


class TestUpdateMissionState:
	def test_basic_progress_counts(self, tmp_path: object) -> None:
		db = _mock_db()
		db.get_work_units_for_mission.return_value = [
			WorkUnit(title="Unit A", status="completed", finished_at="2026-01-01T00:00:00"),
			WorkUnit(title="Unit B", status="failed"),
			WorkUnit(title="Unit C", status="completed", finished_at="2026-01-02T00:00:00"),
		]
		db.get_epochs_for_mission.return_value = [Epoch(), Epoch()]

		mission = Mission(id="m1", objective="Build it")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "Objective: Build it" in state
		assert "2 tasks complete, 1 failed. Epoch 2." in state
		# Last completed should be Unit C (last in reversed order with completed status)
		assert '"Unit C"' in state
		assert "2026-01-02T00:00:00" in state

	def test_no_completed_units(self, tmp_path: object) -> None:
		db = _mock_db()
		db.get_work_units_for_mission.return_value = [
			WorkUnit(title="Unit A", status="pending"),
		]

		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "0 tasks complete, 0 failed. Epoch 0." in state
		assert "Last completed" not in state

	def test_strategy_truncation_at_500(self, tmp_path: object) -> None:
		db = _mock_db()
		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)
		long_strategy = "s" * 600

		update_mission_state(db, mission, cfg, strategy=long_strategy)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "## Strategy" in state
		assert "s" * 500 in state
		assert "s" * 501 not in state

	def test_failed_handoffs_in_active_issues(self, tmp_path: object) -> None:
		db = _mock_db()
		db.get_recent_handoffs.return_value = [
			_make_handoff(work_unit_id="aaaa1111xxxx", status="failed", concerns=["oom"]),
			_make_handoff(work_unit_id="bbbb2222xxxx", status="completed"),
			_make_handoff(work_unit_id="cccc3333xxxx", status="blocked", concerns=["dep missing"]),
		]

		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "## Active Issues" in state
		assert "aaaa1111: oom" in state
		assert "cccc3333: dep missing" in state
		# Completed handoff should not appear as active issue
		assert "bbbb2222" not in state

	def test_active_issues_unknown_when_no_concerns(self, tmp_path: object) -> None:
		db = _mock_db()
		db.get_recent_handoffs.return_value = [
			_make_handoff(status="failed", concerns=[]),
		]

		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "unknown" in state

	def test_reflection_patterns_and_tensions(self, tmp_path: object) -> None:
		db = _mock_db()
		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		reflection = SimpleNamespace(
			patterns=["Pattern A", "Pattern B"],
			tensions=["Tension X"],
			open_questions=["Why does Y happen?"],
		)

		update_mission_state(db, mission, cfg, reflection=reflection)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "## Patterns (from reflection)" in state
		assert "- Pattern A" in state
		assert "- Pattern B" in state
		assert "Tensions:" in state
		assert "- Tension X" in state
		assert "## Open Questions" in state
		assert "- Why does Y happen?" in state

	def test_reflection_no_patterns_no_section(self, tmp_path: object) -> None:
		db = _mock_db()
		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		reflection = SimpleNamespace(patterns=[], tensions=[], open_questions=[])

		update_mission_state(db, mission, cfg, reflection=reflection)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "## Patterns" not in state

	def test_low_confidence_knowledge_items(self, tmp_path: object) -> None:
		db = _mock_db()
		db.get_knowledge_for_mission.return_value = [
			KnowledgeItem(
				source_unit_type="design",
				title="Use REST",
				content="REST over gRPC for simplicity",
				confidence=1.0,
			),
			KnowledgeItem(
				source_unit_type="research",
				title="Maybe caching",
				content="Redis might help with latency",
				confidence=0.5,
			),
		]

		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		# Design item should appear in Key Decisions
		assert "## Key Decisions" in state
		assert "Use REST" in state
		# Low-confidence item should appear in Open Questions
		assert "[0.5] Maybe caching" in state

	def test_low_confidence_skipped_when_reflection_has_open_questions(self, tmp_path: object) -> None:
		"""Low-confidence items still appear but under the existing Open Questions header."""
		db = _mock_db()
		db.get_knowledge_for_mission.return_value = [
			KnowledgeItem(
				source_unit_type="research",
				title="Uncertain thing",
				content="maybe",
				confidence=0.3,
			),
		]

		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)
		reflection = SimpleNamespace(
			patterns=[],
			tensions=[],
			open_questions=["Existing question?"],
		)

		update_mission_state(db, mission, cfg, reflection=reflection)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		# Should NOT add a second "## Open Questions" header
		assert state.count("## Open Questions") == 1
		assert "Existing question?" in state
		assert "[0.3] Uncertain thing" in state

	def test_files_modified_grouped_by_directory(self, tmp_path: object) -> None:
		db = _mock_db()
		db.get_recent_handoffs.return_value = [
			_make_handoff(
				status="completed",
				files_changed=["src/module/foo.py", "src/module/bar.py", "tests/test_foo.py"],
			),
			_make_handoff(
				status="completed",
				files_changed=["src/module/foo.py", "docs/readme.md"],
			),
		]

		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "## Files Modified" in state
		# Grouped by directory
		assert "src/module/: bar.py, foo.py" in state
		assert "tests/: test_foo.py" in state
		assert "docs/: readme.md" in state

	def test_files_in_root_directory(self, tmp_path: object) -> None:
		db = _mock_db()
		db.get_recent_handoffs.return_value = [
			_make_handoff(status="completed", files_changed=["setup.py"]),
		]

		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "./: setup.py" in state

	def test_db_exceptions_produce_empty_defaults(self, tmp_path: object) -> None:
		db = _mock_db()
		db.get_work_units_for_mission.side_effect = RuntimeError("gone")
		db.get_recent_handoffs.side_effect = RuntimeError("gone")
		db.get_epochs_for_mission.side_effect = RuntimeError("gone")

		mission = Mission(id="m1", objective="still works")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "Objective: still works" in state
		assert "0 tasks complete, 0 failed. Epoch 0." in state

	def test_knowledge_exception_swallowed(self, tmp_path: object) -> None:
		db = _mock_db()
		db.get_knowledge_for_mission.side_effect = RuntimeError("corrupt")

		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		# Should not crash; file should still be written
		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "# Mission State" in state

	def test_last_completed_title_truncated_at_80(self, tmp_path: object) -> None:
		db = _mock_db()
		long_title = "A" * 120
		db.get_work_units_for_mission.return_value = [
			WorkUnit(title=long_title, status="completed", finished_at="2026-01-01"),
		]

		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		# Title should be truncated to 80 chars
		assert f'"{long_title[:80]}"' in state
		assert long_title[:81] not in state

	def test_concern_detail_truncated_at_200_in_active_issues(self, tmp_path: object) -> None:
		db = _mock_db()
		long_concern = "c" * 300
		db.get_recent_handoffs.return_value = [
			_make_handoff(concerns=[long_concern]),
		]

		mission = Mission(id="m1", objective="test")
		cfg = _make_config(tmp_path)

		update_mission_state(db, mission, cfg)

		state = (tmp_path / "MISSION_STATE.md").read_text()  # type: ignore[operator]
		assert "c" * 200 in state
		assert "c" * 201 not in state
