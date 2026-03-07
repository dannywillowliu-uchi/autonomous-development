"""Tests for swarm context synthesizer."""

from __future__ import annotations

from unittest.mock import MagicMock

from autodev.swarm.context import ContextSynthesizer
from autodev.swarm.models import (
	AgentRole,
	AgentStatus,
	SwarmAgent,
	SwarmTask,
	TaskStatus,
)


def _make_config() -> MagicMock:
	config = MagicMock()
	config.target.name = "test-project"
	config.target.objective = "Build a compiler"
	config.target.resolved_path = "/tmp/test-project"
	return config


def _make_db() -> MagicMock:
	db = MagicMock()
	db.get_knowledge_for_mission.return_value = []
	return db


class TestContextSynthesizer:
	def test_build_state_empty(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		assert state.cycle_number == 1
		assert state.agents == []
		assert state.tasks == []
		assert state.mission_objective == "Build a compiler"

	def test_build_state_increments_cycle(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state1 = ctx.build_state(agents=[], tasks=[])
		state2 = ctx.build_state(agents=[], tasks=[])
		assert state1.cycle_number == 1
		assert state2.cycle_number == 2

	def test_recent_completions(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title="Done task", status=TaskStatus.COMPLETED, result_summary="Fixed it"),
			SwarmTask(title="Pending task", status=TaskStatus.PENDING),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		assert len(state.recent_completions) == 1
		assert state.recent_completions[0]["title"] == "Done task"

	def test_recent_failures(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title="Failed task", status=TaskStatus.FAILED, attempt_count=2, result_summary="Segfault"),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		assert len(state.recent_failures) == 1
		assert state.recent_failures[0]["attempt"] == 2

	def test_files_in_flight(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		task = SwarmTask(id="t1", title="Fix parser", status=TaskStatus.IN_PROGRESS, files_hint=["src/parser.c"])
		agent = SwarmAgent(name="worker-1", status=AgentStatus.WORKING, current_task_id="t1")
		state = ctx.build_state(agents=[agent], tasks=[task])
		assert "src/parser.c" in state.files_in_flight

	def test_core_test_results_passed_through(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		results = {"pass": 216, "fail": 5, "skip": 0, "total": 221}
		state = ctx.build_state(agents=[], tasks=[], core_test_results=results)
		assert state.core_test_results["pass"] == 216


class TestStagnationDetection:
	def test_no_stagnation_initially(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[], core_test_results={"pass": 200})
		assert state.stagnation_signals == []

	def test_stagnation_after_flat_metric(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		ctx._stagnation_window = 3
		# Simulate 3 cycles with same test count
		for _ in range(3):
			state = ctx.build_state(agents=[], tasks=[], core_test_results={"pass": 200})
		# By cycle 3, should detect stagnation
		assert any(s.metric == "test_pass_count" for s in state.stagnation_signals)

	def test_no_stagnation_when_improving(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		ctx._stagnation_window = 3
		for i in range(5):
			state = ctx.build_state(agents=[], tasks=[], core_test_results={"pass": 200 + i})
		assert not any(s.metric == "test_pass_count" for s in state.stagnation_signals)

	def test_high_failure_rate_signal(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title="t1", status=TaskStatus.FAILED),
			SwarmTask(title="t2", status=TaskStatus.FAILED),
			SwarmTask(title="t3", status=TaskStatus.COMPLETED),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		assert any(s.metric == "high_failure_rate" for s in state.stagnation_signals)


class TestRenderForPlanner:
	def test_render_includes_mission(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		rendered = ctx.render_for_planner(state)
		assert "Build a compiler" in rendered

	def test_render_includes_agents(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		agent = SwarmAgent(name="impl-1", role=AgentRole.IMPLEMENTER, status=AgentStatus.WORKING)
		state = ctx.build_state(agents=[agent], tasks=[])
		rendered = ctx.render_for_planner(state)
		assert "impl-1" in rendered
		assert "implementer" in rendered

	def test_render_includes_stagnation_warnings(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		ctx._stagnation_window = 2
		for _ in range(2):
			state = ctx.build_state(agents=[], tasks=[], core_test_results={"pass": 100})
		rendered = ctx.render_for_planner(state)
		assert "STAGNATION" in rendered

	def test_render_includes_core_tests(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		results = {"pass": 216, "fail": 5, "skip": 0, "total": 221}
		state = ctx.build_state(agents=[], tasks=[], core_test_results=results)
		rendered = ctx.render_for_planner(state)
		assert "216" in rendered
		assert "Core Test" in rendered
