"""Tests for swarm planner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from autodev.config import SwarmConfig
from autodev.swarm.models import (
	AgentStatus,
	DecisionType,
	SwarmState,
	TaskStatus,
)
from autodev.swarm.planner import DrivingPlanner


def _make_controller() -> MagicMock:
	ctrl = MagicMock()
	ctrl._config = MagicMock()
	ctrl._config.target.resolved_path = "/tmp/test"
	ctrl.execute_decisions = AsyncMock(return_value=[])
	ctrl.monitor_agents = AsyncMock(return_value=[])
	ctrl.cleanup = AsyncMock()
	ctrl.build_state = MagicMock(return_value=SwarmState(
		mission_objective="Test mission",
		agents=[],
		tasks=[],
	))
	ctrl.render_state = MagicMock(return_value="## State\nNo agents.")
	return ctrl


def _make_swarm_config(**overrides: object) -> SwarmConfig:
	sc = SwarmConfig()
	for k, v in overrides.items():
		setattr(sc, k, v)
	return sc


class TestParsDecisions:
	def test_parse_valid_json_array(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		decisions_json = json.dumps([
			{
				"type": "spawn",
				"payload": {"role": "implementer", "name": "test-agent"},
				"reasoning": "Need an agent",
				"priority": 5,
			},
			{
				"type": "wait",
				"payload": {"duration": 30, "reason": "waiting"},
				"reasoning": "All good",
				"priority": 0,
			},
		])
		result = planner._parse_decisions(decisions_json)
		assert len(result) == 2
		assert result[0].type == DecisionType.SPAWN
		assert result[0].priority == 5
		assert result[1].type == DecisionType.WAIT

	def test_parse_json_with_surrounding_text(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		response = (
			'Here are my decisions:\n```json\n'
			'[{"type": "wait", "payload": {"duration": 10}, '
			'"reasoning": "ok", "priority": 0}]\n```\nDone.'
		)
		result = planner._parse_decisions(response)
		assert len(result) == 1
		assert result[0].type == DecisionType.WAIT

	def test_parse_empty_array(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		result = planner._parse_decisions("[]")
		assert result == []

	def test_parse_no_json_returns_empty(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		result = planner._parse_decisions("No decisions needed.")
		assert result == []

	def test_parse_invalid_json_returns_empty(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		result = planner._parse_decisions("[{broken json")
		assert result == []

	def test_parse_skips_invalid_decision_type(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		decisions_json = json.dumps([
			{"type": "spawn", "payload": {}, "reasoning": "ok", "priority": 1},
			{"type": "invalid_type", "payload": {}, "reasoning": "bad"},
			{"type": "kill", "payload": {}, "reasoning": "ok", "priority": 2},
		])
		result = planner._parse_decisions(decisions_json)
		assert len(result) == 2
		assert result[0].type == DecisionType.SPAWN
		assert result[1].type == DecisionType.KILL

	def test_parse_all_decision_types(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		all_types = [
			"spawn", "kill", "redirect", "create_task",
			"adjust", "wait", "escalate", "create_skill",
		]
		items = [
			{"type": t, "payload": {}, "reasoning": f"test {t}", "priority": i}
			for i, t in enumerate(all_types)
		]
		result = planner._parse_decisions(json.dumps(items))
		assert len(result) == len(all_types)
		parsed_types = {d.type.value for d in result}
		assert parsed_types == set(all_types)


class TestShouldPlan:
	def _make_state(self, agents=None, tasks=None) -> SwarmState:
		return SwarmState(
			mission_objective="Test",
			agents=agents or [],
			tasks=tasks or [],
		)

	def test_events_trigger_planning(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		state = self._make_state()
		assert planner._should_plan(state, [{"event": "completed"}]) is True

	def test_no_active_agents_with_tasks_triggers(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		task = MagicMock()
		task.status = TaskStatus.PENDING
		state = self._make_state(tasks=[task])
		assert planner._should_plan(state, []) is True

	def test_pending_tasks_with_capacity_triggers(self) -> None:
		planner = DrivingPlanner(
			_make_controller(), _make_swarm_config(max_agents=3)
		)
		agent = MagicMock()
		agent.status = AgentStatus.WORKING
		task = MagicMock()
		task.status = TaskStatus.PENDING
		state = self._make_state(agents=[agent], tasks=[task])
		assert planner._should_plan(state, []) is True

	def test_at_max_agents_no_trigger(self) -> None:
		planner = DrivingPlanner(
			_make_controller(), _make_swarm_config(max_agents=1)
		)
		agent = MagicMock()
		agent.status = AgentStatus.WORKING
		task = MagicMock()
		task.status = TaskStatus.PENDING
		state = self._make_state(agents=[agent], tasks=[task])
		assert planner._should_plan(state, []) is False

	def test_no_events_no_tasks_no_trigger(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		state = self._make_state()
		assert planner._should_plan(state, []) is False


class TestShouldStop:
	def test_all_done_stops(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		task = MagicMock()
		task.status = TaskStatus.COMPLETED
		state = SwarmState(
			mission_objective="Test",
			agents=[],
			tasks=[task],
		)
		assert planner._should_stop(state) is True

	def test_pending_tasks_dont_stop(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		task = MagicMock()
		task.status = TaskStatus.PENDING
		state = SwarmState(
			mission_objective="Test",
			agents=[],
			tasks=[task],
		)
		assert planner._should_stop(state) is False

	def test_active_agents_dont_stop(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		agent = MagicMock()
		agent.status = AgentStatus.WORKING
		task = MagicMock()
		task.status = TaskStatus.COMPLETED
		state = SwarmState(
			mission_objective="Test",
			agents=[agent],
			tasks=[task],
		)
		assert planner._should_stop(state) is False

	def test_no_tasks_dont_stop(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		state = SwarmState(
			mission_objective="Test",
			agents=[],
			tasks=[],
		)
		assert planner._should_stop(state) is False


class TestRecordMetrics:
	def test_records_completion_and_failure_counts(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		completed_task = MagicMock()
		completed_task.status = TaskStatus.COMPLETED
		failed_task = MagicMock()
		failed_task.status = TaskStatus.FAILED
		pending_task = MagicMock()
		pending_task.status = TaskStatus.PENDING
		state = SwarmState(
			mission_objective="Test",
			agents=[],
			tasks=[completed_task, failed_task, pending_task],
			total_cost_usd=1.5,
		)
		planner._record_metrics(state)
		assert planner._completion_history == [1]
		assert planner._failure_history == [1]
		assert planner._cost_history == [1.5]

	def test_records_test_results(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		state = SwarmState(
			mission_objective="Test",
			agents=[],
			tasks=[],
			core_test_results={"pass": 42, "fail": 3},
		)
		planner._record_metrics(state)
		assert planner._test_history == [42]

	def test_no_test_results_skips_test_history(self) -> None:
		planner = DrivingPlanner(_make_controller(), _make_swarm_config())
		state = SwarmState(
			mission_objective="Test",
			agents=[],
			tasks=[],
		)
		planner._record_metrics(state)
		assert planner._test_history == []
		assert planner._completion_history == [0]
