"""Tests for swarm scaling, retry, and pivot execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from autodev.config import SwarmConfig
from autodev.swarm.controller import SwarmController
from autodev.swarm.models import (
	AgentStatus,
	SwarmAgent,
	SwarmTask,
	TaskStatus,
)
from autodev.swarm.stagnation import (
	PivotRecommendation,
	pivots_to_decisions,
)


def _make_config(tmp_path: Path) -> MagicMock:
	config = MagicMock()
	config.target.name = "test-project"
	config.target.objective = "Build a compiler"
	config.target.resolved_path = str(tmp_path)
	config.notification = MagicMock()
	return config


def _make_swarm_config(**overrides: object) -> SwarmConfig:
	sc = SwarmConfig()
	for k, v in overrides.items():
		setattr(sc, k, v)
	return sc


def _make_db() -> MagicMock:
	db = MagicMock()
	db.get_knowledge_for_mission.return_value = []
	return db


def _make_ctrl(tmp_path: Path, **sc_overrides: object) -> SwarmController:
	return SwarmController(
		_make_config(tmp_path),
		_make_swarm_config(**sc_overrides),
		_make_db(),
	)


class TestRequeueFailedTasks:
	def test_requeues_failed_with_budget(self, tmp_path: Path) -> None:
		ctrl = _make_ctrl(tmp_path)
		task = SwarmTask(title="Fix bug", max_attempts=3)
		task.status = TaskStatus.FAILED
		task.attempt_count = 1
		task.claimed_by = "agent-1"
		ctrl._tasks[task.id] = task

		requeued = ctrl.requeue_failed_tasks()
		assert task.id in requeued
		assert task.status == TaskStatus.PENDING
		assert task.claimed_by is None

	def test_does_not_requeue_exhausted(self, tmp_path: Path) -> None:
		ctrl = _make_ctrl(tmp_path)
		task = SwarmTask(title="Fix bug", max_attempts=3)
		task.status = TaskStatus.FAILED
		task.attempt_count = 3
		ctrl._tasks[task.id] = task

		requeued = ctrl.requeue_failed_tasks()
		assert requeued == []
		assert task.status == TaskStatus.FAILED

	def test_ignores_non_failed(self, tmp_path: Path) -> None:
		ctrl = _make_ctrl(tmp_path)
		task = SwarmTask(title="In progress")
		task.status = TaskStatus.IN_PROGRESS
		ctrl._tasks[task.id] = task

		requeued = ctrl.requeue_failed_tasks()
		assert requeued == []


class TestGetIdleAgents:
	def test_returns_idle_agents(self, tmp_path: Path) -> None:
		ctrl = _make_ctrl(tmp_path)
		idle = SwarmAgent(name="idle-1", status=AgentStatus.IDLE)
		working = SwarmAgent(name="working-1", status=AgentStatus.WORKING)
		ctrl._agents[idle.id] = idle
		ctrl._agents[working.id] = working

		result = ctrl.get_idle_agents()
		assert len(result) == 1
		assert result[0].name == "idle-1"

	def test_empty_when_all_working(self, tmp_path: Path) -> None:
		ctrl = _make_ctrl(tmp_path)
		a = SwarmAgent(name="w1", status=AgentStatus.WORKING)
		ctrl._agents[a.id] = a
		assert ctrl.get_idle_agents() == []


class TestGetScalingRecommendation:
	def test_scale_up_when_many_pending(self, tmp_path: Path) -> None:
		ctrl = _make_ctrl(tmp_path)
		agent = SwarmAgent(name="w1", status=AgentStatus.WORKING)
		ctrl._agents[agent.id] = agent
		for i in range(5):
			t = SwarmTask(title=f"Task {i}")
			ctrl._tasks[t.id] = t

		rec = ctrl.get_scaling_recommendation()
		assert rec["scale_up"] > 0

	def test_scale_down_when_idle(self, tmp_path: Path) -> None:
		ctrl = _make_ctrl(tmp_path)
		for i in range(3):
			a = SwarmAgent(name=f"idle-{i}", status=AgentStatus.IDLE)
			ctrl._agents[a.id] = a

		rec = ctrl.get_scaling_recommendation()
		assert rec["scale_down"] == 3

	def test_no_scaling_when_balanced(self, tmp_path: Path) -> None:
		ctrl = _make_ctrl(tmp_path)
		a = SwarmAgent(name="w1", status=AgentStatus.WORKING)
		ctrl._agents[a.id] = a
		t = SwarmTask(title="One task")
		ctrl._tasks[t.id] = t

		rec = ctrl.get_scaling_recommendation()
		assert rec["scale_up"] == 0
		assert rec["scale_down"] == 0


class TestPivotsToDecisions:
	def test_research_pivot(self) -> None:
		pivots = [PivotRecommendation(
			trigger="Tests flat",
			strategy="research_before_implement",
			severity="critical",
			details="Research first.",
		)]
		decisions = pivots_to_decisions(pivots)
		assert len(decisions) == 1
		assert decisions[0]["type"] == "create_task"
		assert decisions[0]["payload"]["priority"] == 3

	def test_reduce_pivot(self) -> None:
		pivots = [PivotRecommendation(
			trigger="Cost rising",
			strategy="reduce_and_focus",
			severity="warning",
		)]
		decisions = pivots_to_decisions(pivots)
		assert len(decisions) == 1
		assert decisions[0]["type"] == "adjust"
		assert decisions[0]["payload"]["max_agents"] == 2

	def test_diagnose_pivot(self) -> None:
		pivots = [PivotRecommendation(
			trigger="High failures",
			strategy="diagnose_systemic",
			severity="critical",
			details="Find root cause.",
		)]
		decisions = pivots_to_decisions(pivots)
		assert len(decisions) == 1
		assert decisions[0]["type"] == "create_task"
		assert "Diagnose" in decisions[0]["payload"]["title"]

	def test_empty_pivots(self) -> None:
		assert pivots_to_decisions([]) == []

	def test_multiple_pivots(self) -> None:
		pivots = [
			PivotRecommendation(trigger="a", strategy="research_before_implement", severity="critical"),
			PivotRecommendation(trigger="b", strategy="reduce_and_focus", severity="warning"),
		]
		decisions = pivots_to_decisions(pivots)
		assert len(decisions) == 2
