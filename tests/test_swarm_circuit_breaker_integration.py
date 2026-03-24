"""Integration tests for circuit breaker wired into SwarmController."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from autodev.config import SwarmConfig
from autodev.swarm.circuit_breaker import SwarmCircuitBreakerState
from autodev.swarm.controller import SwarmController
from autodev.swarm.models import (
	AgentStatus,
	DecisionType,
	PlannerDecision,
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


class TestCircuitBreakerInit:
	def test_controller_has_circuit_breaker(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		assert hasattr(ctrl, "circuit_breaker")
		assert ctrl.circuit_breaker.state == SwarmCircuitBreakerState.CLOSED


class TestCircuitBreakerSpawnGate:
	async def test_spawn_blocked_when_breaker_open(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		# Trip the breaker by recording enough failures
		for _ in range(3):
			ctrl.circuit_breaker.record_failure()
		assert ctrl.circuit_breaker.is_tripped

		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			decision = PlannerDecision(
				type=DecisionType.SPAWN,
				payload={"name": "blocked-agent", "prompt": "do stuff"},
			)
			results = await ctrl.execute_decisions([decision])

		result = results[0]["result"]
		assert result.get("spawned") is False or "error" in result
		assert len(ctrl.agents) == 0

	async def test_spawn_allowed_when_breaker_closed(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		assert not ctrl.circuit_breaker.is_tripped

		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			decision = PlannerDecision(
				type=DecisionType.SPAWN,
				payload={"name": "good-agent", "prompt": "do stuff"},
			)
			results = await ctrl.execute_decisions([decision])

		assert results[0]["success"]
		assert len(ctrl.agents) == 1
		assert ctrl.agents[0].status == AgentStatus.WORKING


class TestCircuitBreakerOutcomeRecording:
	async def test_success_records_on_circuit_breaker(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())

		# Create task + spawn agent
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Fix bug"}),
		])
		task_id = ctrl.tasks[0].id

		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.stdout = AsyncMock()
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.SPAWN, payload={
					"name": "w1", "prompt": "fix", "task_id": task_id,
				}),
			])

		agent = ctrl.agents[0]

		# Record a failure first so we can verify success resets it
		ctrl.circuit_breaker.record_failure(task_id)
		assert ctrl.circuit_breaker.consecutive_failures == 1

		# Simulate agent completion
		ad_result = '{"status":"completed","commits":[],"summary":"done","files_changed":[]}'
		mock_proc.returncode = 0
		ctrl._agent_outputs[agent.id] = f"AD_RESULT:{ad_result}"

		with patch.object(ctrl, "_get_git_changed_files", new=AsyncMock(return_value=[])), \
			patch.object(ctrl, "_auto_commit_task", new=AsyncMock(return_value=None)), \
			patch.object(ctrl, "_write_checkpoint"):
			events = await ctrl.monitor_agents()

		assert len(events) == 1
		assert events[0]["status"] == "completed"
		assert ctrl.circuit_breaker.consecutive_failures == 0

	async def test_failure_records_on_circuit_breaker(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())

		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.stdout = AsyncMock()
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "w2", "prompt": "fail"}),
			])

		agent = ctrl.agents[0]

		# Simulate agent failure
		ad_result = '{"status":"failed","commits":[],"summary":"crashed","files_changed":[]}'
		mock_proc.returncode = 1
		ctrl._agent_outputs[agent.id] = f"AD_RESULT:{ad_result}"

		with patch.object(ctrl, "_get_git_changed_files", new=AsyncMock(return_value=[])), \
			patch.object(ctrl, "_write_checkpoint"):
			events = await ctrl.monitor_agents()

		assert len(events) == 1
		assert events[0]["status"] == "failed"
		assert ctrl.circuit_breaker.consecutive_failures == 1

	async def test_consecutive_failures_trip_breaker(self, tmp_path: Path) -> None:
		"""Three consecutive agent failures should trip the circuit breaker."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())

		for i in range(3):
			mock_proc = MagicMock()
			mock_proc.returncode = None
			mock_proc.stdout = AsyncMock()
			with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
				await ctrl.execute_decisions([
					PlannerDecision(type=DecisionType.SPAWN, payload={
						"name": f"fail-{i}", "prompt": "crash",
					}),
				])

			agent = ctrl.agents[-1]
			ad_result = '{"status":"failed","commits":[],"summary":"crash","files_changed":[]}'
			mock_proc.returncode = 1
			ctrl._agent_outputs[agent.id] = f"AD_RESULT:{ad_result}"

			with patch.object(ctrl, "_get_git_changed_files", new=AsyncMock(return_value=[])), \
				patch.object(ctrl, "_write_checkpoint"):
				await ctrl.monitor_agents()

		assert ctrl.circuit_breaker.is_tripped
		assert ctrl.circuit_breaker.state == SwarmCircuitBreakerState.OPEN
