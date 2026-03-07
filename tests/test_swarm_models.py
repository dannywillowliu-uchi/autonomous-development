"""Tests for swarm data models."""

from autodev.swarm.models import (
	AgentRole,
	AgentStatus,
	DecisionType,
	PlannerDecision,
	StagnationSignal,
	SwarmAgent,
	SwarmState,
	SwarmTask,
	TaskPriority,
	TaskStatus,
)


class TestSwarmAgent:
	def test_defaults(self) -> None:
		agent = SwarmAgent(name="test-agent")
		assert agent.name == "test-agent"
		assert agent.role == AgentRole.GENERAL
		assert agent.status == AgentStatus.SPAWNING
		assert agent.current_task_id is None
		assert agent.tasks_completed == 0
		assert agent.tasks_failed == 0

	def test_role_assignment(self) -> None:
		agent = SwarmAgent(name="researcher", role=AgentRole.RESEARCHER)
		assert agent.role == AgentRole.RESEARCHER


class TestSwarmTask:
	def test_defaults(self) -> None:
		task = SwarmTask(title="Fix parser bug")
		assert task.title == "Fix parser bug"
		assert task.status == TaskStatus.PENDING
		assert task.priority == TaskPriority.NORMAL
		assert task.claimed_by is None
		assert task.attempt_count == 0

	def test_dependencies(self) -> None:
		task = SwarmTask(title="Implement", depends_on=["task-1", "task-2"])
		assert len(task.depends_on) == 2

	def test_priority_ordering(self) -> None:
		assert TaskPriority.CRITICAL > TaskPriority.HIGH > TaskPriority.NORMAL > TaskPriority.LOW


class TestPlannerDecision:
	def test_spawn_decision(self) -> None:
		d = PlannerDecision(
			type=DecisionType.SPAWN,
			payload={"role": "researcher", "prompt": "Investigate parser bug"},
			reasoning="Need to understand root cause before fixing",
			priority=2,
		)
		assert d.type == DecisionType.SPAWN
		assert d.payload["role"] == "researcher"
		assert d.priority == 2

	def test_decision_types_cover_all(self) -> None:
		expected = {"spawn", "kill", "redirect", "create_task", "adjust", "wait", "escalate", "create_skill"}
		actual = {dt.value for dt in DecisionType}
		assert actual == expected


class TestSwarmState:
	def test_empty_state(self) -> None:
		state = SwarmState(mission_objective="Build compiler")
		assert state.mission_objective == "Build compiler"
		assert state.agents == []
		assert state.tasks == []
		assert state.cycle_number == 0
		assert state.stagnation_signals == []

	def test_state_with_agents_and_tasks(self) -> None:
		agent = SwarmAgent(name="worker-1", status=AgentStatus.WORKING)
		task = SwarmTask(title="Fix bug", status=TaskStatus.IN_PROGRESS, claimed_by=agent.id)
		state = SwarmState(
			mission_objective="Fix all bugs",
			agents=[agent],
			tasks=[task],
			cycle_number=5,
			total_cost_usd=3.50,
		)
		assert len(state.agents) == 1
		assert len(state.tasks) == 1
		assert state.cycle_number == 5


class TestStagnationSignal:
	def test_stagnation_signal(self) -> None:
		sig = StagnationSignal(
			metric="test_pass_count",
			value_history=[210.0, 210.0, 210.0],
			cycles_stagnant=3,
			suggested_pivot="Switch to research mode",
		)
		assert sig.cycles_stagnant == 3
		assert len(sig.value_history) == 3
