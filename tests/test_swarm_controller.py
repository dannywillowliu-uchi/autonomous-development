"""Tests for swarm controller."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from autodev.config import SwarmConfig
from autodev.swarm.controller import SwarmController
from autodev.swarm.models import (
	AgentRole,
	AgentStatus,
	DecisionType,
	PlannerDecision,
	TaskPriority,
	TaskStatus,
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


class TestSwarmControllerInit:
	async def test_initialize_creates_team_dir(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		with patch.object(Path, "home", return_value=tmp_path):
			await ctrl.initialize()
		team_dir = tmp_path / ".claude" / "teams" / "autodev-test-project"
		assert team_dir.exists()
		assert (team_dir / "inboxes").exists()
		assert (team_dir / "config.json").exists()

	async def test_team_name_derived_from_target(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		assert ctrl.team_name == "autodev-test-project"


class TestCreateTask:
	async def test_create_task_adds_to_pool(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		decision = PlannerDecision(
			type=DecisionType.CREATE_TASK,
			payload={"title": "Fix parser bug", "priority": 2, "files_hint": ["src/parser.c"]},
		)
		results = await ctrl.execute_decisions([decision])
		assert results[0]["success"]
		assert len(ctrl.tasks) == 1
		assert ctrl.tasks[0].title == "Fix parser bug"
		assert ctrl.tasks[0].priority == TaskPriority.HIGH

	async def test_create_task_with_dependencies(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		d1 = PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Research"})
		d2 = PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Implement", "depends_on": ["task-1"]})
		await ctrl.execute_decisions([d1, d2])
		assert len(ctrl.tasks) == 2


class TestSpawnAgent:
	async def test_spawn_creates_agent(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			decision = PlannerDecision(
				type=DecisionType.SPAWN,
				payload={"role": "researcher", "name": "res-1", "prompt": "Investigate bug"},
			)
			results = await ctrl.execute_decisions([decision])
		assert results[0]["success"]
		assert len(ctrl.agents) == 1
		assert ctrl.agents[0].name == "res-1"
		assert ctrl.agents[0].role == AgentRole.RESEARCHER
		assert ctrl.agents[0].status == AgentStatus.WORKING

	async def test_spawn_respects_max_agents(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(max_agents=1), _make_db())
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			d1 = PlannerDecision(type=DecisionType.SPAWN, payload={"name": "a1", "prompt": "task 1"})
			await ctrl.execute_decisions([d1])
			d2 = PlannerDecision(type=DecisionType.SPAWN, payload={"name": "a2", "prompt": "task 2"})
			results = await ctrl.execute_decisions([d2])
		assert results[0]["result"].get("spawned") is False or results[0]["result"].get("error")

	async def test_spawn_claims_task(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		# Create a task first
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Fix bug"}),
		])
		task_id = ctrl.tasks[0].id

		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			spawn_payload = {"name": "w1", "prompt": "fix", "task_id": task_id}
			d = PlannerDecision(type=DecisionType.SPAWN, payload=spawn_payload)
			await ctrl.execute_decisions([d])
		assert ctrl.tasks[0].status == TaskStatus.CLAIMED


class TestKillAgent:
	async def test_kill_marks_agent_dead(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		# Spawn an agent
		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.terminate = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch.object(ctrl, "_spawn_claude_session", new_callable=AsyncMock, return_value=mock_proc):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "victim", "prompt": "work"}),
			])

		agent_id = ctrl.agents[0].id
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.KILL, payload={"agent_id": agent_id, "reason": "test"}),
		])
		assert ctrl.agents[0].status == AgentStatus.DEAD

	async def test_kill_releases_task(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Fix bug"}),
		])
		task_id = ctrl.tasks[0].id

		mock_proc = MagicMock()
		mock_proc.returncode = None
		mock_proc.terminate = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch.object(ctrl, "_spawn_claude_session", new_callable=AsyncMock, return_value=mock_proc):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "w1", "prompt": "fix", "task_id": task_id}),
			])

		agent_id = ctrl.agents[0].id
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.KILL, payload={"agent_id": agent_id}),
		])
		assert ctrl.tasks[0].status == TaskStatus.PENDING
		assert ctrl.tasks[0].claimed_by is None


class TestAdjust:
	async def test_adjust_max_agents(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.ADJUST, payload={"max_agents": 10}),
		])
		assert ctrl.swarm_config.max_agents == 10


class TestCreateSkill:
	async def test_create_skill_writes_files(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl.execute_decisions([
			PlannerDecision(
				type=DecisionType.CREATE_SKILL,
				payload={
					"name": "test-tool",
					"description": "A test tool",
					"content": "Do the thing.\n\n1. Step one\n2. Step two",
					"supporting_files": {"helper.py": "print('hello')"},
				},
			),
		])
		skill_dir = tmp_path / ".claude" / "skills" / "test-tool"
		assert skill_dir.exists()
		skill_md = (skill_dir / "SKILL.md").read_text()
		assert "name: test-tool" in skill_md
		assert "description: A test tool" in skill_md
		assert "Step one" in skill_md
		assert (skill_dir / "helper.py").read_text() == "print('hello')"


class TestBuildState:
	async def test_build_state_returns_snapshot(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		state = ctrl.build_state(core_test_results={"pass": 200, "fail": 10})
		assert state.mission_objective == "Build a compiler"
		assert state.core_test_results["pass"] == 200


class TestParseAdResult:
	def test_parse_valid_result(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		ad_json = (
			'{"status":"completed","commits":["abc"],'
			'"summary":"Fixed it","files_changed":["a.py"],"discoveries":[]}'
		)
		output = f"Some output\nAD_RESULT:{ad_json}\n"
		result = ctrl._parse_ad_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["commits"] == ["abc"]

	def test_parse_missing_result(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		result = ctrl._parse_ad_result("no result here")
		assert result is None

	def test_parse_result_with_trailing_text(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		output = 'AD_RESULT:{"status":"failed","commits":[],"summary":"error","files_changed":[]}\nsome trailing text'
		result = ctrl._parse_ad_result(output)
		assert result is not None
		assert result["status"] == "failed"


class TestDecisionPriority:
	async def test_higher_priority_executes_first(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		execution_order: list[str] = []
		original_execute = ctrl._execute_one

		async def tracking_execute(decision: PlannerDecision) -> dict:
			execution_order.append(decision.payload.get("title", ""))
			return await original_execute(decision)

		ctrl._execute_one = tracking_execute

		decisions = [
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "low"}, priority=0),
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "high"}, priority=10),
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "mid"}, priority=5),
		]
		await ctrl.execute_decisions(decisions)
		assert execution_order == ["high", "mid", "low"]
