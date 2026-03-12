"""Tests for swarm controller."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autodev.config import SwarmConfig
from autodev.swarm.controller import SwarmController
from autodev.swarm.models import (
	AgentRole,
	AgentStatus,
	DecisionType,
	PlannerDecision,
	SwarmAgent,
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
			PlannerDecision(type=DecisionType.KILL, payload={"agent_id": agent_id, "reason": "test", "force": True}),
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
			PlannerDecision(type=DecisionType.KILL, payload={"agent_id": agent_id, "force": True}),
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
	def test_parse_valid_result_all_fields(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		ad_json = (
			'{"status":"completed","commits":["abc"],'
			'"summary":"Fixed it","files_changed":["a.py"],"discoveries":["found bug"],"concerns":[]}'
		)
		output = f"Some output\nAD_RESULT:{ad_json}\n"
		result = ctrl._parse_ad_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["commits"] == ["abc"]
		assert result["summary"] == "Fixed it"
		assert result["discoveries"] == ["found bug"]

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

	def test_missing_status_returns_none(self, tmp_path: Path) -> None:
		"""'status' is required -- result without it must be rejected."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		output = 'AD_RESULT:{"commits":[],"summary":"oops"}\n'
		result = ctrl._parse_ad_result(output)
		assert result is None

	def test_missing_summary_defaults_to_empty(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		output = 'AD_RESULT:{"status":"completed","commits":[]}\n'
		result = ctrl._parse_ad_result(output)
		assert result is not None
		assert result["summary"] == ""

	def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		output = 'AD_RESULT:{"status": "completed", BROKEN}\n'
		result = ctrl._parse_ad_result(output)
		assert result is None

	def test_no_opening_brace_returns_none(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		output = "AD_RESULT: not json at all\n"
		result = ctrl._parse_ad_result(output)
		assert result is None

	def test_multiple_markers_uses_last(self, tmp_path: Path) -> None:
		"""Earlier AD_RESULT markers (from retries) should be ignored."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		output = (
			'AD_RESULT:{"status":"failed","summary":"first attempt"}\n'
			"some retry output\n"
			'AD_RESULT:{"status":"completed","summary":"second attempt"}\n'
		)
		result = ctrl._parse_ad_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["summary"] == "second attempt"

	def test_large_output_truncated_keeps_tail(self, tmp_path: Path) -> None:
		"""Output exceeding max_output_size is truncated from the start."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		filler = "x" * 5000
		ad_line = 'AD_RESULT:{"status":"completed","summary":"ok"}\n'
		output = filler + "\n" + ad_line
		# Set a small limit; the tail (with AD_RESULT) should survive
		result = ctrl._parse_ad_result(output, max_output_size=200)
		assert result is not None
		assert result["status"] == "completed"

	def test_large_output_truncation_loses_marker(self, tmp_path: Path) -> None:
		"""If AD_RESULT is at the start and gets truncated, return None."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		ad_line = 'AD_RESULT:{"status":"completed","summary":"ok"}\n'
		filler = "x" * 5000
		output = ad_line + filler
		result = ctrl._parse_ad_result(output, max_output_size=200)
		assert result is None

	def test_nested_braces_in_string_values(self, tmp_path: Path) -> None:
		"""Braces inside JSON string values must not break brace counting."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		output = r'AD_RESULT:{"status":"completed","summary":"fixed {nested} braces","data":"{}"}'
		result = ctrl._parse_ad_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert "{nested}" in result["summary"]

	def test_escaped_quotes_in_strings(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		output = r'AD_RESULT:{"status":"completed","summary":"said \"hello\""}'
		result = ctrl._parse_ad_result(output)
		assert result is not None
		assert result["status"] == "completed"

	def test_non_dict_json_returns_none(self, tmp_path: Path) -> None:
		"""A JSON array is not a valid AD_RESULT."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		# This won't parse as object due to brace counting starting at [
		output = 'AD_RESULT:["not","an","object"]\n'
		result = ctrl._parse_ad_result(output)
		assert result is None


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


class TestDeadAgentCleanup:
	def test_cleanup_removes_old_dead_agents(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		agent = SwarmAgent(name="old-dead", status=AgentStatus.DEAD)
		agent.death_time = time.monotonic() - 600  # 10 min ago
		ctrl._agents[agent.id] = agent

		ctrl._cleanup_dead_agents()

		assert agent.id not in ctrl._agents
		assert len(ctrl._dead_agent_history) == 1
		assert ctrl._dead_agent_history[0].name == "old-dead"

	def test_cleanup_keeps_recently_dead_agents(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		agent = SwarmAgent(name="fresh-dead", status=AgentStatus.DEAD)
		agent.death_time = time.monotonic() - 60  # 1 min ago
		ctrl._agents[agent.id] = agent

		ctrl._cleanup_dead_agents()

		assert agent.id in ctrl._agents
		assert len(ctrl._dead_agent_history) == 0

	def test_cleanup_ignores_living_agents(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		agent = SwarmAgent(name="alive", status=AgentStatus.WORKING)
		ctrl._agents[agent.id] = agent

		ctrl._cleanup_dead_agents()

		assert agent.id in ctrl._agents
		assert len(ctrl._dead_agent_history) == 0

	def test_cleanup_removes_stale_process_entries(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		agent = SwarmAgent(name="dead-proc", status=AgentStatus.DEAD)
		agent.death_time = time.monotonic() - 600
		ctrl._agents[agent.id] = agent
		ctrl._processes[agent.id] = MagicMock()

		ctrl._cleanup_dead_agents()

		assert agent.id not in ctrl._processes

	def test_cleanup_bounds_history_size(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		# Pre-fill history to max
		for i in range(50):
			ctrl._dead_agent_history.append(SwarmAgent(name=f"old-{i}"))

		# Add one more dead agent to trigger cleanup
		agent = SwarmAgent(name="overflow", status=AgentStatus.DEAD)
		agent.death_time = time.monotonic() - 600
		ctrl._agents[agent.id] = agent

		ctrl._cleanup_dead_agents()

		assert len(ctrl._dead_agent_history) == 50
		assert ctrl._dead_agent_history[-1].name == "overflow"

	def test_cleanup_sets_death_time_on_legacy_agents(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		agent = SwarmAgent(name="legacy", status=AgentStatus.DEAD)
		# death_time is None (legacy agent)
		ctrl._agents[agent.id] = agent

		ctrl._cleanup_dead_agents()

		# Should set death_time but not remove yet
		assert agent.id in ctrl._agents
		assert agent.death_time is not None

	async def test_monitor_agents_calls_cleanup(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		# Add a dead agent that's old enough to clean
		agent = SwarmAgent(name="monitored-dead", status=AgentStatus.DEAD)
		agent.death_time = time.monotonic() - 600
		ctrl._agents[agent.id] = agent

		await ctrl.monitor_agents()

		assert agent.id not in ctrl._agents
		assert len(ctrl._dead_agent_history) == 1

	def test_kill_sets_death_time(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		agent = SwarmAgent(name="to-kill", status=AgentStatus.DEAD)
		agent.death_time = time.monotonic()
		ctrl._agents[agent.id] = agent

		assert agent.death_time is not None

	def test_build_state_includes_dead_history(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		dead = SwarmAgent(name="hist-agent", status=AgentStatus.DEAD, tasks_completed=3)
		ctrl._dead_agent_history.append(dead)

		state = ctrl.build_state()
		assert len(state.dead_agent_history) == 1
		assert state.dead_agent_history[0].name == "hist-agent"


class TestProcessCrashWithoutAdResult:
	"""Tests for the bug where crashed processes leave tasks stuck in CLAIMED."""

	async def _spawn_agent_with_task(self, ctrl: SwarmController, task_id: str) -> MagicMock:
		mock_proc = MagicMock(returncode=None)
		mock_proc.stdout = None
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "w1", "prompt": "fix", "task_id": task_id}),
			])
		return mock_proc

	async def test_crash_marks_task_failed_and_clears_claimed_by(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Bug fix"}),
		])
		task_id = ctrl.tasks[0].id
		mock_proc = await self._spawn_agent_with_task(ctrl, task_id)
		assert ctrl.tasks[0].status == TaskStatus.CLAIMED

		# Simulate crash: exit code 1, no AD_RESULT
		mock_proc.returncode = 1
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(return_value=b"Segfault\nTraceback...")

		events = await ctrl.monitor_agents()

		assert len(events) >= 1
		assert events[0]["status"] == "failed"
		assert ctrl.tasks[0].status == TaskStatus.FAILED
		assert ctrl.tasks[0].claimed_by is None
		assert "without emitting AD_RESULT" in ctrl.tasks[0].result_summary

	async def test_crash_increments_attempt_count(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Flaky"}),
		])
		task_id = ctrl.tasks[0].id
		mock_proc = await self._spawn_agent_with_task(ctrl, task_id)

		mock_proc.returncode = 137
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(return_value=b"killed")

		await ctrl.monitor_agents()

		assert ctrl.tasks[0].attempt_count == 1

	async def test_crashed_task_can_be_requeued(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Retriable", "max_attempts": 3}),
		])
		task_id = ctrl.tasks[0].id
		mock_proc = await self._spawn_agent_with_task(ctrl, task_id)

		mock_proc.returncode = 1
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(return_value=b"crash")

		await ctrl.monitor_agents()

		requeued = ctrl.requeue_failed_tasks()
		assert task_id in requeued
		assert ctrl.tasks[0].status == TaskStatus.PENDING
		assert ctrl.tasks[0].claimed_by is None

	async def test_stdout_read_exception_still_marks_task_failed(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "IO error"}),
		])
		task_id = ctrl.tasks[0].id
		mock_proc = await self._spawn_agent_with_task(ctrl, task_id)

		mock_proc.returncode = 1
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(side_effect=OSError("pipe broken"))

		await ctrl.monitor_agents()

		assert ctrl.tasks[0].status == TaskStatus.FAILED
		assert ctrl.tasks[0].claimed_by is None


class TestClaimTimeout:
	"""Tests for the configurable task claim timeout."""

	async def test_claimed_task_times_out(self, tmp_path: Path) -> None:
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			task_claim_timeout=60.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Slow task"}),
		])
		task_id = ctrl.tasks[0].id

		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "w1", "prompt": "work", "task_id": task_id}),
			])

		assert ctrl.tasks[0].status == TaskStatus.CLAIMED

		# Backdate claimed_at to exceed timeout
		from datetime import datetime, timedelta, timezone
		old_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
		ctrl.tasks[0].claimed_at = old_time

		events = await ctrl.monitor_agents()

		timeout_events = [e for e in events if e["type"] == "claim_timeout"]
		assert len(timeout_events) == 1
		assert ctrl.tasks[0].status == TaskStatus.FAILED
		assert ctrl.tasks[0].claimed_by is None
		assert "timeout" in ctrl.tasks[0].result_summary.lower()

	async def test_within_timeout_not_affected(self, tmp_path: Path) -> None:
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			task_claim_timeout=1800.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Normal task"}),
		])
		task_id = ctrl.tasks[0].id

		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "w1", "prompt": "work", "task_id": task_id}),
			])

		events = await ctrl.monitor_agents()

		assert ctrl.tasks[0].status == TaskStatus.CLAIMED
		timeout_events = [e for e in events if e.get("type") == "claim_timeout"]
		assert len(timeout_events) == 0

	async def test_timeout_marks_agent_dead(self, tmp_path: Path) -> None:
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			task_claim_timeout=60.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Stuck task"}),
		])
		task_id = ctrl.tasks[0].id

		mock_proc = MagicMock(returncode=None, kill=MagicMock())
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "w1", "prompt": "work", "task_id": task_id}),
			])

		agent = ctrl.agents[0]
		from datetime import datetime, timedelta, timezone
		ctrl.tasks[0].claimed_at = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()

		await ctrl.monitor_agents()

		assert agent.status == AgentStatus.DEAD
		assert agent.tasks_failed == 1
		mock_proc.kill.assert_called_once()

	async def test_timeout_increments_attempt_count(self, tmp_path: Path) -> None:
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			task_claim_timeout=60.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Timeout task"}),
		])
		task_id = ctrl.tasks[0].id

		mock_proc = MagicMock(returncode=None, kill=MagicMock())
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "w1", "prompt": "work", "task_id": task_id}),
			])

		from datetime import datetime, timedelta, timezone
		ctrl.tasks[0].claimed_at = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()

		await ctrl.monitor_agents()

		assert ctrl.tasks[0].attempt_count == 1


class TestOrphanedTaskRecovery:
	"""Tests for stalled/orphaned task detection and recovery."""

	async def test_orphaned_task_reset_to_pending_when_agent_dead(self, tmp_path: Path) -> None:
		"""Task claimed by a dead agent should be reset to PENDING after timeout."""
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			stalled_task_timeout=600.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Orphaned task"}),
		])
		task = ctrl.tasks[0]

		# Create a dead agent that claimed this task
		agent = SwarmAgent(name="dead-worker", status=AgentStatus.DEAD)
		agent.death_time = time.monotonic() - 60
		ctrl._agents[agent.id] = agent

		task.status = TaskStatus.CLAIMED
		task.claimed_by = agent.id
		from datetime import datetime, timedelta, timezone
		task.claimed_at = (datetime.now(timezone.utc) - timedelta(seconds=700)).isoformat()

		events = await ctrl.monitor_agents()

		assert task.status == TaskStatus.PENDING
		assert task.claimed_by is None
		assert task.claimed_at is None
		recovered = [e for e in events if e["type"] == "orphaned_task_recovered"]
		assert len(recovered) == 1
		assert recovered[0]["task_id"] == task.id

	async def test_orphaned_task_not_recovered_if_agent_alive(self, tmp_path: Path) -> None:
		"""Task claimed by a living agent should NOT be reset."""
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			stalled_task_timeout=600.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Active task"}),
		])
		task = ctrl.tasks[0]

		agent = SwarmAgent(name="alive-worker", status=AgentStatus.WORKING)
		ctrl._agents[agent.id] = agent

		task.status = TaskStatus.CLAIMED
		task.claimed_by = agent.id
		from datetime import datetime, timedelta, timezone
		task.claimed_at = (datetime.now(timezone.utc) - timedelta(seconds=700)).isoformat()

		events = await ctrl.monitor_agents()

		assert task.status == TaskStatus.CLAIMED
		recovered = [e for e in events if e.get("type") == "orphaned_task_recovered"]
		assert len(recovered) == 0

	async def test_orphaned_task_not_recovered_within_timeout(self, tmp_path: Path) -> None:
		"""Task claimed by a dead agent should NOT be reset if within timeout."""
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			stalled_task_timeout=600.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Recent task"}),
		])
		task = ctrl.tasks[0]

		agent = SwarmAgent(name="dead-recent", status=AgentStatus.DEAD)
		agent.death_time = time.monotonic()
		ctrl._agents[agent.id] = agent

		task.status = TaskStatus.CLAIMED
		task.claimed_by = agent.id
		from datetime import datetime, timedelta, timezone
		task.claimed_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()

		events = await ctrl.monitor_agents()

		assert task.status == TaskStatus.CLAIMED
		recovered = [e for e in events if e.get("type") == "orphaned_task_recovered"]
		assert len(recovered) == 0

	async def test_orphaned_task_recovered_when_agent_missing(self, tmp_path: Path) -> None:
		"""Task claimed by an agent no longer in the pool should be recovered."""
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			stalled_task_timeout=600.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Abandoned task"}),
		])
		task = ctrl.tasks[0]

		# Agent is not in _agents at all (already cleaned up)
		task.status = TaskStatus.CLAIMED
		task.claimed_by = "nonexistent-agent-id"
		from datetime import datetime, timedelta, timezone
		task.claimed_at = (datetime.now(timezone.utc) - timedelta(seconds=700)).isoformat()

		events = await ctrl.monitor_agents()

		assert task.status == TaskStatus.PENDING
		assert task.claimed_by is None
		recovered = [e for e in events if e["type"] == "orphaned_task_recovered"]
		assert len(recovered) == 1

	async def test_in_progress_task_recovered_when_agent_dead(self, tmp_path: Path) -> None:
		"""IN_PROGRESS tasks should also be recovered when agent dies."""
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			stalled_task_timeout=600.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "In progress task"}),
		])
		task = ctrl.tasks[0]

		agent = SwarmAgent(name="dead-ip", status=AgentStatus.DEAD)
		ctrl._agents[agent.id] = agent

		task.status = TaskStatus.IN_PROGRESS
		task.claimed_by = agent.id
		from datetime import datetime, timedelta, timezone
		task.claimed_at = (datetime.now(timezone.utc) - timedelta(seconds=700)).isoformat()

		events = await ctrl.monitor_agents()

		assert task.status == TaskStatus.PENDING
		recovered = [e for e in events if e["type"] == "orphaned_task_recovered"]
		assert len(recovered) == 1
		assert recovered[0]["previous_status"] == "in_progress"

	async def test_orphaned_task_no_claimed_by_reset_immediately(self, tmp_path: Path) -> None:
		"""CLAIMED task with no claimed_by should be reset to PENDING immediately."""
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			stalled_task_timeout=600.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "No owner"}),
		])
		task = ctrl.tasks[0]
		task.status = TaskStatus.CLAIMED
		task.claimed_by = None

		await ctrl.monitor_agents()

		assert task.status == TaskStatus.PENDING

	async def test_custom_stalled_timeout(self, tmp_path: Path) -> None:
		"""Custom stalled_task_timeout should be respected."""
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(), _make_db(),
			stalled_task_timeout=30.0,
		)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Quick timeout"}),
		])
		task = ctrl.tasks[0]

		agent = SwarmAgent(name="dead-fast", status=AgentStatus.DEAD)
		ctrl._agents[agent.id] = agent

		task.status = TaskStatus.CLAIMED
		task.claimed_by = agent.id
		from datetime import datetime, timedelta, timezone
		task.claimed_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()

		events = await ctrl.monitor_agents()

		assert task.status == TaskStatus.PENDING
		recovered = [e for e in events if e["type"] == "orphaned_task_recovered"]
		assert len(recovered) == 1


class TestTaskStateSync:
	"""Tests for the bug where completed agents' tasks stay PENDING.

	Root cause: when create_task and spawn are in the same batch of decisions,
	the planner can't reference auto-generated task IDs. The spawn's task_id
	is either None or a placeholder, so agent.current_task_id doesn't match
	any task. When the agent completes, monitor_agents skips the task update.
	"""

	async def test_spawn_auto_links_to_batch_created_task(self, tmp_path: Path) -> None:
		"""When create_task and spawn are in the same batch, task_id is auto-resolved."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			decisions = [
				PlannerDecision(
					type=DecisionType.CREATE_TASK,
					payload={"title": "Fix parser bug", "priority": 2},
					priority=10,
				),
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={"role": "implementer", "name": "parser-fixer", "prompt": "Fix it"},
					priority=9,
				),
			]
			await ctrl.execute_decisions(decisions)

		assert len(ctrl.tasks) == 1
		assert len(ctrl.agents) == 1
		# Task should be CLAIMED by the spawned agent
		assert ctrl.tasks[0].status == TaskStatus.CLAIMED
		assert ctrl.tasks[0].claimed_by == ctrl.agents[0].id
		# Agent should have the correct task_id
		assert ctrl.agents[0].current_task_id == ctrl.tasks[0].id

	async def test_spawn_resolves_placeholder_task_id(self, tmp_path: Path) -> None:
		"""A spawn with a non-existent task_id gets resolved to a batch-created task."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			decisions = [
				PlannerDecision(
					type=DecisionType.CREATE_TASK,
					payload={"title": "Research float80"},
					priority=10,
				),
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={
						"role": "researcher",
						"name": "float80-researcher",
						"prompt": "Research float80",
						"task_id": "<task_id from above>",
					},
					priority=9,
				),
			]
			await ctrl.execute_decisions(decisions)

		assert ctrl.tasks[0].status == TaskStatus.CLAIMED
		assert ctrl.agents[0].current_task_id == ctrl.tasks[0].id

	async def test_multiple_batch_tasks_linked_in_order(self, tmp_path: Path) -> None:
		"""Multiple create_task + spawn pairs auto-link in execution order."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			decisions = [
				PlannerDecision(
					type=DecisionType.CREATE_TASK, payload={"title": "Task A"}, priority=10,
				),
				PlannerDecision(
					type=DecisionType.CREATE_TASK, payload={"title": "Task B"}, priority=9,
				),
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={"name": "worker-a", "prompt": "Do A"},
					priority=8,
				),
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={"name": "worker-b", "prompt": "Do B"},
					priority=7,
				),
			]
			await ctrl.execute_decisions(decisions)

		assert len(ctrl.tasks) == 2
		assert len(ctrl.agents) == 2
		# Both tasks should be CLAIMED
		assert ctrl.tasks[0].status == TaskStatus.CLAIMED
		assert ctrl.tasks[1].status == TaskStatus.CLAIMED
		# Each agent should be linked to a different task
		agent_task_ids = {a.current_task_id for a in ctrl.agents}
		task_ids = {t.id for t in ctrl.tasks}
		assert agent_task_ids == task_ids

	async def test_spawn_with_valid_task_id_not_overridden(self, tmp_path: Path) -> None:
		"""A spawn with a valid task_id should NOT be auto-resolved."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		# Create task first (separate batch)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Existing task"}),
		])
		task_id = ctrl.tasks[0].id

		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={"name": "w1", "prompt": "work", "task_id": task_id},
				),
			])

		assert ctrl.agents[0].current_task_id == task_id

	async def test_completed_agent_marks_task_completed(self, tmp_path: Path) -> None:
		"""End-to-end: create task + spawn in same batch, agent completes -> task COMPLETED."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			decisions = [
				PlannerDecision(
					type=DecisionType.CREATE_TASK,
					payload={"title": "Fix the bug"},
					priority=10,
				),
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={"name": "bugfixer", "prompt": "Fix it"},
					priority=9,
				),
			]
			await ctrl.execute_decisions(decisions)

		assert ctrl.tasks[0].status == TaskStatus.CLAIMED

		# Simulate agent completing successfully
		mock_proc.returncode = 0
		ad_result = (
			'{"status":"completed","summary":"Fixed the bug",'
			'"commits":["abc"],"files_changed":["a.py"]}'
		)
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(
			return_value=f"output\nAD_RESULT:{ad_result}\n".encode()
		)

		events = await ctrl.monitor_agents()

		assert len(events) == 1
		assert events[0]["status"] == "completed"
		assert ctrl.tasks[0].status == TaskStatus.COMPLETED
		assert ctrl.tasks[0].result_summary == "Fixed the bug"
		assert ctrl.agents[0].tasks_completed == 1

	async def test_monitor_resolves_task_via_claimed_by_fallback(self, tmp_path: Path) -> None:
		"""If current_task_id doesn't match, task is found via claimed_by."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())

		# Create a task
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Bug fix"}),
		])
		task = ctrl.tasks[0]

		# Spawn agent with correct task_id
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={"name": "w1", "prompt": "fix", "task_id": task.id},
				),
			])

		agent = ctrl.agents[0]
		assert task.claimed_by == agent.id

		# Corrupt the agent's current_task_id to simulate the bug
		agent.current_task_id = "nonexistent-id"

		# Simulate agent completing
		mock_proc.returncode = 0
		ad_result = '{"status":"completed","summary":"Done"}'
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(return_value=f"AD_RESULT:{ad_result}".encode())

		await ctrl.monitor_agents()

		# Task should still be marked COMPLETED via claimed_by fallback
		assert task.status == TaskStatus.COMPLETED
		assert agent.tasks_completed == 1

	async def test_scaling_signal_correct_after_completion(self, tmp_path: Path) -> None:
		"""After agents complete, scaling signal should NOT say 'scale UP'."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			decisions = [
				PlannerDecision(
					type=DecisionType.CREATE_TASK, payload={"title": "Task 1"}, priority=10,
				),
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={"name": "w1", "prompt": "do task 1"},
					priority=9,
				),
			]
			await ctrl.execute_decisions(decisions)

		# Simulate completion
		mock_proc.returncode = 0
		ad_result = '{"status":"completed","summary":"Done"}'
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(return_value=f"AD_RESULT:{ad_result}".encode())
		await ctrl.monitor_agents()

		# Task should be COMPLETED, not PENDING
		assert ctrl.tasks[0].status == TaskStatus.COMPLETED
		# Scaling should NOT recommend scale up (no pending tasks)
		rec = ctrl.get_scaling_recommendation()
		assert rec["scale_up"] == 0

	async def test_invalid_task_id_resolved_to_pending_task(self, tmp_path: Path) -> None:
		"""Spawn with invalid task_id resolves to an unclaimed PENDING task from a prior batch."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())

		# Create task in a SEPARATE batch (simulating prior cycle)
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Parser fix"}),
		])
		real_task_id = ctrl.tasks[0].id

		# Spawn with an invalid task_id (planner made up an ID)
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={
						"name": "parser-fixer",
						"prompt": "Fix the parser",
						"task_id": "fake-task-id-from-planner",
					},
				),
			])

		# Agent should be linked to the real task, not the fake ID
		assert ctrl.agents[0].current_task_id == real_task_id
		assert ctrl.tasks[0].status == TaskStatus.CLAIMED
		assert ctrl.tasks[0].claimed_by == ctrl.agents[0].id

	async def test_invalid_task_id_agent_completion_updates_task(self, tmp_path: Path) -> None:
		"""End-to-end: invalid task_id resolved -> agent completes -> task COMPLETED."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())

		# Create task in prior batch
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Fix bug"}),
		])

		# Spawn with invalid task_id
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={
						"name": "bugfixer",
						"prompt": "Fix it",
						"task_id": "nonexistent-id",
					},
				),
			])

		# Simulate successful completion
		mock_proc.returncode = 0
		ad_result = '{"status":"completed","summary":"Fixed the bug"}'
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(return_value=f"AD_RESULT:{ad_result}".encode())
		await ctrl.monitor_agents()

		assert ctrl.tasks[0].status == TaskStatus.COMPLETED
		assert ctrl.agents[0].tasks_completed == 1

	async def test_no_pending_tasks_clears_invalid_task_id(self, tmp_path: Path) -> None:
		"""When no unclaimed tasks exist, invalid task_id is cleared (not left dangling)."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())

		# No tasks in pool
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={
						"name": "orphan",
						"prompt": "work",
						"task_id": "bogus-id",
					},
				),
			])

		# Agent should have current_task_id cleared (not pointing to bogus ID)
		assert ctrl.agents[0].current_task_id is None


class TestHandleCreateHook:
	async def test_creates_hook_in_settings(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		result = await ctrl._handle_create_hook({
			"event": "PreToolUse",
			"matcher": "Bash",
			"type": "command",
			"command": "echo safety check",
		})
		assert result["hook_added"] is True
		assert result["event"] == "PreToolUse"
		settings_path = Path(tmp_path) / ".claude" / "settings.json"
		import json
		data = json.loads(settings_path.read_text())
		assert len(data["hooks"]["PreToolUse"]) == 1
		hook = data["hooks"]["PreToolUse"][0]
		assert hook["matcher"] == "Bash"
		assert hook["command"] == "echo safety check"
		assert hook["type"] == "command"

	async def test_creates_prompt_hook(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl._handle_create_hook({
			"event": "PostToolUse",
			"matcher": "Write",
			"type": "prompt",
			"prompt": "Check output format",
		})
		settings_path = Path(tmp_path) / ".claude" / "settings.json"
		import json
		data = json.loads(settings_path.read_text())
		hook = data["hooks"]["PostToolUse"][0]
		assert hook["type"] == "prompt"
		assert hook["prompt"] == "Check output format"

	async def test_appends_to_existing_hooks(self, tmp_path: Path) -> None:
		import json
		settings_dir = Path(tmp_path) / ".claude"
		settings_dir.mkdir(parents=True)
		settings_path = settings_dir / "settings.json"
		settings_path.write_text(json.dumps({
			"hooks": {"PreToolUse": [{"matcher": "Edit", "type": "command", "command": "lint"}]}
		}))
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl._handle_create_hook({
			"event": "PreToolUse",
			"matcher": "Bash",
			"type": "command",
			"command": "check",
		})
		data = json.loads(settings_path.read_text())
		assert len(data["hooks"]["PreToolUse"]) == 2

	async def test_background_hook(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl._handle_create_hook({
			"event": "PreToolUse",
			"matcher": "Bash",
			"type": "command",
			"command": "log",
			"background": True,
		})
		import json
		settings_path = Path(tmp_path) / ".claude" / "settings.json"
		data = json.loads(settings_path.read_text())
		assert data["hooks"]["PreToolUse"][0]["background"] is True


class TestHandleRegisterMcp:
	async def test_registers_stdio_server(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		result = await ctrl._handle_register_mcp({
			"name": "my-mcp",
			"type": "stdio",
			"command": "npx",
			"args": ["-y", "my-mcp-server"],
		})
		assert result["registered"] is True
		assert result["name"] == "my-mcp"
		import json
		mcp_path = Path(tmp_path) / ".mcp.json"
		data = json.loads(mcp_path.read_text())
		assert data["mcpServers"]["my-mcp"]["command"] == "npx"
		assert data["mcpServers"]["my-mcp"]["args"] == ["-y", "my-mcp-server"]

	async def test_registers_sse_server(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl._handle_register_mcp({
			"name": "remote-mcp",
			"type": "sse",
			"url": "https://example.com/mcp",
		})
		import json
		mcp_path = Path(tmp_path) / ".mcp.json"
		data = json.loads(mcp_path.read_text())
		assert data["mcpServers"]["remote-mcp"]["url"] == "https://example.com/mcp"

	async def test_registers_with_env(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl._handle_register_mcp({
			"name": "with-env",
			"type": "stdio",
			"command": "cmd",
			"env": {"API_KEY": "secret"},
		})
		import json
		mcp_path = Path(tmp_path) / ".mcp.json"
		data = json.loads(mcp_path.read_text())
		assert data["mcpServers"]["with-env"]["env"] == {"API_KEY": "secret"}

	async def test_appends_to_existing_config(self, tmp_path: Path) -> None:
		import json
		mcp_path = Path(tmp_path) / ".mcp.json"
		mcp_path.write_text(json.dumps({
			"mcpServers": {"existing": {"command": "old"}}
		}))
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl._handle_register_mcp({
			"name": "new-server",
			"type": "stdio",
			"command": "new-cmd",
		})
		data = json.loads(mcp_path.read_text())
		assert "existing" in data["mcpServers"]
		assert "new-server" in data["mcpServers"]

	async def test_global_scope_writes_to_home(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		with patch.object(Path, "home", return_value=tmp_path):
			await ctrl._handle_register_mcp({
				"name": "global-mcp",
				"scope": "global",
				"type": "stdio",
				"command": "global-cmd",
			})
		import json
		mcp_path = tmp_path / ".claude.json"
		data = json.loads(mcp_path.read_text())
		assert "global-mcp" in data["mcpServers"]


class TestHandleCreateAgentDef:
	async def test_creates_agent_definition(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		result = await ctrl._handle_create_agent_def({
			"name": "code-reviewer",
			"description": "Reviews code for bugs",
			"tools": ["Read", "Grep", "Glob"],
			"model": "sonnet",
			"system_prompt": "You are a code reviewer.",
		})
		assert result["created"] is True
		assert result["name"] == "code-reviewer"
		agent_path = Path(tmp_path) / ".claude" / "agents" / "code-reviewer.md"
		assert agent_path.exists()
		content = agent_path.read_text()
		assert "name: code-reviewer" in content
		assert "description: Reviews code for bugs" in content
		assert "allowed-tools: Read, Grep, Glob" in content
		assert "model: sonnet" in content
		assert "You are a code reviewer." in content

	async def test_creates_minimal_agent(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl._handle_create_agent_def({
			"name": "simple-agent",
		})
		agent_path = Path(tmp_path) / ".claude" / "agents" / "simple-agent.md"
		content = agent_path.read_text()
		assert "name: simple-agent" in content
		assert "description" not in content
		assert "allowed-tools" not in content

	async def test_creates_agent_with_disallowed_tools(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		await ctrl._handle_create_agent_def({
			"name": "safe-agent",
			"disallowed_tools": ["Bash", "Write"],
		})
		agent_path = Path(tmp_path) / ".claude" / "agents" / "safe-agent.md"
		content = agent_path.read_text()
		assert "disallowed-tools: Bash, Write" in content


class TestHandleUseSkill:
	async def test_sends_directive_to_agent_inbox(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		with patch.object(ctrl, "_write_to_inbox") as mock_write:
			result = await ctrl._handle_use_skill({
				"agent_name": "worker-1",
				"skill_name": "commit",
				"args": "-m 'fix bug'",
			})
		assert result["directed"] is True
		assert result["skill"] == "commit"
		assert result["agent_name"] == "worker-1"
		mock_write.assert_called_once()
		call_args = mock_write.call_args
		assert call_args[0][0] == "worker-1"
		msg = call_args[0][1]
		assert msg["type"] == "directive"
		assert "/commit -m 'fix bug'" in msg["text"]

	async def test_sends_skill_without_args(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		with patch.object(ctrl, "_write_to_inbox") as mock_write:
			await ctrl._handle_use_skill({
				"agent_name": "worker-2",
				"skill_name": "verify-all",
			})
		msg = mock_write.call_args[0][1]
		assert msg["text"] == "Invoke /verify-all"


class TestHandleEscalate:
	async def test_escalate_returns_reason(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		result = await ctrl._handle_escalate({
			"reason": "Tests stuck in infinite loop",
		})
		assert result["escalated"] is True
		assert result["reason"] == "Tests stuck in infinite loop"

	async def test_escalate_default_reason(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		result = await ctrl._handle_escalate({})
		assert result["escalated"] is True
		assert result["reason"] == "Planner needs human input"


class TestHandleWait:
	async def test_wait_returns_duration(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		with patch("autodev.swarm.controller.asyncio.sleep", new=AsyncMock()):
			result = await ctrl._handle_wait({"duration": 5, "reason": "agents busy"})
		assert result["waited"] == 5
		assert result["reason"] == "agents busy"

	async def test_wait_caps_at_120(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		with patch("autodev.swarm.controller.asyncio.sleep", new=AsyncMock()):
			result = await ctrl._handle_wait({"duration": 999})
		assert result["waited"] == 120

	async def test_wait_default_duration(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		with patch("autodev.swarm.controller.asyncio.sleep", new=AsyncMock()):
			result = await ctrl._handle_wait({})
		assert result["waited"] == 10


class TestParseAgentCost:
	def test_standard_format(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		assert ctrl._parse_agent_cost("Some output\nTotal cost: $1.23\nDone") == 1.23

	def test_with_tab(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		assert ctrl._parse_agent_cost("Total cost:\t$0.50") == 0.50

	def test_not_found(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		assert ctrl._parse_agent_cost("random text with no cost info") == 0.0

	def test_large_amount(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		assert ctrl._parse_agent_cost("Total cost: $123.45") == 123.45

	def test_integer_amount(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		assert ctrl._parse_agent_cost("Total cost: $5") == 5.0


class TestCompletionReport:
	async def test_completion_report_generation(self, tmp_path: Path) -> None:
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		ctrl._start_commit = None
		ctrl._total_cost_usd = 2.50
		ctrl._agent_costs = {"worker-1": 1.50, "worker-2": 1.00}

		# Add tasks
		await ctrl.execute_decisions([
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Task A"}),
			PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Task B"}),
		])
		ctrl.tasks[0].status = TaskStatus.COMPLETED
		ctrl.tasks[0].result_summary = "Done A"
		ctrl.tasks[1].status = TaskStatus.FAILED
		ctrl.tasks[1].result_summary = "Failed B"
		ctrl.tasks[1].attempt_count = 2

		report = await ctrl._generate_completion_report()

		assert "# Swarm Completion Report" in report
		assert "Tasks completed: 1" in report
		assert "Tasks failed: 1" in report
		assert "$2.50" in report
		assert "worker-1: $1.50" in report
		assert "Done A" in report
		assert "Failed B" in report
		# Report file written
		report_path = tmp_path / ".autodev-swarm-report.md"
		assert report_path.exists()
		assert "Completion Report" in report_path.read_text()


class TestCommitPerTask:
	async def test_commit_on_success(self, tmp_path: Path) -> None:
		"""Verify git commit is called after successful task completion."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Fix bug"}, priority=10),
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "w1", "prompt": "fix"}, priority=9),
			])

		# Simulate completion
		mock_proc.returncode = 0
		ad_result = '{"status":"completed","summary":"Fixed"}'
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(return_value=f"AD_RESULT:{ad_result}".encode())

		with (
			patch.object(ctrl, "_get_git_changed_files", new=AsyncMock(return_value=["a.py"])),
			patch.object(ctrl, "_auto_commit_task", new=AsyncMock(return_value="abc123")) as mock_commit,
		):
			events = await ctrl.monitor_agents()

		mock_commit.assert_called_once_with("Fix bug", "w1")
		assert events[0]["commit_hash"] == "abc123"

	async def test_no_commit_on_failure(self, tmp_path: Path) -> None:
		"""Verify git commit is NOT called after failed task completion."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Fix bug"}, priority=10),
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "w1", "prompt": "fix"}, priority=9),
			])

		# Simulate failure
		mock_proc.returncode = 1
		ad_result = '{"status":"failed","summary":"Crashed"}'
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(return_value=f"AD_RESULT:{ad_result}".encode())

		with (
			patch.object(ctrl, "_get_git_changed_files", new=AsyncMock(return_value=[])),
			patch.object(ctrl, "_auto_commit_task", new=AsyncMock()) as mock_commit,
		):
			await ctrl.monitor_agents()

		mock_commit.assert_not_called()


class TestStaleTaskDetection:
	async def test_stale_task_hint_added(self, tmp_path: Path) -> None:
		"""Verify prompt includes stale-task note when file was recently modified."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		# Simulate that worker-1 recently changed src/parser.py
		ctrl._recent_changes = {"worker-1": ["src/parser.py"]}

		# Create task with files_hint matching the recently changed file
		await ctrl.execute_decisions([
			PlannerDecision(
				type=DecisionType.CREATE_TASK,
				payload={"title": "Fix parser", "files_hint": ["src/parser.py"]},
			),
		])
		task_id = ctrl.tasks[0].id

		mock_proc = MagicMock(returncode=None)
		captured_prompt = None

		async def capture_spawn(agent, prompt):
			nonlocal captured_prompt
			captured_prompt = prompt
			return mock_proc

		with patch.object(ctrl, "_spawn_claude_session", side_effect=capture_spawn):
			await ctrl.execute_decisions([
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={"name": "w2", "prompt": "Fix the parser bug", "task_id": task_id},
				),
			])

		assert captured_prompt is not None
		assert "src/parser.py was recently modified by worker-1" in captured_prompt

	async def test_no_stale_hint_when_no_overlap(self, tmp_path: Path) -> None:
		"""No stale-task note when files don't overlap."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		ctrl._recent_changes = {"worker-1": ["src/other.py"]}

		await ctrl.execute_decisions([
			PlannerDecision(
				type=DecisionType.CREATE_TASK,
				payload={"title": "Fix parser", "files_hint": ["src/parser.py"]},
			),
		])
		task_id = ctrl.tasks[0].id

		captured_prompt = None

		async def capture_spawn(agent, prompt):
			nonlocal captured_prompt
			captured_prompt = prompt
			return MagicMock(returncode=None)

		with patch.object(ctrl, "_spawn_claude_session", side_effect=capture_spawn):
			await ctrl.execute_decisions([
				PlannerDecision(
					type=DecisionType.SPAWN,
					payload={"name": "w2", "prompt": "Fix the parser bug", "task_id": task_id},
				),
			])

		assert "recently modified" not in captured_prompt


class TestCostTracking:
	async def test_cost_accumulated_on_completion(self, tmp_path: Path) -> None:
		"""Verify agent cost is accumulated in controller state."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		mock_proc = MagicMock(returncode=None)
		with patch.object(ctrl, "_spawn_claude_session", new=AsyncMock(return_value=mock_proc)):
			await ctrl.execute_decisions([
				PlannerDecision(type=DecisionType.CREATE_TASK, payload={"title": "Task"}, priority=10),
				PlannerDecision(type=DecisionType.SPAWN, payload={"name": "w1", "prompt": "do"}, priority=9),
			])

		mock_proc.returncode = 0
		output = 'Some output\nTotal cost: $3.75\nAD_RESULT:{"status":"completed","summary":"Done"}'
		mock_proc.stdout = AsyncMock()
		mock_proc.stdout.read = AsyncMock(return_value=output.encode())

		with (
			patch.object(ctrl, "_get_git_changed_files", new=AsyncMock(return_value=[])),
			patch.object(ctrl, "_auto_commit_task", new=AsyncMock(return_value=None)),
		):
			events = await ctrl.monitor_agents()

		assert ctrl._total_cost_usd == 3.75
		assert ctrl._agent_costs["w1"] == 3.75
		assert events[0]["cost_usd"] == 3.75


class TestRunTraceReview:
	@pytest.mark.asyncio
	async def test_writes_review_and_learnings(self, tmp_path: Path) -> None:
		"""Trace review writes REVIEW.md and appends recommendations to learnings."""
		config = _make_config(tmp_path)
		db = _make_db()
		# Provide traces with a file hotspot so recommendations are generated
		db.get_agent_traces = lambda run_id=None, limit=50: [
			{
				"id": "t1", "run_id": "test-run", "agent_name": "a1",
				"agent_id": "aid-1", "task_id": "tid-1", "task_title": "task 1",
				"started_at": "2026-01-01T00:00:00", "ended_at": "2026-01-01T00:02:00",
				"duration_s": 120.0, "exit_code": 0, "cost_usd": 1.0,
				"files_changed": ["src/shared.py"], "trace_path": "", "output_tail": "",
			},
			{
				"id": "t2", "run_id": "test-run", "agent_name": "a2",
				"agent_id": "aid-2", "task_id": "tid-2", "task_title": "task 2",
				"started_at": "2026-01-01T00:00:00", "ended_at": "2026-01-01T00:03:00",
				"duration_s": 180.0, "exit_code": 1, "cost_usd": 2.0,
				"files_changed": ["src/shared.py"], "trace_path": "",
				"output_tail": "Error: something broke",
			},
		] if run_id == "test-run" else []
		ctrl = SwarmController(config, _make_swarm_config(), db)
		ctrl._run_id = "test-run"

		await ctrl._run_trace_review()

		# Verify REVIEW.md was written
		review_path = ctrl._trace_dir / "REVIEW.md"
		assert review_path.exists()
		content = review_path.read_text()
		assert "# Trace Review: test-run" in content
		assert "src/shared.py" in content

		# Verify learnings file was updated
		learnings_path = tmp_path / ".autodev-swarm-learnings.md"
		assert learnings_path.exists()
		learnings_content = learnings_path.read_text()
		assert "src/shared.py" in learnings_content

	@pytest.mark.asyncio
	async def test_skips_when_no_traces(self, tmp_path: Path) -> None:
		"""Trace review is a no-op when no traces exist."""
		config = _make_config(tmp_path)
		db = _make_db()
		db.get_agent_traces = lambda run_id=None, limit=50: []
		ctrl = SwarmController(config, _make_swarm_config(), db)

		await ctrl._run_trace_review()

		review_path = ctrl._trace_dir / "REVIEW.md"
		assert not review_path.exists()

	@pytest.mark.asyncio
	async def test_does_not_crash_on_error(self, tmp_path: Path) -> None:
		"""Trace review failure should not propagate."""
		config = _make_config(tmp_path)
		db = _make_db()
		db.get_agent_traces = MagicMock(side_effect=RuntimeError("db error"))
		ctrl = SwarmController(config, _make_swarm_config(), db)

		# Should not raise
		await ctrl._run_trace_review()
