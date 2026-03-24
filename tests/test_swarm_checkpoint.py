"""Tests for swarm checkpoint/resume functionality."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autodev.config import SwarmConfig
from autodev.swarm.controller import SwarmController
from autodev.swarm.models import (
	AgentRole,
	AgentStatus,
	SwarmAgent,
	SwarmTask,
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


def _make_controller(tmp_path: Path) -> SwarmController:
	"""Create a controller with mock agents and tasks for testing."""
	ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())

	agent = SwarmAgent(
		id="agent-001",
		name="impl-1",
		role=AgentRole.IMPLEMENTER,
		status=AgentStatus.WORKING,
		current_task_id="task-001",
		spawned_at="2026-01-01T00:00:00+00:00",
		tasks_completed=2,
		tasks_failed=1,
	)
	ctrl._agents["agent-001"] = agent

	task = SwarmTask(
		id="task-001",
		title="Fix parser bug",
		description="The parser fails on nested expressions",
		priority=TaskPriority.HIGH,
		status=TaskStatus.CLAIMED,
		claimed_by="agent-001",
		depends_on=["task-000"],
		files_hint=["src/parser.py"],
		attempt_count=1,
		max_attempts=3,
		result_summary="",
	)
	ctrl._tasks["task-001"] = task

	pending_task = SwarmTask(
		id="task-002",
		title="Add tests",
		description="Write unit tests",
		priority=TaskPriority.NORMAL,
		status=TaskStatus.PENDING,
	)
	ctrl._tasks["task-002"] = pending_task

	ctrl._total_cost_usd = 1.23
	ctrl._agent_costs = {"impl-1": 1.23}
	ctrl._start_commit = "abc123"

	return ctrl


class TestWriteCheckpoint:
	def test_write_checkpoint_creates_valid_json(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path)
		ctrl._write_checkpoint()

		checkpoint_path = tmp_path / ".autodev-swarm-checkpoint.json"
		assert checkpoint_path.exists()

		data = json.loads(checkpoint_path.read_text())
		assert data["version"] == 1
		assert data["run_id"] == ctrl._run_id
		assert data["team_name"] == ctrl._team_name
		assert data["start_commit"] == "abc123"
		assert data["total_cost_usd"] == 1.23
		assert data["agent_costs"] == {"impl-1": 1.23}
		assert "config_hash" in data
		assert isinstance(data["config_hash"], str)
		assert len(data["config_hash"]) == 16

		# Verify agents
		assert "agent-001" in data["agents"]
		agent_data = data["agents"]["agent-001"]
		assert agent_data["name"] == "impl-1"
		assert agent_data["role"] == "implementer"
		assert agent_data["status"] == "working"
		assert agent_data["current_task_id"] == "task-001"
		assert agent_data["tasks_completed"] == 2
		assert agent_data["tasks_failed"] == 1
		assert agent_data["pid"] is None  # no process attached

		# Verify tasks
		assert "task-001" in data["tasks"]
		task_data = data["tasks"]["task-001"]
		assert task_data["title"] == "Fix parser bug"
		assert task_data["priority"] == 2  # HIGH
		assert task_data["status"] == "claimed"
		assert task_data["claimed_by"] == "agent-001"
		assert task_data["depends_on"] == ["task-000"]
		assert task_data["files_hint"] == ["src/parser.py"]

		# Verify planner stub
		assert data["planner"]["cycle_count"] is None
		assert data["planner"]["test_history"] == []

	def test_checkpoint_with_planner_state(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path)
		ctrl.update_checkpoint_planner_state(
			cycle_count=5,
			test_history=[10, 12, 15],
			completion_history=[1, 2, 3],
			failure_history=[0, 1, 0],
			cost_history=[0.5, 1.0, 1.23],
		)
		ctrl._write_checkpoint()

		data = json.loads((tmp_path / ".autodev-swarm-checkpoint.json").read_text())
		assert data["planner"]["cycle_count"] == 5
		assert data["planner"]["test_history"] == [10, 12, 15]


class TestResumeFromCheckpoint:
	def test_resume_restores_tasks(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path)
		ctrl._write_checkpoint()

		config = _make_config(tmp_path)
		restored = SwarmController.from_checkpoint(config, _make_swarm_config(), _make_db())

		assert len(restored._tasks) == 2
		task = restored._tasks["task-001"]
		assert task.title == "Fix parser bug"
		assert task.priority == TaskPriority.HIGH
		assert task.status == TaskStatus.CLAIMED
		assert task.claimed_by == "agent-001"
		assert task.depends_on == ["task-000"]
		assert task.files_hint == ["src/parser.py"]
		assert task.attempt_count == 1
		assert task.max_attempts == 3

		task2 = restored._tasks["task-002"]
		assert task2.status == TaskStatus.PENDING

		assert restored._total_cost_usd == 1.23
		assert restored._agent_costs == {"impl-1": 1.23}
		assert restored._run_id == ctrl._run_id
		assert restored._start_commit == "abc123"

	def test_resume_marks_dead_pid_as_dead(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path)
		ctrl._write_checkpoint()

		# Patch the checkpoint to include a non-existent PID
		checkpoint_path = tmp_path / ".autodev-swarm-checkpoint.json"
		data = json.loads(checkpoint_path.read_text())
		data["agents"]["agent-001"]["pid"] = 99999
		data["agents"]["agent-001"]["status"] = "working"
		checkpoint_path.write_text(json.dumps(data))

		config = _make_config(tmp_path)
		# os.kill(99999, 0) should raise OSError for non-existent process
		restored = SwarmController.from_checkpoint(config, _make_swarm_config(), _make_db())

		agent = restored._agents["agent-001"]
		assert agent.status == AgentStatus.DEAD

	def test_resume_marks_alive_pid_as_working(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path)
		ctrl._write_checkpoint()

		# Use our own PID (guaranteed alive)
		my_pid = os.getpid()
		checkpoint_path = tmp_path / ".autodev-swarm-checkpoint.json"
		data = json.loads(checkpoint_path.read_text())
		data["agents"]["agent-001"]["pid"] = my_pid
		data["agents"]["agent-001"]["status"] = "dead"  # should be overridden
		checkpoint_path.write_text(json.dumps(data))

		config = _make_config(tmp_path)
		restored = SwarmController.from_checkpoint(config, _make_swarm_config(), _make_db())

		agent = restored._agents["agent-001"]
		assert agent.status == AgentStatus.WORKING

	def test_resume_no_checkpoint_raises(self, tmp_path: Path) -> None:
		config = _make_config(tmp_path)
		with pytest.raises(FileNotFoundError, match="No checkpoint file"):
			SwarmController.from_checkpoint(config, _make_swarm_config(), _make_db())


class TestCheckpointAtomicWrite:
	def test_checkpoint_atomic_write(self, tmp_path: Path) -> None:
		"""Verify the .tmp file doesn't persist after write (atomic rename)."""
		ctrl = _make_controller(tmp_path)
		ctrl._write_checkpoint()

		checkpoint_path = tmp_path / ".autodev-swarm-checkpoint.json"
		tmp_file = checkpoint_path.with_suffix(".tmp")
		assert checkpoint_path.exists()
		assert not tmp_file.exists()


class TestDetectOrphanAgents:
	def test_removes_stale_checkpoint_when_all_dead(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path)
		ctrl._write_checkpoint()

		# Patch PID to a non-existent one
		checkpoint_path = tmp_path / ".autodev-swarm-checkpoint.json"
		data = json.loads(checkpoint_path.read_text())
		data["agents"]["agent-001"]["pid"] = 99999
		checkpoint_path.write_text(json.dumps(data))

		# Fresh controller should remove stale checkpoint
		ctrl2 = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		ctrl2._detect_orphan_agents()
		assert not checkpoint_path.exists()

	def test_keeps_checkpoint_when_alive(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path)
		ctrl._write_checkpoint()

		# Use our own PID
		checkpoint_path = tmp_path / ".autodev-swarm-checkpoint.json"
		data = json.loads(checkpoint_path.read_text())
		data["agents"]["agent-001"]["pid"] = os.getpid()
		checkpoint_path.write_text(json.dumps(data))

		ctrl2 = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		ctrl2._detect_orphan_agents()
		assert checkpoint_path.exists()


class TestConfigDrift:
	def test_config_hash_deterministic(self, tmp_path: Path) -> None:
		ctrl = _make_controller(tmp_path)
		h1 = ctrl._config_hash()
		h2 = ctrl._config_hash()
		assert h1 == h2
		assert len(h1) == 16

	def test_config_hash_changes_with_config(self, tmp_path: Path) -> None:
		ctrl1 = SwarmController(
			_make_config(tmp_path), _make_swarm_config(max_agents=4), _make_db()
		)
		ctrl2 = SwarmController(
			_make_config(tmp_path), _make_swarm_config(max_agents=8), _make_db()
		)
		assert ctrl1._config_hash() != ctrl2._config_hash()

	def test_resume_logs_config_drift(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
		ctrl = SwarmController(
			_make_config(tmp_path), _make_swarm_config(max_agents=4), _make_db()
		)
		ctrl._agents["a1"] = SwarmAgent(id="a1", name="x", status=AgentStatus.DEAD)
		ctrl._tasks["t1"] = SwarmTask(id="t1", title="y", status=TaskStatus.PENDING)
		ctrl._write_checkpoint()

		# Resume with different config
		config = _make_config(tmp_path)
		with caplog.at_level("WARNING"):
			restored = SwarmController.from_checkpoint(
				config, _make_swarm_config(max_agents=8), _make_db()
			)
		assert "Config drift detected" in caplog.text
		# Should still restore successfully
		assert len(restored._tasks) == 1


class TestCorruptCheckpoint:
	def test_corrupt_json_raises_value_error(self, tmp_path: Path) -> None:
		checkpoint_path = tmp_path / ".autodev-swarm-checkpoint.json"
		checkpoint_path.write_text("not valid json {{{")

		config = _make_config(tmp_path)
		with pytest.raises(ValueError, match="Corrupt checkpoint file"):
			SwarmController.from_checkpoint(config, _make_swarm_config(), _make_db())

	def test_empty_file_raises_value_error(self, tmp_path: Path) -> None:
		checkpoint_path = tmp_path / ".autodev-swarm-checkpoint.json"
		checkpoint_path.write_text("")

		config = _make_config(tmp_path)
		with pytest.raises(ValueError, match="Corrupt checkpoint file"):
			SwarmController.from_checkpoint(config, _make_swarm_config(), _make_db())


class TestCheckpointRoundTrip:
	def test_save_load_round_trip_preserves_all_fields(self, tmp_path: Path) -> None:
		"""Full round-trip: save checkpoint, load it, verify everything matches."""
		ctrl = _make_controller(tmp_path)
		ctrl.update_checkpoint_planner_state(
			cycle_count=7,
			test_history=[5, 8, 12],
			completion_history=[1, 3, 5],
			failure_history=[0, 0, 1],
			cost_history=[0.1, 0.5, 1.23],
		)
		ctrl._write_checkpoint()

		config = _make_config(tmp_path)
		restored = SwarmController.from_checkpoint(config, _make_swarm_config(), _make_db())

		# Scalar state
		assert restored._run_id == ctrl._run_id
		assert restored._team_name == ctrl._team_name
		assert restored._start_commit == ctrl._start_commit
		assert restored._total_cost_usd == ctrl._total_cost_usd
		assert restored._agent_costs == ctrl._agent_costs

		# Agent state
		assert set(restored._agents.keys()) == set(ctrl._agents.keys())
		orig_agent = ctrl._agents["agent-001"]
		rest_agent = restored._agents["agent-001"]
		assert rest_agent.name == orig_agent.name
		assert rest_agent.role == orig_agent.role
		assert rest_agent.current_task_id == orig_agent.current_task_id
		assert rest_agent.tasks_completed == orig_agent.tasks_completed

		# Task state
		assert set(restored._tasks.keys()) == set(ctrl._tasks.keys())
		orig_task = ctrl._tasks["task-001"]
		rest_task = restored._tasks["task-001"]
		assert rest_task.title == orig_task.title
		assert rest_task.priority == orig_task.priority
		assert rest_task.status == orig_task.status
		assert rest_task.depends_on == orig_task.depends_on
		assert rest_task.files_hint == orig_task.files_hint

		# Planner state
		assert restored._checkpoint_planner_state["cycle_count"] == 7
		assert restored._checkpoint_planner_state["test_history"] == [5, 8, 12]
