"""Tests for swarm worker prompt builder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from autodev.config import SwarmConfig
from autodev.swarm.models import (
	AgentRole,
	AgentStatus,
	SwarmAgent,
	SwarmTask,
	TaskPriority,
	TaskStatus,
)
from autodev.swarm.worker_prompt import (
	_file_conflict_section,
	_identity_section,
	_inbox_section,
	_mcp_section,
	_peer_section,
	_result_protocol_section,
	_skill_creation_section,
	_task_pool_section,
	build_worker_prompt,
)


def _make_agent(**overrides: object) -> SwarmAgent:
	defaults = {"name": "worker-1", "role": AgentRole.IMPLEMENTER, "status": AgentStatus.WORKING}
	defaults.update(overrides)
	return SwarmAgent(**defaults)


def _make_task(**overrides: object) -> SwarmTask:
	defaults = {"title": "Fix bug", "description": "fix it"}
	defaults.update(overrides)
	return SwarmTask(**defaults)


def _make_config(tmp_path: Path) -> MagicMock:
	config = MagicMock()
	config.target.resolved_path = str(tmp_path)
	config.target.objective = "Build compiler"
	return config


def _make_swarm_config(**overrides: object) -> SwarmConfig:
	sc = SwarmConfig()
	for k, v in overrides.items():
		setattr(sc, k, v)
	return sc


class TestIdentitySection:
	def test_includes_agent_name_and_role(self) -> None:
		agent = _make_agent()
		text = _identity_section(agent, "autodev-test")
		assert "worker-1" in text
		assert "implementer" in text
		assert "autodev-test" in text


class TestPeerSection:
	def test_no_peers(self) -> None:
		agent = _make_agent()
		text = _peer_section(agent, [agent])
		assert "No other agents" in text

	def test_shows_active_peers(self) -> None:
		agent = _make_agent(name="worker-1")
		peer = _make_agent(name="worker-2", role=AgentRole.RESEARCHER)
		text = _peer_section(agent, [agent, peer])
		assert "worker-2" in text
		assert "researcher" in text

	def test_excludes_dead_peers(self) -> None:
		agent = _make_agent(name="worker-1")
		dead = _make_agent(name="dead-agent", status=AgentStatus.DEAD)
		text = _peer_section(agent, [agent, dead])
		assert "No other agents" in text


class TestTaskPoolSection:
	def test_empty_when_no_pending(self) -> None:
		task = _make_task(status=TaskStatus.COMPLETED)
		text = _task_pool_section([task])
		assert text == ""

	def test_shows_pending_tasks(self) -> None:
		task = _make_task(title="Write tests", priority=TaskPriority.HIGH)
		text = _task_pool_section([task])
		assert "Write tests" in text
		assert "HIGH" in text

	def test_truncates_long_list(self) -> None:
		tasks = [_make_task(title=f"Task {i}") for i in range(8)]
		text = _task_pool_section(tasks)
		assert "3 more" in text


class TestFileConflictSection:
	def test_no_conflicts(self) -> None:
		agent = _make_agent()
		text = _file_conflict_section(agent, [agent], [])
		assert text == ""

	def test_shows_conflicting_files(self) -> None:
		agent = _make_agent(name="a1")
		task = _make_task(files_hint=["src/main.py", "src/lib.py"])
		other = _make_agent(name="a2", current_task_id=task.id)
		text = _file_conflict_section(agent, [agent, other], [task])
		assert "src/main.py" in text
		assert "src/lib.py" in text


class TestInboxSection:
	def test_includes_inbox_paths(self) -> None:
		agent = _make_agent()
		text = _inbox_section(agent, "autodev-test")
		assert "autodev-test" in text
		assert agent.name in text
		assert "team-lead.json" in text


class TestMcpSection:
	def test_empty_when_disabled(self) -> None:
		sc = _make_swarm_config(inherit_global_mcps=False)
		assert _mcp_section(sc) == ""

	def test_lists_tools_when_enabled(self) -> None:
		sc = _make_swarm_config(inherit_global_mcps=True)
		text = _mcp_section(sc)
		assert "browser" in text.lower()
		assert "nanobanana" in text.lower()

	def test_shows_allowed_mcps(self) -> None:
		sc = _make_swarm_config(inherit_global_mcps=True, allowed_mcps=["obsidian", "stitch"])
		text = _mcp_section(sc)
		assert "obsidian" in text
		assert "stitch" in text


class TestSkillCreationSection:
	def test_includes_skill_format(self) -> None:
		text = _skill_creation_section()
		assert "SKILL.md" in text
		assert "frontmatter" in text


class TestResultProtocol:
	def test_includes_ad_result(self) -> None:
		text = _result_protocol_section()
		assert "AD_RESULT" in text
		assert "discoveries" in text


class TestBuildWorkerPrompt:
	def test_includes_task_and_identity(self, tmp_path: Path) -> None:
		agent = _make_agent()
		prompt = build_worker_prompt(
			agent=agent,
			task_prompt="Fix the compiler bug in codegen.c",
			team_name="autodev-test",
			agents=[agent],
			tasks=[],
			config=_make_config(tmp_path),
			swarm_config=_make_swarm_config(),
		)
		assert "Fix the compiler bug" in prompt
		assert "worker-1" in prompt
		assert "AD_RESULT" in prompt

	def test_includes_peer_info(self, tmp_path: Path) -> None:
		agent = _make_agent(name="a1")
		peer = _make_agent(name="a2", role=AgentRole.TESTER)
		prompt = build_worker_prompt(
			agent=agent,
			task_prompt="Do stuff",
			team_name="autodev-test",
			agents=[agent, peer],
			tasks=[],
			config=_make_config(tmp_path),
			swarm_config=_make_swarm_config(),
		)
		assert "a2" in prompt
		assert "tester" in prompt
