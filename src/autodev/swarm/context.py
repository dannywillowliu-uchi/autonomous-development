"""Context synthesizer -- aggregates swarm state into a coherent snapshot for the planner."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autodev.swarm.models import (
	AgentStatus,
	StagnationSignal,
	SwarmAgent,
	SwarmState,
	SwarmTask,
	TaskStatus,
)

if TYPE_CHECKING:
	from autodev.config import MissionConfig
	from autodev.db import Database

logger = logging.getLogger(__name__)


class ContextSynthesizer:
	"""Builds a SwarmState snapshot from all available signals.

	Reads from:
	- Database (tasks, agents, handoffs, knowledge)
	- Team inboxes (file-based JSON messages)
	- Core test results
	- Skill/tool registries
	- Git state
	"""

	def __init__(self, config: MissionConfig, db: Database, team_name: str) -> None:
		self._config = config
		self._db = db
		self._team_name = team_name
		self._cycle_number = 0
		self._start_time: float | None = None
		self._metric_history: dict[str, list[float]] = {}
		self._stagnation_window = 5  # cycles to look back

	def build_state(
		self,
		agents: list[SwarmAgent],
		tasks: list[SwarmTask],
		core_test_results: dict[str, Any] | None = None,
		total_cost_usd: float = 0.0,
		wall_time_seconds: float = 0.0,
	) -> SwarmState:
		"""Build a complete SwarmState snapshot for the planner."""
		self._cycle_number += 1

		recent_completions = self._get_recent_completions(tasks)
		recent_failures = self._get_recent_failures(tasks)
		discoveries = self._get_recent_discoveries(tasks)
		skills = self._discover_skills()
		tools = self._discover_tools()
		stagnation = self._detect_stagnation(tasks, core_test_results)
		files_in_flight = self._get_files_in_flight(agents, tasks)

		return SwarmState(
			mission_objective=self._config.target.objective,
			agents=agents,
			tasks=tasks,
			recent_completions=recent_completions,
			recent_failures=recent_failures,
			recent_discoveries=discoveries,
			available_skills=skills,
			available_tools=tools,
			stagnation_signals=stagnation,
			core_test_results=core_test_results or {},
			cycle_number=self._cycle_number,
			total_cost_usd=total_cost_usd,
			wall_time_seconds=wall_time_seconds,
			files_in_flight=files_in_flight,
		)

	def render_for_planner(self, state: SwarmState) -> str:
		"""Render SwarmState as a structured text block for the planner prompt."""
		sections = []

		# Mission
		sections.append(f"## Mission\n{state.mission_objective}")

		# Agents
		agent_lines = []
		for a in state.agents:
			task_info = f" (task: {a.current_task_id})" if a.current_task_id else ""
			agent_lines.append(
				f"- {a.name} [{a.role.value}] status={a.status.value}{task_info} "
				f"completed={a.tasks_completed} failed={a.tasks_failed}"
			)
		if agent_lines:
			sections.append("## Active Agents\n" + "\n".join(agent_lines))
		else:
			sections.append("## Active Agents\nNone currently active.")

		# Task pool
		pending = [t for t in state.tasks if t.status == TaskStatus.PENDING]
		in_progress = [t for t in state.tasks if t.status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS)]
		if pending or in_progress:
			task_lines = []
			for t in in_progress:
				task_lines.append(f"- [IN PROGRESS] {t.title} (claimed by: {t.claimed_by}, attempt {t.attempt_count})")
			for t in pending:
				deps = f" (blocked by: {', '.join(t.depends_on)})" if t.depends_on else ""
				task_lines.append(f"- [PENDING] {t.title}{deps}")
			sections.append("## Task Pool\n" + "\n".join(task_lines))

		# Recent completions
		if state.recent_completions:
			comp_lines = [f"- {c.get('title', '?')}: {c.get('summary', '')}" for c in state.recent_completions[:5]]
			sections.append("## Recent Completions\n" + "\n".join(comp_lines))

		# Recent failures
		if state.recent_failures:
			fail_lines = [
				f"- {f.get('title', '?')} (attempt {f.get('attempt', '?')}): {f.get('error', '')}"
				for f in state.recent_failures[:5]
			]
			sections.append("## Recent Failures\n" + "\n".join(fail_lines))

		# Discoveries
		if state.recent_discoveries:
			disc_lines = [f"- {d}" for d in state.recent_discoveries[:10]]
			sections.append("## Recent Discoveries\n" + "\n".join(disc_lines))

		# Core test results
		if state.core_test_results:
			tr = state.core_test_results
			sections.append(
				f"## Core Test Results\n"
				f"Pass: {tr.get('pass', '?')} | Fail: {tr.get('fail', '?')} | "
				f"Skip: {tr.get('skip', '?')} | Total: {tr.get('total', '?')}"
			)

		# Stagnation
		if state.stagnation_signals:
			stag_lines = []
			for s in state.stagnation_signals:
				stag_lines.append(
					f"- {s.metric}: stagnant for {s.cycles_stagnant} cycles "
					f"(values: {s.value_history[-3:]}). Suggested pivot: {s.suggested_pivot}"
				)
			sections.append("## STAGNATION WARNINGS\n" + "\n".join(stag_lines))

		# Available skills/tools
		if state.available_skills:
			sections.append("## Available Skills\n" + ", ".join(state.available_skills))
		if state.available_tools:
			sections.append("## Available Tools\n" + ", ".join(state.available_tools))

		# Files in flight
		if state.files_in_flight:
			sections.append("## Files Currently Being Modified\n" + "\n".join(f"- {f}" for f in state.files_in_flight))

		# Meta
		sections.append(
			f"## Meta\n"
			f"Cycle: {state.cycle_number} | Cost: ${state.total_cost_usd:.2f} | "
			f"Wall time: {state.wall_time_seconds / 60:.1f}min"
		)

		return "\n\n".join(sections)

	def _get_recent_completions(self, tasks: list[SwarmTask]) -> list[dict[str, Any]]:
		"""Get tasks completed since last cycle."""
		return [
			{"title": t.title, "summary": t.result_summary, "id": t.id}
			for t in tasks
			if t.status == TaskStatus.COMPLETED
		]

	def _get_recent_failures(self, tasks: list[SwarmTask]) -> list[dict[str, Any]]:
		"""Get tasks that failed since last cycle."""
		return [
			{
				"title": t.title,
				"attempt": t.attempt_count,
				"error": t.result_summary,
				"id": t.id,
				"retries_left": t.max_attempts - t.attempt_count,
			}
			for t in tasks
			if t.status == TaskStatus.FAILED
		]

	def _get_recent_discoveries(self, tasks: list[SwarmTask] | None = None) -> list[str]:
		"""Read discoveries from inbox messages, task results, and knowledge items."""
		discoveries: list[str] = []

		# Read from team inbox (all message types, not just keyword-filtered)
		inbox_dir = Path.home() / ".claude" / "teams" / self._team_name / "inboxes"
		if inbox_dir.exists():
			for inbox_file in inbox_dir.glob("*.json"):
				try:
					messages = json.loads(inbox_file.read_text())
					for msg in messages[-20:]:
						msg_type = msg.get("type", "")
						text = msg.get("text", "")
						sender = msg.get("from", inbox_file.stem)
						if msg_type in ("discovery", "blocked", "question"):
							discoveries.append(f"[{sender}] ({msg_type}) {text}")
						elif "discovery:" in text.lower() or "found:" in text.lower():
							discoveries.append(f"[{sender}] {text}")
				except (json.JSONDecodeError, OSError):
					pass

		# Extract discoveries from completed task results
		if tasks:
			for t in tasks:
				if t.status == TaskStatus.COMPLETED and t.result_summary:
					try:
						result = json.loads(t.result_summary) if t.result_summary.startswith("{") else None
						if result and result.get("discoveries"):
							for d in result["discoveries"]:
								discoveries.append(f"[task:{t.title[:30]}] {d}")
					except (json.JSONDecodeError, ValueError):
						pass

		# Read from DB knowledge items
		try:
			knowledge = self._db.get_knowledge_for_mission(self._config.target.name)
			for item in knowledge[-10:]:
				discoveries.append(f"[knowledge] {item.content}")
		except Exception:
			pass

		return discoveries

	def _discover_skills(self) -> list[str]:
		"""Find available skills in .claude/skills/."""
		skills: list[str] = []
		skills_dir = Path(self._config.target.resolved_path) / ".claude" / "skills"
		if skills_dir.exists():
			for skill_dir in skills_dir.iterdir():
				if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
					skills.append(skill_dir.name)
		return skills

	def _discover_tools(self) -> list[str]:
		"""Find available synthesized tools."""
		tools: list[str] = []
		try:
			from autodev.mcp_registry import MCPToolRegistry
			registry = MCPToolRegistry(self._db)
			all_tools = registry.list_all()
			tools = [t.name for t in all_tools[:20]]
		except Exception:
			pass
		return tools

	def _detect_stagnation(
		self,
		tasks: list[SwarmTask],
		core_test_results: dict[str, Any] | None,
	) -> list[StagnationSignal]:
		"""Detect stagnation across multiple metrics."""
		signals: list[StagnationSignal] = []

		# Track test pass count
		if core_test_results:
			pass_count = core_test_results.get("pass", 0)
			self._track_metric("test_pass_count", float(pass_count))
			stag = self._check_metric_stagnation(
				"test_pass_count",
				"Switch from implementation to research. Understand WHY tests are failing before trying more fixes.",
			)
			if stag:
				signals.append(stag)

		# Track completion rate
		completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
		self._track_metric("completed_tasks", float(completed))
		stag = self._check_metric_stagnation(
			"completed_tasks",
			"Reduce parallelism. Focus agents on fewer, higher-impact tasks.",
		)
		if stag:
			signals.append(stag)

		# Track failure rate
		failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
		if failed > 0:
			total = max(completed + failed, 1)
			fail_rate = failed / total
			self._track_metric("failure_rate", fail_rate)
			if fail_rate > 0.5:
				signals.append(StagnationSignal(
					metric="high_failure_rate",
					value_history=self._metric_history.get("failure_rate", [])[-5:],
					cycles_stagnant=1,
					suggested_pivot=(
						"More than half of tasks are failing. "
						"Spawn a research agent to diagnose the systemic issue."
					),
				))

		return signals

	def _track_metric(self, name: str, value: float) -> None:
		"""Record a metric value for stagnation detection."""
		if name not in self._metric_history:
			self._metric_history[name] = []
		self._metric_history[name].append(value)
		# Keep bounded
		if len(self._metric_history[name]) > 20:
			self._metric_history[name] = self._metric_history[name][-20:]

	def _check_metric_stagnation(self, name: str, pivot_suggestion: str) -> StagnationSignal | None:
		"""Check if a metric has been flat for too long."""
		history = self._metric_history.get(name, [])
		if len(history) < self._stagnation_window:
			return None

		recent = history[-self._stagnation_window:]
		if len(set(recent)) == 1:  # all values identical
			return StagnationSignal(
				metric=name,
				value_history=recent,
				cycles_stagnant=self._stagnation_window,
				suggested_pivot=pivot_suggestion,
			)
		return None

	def _get_files_in_flight(self, agents: list[SwarmAgent], tasks: list[SwarmTask]) -> list[str]:
		"""Get files currently being modified by active agents."""
		files: list[str] = []
		active_task_ids = {a.current_task_id for a in agents if a.current_task_id and a.status == AgentStatus.WORKING}
		for t in tasks:
			if t.id in active_task_ids and t.files_hint:
				files.extend(t.files_hint)
		return sorted(set(files))
