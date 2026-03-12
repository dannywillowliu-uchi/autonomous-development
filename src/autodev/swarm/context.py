"""Context synthesizer -- aggregates swarm state into a coherent snapshot for the planner."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
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
	from autodev.swarm.capabilities import CapabilityManifest

logger = logging.getLogger(__name__)

# Structured report fields that workers can include in inbox messages
STRUCTURED_REPORT_FIELDS = ("status", "progress", "files_changed", "tests_passing", "error")

# Inbox rotation defaults
DEFAULT_MAX_INBOX_BYTES = 500 * 1024  # 500 KB
DEFAULT_MAX_INBOX_MESSAGES = 500
DEFAULT_KEEP_MESSAGES = 200  # messages to keep after rotation


def rotate_inbox(
	inbox_path: Path,
	*,
	max_bytes: int = DEFAULT_MAX_INBOX_BYTES,
	max_messages: int = DEFAULT_MAX_INBOX_MESSAGES,
	keep_messages: int = DEFAULT_KEEP_MESSAGES,
) -> bool:
	"""Rotate an inbox file if it exceeds size or message count limits.

	Keeps the most recent `keep_messages` messages. Uses the same atomic
	write pattern (fcntl.flock + tempfile.mkstemp + os.rename) as the
	controller's _write_to_inbox to avoid data corruption.

	Returns True if rotation occurred.
	"""
	if not inbox_path.exists():
		return False

	# Quick size check before acquiring lock
	try:
		file_size = inbox_path.stat().st_size
	except OSError:
		return False

	if file_size <= max_bytes:
		# Still need to check message count, but skip if file is tiny
		if file_size < 1024:
			return False

	lock_path = inbox_path.with_suffix(".lock")
	tmp_fd = None
	tmp_path = None
	try:
		with open(lock_path, "w") as lock_file:
			fcntl.flock(lock_file, fcntl.LOCK_EX)

			try:
				raw = inbox_path.read_text()
				messages = json.loads(raw)
			except (json.JSONDecodeError, OSError):
				return False

			if not isinstance(messages, list):
				return False

			needs_rotation = len(raw.encode()) > max_bytes or len(messages) > max_messages
			if not needs_rotation:
				return False

			truncated = messages[-keep_messages:]
			new_data = json.dumps(truncated, indent=2).encode()

			tmp_fd, tmp_path = tempfile.mkstemp(
				dir=str(inbox_path.parent), suffix=".tmp"
			)
			os.write(tmp_fd, new_data)
			os.close(tmp_fd)
			tmp_fd = None
			os.rename(tmp_path, str(inbox_path))
			tmp_path = None

			logger.info(
				"Rotated inbox %s: %d -> %d messages",
				inbox_path.name, len(messages), len(truncated),
			)
			return True
	except OSError as e:
		logger.warning("Could not rotate inbox %s: %s", inbox_path.name, e)
		return False
	finally:
		if tmp_fd is not None:
			os.close(tmp_fd)
		if tmp_path is not None:
			try:
				os.unlink(tmp_path)
			except OSError:
				pass


def parse_structured_report(msg: dict[str, Any]) -> dict[str, Any]:
	"""Extract structured fields from an inbox message.

	Returns a dict with the standard fields (from, type, text) plus any
	structured report fields (status, progress, files_changed, tests_passing,
	error) that are present and valid. Unstructured messages return only
	the base fields.
	"""
	result: dict[str, Any] = {
		"from": msg.get("from", "unknown"),
		"type": msg.get("type", ""),
		"text": msg.get("text", ""),
	}

	status = msg.get("status")
	if isinstance(status, str) and status in ("working", "blocked", "completed"):
		result["status"] = status

	progress = msg.get("progress")
	if isinstance(progress, str) and progress:
		result["progress"] = progress

	files_changed = msg.get("files_changed")
	if isinstance(files_changed, list) and all(isinstance(f, str) for f in files_changed):
		result["files_changed"] = files_changed

	tests_passing = msg.get("tests_passing")
	if isinstance(tests_passing, (int, float)) and tests_passing >= 0:
		result["tests_passing"] = int(tests_passing)

	error = msg.get("error")
	if isinstance(error, str) and error:
		result["error"] = error

	if "timestamp" in msg:
		result["timestamp"] = msg["timestamp"]

	return result


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
		dead_agent_history: list[SwarmAgent] | None = None,
		capabilities: CapabilityManifest | None = None,
		recent_file_changes: dict[str, list[str]] | None = None,
		agent_costs: dict[str, float] | None = None,
	) -> SwarmState:
		"""Build a complete SwarmState snapshot for the planner."""
		self._cycle_number += 1

		recent_completions = self._get_recent_completions(tasks)
		recent_failures = self._get_recent_failures(tasks)
		discoveries = self._get_recent_discoveries(tasks)
		skills = self._discover_skills()
		tools = self._discover_tools()
		stagnation = self._detect_stagnation(tasks, core_test_results, dead_agent_history)
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
			capabilities=capabilities,
			dead_agent_history=dead_agent_history or [],
			recent_file_changes=recent_file_changes or {},
			agent_costs=agent_costs or {},
		)

	def get_agent_reports(self) -> dict[str, dict[str, Any]]:
		"""Get the latest structured report from each agent.

		Returns a dict mapping agent name -> their most recent structured report
		(parsed via parse_structured_report). Only includes messages with a
		"status" field. Useful for the planner to get a quick snapshot of
		what each agent is doing.
		"""
		reports: dict[str, dict[str, Any]] = {}
		inbox_dir = Path.home() / ".claude" / "teams" / self._team_name / "inboxes"
		if not inbox_dir.exists():
			return reports

		for inbox_file in inbox_dir.glob("*.json"):
			try:
				messages = json.loads(inbox_file.read_text())
			except (json.JSONDecodeError, OSError):
				continue
			if not isinstance(messages, list):
				continue

			# Walk backwards to find the latest structured report from each sender
			for msg in reversed(messages):
				if not isinstance(msg, dict):
					continue
				sender = msg.get("from", inbox_file.stem)
				if sender in reports:
					continue
				parsed = parse_structured_report(msg)
				if "status" in parsed:
					reports[sender] = parsed

		return reports

	def render_for_planner(self, state: SwarmState) -> str:
		"""Render SwarmState as a structured text block for the planner prompt."""
		sections = []

		# Mission
		sections.append(f"## Mission\n{state.mission_objective}")

		# Human directives (highest priority -- show before everything else)
		directives = self._get_human_directives()
		if directives:
			dir_lines = [f"- {d}" for d in directives]
			sections.append(
				"## HUMAN DIRECTIVES (PRIORITY)\n"
				"The human operator has injected the following directives. "
				"Treat these as highest priority and act on them immediately.\n"
				+ "\n".join(dir_lines)
			)

		# Task progress summary
		task_counts = self._count_task_statuses(state.tasks)
		total = len(state.tasks)
		if total > 0:
			sections.append(
				f"## Task Progress\n"
				f"{task_counts['completed']}/{total} completed | "
				f"{task_counts['in_progress']} in progress | "
				f"{task_counts['pending']} pending | "
				f"{task_counts['blocked']} blocked | "
				f"{task_counts['failed']} failed"
			)

		# Build lookup maps for dependency resolution and agent-task cross-referencing
		task_by_id = {t.id: t for t in state.tasks}
		agent_task_map = self._build_agent_task_map(state.agents, state.dead_agent_history, task_by_id)

		# Agents (with elapsed time)
		agent_lines = []
		now = datetime.now(timezone.utc)
		for a in state.agents:
			task_title = ""
			if a.current_task_id and a.current_task_id in task_by_id:
				task_title = f" task: \"{task_by_id[a.current_task_id].title}\""
			elif a.current_task_id:
				task_title = f" task: {a.current_task_id}"
			elapsed = self._format_elapsed(a.spawned_at, now)
			agent_lines.append(
				f"- {a.name} [{a.role.value}] status={a.status.value}{task_title} "
				f"elapsed={elapsed} completed={a.tasks_completed} failed={a.tasks_failed}"
			)
		if agent_lines:
			sections.append("## Active Agents\n" + "\n".join(agent_lines))
		else:
			sections.append("## Active Agents\nNone currently active.")

		# Agent progress reports (structured inbox data)
		agent_reports = self.get_agent_reports()
		if agent_reports:
			report_lines = []
			for name, report in agent_reports.items():
				parts = [f"- **{name}**: status={report['status']}"]
				if report.get("progress"):
					parts.append(f"progress=\"{report['progress']}\"")
				if report.get("tests_passing") is not None:
					parts.append(f"tests_passing={report['tests_passing']}")
				if report.get("files_changed"):
					parts.append(f"files=[{', '.join(report['files_changed'][:5])}]")
				if report.get("error"):
					parts.append(f"error=\"{report['error'][:100]}\"")
				report_lines.append(" ".join(parts))
			sections.append("## Agent Progress Reports\n" + "\n".join(report_lines))

		# Task pool -- all non-completed tasks with dependency status
		pending = [t for t in state.tasks if t.status == TaskStatus.PENDING]
		blocked = [t for t in state.tasks if t.status == TaskStatus.BLOCKED]
		in_progress = [t for t in state.tasks if t.status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS)]
		if pending or in_progress or blocked:
			task_lines = []
			for t in in_progress:
				claimer = self._resolve_claimer_name(t.claimed_by, state.agents)
				task_lines.append(
					f"- [IN PROGRESS] {t.title} (id: {t.id}, agent: {claimer}, attempt {t.attempt_count})"
				)
			for t in blocked:
				dep_info = self._render_dependency_status(t.depends_on, task_by_id)
				task_lines.append(f"- [BLOCKED] {t.title} (id: {t.id}){dep_info}")
			for t in pending:
				dep_info = self._render_dependency_status(t.depends_on, task_by_id)
				blocked_flag = " **BLOCKED**" if self._has_unmet_deps(t.depends_on, task_by_id) else ""
				task_lines.append(f"- [PENDING] {t.title} (id: {t.id}){dep_info}{blocked_flag}")
			sections.append("## Task Pool\n" + "\n".join(task_lines))

		# Completed tasks (brief)
		completed = [t for t in state.tasks if t.status == TaskStatus.COMPLETED]
		if completed:
			comp_lines = []
			for t in completed:
				summary = t.result_summary[:120] if t.result_summary else "no summary"
				comp_lines.append(f"- {t.title}: {summary}")
			sections.append("## Completed Tasks\n" + "\n".join(comp_lines[:10]))

		# Failed tasks (with retry info)
		failed = [t for t in state.tasks if t.status == TaskStatus.FAILED]
		if failed:
			fail_lines = []
			for t in failed:
				retries_left = t.max_attempts - t.attempt_count
				error = t.result_summary[:120] if t.result_summary else "no details"
				retry_info = f", {retries_left} retries left" if retries_left > 0 else ", NO retries left"
				fail_lines.append(f"- {t.title} (attempt {t.attempt_count}{retry_info}): {error}")
			sections.append("## Failed Tasks\n" + "\n".join(fail_lines[:10]))

		# Discoveries (grouped by source, capped at 10 total)
		if state.recent_discoveries:
			grouped = self._group_discoveries(state.recent_discoveries)
			disc_lines: list[str] = []
			total = 0
			for source, items in grouped.items():
				if total >= 10:
					break
				disc_lines.append(f"**{source}**")
				for item in items:
					if total >= 10:
						break
					disc_lines.append(f"  - {item}")
					total += 1
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

		# Capability manifest
		if state.capabilities:
			cap_lines: list[str] = []
			if state.capabilities.skills:
				cap_lines.append("### Skills")
				for s in state.capabilities.skills:
					desc = f" -- {s.description}" if s.description else ""
					cap_lines.append(f"- `{s.invocation}`{desc}")
			if state.capabilities.agents:
				cap_lines.append("### Agent Definitions")
				for a in state.capabilities.agents:
					desc = f" -- {a.description}" if a.description else ""
					model = f" (model: {a.model})" if a.model else ""
					cap_lines.append(f"- **{a.name}**{model}{desc}")
			if state.capabilities.hooks:
				cap_lines.append("### Hooks")
				for h in state.capabilities.hooks:
					cap_lines.append(f"- {h.event}: matcher={h.matcher} type={h.hook_type}")
			if state.capabilities.mcp_servers:
				cap_lines.append("### MCP Servers")
				for m in state.capabilities.mcp_servers:
					tools_info = f" tools: {', '.join(m.tools[:5])}" if m.tools else ""
					cap_lines.append(f"- **{m.name}** [{m.server_type}]{tools_info}")
			if cap_lines:
				sections.append("## Available Capabilities\n" + "\n".join(cap_lines))

		# Recent changes (files modified by recently completed agents)
		if state.recent_file_changes:
			change_lines = []
			for agent_name, files in state.recent_file_changes.items():
				file_list = ", ".join(files[:10])
				if len(files) > 10:
					file_list += f" (+{len(files) - 10} more)"
				change_lines.append(f"- **{agent_name}**: {file_list}")
			sections.append("## Recent Changes\n" + "\n".join(change_lines))

		# Files in flight
		if state.files_in_flight:
			sections.append("## Files Currently Being Modified\n" + "\n".join(f"- {f}" for f in state.files_in_flight))

		# Dead agents with accomplishment summaries
		if state.dead_agent_history:
			recent_dead = state.dead_agent_history[-10:]
			dead_lines = []
			for a in recent_dead:
				summary = agent_task_map.get(a.id, "")
				summary_text = f" -- {summary}" if summary else ""
				dead_lines.append(
					f"- {a.name} [{a.role.value}] completed={a.tasks_completed} "
					f"failed={a.tasks_failed}{summary_text}"
				)
			sections.append("## Recently Cleaned Up Agents\n" + "\n".join(dead_lines))

		# Completed work summary from dead agents with successful completions
		completed_agents = [a for a in state.dead_agent_history if a.tasks_completed > 0]
		if completed_agents:
			work_lines = self._build_completed_work_summary(
				completed_agents, agent_task_map, state.recent_discoveries,
			)
			if work_lines:
				sections.append("## Completed Work Summary\n" + "\n".join(work_lines))

		# Meta
		meta_parts = [
			f"Cycle: {state.cycle_number}",
			f"Cost: ${state.total_cost_usd:.2f}",
			f"Wall time: {state.wall_time_seconds / 60:.1f}min",
		]
		if state.total_cost_usd > 0 and state.agent_costs:
			avg_cost = state.total_cost_usd / len(state.agent_costs)
			meta_parts.append(f"Avg cost/agent: ${avg_cost:.2f}")
			top_spender = max(state.agent_costs.items(), key=lambda x: x[1])
			meta_parts.append(f"Top spender: {top_spender[0]} (${top_spender[1]:.2f})")
		sections.append("## Meta\n" + " | ".join(meta_parts))

		return "\n\n".join(sections)

	@staticmethod
	def _count_task_statuses(tasks: list[SwarmTask]) -> dict[str, int]:
		"""Count tasks by status category."""
		counts = {"completed": 0, "in_progress": 0, "pending": 0, "blocked": 0, "failed": 0}
		for t in tasks:
			if t.status == TaskStatus.COMPLETED:
				counts["completed"] += 1
			elif t.status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS):
				counts["in_progress"] += 1
			elif t.status == TaskStatus.BLOCKED:
				counts["blocked"] += 1
			elif t.status == TaskStatus.FAILED:
				counts["failed"] += 1
			elif t.status == TaskStatus.PENDING:
				# Check if effectively blocked by unmet dependencies
				if t.depends_on:
					counts["blocked"] += 1
				else:
					counts["pending"] += 1
		return counts

	@staticmethod
	def _format_elapsed(spawned_at: str, now: datetime) -> str:
		"""Format elapsed time since agent spawn as a human-readable string."""
		try:
			spawned = datetime.fromisoformat(spawned_at)
			delta = (now - spawned).total_seconds()
			if delta < 60:
				return f"{int(delta)}s"
			if delta < 3600:
				return f"{int(delta // 60)}m{int(delta % 60)}s"
			return f"{int(delta // 3600)}h{int((delta % 3600) // 60)}m"
		except (ValueError, TypeError):
			return "?"

	@staticmethod
	def _render_dependency_status(depends_on: list[str], task_by_id: dict[str, SwarmTask]) -> str:
		"""Render dependency info with resolution status for each dep."""
		if not depends_on:
			return ""
		parts = []
		for dep_id in depends_on:
			dep_task = task_by_id.get(dep_id)
			if dep_task:
				parts.append(f"{dep_task.title} [{dep_task.status.value}]")
			else:
				parts.append(f"{dep_id} [unknown]")
		return f" (waiting on: {', '.join(parts)})"

	@staticmethod
	def _has_unmet_deps(depends_on: list[str], task_by_id: dict[str, SwarmTask]) -> bool:
		"""Check if any dependency is not yet completed."""
		for dep_id in depends_on:
			dep_task = task_by_id.get(dep_id)
			if not dep_task or dep_task.status != TaskStatus.COMPLETED:
				return True
		return False

	@staticmethod
	def _resolve_claimer_name(claimed_by: str | None, agents: list[SwarmAgent]) -> str:
		"""Resolve agent ID to name for display."""
		if not claimed_by:
			return "?"
		for a in agents:
			if a.id == claimed_by:
				return a.name
		return claimed_by[:8]

	@staticmethod
	def _build_agent_task_map(
		agents: list[SwarmAgent],
		dead_agents: list[SwarmAgent],
		task_by_id: dict[str, SwarmTask],
	) -> dict[str, str]:
		"""Build a map of agent_id -> brief accomplishment summary from their tasks."""
		result: dict[str, str] = {}
		for a in list(agents) + list(dead_agents):
			if not a.current_task_id:
				continue
			task = task_by_id.get(a.current_task_id)
			if not task:
				continue
			summary = task.result_summary[:100] if task.result_summary else ""
			if task.status == TaskStatus.COMPLETED:
				result[a.id] = f"completed \"{task.title}\": {summary}" if summary else f"completed \"{task.title}\""
			elif task.status == TaskStatus.FAILED:
				result[a.id] = f"failed \"{task.title}\": {summary}" if summary else f"failed \"{task.title}\""
		return result

	@staticmethod
	def _group_discoveries(discoveries: list[str]) -> dict[str, list[str]]:
		"""Group discoveries by source (extracted from [source] prefix)."""
		from collections import OrderedDict
		grouped: OrderedDict[str, list[str]] = OrderedDict()
		for d in discoveries[-20:]:
			if d.startswith("[") and "] " in d:
				bracket_end = d.index("] ")
				source = d[1:bracket_end]
				text = d[bracket_end + 2:]
				# Strip message type prefix like "(discovery) "
				if text.startswith("(") and ") " in text:
					paren_end = text.index(") ")
					text = text[paren_end + 2:]
			else:
				source = "general"
				text = d
			grouped.setdefault(source, []).append(text)
		return grouped

	@staticmethod
	def _build_completed_work_summary(
		completed_agents: list[SwarmAgent],
		agent_task_map: dict[str, str],
		discoveries: list[str],
	) -> list[str]:
		"""Build a summary of what each successfully-completed dead agent accomplished."""
		lines: list[str] = []
		for a in completed_agents[-10:]:
			task_summary = agent_task_map.get(a.id, "")
			agent_discoveries = [
				d for d in discoveries
				if f"[{a.name}]" in d
			]
			parts = [f"**{a.name}** ({a.tasks_completed} task(s) completed)"]
			if task_summary:
				parts.append(f"  Result: {task_summary}")
			for d in agent_discoveries[:3]:
				# Strip the [sender] prefix for cleaner display
				text = d.split("] ", 1)[-1] if "] " in d else d
				parts.append(f"  Discovery: {text}")
			lines.append("\n".join(parts))
		return lines

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
				# Lazy rotation: trim oversized inboxes during reads
				rotate_inbox(inbox_file)
				try:
					messages = json.loads(inbox_file.read_text())
					for msg in messages[-20:]:
						msg_type = msg.get("type", "")
						text = msg.get("text", "")
						sender = msg.get("from", inbox_file.stem)
						if msg_type in ("discovery", "blocked", "question", "report", "directive"):
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

	def _get_human_directives(self) -> list[str]:
		"""Get human directives from the team-lead inbox."""
		directives: list[str] = []
		inbox_dir = Path.home() / ".claude" / "teams" / self._team_name / "inboxes"
		leader_inbox = inbox_dir / "team-lead.json"
		if leader_inbox.exists():
			try:
				messages = json.loads(leader_inbox.read_text())
				for msg in messages[-20:]:
					if msg.get("type") == "directive":
						ts = msg.get("timestamp", "")
						text = msg.get("text", "")
						directives.append(f"[{ts}] {text}")
			except (json.JSONDecodeError, OSError):
				pass
		return directives

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
		dead_agent_history: list[SwarmAgent] | None = None,
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

		# Track completion rate (suppress if agents recently died with completions)
		completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
		self._track_metric("completed_tasks", float(completed))
		stag = self._check_metric_stagnation(
			"completed_tasks",
			"Reduce parallelism. Focus agents on fewer, higher-impact tasks.",
		)
		if stag:
			# Don't signal stagnation if dead agents recently completed work --
			# the task pool status may lag behind actual completions
			recent_dead_completions = sum(
				a.tasks_completed for a in (dead_agent_history or [])
			)
			if recent_dead_completions == 0:
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
