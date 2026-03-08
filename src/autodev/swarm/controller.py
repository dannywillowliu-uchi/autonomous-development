"""Swarm controller -- translates planner decisions into Claude Code native operations.

Uses TeammateTool (teams, inboxes), Task system (shared pool, dependencies),
and Skills (SKILL.md creation) as the execution substrate. The controller
does NOT contain intelligence -- it's a thin execution layer. All intelligence
lives in the planner.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autodev.swarm.context import ContextSynthesizer
from autodev.swarm.models import (
	AgentRole,
	AgentStatus,
	DecisionType,
	PlannerDecision,
	SwarmAgent,
	SwarmState,
	SwarmTask,
	TaskPriority,
	TaskStatus,
	_new_id,
	_now_iso,
)

if TYPE_CHECKING:
	from autodev.config import MissionConfig, SwarmConfig
	from autodev.db import Database

logger = logging.getLogger(__name__)


class SwarmController:
	"""Manages swarm lifecycle and executes planner decisions.

	Responsibilities:
	- Spawn/kill agents via Claude Code subprocess
	- Manage the shared task pool
	- Read/write team inboxes
	- Track agent state and heartbeats
	- Execute planner decisions
	"""

	def __init__(
		self,
		config: MissionConfig,
		swarm_config: SwarmConfig,
		db: Database,
	) -> None:
		self._config = config
		self._swarm_config = swarm_config
		self._db = db
		self._team_name = f"autodev-{config.target.name}"
		self._agents: dict[str, SwarmAgent] = {}
		self._tasks: dict[str, SwarmTask] = {}
		self._processes: dict[str, asyncio.subprocess.Process] = {}
		self._context = ContextSynthesizer(config, db, self._team_name)
		self._start_time = time.monotonic()
		self._total_cost_usd = 0.0
		self._running = False

	@property
	def team_name(self) -> str:
		return self._team_name

	@property
	def agents(self) -> list[SwarmAgent]:
		return list(self._agents.values())

	@property
	def tasks(self) -> list[SwarmTask]:
		return list(self._tasks.values())

	@property
	def swarm_config(self) -> SwarmConfig:
		return self._swarm_config

	async def initialize(self) -> None:
		"""Set up team directory and initial state."""
		team_dir = Path.home() / ".claude" / "teams" / self._team_name
		team_dir.mkdir(parents=True, exist_ok=True)
		(team_dir / "inboxes").mkdir(exist_ok=True)

		config_path = team_dir / "config.json"
		if not config_path.exists():
			config_path.write_text(json.dumps({
				"team_name": self._team_name,
				"leader": "planner",
				"created_at": _now_iso(),
			}))

		# Initialize leader inbox
		leader_inbox = team_dir / "inboxes" / "team-lead.json"
		if not leader_inbox.exists():
			leader_inbox.write_text("[]")

		self._running = True
		logger.info("Swarm controller initialized for team %s", self._team_name)

	def build_state(self, core_test_results: dict[str, Any] | None = None) -> SwarmState:
		"""Build current swarm state snapshot for the planner."""
		wall_time = time.monotonic() - self._start_time
		return self._context.build_state(
			agents=self.agents,
			tasks=self.tasks,
			core_test_results=core_test_results,
			total_cost_usd=self._total_cost_usd,
			wall_time_seconds=wall_time,
		)

	def render_state(self, state: SwarmState) -> str:
		"""Render state as text for the planner prompt."""
		return self._context.render_for_planner(state)

	async def execute_decisions(self, decisions: list[PlannerDecision]) -> list[dict[str, Any]]:
		"""Execute a list of planner decisions. Returns results for each."""
		results = []
		for decision in sorted(decisions, key=lambda d: d.priority, reverse=True):
			try:
				result = await self._execute_one(decision)
				results.append({"decision": decision.type.value, "success": True, "result": result})
			except Exception as e:
				logger.error("Failed to execute decision %s: %s", decision.type, e)
				results.append({"decision": decision.type.value, "success": False, "error": str(e)})
		return results

	async def _execute_one(self, decision: PlannerDecision) -> dict[str, Any]:
		"""Execute a single planner decision."""
		handlers: dict[DecisionType, Any] = {
			DecisionType.SPAWN: self._handle_spawn,
			DecisionType.KILL: self._handle_kill,
			DecisionType.REDIRECT: self._handle_redirect,
			DecisionType.CREATE_TASK: self._handle_create_task,
			DecisionType.ADJUST: self._handle_adjust,
			DecisionType.WAIT: self._handle_wait,
			DecisionType.ESCALATE: self._handle_escalate,
			DecisionType.CREATE_SKILL: self._handle_create_skill,
		}
		handler = handlers.get(decision.type)
		if not handler:
			raise ValueError(f"Unknown decision type: {decision.type}")
		return await handler(decision.payload)

	# -- Decision Handlers --

	async def _handle_spawn(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Spawn a new agent in the swarm."""
		# Enforce max_agents
		active = [a for a in self._agents.values() if a.status in (AgentStatus.WORKING, AgentStatus.SPAWNING)]
		if self._swarm_config.max_agents > 0 and len(active) >= self._swarm_config.max_agents:
			return {"error": f"At max agents ({self._swarm_config.max_agents})", "spawned": False}

		role = AgentRole(payload.get("role", "general"))
		task_id = payload.get("task_id")
		name = payload.get("name", f"{role.value}-{_new_id()[:6]}")
		prompt = payload.get("prompt", "")

		agent = SwarmAgent(
			name=name,
			role=role,
			status=AgentStatus.SPAWNING,
			current_task_id=task_id,
		)
		self._agents[agent.id] = agent

		if task_id and task_id in self._tasks:
			self._tasks[task_id].status = TaskStatus.CLAIMED
			self._tasks[task_id].claimed_by = agent.id

		worker_prompt = self._build_worker_prompt(agent, prompt)
		proc = await self._spawn_claude_session(agent, worker_prompt)
		if proc:
			self._processes[agent.id] = proc
			agent.status = AgentStatus.WORKING
			logger.info("Spawned agent %s (%s) for task %s", agent.name, role.value, task_id)
		else:
			agent.status = AgentStatus.DEAD
			logger.error("Failed to spawn agent %s", agent.name)

		return {"agent_id": agent.id, "name": agent.name, "status": agent.status.value}

	async def _handle_kill(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Kill an agent gracefully. Refuses to kill agents younger than 5 minutes."""
		agent_id = payload["agent_id"]
		reason = payload.get("reason", "planner decision")
		force = payload.get("force", False)

		agent = self._agents.get(agent_id)
		if not agent:
			return {"error": f"Agent {agent_id} not found"}

		# Guard: don't kill agents that are still working and young
		if not force and agent.status == AgentStatus.WORKING and reason != "swarm shutdown":
			from datetime import datetime, timezone
			try:
				spawned = datetime.fromisoformat(agent.spawned_at)
				age_seconds = (datetime.now(timezone.utc) - spawned).total_seconds()
				if age_seconds < 300:  # 5 minutes minimum
					logger.info(
						"Refusing to kill agent %s (age %.0fs < 300s). Let it work.",
						agent.name, age_seconds,
					)
					return {"agent_id": agent_id, "killed": False, "reason": "too young"}
			except (ValueError, TypeError):
				pass

		self._write_to_inbox(agent.name, {
			"type": "shutdown_request",
			"reason": reason,
			"from": "planner",
		})

		proc = self._processes.get(agent_id)
		if proc and proc.returncode is None:
			proc.terminate()
			try:
				await asyncio.wait_for(proc.wait(), timeout=10)
			except asyncio.TimeoutError:
				proc.kill()

		agent.status = AgentStatus.DEAD
		if agent.current_task_id and agent.current_task_id in self._tasks:
			task = self._tasks[agent.current_task_id]
			if task.status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS):
				task.status = TaskStatus.PENDING
				task.claimed_by = None

		logger.info("Killed agent %s: %s", agent.name, reason)
		return {"agent_id": agent_id, "killed": True}

	async def _handle_redirect(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Redirect an agent to a different task (kill + respawn)."""
		agent_id = payload["agent_id"]
		old_agent = self._agents.get(agent_id, SwarmAgent())

		await self._handle_kill({"agent_id": agent_id, "reason": "redirected by planner"})
		return await self._handle_spawn({
			"role": old_agent.role.value,
			"task_id": payload.get("new_task_id"),
			"prompt": payload.get("prompt", ""),
			"name": payload.get("name", old_agent.name),
		})

	async def _handle_create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Add a task to the shared pool."""
		task = SwarmTask(
			title=payload["title"],
			description=payload.get("description", ""),
			priority=TaskPriority(payload.get("priority", 1)),
			depends_on=payload.get("depends_on", []),
			files_hint=payload.get("files_hint", []),
			max_attempts=payload.get("max_attempts", 3),
		)
		self._tasks[task.id] = task
		logger.info("Created task %s: %s", task.id, task.title)
		return {"task_id": task.id, "title": task.title}

	async def _handle_adjust(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Adjust runtime parameters."""
		adjusted = {}
		for key in ("max_agents", "planner_cooldown", "stagnation_threshold", "min_agents"):
			if key in payload:
				setattr(self._swarm_config, key, payload[key])
				adjusted[key] = payload[key]
		logger.info("Adjusted swarm config: %s", adjusted)
		return adjusted

	async def _handle_wait(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Planner explicitly waits."""
		duration = min(payload.get("duration", 10), 120)
		reason = payload.get("reason", "waiting for agents")
		logger.info("Planner waiting %ds: %s", duration, reason)
		await asyncio.sleep(duration)
		return {"waited": duration, "reason": reason}

	async def _handle_escalate(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Escalate to human via Telegram notification."""
		reason = payload.get("reason", "Planner needs human input")
		logger.warning("ESCALATION: %s", reason)
		try:
			from autodev.notifier import TelegramNotifier
			notifier = TelegramNotifier(self._config.notification)
			await notifier.send(f"[autodev swarm] ESCALATION: {reason}", level="error")
		except Exception:
			pass
		return {"escalated": True, "reason": reason}

	async def _handle_create_skill(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Create a Claude Code skill (SKILL.md) in the project."""
		name = payload["name"]
		description = payload.get("description", "")
		content = payload["content"]

		skills_dir = Path(self._config.target.resolved_path) / ".claude" / "skills" / name
		skills_dir.mkdir(parents=True, exist_ok=True)

		frontmatter_lines = [
			"---",
			f"name: {name}",
			f"description: {description}",
		]
		if payload.get("disable_model_invocation"):
			frontmatter_lines.append("disable-model-invocation: true")
		if payload.get("allowed_tools"):
			frontmatter_lines.append(f"allowed-tools: {payload['allowed_tools']}")
		frontmatter_lines.append("---")

		skill_text = "\n".join(frontmatter_lines) + "\n\n" + content
		(skills_dir / "SKILL.md").write_text(skill_text)

		for filename, file_content in payload.get("supporting_files", {}).items():
			(skills_dir / filename).write_text(file_content)

		logger.info("Created skill: %s at %s", name, skills_dir)
		return {"skill_name": name, "path": str(skills_dir)}

	# -- Worker Prompt --

	def _build_worker_prompt(self, agent: SwarmAgent, task_prompt: str) -> str:
		"""Build the full prompt for a worker agent with swarm context."""
		from autodev.swarm.worker_prompt import build_worker_prompt

		return build_worker_prompt(
			agent=agent,
			task_prompt=task_prompt,
			team_name=self._team_name,
			agents=list(self._agents.values()),
			tasks=list(self._tasks.values()),
			config=self._config,
			swarm_config=self._swarm_config,
		)

	# -- Inbox I/O --

	def _write_to_inbox(self, agent_name: str, message: dict[str, Any]) -> None:
		"""Write a message to an agent's inbox."""
		inbox_path = Path.home() / ".claude" / "teams" / self._team_name / "inboxes" / f"{agent_name}.json"
		try:
			inbox_path.parent.mkdir(parents=True, exist_ok=True)
			messages = json.loads(inbox_path.read_text()) if inbox_path.exists() else []
		except (json.JSONDecodeError, OSError):
			messages = []
		try:
			messages.append({**message, "timestamp": _now_iso()})
			inbox_path.write_text(json.dumps(messages, indent=2))
		except OSError as e:
			logger.warning("Could not write to inbox %s: %s", agent_name, e)

	def read_leader_inbox(self) -> list[dict[str, Any]]:
		"""Read messages sent to the team leader."""
		inbox_path = Path.home() / ".claude" / "teams" / self._team_name / "inboxes" / "team-lead.json"
		try:
			if inbox_path.exists():
				return json.loads(inbox_path.read_text())
		except (json.JSONDecodeError, OSError):
			pass
		return []

	# -- Subprocess --

	async def _spawn_claude_session(
		self, agent: SwarmAgent, prompt: str
	) -> asyncio.subprocess.Process | None:
		"""Spawn a Claude Code subprocess for an agent.

		Unlike batch-mode workers (which use --setting-sources project to
		isolate from global config), swarm agents inherit global MCP servers
		by default so they can use browser automation, nanobanana, obsidian,
		and other tools configured in ~/.claude.json.
		"""
		from autodev.config import build_claude_cmd, claude_subprocess_env

		# Do NOT pass setting_sources="project" -- swarm agents need
		# global MCP servers (browser-use, nanobanana, stitch, etc.)
		setting_sources = None if self._swarm_config.inherit_global_mcps else "project"
		cmd = build_claude_cmd(
			self._config,
			model=self._config.scheduler.model,
			prompt=prompt,
			setting_sources=setting_sources,
			permission_mode="auto",
			max_turns=200,
		)
		env = claude_subprocess_env(self._config)
		env["AUTODEV_TEAM_NAME"] = self._team_name
		env["AUTODEV_AGENT_ID"] = agent.id
		env["AUTODEV_AGENT_NAME"] = agent.name
		env["AUTODEV_AGENT_ROLE"] = agent.role.value

		try:
			proc = await asyncio.create_subprocess_exec(
				*cmd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=str(self._config.target.resolved_path),
				env=env,
			)
			return proc
		except Exception as e:
			logger.error("Failed to spawn Claude session for %s: %s", agent.name, e)
			return None

	# -- Agent Monitoring --

	async def monitor_agents(self) -> list[dict[str, Any]]:
		"""Check all active agents, collect completions and failures."""
		events: list[dict[str, Any]] = []

		for agent_id, proc in list(self._processes.items()):
			agent = self._agents.get(agent_id)
			if not agent:
				continue

			if proc.returncode is not None:
				stdout_bytes = await proc.stdout.read() if proc.stdout else b""
				output = stdout_bytes.decode(errors="replace")

				result = self._parse_ad_result(output)
				status = result.get("status", "failed") if result else "failed"

				agent.status = AgentStatus.DEAD
				if status == "completed":
					agent.tasks_completed += 1
				else:
					agent.tasks_failed += 1

				if agent.current_task_id and agent.current_task_id in self._tasks:
					task = self._tasks[agent.current_task_id]
					task.status = TaskStatus.COMPLETED if status == "completed" else TaskStatus.FAILED
					task.result_summary = result.get("summary", "") if result else output[-500:]
					task.completed_at = _now_iso()
					task.attempt_count += 1

				events.append({
					"type": "agent_completed",
					"agent_id": agent_id,
					"agent_name": agent.name,
					"status": status,
					"result": result,
				})
				del self._processes[agent_id]

		return events

	def _parse_ad_result(self, output: str) -> dict[str, Any] | None:
		"""Parse AD_RESULT JSON from worker stdout."""
		marker = "AD_RESULT:"
		idx = output.rfind(marker)
		if idx == -1:
			return None
		json_str = output[idx + len(marker):].strip()
		try:
			brace_count = 0
			end = 0
			for i, c in enumerate(json_str):
				if c == "{":
					brace_count += 1
				elif c == "}":
					brace_count -= 1
					if brace_count == 0:
						end = i + 1
						break
			if end > 0:
				return json.loads(json_str[:end])
		except json.JSONDecodeError:
			pass
		return None

	def requeue_failed_tasks(self) -> list[str]:
		"""Re-queue failed tasks that haven't exhausted their retry budget."""
		requeued = []
		for task in self._tasks.values():
			if task.status == TaskStatus.FAILED and task.attempt_count < task.max_attempts:
				task.status = TaskStatus.PENDING
				task.claimed_by = None
				requeued.append(task.id)
				logger.info(
					"Re-queued task %s (%s), attempt %d/%d",
					task.id, task.title, task.attempt_count, task.max_attempts,
				)
		return requeued

	def get_idle_agents(self, idle_seconds: float = 120.0) -> list[SwarmAgent]:
		"""Get agents that have been idle (no task) for too long."""
		idle = []
		for agent in self._agents.values():
			if agent.status == AgentStatus.IDLE and not agent.current_task_id:
				idle.append(agent)
		return idle

	def get_scaling_recommendation(self) -> dict[str, int]:
		"""Get scaling recommendation based on current state."""
		active = [a for a in self._agents.values() if a.status in (AgentStatus.WORKING, AgentStatus.SPAWNING)]
		pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
		idle = [a for a in self._agents.values() if a.status == AgentStatus.IDLE]

		rec: dict[str, int] = {"scale_up": 0, "scale_down": 0}
		if len(pending) > 2 * max(len(active), 1):
			rec["scale_up"] = min(len(pending) - len(active), 3)
		if len(idle) >= 2:
			rec["scale_down"] = len(idle)
		return rec

	async def cleanup(self) -> None:
		"""Shut down all agents and clean up."""
		self._running = False
		for agent_id in list(self._processes.keys()):
			await self._handle_kill({"agent_id": agent_id, "reason": "swarm shutdown"})
		logger.info("Swarm controller cleaned up")
