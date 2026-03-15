"""Swarm controller -- translates planner decisions into Claude Code native operations.

Uses TeammateTool (teams, inboxes), Task system (shared pool, dependencies),
and Skills (SKILL.md creation) as the execution substrate. The controller
does NOT contain intelligence -- it's a thin execution layer. All intelligence
lives in the planner.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import tempfile
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
		task_claim_timeout: float = 1800.0,
		stalled_task_timeout: float = 600.0,
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
		self._agent_costs: dict[str, float] = {}
		self._recent_changes: dict[str, list[str]] = {}
		self._start_commit: str | None = None
		self._running = False
		self._dead_agent_history: list[SwarmAgent] = []
		self._max_dead_history = 50
		self._task_claim_timeout = task_claim_timeout
		self._stalled_task_timeout = stalled_task_timeout
		# Trace system
		self._run_id = _now_iso().replace(":", "-").replace("T", "_")[:19]
		self._trace_dir = Path(config.target.resolved_path) / ".autodev-traces" / self._run_id
		self._trace_tasks: dict[str, asyncio.Task] = {}  # agent_id -> streaming task
		self._agent_outputs: dict[str, str] = {}  # agent_id -> accumulated output
		self._agent_spawn_times: dict[str, str] = {}  # agent_id -> ISO timestamp

	@property
	def team_name(self) -> str:
		return self._team_name

	@property
	def dead_agent_history(self) -> list[SwarmAgent]:
		return list(self._dead_agent_history)

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

		from autodev.swarm.capabilities import scan_capabilities
		self._capabilities = scan_capabilities(Path(self._config.target.resolved_path))

		# Capture starting commit for completion report diff
		try:
			proc = await asyncio.create_subprocess_exec(
				"git", "rev-parse", "HEAD",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=str(self._config.target.resolved_path),
			)
			stdout, _ = await proc.communicate()
			if proc.returncode == 0:
				self._start_commit = stdout.decode().strip()
		except Exception:
			pass

		self._running = True
		logger.info("Swarm controller initialized for team %s", self._team_name)

	def build_state(self, core_test_results: dict[str, Any] | None = None) -> SwarmState:
		"""Build current swarm state snapshot for the planner."""
		wall_time = time.monotonic() - self._start_time
		capabilities = getattr(self, "_capabilities", None)
		return self._context.build_state(
			agents=self.agents,
			tasks=self.tasks,
			core_test_results=core_test_results,
			total_cost_usd=self._total_cost_usd,
			wall_time_seconds=wall_time,
			dead_agent_history=self.dead_agent_history,
			capabilities=capabilities,
			recent_file_changes=dict(self._recent_changes),
			agent_costs=dict(self._agent_costs),
		)

	def render_state(self, state: SwarmState) -> str:
		"""Render state as text for the planner prompt."""
		return self._context.render_for_planner(state)

	async def execute_decisions(self, decisions: list[PlannerDecision]) -> list[dict[str, Any]]:
		"""Execute a list of planner decisions. Returns results for each.

		Tracks tasks created during this batch so that spawn decisions
		without a valid task_id can be auto-linked to the correct task.
		"""
		results = []
		# Track task IDs created in this batch (in execution order) so
		# spawns can reference tasks that didn't exist when the planner
		# generated its response.
		batch_created_task_ids: list[str] = []

		for decision in sorted(decisions, key=lambda d: d.priority, reverse=True):
			try:
				# Auto-resolve spawn task_id from batch-created tasks
				if decision.type == DecisionType.SPAWN:
					self._resolve_spawn_task_id(decision.payload, batch_created_task_ids)

				result = await self._execute_one(decision)
				results.append({"decision": decision.type.value, "success": True, "result": result})

				# Track tasks created in this batch
				if decision.type == DecisionType.CREATE_TASK and results[-1]["success"]:
					task_id = result.get("task_id")
					if task_id:
						batch_created_task_ids.append(task_id)
			except Exception as e:
				logger.error("Failed to execute decision %s: %s", decision.type, e)
				results.append({"decision": decision.type.value, "success": False, "error": str(e)})
		return results

	def _resolve_spawn_task_id(
		self, payload: dict[str, Any], batch_created_task_ids: list[str]
	) -> None:
		"""Auto-resolve a spawn's task_id when it doesn't match any task.

		When the planner creates tasks and spawns agents in the same batch,
		the auto-generated task IDs aren't known to the planner. This method
		finds the right task by checking batch-created tasks first, then
		falling back to unclaimed PENDING tasks in the pool.
		"""
		task_id = payload.get("task_id")
		if task_id and task_id in self._tasks:
			return  # Already valid

		# Try unclaimed batch-created tasks (in order)
		for tid in batch_created_task_ids:
			task = self._tasks.get(tid)
			if task and task.status == TaskStatus.PENDING and not task.claimed_by:
				if task_id:
					logger.info(
						"Resolved spawn task_id %r -> %s (%s)",
						task_id, tid, task.title,
					)
				payload["task_id"] = tid
				return

		# Fallback: find any unclaimed PENDING task in the pool.
		# This triggers when task_id is empty OR when it's an invalid ID
		# that doesn't match any real task (common when planners reference
		# auto-generated IDs they don't know).
		if not task_id or task_id not in self._tasks:
			for task in self._tasks.values():
				if task.status == TaskStatus.PENDING and not task.claimed_by:
					logger.info(
						"Auto-assigned spawn %r to unclaimed task %s (%s)",
						payload.get("name", "?"), task.id, task.title,
					)
					payload["task_id"] = task.id
					return

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
			DecisionType.CREATE_HOOK: self._handle_create_hook,
			DecisionType.REGISTER_MCP: self._handle_register_mcp,
			DecisionType.CREATE_AGENT_DEF: self._handle_create_agent_def,
			DecisionType.USE_SKILL: self._handle_use_skill,
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
			self._tasks[task_id].claimed_at = _now_iso()
		elif task_id:
			logger.warning(
				"Agent %s spawned with invalid task_id %r (not in task pool). "
				"Task will NOT be linked -- agent work won't update any task status.",
				name, task_id,
			)
			agent.current_task_id = None

		# Stale-task detection: warn if files were recently modified by another agent
		if task_id and task_id in self._tasks:
			task = self._tasks[task_id]
			if task.files_hint and self._recent_changes:
				stale_notes = []
				for hint_file in task.files_hint:
					for prev_agent, changed_files in self._recent_changes.items():
						if hint_file in changed_files and prev_agent != name:
							stale_notes.append(
								f"Note: {hint_file} was recently modified by {prev_agent}. "
								f"Check if the issue is already resolved before making changes."
							)
				if stale_notes:
					prompt = "\n".join(stale_notes) + "\n\n" + prompt

		worker_prompt = self._build_worker_prompt(agent, prompt)
		proc = await self._spawn_claude_session(agent, worker_prompt)
		if proc:
			self._processes[agent.id] = proc
			agent.status = AgentStatus.WORKING
			logger.info("Spawned agent %s (%s) for task %s", agent.name, role.value, task_id)
		else:
			agent.status = AgentStatus.DEAD
			agent.death_time = time.monotonic()
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
		agent.death_time = time.monotonic()
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
			from autodev.notifier import NotificationPriority, TelegramNotifier
			tg = self._config.notifications.telegram
			if tg.bot_token and tg.chat_id:
				notifier = TelegramNotifier(tg.bot_token, tg.chat_id)
				await notifier.send(
					f"[autodev swarm] ESCALATION: {reason}",
					priority=NotificationPriority.HIGH,
				)
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

	async def _handle_create_hook(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Install a hook into the project's .claude/settings.json."""
		settings_path = Path(self._config.target.resolved_path) / ".claude" / "settings.json"
		settings_path.parent.mkdir(parents=True, exist_ok=True)
		settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
		hooks = settings.setdefault("hooks", {})
		event_hooks = hooks.setdefault(payload["event"], [])
		hook_entry: dict[str, Any] = {"matcher": payload.get("matcher", ""), "type": payload["type"]}
		if payload["type"] == "command":
			hook_entry["command"] = payload["command"]
		else:
			hook_entry["prompt"] = payload.get("prompt", "")
		if payload.get("background"):
			hook_entry["background"] = True
		event_hooks.append(hook_entry)
		settings_path.write_text(json.dumps(settings, indent=2))
		return {"event": payload["event"], "hook_added": True, "path": str(settings_path)}

	async def _handle_register_mcp(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Add an MCP server to .mcp.json."""
		if payload.get("scope") == "global":
			mcp_path = Path.home() / ".claude.json"
		else:
			mcp_path = Path(self._config.target.resolved_path) / ".mcp.json"
		config = json.loads(mcp_path.read_text()) if mcp_path.exists() else {}
		servers = config.setdefault("mcpServers", {})
		entry: dict[str, Any] = {"type": payload.get("type", "stdio")}
		if entry["type"] == "stdio":
			entry["command"] = payload["command"]
			if payload.get("args"):
				entry["args"] = payload["args"]
		else:
			entry["url"] = payload["url"]
		if payload.get("env"):
			entry["env"] = payload["env"]
		servers[payload["name"]] = entry
		mcp_path.write_text(json.dumps(config, indent=2))
		return {"name": payload["name"], "registered": True, "path": str(mcp_path)}

	async def _handle_create_agent_def(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Write a .claude/agents/<name>.md file."""
		agents_dir = Path(self._config.target.resolved_path) / ".claude" / "agents"
		agents_dir.mkdir(parents=True, exist_ok=True)
		lines = ["---"]
		lines.append(f"name: {payload['name']}")
		if payload.get("description"):
			lines.append(f"description: {payload['description']}")
		if payload.get("tools"):
			lines.append(f"allowed-tools: {', '.join(payload['tools'])}")
		if payload.get("disallowed_tools"):
			lines.append(f"disallowed-tools: {', '.join(payload['disallowed_tools'])}")
		if payload.get("model"):
			lines.append(f"model: {payload['model']}")
		lines.append("---")
		lines.append("")
		lines.append(payload.get("system_prompt", ""))
		agent_path = agents_dir / f"{payload['name']}.md"
		agent_path.write_text("\n".join(lines))
		return {"name": payload["name"], "created": True, "path": str(agent_path)}

	async def _handle_use_skill(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Instruct an active agent to invoke a skill via inbox message."""
		agent_name = payload["agent_name"]
		skill_name = payload["skill_name"]
		args = payload.get("args", "")
		self._write_to_inbox(agent_name, {
			"type": "directive",
			"from": "planner",
			"text": f"Invoke /{skill_name} {args}".strip(),
		})
		return {"agent_name": agent_name, "skill": skill_name, "directed": True}

	# -- Auth Protocol --

	async def handle_auth_request(
		self,
		service: str,
		url: str,
		purpose: str,
		requesting_agent: str,
		signup_ok: bool = False,
	) -> dict[str, Any]:
		"""Process an auth_request message from a worker's inbox.

		Delegates to AuthGateway, then writes an auth_response back to the
		requesting worker's inbox so it can continue.
		"""
		logger.info(
			"Auth request from %s: service=%s url=%s purpose=%s",
			requesting_agent, service, url, purpose,
		)

		# Notify via Telegram
		try:
			from autodev.notifier import TelegramNotifier
			tg = self._config.notifications.telegram
			if tg.bot_token and tg.chat_id:
				notifier = TelegramNotifier(tg.bot_token, tg.chat_id)
				await notifier.send_auth_request(service, purpose, url)
		except Exception:
			pass

		# Delegate to AuthGateway
		result: dict[str, Any] = {"service": service, "success": False}
		try:
			from autodev.auth.vault import KeychainVault
			from autodev.auth.gateway import AuthGateway
			from autodev.auth.browser import HeadlessAuthHandler

			vault = KeychainVault()
			notifier_inst = None
			try:
				from autodev.notifier import TelegramNotifier as TN
				tg = self._config.notifications.telegram
				if tg.bot_token and tg.chat_id:
					notifier_inst = TN(tg.bot_token, tg.chat_id)
			except Exception:
				pass

			browser = HeadlessAuthHandler(vault, notifier_inst)
			gateway = AuthGateway(vault, browser, notifier_inst)
			auth_result = await gateway.authenticate(
				service=service,
				purpose=purpose,
				url=url,
				signup_ok=signup_ok,
			)
			result = {
				"service": service,
				"success": auth_result.success,
				"credential_type": auth_result.credential_type,
				"error": auth_result.error,
			}
			await browser.close()
		except ImportError:
			logger.warning("Auth gateway modules not available, returning failure")
			result = {
				"service": service,
				"success": False,
				"error": "Auth gateway not installed",
			}
		except Exception as e:
			logger.error("Auth gateway error for %s: %s", service, e)
			result = {
				"service": service,
				"success": False,
				"error": str(e),
			}

		# Write auth_response to the requesting worker's inbox
		instructions = ""
		if result["success"]:
			cred_type = result.get("credential_type", "")
			instructions = (
				f"Token stored in Keychain as autodev/{service}. "
				f"Credential type: {cred_type}."
			)
		else:
			instructions = f"Auth failed: {result.get('error', 'unknown error')}"

		self._write_to_inbox(requesting_agent, {
			"type": "auth_response",
			"from": "planner",
			"service": service,
			"success": result["success"],
			"credential_type": result.get("credential_type", ""),
			"instructions": instructions,
			"text": f"Auth response for {service}: {'success' if result['success'] else 'failed'}",
		})

		return result

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
		"""Write a message to an agent's inbox using atomic file operations."""
		inbox_path = Path.home() / ".claude" / "teams" / self._team_name / "inboxes" / f"{agent_name}.json"
		inbox_path.parent.mkdir(parents=True, exist_ok=True)
		lock_path = inbox_path.with_suffix(".lock")
		tmp_fd = None
		tmp_path = None
		try:
			with open(lock_path, "w") as lock_file:
				fcntl.flock(lock_file, fcntl.LOCK_EX)
				try:
					messages = json.loads(inbox_path.read_text()) if inbox_path.exists() else []
					if not isinstance(messages, list):
						messages = []
				except (json.JSONDecodeError, OSError):
					messages = []
				messages.append({**message, "timestamp": _now_iso()})
				tmp_fd, tmp_path = tempfile.mkstemp(
					dir=str(inbox_path.parent), suffix=".tmp"
				)
				os.write(tmp_fd, json.dumps(messages, indent=2).encode())
				os.close(tmp_fd)
				tmp_fd = None
				os.rename(tmp_path, str(inbox_path))
				tmp_path = None
		except OSError as e:
			logger.warning("Could not write to inbox %s: %s", agent_name, e)
		finally:
			if tmp_fd is not None:
				os.close(tmp_fd)
			if tmp_path is not None:
				try:
					os.unlink(tmp_path)
				except OSError:
					pass

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

		Output is streamed to a trace file at .autodev-traces/{run_id}/{agent}.log.
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
			# Start streaming output to trace file
			self._agent_spawn_times[agent.id] = _now_iso()
			self._agent_outputs[agent.id] = ""
			trace_task = asyncio.create_task(
				self._stream_agent_output(agent.id, agent.name, proc)
			)
			self._trace_tasks[agent.id] = trace_task
			return proc
		except Exception as e:
			logger.error("Failed to spawn Claude session for %s: %s", agent.name, e)
			return None

	async def _stream_agent_output(
		self, agent_id: str, agent_name: str, proc: asyncio.subprocess.Process
	) -> None:
		"""Stream agent stdout/stderr to a trace file, avoiding pipe deadlocks."""
		self._trace_dir.mkdir(parents=True, exist_ok=True)
		trace_path = self._trace_dir / f"{agent_name}-{agent_id[:8]}.log"
		try:
			with open(trace_path, "w") as f:
				f.write(f"# Agent: {agent_name} ({agent_id})\n")
				f.write(f"# Started: {self._agent_spawn_times.get(agent_id, '')}\n\n")

				async def read_stream(stream: asyncio.StreamReader, prefix: str) -> None:
					while True:
						line = await stream.readline()
						if not line:
							break
						decoded = line.decode(errors="replace")
						f.write(f"{prefix}{decoded}")
						f.flush()
						if prefix == "[OUT] ":
							self._agent_outputs[agent_id] = (
								self._agent_outputs.get(agent_id, "") + decoded
							)

				tasks = []
				if proc.stdout:
					tasks.append(asyncio.create_task(read_stream(proc.stdout, "[OUT] ")))
				if proc.stderr:
					tasks.append(asyncio.create_task(read_stream(proc.stderr, "[ERR] ")))
				if tasks:
					await asyncio.gather(*tasks)
		except Exception as e:
			logger.warning("Trace streaming error for %s: %s", agent_name, e)

	# -- Agent Monitoring --

	async def monitor_agents(self) -> list[dict[str, Any]]:
		"""Check all active agents, collect completions and failures."""
		events: list[dict[str, Any]] = []

		for agent_id, proc in list(self._processes.items()):
			agent = self._agents.get(agent_id)
			if not agent:
				continue

			if proc.returncode is not None:
				# Wait for trace streaming to finish
				trace_task = self._trace_tasks.pop(agent_id, None)
				if trace_task and not trace_task.done():
					try:
						await asyncio.wait_for(trace_task, timeout=5.0)
					except (asyncio.TimeoutError, Exception):
						pass

				# Use output accumulated by the streaming task, fall back to pipe read
				output = self._agent_outputs.pop(agent_id, "")
				if not output:
					try:
						stdout_bytes = await proc.stdout.read() if proc.stdout else b""
						output = stdout_bytes.decode(errors="replace")
					except Exception:
						output = ""

				result = self._parse_ad_result(output)
				if result:
					status = result.get("status", "failed")
				else:
					status = "failed"
					result = {
						"status": "failed",
						"summary": f"Process exited (code {proc.returncode}) without emitting AD_RESULT",
						"commits": [],
						"files_changed": [],
					}

				# Parse cost from agent output
				agent_cost = self._parse_agent_cost(output)
				if agent_cost > 0:
					self._total_cost_usd += agent_cost
					self._agent_costs[agent.name] = self._agent_costs.get(agent.name, 0.0) + agent_cost

				agent.status = AgentStatus.DEAD
				agent.death_time = time.monotonic()
				if status == "completed":
					agent.tasks_completed += 1
				else:
					agent.tasks_failed += 1

				task = self._resolve_agent_task(agent)
				if task:
					task.status = TaskStatus.COMPLETED if status == "completed" else TaskStatus.FAILED
					task.result_summary = result.get("summary", "")
					task.completed_at = _now_iso()
					task.attempt_count += 1
					if task.status == TaskStatus.FAILED:
						task.claimed_by = None

				# Collect diff info for planner visibility
				files_changed = await self._get_git_changed_files()
				if files_changed:
					self._recent_changes[agent.name] = files_changed

				# Commit-per-task on success
				commit_hash = None
				if status == "completed" and task:
					commit_hash = await self._auto_commit_task(task.title, agent.name)

				# Save trace to DB
				trace_path = self._trace_dir / f"{agent.name}-{agent_id[:8]}.log"
				self._save_agent_trace(
					agent=agent,
					task=task,
					exit_code=proc.returncode,
					cost_usd=agent_cost,
					files_changed=files_changed or [],
					trace_path=str(trace_path) if trace_path.exists() else "",
					output=output,
				)

				event: dict[str, Any] = {
					"type": "agent_completed",
					"agent_id": agent_id,
					"agent_name": agent.name,
					"status": status,
					"result": result,
					"cost_usd": agent_cost,
					"files_changed": files_changed,
				}
				if commit_hash:
					event["commit_hash"] = commit_hash
				events.append(event)
				del self._processes[agent_id]

		self._recover_orphaned_tasks(events)
		self._check_claim_timeouts(events)
		self._cleanup_dead_agents()
		return events

	def _save_agent_trace(
		self,
		agent: SwarmAgent,
		task: SwarmTask | None,
		exit_code: int | None,
		cost_usd: float,
		files_changed: list[str],
		trace_path: str,
		output: str,
	) -> None:
		"""Save agent trace metadata to DB."""
		try:
			spawn_time = self._agent_spawn_times.pop(agent.id, "")
			ended_at = _now_iso()
			duration_s = 0.0
			if spawn_time:
				from datetime import datetime
				try:
					start = datetime.fromisoformat(spawn_time)
					end = datetime.fromisoformat(ended_at)
					duration_s = (end - start).total_seconds()
				except (ValueError, TypeError):
					pass
			self._db.save_agent_trace(
				run_id=self._run_id,
				agent_name=agent.name,
				agent_id=agent.id,
				task_id=task.id if task else "",
				task_title=task.title if task else "",
				started_at=spawn_time,
				ended_at=ended_at,
				duration_s=duration_s,
				exit_code=exit_code,
				cost_usd=cost_usd,
				files_changed=files_changed,
				trace_path=trace_path,
				output_tail=output[-2000:] if output else "",
			)
		except Exception as e:
			logger.warning("Failed to save agent trace for %s: %s", agent.name, e)

	def _resolve_agent_task(self, agent: SwarmAgent) -> SwarmTask | None:
		"""Find the task associated with an agent.

		Primary: use agent.current_task_id for direct lookup.
		Fallback: search for a task with claimed_by == agent.id.

		This fallback handles cases where:
		- current_task_id was set to a placeholder that didn't match
		- current_task_id is None but the task was claimed through another path
		"""
		if agent.current_task_id and agent.current_task_id in self._tasks:
			return self._tasks[agent.current_task_id]

		# Fallback: search by claimed_by
		for task in self._tasks.values():
			if task.claimed_by == agent.id and task.status in (
				TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS
			):
				logger.info(
					"Resolved task for agent %s via claimed_by fallback: %s (%s)",
					agent.name, task.id, task.title,
				)
				return task

		if agent.current_task_id:
			logger.warning(
				"Agent %s has current_task_id %s but no matching task found",
				agent.name, agent.current_task_id,
			)
		return None

	def _recover_orphaned_tasks(self, events: list[dict[str, Any]]) -> None:
		"""Reset CLAIMED tasks to PENDING when their agent is dead or missing.

		A task is considered orphaned when:
		- It's in CLAIMED or IN_PROGRESS status
		- Its claiming agent is dead, missing from the agent pool, or has no process
		- It has been claimed for longer than _stalled_task_timeout
		"""
		from datetime import datetime, timezone

		now = datetime.now(timezone.utc)
		for task in self._tasks.values():
			if task.status not in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS):
				continue
			if not task.claimed_by:
				# No agent claim -- reset immediately
				task.status = TaskStatus.PENDING
				continue
			if not task.claimed_at:
				continue

			agent = self._agents.get(task.claimed_by)
			agent_alive = agent is not None and agent.status not in (AgentStatus.DEAD, AgentStatus.SHUTTING_DOWN)
			if agent_alive:
				continue

			# Agent is dead or missing -- check if enough time has passed
			try:
				claimed = datetime.fromisoformat(task.claimed_at)
				elapsed = (now - claimed).total_seconds()
			except (ValueError, TypeError):
				continue

			if elapsed < self._stalled_task_timeout:
				continue

			old_status = task.status
			task.status = TaskStatus.PENDING
			task.claimed_by = None
			task.claimed_at = None

			events.append({
				"type": "orphaned_task_recovered",
				"task_id": task.id,
				"task_title": task.title,
				"agent_id": agent.id if agent else None,
				"elapsed_seconds": elapsed,
				"previous_status": old_status.value,
			})
			logger.warning(
				"Recovered orphaned task %s (%s): agent dead/missing after %.0fs",
				task.id, task.title, elapsed,
			)

	def _check_claim_timeouts(self, events: list[dict[str, Any]]) -> None:
		"""Mark tasks as FAILED if they've been CLAIMED beyond the timeout."""
		from datetime import datetime, timezone

		now = datetime.now(timezone.utc)
		for task in self._tasks.values():
			if task.status not in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS):
				continue
			if not task.claimed_at:
				continue
			try:
				claimed = datetime.fromisoformat(task.claimed_at)
				elapsed = (now - claimed).total_seconds()
			except (ValueError, TypeError):
				continue
			if elapsed <= self._task_claim_timeout:
				continue

			agent_id = task.claimed_by
			task.status = TaskStatus.FAILED
			task.result_summary = f"Claim timeout after {elapsed:.0f}s (limit {self._task_claim_timeout:.0f}s)"
			task.completed_at = _now_iso()
			task.attempt_count += 1
			task.claimed_by = None

			if agent_id and agent_id in self._agents:
				self._agents[agent_id].status = AgentStatus.DEAD
				self._agents[agent_id].tasks_failed += 1
			if agent_id and agent_id in self._processes:
				proc = self._processes[agent_id]
				if proc.returncode is None:
					proc.kill()

			events.append({
				"type": "claim_timeout",
				"task_id": task.id,
				"task_title": task.title,
				"agent_id": agent_id,
				"elapsed_seconds": elapsed,
			})
			logger.warning(
				"Task %s (%s) claim timed out after %.0fs",
				task.id, task.title, elapsed,
			)

	def _cleanup_dead_agents(self, dead_threshold_seconds: float = 300.0) -> None:
		"""Remove agents that have been DEAD for longer than the threshold.

		Moves cleaned-up agents to a bounded history list for context rendering.
		"""
		now = time.monotonic()
		to_remove: list[str] = []

		for agent_id, agent in self._agents.items():
			if agent.status != AgentStatus.DEAD:
				continue
			if agent.death_time is None:
				# Legacy agent without death_time -- set it now
				agent.death_time = now
				continue
			if now - agent.death_time >= dead_threshold_seconds:
				to_remove.append(agent_id)

		for agent_id in to_remove:
			agent = self._agents.pop(agent_id)
			self._processes.pop(agent_id, None)
			self._dead_agent_history.append(agent)
			logger.debug("Cleaned up dead agent %s (%s)", agent.name, agent_id)

		# Keep history bounded
		if len(self._dead_agent_history) > self._max_dead_history:
			self._dead_agent_history = self._dead_agent_history[-self._max_dead_history:]

	# 10 MB default cap on stdout read to prevent memory exhaustion
	MAX_OUTPUT_SIZE = 10 * 1024 * 1024

	def _parse_ad_result(self, output: str, *, max_output_size: int = 0) -> dict[str, Any] | None:
		"""Parse AD_RESULT JSON from worker stdout.

		Uses the *last* AD_RESULT marker (earlier ones may be from retries).
		Validates that the parsed object contains a 'status' field and defaults
		'summary' to "" when absent.
		"""
		limit = max_output_size or self.MAX_OUTPUT_SIZE
		if len(output) > limit:
			output = output[-limit:]

		marker = "AD_RESULT:"
		idx = output.rfind(marker)
		if idx == -1:
			return None

		json_str = output[idx + len(marker):].strip()

		# Find the end of the top-level JSON object via brace counting.
		# Characters inside strings (including escaped quotes) are tracked
		# so nested braces in string values don't confuse the counter.
		brace_count = 0
		end = 0
		in_string = False
		escape_next = False
		for i, c in enumerate(json_str):
			if escape_next:
				escape_next = False
				continue
			if c == "\\" and in_string:
				escape_next = True
				continue
			if c == '"' and not escape_next:
				in_string = not in_string
				continue
			if in_string:
				continue
			if c == "{":
				brace_count += 1
			elif c == "}":
				brace_count -= 1
				if brace_count == 0:
					end = i + 1
					break

		if end == 0:
			logger.warning("AD_RESULT marker found but no valid JSON object follows")
			return None

		try:
			data = json.loads(json_str[:end])
		except json.JSONDecodeError as exc:
			logger.warning("AD_RESULT JSON parse failed: %s", exc)
			return None

		if not isinstance(data, dict):
			logger.warning("AD_RESULT is not a JSON object")
			return None

		if "status" not in data:
			logger.warning("AD_RESULT missing required 'status' field")
			return None

		data.setdefault("summary", "")
		return data

	def _parse_agent_cost(self, output: str) -> float:
		"""Extract cost from Claude Code agent output.

		Claude Code prints lines like 'Total cost: $1.23' or 'Total cost:\t$0.50'.
		Returns 0.0 if not found.
		"""
		match = re.search(r"Total cost:[\s\t]*\$([0-9]+(?:\.[0-9]+)?)", output)
		if match:
			try:
				return float(match.group(1))
			except ValueError:
				return 0.0
		return 0.0

	async def _get_git_changed_files(self) -> list[str]:
		"""Run git diff --name-only to get list of changed files."""
		try:
			proc = await asyncio.create_subprocess_exec(
				"git", "diff", "--name-only", "HEAD~1",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=str(self._config.target.resolved_path),
			)
			stdout, _ = await proc.communicate()
			if proc.returncode == 0:
				return [f for f in stdout.decode().strip().split("\n") if f]
		except Exception:
			pass
		return []

	async def _auto_commit_task(self, task_title: str, agent_name: str) -> str | None:
		"""Auto-commit changes after successful task completion. Returns commit hash or None."""
		cwd = str(self._config.target.resolved_path)
		try:
			# Stage all changes
			proc = await asyncio.create_subprocess_exec(
				"git", "add", "-A",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=cwd,
			)
			await proc.communicate()

			# Check if there are staged changes
			proc = await asyncio.create_subprocess_exec(
				"git", "diff", "--cached", "--quiet",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=cwd,
			)
			await proc.communicate()
			if proc.returncode == 0:
				return None  # No staged changes

			# Commit
			msg = f"autodev: {task_title} (agent: {agent_name})"
			proc = await asyncio.create_subprocess_exec(
				"git", "commit", "-m", msg,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=cwd,
			)
			await proc.communicate()
			if proc.returncode != 0:
				return None

			# Get commit hash
			proc = await asyncio.create_subprocess_exec(
				"git", "rev-parse", "HEAD",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=cwd,
			)
			stdout, _ = await proc.communicate()
			if proc.returncode == 0:
				commit_hash = stdout.decode().strip()
				logger.info("Auto-committed task '%s': %s", task_title, commit_hash[:8])
				return commit_hash
		except Exception as e:
			logger.warning("Auto-commit failed for task '%s': %s", task_title, e)
		return None

	async def _generate_completion_report(self) -> str:
		"""Generate a swarm completion report."""
		duration = time.monotonic() - self._start_time
		completed = [t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED]
		failed = [t for t in self._tasks.values() if t.status == TaskStatus.FAILED]

		lines = [
			"# Swarm Completion Report",
			"",
			"## Summary",
			f"- Duration: {duration / 60:.1f} minutes",
			f"- Cycles: {self._context._cycle_number}",
			f"- Tasks completed: {len(completed)}",
			f"- Tasks failed: {len(failed)}",
			f"- Total cost: ${self._total_cost_usd:.2f}",
			"",
		]

		# Per-agent cost breakdown
		if self._agent_costs:
			lines.append("## Agent Cost Breakdown")
			for agent_name, cost in sorted(self._agent_costs.items(), key=lambda x: x[1], reverse=True):
				lines.append(f"- {agent_name}: ${cost:.2f}")
			lines.append("")

		# Completed tasks
		if completed:
			lines.append("## Completed Tasks")
			for t in completed:
				summary = t.result_summary[:120] if t.result_summary else "no summary"
				lines.append(f"- {t.title}: {summary}")
			lines.append("")

		# Failed tasks
		if failed:
			lines.append("## Failed Tasks")
			for t in failed:
				summary = t.result_summary[:120] if t.result_summary else "no details"
				lines.append(f"- {t.title} (attempts: {t.attempt_count}): {summary}")
			lines.append("")

		# Files changed (git diff --stat vs starting commit)
		if self._start_commit:
			try:
				proc = await asyncio.create_subprocess_exec(
					"git", "diff", "--stat", self._start_commit,
					stdout=asyncio.subprocess.PIPE,
					stderr=asyncio.subprocess.PIPE,
					cwd=str(self._config.target.resolved_path),
				)
				stdout, _ = await proc.communicate()
				if proc.returncode == 0 and stdout:
					lines.append("## Files Changed")
					lines.append("```")
					lines.append(stdout.decode().strip())
					lines.append("```")
					lines.append("")
			except Exception:
				pass

		report = "\n".join(lines)

		# Write to project root
		report_path = Path(self._config.target.resolved_path) / ".autodev-swarm-report.md"
		try:
			report_path.write_text(report)
			logger.info("Completion report written to %s", report_path)
		except OSError as e:
			logger.warning("Failed to write completion report: %s", e)

		return report

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
		await self._generate_completion_report()
		await self._run_trace_review()
		await self._record_metrics()
		logger.info("Swarm controller cleaned up")

	async def _record_metrics(self) -> None:
		"""Record swarm run metrics for trend analysis."""
		try:
			from autodev.metrics import MetricsTracker, SwarmMetrics

			project_path = Path(self._config.target.resolved_path)
			tracker = MetricsTracker(project_path)

			completed = [t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED]
			failed = [t for t in self._tasks.values() if t.status == TaskStatus.FAILED]
			total_tasks = len(completed) + len(failed)
			duration = time.monotonic() - self._start_time

			# Agent success rate: completed agents / total finished agents
			finished_agents = [
				a for a in list(self._agents.values()) + self._dead_agent_history
				if a.status in (AgentStatus.COMPLETED, AgentStatus.FAILED)
			]
			completed_agents = [a for a in finished_agents if a.status == AgentStatus.COMPLETED]
			agent_success_rate = len(completed_agents) / len(finished_agents) if finished_agents else 0.0

			metrics = SwarmMetrics(
				run_id=self._run_id,
				timestamp=_now_iso(),
				test_count=0,
				test_pass_rate=0.0,
				total_cost_usd=self._total_cost_usd,
				cost_per_task=self._total_cost_usd / total_tasks if total_tasks > 0 else 0.0,
				agent_success_rate=agent_success_rate,
				total_duration_s=duration,
				tasks_completed=len(completed),
				tasks_failed=len(failed),
			)
			tracker.record_run(metrics)
			logger.info("Recorded swarm metrics for run %s", self._run_id)
		except Exception:
			logger.warning("Failed to record metrics", exc_info=True)

	async def _run_trace_review(self) -> None:
		"""Run trace analysis and write review report after swarm completion."""
		try:
			from autodev.swarm.learnings import SwarmLearnings
			from autodev.trace_review import TraceAnalyzer

			project_path = Path(self._config.target.resolved_path)
			analyzer = TraceAnalyzer(self._db, project_path)
			analysis = await analyzer.analyze_run(self._run_id)

			if analysis.total_agents == 0:
				logger.info("No agent traces found for run %s, skipping trace review", self._run_id)
				return

			report = await analyzer.generate_report(analysis)

			# Write full report to .autodev-traces/{run_id}/REVIEW.md
			review_dir = self._trace_dir
			review_dir.mkdir(parents=True, exist_ok=True)
			review_path = review_dir / "REVIEW.md"
			review_path.write_text(report)
			logger.info("Trace review written to %s", review_path)

			# Append key recommendations to learnings file
			if analysis.recommendations:
				learnings = SwarmLearnings(project_path)
				for rec in analysis.recommendations:
					learnings.add_discovery(f"trace-review/{self._run_id}", rec)
				logger.info("Appended %d trace recommendations to learnings", len(analysis.recommendations))

		except Exception:
			logger.warning("Trace review failed (non-fatal)", exc_info=True)
