"""Driving planner -- the brain of the swarm.

Runs as a series of LLM calls in a Python async loop (not a persistent
session). Each cycle: observe state -> reason -> decide -> execute.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autodev.swarm.evaluator import CycleEvaluator
from autodev.swarm.models import (
	AgentStatus,
	DecisionType,
	PlannerDecision,
	SwarmState,
	TaskStatus,
)
from autodev.swarm.prompts import (
	ANALYSIS_PROMPT_TEMPLATE,
	CYCLE_PROMPT_TEMPLATE,
	DECISION_FROM_ANALYSIS_PROMPT,
	INITIAL_PLANNING_PROMPT,
	SYSTEM_PROMPT,
)
from autodev.swarm.stagnation import (
	analyze_stagnation,
	format_pivots_for_planner,
)

if TYPE_CHECKING:
	from autodev.config import SwarmConfig
	from autodev.swarm.controller import SwarmController

logger = logging.getLogger(__name__)


class DrivingPlanner:
	"""The planner loop that drives the swarm.

	Each cycle:
	1. Build state snapshot from controller
	2. Call LLM with state + system prompt
	3. Parse structured decisions from response
	4. Pass decisions to controller for execution
	5. Sleep until next trigger
	"""

	# After this many consecutive parse failures, escalate and stop
	MAX_CONSECUTIVE_PARSE_FAILURES = 5

	def __init__(
		self,
		controller: SwarmController,
		swarm_config: SwarmConfig,
	) -> None:
		self._controller = controller
		self._config = swarm_config
		self._running = False
		self._cycle_count = 0
		self._test_history: list[int] = []
		self._completion_history: list[int] = []
		self._failure_history: list[int] = []
		self._cost_history: list[float] = []
		self._log_events: list[dict[str, Any]] = []
		self._last_plan_time: float = 0
		self._consecutive_parse_failures = 0
		self._evaluator = CycleEvaluator()
		self._task_failure_counts: dict[str, list[str]] = {}
		self._daemon_idling = False
		self._last_directive_text: str | None = None

		from autodev.swarm.learnings import SwarmLearnings
		self._learnings = SwarmLearnings(
			Path(controller._config.target.resolved_path)
		)

	async def run(self, core_test_runner: Any = None) -> None:
		"""Main planner loop. Runs until stopped or escalated."""
		self._running = True
		logger.info("Driving planner started")

		try:
			# Initial planning cycle
			state = self._build_state(core_test_runner)
			decisions = await self._initial_plan(state)
			decisions = self._validate_decisions(decisions, state)
			await self._controller.execute_decisions(decisions)

			self._write_state_file(state)

			# Main loop
			while self._running:
				if self._daemon_idling:
					await asyncio.sleep(60)  # Check inbox every 60s when idle
				else:
					await asyncio.sleep(self._config.planner_cooldown)

				# Monitor agents, collect events
				events = await self._controller.monitor_agents()

				# Record learnings from completed/failed agents
				self._record_learnings(events)

				# Re-queue failed tasks with retry budget
				requeued = self._controller.requeue_failed_tasks()
				if requeued:
					events.append({
						"type": "tasks_requeued",
						"task_ids": requeued,
					})

				# Build fresh state
				state = self._build_state(core_test_runner)
				self._record_metrics(state)
				self._write_state_file(state)

				# Grade the cycle (evaluator runs every iteration, not just planning cycles)
				self._evaluator.grade_cycle(state)

				# Check for new directives in inbox (even when idling)
				if self._daemon_idling:
					has_directive = self._check_inbox_for_directives(state)
					if has_directive:
						self._daemon_idling = False
						logger.info("Daemon mode: new directive received, resuming planning")
						directive_preview = (self._last_directive_text or "")[:200]
						asyncio.create_task(self._controller._notify(
							f"[autodev] New directive received: {directive_preview}"
						))
						decisions = await self._plan_cycle(state)
						decisions = self._validate_decisions(decisions, state)
						results = await self._controller.execute_decisions(decisions)
						self._log_cycle(decisions, results)
				elif self._should_plan(state, events):
					decisions = await self._plan_cycle(state)
					decisions = self._validate_decisions(decisions, state)
					results = await self._controller.execute_decisions(decisions)
					self._log_cycle(decisions, results)

				# Check termination conditions
				if self._should_stop(state):
					logger.info("Planner stopping: termination condition met")
					break

		except Exception as e:
			logger.error("Planner loop error: %s", e, exc_info=True)
			raise
		finally:
			self._running = False
			await self._controller.cleanup()
			logger.info(
				"Driving planner stopped after %d cycles",
				self._cycle_count,
			)

	def stop(self) -> None:
		"""Signal the planner to stop."""
		self._running = False

	def _validate_decisions(
		self, decisions: list[PlannerDecision], state: SwarmState
	) -> list[PlannerDecision]:
		"""Rule-based validation gate for planner decisions.

		Filters out bad decisions before they reach execute_decisions().
		All checks are algorithmic -- no LLM calls.
		"""
		if not decisions:
			return decisions

		validated: list[PlannerDecision] = []
		rejected_reasons: list[str] = []

		# Pre-compute existing task titles for duplicate detection
		existing_titles = [
			t.title.lower().strip()
			for t in state.tasks
			if t.status in (TaskStatus.PENDING, TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS)
		]

		# Track files claimed by decisions already accepted this batch
		accepted_files: list[set[str]] = []

		# Track spawn count against budget
		active_agents = sum(
			1 for a in state.agents
			if a.status in (AgentStatus.WORKING, AgentStatus.SPAWNING)
		)
		max_agents = self._config.max_agents
		spawns_allowed = (max_agents - active_agents) if max_agents > 0 else float("inf")
		spawns_accepted = 0

		# Collect create_task decisions for circular dependency check
		new_task_decisions: list[tuple[int, PlannerDecision]] = []

		for i, decision in enumerate(decisions):
			reject_reason = self._check_decision(
				decision, state, existing_titles, accepted_files,
				spawns_allowed, spawns_accepted,
			)
			if reject_reason:
				rejected_reasons.append(
					f"Decision #{i} ({decision.type.value}): {reject_reason}"
				)
				continue

			# Track accepted state
			if decision.type == DecisionType.SPAWN:
				spawns_accepted += 1
			files = self._extract_files_hint(decision)
			if files:
				accepted_files.append(files)
			if decision.type == DecisionType.CREATE_TASK:
				title = decision.payload.get("title", "").lower().strip()
				if title:
					existing_titles.append(title)
				new_task_decisions.append((len(validated), decision))

			validated.append(decision)

		# Circular dependency check across new create_task decisions
		validated = self._filter_circular_deps(validated, new_task_decisions, rejected_reasons)

		for reason in rejected_reasons:
			logger.warning("Validation gate rejected: %s", reason)

		if rejected_reasons:
			logger.info(
				"Validation gate: %d/%d decisions passed, %d rejected",
				len(validated), len(decisions), len(rejected_reasons),
			)

		return validated

	def _check_decision(
		self,
		decision: PlannerDecision,
		state: SwarmState,
		existing_titles: list[str],
		accepted_files: list[set[str]],
		spawns_allowed: float,
		spawns_accepted: int,
	) -> str | None:
		"""Check a single decision against validation rules.

		Returns a rejection reason string, or None if the decision is valid.
		"""
		# EMPTY PROMPT: reject spawn with empty or whitespace-only prompt
		if decision.type == DecisionType.SPAWN:
			prompt = decision.payload.get("prompt", "")
			if not prompt or not prompt.strip():
				return "spawn with empty prompt"

		# SPAWN BUDGET: don't exceed max_agents
		if decision.type == DecisionType.SPAWN:
			if spawns_accepted >= spawns_allowed:
				return (
					f"spawn budget exceeded "
					f"(allowed={int(spawns_allowed)}, already accepted={spawns_accepted})"
				)

		# DUPLICATE TASKS: >80% title similarity with existing tasks
		if decision.type == DecisionType.CREATE_TASK:
			title = decision.payload.get("title", "").lower().strip()
			if title:
				for existing in existing_titles:
					ratio = SequenceMatcher(None, title, existing).ratio()
					if ratio > 0.80:
						return f"duplicate task ('{title}' ~{ratio:.0%} similar to '{existing}')"

		# FILE OVERLAP: flag spawn/create_task that overlap with already-accepted decisions
		if decision.type in (DecisionType.SPAWN, DecisionType.CREATE_TASK):
			files = self._extract_files_hint(decision)
			if files:
				for prev_files in accepted_files:
					overlap = files & prev_files
					if overlap:
						return f"file overlap with prior decision: {overlap}"

		return None

	@staticmethod
	def _extract_files_hint(decision: PlannerDecision) -> set[str]:
		"""Extract files_hint from a decision payload as a set of paths."""
		hint = decision.payload.get("files_hint", [])
		if isinstance(hint, str):
			return {f.strip() for f in hint.split(",") if f.strip()}
		if isinstance(hint, list):
			return {str(f).strip() for f in hint if str(f).strip()}
		return set()

	@staticmethod
	def _filter_circular_deps(
		validated: list[PlannerDecision],
		new_task_decisions: list[tuple[int, PlannerDecision]],
		rejected_reasons: list[str],
	) -> list[PlannerDecision]:
		"""Detect and remove create_task decisions that form dependency cycles."""
		if len(new_task_decisions) < 2:
			return validated

		# First pass: collect all task IDs
		task_ids: dict[str, int] = {}
		for idx, dec in new_task_decisions:
			tid = dec.payload.get("task_id", dec.payload.get("title", f"__idx_{idx}"))
			task_ids[tid] = idx

		new_ids = set(task_ids.keys())

		# Second pass: build dependency graph (only edges within the new batch)
		adjacency: dict[str, list[str]] = {tid: [] for tid in new_ids}
		in_degree: dict[str, int] = {tid: 0 for tid in new_ids}
		for idx, dec in new_task_decisions:
			tid = dec.payload.get("task_id", dec.payload.get("title", f"__idx_{idx}"))
			deps = dec.payload.get("depends_on", [])
			if isinstance(deps, str):
				deps = [d.strip() for d in deps.split(",") if d.strip()]
			for dep in deps:
				if dep in new_ids:
					adjacency[dep].append(tid)
					in_degree[tid] += 1

		# Kahn's algorithm to detect cycles
		queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
		visited = 0
		while queue:
			node = queue.popleft()
			visited += 1
			for neighbor in adjacency.get(node, []):
				in_degree[neighbor] -= 1
				if in_degree[neighbor] == 0:
					queue.append(neighbor)

		if visited == len(new_ids):
			return validated

		# Cycle detected -- remove all tasks involved in cycles
		cycle_ids = {tid for tid, deg in in_degree.items() if deg > 0}
		indices_to_remove: set[int] = set()
		for tid in cycle_ids:
			idx = task_ids[tid]
			indices_to_remove.add(idx)
			rejected_reasons.append(
				f"Decision (create_task '{tid}'): circular dependency detected"
			)

		if indices_to_remove:
			validated = [d for i, d in enumerate(validated) if i not in indices_to_remove]

		return validated

	def _build_state(self, core_test_runner: Any = None) -> SwarmState:
		"""Build current swarm state, optionally running core tests."""
		core_results = None
		if core_test_runner:
			try:
				core_results = core_test_runner()
			except Exception as e:
				logger.warning("Core test runner failed: %s", e)
		return self._controller.build_state(core_test_results=core_results)

	def _record_learnings(self, events: list[dict[str, Any]]) -> None:
		"""Extract learnings from agent completion/failure events."""
		for ev in events:
			if ev.get("type") != "agent_completed":
				continue
			agent_name = ev.get("agent_name", "?")
			status = ev.get("status", "failed")
			result = ev.get("result") or {}
			task_title = ev.get("task_title", agent_name)

			if status == "completed":
				summary = result.get("summary", "")
				if summary:
					self._learnings.add_successful_approach(agent_name, summary)
				for disc in result.get("discoveries", []):
					self._learnings.add_discovery(agent_name, disc)
			else:
				error = result.get("error", result.get("summary", "unknown error"))
				if error:
					self._learnings.add_failed_approach(
						agent_name, str(error), result.get("attempt", 1)
					)

				# Track failures per task for reflection
				failures = self._task_failure_counts.setdefault(task_title, [])
				failures.append(str(error) if error else "unknown error")
				if len(failures) >= 2:
					self._learnings.add_reflection(
						task_title,
						failures,
						f"Task '{task_title}' has failed {len(failures)} times. "
						"Consider: Was the task description clear enough? "
						"Was the right agent role assigned? Were dependencies satisfied?",
					)

	def _record_metrics(self, state: SwarmState) -> None:
		"""Record metrics for stagnation detection."""
		if state.core_test_results:
			self._test_history.append(
				state.core_test_results.get("pass", 0)
			)
		completed = sum(
			1 for t in state.tasks if t.status == TaskStatus.COMPLETED
		)
		failed = sum(
			1 for t in state.tasks if t.status == TaskStatus.FAILED
		)
		self._completion_history.append(completed)
		self._failure_history.append(failed)
		self._cost_history.append(state.total_cost_usd)

	def _check_inbox_for_directives(self, state: SwarmState) -> bool:
		"""Check if the team inbox has new directive messages."""
		try:
			inbox_path = (
				Path.home() / ".claude" / "teams"
				/ self._controller.team_name / "inboxes" / "team-lead.json"
			)
			if not inbox_path.exists():
				return False
			messages = json.loads(inbox_path.read_text())
			for msg in messages:
				if msg.get("type") == "directive":
					self._last_directive_text = msg.get("text", "")
					logger.info("Found directive: %s", self._last_directive_text[:100])
					return True
		except Exception:
			pass
		return False

	def _should_plan(
		self, state: SwarmState, events: list[dict[str, Any]]
	) -> bool:
		"""Decide whether to run a planning cycle."""
		# Enforce minimum interval between cycles, with exponential backoff
		# on consecutive parse failures to avoid burning credits
		elapsed = time.monotonic() - self._last_plan_time
		min_interval = max(60, self._config.planner_cooldown)
		if self._consecutive_parse_failures > 0:
			# Double the interval for each consecutive failure: 120s, 240s, 480s, ...
			backoff = min_interval * (2 ** self._consecutive_parse_failures)
			min_interval = min(backoff, 600)  # cap at 10 minutes
		if elapsed < min_interval:
			return False

		if events:
			return True

		active = [
			a for a in state.agents if a.status == AgentStatus.WORKING
		]
		if not active and state.tasks:
			return True

		pending = [
			t for t in state.tasks if t.status == TaskStatus.PENDING
		]
		if pending and (
			self._config.max_agents == 0
			or len(active) < self._config.max_agents
		):
			return True

		return False

	def _should_stop(self, state: SwarmState) -> bool:
		"""Check termination conditions."""
		# Stop if planner LLM is consistently producing unparseable output
		if self._consecutive_parse_failures >= self.MAX_CONSECUTIVE_PARSE_FAILURES:
			logger.error(
				"Stopping: %d consecutive planner parse failures. "
				"LLM output is likely being truncated.",
				self._consecutive_parse_failures,
			)
			return True

		active = [
			a for a in state.agents if a.status == AgentStatus.WORKING
		]
		pending = [
			t for t in state.tasks
			if t.status in (
				TaskStatus.PENDING,
				TaskStatus.CLAIMED,
				TaskStatus.IN_PROGRESS,
			)
		]
		if not active and not pending and state.tasks:
			if self._config.daemon_mode:
				if not self._daemon_idling:
					logger.info("Daemon mode: all tasks complete, idling for new directives")
					self._daemon_idling = True
					asyncio.create_task(self._controller._notify(
						"[autodev] Swarm idling -- waiting for new directives"
					))
				return False
			return True
		if self._daemon_idling:
			self._daemon_idling = False
			logger.info("Daemon mode: new work detected, resuming")
		return False

	async def _initial_plan(
		self, state: SwarmState
	) -> list[PlannerDecision]:
		"""Generate the initial task decomposition and agent spawns."""
		self._last_plan_time = time.monotonic()
		self._cycle_count += 1
		state_text = self._controller.render_state(state)

		max_hint = self._config.max_agents if self._config.max_agents > 0 else 8

		# Add accumulated learnings from previous runs
		learnings_text = self._learnings.get_for_planner()
		if learnings_text:
			state_text += "\n\n" + learnings_text

		prompt = INITIAL_PLANNING_PROMPT.format(
			objective=state.mission_objective,
			state_text=state_text,
			min_agents=self._config.min_agents,
			max_agents_hint=max_hint,
		)

		response = await self._call_llm(prompt)
		return self._parse_decisions(response)

	async def _plan_cycle(
		self, state: SwarmState
	) -> list[PlannerDecision]:
		"""Run one planning cycle."""
		self._last_plan_time = time.monotonic()
		self._cycle_count += 1
		state_text = self._controller.render_state(state)

		pivots = analyze_stagnation(
			cycle_number=self._cycle_count,
			test_history=self._test_history,
			completion_history=self._completion_history,
			failure_history=self._failure_history,
			cost_history=self._cost_history,
			threshold=self._config.stagnation_threshold,
		)
		pivot_text = format_pivots_for_planner(pivots)
		if pivot_text:
			state_text += "\n\n" + pivot_text
		for p in pivots:
			self._learnings.add_stagnation_insight(p.trigger, p.strategy)

		# Add scaling recommendation
		scaling = self._controller.get_scaling_recommendation()
		if scaling.get("scale_up", 0) > 0:
			up = scaling["scale_up"]
			state_text += f"\n\n## Scaling Signal\nScale UP: {up} more agents recommended (pending >> active)"
		if scaling.get("scale_down", 0) > 0:
			state_text += f"\n\n## Scaling Signal\nScale DOWN: {scaling['scale_down']} idle agents could be killed"

		# Add accumulated learnings from previous runs
		learnings_text = self._learnings.get_for_planner()
		if learnings_text:
			state_text += "\n\n" + learnings_text

		# Inject cycle evaluator feedback
		evaluator_feedback = self._evaluator.get_feedback()
		if evaluator_feedback:
			state_text += f"\n\n## Previous Cycle Grade\n{evaluator_feedback}"

		# Two-step planning: analysis then decisions
		if self._config.two_step_planning:
			result = await self._two_step_plan(state_text)
			if result is not None:
				return result
			# Fall through to single-call on failure

		prompt = CYCLE_PROMPT_TEMPLATE.format(state_text=state_text)
		response = await self._call_llm(prompt)
		return self._parse_decisions(response)

	async def _two_step_plan(
		self, state_text: str
	) -> list[PlannerDecision] | None:
		"""Two-step planning: analysis call then decision call.

		Returns None if the analysis fails, signaling fallback to single-call.
		"""
		# Step 1: Analysis
		analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(state_text=state_text)
		analysis_response = await self._call_llm(analysis_prompt)
		analysis = self._parse_analysis(analysis_response)

		# Gate: validate analysis is coherent
		if analysis is None:
			logger.info("Two-step analysis parse failed, falling back to single-call")
			return None

		valid_statuses = {"on_track", "stagnating", "blocked", "recovering"}
		if analysis.get("status") not in valid_statuses:
			logger.info(
				"Two-step analysis has invalid status '%s', falling back to single-call",
				analysis.get("status"),
			)
			return None

		# Step 2: Decisions from analysis
		decision_prompt = DECISION_FROM_ANALYSIS_PROMPT.format(
			analysis_json=json.dumps(analysis, indent=2),
			state_summary=state_text[:2000],
			decision_types_reference="(See system prompt for decision type reference)",
		)
		response = await self._call_llm(decision_prompt)
		return self._parse_decisions(response)

	def _parse_analysis(self, response: str) -> dict[str, Any] | None:
		"""Parse the analysis JSON from the first step of two-step planning.

		Returns None on parse failure (caller should fall back to single-call).
		"""
		text = response.strip()

		start = text.find("{")
		if start == -1:
			return None

		# Find matching closing brace
		depth = 0
		end = start
		for i in range(start, len(text)):
			if text[i] == "{":
				depth += 1
			elif text[i] == "}":
				depth -= 1
				if depth == 0:
					end = i + 1
					break

		json_text = text[start:end]
		try:
			result = json.loads(json_text)
			if isinstance(result, dict):
				return result
		except json.JSONDecodeError:
			# Attempt repair using existing logic (adapted for objects)
			repaired = self._repair_truncated_json(json_text)
			if repaired is not None:
				try:
					result = json.loads(repaired)
					if isinstance(result, dict):
						return result
				except json.JSONDecodeError:
					pass

		return None

	async def _call_llm(self, prompt: str) -> str:
		"""Call the planner LLM via Claude Code subprocess."""
		from autodev.config import build_claude_cmd, claude_subprocess_env

		config = self._controller._config
		full_prompt = SYSTEM_PROMPT + "\n\n" + prompt

		cmd = build_claude_cmd(
			config,
			prompt=full_prompt,
			model=self._config.planner_model,
			output_format="text",
		)
		env = claude_subprocess_env(config)

		try:
			proc = await asyncio.create_subprocess_exec(
				*cmd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=str(config.target.resolved_path),
				env=env,
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(),
				timeout=300,
			)
			return stdout.decode(errors="replace")
		except asyncio.TimeoutError:
			logger.error("Planner LLM call timed out")
			return "[]"
		except Exception as e:
			logger.error("Planner LLM call failed: %s", e)
			return "[]"

	def _parse_decisions(self, response: str) -> list[PlannerDecision]:
		"""Parse structured decisions from LLM response.

		Tracks consecutive parse failures. If JSON is truncated (common when
		LLM output hits max_tokens), attempts repair by closing open strings,
		objects, and arrays.
		"""
		text = response.strip()

		start = text.find("[")
		if start == -1:
			logger.warning("No JSON array found in planner response")
			self._consecutive_parse_failures += 1
			return []

		depth = 0
		end = start
		for i in range(start, len(text)):
			if text[i] == "[":
				depth += 1
			elif text[i] == "]":
				depth -= 1
				if depth == 0:
					end = i + 1
					break

		json_text = text[start:end]
		raw = None
		try:
			raw = json.loads(json_text)
		except json.JSONDecodeError:
			# Attempt to repair truncated JSON
			repaired = self._repair_truncated_json(json_text)
			if repaired is not None:
				try:
					raw = json.loads(repaired)
					logger.info("Repaired truncated JSON from planner response")
				except json.JSONDecodeError:
					pass

		if raw is None:
			self._consecutive_parse_failures += 1
			logger.warning(
				"Failed to parse planner decisions (consecutive failures: %d/%d)",
				self._consecutive_parse_failures,
				self.MAX_CONSECUTIVE_PARSE_FAILURES,
			)
			return []

		# Reset on successful parse
		self._consecutive_parse_failures = 0

		decisions: list[PlannerDecision] = []
		for item in raw:
			try:
				dtype = DecisionType(item.get("type", ""))
				decisions.append(PlannerDecision(
					type=dtype,
					payload=item.get("payload", {}),
					reasoning=item.get("reasoning", ""),
					priority=int(item.get("priority", 0)),
				))
			except (ValueError, KeyError) as e:
				logger.warning(
					"Skipping invalid decision: %s (%s)", item, e
				)

		logger.info(
			"Parsed %d decisions from planner (cycle %d)",
			len(decisions), self._cycle_count,
		)
		return decisions

	@staticmethod
	def _repair_truncated_json(text: str) -> str | None:
		"""Attempt to repair truncated JSON by closing open structures.

		Handles the common case where the LLM output was cut off mid-response,
		producing unterminated strings, objects, or arrays.
		"""
		# Truncate to the last complete-looking object boundary
		# Find the last complete }, then close the array
		last_brace = text.rfind("}")
		if last_brace == -1:
			return None

		# Take everything up to and including the last }
		candidate = text[:last_brace + 1]

		# Check if we're inside an unterminated string by counting unescaped quotes
		in_string = False
		for i, ch in enumerate(candidate):
			if ch == '"' and (i == 0 or candidate[i - 1] != "\\"):
				in_string = not in_string

		# If we ended inside a string, back up to before the last unmatched quote
		if in_string:
			# Find the last unescaped quote and truncate before it
			for i in range(len(candidate) - 1, -1, -1):
				if candidate[i] == '"' and (i == 0 or candidate[i - 1] != "\\"):
					candidate = candidate[:i]
					break
			# Now find the last } again
			last_brace = candidate.rfind("}")
			if last_brace == -1:
				return None
			candidate = candidate[:last_brace + 1]

		# Close any remaining open brackets
		open_brackets = candidate.count("[") - candidate.count("]")
		open_braces = candidate.count("{") - candidate.count("}")
		candidate += "}" * max(open_braces, 0)
		candidate += "]" * max(open_brackets, 0)

		return candidate

	def _write_state_file(self, state: SwarmState) -> None:
		"""Write swarm state to a JSON file for the TUI dashboard."""
		state_path = Path(self._controller._config.target.resolved_path) / ".autodev-swarm-state.json"
		try:
			data = {
				"cycle": self._cycle_count,
				"mission": state.mission_objective[:100],
				"team_name": self._controller.team_name,
				"agents": [
					{
						"id": a.id[:8],
						"name": a.name,
						"role": a.role.value,
						"status": a.status.value,
						"task_id": a.current_task_id[:8] if a.current_task_id else None,
						"completed": a.tasks_completed,
						"failed": a.tasks_failed,
					}
					for a in state.agents
				],
				"tasks": [
					{
						"id": t.id[:8],
						"title": t.title[:60],
						"status": t.status.value,
						"claimed_by": t.claimed_by[:8] if t.claimed_by else None,
						"attempts": t.attempt_count,
						"max_attempts": t.max_attempts,
						"priority": t.priority.name,
					}
					for t in state.tasks
				],
				"core_tests": state.core_test_results or {},
				"stagnation": [
					{
						"metric": s.metric,
						"cycles": s.cycles_stagnant,
						"pivot": s.suggested_pivot[:80],
					}
					for s in state.stagnation_signals
				],
				"discoveries": state.recent_discoveries[:10],
				"cost_usd": state.total_cost_usd,
				"wall_minutes": state.wall_time_seconds / 60,
				"test_history": self._test_history[-20:],
				"completion_history": self._completion_history[-20:],
				"failure_history": self._failure_history[-20:],
				"log_events": self._log_events[-30:],
			}
			state_path.write_text(json.dumps(data, indent=2))
		except Exception as e:
			logger.debug("Failed to write state file: %s", e)

	def _log_cycle(
		self,
		decisions: list[PlannerDecision],
		results: list[dict[str, Any]],
	) -> None:
		"""Log a planning cycle summary."""
		succeeded = sum(1 for r in results if r.get("success"))
		failed = len(results) - succeeded
		decision_types = [d.type.value for d in decisions]
		entry = {
			"cycle": self._cycle_count,
			"decisions": decision_types,
			"ok": succeeded,
			"failed": failed,
			"reasonings": [d.reasoning[:80] for d in decisions[:5]],
		}
		self._log_events.append(entry)
		logger.info(
			"Cycle %d: %d decisions (%s), %d ok, %d failed",
			self._cycle_count,
			len(decisions),
			", ".join(decision_types),
			succeeded,
			failed,
		)
