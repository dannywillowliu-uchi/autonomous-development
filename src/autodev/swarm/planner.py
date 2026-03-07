"""Driving planner -- the brain of the swarm.

Runs as a series of LLM calls in a Python async loop (not a persistent
session). Each cycle: observe state -> reason -> decide -> execute.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from autodev.swarm.models import (
	AgentStatus,
	DecisionType,
	PlannerDecision,
	SwarmState,
	TaskStatus,
)
from autodev.swarm.prompts import (
	CYCLE_PROMPT_TEMPLATE,
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

	async def run(self, core_test_runner: Any = None) -> None:
		"""Main planner loop. Runs until stopped or escalated."""
		self._running = True
		logger.info("Driving planner started")

		try:
			# Initial planning cycle
			state = self._build_state(core_test_runner)
			decisions = await self._initial_plan(state)
			await self._controller.execute_decisions(decisions)

			# Main loop
			while self._running:
				await asyncio.sleep(self._config.planner_cooldown)

				# Monitor agents, collect events
				events = await self._controller.monitor_agents()

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

				# Check if we should plan
				if self._should_plan(state, events):
					decisions = await self._plan_cycle(state)
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

	def _build_state(self, core_test_runner: Any = None) -> SwarmState:
		"""Build current swarm state, optionally running core tests."""
		core_results = None
		if core_test_runner:
			try:
				core_results = core_test_runner()
			except Exception as e:
				logger.warning("Core test runner failed: %s", e)
		return self._controller.build_state(core_test_results=core_results)

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

	def _should_plan(
		self, state: SwarmState, events: list[dict[str, Any]]
	) -> bool:
		"""Decide whether to run a planning cycle."""
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
			return True
		return False

	async def _initial_plan(
		self, state: SwarmState
	) -> list[PlannerDecision]:
		"""Generate the initial task decomposition and agent spawns."""
		self._cycle_count += 1
		state_text = self._controller.render_state(state)

		max_hint = self._config.max_agents if self._config.max_agents > 0 else 8
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

		# Add scaling recommendation
		scaling = self._controller.get_scaling_recommendation()
		if scaling.get("scale_up", 0) > 0:
			up = scaling["scale_up"]
			state_text += f"\n\n## Scaling Signal\nScale UP: {up} more agents recommended (pending >> active)"
		if scaling.get("scale_down", 0) > 0:
			state_text += f"\n\n## Scaling Signal\nScale DOWN: {scaling['scale_down']} idle agents could be killed"

		prompt = CYCLE_PROMPT_TEMPLATE.format(state_text=state_text)
		response = await self._call_llm(prompt)
		return self._parse_decisions(response)

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
		"""Parse structured decisions from LLM response."""
		text = response.strip()

		start = text.find("[")
		if start == -1:
			logger.warning("No JSON array found in planner response")
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

		try:
			raw = json.loads(text[start:end])
		except json.JSONDecodeError as e:
			logger.warning("Failed to parse planner decisions: %s", e)
			return []

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

	def _log_cycle(
		self,
		decisions: list[PlannerDecision],
		results: list[dict[str, Any]],
	) -> None:
		"""Log a planning cycle summary."""
		succeeded = sum(1 for r in results if r.get("success"))
		failed = len(results) - succeeded
		decision_types = [d.type.value for d in decisions]
		logger.info(
			"Cycle %d: %d decisions (%s), %d ok, %d failed",
			self._cycle_count,
			len(decisions),
			", ".join(decision_types),
			succeeded,
			failed,
		)
