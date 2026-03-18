"""Tests for GOAL.md fitness integration in swarm worker prompts."""

from __future__ import annotations

from autodev.config import GoalConfig, MissionConfig, SwarmConfig
from autodev.goal import (
	FitnessResult,
	GoalAction,
	GoalComponent,
	GoalSpec,
)
from autodev.swarm.models import AgentRole, AgentStatus, SwarmAgent
from autodev.swarm.worker_prompt import build_worker_prompt
from autodev.worker import build_goal_context


def _make_agent(**overrides: object) -> SwarmAgent:
	defaults = dict(
		id="agent-1",
		name="test-agent",
		role=AgentRole.IMPLEMENTER,
		status=AgentStatus.WORKING,
	)
	defaults.update(overrides)
	return SwarmAgent(**defaults)


def _make_spec(**overrides: object) -> GoalSpec:
	defaults = dict(
		name="Test Goal",
		fitness_command="python measure.py",
		target_score=1.0,
	)
	defaults.update(overrides)
	return GoalSpec(**defaults)


class TestSwarmWorkerPromptGoalSection:
	"""Tests that build_worker_prompt includes/omits goal fitness section."""

	def test_goal_section_included_when_context_provided(
		self, config: MissionConfig,
	) -> None:
		agent = _make_agent()
		goal_ctx = build_goal_context(
			_make_spec(),
			FitnessResult(composite=0.75),
		)
		prompt = build_worker_prompt(
			agent=agent,
			task_prompt="Do something",
			team_name="test-team",
			agents=[agent],
			tasks=[],
			config=config,
			swarm_config=SwarmConfig(),
			goal_context=goal_ctx,
		)
		assert "## Goal Fitness" in prompt
		assert "Test Goal" in prompt
		assert "0.750" in prompt

	def test_goal_section_omitted_when_no_context(
		self, config: MissionConfig,
	) -> None:
		agent = _make_agent()
		prompt = build_worker_prompt(
			agent=agent,
			task_prompt="Do something",
			team_name="test-team",
			agents=[agent],
			tasks=[],
			config=config,
			swarm_config=SwarmConfig(),
			goal_context="",
		)
		assert "## Goal Fitness" not in prompt

	def test_goal_section_omitted_when_default(
		self, config: MissionConfig,
	) -> None:
		agent = _make_agent()
		prompt = build_worker_prompt(
			agent=agent,
			task_prompt="Do something",
			team_name="test-team",
			agents=[agent],
			tasks=[],
			config=config,
			swarm_config=SwarmConfig(),
		)
		assert "## Goal Fitness" not in prompt

	def test_fitness_command_in_prompt(self, config: MissionConfig) -> None:
		agent = _make_agent()
		goal_ctx = build_goal_context(
			_make_spec(fitness_command="./run_fitness.sh"),
			FitnessResult(composite=0.5),
		)
		prompt = build_worker_prompt(
			agent=agent,
			task_prompt="Do something",
			team_name="test-team",
			agents=[agent],
			tasks=[],
			config=config,
			swarm_config=SwarmConfig(),
			goal_context=goal_ctx,
		)
		assert "./run_fitness.sh" in prompt

	def test_component_scores_in_prompt(self, config: MissionConfig) -> None:
		spec = _make_spec(components=[
			GoalComponent(name="tests", command="pytest -q", weight=0.6),
			GoalComponent(name="lint", command="ruff check .", weight=0.4),
		])
		fitness = FitnessResult(
			composite=0.8,
			components={"tests": 0.9, "lint": 0.65},
		)
		goal_ctx = build_goal_context(spec, fitness)
		prompt = build_worker_prompt(
			agent=_make_agent(),
			task_prompt="Do something",
			team_name="test-team",
			agents=[],
			tasks=[],
			config=config,
			swarm_config=SwarmConfig(),
			goal_context=goal_ctx,
		)
		assert "tests" in prompt
		assert "lint" in prompt
		assert "0.900" in prompt
		assert "0.650" in prompt

	def test_constraints_in_prompt(self, config: MissionConfig) -> None:
		spec = _make_spec(constraints=["No new deps", "Keep API stable"])
		goal_ctx = build_goal_context(spec, FitnessResult(composite=0.6))
		prompt = build_worker_prompt(
			agent=_make_agent(),
			task_prompt="Do something",
			team_name="test-team",
			agents=[],
			tasks=[],
			config=config,
			swarm_config=SwarmConfig(),
			goal_context=goal_ctx,
		)
		assert "No new deps" in prompt
		assert "Keep API stable" in prompt

	def test_fitness_protocol_in_prompt(self, config: MissionConfig) -> None:
		goal_ctx = build_goal_context(
			_make_spec(),
			FitnessResult(composite=0.5),
		)
		prompt = build_worker_prompt(
			agent=_make_agent(),
			task_prompt="Do something",
			team_name="test-team",
			agents=[],
			tasks=[],
			config=config,
			swarm_config=SwarmConfig(),
			goal_context=goal_ctx,
		)
		assert "Fitness Protocol" in prompt
		assert "re-measure fitness" in prompt
		assert "revert" in prompt.lower()


class TestBuildGoalContextRankedActions:
	"""Tests for ranked action catalog in build_goal_context."""

	def test_ranked_actions_included(self) -> None:
		spec = _make_spec(actions=[
			GoalAction(description="Add tests", estimated_impact="high"),
			GoalAction(description="Fix lint", estimated_impact="low"),
		])
		fitness = FitnessResult(composite=0.5)
		ctx = build_goal_context(spec, fitness)
		assert "Ranked Actions" in ctx
		assert "Add tests" in ctx
		assert "Fix lint" in ctx
		assert "[high]" in ctx
		assert "[low]" in ctx

	def test_ranked_actions_with_files_hint(self) -> None:
		spec = _make_spec(actions=[
			GoalAction(
				description="Fix module",
				files_hint=["src/foo.py", "src/bar.py"],
				estimated_impact="medium",
			),
		])
		fitness = FitnessResult(composite=0.5)
		ctx = build_goal_context(spec, fitness)
		assert "src/foo.py" in ctx
		assert "src/bar.py" in ctx

	def test_no_actions_section_when_empty(self) -> None:
		spec = _make_spec(actions=[])
		fitness = FitnessResult(composite=0.5)
		ctx = build_goal_context(spec, fitness)
		assert "Ranked Actions" not in ctx

	def test_actions_ordered_by_impact(self) -> None:
		spec = _make_spec(actions=[
			GoalAction(description="Low task", estimated_impact="low"),
			GoalAction(description="High task", estimated_impact="high"),
		])
		fitness = FitnessResult(composite=0.5)
		ctx = build_goal_context(spec, fitness)
		high_pos = ctx.index("[high]")
		low_pos = ctx.index("[low]")
		assert high_pos < low_pos

	def test_failed_fitness_returns_empty(self) -> None:
		spec = _make_spec(actions=[
			GoalAction(description="Something", estimated_impact="high"),
		])
		fitness = FitnessResult(success=False, error="timeout")
		ctx = build_goal_context(spec, fitness)
		assert ctx == ""

	def test_revert_disabled_no_revert_line(self) -> None:
		spec = _make_spec()
		fitness = FitnessResult(composite=0.5)
		goal_config = GoalConfig(revert_on_regression=False)
		ctx = build_goal_context(spec, fitness, goal_config=goal_config)
		assert "REGRESSES" not in ctx
