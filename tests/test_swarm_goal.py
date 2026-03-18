"""Tests for GOAL.md integration into swarm mode."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from autodev.goal import FitnessResult, GoalComponent, GoalSpec
from autodev.swarm.context import ContextSynthesizer
from autodev.swarm.models import (
	AgentStatus,
	SwarmAgent,
	SwarmState,
	SwarmTask,
	TaskStatus,
)
from autodev.swarm.planner import DrivingPlanner
from autodev.swarm.prompts import (
	GOAL_CONTEXT_TEMPLATE,
	GOAL_SYSTEM_PROMPT_ADDENDUM,
)


def _make_config(goal_enabled: bool = True) -> MagicMock:
	config = MagicMock()
	config.target.name = "test-project"
	config.target.objective = "Build a compiler"
	config.target.resolved_path = "/tmp/test-project"
	config.goal.enabled = goal_enabled
	config.goal.auto_detect = True
	config.goal.goal_file = "GOAL.md"
	config.goal.fitness_timeout = 60
	config.goal.revert_on_regression = True
	config.goal.min_improvement = 0.01
	config.goal.target_score = 1.0
	config.goal.max_iterations = 0
	config.goal.log_file = ".goal-iterations.jsonl"
	return config


def _make_db() -> MagicMock:
	db = MagicMock()
	db.get_knowledge_for_mission.return_value = []
	return db


def _make_goal_spec(target: float = 1.0) -> GoalSpec:
	return GoalSpec(
		name="Test Goal",
		description="A test fitness function",
		fitness_command="echo 0.75",
		components=[
			GoalComponent(name="tests", command="echo 0.8", weight=2.0),
			GoalComponent(name="lint", command="echo 0.9", weight=1.0),
		],
		target_score=target,
		constraints=["No regressions allowed"],
	)


def _make_fitness_result(
	composite: float = 0.75,
	components: dict[str, float] | None = None,
	success: bool = True,
) -> FitnessResult:
	return FitnessResult(
		composite=composite,
		components=components or {"tests": 0.8, "lint": 0.9},
		success=success,
	)


# -- SwarmState with goal fields --


class TestSwarmStateGoalFields:
	def test_default_goal_fields_are_none(self) -> None:
		state = SwarmState()
		assert state.goal_spec is None
		assert state.current_fitness is None
		assert state.score_history == []
		assert state.goal_met is False

	def test_goal_fields_populated(self) -> None:
		spec = _make_goal_spec()
		fitness = _make_fitness_result()
		state = SwarmState(
			goal_spec=spec,
			current_fitness=fitness,
			score_history=[0.5, 0.6, 0.75],
			goal_met=False,
		)
		assert state.goal_spec.name == "Test Goal"
		assert state.current_fitness.composite == 0.75
		assert len(state.score_history) == 3
		assert state.goal_met is False

	def test_goal_met_flag(self) -> None:
		spec = _make_goal_spec(target=0.7)
		fitness = _make_fitness_result(composite=0.8)
		state = SwarmState(
			goal_spec=spec,
			current_fitness=fitness,
			goal_met=True,
		)
		assert state.goal_met is True


# -- Context rendering with goal section --


class TestContextGoalRendering:
	def test_render_includes_goal_progress_section(self) -> None:
		spec = _make_goal_spec(target=1.0)
		fitness = _make_fitness_result(composite=0.75)
		state = SwarmState(
			mission_objective="Build something",
			goal_spec=spec,
			current_fitness=fitness,
			score_history=[0.5, 0.6, 0.75],
		)
		rendered = ContextSynthesizer._render_goal_progress(state)
		assert "## Goal Progress" in rendered
		assert "Test Goal" in rendered
		assert "0.750" in rendered
		assert "1.000" in rendered
		assert "IMPROVING" in rendered

	def test_render_shows_regression_trend(self) -> None:
		spec = _make_goal_spec()
		fitness = _make_fitness_result(composite=0.5)
		state = SwarmState(
			goal_spec=spec,
			current_fitness=fitness,
			score_history=[0.8, 0.7, 0.5],
		)
		rendered = ContextSynthesizer._render_goal_progress(state)
		assert "REGRESSING" in rendered

	def test_render_shows_flat_trend(self) -> None:
		spec = _make_goal_spec()
		fitness = _make_fitness_result(composite=0.75)
		state = SwarmState(
			goal_spec=spec,
			current_fitness=fitness,
			score_history=[0.75, 0.75, 0.75],
		)
		rendered = ContextSynthesizer._render_goal_progress(state)
		assert "FLAT" in rendered

	def test_render_shows_goal_met(self) -> None:
		spec = _make_goal_spec(target=0.7)
		fitness = _make_fitness_result(composite=0.8)
		state = SwarmState(
			goal_spec=spec,
			current_fitness=fitness,
			goal_met=True,
		)
		rendered = ContextSynthesizer._render_goal_progress(state)
		assert "GOAL MET" in rendered

	def test_render_shows_component_breakdown(self) -> None:
		spec = _make_goal_spec()
		fitness = _make_fitness_result(components={"tests": 0.8, "lint": 0.9})
		state = SwarmState(
			goal_spec=spec,
			current_fitness=fitness,
		)
		rendered = ContextSynthesizer._render_goal_progress(state)
		assert "tests" in rendered
		assert "lint" in rendered
		assert "w=2.0" in rendered
		assert "w=1.0" in rendered

	def test_render_shows_constraints(self) -> None:
		spec = _make_goal_spec()
		fitness = _make_fitness_result()
		state = SwarmState(
			goal_spec=spec,
			current_fitness=fitness,
		)
		rendered = ContextSynthesizer._render_goal_progress(state)
		assert "No regressions allowed" in rendered

	def test_render_empty_when_no_goal(self) -> None:
		state = SwarmState()
		rendered = ContextSynthesizer._render_goal_progress(state)
		assert rendered == ""

	def test_render_for_planner_includes_goal(self) -> None:
		"""Integration: render_for_planner includes goal section when active."""
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		spec = _make_goal_spec()
		fitness = _make_fitness_result()
		state = SwarmState(
			mission_objective="Build a compiler",
			goal_spec=spec,
			current_fitness=fitness,
			score_history=[0.5, 0.75],
		)
		rendered = ctx.render_for_planner(state)
		assert "## Goal Progress" in rendered
		assert "## Mission" in rendered


# -- Context build_state with goal evaluation --


class TestContextBuildStateGoal:
	@patch("autodev.swarm.context.ContextSynthesizer._load_goal_spec")
	def test_build_state_without_goal(self, mock_load: MagicMock) -> None:
		ctx = ContextSynthesizer(_make_config(goal_enabled=False), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		assert state.goal_spec is None
		assert state.current_fitness is None
		assert state.goal_met is False

	@patch("autodev.swarm.context.Path")
	def test_load_goal_spec_parses_file(self, mock_path_cls: MagicMock, tmp_path: Path) -> None:
		goal_md = tmp_path / "GOAL.md"
		goal_md.write_text(
			"# Goal: Test\n\n## Fitness\necho 0.5\n\n## Target\n0.8\n"
		)
		config = _make_config()
		config.target.resolved_path = str(tmp_path)
		ctx = ContextSynthesizer(config, _make_db(), "test-team")
		# Reset mock so Path works normally
		mock_path_cls.side_effect = Path
		ctx._load_goal_spec()
		assert ctx._goal_spec is not None
		assert ctx._goal_spec.name == "Test"
		assert ctx._goal_spec.target_score == 0.8

	def test_evaluate_goal_fitness_no_spec(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		ctx._goal_loaded = True  # skip loading
		ctx._goal_spec = None
		spec, fitness, met = ctx._evaluate_goal_fitness()
		assert spec is None
		assert fitness is None
		assert met is False

	@patch("autodev.goal.run_fitness")
	def test_evaluate_goal_fitness_with_spec(self, mock_run: MagicMock) -> None:
		mock_run.return_value = FitnessResult(composite=0.85, success=True)
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		ctx._goal_loaded = True
		ctx._goal_spec = _make_goal_spec(target=1.0)
		spec, fitness, met = ctx._evaluate_goal_fitness()
		assert spec is not None
		assert fitness is not None
		assert fitness.composite == 0.85
		assert met is False
		assert ctx._score_history == [0.85]

	@patch("autodev.goal.run_fitness")
	def test_evaluate_goal_fitness_met(self, mock_run: MagicMock) -> None:
		mock_run.return_value = FitnessResult(composite=1.0, success=True)
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		ctx._goal_loaded = True
		ctx._goal_spec = _make_goal_spec(target=0.9)
		spec, fitness, met = ctx._evaluate_goal_fitness()
		assert met is True

	@patch("autodev.goal.run_fitness")
	def test_score_history_accumulates(self, mock_run: MagicMock) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		ctx._goal_loaded = True
		ctx._goal_spec = _make_goal_spec()
		for score in [0.5, 0.6, 0.7]:
			mock_run.return_value = FitnessResult(composite=score, success=True)
			ctx._evaluate_goal_fitness()
		assert ctx._score_history == [0.5, 0.6, 0.7]


# -- Planner stopping condition --


class TestPlannerGoalStopping:
	def _make_planner(self) -> DrivingPlanner:
		controller = MagicMock()
		controller._config.target.resolved_path = "/tmp/test"
		swarm_config = MagicMock()
		swarm_config.planner_cooldown = 60
		swarm_config.max_agents = 4
		swarm_config.min_agents = 1
		swarm_config.stagnation_threshold = 3
		swarm_config.two_step_planning = False
		swarm_config.daemon_mode = False
		return DrivingPlanner(controller, swarm_config)

	def test_should_stop_when_goal_met_no_active(self) -> None:
		planner = self._make_planner()
		state = SwarmState(
			goal_spec=_make_goal_spec(target=0.7),
			current_fitness=_make_fitness_result(composite=0.8),
			goal_met=True,
			agents=[],
			tasks=[SwarmTask(status=TaskStatus.COMPLETED)],
		)
		assert planner._should_stop(state) is True

	def test_should_not_stop_when_goal_met_agents_active(self) -> None:
		planner = self._make_planner()
		state = SwarmState(
			goal_spec=_make_goal_spec(target=0.7),
			current_fitness=_make_fitness_result(composite=0.8),
			goal_met=True,
			agents=[SwarmAgent(name="worker", status=AgentStatus.WORKING)],
			tasks=[SwarmTask(status=TaskStatus.IN_PROGRESS)],
		)
		assert planner._should_stop(state) is False

	def test_should_not_stop_when_goal_not_met(self) -> None:
		planner = self._make_planner()
		state = SwarmState(
			goal_spec=_make_goal_spec(target=1.0),
			current_fitness=_make_fitness_result(composite=0.5),
			goal_met=False,
			agents=[SwarmAgent(name="worker", status=AgentStatus.WORKING)],
			tasks=[SwarmTask(status=TaskStatus.IN_PROGRESS)],
		)
		assert planner._should_stop(state) is False


# -- Planner goal delta injection --


class TestPlannerGoalDelta:
	def _make_planner(self) -> DrivingPlanner:
		controller = MagicMock()
		controller._config.target.resolved_path = "/tmp/test"
		swarm_config = MagicMock()
		swarm_config.planner_cooldown = 60
		swarm_config.max_agents = 4
		swarm_config.min_agents = 1
		swarm_config.stagnation_threshold = 3
		swarm_config.two_step_planning = False
		swarm_config.daemon_mode = False
		return DrivingPlanner(controller, swarm_config)

	def test_inject_goal_delta_first_cycle(self) -> None:
		planner = self._make_planner()
		state = SwarmState(
			current_fitness=_make_fitness_result(composite=0.75),
		)
		result = planner._inject_goal_delta("base text", state)
		# First cycle has no previous score, so no delta section
		assert "Score Delta" not in result
		assert planner._prev_fitness_score == 0.75

	def test_inject_goal_delta_improvement(self) -> None:
		planner = self._make_planner()
		planner._prev_fitness_score = 0.5
		state = SwarmState(
			current_fitness=_make_fitness_result(composite=0.75),
		)
		result = planner._inject_goal_delta("base text", state)
		assert "Score Delta" in result
		assert "+0.250" in result
		assert "REGRESSED" not in result

	def test_inject_goal_delta_regression(self) -> None:
		planner = self._make_planner()
		planner._prev_fitness_score = 0.8
		state = SwarmState(
			current_fitness=_make_fitness_result(composite=0.6),
		)
		result = planner._inject_goal_delta("base text", state)
		assert "REGRESSED" in result
		assert "-0.200" in result

	def test_inject_goal_met_notice(self) -> None:
		planner = self._make_planner()
		state = SwarmState(
			current_fitness=_make_fitness_result(composite=1.0),
			goal_met=True,
		)
		result = planner._inject_goal_delta("base text", state)
		assert "GOAL MET" in result
		assert "Do NOT create new tasks" in result

	def test_inject_no_goal(self) -> None:
		planner = self._make_planner()
		state = SwarmState()
		result = planner._inject_goal_delta("base text", state)
		assert result == "base text"


# -- Planner system prompt --


class TestPlannerGoalSystemPrompt:
	def _make_planner(self) -> DrivingPlanner:
		controller = MagicMock()
		controller._config.target.resolved_path = "/tmp/test"
		swarm_config = MagicMock()
		swarm_config.planner_cooldown = 60
		swarm_config.max_agents = 4
		swarm_config.min_agents = 1
		swarm_config.stagnation_threshold = 3
		swarm_config.two_step_planning = False
		swarm_config.daemon_mode = False
		return DrivingPlanner(controller, swarm_config)

	def test_system_prompt_without_goal(self) -> None:
		planner = self._make_planner()
		prompt = planner._get_system_prompt(None)
		assert "Goal-Based Fitness Scoring" not in prompt

	def test_system_prompt_with_goal(self) -> None:
		planner = self._make_planner()
		state = SwarmState(goal_spec=_make_goal_spec())
		prompt = planner._get_system_prompt(state)
		assert "Goal-Based Fitness Scoring" in prompt
		assert "Prioritize by score impact" in prompt

	def test_system_prompt_no_goal_spec(self) -> None:
		planner = self._make_planner()
		state = SwarmState()  # goal_spec is None
		prompt = planner._get_system_prompt(state)
		assert "Goal-Based Fitness Scoring" not in prompt


# -- Prompt templates --


class TestGoalPromptTemplates:
	def test_goal_context_template_renders(self) -> None:
		rendered = GOAL_CONTEXT_TEMPLATE.format(
			prev_score=0.5,
			current_score=0.75,
			delta=0.25,
			regression_warning="",
		)
		assert "0.500" in rendered
		assert "0.750" in rendered
		assert "+0.250" in rendered

	def test_goal_context_template_with_regression(self) -> None:
		rendered = GOAL_CONTEXT_TEMPLATE.format(
			prev_score=0.8,
			current_score=0.6,
			delta=-0.2,
			regression_warning="**WARNING: Score REGRESSED!**\n",
		)
		assert "REGRESSED" in rendered

	def test_goal_system_prompt_addendum_content(self) -> None:
		assert "Prioritize by score impact" in GOAL_SYSTEM_PROMPT_ADDENDUM
		assert "Detect regressions" in GOAL_SYSTEM_PROMPT_ADDENDUM
		assert "Goal-based stopping" in GOAL_SYSTEM_PROMPT_ADDENDUM
