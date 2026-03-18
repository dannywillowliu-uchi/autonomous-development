"""Tests for GOAL.md fitness integration in mission mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodev.config import GoalConfig, MissionConfig
from autodev.continuous_controller import ContinuousController
from autodev.db import Database
from autodev.goal import FitnessResult
from autodev.models import Mission, WorkUnit
from autodev.worker import render_mission_worker_prompt

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOAL_MD_CONTENT = """\
# Goal: test coverage

Maximize test coverage.

## Fitness

echo 0.75

## Components

- unit tests (weight: 0.6): echo 0.8
- integration tests (weight: 0.4): echo 0.7

## Target

0.95

## Constraints

- No mocking of database connections
- Must run in under 5 minutes

## Actions

- Add missing unit tests for auth module [files: src/auth.py, tests/test_auth.py] [impact: high]
- Refactor integration test fixtures [files: tests/conftest.py] [impact: medium]
"""


@pytest.fixture()
def goal_config(config: MissionConfig) -> MissionConfig:
	"""Config with goal tracking enabled."""
	config.goal = GoalConfig(enabled=True, goal_file="GOAL.md", auto_detect=True)
	return config


@pytest.fixture()
def goal_file(config: MissionConfig) -> Path:
	"""Create a GOAL.md in the target path."""
	p = Path(config.target.path) / "GOAL.md"
	p.write_text(GOAL_MD_CONTENT)
	return p


# ---------------------------------------------------------------------------
# _init_goal_tracking
# ---------------------------------------------------------------------------

class TestGoalInit:
	def test_init_with_goal_file(self, goal_config: MissionConfig, db: Database, goal_file: Path) -> None:
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		assert ctrl._goal_spec is not None
		assert ctrl._goal_spec.name == "test coverage"
		assert ctrl._goal_iteration_log is not None

	def test_init_no_goal_file(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		ctrl._init_goal_tracking()
		assert ctrl._goal_spec is None
		assert ctrl._goal_iteration_log is None

	def test_init_auto_detect(self, config: MissionConfig, db: Database, goal_file: Path) -> None:
		"""auto_detect=True (default) should find GOAL.md without explicit enable."""
		config.goal.auto_detect = True
		config.goal.enabled = False
		ctrl = ContinuousController(config, db)
		ctrl._init_goal_tracking()
		assert ctrl._goal_spec is not None

	def test_init_disabled(self, config: MissionConfig, db: Database, goal_file: Path) -> None:
		"""When both enabled=False and auto_detect=False, skip even if file exists."""
		config.goal.enabled = False
		config.goal.auto_detect = False
		ctrl = ContinuousController(config, db)
		ctrl._init_goal_tracking()
		assert ctrl._goal_spec is None

	def test_init_invalid_goal_file(self, config: MissionConfig, db: Database) -> None:
		"""Invalid GOAL.md content should disable tracking gracefully."""
		config.goal.enabled = True
		p = Path(config.target.path) / "GOAL.md"
		p.write_text("this is not a valid goal file")
		ctrl = ContinuousController(config, db)
		ctrl._init_goal_tracking()
		assert ctrl._goal_spec is None


# ---------------------------------------------------------------------------
# Fitness measurement
# ---------------------------------------------------------------------------

class TestFitnessMeasurement:
	def test_measure_fitness_no_goal(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl._measure_fitness() is None

	def test_measure_fitness_success(self, goal_config: MissionConfig, db: Database, goal_file: Path) -> None:
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		result = ctrl._measure_fitness()
		assert result is not None
		assert result.success
		# Weighted: (0.8 * 0.6 + 0.7 * 0.4) / 1.0 = 0.76
		assert result.composite == pytest.approx(0.76)

	def test_measure_fitness_command_failure(self, goal_config: MissionConfig, db: Database) -> None:
		"""Fitness command failure should return a result with success=False."""
		p = Path(goal_config.target.path) / "GOAL.md"
		p.write_text("# Goal: broken\n\n## Fitness\n\nexit 1\n")
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		result = ctrl._measure_fitness()
		assert result is not None
		assert not result.success


# ---------------------------------------------------------------------------
# Goal iteration recording
# ---------------------------------------------------------------------------

class TestGoalIteration:
	def test_record_iteration(self, goal_config: MissionConfig, db: Database, goal_file: Path) -> None:
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		before = FitnessResult(composite=0.5)
		after = FitnessResult(composite=0.7)
		ctrl._record_goal_iteration(before, after, "epoch 1")

		assert ctrl._goal_iteration_log is not None
		latest = ctrl._goal_iteration_log.latest()
		assert latest is not None
		assert latest.after.composite == 0.7
		assert latest.delta == pytest.approx(0.2)

	def test_record_iteration_no_log(self, config: MissionConfig, db: Database) -> None:
		"""No-op when goal tracking isn't active."""
		ctrl = ContinuousController(config, db)
		before = FitnessResult(composite=0.5)
		after = FitnessResult(composite=0.7)
		ctrl._record_goal_iteration(before, after, "epoch 1")
		# Should not raise


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

class TestRegressionDetection:
	def test_no_regression(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		before = FitnessResult(composite=0.5)
		after = FitnessResult(composite=0.6)
		assert not ctrl._check_goal_regression(before, after)

	def test_regression_detected(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		before = FitnessResult(composite=0.7)
		after = FitnessResult(composite=0.5)
		assert ctrl._check_goal_regression(before, after)

	def test_small_regression_within_tolerance(self, config: MissionConfig, db: Database) -> None:
		"""Regression smaller than min_improvement should not trigger."""
		config.goal.min_improvement = 0.05
		ctrl = ContinuousController(config, db)
		before = FitnessResult(composite=0.7)
		after = FitnessResult(composite=0.68)  # Only 0.02 regression, below 0.05 threshold
		assert not ctrl._check_goal_regression(before, after)

	def test_regression_with_failed_measurement(self, config: MissionConfig, db: Database) -> None:
		"""Failed measurements should not trigger regression."""
		ctrl = ContinuousController(config, db)
		before = FitnessResult(composite=0.7, success=True)
		after = FitnessResult(composite=0.0, success=False)
		assert not ctrl._check_goal_regression(before, after)


# ---------------------------------------------------------------------------
# Goal-based stopping
# ---------------------------------------------------------------------------

class TestGoalStopping:
	def test_goal_met_stops(self, goal_config: MissionConfig, db: Database, goal_file: Path) -> None:
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		# Record an iteration that meets the target
		before = FitnessResult(composite=0.8)
		after = FitnessResult(composite=0.96)  # Above 0.95 target
		ctrl._record_goal_iteration(before, after, "final epoch")
		assert ctrl._check_goal_met()

	def test_goal_not_met(self, goal_config: MissionConfig, db: Database, goal_file: Path) -> None:
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		before = FitnessResult(composite=0.5)
		after = FitnessResult(composite=0.7)  # Below 0.95 target
		ctrl._record_goal_iteration(before, after, "epoch 1")
		assert not ctrl._check_goal_met()

	def test_goal_met_in_should_stop(self, goal_config: MissionConfig, db: Database, goal_file: Path) -> None:
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		before = FitnessResult(composite=0.9)
		after = FitnessResult(composite=0.96)
		ctrl._record_goal_iteration(before, after, "final")
		assert ctrl._should_stop(Mission(id="m1")) == "goal_met"

	def test_no_goal_no_stop(self, config: MissionConfig, db: Database) -> None:
		"""Without goal tracking, goal_met should never fire."""
		ctrl = ContinuousController(config, db)
		assert ctrl._should_stop(Mission(id="m1")) == ""

	def test_max_iterations_stops(self, goal_config: MissionConfig, db: Database, goal_file: Path) -> None:
		"""Max iterations should trigger stopping even when score not met."""
		goal_config.goal.max_iterations = 3
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		ctrl._goal_epoch_count = 3
		before = FitnessResult(composite=0.5)
		after = FitnessResult(composite=0.6)  # Below target
		ctrl._record_goal_iteration(before, after, "epoch 3")
		assert ctrl._check_goal_met()


# ---------------------------------------------------------------------------
# Worker prompt includes goal context
# ---------------------------------------------------------------------------

class TestWorkerGoalContext:
	def test_worker_prompt_includes_fitness_command(self) -> None:
		unit = WorkUnit(
			id="u1", title="Test task", description="Do something",
			files_hint="src/foo.py",
		)
		config = MissionConfig()
		config.target.name = "test-proj"
		config.target.verification.command = "pytest -q"

		goal_ctx = (
			"Fitness check command: echo 0.75\n"
			"Run this after your changes to verify you haven't regressed the fitness score.\n"
			"Current score: 0.750 / 0.950\n"
			"Constraints:\n"
			"  - No mocking of database connections"
		)

		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/test",
			branch_name="mc/unit-u1",
			goal_context=goal_ctx,
		)

		assert "Fitness check command: echo 0.75" in prompt
		assert "No mocking of database connections" in prompt
		assert "Current score: 0.750 / 0.950" in prompt

	def test_worker_prompt_no_goal(self) -> None:
		"""Without goal_context, prompt should not include goal section."""
		unit = WorkUnit(
			id="u1", title="Test task", description="Do something",
			files_hint="src/foo.py",
		)
		config = MissionConfig()
		config.target.name = "test-proj"
		config.target.verification.command = "pytest -q"

		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/test",
			branch_name="mc/unit-u1",
		)

		assert "Goal Fitness" not in prompt


# ---------------------------------------------------------------------------
# Build goal context methods
# ---------------------------------------------------------------------------

class TestBuildGoalContext:
	def test_build_goal_context_no_goal(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl._build_goal_context() == ""

	def test_build_goal_context_with_goal(self, goal_config: MissionConfig, db: Database, goal_file: Path) -> None:
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		ctx = ctrl._build_goal_context()
		assert "test coverage" in ctx
		assert "Prioritize work units" in ctx

	def test_build_goal_context_with_scores(self, goal_config: MissionConfig, db: Database, goal_file: Path) -> None:
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		before = FitnessResult(composite=0.5)
		after = FitnessResult(composite=0.7, components={"unit tests": 0.8, "integration tests": 0.6})
		ctrl._record_goal_iteration(before, after, "epoch 1")
		ctx = ctrl._build_goal_context()
		assert "0.700" in ctx
		assert "Trend:" in ctx

	def test_build_worker_goal_context_no_goal(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl._build_worker_goal_context() == ""

	def test_build_worker_goal_context_with_actions(
		self, goal_config: MissionConfig, db: Database, goal_file: Path,
	) -> None:
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		ctx = ctrl._build_worker_goal_context()
		assert "echo 0.75" in ctx
		assert "No mocking of database connections" in ctx
		assert "Add missing unit tests" in ctx
		assert "[high]" in ctx


# ---------------------------------------------------------------------------
# Goal score summary
# ---------------------------------------------------------------------------

class TestGoalScoreSummary:
	def test_summary_no_goal(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		assert ctrl._goal_score_summary() == ""

	def test_summary_with_data(self, goal_config: MissionConfig, db: Database, goal_file: Path) -> None:
		ctrl = ContinuousController(goal_config, db)
		ctrl._init_goal_tracking()
		before = FitnessResult(composite=0.5)
		after = FitnessResult(composite=0.7)
		ctrl._record_goal_iteration(before, after, "epoch 1")
		summary = ctrl._goal_score_summary()
		assert "test coverage" in summary
		assert "0.700" in summary
