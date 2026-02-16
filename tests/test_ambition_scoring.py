"""Tests for ambition scoring and replan logic in strategist and controller."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig
from mission_control.continuous_controller import ContinuousController, WorkerCompletion
from mission_control.db import Database
from mission_control.models import BacklogItem, Epoch, Mission, Plan, WorkUnit
from mission_control.strategist import Strategist

# -- Helpers --


def _make_strategist(db: Database | None = None) -> Strategist:
	config = MissionConfig()
	if db is None:
		db = MagicMock()
		db.get_all_missions.return_value = []
		db.get_pending_backlog.return_value = []
	return Strategist(config=config, db=db)


def _make_units(specs: list[tuple[str, str, str]]) -> list[WorkUnit]:
	"""Create units from (title, description, files_hint) tuples."""
	return [
		WorkUnit(id=f"wu{i}", title=title, description=desc, files_hint=files)
		for i, (title, desc, files) in enumerate(specs)
	]


# -- evaluate_ambition tests --


class TestEvaluateAmbition:
	def test_empty_units_returns_1(self) -> None:
		s = _make_strategist()
		assert s.evaluate_ambition([]) == 1

	def test_single_lint_fix_scores_low(self) -> None:
		s = _make_strategist()
		units = _make_units([("Fix lint errors", "Run ruff and fix formatting issues", "src/main.py")])
		score = s.evaluate_ambition(units)
		assert 1 <= score <= 3

	def test_single_typo_fix_scores_low(self) -> None:
		s = _make_strategist()
		units = _make_units([("Fix typo in README", "Minor typo correction", "README.md")])
		score = s.evaluate_ambition(units)
		assert 1 <= score <= 3

	def test_formatting_cleanup_scores_low(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Fix whitespace issues", "cleanup formatting", "a.py"),
			("Fix style nits", "minor style cleanup", "b.py"),
		])
		score = s.evaluate_ambition(units)
		assert 1 <= score <= 4

	def test_new_feature_scores_moderate(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Add user authentication", "Implement JWT-based auth", "src/auth.py, src/middleware.py"),
			("Add auth tests", "Create test suite for auth", "tests/test_auth.py"),
		])
		score = s.evaluate_ambition(units)
		assert 4 <= score <= 7

	def test_architecture_change_scores_high(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Redesign event system", "New architecture for event-driven pipeline", "src/events.py, src/pipeline.py"),
			("Build distributed queue", "New system for async task distribution", "src/queue.py, src/worker.py"),
			("Integrate message bus", "Integration with event bus infrastructure", "src/bus.py, src/config.py"),
		])
		score = s.evaluate_ambition(units)
		assert score >= 7

	def test_new_system_scores_high(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Build new module for caching", "New system with Redis backend",
			 "src/cache/engine.py, src/cache/store.py"),
			("Add cache migration", "Migration scripts for cache infrastructure",
			 "src/cache/migrate.py"),
		])
		score = s.evaluate_ambition(units)
		assert score >= 6

	def test_many_files_increases_score(self) -> None:
		s = _make_strategist()
		few_files = _make_units([
			("Add feature X", "Implement feature X", "src/x.py"),
		])
		many_files = _make_units([
			("Add feature X", "Implement feature X",
			 "src/x.py, src/y.py, src/z.py, src/a.py, src/b.py, "
			 "src/c.py, src/d.py, src/e.py, src/f.py, src/g.py"),
		])
		score_few = s.evaluate_ambition(few_files)
		score_many = s.evaluate_ambition(many_files)
		assert score_many >= score_few

	def test_many_units_increases_score(self) -> None:
		s = _make_strategist()
		few = _make_units([
			("Add feature", "Implement it", "a.py"),
		])
		many = _make_units([
			(f"Add feature {i}", f"Implement feature {i}", f"src/mod{i}.py")
			for i in range(6)
		])
		score_few = s.evaluate_ambition(few)
		score_many = s.evaluate_ambition(many)
		assert score_many >= score_few

	def test_score_clamped_1_to_10(self) -> None:
		s = _make_strategist()
		# Extreme high
		units = _make_units([
			(f"Redesign system {i}", f"New architecture for distributed infrastructure component {i}",
			 ", ".join(f"src/mod{j}.py" for j in range(20)))
			for i in range(10)
		])
		score = s.evaluate_ambition(units)
		assert 1 <= score <= 10

	def test_mixed_unit_types(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Fix lint in config", "Minor lint cleanup", "src/config.py"),
			("Redesign auth layer", "New architecture for auth", "src/auth.py, src/middleware.py, src/tokens.py"),
			("Add feature flag", "Implement toggle logic", "src/flags.py"),
		])
		score = s.evaluate_ambition(units)
		# Mix of low and high should land moderate-to-high
		assert 4 <= score <= 9

	def test_refactor_scores_moderate(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Refactor database layer", "Improve error handling and connection pooling", "src/db.py, src/pool.py"),
			("Update API endpoints", "Enhance endpoint validation", "src/api.py"),
		])
		score = s.evaluate_ambition(units)
		assert 4 <= score <= 7

	def test_no_files_hint(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Add new feature", "Build something interesting", ""),
		])
		score = s.evaluate_ambition(units)
		# Should still produce a valid score
		assert 1 <= score <= 10

	def test_description_contributes_to_scoring(self) -> None:
		s = _make_strategist()
		# Title is generic but description has high keywords
		units = _make_units([
			("Task 1", "Redesign the distributed architecture for the pipeline", "src/pipe.py"),
		])
		score = s.evaluate_ambition(units)
		assert score >= 5


# -- should_replan tests --


class TestShouldReplan:
	def test_high_ambition_no_replan(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Important task", priority_score=9.0)]
		should, reason = s.should_replan(7, items)
		assert should is False
		assert reason == ""

	def test_exactly_4_no_replan(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Important task", priority_score=9.0)]
		should, reason = s.should_replan(4, items)
		assert should is False

	def test_low_ambition_no_backlog(self) -> None:
		s = _make_strategist()
		should, reason = s.should_replan(2, [])
		assert should is False
		assert "No higher-priority" in reason

	def test_low_ambition_low_priority_backlog(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Trivial task", priority_score=2.0)]
		should, reason = s.should_replan(2, items)
		assert should is False
		assert "No high-priority" in reason

	def test_low_ambition_high_priority_triggers_replan(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Critical auth fix", priority_score=9.0)]
		should, reason = s.should_replan(2, items)
		assert should is True
		assert "Critical auth fix" in reason
		assert "Ambition score 2" in reason

	def test_ambition_3_triggers_replan(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Build new API", priority_score=8.0)]
		should, reason = s.should_replan(3, items)
		assert should is True
		assert "Build new API" in reason

	def test_ambition_1_triggers_replan(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Redesign DB", priority_score=7.5)]
		should, reason = s.should_replan(1, items)
		assert should is True

	def test_pinned_score_used_when_present(self) -> None:
		s = _make_strategist()
		# Low priority_score but high pinned_score
		items = [BacklogItem(title="Pinned task", priority_score=2.0, pinned_score=9.0)]
		should, reason = s.should_replan(2, items)
		assert should is True
		assert "priority=9.0" in reason

	def test_pinned_score_none_uses_priority(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Regular task", priority_score=8.0, pinned_score=None)]
		should, reason = s.should_replan(2, items)
		assert should is True
		assert "priority=8.0" in reason

	def test_mixed_priorities_uses_first_high(self) -> None:
		s = _make_strategist()
		items = [
			BacklogItem(title="Low task", priority_score=3.0),
			BacklogItem(title="High task", priority_score=8.0),
		]
		should, reason = s.should_replan(2, items)
		# Only the first high-priority item (above 5.0) is referenced
		assert should is True
		assert "High task" in reason


# -- Controller integration tests --


class TestControllerStrategistIntegration:
	@pytest.mark.asyncio
	async def test_strategist_evaluate_ambition_used_in_dispatch(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When strategist is set, its evaluate_ambition is used instead of _score_ambition."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 5
		ctrl = ContinuousController(config, db)

		mock_strategist = MagicMock(spec=Strategist)
		mock_strategist.evaluate_ambition.return_value = 9
		mock_strategist.should_replan.return_value = (False, "")
		ctrl._strategist = mock_strategist

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [
					WorkUnit(id=f"wu-{i}", plan_id=plan.id, title=f"Task {i}", priority=2)
					for i in range(3)
				]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
		):
			result = await ctrl.run()

		mock_strategist.evaluate_ambition.assert_called_once()
		assert result.ambition_score == 9
		assert ctrl.ambition_score == 9

	@pytest.mark.asyncio
	async def test_strategist_should_replan_logs_warning(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When strategist recommends replanning, a warning is logged."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 5
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		mock_strategist = MagicMock(spec=Strategist)
		mock_strategist.evaluate_ambition.return_value = 2
		mock_strategist.should_replan.return_value = (True, "Low ambition, replan needed")
		ctrl._strategist = mock_strategist

		# Insert backlog items for should_replan to find
		db.insert_backlog_item(BacklogItem(
			id="bl1", title="Big task", priority_score=9.0, status="pending",
		))

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [WorkUnit(id="wu-1", plan_id=plan.id, title="Fix lint", priority=8)]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()

		warning_messages: list[str] = []

		def capture_warning(msg: str, *args: object, **kwargs: object) -> None:
			warning_messages.append(msg % args if args else msg)

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch.object(
				logging.getLogger("mission_control.continuous_controller"),
				"warning", side_effect=capture_warning,
			),
		):
			result = await ctrl.run()

		assert result.ambition_score == 2
		replan_warnings = [m for m in warning_messages if "replan" in m.lower()]
		assert len(replan_warnings) > 0

	@pytest.mark.asyncio
	async def test_no_strategist_uses_score_ambition(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Without a strategist, the controller falls back to _score_ambition."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 5
		ctrl = ContinuousController(config, db)
		# No strategist set -- ctrl._strategist is None

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [
					WorkUnit(id=f"wu-{i}", plan_id=plan.id, title=f"Task {i}", priority=2,
						files_hint=f"src/mod{i}.py")
					for i in range(3)
				]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch.object(ctrl, "_score_ambition", return_value=6) as mock_score,
		):
			result = await ctrl.run()

		mock_score.assert_called_once()
		assert result.ambition_score == 6

	@pytest.mark.asyncio
	async def test_ambition_score_written_to_db(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Ambition score from strategist should be persisted to the mission DB record."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 5
		ctrl = ContinuousController(config, db)

		mock_strategist = MagicMock(spec=Strategist)
		mock_strategist.evaluate_ambition.return_value = 7
		mock_strategist.should_replan.return_value = (False, "")
		ctrl._strategist = mock_strategist

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [WorkUnit(id="wu-1", plan_id=plan.id, title="Build system", priority=1)]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
		):
			result = await ctrl.run()

		assert result.ambition_score == 7

		# Check DB persistence
		db_mission = db.get_latest_mission()
		assert db_mission is not None
		assert db_mission.ambition_score == 7

	@pytest.mark.asyncio
	async def test_should_replan_false_no_warning(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When should_replan returns False, no replan warning is logged."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 5
		ctrl = ContinuousController(config, db)

		mock_strategist = MagicMock(spec=Strategist)
		mock_strategist.evaluate_ambition.return_value = 5
		mock_strategist.should_replan.return_value = (False, "")
		ctrl._strategist = mock_strategist

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "",
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [WorkUnit(id="wu-1", plan_id=plan.id, title="Task", priority=3)]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()

		warning_messages: list[str] = []

		def capture_warning(msg: str, *args: object, **kwargs: object) -> None:
			warning_messages.append(msg % args if args else msg)

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch.object(
				logging.getLogger("mission_control.continuous_controller"),
				"warning", side_effect=capture_warning,
			),
		):
			await ctrl.run()

		replan_warnings = [m for m in warning_messages if "replan" in m.lower()]
		assert len(replan_warnings) == 0
