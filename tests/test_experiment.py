"""Tests for experiment mode: CLI flag, DB operations, model, prompt, and controller skip."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mission_control.cli import build_parser
from mission_control.db import Database
from mission_control.models import Epoch, ExperimentResult, Handoff, Mission, Plan, WorkUnit, _now_iso
from mission_control.worker import render_mission_worker_prompt

# -- DB experiment_results CRUD tests --


@pytest.fixture
def db(tmp_path: Path) -> Database:
	d = Database(tmp_path / "test.db")
	yield d
	d.close()


def _create_mission_and_unit(db: Database, mission_id: str, unit_id: str) -> None:
	"""Helper: insert a mission and work unit to satisfy FK constraints."""
	from mission_control.models import Mission, Plan
	mission = Mission(id=mission_id, objective="test")
	db.insert_mission(mission)
	plan = Plan(id="plan-" + unit_id, objective="test")
	db.insert_plan(plan)
	unit = WorkUnit(id=unit_id, plan_id=plan.id, title="test unit")
	db.insert_work_unit(unit)


class TestExperimentResultDB:
	def test_insert_and_get(self, db: Database) -> None:
		_create_mission_and_unit(db, "m-1", "wu-1")
		r = ExperimentResult(
			id="exp-1",
			work_unit_id="wu-1",
			mission_id="m-1",
			approach_count=2,
			comparison_report='{"a": 1}',
			recommended_approach="A",
		)
		db.insert_experiment_result(r)
		fetched = db.get_experiment_result("exp-1")
		assert fetched is not None
		assert fetched.id == "exp-1"
		assert fetched.work_unit_id == "wu-1"
		assert fetched.mission_id == "m-1"
		assert fetched.approach_count == 2
		assert fetched.comparison_report == '{"a": 1}'
		assert fetched.recommended_approach == "A"

	def test_get_results_for_mission(self, db: Database) -> None:
		from mission_control.models import Mission, Plan
		mission = Mission(id="m-1", objective="test")
		db.insert_mission(mission)
		plan = Plan(id="plan-multi", objective="test")
		db.insert_plan(plan)
		for i in range(3):
			unit = WorkUnit(id=f"wu-{i}", plan_id=plan.id, title=f"unit {i}")
			db.insert_work_unit(unit)
			db.insert_experiment_result(ExperimentResult(
				id=f"exp-{i}",
				work_unit_id=f"wu-{i}",
				mission_id="m-1",
				comparison_report=f'{{"run": {i}}}',
			))
		# Different mission
		mission2 = Mission(id="m-2", objective="other")
		db.insert_mission(mission2)
		unit_other = WorkUnit(id="wu-other", plan_id=plan.id, title="other")
		db.insert_work_unit(unit_other)
		db.insert_experiment_result(ExperimentResult(
			id="exp-other",
			work_unit_id="wu-other",
			mission_id="m-2",
		))
		results = db.get_experiment_results_for_mission("m-1")
		assert len(results) == 3
		assert all(r.mission_id == "m-1" for r in results)


# -- Worker prompt template tests --


class TestExperimentWorkerPrompt:
	def test_render_mission_prompt_for_experiment_unit(self) -> None:
		"""Experiment units with unit_type='research' use the research template (skip merge)."""
		from mission_control.worker import render_mission_worker_prompt

		config = MagicMock()
		config.target.name = "test-project"
		config.target.verification.command = "pytest"

		unit = WorkUnit(
			title="Compare caching strategies",
			description="Try Redis vs Memcached",
			unit_type="research",  # experiment units routed as research to skip merge
			experiment_mode=True,
		)
		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/ws",
			branch_name="mc/unit-test",
		)
		assert "Compare caching strategies" in prompt
		assert "Try Redis vs Memcached" in prompt


# -- CLI parser tests --


class TestExperimentCLIFlag:
	def test_experiment_flag_exists(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission"])
		assert hasattr(args, "experiment")
		assert args.experiment is False


# -- Experiment prompt rendering tests --


class TestExperimentPromptRendering:
	"""Verify template selection based on unit_type and experiment_mode."""

	def _make_config(self) -> MagicMock:
		config = MagicMock()
		config.target.name = "test-project"
		config.target.verification.command = "pytest -q"
		return config

	def test_experiment_unit_selects_experiment_template(self) -> None:
		"""unit_type='experiment' should use EXPERIMENT_WORKER_PROMPT_TEMPLATE."""
		config = self._make_config()
		unit = WorkUnit(
			title="Compare caching strategies",
			description="Try Redis vs Memcached",
			unit_type="experiment",
			experiment_mode=True,
		)
		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/ws",
			branch_name="mc/unit-test",
		)
		assert "experiment agent" in prompt.lower()
		assert "Experiment Task" in prompt
		assert "Compare caching strategies" in prompt
		assert "Try Redis vs Memcached" in prompt
		assert "comparison report" in prompt.lower()

	def test_experiment_template_includes_json_comparison_format(self) -> None:
		"""Experiment prompt should instruct JSON comparison report output."""
		config = self._make_config()
		unit = WorkUnit(
			title="Benchmark approaches",
			description="Test A vs B",
			unit_type="experiment",
			experiment_mode=True,
		)
		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/ws",
			branch_name="mc/unit-test",
		)
		assert "comparison_report" in prompt
		assert "recommended_approach" in prompt


# -- Controller experiment completion flow tests --


@pytest.fixture
def db_runtime(tmp_path: Path) -> Database:
	db_path = tmp_path / "test.db"
	d = Database(str(db_path))
	return d


class TestControllerExperimentCompletion:
	"""Test experiment unit handling in _process_completions()."""

	def _insert_plan(self, db: Database, plan_id: str) -> None:
		"""Insert a Plan record to satisfy FK constraints for work units."""
		db.insert_plan(Plan(id=plan_id, objective="test"))

	def _make_controller(self, db: Database) -> MagicMock:
		"""Create a minimal mock controller with the experiment completion logic."""
		from mission_control.continuous_controller import ContinuousController
		from mission_control.degradation import DegradationManager

		config = MagicMock()
		config.target.resolved_path = Path("/tmp/fake")
		config.continuous = MagicMock()
		controller = ContinuousController.__new__(ContinuousController)
		controller.config = config
		controller.db = db
		controller.running = True
		controller._green_branch = MagicMock()
		controller._planner = MagicMock()
		controller._notifier = None
		controller._event_stream = None
		controller._completion_queue = asyncio.Queue()
		controller._total_merged = 0
		controller._total_failed = 0
		controller._completed_unit_ids = set()
		controller._state_changelog = []
		controller._backlog_item_ids = []
		controller._degradation = DegradationManager()
		return controller

	def test_experiment_unit_skips_merge(self, db_runtime: Database) -> None:
		"""Experiment units should not call merge_unit on green branch."""
		from mission_control.continuous_controller import WorkerCompletion

		controller = self._make_controller(db_runtime)
		mission = Mission(id="m-1", objective="test")
		db_runtime.insert_mission(mission)

		epoch = Epoch(id="ep-1", mission_id="m-1")
		db_runtime.insert_epoch(epoch)

		self._insert_plan(db_runtime, "p-1")
		unit = WorkUnit(
			id="wu-exp-1",
			plan_id="p-1",
			title="Compare caching",
			description="Redis vs Memcached",
			unit_type="experiment",
			experiment_mode=True,
			status="completed",
			finished_at=_now_iso(),
		)
		db_runtime.insert_work_unit(unit)

		handoff = Handoff(
			id="h-1",
			work_unit_id="wu-exp-1",
			epoch_id="ep-1",
			status="completed",
			summary="Redis is 2x faster",
			discoveries=["Redis approach is faster with lower latency"],
		)
		db_runtime.insert_handoff(handoff)

		completion = WorkerCompletion(
			unit=unit,
			handoff=handoff,
			workspace="/tmp/ws",
			epoch=epoch,
		)

		# Put completion on queue and process
		controller._completion_queue.put_nowait(completion)
		controller.running = False  # Process just one item

		result = MagicMock()
		result.objective_met = False

		asyncio.run(
			controller._process_completions(mission, result)
		)

		# merge_unit should NOT have been called
		controller._green_branch.merge_unit.assert_not_called()

	def test_experiment_result_stored_in_db(self, db_runtime: Database) -> None:
		"""ExperimentResult should be inserted into DB for completed experiment units."""
		from mission_control.continuous_controller import WorkerCompletion

		controller = self._make_controller(db_runtime)
		mission = Mission(id="m-2", objective="test")
		db_runtime.insert_mission(mission)

		epoch = Epoch(id="ep-2", mission_id="m-2")
		db_runtime.insert_epoch(epoch)

		self._insert_plan(db_runtime, "p-2")
		unit = WorkUnit(
			id="wu-exp-2",
			plan_id="p-2",
			title="Benchmark serializers",
			description="JSON vs MessagePack",
			unit_type="experiment",
			experiment_mode=True,
			status="completed",
			finished_at=_now_iso(),
		)
		db_runtime.insert_work_unit(unit)

		handoff = Handoff(
			id="h-2",
			work_unit_id="wu-exp-2",
			epoch_id="ep-2",
			status="completed",
			summary="MessagePack is 3x faster for binary data",
			discoveries=["approach A: JSON - 100ms avg", "approach B: msgpack - 33ms avg"],
		)
		db_runtime.insert_handoff(handoff)

		completion = WorkerCompletion(
			unit=unit,
			handoff=handoff,
			workspace="/tmp/ws",
			epoch=epoch,
		)

		controller._completion_queue.put_nowait(completion)
		controller.running = False

		result = MagicMock()
		asyncio.run(
			controller._process_completions(mission, result)
		)

		# Check experiment_result was stored
		results = db_runtime.get_experiment_results_for_mission("m-2")
		assert len(results) == 1
		assert results[0].work_unit_id == "wu-exp-2"
		assert results[0].mission_id == "m-2"
		assert results[0].epoch_id == "ep-2"

	def test_failed_experiment_increments_failed_counter(self, db_runtime: Database) -> None:
		"""Failed experiment units should increment _total_failed."""
		from mission_control.continuous_controller import WorkerCompletion

		controller = self._make_controller(db_runtime)
		mission = Mission(id="m-6", objective="test")
		db_runtime.insert_mission(mission)

		epoch = Epoch(id="ep-6", mission_id="m-6")
		db_runtime.insert_epoch(epoch)

		self._insert_plan(db_runtime, "p-6")
		unit = WorkUnit(
			id="wu-exp-6",
			plan_id="p-6",
			title="Failed experiment",
			description="Could not complete",
			unit_type="experiment",
			experiment_mode=True,
			status="failed",
			finished_at=_now_iso(),
		)
		db_runtime.insert_work_unit(unit)

		handoff = Handoff(
			id="h-6",
			work_unit_id="wu-exp-6",
			epoch_id="ep-6",
			status="failed",
			summary="Timed out",
		)
		db_runtime.insert_handoff(handoff)

		completion = WorkerCompletion(
			unit=unit,
			handoff=handoff,
			workspace="/tmp/ws",
			epoch=epoch,
		)

		controller._completion_queue.put_nowait(completion)
		controller.running = False

		result = MagicMock()
		asyncio.run(
			controller._process_completions(mission, result)
		)

		assert controller._total_failed == 1
		assert controller._total_merged == 0
		# merge_unit should NOT be called even for failed experiment units
		controller._green_branch.merge_unit.assert_not_called()
