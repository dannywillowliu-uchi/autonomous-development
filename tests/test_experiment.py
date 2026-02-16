"""Tests for experiment mode: CLI flag, DB operations, model, prompt, and controller skip."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mission_control.cli import build_parser
from mission_control.db import Database
from mission_control.models import Epoch, ExperimentResult, Handoff, Mission, Plan, WorkUnit, _now_iso
from mission_control.worker import render_mission_worker_prompt

# -- ExperimentResult dataclass tests --


class TestExperimentResultDataclass:
	def test_default_fields(self) -> None:
		r = ExperimentResult()
		assert r.work_unit_id == ""
		assert r.epoch_id is None
		assert r.mission_id == ""
		assert r.approach_count == 2
		assert r.comparison_report == ""
		assert r.recommended_approach == ""
		assert r.id  # auto-generated
		assert r.timestamp  # auto-generated
		assert r.created_at  # auto-generated

	def test_custom_fields(self) -> None:
		r = ExperimentResult(
			id="exp-1",
			work_unit_id="wu-1",
			epoch_id="ep-1",
			mission_id="m-1",
			approach_count=3,
			comparison_report='{"winner": "A"}',
			recommended_approach="Approach A",
		)
		assert r.id == "exp-1"
		assert r.work_unit_id == "wu-1"
		assert r.epoch_id == "ep-1"
		assert r.mission_id == "m-1"
		assert r.approach_count == 3
		assert r.comparison_report == '{"winner": "A"}'
		assert r.recommended_approach == "Approach A"

	def test_multiple_instances_unique_ids(self) -> None:
		r1 = ExperimentResult()
		r2 = ExperimentResult()
		assert r1.id != r2.id


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

	def test_get_nonexistent(self, db: Database) -> None:
		assert db.get_experiment_result("nope") is None

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

	def test_get_results_for_mission_empty(self, db: Database) -> None:
		results = db.get_experiment_results_for_mission("no-mission")
		assert results == []


# -- WorkUnit experiment_mode persistence --


class TestWorkUnitExperimentMode:
	def test_workunit_experiment_mode_default(self) -> None:
		unit = WorkUnit()
		assert unit.experiment_mode is False

	def test_workunit_experiment_mode_true(self) -> None:
		unit = WorkUnit(experiment_mode=True)
		assert unit.experiment_mode is True

	def test_workunit_experiment_mode_persists(self, db: Database) -> None:
		from mission_control.models import Plan
		plan = Plan(id="plan-exp", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(
			id="wu-exp",
			plan_id=plan.id,
			title="experiment unit",
			experiment_mode=True,
		)
		db.insert_work_unit(unit)
		fetched = db.get_work_unit("wu-exp")
		assert fetched is not None
		assert fetched.experiment_mode is True

	def test_workunit_experiment_mode_false_persists(self, db: Database) -> None:
		from mission_control.models import Plan
		plan = Plan(id="plan-normal", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(
			id="wu-normal",
			plan_id=plan.id,
			title="normal unit",
			experiment_mode=False,
		)
		db.insert_work_unit(unit)
		fetched = db.get_work_unit("wu-normal")
		assert fetched is not None
		assert fetched.experiment_mode is False


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

	def test_render_mission_prompt_normal_unit(self) -> None:
		"""Normal implementation units render with the mission template."""
		from mission_control.worker import render_mission_worker_prompt

		config = MagicMock()
		config.target.name = "test-project"
		config.target.verification.command = "pytest"

		unit = WorkUnit(
			title="Add feature X",
			description="Implement feature",
			unit_type="implementation",
			experiment_mode=False,
		)
		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/ws",
			branch_name="mc/unit-test",
		)
		assert "Add feature X" in prompt
		assert "Implement feature" in prompt


# -- Experiment units skip merge in _process_completions --


class TestExperimentSkipMerge:
	def test_research_units_skip_merge(self) -> None:
		"""Verify that research-type units (used for experiments) skip merge logic."""
		unit = WorkUnit(
			unit_type="research",
			experiment_mode=True,
		)
		# The controller checks unit.unit_type == "research" to skip merge
		assert unit.unit_type == "research"
		assert unit.experiment_mode is True

	def test_implementation_units_do_not_skip_merge(self) -> None:
		unit = WorkUnit(
			unit_type="implementation",
			experiment_mode=False,
		)
		assert unit.unit_type != "research"


# -- CLI parser tests --


class TestExperimentCLIFlag:
	def test_experiment_flag_exists(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission"])
		assert hasattr(args, "experiment")
		assert args.experiment is False

	def test_experiment_flag_set(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--experiment"])
		assert args.experiment is True

	def test_experiment_with_other_flags(self) -> None:
		parser = build_parser()
		args = parser.parse_args([
			"mission", "--experiment", "--dry-run", "--workers", "4",
		])
		assert args.experiment is True
		assert args.dry_run is True
		assert args.workers == 4

	def test_experiment_and_strategist_flags(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--experiment", "--strategist"])
		assert args.experiment is True
		assert args.strategist is True


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

	def test_research_unit_selects_research_template(self) -> None:
		"""unit_type='research' should use RESEARCH_WORKER_PROMPT_TEMPLATE."""
		config = self._make_config()
		unit = WorkUnit(
			title="Investigate API patterns",
			description="Read the codebase",
			unit_type="research",
		)
		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/ws",
			branch_name="mc/unit-test",
		)
		assert "research agent" in prompt.lower()
		assert "Research Task" in prompt

	def test_implementation_unit_selects_mission_template(self) -> None:
		"""unit_type='implementation' should use MISSION_WORKER_PROMPT_TEMPLATE."""
		config = self._make_config()
		unit = WorkUnit(
			title="Add feature X",
			description="Implement feature",
			unit_type="implementation",
		)
		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/ws",
			branch_name="mc/unit-test",
		)
		assert "Constraints" in prompt
		assert "No TODOs" in prompt

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

	def test_experiment_template_instructs_no_commits(self) -> None:
		"""Experiment prompt should tell worker not to commit."""
		config = self._make_config()
		unit = WorkUnit(
			title="Test approaches",
			description="Compare two implementations",
			unit_type="experiment",
			experiment_mode=True,
		)
		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/ws",
			branch_name="mc/unit-test",
		)
		assert "Do NOT commit" in prompt

	def test_experiment_prompt_includes_experience_context(self) -> None:
		"""Experience context should be included in experiment prompts."""
		config = self._make_config()
		unit = WorkUnit(
			title="Experiment task",
			description="Compare approaches",
			unit_type="experiment",
			experiment_mode=True,
		)
		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/ws",
			branch_name="mc/unit-test",
			experience_context="Previous experiment showed approach A is 2x faster",
		)
		assert "Previous experiment showed approach A is 2x faster" in prompt

	def test_experiment_prompt_includes_mission_state(self) -> None:
		"""Mission state should be included in experiment prompts."""
		config = self._make_config()
		unit = WorkUnit(
			title="Experiment task",
			description="Compare approaches",
			unit_type="experiment",
			experiment_mode=True,
		)
		prompt = render_mission_worker_prompt(
			unit=unit,
			config=config,
			workspace_path="/tmp/ws",
			branch_name="mc/unit-test",
			mission_state="3 units completed, 1 failed",
		)
		assert "3 units completed, 1 failed" in prompt


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
		controller._state_changelog = []
		controller._backlog_item_ids = []
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
			discoveries=json.dumps(["Redis approach is faster with lower latency"]),
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
			discoveries=json.dumps(["approach A: JSON - 100ms avg", "approach B: msgpack - 33ms avg"]),
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

	def test_experiment_completed_event_logged(self, db_runtime: Database) -> None:
		"""An 'experiment_completed' UnitEvent should be logged."""
		from mission_control.continuous_controller import WorkerCompletion

		controller = self._make_controller(db_runtime)
		mission = Mission(id="m-3", objective="test")
		db_runtime.insert_mission(mission)

		epoch = Epoch(id="ep-3", mission_id="m-3")
		db_runtime.insert_epoch(epoch)

		self._insert_plan(db_runtime, "p-3")
		unit = WorkUnit(
			id="wu-exp-3",
			plan_id="p-3",
			title="Compare algorithms",
			description="Quick sort vs merge sort",
			unit_type="experiment",
			experiment_mode=True,
			status="completed",
			finished_at=_now_iso(),
		)
		db_runtime.insert_work_unit(unit)

		handoff = Handoff(
			id="h-3",
			work_unit_id="wu-exp-3",
			epoch_id="ep-3",
			status="completed",
			summary="Quick sort faster for small arrays",
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

		# Check event logged
		events = db_runtime.conn.execute(
			"SELECT * FROM unit_events WHERE work_unit_id = ? AND event_type = 'experiment_completed'",
			("wu-exp-3",),
		).fetchall()
		assert len(events) == 1

	def test_experiment_handoff_fed_to_planner(self, db_runtime: Database) -> None:
		"""Handoff from experiment units should be ingested by the planner."""
		from mission_control.continuous_controller import WorkerCompletion

		controller = self._make_controller(db_runtime)
		mission = Mission(id="m-4", objective="test")
		db_runtime.insert_mission(mission)

		epoch = Epoch(id="ep-4", mission_id="m-4")
		db_runtime.insert_epoch(epoch)

		self._insert_plan(db_runtime, "p-4")
		unit = WorkUnit(
			id="wu-exp-4",
			plan_id="p-4",
			title="Test DB engines",
			description="PostgreSQL vs MySQL",
			unit_type="experiment",
			experiment_mode=True,
			status="completed",
			finished_at=_now_iso(),
		)
		db_runtime.insert_work_unit(unit)

		handoff = Handoff(
			id="h-4",
			work_unit_id="wu-exp-4",
			epoch_id="ep-4",
			status="completed",
			summary="PostgreSQL handles complex queries better",
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

		# Planner should have received the handoff
		controller._planner.ingest_handoff.assert_called_once_with(handoff)

	def test_experiment_increments_merged_counter(self, db_runtime: Database) -> None:
		"""Completed experiment units should increment _total_merged (counted as successful)."""
		from mission_control.continuous_controller import WorkerCompletion

		controller = self._make_controller(db_runtime)
		mission = Mission(id="m-5", objective="test")
		db_runtime.insert_mission(mission)

		epoch = Epoch(id="ep-5", mission_id="m-5")
		db_runtime.insert_epoch(epoch)

		self._insert_plan(db_runtime, "p-5")
		unit = WorkUnit(
			id="wu-exp-5",
			plan_id="p-5",
			title="Test experiment",
			description="A vs B",
			unit_type="experiment",
			experiment_mode=True,
			status="completed",
			finished_at=_now_iso(),
		)
		db_runtime.insert_work_unit(unit)

		handoff = Handoff(
			id="h-5",
			work_unit_id="wu-exp-5",
			epoch_id="ep-5",
			status="completed",
			summary="A is better",
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

		assert controller._total_merged == 1
		assert controller._total_failed == 0

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

	def test_experiment_adds_state_changelog_entry(self, db_runtime: Database) -> None:
		"""Completed experiment should add an entry to _state_changelog."""
		from mission_control.continuous_controller import WorkerCompletion

		controller = self._make_controller(db_runtime)
		mission = Mission(id="m-7", objective="test")
		db_runtime.insert_mission(mission)

		epoch = Epoch(id="ep-7", mission_id="m-7")
		db_runtime.insert_epoch(epoch)

		self._insert_plan(db_runtime, "p-7")
		unit = WorkUnit(
			id="wu-exp-7abc",
			plan_id="p-7",
			title="Changelog test",
			description="Test changelog",
			unit_type="experiment",
			experiment_mode=True,
			status="completed",
			finished_at=_now_iso(),
		)
		db_runtime.insert_work_unit(unit)

		handoff = Handoff(
			id="h-7",
			work_unit_id="wu-exp-7abc",
			epoch_id="ep-7",
			status="completed",
			summary="Experiment completed successfully",
		)
		db_runtime.insert_handoff(handoff)

		# Mock _update_mission_state to avoid file I/O
		controller._update_mission_state = MagicMock()

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

		assert len(controller._state_changelog) == 1
		assert "experiment completed" in controller._state_changelog[0]
		assert "wu-exp-7" in controller._state_changelog[0]
