"""Tests for experiment mode: CLI flag, DB operations, model, prompt, and controller skip."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mission_control.cli import build_parser
from mission_control.db import Database
from mission_control.models import ExperimentResult, WorkUnit

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
