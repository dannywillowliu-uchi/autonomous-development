"""Tests for causal outcome attribution: signal model, DB persistence, and risk computation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_control.causal import CausalAttributor, CausalSignal
from mission_control.db import Database
from mission_control.models import WorkUnit


@pytest.fixture
def db(tmp_path: Path) -> Database:
	d = Database(tmp_path / "test.db")
	yield d
	d.close()


class TestCausalSignalDefaults:
	def test_default_fields(self) -> None:
		s = CausalSignal()
		assert len(s.id) == 12
		assert s.work_unit_id == ""
		assert s.outcome == ""
		assert s.unit_type == "implementation"
		assert s.has_dependencies is False
		assert s.has_overlap is False

	def test_id_generation_unique(self) -> None:
		s1 = CausalSignal()
		s2 = CausalSignal()
		assert s1.id != s2.id


class TestCausalSignalDB:
	def test_insert_and_retrieve(self, db: Database) -> None:
		# Insert prerequisite records for FK constraints
		from mission_control.models import Mission, Plan
		mission = Mission(id="m-1", objective="test")
		db.insert_mission(mission)
		plan = Plan(id="p-1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(id="wu-1", plan_id="p-1", title="test unit")
		db.insert_work_unit(unit)

		signal = CausalSignal(
			id="cs-1",
			work_unit_id="wu-1",
			mission_id="m-1",
			epoch_id="ep-1",
			specialist="test-writer",
			model="claude-sonnet",
			file_count=3,
			has_dependencies=True,
			attempt=1,
			unit_type="implementation",
			epoch_size=10,
			concurrent_units=4,
			has_overlap=True,
			outcome="merged",
			failure_stage="",
		)
		db.insert_causal_signal(signal)

		results = db.get_causal_signals_for_mission("m-1")
		assert len(results) == 1
		r = results[0]
		assert r.id == "cs-1"
		assert r.specialist == "test-writer"
		assert r.model == "claude-sonnet"
		assert r.file_count == 3
		assert r.has_dependencies is True
		assert r.has_overlap is True
		assert r.outcome == "merged"
		assert r.epoch_size == 10
		assert r.concurrent_units == 4

	def test_get_signals_filters_by_mission(self, db: Database) -> None:
		from mission_control.models import Mission, Plan
		for mid in ["m-1", "m-2"]:
			db.insert_mission(Mission(id=mid, objective="test"))
		plan = Plan(id="p-1", objective="test")
		db.insert_plan(plan)
		for i in range(3):
			uid = f"wu-{i}"
			db.insert_work_unit(WorkUnit(id=uid, plan_id="p-1", title=f"unit {i}"))
			db.insert_causal_signal(CausalSignal(
				id=f"cs-{i}", work_unit_id=uid,
				mission_id="m-1" if i < 2 else "m-2",
				outcome="merged",
			))
		assert len(db.get_causal_signals_for_mission("m-1")) == 2
		assert len(db.get_causal_signals_for_mission("m-2")) == 1


class TestCountCausalOutcomes:
	def _seed_signals(self, db: Database, outcomes: list[tuple[str, str]]) -> None:
		from mission_control.models import Mission, Plan
		db.insert_mission(Mission(id="m-1", objective="test"))
		db.insert_plan(Plan(id="p-1", objective="test"))
		for i, (specialist, outcome) in enumerate(outcomes):
			uid = f"wu-{i}"
			db.insert_work_unit(WorkUnit(id=uid, plan_id="p-1", title=f"unit {i}"))
			db.insert_causal_signal(CausalSignal(
				id=f"cs-{i}", work_unit_id=uid, mission_id="m-1",
				specialist=specialist, outcome=outcome,
			))

	def test_basic_counts(self, db: Database) -> None:
		self._seed_signals(db, [
			("test-writer", "merged"),
			("test-writer", "merged"),
			("test-writer", "failed"),
			("refactorer", "merged"),
		])
		counts = db.count_causal_outcomes("specialist", "test-writer")
		assert counts["merged"] == 2
		assert counts["failed"] == 1

	def test_no_data_returns_empty(self, db: Database) -> None:
		counts = db.count_causal_outcomes("specialist", "nonexistent")
		assert counts == {}

	def test_bucketed_file_count(self, db: Database) -> None:
		from mission_control.models import Mission, Plan
		db.insert_mission(Mission(id="m-1", objective="test"))
		db.insert_plan(Plan(id="p-1", objective="test"))
		# Insert signals with various file counts
		file_counts = [1, 2, 3, 5, 7, 8]
		outcomes = ["merged", "merged", "failed", "merged", "failed", "failed"]
		for i, (fc, outcome) in enumerate(zip(file_counts, outcomes)):
			uid = f"wu-{i}"
			db.insert_work_unit(WorkUnit(id=uid, plan_id="p-1", title=f"unit {i}"))
			db.insert_causal_signal(CausalSignal(
				id=f"cs-{i}", work_unit_id=uid, mission_id="m-1",
				file_count=fc, outcome=outcome,
			))
		# Bucket "2-3" should have file_count 2 and 3
		counts_23 = db.count_causal_outcomes_bucketed("2-3")
		assert counts_23.get("merged", 0) == 1
		assert counts_23.get("failed", 0) == 1
		# Bucket "6+" should have file_count 7 and 8
		counts_6p = db.count_causal_outcomes_bucketed("6+")
		assert counts_6p.get("failed", 0) == 2
		# Invalid bucket
		assert db.count_causal_outcomes_bucketed("invalid") == {}


class TestPFailure:
	def _seed(self, db: Database, specialist: str, merged: int, failed: int) -> None:
		from mission_control.models import Mission, Plan
		try:
			db.insert_mission(Mission(id="m-1", objective="test"))
		except Exception:
			pass
		try:
			db.insert_plan(Plan(id="p-1", objective="test"))
		except Exception:
			pass
		idx = 0
		for _ in range(merged):
			uid = f"wu-m-{specialist}-{idx}"
			db.insert_work_unit(WorkUnit(id=uid, plan_id="p-1", title="t"))
			db.insert_causal_signal(CausalSignal(
				id=f"cs-m-{specialist}-{idx}", work_unit_id=uid, mission_id="m-1",
				specialist=specialist, outcome="merged",
			))
			idx += 1
		for _ in range(failed):
			uid = f"wu-f-{specialist}-{idx}"
			db.insert_work_unit(WorkUnit(id=uid, plan_id="p-1", title="t"))
			db.insert_causal_signal(CausalSignal(
				id=f"cs-f-{specialist}-{idx}", work_unit_id=uid, mission_id="m-1",
				specialist=specialist, outcome="failed",
			))
			idx += 1

	def test_basic_probability(self, db: Database) -> None:
		self._seed(db, "test-writer", merged=7, failed=3)
		attr = CausalAttributor(db)
		p = attr.p_failure("specialist", "test-writer")
		assert p is not None
		assert abs(p - 0.3) < 0.01

	def test_min_samples_threshold(self, db: Database) -> None:
		self._seed(db, "test-writer", merged=2, failed=1)
		attr = CausalAttributor(db)
		# Only 3 samples, below default min_samples=5
		p = attr.p_failure("specialist", "test-writer")
		assert p is None

	def test_all_merged(self, db: Database) -> None:
		self._seed(db, "refactorer", merged=10, failed=0)
		attr = CausalAttributor(db)
		p = attr.p_failure("specialist", "refactorer")
		assert p is not None
		assert p == 0.0

	def test_all_failed(self, db: Database) -> None:
		self._seed(db, "debugger", merged=0, failed=6)
		attr = CausalAttributor(db)
		p = attr.p_failure("specialist", "debugger")
		assert p is not None
		assert p == 1.0

	def test_custom_min_samples(self, db: Database) -> None:
		self._seed(db, "test-writer", merged=2, failed=1)
		attr = CausalAttributor(db)
		p = attr.p_failure("specialist", "test-writer", min_samples=3)
		assert p is not None
		assert abs(p - 1 / 3) < 0.01


class TestTopRiskFactors:
	def _seed_many(self, db: Database) -> None:
		from mission_control.models import Mission, Plan
		db.insert_mission(Mission(id="m-1", objective="test"))
		db.insert_plan(Plan(id="p-1", objective="test"))
		idx = 0
		# test-writer: 3/10 failed (30%)
		for outcome in (["merged"] * 7 + ["failed"] * 3):
			uid = f"wu-tw-{idx}"
			db.insert_work_unit(WorkUnit(id=uid, plan_id="p-1", title="t"))
			db.insert_causal_signal(CausalSignal(
				id=f"cs-tw-{idx}", work_unit_id=uid, mission_id="m-1",
				specialist="test-writer", outcome=outcome, unit_type="implementation",
			))
			idx += 1
		# implementation type: 4/10 failed (40%)
		for outcome in (["merged"] * 6 + ["failed"] * 4):
			uid = f"wu-impl-{idx}"
			db.insert_work_unit(WorkUnit(id=uid, plan_id="p-1", title="t"))
			db.insert_causal_signal(CausalSignal(
				id=f"cs-impl-{idx}", work_unit_id=uid, mission_id="m-1",
				specialist="", unit_type="implementation", outcome=outcome,
			))
			idx += 1

	def test_sorted_by_risk(self, db: Database) -> None:
		self._seed_many(db)
		attr = CausalAttributor(db)
		unit = WorkUnit(
			specialist="test-writer", unit_type="implementation",
			files_hint="a.py,b.py",
		)
		risks = attr.top_risk_factors(unit)
		assert len(risks) > 0
		# Should be sorted descending by p_failure
		for i in range(len(risks) - 1):
			assert risks[i][1] >= risks[i + 1][1]

	def test_limited_results(self, db: Database) -> None:
		self._seed_many(db)
		attr = CausalAttributor(db)
		unit = WorkUnit(
			specialist="test-writer", unit_type="implementation",
			files_hint="a.py,b.py",
		)
		risks = attr.top_risk_factors(unit, limit=1)
		assert len(risks) <= 1

	def test_empty_history(self, db: Database) -> None:
		attr = CausalAttributor(db)
		unit = WorkUnit(specialist="test-writer", unit_type="implementation")
		risks = attr.top_risk_factors(unit)
		assert risks == []


class TestBucketFileCount:
	def test_zero_files(self) -> None:
		unit = WorkUnit(files_hint="")
		assert CausalAttributor._bucket_file_count(unit) == "1"

	def test_one_file(self) -> None:
		unit = WorkUnit(files_hint="a.py")
		assert CausalAttributor._bucket_file_count(unit) == "1"

	def test_two_files(self) -> None:
		unit = WorkUnit(files_hint="a.py,b.py")
		assert CausalAttributor._bucket_file_count(unit) == "2-3"

	def test_three_files(self) -> None:
		unit = WorkUnit(files_hint="a.py,b.py,c.py")
		assert CausalAttributor._bucket_file_count(unit) == "2-3"

	def test_five_files(self) -> None:
		unit = WorkUnit(files_hint="a.py,b.py,c.py,d.py,e.py")
		assert CausalAttributor._bucket_file_count(unit) == "4-5"

	def test_many_files(self) -> None:
		unit = WorkUnit(files_hint="a.py,b.py,c.py,d.py,e.py,f.py,g.py")
		assert CausalAttributor._bucket_file_count(unit) == "6+"


class TestFormatRiskSection:
	def test_empty_risks(self) -> None:
		assert CausalAttributor.format_risk_section([]) == ""

	def test_formatted_output(self) -> None:
		risks = [
			("specialist=test-writer", 0.30),
			("file_count=6+", 0.50),
		]
		output = CausalAttributor.format_risk_section(risks)
		assert "## Causal Risk Factors" in output
		assert "specialist=test-writer: 30% historical failure rate" in output
		assert "file_count=6+: 50% historical failure rate" in output

	def test_single_risk(self) -> None:
		output = CausalAttributor.format_risk_section([("model=gpt-4", 0.25)])
		assert "model=gpt-4: 25% historical failure rate" in output


class TestSignalFromWorkUnit:
	def test_signal_construction_from_unit(self) -> None:
		unit = WorkUnit(
			id="wu-1",
			specialist="test-writer",
			unit_type="implementation",
			files_hint="a.py,b.py,c.py",
			depends_on="wu-0",
			attempt=2,
		)
		signal = CausalSignal(
			work_unit_id=unit.id,
			mission_id="m-1",
			epoch_id="ep-1",
			specialist=unit.specialist,
			model="claude-sonnet",
			file_count=len(unit.files_hint.split(",")) if unit.files_hint else 0,
			has_dependencies=bool(unit.depends_on),
			attempt=unit.attempt,
			unit_type=unit.unit_type,
			epoch_size=10,
			concurrent_units=3,
			has_overlap=False,
			outcome="merged",
		)
		assert signal.work_unit_id == "wu-1"
		assert signal.specialist == "test-writer"
		assert signal.file_count == 3
		assert signal.has_dependencies is True
		assert signal.attempt == 2
		assert signal.outcome == "merged"


class TestCausalAttributorRecord:
	def test_record_persists_signal(self, db: Database) -> None:
		from mission_control.models import Mission, Plan
		db.insert_mission(Mission(id="m-1", objective="test"))
		db.insert_plan(Plan(id="p-1", objective="test"))
		db.insert_work_unit(WorkUnit(id="wu-1", plan_id="p-1", title="t"))

		attr = CausalAttributor(db)
		signal = CausalSignal(
			id="cs-rec-1", work_unit_id="wu-1", mission_id="m-1",
			outcome="failed", failure_stage="merge",
		)
		attr.record(signal)

		results = db.get_causal_signals_for_mission("m-1")
		assert len(results) == 1
		assert results[0].outcome == "failed"
		assert results[0].failure_stage == "merge"

	def test_record_handles_error_gracefully(self, db: Database) -> None:
		"""Recording errors should be swallowed (non-critical path)."""
		attr = CausalAttributor(db)
		# Signal with non-existent FK -- depending on DB enforcement this may fail
		# but should not raise
		signal = CausalSignal(
			id="cs-bad", work_unit_id="nonexistent", mission_id="nonexistent",
			outcome="merged",
		)
		# Should not raise even if insert fails
		attr.record(signal)
