"""Tests for BatchAnalyzer heuristic signal detection."""

from __future__ import annotations

import json

from mission_control.batch_analyzer import BatchAnalyzer, format_failure_stages
from mission_control.db import Database
from mission_control.models import (
	Epoch,
	Handoff,
	KnowledgeItem,
	Mission,
	Plan,
	UnitEvent,
	WorkUnit,
)


def _setup_mission(db: Database) -> str:
	"""Insert a mission + epoch + plan and return mission_id."""
	mid = "m1"
	db.insert_mission(Mission(id=mid, objective="test"))
	db.insert_epoch(Epoch(id="ep1", mission_id=mid, number=1))
	db.insert_plan(Plan(id="p1", objective="test"))
	return mid


class TestComputeHotspots:
	def test_no_handoffs_empty(self, db: Database) -> None:
		mid = _setup_mission(db)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.file_hotspots == []

	def test_file_touched_three_times(self, db: Database) -> None:
		mid = _setup_mission(db)
		for i in range(3):
			wu = WorkUnit(
				id=f"wu{i}", plan_id="p1", title=f"Task {i}",
				epoch_id="ep1", status="completed",
			)
			db.insert_work_unit(wu)
			db.insert_handoff(Handoff(
				work_unit_id=f"wu{i}", epoch_id="ep1",
				status="completed", files_changed=["src/hot.py"],
			))

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert len(signals.file_hotspots) == 1
		assert signals.file_hotspots[0][0] == "src/hot.py"
		assert signals.file_hotspots[0][1] == 3

	def test_file_below_threshold_excluded(self, db: Database) -> None:
		mid = _setup_mission(db)
		for i in range(2):
			wu = WorkUnit(
				id=f"wu{i}", plan_id="p1", title=f"Task {i}",
				epoch_id="ep1", status="completed",
			)
			db.insert_work_unit(wu)
			db.insert_handoff(Handoff(
				work_unit_id=f"wu{i}", epoch_id="ep1",
				status="completed", files_changed=["src/cool.py"],
			))

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.file_hotspots == []


class TestFailureClustering:
	def test_clusters_by_concern(self, db: Database) -> None:
		mid = _setup_mission(db)
		for i in range(3):
			wu = WorkUnit(
				id=f"wu{i}", plan_id="p1", title=f"Task {i}",
				epoch_id="ep1", status="failed",
			)
			db.insert_work_unit(wu)
			db.insert_handoff(Handoff(
				work_unit_id=f"wu{i}", epoch_id="ep1",
				status="failed", concerns=["Import error in auth module"],
			))

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert len(signals.failure_clusters) >= 1
		assert "Import error in auth module" in signals.failure_clusters

	def test_no_failures_empty(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			epoch_id="ep1", status="completed",
		)
		db.insert_work_unit(wu)
		db.insert_handoff(Handoff(
			work_unit_id="wu1", epoch_id="ep1",
			status="completed",
		))

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_clusters == {}


class TestStalledDetection:
	def test_area_attempted_twice_no_success(self, db: Database) -> None:
		mid = _setup_mission(db)
		for i in range(2):
			wu = WorkUnit(
				id=f"wu{i}", plan_id="p1", title="Fix auth",
				epoch_id="ep1", status="failed",
				files_hint="src/auth.py",
			)
			db.insert_work_unit(wu)

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert "src/auth.py" in signals.stalled_areas

	def test_area_eventually_succeeded(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu1 = WorkUnit(
			id="wu1", plan_id="p1", title="Fix auth",
			epoch_id="ep1", status="failed",
			files_hint="src/auth.py",
		)
		wu2 = WorkUnit(
			id="wu2", plan_id="p1", title="Fix auth",
			epoch_id="ep1", status="completed",
			files_hint="src/auth.py",
		)
		db.insert_work_unit(wu1)
		db.insert_work_unit(wu2)

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert "src/auth.py" not in signals.stalled_areas

	def test_single_attempt_not_stalled(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(
			id="wu1", plan_id="p1", title="Fix auth",
			epoch_id="ep1", status="failed",
			files_hint="src/auth.py",
		)
		db.insert_work_unit(wu)

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.stalled_areas == []


class TestEffortDistribution:
	def test_effort_percentages(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu1 = WorkUnit(
			id="wu1", plan_id="p1", title="Auth",
			epoch_id="ep1", status="completed",
			files_hint="src/auth.py", cost_usd=3.0,
		)
		wu2 = WorkUnit(
			id="wu2", plan_id="p1", title="Tests",
			epoch_id="ep1", status="completed",
			files_hint="tests/", cost_usd=1.0,
		)
		db.insert_work_unit(wu1)
		db.insert_work_unit(wu2)

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.effort_distribution["src/auth.py"] == 0.75
		assert signals.effort_distribution["tests/"] == 0.25

	def test_zero_cost_no_division_error(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			epoch_id="ep1", status="completed",
			cost_usd=0.0,
		)
		db.insert_work_unit(wu)

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert isinstance(signals.effort_distribution, dict)


class TestRetryDepth:
	def test_retried_units_tracked(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(
			id="wu1", plan_id="p1", title="Retry task",
			epoch_id="ep1", status="completed", attempt=2,
		)
		db.insert_work_unit(wu)

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.retry_depth == {"wu1": 2}

	def test_no_retries_empty(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(
			id="wu1", plan_id="p1", title="First try",
			epoch_id="ep1", status="completed", attempt=0,
		)
		db.insert_work_unit(wu)

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.retry_depth == {}


class TestKnowledgeGaps:
	def test_gap_between_research_and_implementation(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(
			id="wu1", plan_id="p1", title="Implement auth",
			epoch_id="ep1", status="completed",
			unit_type="implementation", files_hint="src/auth.py",
		)
		db.insert_work_unit(wu)
		# Research only covers "api" scope, not "src/auth.py"
		ki = KnowledgeItem(
			mission_id=mid, source_unit_id="r1",
			source_unit_type="research", title="API research",
			content="Found patterns", scope="api",
		)
		db.insert_knowledge_item(ki)

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert "src/auth.py" in signals.knowledge_gaps

	def test_no_gap_when_research_covers_area(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(
			id="wu1", plan_id="p1", title="Implement auth",
			epoch_id="ep1", status="completed",
			unit_type="implementation", files_hint="src/auth.py",
		)
		db.insert_work_unit(wu)
		ki = KnowledgeItem(
			mission_id=mid, source_unit_id="r1",
			source_unit_type="research", title="Auth research",
			content="Found patterns", scope="src/auth.py",
		)
		db.insert_knowledge_item(ki)

		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert "src/auth.py" not in signals.knowledge_gaps


def _insert_merge_failed_event(
	db: Database, work_unit_id: str, failure_stage: str, mission_id: str = "m1",
) -> None:
	"""Helper to insert a merge_failed unit event with a failure_stage in details."""
	wu = WorkUnit(
		id=work_unit_id, plan_id="p1", title="Task",
		epoch_id="ep1", status="failed",
	)
	db.insert_work_unit(wu)
	db.insert_unit_event(UnitEvent(
		mission_id=mission_id,
		epoch_id="ep1",
		work_unit_id=work_unit_id,
		event_type="merge_failed",
		details=json.dumps({"failure_stage": failure_stage}),
	))


class TestFailureStages:
	def test_acceptance_criteria_stage(self, db: Database) -> None:
		mid = _setup_mission(db)
		_insert_merge_failed_event(db, "wu1", "acceptance_criteria", mid)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {"acceptance_criteria": 1}

	def test_verification_stage(self, db: Database) -> None:
		mid = _setup_mission(db)
		_insert_merge_failed_event(db, "wu1", "pre_merge_verification", mid)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {"verification": 1}

	def test_merge_conflict_stage(self, db: Database) -> None:
		mid = _setup_mission(db)
		_insert_merge_failed_event(db, "wu1", "merge_conflict", mid)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {"merge_conflict": 1}

	def test_timeout_stage(self, db: Database) -> None:
		mid = _setup_mission(db)
		_insert_merge_failed_event(db, "wu1", "timeout", mid)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {"timeout": 1}

	def test_infrastructure_stage_from_fetch(self, db: Database) -> None:
		mid = _setup_mission(db)
		_insert_merge_failed_event(db, "wu1", "fetch", mid)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {"infrastructure": 1}

	def test_infrastructure_stage_from_exception(self, db: Database) -> None:
		mid = _setup_mission(db)
		_insert_merge_failed_event(db, "wu1", "exception", mid)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {"infrastructure": 1}

	def test_infrastructure_stage_from_execution(self, db: Database) -> None:
		mid = _setup_mission(db)
		_insert_merge_failed_event(db, "wu1", "execution", mid)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {"infrastructure": 1}

	def test_multiple_stages_counted(self, db: Database) -> None:
		mid = _setup_mission(db)
		_insert_merge_failed_event(db, "wu1", "acceptance_criteria", mid)
		_insert_merge_failed_event(db, "wu2", "acceptance_criteria", mid)
		_insert_merge_failed_event(db, "wu3", "merge_conflict", mid)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages["acceptance_criteria"] == 2
		assert signals.failure_stages["merge_conflict"] == 1

	def test_no_merge_failed_events_empty(self, db: Database) -> None:
		mid = _setup_mission(db)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {}

	def test_non_merge_failed_events_ignored(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(id="wu1", plan_id="p1", title="Task", epoch_id="ep1", status="completed")
		db.insert_work_unit(wu)
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep1", work_unit_id="wu1",
			event_type="merged",
			details=json.dumps({"failure_stage": "acceptance_criteria"}),
		))
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {}

	def test_empty_details_json_skipped(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(id="wu1", plan_id="p1", title="Task", epoch_id="ep1", status="failed")
		db.insert_work_unit(wu)
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep1", work_unit_id="wu1",
			event_type="merge_failed", details="",
		))
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {}

	def test_malformed_details_json_skipped(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(id="wu1", plan_id="p1", title="Task", epoch_id="ep1", status="failed")
		db.insert_work_unit(wu)
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep1", work_unit_id="wu1",
			event_type="merge_failed", details="not valid json",
		))
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {}

	def test_post_merge_verification_normalized(self, db: Database) -> None:
		mid = _setup_mission(db)
		_insert_merge_failed_event(db, "wu1", "post_merge_verification", mid)
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze(mid)
		assert signals.failure_stages == {"verification": 1}


class TestFormatFailureStages:
	def test_empty_dict_returns_empty_string(self) -> None:
		assert format_failure_stages({}) == ""

	def test_single_stage(self) -> None:
		result = format_failure_stages({"acceptance_criteria": 3})
		assert "Failure breakdown (3 total):" in result
		assert "acceptance_criteria: 3" in result

	def test_multiple_stages_sorted_by_count(self) -> None:
		result = format_failure_stages({
			"merge_conflict": 1,
			"acceptance_criteria": 5,
			"timeout": 2,
		})
		lines = result.strip().split("\n")
		assert "8 total" in lines[0]
		assert "acceptance_criteria: 5" in lines[1]
		assert "timeout: 2" in lines[2]
		assert "merge_conflict: 1" in lines[3]


class TestAnalyzeWithDBErrors:
	def test_nonexistent_mission_returns_empty(self, db: Database) -> None:
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze("nonexistent")
		assert signals.file_hotspots == []
		assert signals.failure_clusters == {}
		assert signals.failure_stages == {}
		assert signals.stalled_areas == []
