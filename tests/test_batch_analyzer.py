"""Tests for BatchAnalyzer heuristic signal detection."""

from __future__ import annotations

import json

from mission_control.batch_analyzer import (
	BatchAnalyzer,
	EpochCostSummary,
	format_cost_trend,
	format_failure_stages,
)
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


class TestEpochCostSummary:
	def test_no_epochs_returns_empty(self, db: Database) -> None:
		_setup_mission(db)
		analyzer = BatchAnalyzer(db)
		# Mission has one epoch from _setup_mission but no events/units with cost data
		# Test with a mission that has no epochs
		db.insert_mission(Mission(id="m_empty", objective="empty"))
		result = analyzer.get_epoch_cost_summary("m_empty")
		assert result == []

	def test_single_epoch_with_units_and_events(self, db: Database) -> None:
		mid = _setup_mission(db)

		for i in range(3):
			wu = WorkUnit(
				id=f"cwu{i}", plan_id="p1", title=f"Cost task {i}",
				epoch_id="ep1", status="completed" if i < 2 else "failed",
				cost_usd=1.0, input_tokens=1000, output_tokens=500,
			)
			db.insert_work_unit(wu)
			db.insert_unit_event(UnitEvent(
				mission_id=mid, epoch_id="ep1", work_unit_id=f"cwu{i}",
				event_type="dispatched",
			))

		# Two merged, one failed
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep1", work_unit_id="cwu0",
			event_type="merged",
		))
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep1", work_unit_id="cwu1",
			event_type="merged",
		))
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep1", work_unit_id="cwu2",
			event_type="failed",
		))

		analyzer = BatchAnalyzer(db)
		result = analyzer.get_epoch_cost_summary(mid)
		assert len(result) == 1
		s = result[0]
		assert s.epoch_id == "ep1"
		assert s.total_cost_usd == 3.0  # 3 units * $1
		assert s.total_input_tokens == 3000
		assert s.total_output_tokens == 1500
		assert s.units_dispatched == 3
		assert s.units_merged == 2
		assert s.units_failed == 1
		assert s.avg_cost_per_unit == 1.0
		assert s.success_rate == round(2 / 3, 2)

	def test_multiple_epochs(self, db: Database) -> None:
		mid = _setup_mission(db)
		db.insert_epoch(Epoch(id="ep2", mission_id=mid, number=2))

		wu1 = WorkUnit(
			id="mwu1", plan_id="p1", title="E1 task",
			epoch_id="ep1", status="completed",
			cost_usd=2.0, input_tokens=500, output_tokens=200,
		)
		wu2 = WorkUnit(
			id="mwu2", plan_id="p1", title="E2 task",
			epoch_id="ep2", status="completed",
			cost_usd=4.0, input_tokens=800, output_tokens=400,
		)
		db.insert_work_unit(wu1)
		db.insert_work_unit(wu2)

		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep1", work_unit_id="mwu1",
			event_type="dispatched",
		))
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep1", work_unit_id="mwu1",
			event_type="merged",
		))
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep2", work_unit_id="mwu2",
			event_type="dispatched",
		))
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep2", work_unit_id="mwu2",
			event_type="merged",
		))

		analyzer = BatchAnalyzer(db)
		result = analyzer.get_epoch_cost_summary(mid)
		assert len(result) == 2
		assert result[0].epoch_id == "ep1"
		assert result[1].epoch_id == "ep2"
		assert result[0].total_cost_usd == 2.0
		assert result[1].total_cost_usd == 4.0

	def test_zero_dispatched_no_division_error(self, db: Database) -> None:
		mid = _setup_mission(db)
		# Epoch exists but no events
		analyzer = BatchAnalyzer(db)
		result = analyzer.get_epoch_cost_summary(mid)
		assert len(result) == 1
		assert result[0].avg_cost_per_unit == 0.0
		assert result[0].success_rate == 0.0

	def test_merge_failed_counts_as_failed(self, db: Database) -> None:
		mid = _setup_mission(db)
		wu = WorkUnit(
			id="mfwu", plan_id="p1", title="Merge fail task",
			epoch_id="ep1", status="failed", cost_usd=1.0,
		)
		db.insert_work_unit(wu)
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep1", work_unit_id="mfwu",
			event_type="dispatched",
		))
		db.insert_unit_event(UnitEvent(
			mission_id=mid, epoch_id="ep1", work_unit_id="mfwu",
			event_type="merge_failed",
		))

		analyzer = BatchAnalyzer(db)
		result = analyzer.get_epoch_cost_summary(mid)
		assert result[0].units_failed == 1
		assert result[0].units_merged == 0

	def test_nonexistent_mission_returns_empty(self, db: Database) -> None:
		analyzer = BatchAnalyzer(db)
		result = analyzer.get_epoch_cost_summary("nonexistent")
		assert result == []


class TestFormatCostTrend:
	def test_empty_list_returns_empty_string(self) -> None:
		assert format_cost_trend([]) == ""

	def test_single_epoch_renders_table(self) -> None:
		summaries = [EpochCostSummary(
			epoch_id="abcd1234",
			total_cost_usd=5.25,
			total_input_tokens=10000,
			total_output_tokens=5000,
			units_dispatched=4,
			units_merged=3,
			units_failed=1,
			avg_cost_per_unit=1.3125,
			success_rate=0.75,
		)]
		result = format_cost_trend(summaries)
		assert "| Epoch |" in result
		assert "abcd1234" in result
		assert "5.25" in result
		assert "10,000" in result
		assert "5,000" in result
		assert "75%" in result

	def test_last_n_limits_output(self) -> None:
		summaries = [
			EpochCostSummary(epoch_id=f"ep{i}xxxxx")
			for i in range(10)
		]
		result = format_cost_trend(summaries, last_n=3)
		# Should only contain the last 3 (epoch_id[:8])
		assert "ep7xxxxx" in result
		assert "ep8xxxxx" in result
		assert "ep9xxxxx" in result
		assert "ep0xxxxx" not in result
		assert "ep6xxxxx" not in result

	def test_default_last_n_is_five(self) -> None:
		summaries = [
			EpochCostSummary(epoch_id=f"ep{i}xxxxx")
			for i in range(8)
		]
		result = format_cost_trend(summaries)
		# Default is 5, so epochs 3-7 visible
		assert "ep3xxxxx" in result
		assert "ep7xxxxx" in result
		assert "ep2xxxxx" not in result


class TestAnalyzeWithDBErrors:
	def test_nonexistent_mission_returns_empty(self, db: Database) -> None:
		analyzer = BatchAnalyzer(db)
		signals = analyzer.analyze("nonexistent")
		assert signals.file_hotspots == []
		assert signals.failure_clusters == {}
		assert signals.failure_stages == {}
		assert signals.stalled_areas == []
