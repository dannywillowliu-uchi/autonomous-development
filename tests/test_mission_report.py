"""Tests for mission_report module."""

from __future__ import annotations

import json
from pathlib import Path

from mission_control.config import MissionConfig, TargetConfig
from mission_control.continuous_controller import ContinuousMissionResult
from mission_control.db import Database
from mission_control.mission_report import generate_mission_report
from mission_control.models import Epoch, Handoff, Mission, Plan, UnitEvent, WorkUnit


def _make_config(tmp_path: Path) -> MissionConfig:
	config = MissionConfig()
	config.target = TargetConfig(name="test", path=str(tmp_path), objective="test objective")
	return config


def _seed_mission(db: Database) -> tuple[Mission, Epoch, Plan]:
	mission = Mission(objective="implement feature X", status="running")
	db.insert_mission(mission)

	plan = Plan(objective=mission.objective, status="active")
	db.insert_plan(plan)

	epoch = Epoch(mission_id=mission.id, number=1)
	db.insert_epoch(epoch)

	return mission, epoch, plan


def _seed_unit_and_handoff(
	db: Database,
	mission: Mission,
	epoch: Epoch,
	plan: Plan,
	*,
	unit_id: str = "unit001",
	files: list[str] | None = None,
	handoff_status: str = "completed",
) -> tuple[WorkUnit, Handoff]:
	unit = WorkUnit(id=unit_id, plan_id=plan.id, title=f"task {unit_id}", status="completed")
	unit.epoch_id = epoch.id
	db.insert_work_unit(unit)

	handoff = Handoff(
		work_unit_id=unit.id,
		round_id="",
		epoch_id=epoch.id,
		status=handoff_status,
		summary=f"did {unit_id}",
		files_changed=files or [],
	)
	db.insert_handoff(handoff)
	return unit, handoff


class TestGenerateReport:
	def test_basic(self, tmp_path: Path) -> None:
		db = Database()
		config = _make_config(tmp_path)
		mission, epoch, plan = _seed_mission(db)

		_seed_unit_and_handoff(
			db, mission, epoch, plan,
			unit_id="u001", files=["src/foo.py", "src/bar.py"],
		)
		_seed_unit_and_handoff(
			db, mission, epoch, plan,
			unit_id="u002", files=["src/baz.py"],
			handoff_status="failed",
		)

		# Insert events
		db.insert_unit_event(UnitEvent(
			mission_id=mission.id, epoch_id=epoch.id,
			work_unit_id="u001", event_type="dispatched",
			timestamp="2026-01-01T00:00:00+00:00",
		))
		db.insert_unit_event(UnitEvent(
			mission_id=mission.id, epoch_id=epoch.id,
			work_unit_id="u001", event_type="merged",
			timestamp="2026-01-01T00:05:00+00:00",
		))

		result = ContinuousMissionResult(
			mission_id=mission.id,
			objective=mission.objective,
			objective_met=False,
			total_units_dispatched=2,
			total_units_merged=1,
			total_units_failed=1,
			wall_time_seconds=300.5,
			stopped_reason="wall_time_exceeded",
			final_verification_passed=True,
			final_verification_output="all passed",
		)

		report = generate_mission_report(result, mission, db, config)

		assert report["objective"] == "implement feature X"
		assert report["outcome"]["objective_met"] is False
		assert report["outcome"]["stopped_reason"] == "wall_time_exceeded"
		assert report["mission_id"] == mission.id
		assert report["units_dispatched"] == 2
		assert report["units_merged"] == 1
		assert report["units_failed"] == 1
		assert report["wall_time_seconds"] == 300.5
		assert report["total_cost_usd"] == mission.total_cost_usd
		assert report["files_changed"] == ["src/bar.py", "src/baz.py", "src/foo.py"]
		assert report["verification_passed"] is True
		assert report["verification_output"] == "all passed"
		assert len(report["timeline"]) == 2
		assert report["timeline"][0]["event_type"] == "dispatched"
		assert report["timeline"][1]["event_type"] == "merged"

		db.close()

	def test_empty_mission(self, tmp_path: Path) -> None:
		db = Database()
		config = _make_config(tmp_path)
		mission = Mission(objective="empty mission", status="running")
		db.insert_mission(mission)

		result = ContinuousMissionResult(
			mission_id=mission.id,
			objective=mission.objective,
		)

		report = generate_mission_report(result, mission, db, config)

		assert report["objective"] == "empty mission"
		assert report["timeline"] == []
		assert report["files_changed"] == []
		assert report["units_dispatched"] == 0
		assert report["units_merged"] == 0
		assert report["units_failed"] == 0

		db.close()

	def test_writes_file(self, tmp_path: Path) -> None:
		db = Database()
		config = _make_config(tmp_path)
		mission, epoch, plan = _seed_mission(db)

		_seed_unit_and_handoff(
			db, mission, epoch, plan,
			unit_id="u100", files=["README.md"],
		)

		result = ContinuousMissionResult(
			mission_id=mission.id,
			objective=mission.objective,
			total_units_dispatched=1,
			total_units_merged=1,
			wall_time_seconds=60.0,
			final_verification_passed=True,
		)

		generate_mission_report(result, mission, db, config)

		report_path = tmp_path / "mission_report.json"
		assert report_path.exists()

		written = json.loads(report_path.read_text())
		assert written["mission_id"] == mission.id
		assert written["files_changed"] == ["README.md"]

		db.close()

	def test_timeline_ordering(self, tmp_path: Path) -> None:
		db = Database()
		config = _make_config(tmp_path)
		mission, epoch, plan = _seed_mission(db)

		# Create work units so FK constraints pass
		wu1 = WorkUnit(id="u1", plan_id=plan.id, title="task u1", status="completed")
		wu2 = WorkUnit(id="u2", plan_id=plan.id, title="task u2", status="completed")
		db.insert_work_unit(wu1)
		db.insert_work_unit(wu2)

		# Insert events out of chronological order
		db.insert_unit_event(UnitEvent(
			mission_id=mission.id, epoch_id=epoch.id,
			work_unit_id="u1", event_type="merged",
			timestamp="2026-01-01T00:10:00+00:00",
		))
		db.insert_unit_event(UnitEvent(
			mission_id=mission.id, epoch_id=epoch.id,
			work_unit_id="u1", event_type="dispatched",
			timestamp="2026-01-01T00:01:00+00:00",
		))
		db.insert_unit_event(UnitEvent(
			mission_id=mission.id, epoch_id=epoch.id,
			work_unit_id="u2", event_type="dispatched",
			timestamp="2026-01-01T00:05:00+00:00",
		))

		result = ContinuousMissionResult(mission_id=mission.id, objective=mission.objective)

		report = generate_mission_report(result, mission, db, config)

		timestamps = [e["timestamp"] for e in report["timeline"]]
		assert timestamps == sorted(timestamps)
		assert len(timestamps) == 3
		assert report["timeline"][0]["event_type"] == "dispatched"
		assert report["timeline"][0]["work_unit_id"] == "u1"
		assert report["timeline"][1]["event_type"] == "dispatched"
		assert report["timeline"][1]["work_unit_id"] == "u2"
		assert report["timeline"][2]["event_type"] == "merged"

		db.close()


class TestMergeQuality:
	"""Tests for the merge_quality section of the mission report."""

	def _insert_event(
		self,
		db: Database,
		mission: Mission,
		epoch: Epoch,
		unit_id: str,
		event_type: str,
		ts: str,
	) -> None:
		db.insert_unit_event(UnitEvent(
			mission_id=mission.id,
			epoch_id=epoch.id,
			work_unit_id=unit_id,
			event_type=event_type,
			timestamp=ts,
		))

	def test_perfect_merge_quality(self, tmp_path: Path) -> None:
		"""All merges succeed -- score should be 1.0."""
		db = Database()
		config = _make_config(tmp_path)
		mission, epoch, plan = _seed_mission(db)

		for i in range(3):
			uid = f"u{i}"
			wu = WorkUnit(id=uid, plan_id=plan.id, title=f"task {uid}", status="completed")
			wu.epoch_id = epoch.id
			db.insert_work_unit(wu)
			self._insert_event(db, mission, epoch, uid, "dispatched", f"2026-01-01T00:0{i}:00+00:00")
			self._insert_event(db, mission, epoch, uid, "merged", f"2026-01-01T00:0{i}:30+00:00")

		result = ContinuousMissionResult(mission_id=mission.id, objective=mission.objective)
		report = generate_mission_report(result, mission, db, config)

		mq = report["merge_quality"]
		assert mq["total_merge_attempts"] == 3
		assert mq["successful_merges"] == 3
		assert mq["merge_failures"] == 0
		assert mq["retries"] == 0
		assert mq["reverts"] == 0
		assert mq["merge_quality_score"] == 1.0

		db.close()

	def test_failures_and_reverts(self, tmp_path: Path) -> None:
		"""Mix of merges, failures, retries, and reverts."""
		db = Database()
		config = _make_config(tmp_path)
		mission, epoch, plan = _seed_mission(db)

		# u0: dispatched -> merged (success)
		wu0 = WorkUnit(id="u0", plan_id=plan.id, title="task u0", status="completed")
		wu0.epoch_id = epoch.id
		db.insert_work_unit(wu0)
		self._insert_event(db, mission, epoch, "u0", "dispatched", "2026-01-01T00:00:00+00:00")
		self._insert_event(db, mission, epoch, "u0", "merged", "2026-01-01T00:01:00+00:00")

		# u1: dispatched -> merge_failed -> retry_queued -> merged (success on retry)
		wu1 = WorkUnit(id="u1", plan_id=plan.id, title="task u1", status="completed")
		wu1.epoch_id = epoch.id
		db.insert_work_unit(wu1)
		self._insert_event(db, mission, epoch, "u1", "dispatched", "2026-01-01T00:02:00+00:00")
		self._insert_event(db, mission, epoch, "u1", "merge_failed", "2026-01-01T00:03:00+00:00")
		self._insert_event(db, mission, epoch, "u1", "retry_queued", "2026-01-01T00:03:30+00:00")
		self._insert_event(db, mission, epoch, "u1", "merged", "2026-01-01T00:04:00+00:00")

		# u2: dispatched -> merged -> reverted
		wu2 = WorkUnit(id="u2", plan_id=plan.id, title="task u2", status="reverted")
		wu2.epoch_id = epoch.id
		db.insert_work_unit(wu2)
		self._insert_event(db, mission, epoch, "u2", "dispatched", "2026-01-01T00:05:00+00:00")
		self._insert_event(db, mission, epoch, "u2", "merged", "2026-01-01T00:06:00+00:00")
		self._insert_event(db, mission, epoch, "u2", "revert", "2026-01-01T00:07:00+00:00")

		result = ContinuousMissionResult(mission_id=mission.id, objective=mission.objective)
		report = generate_mission_report(result, mission, db, config)

		mq = report["merge_quality"]
		# merged events: u0, u1, u2 = 3 successful
		assert mq["successful_merges"] == 3
		# merge_failed events: u1 = 1
		assert mq["merge_failures"] == 1
		# total attempts = 3 + 1 = 4
		assert mq["total_merge_attempts"] == 4
		assert mq["retries"] == 1
		assert mq["reverts"] == 1
		# score = 3/4 = 0.75
		assert mq["merge_quality_score"] == 0.75

		db.close()

	def test_empty_timeline(self, tmp_path: Path) -> None:
		"""No events at all -- score should be None."""
		db = Database()
		config = _make_config(tmp_path)
		mission = Mission(objective="empty", status="running")
		db.insert_mission(mission)

		result = ContinuousMissionResult(mission_id=mission.id, objective=mission.objective)
		report = generate_mission_report(result, mission, db, config)

		mq = report["merge_quality"]
		assert mq["total_merge_attempts"] == 0
		assert mq["successful_merges"] == 0
		assert mq["merge_failures"] == 0
		assert mq["retries"] == 0
		assert mq["reverts"] == 0
		assert mq["merge_quality_score"] is None

		db.close()

	def test_all_failures_score_zero(self, tmp_path: Path) -> None:
		"""All merge attempts fail -- score should be 0.0."""
		db = Database()
		config = _make_config(tmp_path)
		mission, epoch, plan = _seed_mission(db)

		for i in range(2):
			uid = f"u{i}"
			wu = WorkUnit(id=uid, plan_id=plan.id, title=f"task {uid}", status="failed")
			wu.epoch_id = epoch.id
			db.insert_work_unit(wu)
			self._insert_event(db, mission, epoch, uid, "dispatched", f"2026-01-01T00:0{i}:00+00:00")
			self._insert_event(db, mission, epoch, uid, "merge_failed", f"2026-01-01T00:0{i}:30+00:00")

		result = ContinuousMissionResult(mission_id=mission.id, objective=mission.objective)
		report = generate_mission_report(result, mission, db, config)

		mq = report["merge_quality"]
		assert mq["total_merge_attempts"] == 2
		assert mq["successful_merges"] == 0
		assert mq["merge_failures"] == 2
		assert mq["merge_quality_score"] == 0.0

		db.close()

	def test_basic_report_includes_merge_quality(self, tmp_path: Path) -> None:
		"""The existing basic test scenario should include merge_quality."""
		db = Database()
		config = _make_config(tmp_path)
		mission, epoch, plan = _seed_mission(db)

		wu = WorkUnit(id="u001", plan_id=plan.id, title="task u001", status="completed")
		wu.epoch_id = epoch.id
		db.insert_work_unit(wu)
		db.insert_unit_event(UnitEvent(
			mission_id=mission.id, epoch_id=epoch.id,
			work_unit_id="u001", event_type="dispatched",
			timestamp="2026-01-01T00:00:00+00:00",
		))
		db.insert_unit_event(UnitEvent(
			mission_id=mission.id, epoch_id=epoch.id,
			work_unit_id="u001", event_type="merged",
			timestamp="2026-01-01T00:05:00+00:00",
		))

		result = ContinuousMissionResult(mission_id=mission.id, objective=mission.objective)
		report = generate_mission_report(result, mission, db, config)

		assert "merge_quality" in report
		mq = report["merge_quality"]
		assert mq["successful_merges"] == 1
		assert mq["total_merge_attempts"] == 1
		assert mq["merge_quality_score"] == 1.0

		db.close()
