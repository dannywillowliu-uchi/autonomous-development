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


class TestGenerateReportBasic:
	def test_generate_report_basic(self, tmp_path: Path) -> None:
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


class TestGenerateReportBacklogItemIds:
	def test_backlog_item_ids_appear_in_report(self, tmp_path: Path) -> None:
		db = Database()
		config = _make_config(tmp_path)
		mission, epoch, plan = _seed_mission(db)

		_seed_unit_and_handoff(
			db, mission, epoch, plan,
			unit_id="u010", files=["src/main.py"],
		)

		result = ContinuousMissionResult(
			mission_id=mission.id,
			objective=mission.objective,
			total_units_dispatched=1,
			total_units_merged=1,
			wall_time_seconds=120.0,
			backlog_item_ids=["bl_001", "bl_002", "bl_003"],
		)

		report = generate_mission_report(result, mission, db, config)

		assert report["backlog_item_ids"] == ["bl_001", "bl_002", "bl_003"]

		# Verify it's also written to disk
		report_path = tmp_path / "mission_report.json"
		written = json.loads(report_path.read_text())
		assert written["backlog_item_ids"] == ["bl_001", "bl_002", "bl_003"]

		db.close()

	def test_backlog_item_ids_empty_when_none(self, tmp_path: Path) -> None:
		db = Database()
		config = _make_config(tmp_path)
		mission = Mission(objective="no backlog", status="running")
		db.insert_mission(mission)

		result = ContinuousMissionResult(
			mission_id=mission.id,
			objective=mission.objective,
		)

		report = generate_mission_report(result, mission, db, config)

		assert report["backlog_item_ids"] == []

		db.close()


class TestGenerateReportEmptyMission:
	def test_generate_report_empty_mission(self, tmp_path: Path) -> None:
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


class TestGenerateReportWritesFile:
	def test_generate_report_writes_file(self, tmp_path: Path) -> None:
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


class TestGenerateReportTimelineOrdering:
	def test_generate_report_timeline_ordering(self, tmp_path: Path) -> None:
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
