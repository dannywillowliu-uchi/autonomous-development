"""Tests for the coordinator module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mission_control.config import MissionConfig, TargetConfig, VerificationConfig
from mission_control.coordinator import Coordinator, CoordinatorReport
from mission_control.db import Database
from mission_control.models import Plan, Snapshot, WorkUnit


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


@pytest.fixture()
def config(tmp_path: object) -> MissionConfig:
	cfg = MissionConfig()
	cfg.target = TargetConfig(
		name="test-proj",
		path="/tmp/test",
		branch="main",
		objective="Fix all the things",
		verification=VerificationConfig(command="pytest -q"),
	)
	cfg.scheduler.parallel.num_workers = 2
	return cfg


class TestCoordinatorReport:
	def test_defaults(self) -> None:
		report = CoordinatorReport()
		assert report.plan_id == ""
		assert report.total_units == 0
		assert report.workers_spawned == 0
		assert report.wall_time_seconds == 0.0
		assert report.stopped_reason == ""


class TestCoordinator:
	async def test_planner_no_units_exits_early(
		self, db: Database, config: MissionConfig,
	) -> None:
		"""When planner produces 0 units, coordinator exits immediately."""
		empty_plan = Plan(objective="test", status="active", total_units=0)

		with (
			patch("mission_control.coordinator.snapshot_project_health", new_callable=AsyncMock) as mock_snap,
			patch("mission_control.coordinator.create_plan", new_callable=AsyncMock) as mock_plan,
		):
			mock_snap.return_value = Snapshot()
			mock_plan.return_value = empty_plan

			coord = Coordinator(config, db)
			report = await coord.run()

		assert report.stopped_reason == "planner_produced_no_units"
		assert report.total_units == 0
		assert report.wall_time_seconds > 0

	async def test_all_units_done_triggers_shutdown(
		self, db: Database, config: MissionConfig, tmp_path: object,
	) -> None:
		"""Coordinator shuts down when all units are completed."""
		plan = Plan(id="p1", objective="test", status="active", total_units=1)

		# Pre-seed a completed unit so monitor sees all done immediately
		db.insert_plan(plan)
		wu = WorkUnit(id="wu1", plan_id="p1", title="Task 1", status="completed")
		db.insert_work_unit(wu)

		mock_pool = AsyncMock()
		mock_pool.acquire = AsyncMock(side_effect=[
			# merge workspace
			type("P", (), {"__str__": lambda s: "/tmp/merge"})(),
			# worker workspaces
			type("P", (), {"__str__": lambda s: "/tmp/w1"})(),
			type("P", (), {"__str__": lambda s: "/tmp/w2"})(),
		])
		mock_pool.initialize = AsyncMock()
		mock_pool.cleanup = AsyncMock()

		mock_backend = AsyncMock()
		mock_backend._pool = mock_pool
		mock_backend.initialize = AsyncMock()
		mock_backend.cleanup = AsyncMock()

		with (
			patch("mission_control.coordinator.snapshot_project_health", new_callable=AsyncMock) as mock_snap,
			patch("mission_control.coordinator.create_plan", new_callable=AsyncMock) as mock_create,
			patch("mission_control.coordinator.LocalBackend", return_value=mock_backend),
			patch("mission_control.coordinator.MergeQueue") as mock_mq_cls,
			patch("mission_control.coordinator.WorkerAgent") as mock_wa_cls,
		):
			mock_snap.return_value = Snapshot()
			mock_create.return_value = plan

			# MergeQueue mock
			mock_mq = AsyncMock()
			mock_mq.run = AsyncMock()
			mock_mq.stop = lambda: None
			mock_mq_cls.return_value = mock_mq

			# WorkerAgent mock
			mock_agent = AsyncMock()
			mock_agent.run = AsyncMock()
			mock_agent.running = False  # Already stopped
			mock_agent.stop = lambda: None
			mock_wa_cls.return_value = mock_agent

			coord = Coordinator(config, db, num_workers=2)
			report = await coord.run()

		assert report.stopped_reason in ("all_units_done", "all_workers_stopped")
		assert report.plan_id == "p1"

	async def test_stop_sets_running_false(self, db: Database, config: MissionConfig) -> None:
		coord = Coordinator(config, db)
		assert coord.running is True
		coord.stop()
		assert coord.running is False

	async def test_workers_count_from_config(self, db: Database, config: MissionConfig) -> None:
		coord = Coordinator(config, db)
		assert coord.num_workers == 2

	async def test_workers_count_override(self, db: Database, config: MissionConfig) -> None:
		coord = Coordinator(config, db, num_workers=8)
		assert coord.num_workers == 8
