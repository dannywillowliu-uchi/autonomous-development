"""Tests for DashboardProvider snapshot building."""

from __future__ import annotations

import pytest

from mission_control.dashboard.provider import (
	DashboardProvider,
	DashboardSnapshot,
	_build_events,
	_derive_workers_from_plan,
)
from mission_control.db import Database
from mission_control.models import (
	MergeRequest,
	Mission,
	Plan,
	Round,
	Worker,
	WorkUnit,
)

# -- Helpers --


def _make_db() -> Database:
	return Database(":memory:")


def _insert_mission(db: Database, mission_id: str = "m1", **kwargs) -> Mission:
	defaults = {
		"id": mission_id,
		"objective": "Ship the feature",
		"status": "running",
		"total_cost_usd": 1.50,
	}
	defaults.update(kwargs)
	m = Mission(**defaults)
	db.insert_mission(m)
	return m


def _insert_round(
	db: Database,
	mission_id: str = "m1",
	round_id: str = "r1",
	number: int = 1,
	plan_id: str | None = "plan1",
	**kwargs,
) -> Round:
	defaults = {
		"id": round_id,
		"mission_id": mission_id,
		"number": number,
		"status": "executing",
		"plan_id": plan_id,
		"objective_score": 0.0,
		"cost_usd": 0.25,
	}
	defaults.update(kwargs)
	r = Round(**defaults)
	db.insert_round(r)
	return r


def _insert_plan(db: Database, plan_id: str = "plan1", **kwargs) -> Plan:
	defaults = {"id": plan_id, "objective": "test plan"}
	defaults.update(kwargs)
	p = Plan(**defaults)
	db.insert_plan(p)
	return p


def _insert_work_unit(db: Database, unit_id: str = "wu1", plan_id: str = "plan1", **kwargs) -> WorkUnit:
	defaults = {
		"id": unit_id,
		"plan_id": plan_id,
		"title": f"Task {unit_id}",
		"status": "pending",
	}
	defaults.update(kwargs)
	wu = WorkUnit(**defaults)
	db.insert_work_unit(wu)
	return wu


def _insert_worker(db: Database, worker_id: str = "w1", **kwargs) -> Worker:
	defaults = {
		"id": worker_id,
		"workspace_path": f"/tmp/{worker_id}",
		"status": "idle",
	}
	defaults.update(kwargs)
	w = Worker(**defaults)
	db.insert_worker(w)
	return w


def _insert_merge_request(db: Database, mr_id: str = "mr1", **kwargs) -> MergeRequest:
	defaults = {
		"id": mr_id,
		"work_unit_id": "wu1",
		"worker_id": "w1",
		"branch_name": f"mc/unit-{mr_id}",
		"commit_hash": "abc123",
		"status": "pending",
		"position": 1,
	}
	defaults.update(kwargs)
	mr = MergeRequest(**defaults)
	db.insert_merge_request(mr)
	return mr


def _make_provider(db: Database) -> DashboardProvider:
	"""Create a DashboardProvider backed by an in-memory Database."""
	provider = DashboardProvider(":memory:")
	provider._db = db
	return provider


# -- Populated DB for the "full" tests --


def _populated_db() -> Database:
	"""Build a DB with mission, 2 rounds, workers, units, and merge requests."""
	db = _make_db()

	_insert_mission(db, "m1", total_cost_usd=2.00)

	_insert_plan(db, "plan1")
	_insert_plan(db, "plan2")

	_insert_round(db, "m1", "r1", number=1, plan_id="plan1", objective_score=40.0, cost_usd=0.80, status="completed")
	_insert_round(db, "m1", "r2", number=2, plan_id="plan2", objective_score=75.0, cost_usd=1.20, status="executing")

	# Workers
	_insert_worker(db, "w1", status="working", current_unit_id="wu3", units_completed=2, units_failed=0)
	_insert_worker(db, "w2", status="idle", units_completed=1, units_failed=1)
	_insert_worker(db, "w3", status="dead", units_completed=0, units_failed=0)

	# Work units for current round (plan2)
	_insert_work_unit(db, "wu1", "plan2", status="completed", finished_at="2025-06-01T10:00:00", title="Setup DB")
	_insert_work_unit(db, "wu2", "plan2", status="completed", finished_at="2025-06-01T11:00:00", title="Add API")
	_insert_work_unit(db, "wu3", "plan2", status="running", claimed_at="2025-06-01T12:00:00", title="Write tests")
	_insert_work_unit(db, "wu4", "plan2", status="pending", title="Deploy")
	_insert_work_unit(db, "wu5", "plan2", status="failed", finished_at="2025-06-01T09:30:00", title="Broken lint")
	_insert_work_unit(
		db, "wu6", "plan2", status="blocked",
		finished_at="2025-06-01T09:45:00", title="Blocked task", depends_on="wu3",
	)

	# Merge requests for plan2's units
	_insert_merge_request(
		db, "mr1", work_unit_id="wu1", worker_id="w1",
		status="merged", merged_at="2025-06-01T10:05:00", position=1,
	)
	_insert_merge_request(db, "mr2", work_unit_id="wu2", worker_id="w2", status="pending", position=2)
	_insert_merge_request(db, "mr3", work_unit_id="wu3", worker_id="w1", status="pending", position=3)

	return db


# -- Tests --


class TestSnapshotNoMission:
	def test_snapshot_no_mission(self) -> None:
		"""Empty DB returns snapshot with mission=None, phase='idle'."""
		db = _make_db()
		provider = _make_provider(db)
		snap = provider.refresh()

		assert snap.mission is None
		assert snap.phase == "idle"
		assert snap.workers == []
		assert snap.units_total == 0
		assert snap.score_history == []
		assert snap.recent_events == []
		assert snap.timestamp != ""


class TestSnapshotWithMission:
	def test_snapshot_with_mission(self) -> None:
		"""Populated DB produces a fully hydrated snapshot."""
		db = _populated_db()
		provider = _make_provider(db)
		snap = provider.refresh()

		# Mission
		assert snap.mission is not None
		assert snap.mission.id == "m1"
		assert snap.mission.objective == "Ship the feature"
		assert snap.total_cost == 2.00

		# Current round is r2 (latest by number)
		assert snap.current_round is not None
		assert snap.current_round.id == "r2"
		assert snap.current_round.number == 2
		assert snap.phase == "executing"
		assert snap.round_cost == 1.20

		# Workers
		assert snap.workers_active == 1  # w1 is "working"
		assert snap.workers_idle == 1    # w2 is "idle"
		assert snap.workers_dead == 1    # w3 is "dead"
		assert len(snap.workers) == 3

		# Work units (from plan2)
		assert snap.units_total == 6
		assert snap.units_completed == 2  # wu1, wu2
		assert snap.units_running == 1    # wu3
		assert snap.units_pending == 1    # wu4 (pending only, not claimed)
		assert snap.units_failed == 1     # wu5
		assert snap.units_blocked == 1    # wu6

		# Merge queue
		assert snap.merge_queue_depth == 2  # mr2 + mr3 pending

		# Score history
		assert len(snap.score_history) == 2
		assert snap.score_history[0] == (1, 40.0)
		assert snap.score_history[1] == (2, 75.0)


class TestScoreHistory:
	def test_score_history(self) -> None:
		"""score_history is list of (round_number, score) tuples from rounds."""
		db = _make_db()
		_insert_mission(db, "m1")
		_insert_round(db, "m1", "r1", number=1, plan_id=None, objective_score=10.0)
		_insert_round(db, "m1", "r2", number=2, plan_id=None, objective_score=30.0)
		_insert_round(db, "m1", "r3", number=3, plan_id=None, objective_score=55.0)

		provider = _make_provider(db)
		snap = provider.refresh()

		assert snap.score_history == [(1, 10.0), (2, 30.0), (3, 55.0)]


class TestWorkerInfos:
	def test_worker_infos(self) -> None:
		"""Workers list has resolved unit titles."""
		db = _make_db()
		_insert_mission(db, "m1")
		_insert_plan(db, "plan1")
		_insert_round(db, "m1", "r1", number=1, plan_id="plan1", status="executing")

		_insert_work_unit(db, "wu1", "plan1", title="Implement auth")
		_insert_worker(db, "w1", status="working", current_unit_id="wu1", units_completed=5, units_failed=1)
		_insert_worker(db, "w2", status="idle")

		provider = _make_provider(db)
		snap = provider.refresh()

		assert len(snap.workers) == 2

		w1_info = next(w for w in snap.workers if w.id == "w1")
		assert w1_info.current_unit_title == "Implement auth"
		assert w1_info.units_completed == 5
		assert w1_info.units_failed == 1
		assert w1_info.status == "working"

		w2_info = next(w for w in snap.workers if w.id == "w2")
		assert w2_info.current_unit_title == ""
		assert w2_info.status == "idle"


class TestRecentEvents:
	def test_recent_events(self) -> None:
		"""Events are built from completed/claimed units and merged MRs, sorted descending."""
		db = _make_db()
		_insert_mission(db, "m1")
		_insert_plan(db, "plan1")
		_insert_round(db, "m1", "r1", number=1, plan_id="plan1", status="executing")

		_insert_work_unit(db, "wu1", "plan1", status="completed", finished_at="2025-06-01T10:00:00", title="Setup")
		_insert_work_unit(db, "wu2", "plan1", status="running", claimed_at="2025-06-01T12:00:00", title="Build API")
		_insert_work_unit(db, "wu3", "plan1", status="failed", finished_at="2025-06-01T11:00:00", title="Broken")

		_insert_merge_request(
			db, "mr1", work_unit_id="wu1", worker_id="w1",
			status="merged", merged_at="2025-06-01T10:05:00",
			branch_name="mc/unit-wu1",
		)

		provider = _make_provider(db)
		snap = provider.refresh()

		assert len(snap.recent_events) == 4

		# Descending by timestamp
		timestamps = [e.timestamp for e in snap.recent_events]
		assert timestamps == sorted(timestamps, reverse=True)

		# Verify event types
		types = {e.event_type for e in snap.recent_events}
		assert types == {"completed", "claimed", "merged", "failed"}

		# Verify event messages
		messages = [e.message for e in snap.recent_events]
		assert 'Completed "Setup"' in messages
		assert 'Claimed "Build API"' in messages
		assert 'Failed "Broken"' in messages
		assert "Merged mc/unit-wu1" in messages


class TestCompletionPct:
	def test_completion_pct_normal(self) -> None:
		"""completion_pct = (completed / total) * 100."""
		snap = DashboardSnapshot(units_total=10, units_completed=3)
		assert snap.completion_pct == pytest.approx(30.0)

	def test_completion_pct_zero_total(self) -> None:
		"""completion_pct is 0 when no units exist."""
		snap = DashboardSnapshot(units_total=0, units_completed=0)
		assert snap.completion_pct == 0.0


class TestScoreDelta:
	def test_score_delta_with_history(self) -> None:
		"""score_delta is difference between last two scores."""
		snap = DashboardSnapshot(
			score_history=[(1, 40.0), (2, 75.0)],
		)
		assert snap.score_delta == pytest.approx(35.0)

	def test_score_delta_empty(self) -> None:
		"""score_delta is 0 with no history."""
		snap = DashboardSnapshot()
		assert snap.score_delta == 0.0

	def test_score_delta_negative(self) -> None:
		"""score_delta can be negative if score regressed."""
		snap = DashboardSnapshot(
			score_history=[(1, 80.0), (2, 60.0)],
		)
		assert snap.score_delta == pytest.approx(-20.0)


class TestMergeQueueDepth:
	def test_merge_queue_depth(self) -> None:
		"""Only pending merge requests are counted."""
		db = _make_db()
		_insert_mission(db, "m1")
		_insert_plan(db, "plan1")
		_insert_round(db, "m1", "r1", number=1, plan_id="plan1", status="executing")

		_insert_work_unit(db, "wu1", "plan1")
		_insert_work_unit(db, "wu2", "plan1")
		_insert_work_unit(db, "wu3", "plan1")

		_insert_merge_request(db, "mr1", work_unit_id="wu1", status="pending", position=1)
		_insert_merge_request(
			db, "mr2", work_unit_id="wu2",
			status="merged", merged_at="2025-01-01T00:00:00", position=2,
		)
		_insert_merge_request(db, "mr3", work_unit_id="wu3", status="pending", position=3)

		provider = _make_provider(db)
		snap = provider.refresh()

		assert snap.merge_queue_depth == 2


class TestBuildEventsEmpty:
	def test_build_events_empty(self) -> None:
		"""No units = no events."""
		events = _build_events([], [])
		assert events == []


class TestRefreshUpdatesSnapshot:
	def test_refresh_updates_snapshot(self) -> None:
		"""Calling refresh() produces a new snapshot visible via get_snapshot()."""
		db = _make_db()
		provider = _make_provider(db)

		# Initial snapshot has no mission
		snap1 = provider.refresh()
		assert snap1.mission is None

		# Add data, refresh again
		_insert_mission(db, "m1")
		snap2 = provider.refresh()
		assert snap2.mission is not None
		assert snap2.mission.id == "m1"

		# get_snapshot returns the latest
		cached = provider.get_snapshot()
		assert cached.mission is not None
		assert cached.mission.id == "m1"
		assert cached.timestamp == snap2.timestamp


class TestBlockedUnits:
	def test_blocked_event_generated(self) -> None:
		"""A blocked unit with finished_at produces a 'blocked' event."""
		db = _make_db()
		_insert_mission(db, "m1")
		_insert_plan(db, "plan1")
		_insert_round(db, "m1", "r1", number=1, plan_id="plan1", status="executing")
		_insert_work_unit(
			db, "wu1", "plan1", status="blocked",
			finished_at="2025-06-01T10:00:00", title="Waiting on deps",
		)

		provider = _make_provider(db)
		snap = provider.refresh()

		blocked_events = [e for e in snap.recent_events if e.event_type == "blocked"]
		assert len(blocked_events) == 1
		assert 'Blocked "Waiting on deps"' in blocked_events[0].message

	def test_blocked_not_counted_as_pending(self) -> None:
		"""Blocked units don't inflate the pending count."""
		db = _make_db()
		_insert_mission(db, "m1")
		_insert_plan(db, "plan1")
		_insert_round(db, "m1", "r1", number=1, plan_id="plan1", status="executing")
		_insert_work_unit(db, "wu1", "plan1", status="pending", title="Legit pending")
		_insert_work_unit(db, "wu2", "plan1", status="blocked", title="Blocked task")

		provider = _make_provider(db)
		snap = provider.refresh()

		assert snap.units_pending == 1
		assert snap.units_blocked == 1


class TestDeriveWorkersFromPlan:
	def test_running_units_become_workers(self) -> None:
		"""Running work units are shown as workers when no Worker rows exist."""
		db = _make_db()
		_insert_mission(db, "m1")
		_insert_plan(db, "plan1")
		rnd = _insert_round(db, "m1", "r1", number=1, plan_id="plan1", status="executing")
		_insert_work_unit(db, "wu1", "plan1", status="running", title="Active task")
		_insert_work_unit(db, "wu2", "plan1", status="pending", title="Waiting")

		infos, active = _derive_workers_from_plan(db, rnd)
		assert active == 1
		assert len(infos) == 1
		assert infos[0].current_unit_title == "Active task"
		assert infos[0].status == "working"

	def test_snapshot_uses_plan_derive_without_worker_rows(self) -> None:
		"""Full snapshot derives workers from plan when no Worker DB rows exist."""
		db = _make_db()
		_insert_mission(db, "m1")
		_insert_plan(db, "plan1")
		_insert_round(db, "m1", "r1", number=1, plan_id="plan1", status="executing")
		_insert_work_unit(db, "wu1", "plan1", status="running", title="Building API")
		_insert_work_unit(db, "wu2", "plan1", status="claimed", title="Writing tests")

		provider = _make_provider(db)
		snap = provider.refresh()

		assert snap.workers_active == 2
		assert len(snap.workers) == 2
		titles = {w.current_unit_title for w in snap.workers}
		assert titles == {"Building API", "Writing tests"}


class TestChainAggregation:
	def test_provider_single_mission_no_chain(self) -> None:
		"""Non-chained mission (chain_id='') works identically to before."""
		db = _make_db()
		_insert_mission(db, "m1", chain_id="")
		_insert_plan(db, "plan1")
		_insert_round(db, "m1", "r1", number=1, plan_id="plan1", objective_score=50.0)

		provider = _make_provider(db)
		snap = provider.refresh()
		assert snap.mission is not None
		assert snap.mission.id == "m1"
		assert snap.score_history == [(1, 50.0)]

	def test_provider_chain_aggregates_rounds(self) -> None:
		"""Chained missions aggregate rounds across all chain members."""
		db = _make_db()
		_insert_mission(db, "m1", chain_id="chain-1", started_at="2025-01-01T00:00:00")
		_insert_mission(db, "m2", chain_id="chain-1", started_at="2025-01-02T00:00:00")
		_insert_plan(db, "plan1")
		_insert_plan(db, "plan2")
		_insert_round(db, "m1", "r1", number=1, plan_id="plan1", objective_score=30.0, cost_usd=0.50)
		_insert_round(db, "m2", "r2", number=1, plan_id="plan2", objective_score=70.0, cost_usd=1.00)

		provider = _make_provider(db)
		snap = provider.refresh()
		# m2 is latest mission, so it's the current mission
		assert snap.mission is not None
		assert snap.mission.id == "m2"
		# Rounds renumbered sequentially: m1's round=1, m2's round=2
		assert len(snap.score_history) == 2
		assert snap.score_history[0] == (1, 30.0)
		assert snap.score_history[1] == (2, 70.0)

	def test_provider_chain_aggregates_cost(self) -> None:
		"""Total cost sums across all chain missions."""
		db = _make_db()
		_insert_mission(db, "m1", chain_id="chain-1", total_cost_usd=1.50, started_at="2025-01-01T00:00:00")
		_insert_mission(db, "m2", chain_id="chain-1", total_cost_usd=2.50, started_at="2025-01-02T00:00:00")

		provider = _make_provider(db)
		snap = provider.refresh()
		assert snap.total_cost == 4.00


class TestLiveUIPath:
	def test_ui_path_uses_importlib_resources(self) -> None:
		"""_UI_PATH resolves via importlib.resources, not __file__."""
		from mission_control.dashboard.live import _UI_PATH
		# Should be a Traversable from importlib.resources, not a raw Path
		assert _UI_PATH.name == "live_ui.html"
		assert _UI_PATH.is_file()

	def test_ui_path_readable(self) -> None:
		"""_UI_PATH can be read as text (the HTML file exists in the package)."""
		from mission_control.dashboard.live import _UI_PATH
		content = _UI_PATH.read_text()
		assert len(content) > 0
