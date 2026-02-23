"""End-to-end integration tests for the dispatch-merge-complete cycle.

Tests the critical path: controller dispatches unit -> worker executes ->
MC_RESULT parsed -> green branch merge -> handoff ingested -> planner re-plans.

Uses real git repos (no mocks for git operations) following the pattern from
test_green_branch.py (TestGreenBranchRealGit).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from mission_control.backends.base import WorkerBackend, WorkerHandle
from mission_control.config import (
	GreenBranchConfig,
	MissionConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.green_branch import GreenBranchManager
from mission_control.models import Epoch, Handoff, Mission, Plan, UnitEvent, WorkUnit

# ---------------------------------------------------------------------------
# Git helpers (same pattern as test_green_branch.py)
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: str | Path) -> str:
	"""Run a git command synchronously, raise on failure."""
	result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
	return result.stdout.strip()


def _setup_source_repo(tmp_path: Path) -> tuple[Path, Path]:
	"""Create a bare source repo and a workspace clone with initial commits.

	Returns (source_repo, workspace) paths.
	"""
	source = tmp_path / "source.git"
	source.mkdir()
	_run(["git", "init", "--bare"], source)

	setup_clone = tmp_path / "setup-clone"
	_run(["git", "clone", str(source), str(setup_clone)], tmp_path)
	_run(["git", "config", "user.email", "test@test.com"], setup_clone)
	_run(["git", "config", "user.name", "Test"], setup_clone)

	# Initial commit on main
	(setup_clone / "README.md").write_text("# Test Project\n")
	_run(["git", "add", "README.md"], setup_clone)
	_run(["git", "commit", "-m", "Initial commit"], setup_clone)
	_run(["git", "push", "origin", "main"], setup_clone)

	# Second commit
	(setup_clone / "app.py").write_text("print('hello')\n")
	_run(["git", "add", "app.py"], setup_clone)
	_run(["git", "commit", "-m", "Add app.py"], setup_clone)
	_run(["git", "push", "origin", "main"], setup_clone)

	# Create mc/green and mc/working branches
	main_hash = _run(["git", "rev-parse", "main"], setup_clone)
	_run(["git", "branch", "mc/green", main_hash], setup_clone)
	_run(["git", "push", "origin", "mc/green"], setup_clone)
	_run(["git", "branch", "mc/working", main_hash], setup_clone)
	_run(["git", "push", "origin", "mc/working"], setup_clone)

	# Workspace clone for GreenBranchManager
	workspace = tmp_path / "workspace"
	_run(["git", "clone", str(source), str(workspace)], tmp_path)
	_run(["git", "config", "user.email", "test@test.com"], workspace)
	_run(["git", "config", "user.name", "Test"], workspace)
	_run(["git", "branch", "mc/green", "origin/mc/green"], workspace)
	_run(["git", "branch", "mc/working", "origin/mc/working"], workspace)

	return source, workspace


def _make_worker_clone(tmp_path: Path, source: Path, name: str) -> Path:
	"""Create a worker clone from the source repo and return its path."""
	worker = tmp_path / name
	_run(["git", "clone", str(source), str(worker)], tmp_path)
	_run(["git", "config", "user.email", "worker@test.com"], worker)
	_run(["git", "config", "user.name", "Worker"], worker)
	return worker


def _real_config(source: Path) -> MissionConfig:
	"""Build a MissionConfig pointing at real repos."""
	mc = MissionConfig()
	mc.target = TargetConfig(
		name="test",
		path=str(source),
		branch="main",
		verification=VerificationConfig(command="true"),
	)
	mc.green_branch = GreenBranchConfig(
		working_branch="mc/working",
		green_branch="mc/green",
		reset_on_init=False,
	)
	return mc


# ---------------------------------------------------------------------------
# Mock WorkerBackend -- creates real git branches with real commits
# ---------------------------------------------------------------------------


class MockWorkerBackend(WorkerBackend):
	"""Mock backend that simulates workers by creating real git branches and commits.

	Instead of spawning Claude sessions, it:
	1. Creates a real git clone from the source repo
	2. Creates a branch, commits a file change
	3. Returns realistic MC_RESULT output with handoff data
	"""

	def __init__(self, source: Path, tmp_path: Path) -> None:
		self.source = source
		self.tmp_path = tmp_path
		self._workspaces: dict[str, Path] = {}
		self._counter = 0

	async def provision_workspace(
		self, worker_id: str, source_repo: str, base_branch: str,
	) -> str:
		self._counter += 1
		clone = _make_worker_clone(
			self.tmp_path, self.source, f"mock-worker-{self._counter}",
		)
		# Fetch and checkout the base branch so workers start from mc/green
		_run(["git", "fetch", "origin", f"{base_branch}:{base_branch}"], clone)
		_run(["git", "checkout", base_branch], clone)
		self._workspaces[worker_id] = clone
		return str(clone)

	def simulate_worker(
		self, worker_id: str, branch_name: str, filename: str, content: str,
	) -> tuple[str, dict[str, object]]:
		"""Simulate a worker: create branch, commit file, return MC_RESULT dict.

		Returns (commit_hash, mc_result_dict).
		"""
		clone = self._workspaces[worker_id]
		_run(["git", "checkout", "-b", branch_name], clone)
		(clone / filename).write_text(content)
		_run(["git", "add", filename], clone)
		_run(["git", "commit", "-m", f"Add {filename}"], clone)
		commit_hash = _run(["git", "rev-parse", "HEAD"], clone)

		mc_result: dict[str, object] = {
			"status": "completed",
			"commits": [commit_hash],
			"summary": f"Added {filename}",
			"files_changed": [filename],
			"discoveries": [f"Implemented {filename}"],
			"concerns": [],
		}
		return commit_hash, mc_result

	async def spawn(
		self, worker_id: str, workspace_path: str, command: list[str], timeout: int,
	) -> WorkerHandle:
		return WorkerHandle(worker_id=worker_id, pid=99999, workspace_path=workspace_path)

	async def check_status(self, handle: WorkerHandle) -> str:
		return "completed"

	async def get_output(self, handle: WorkerHandle) -> str:
		return ""

	async def kill(self, handle: WorkerHandle) -> None:
		pass

	async def release_workspace(self, workspace_path: str) -> None:
		pass

	async def cleanup(self) -> None:
		pass


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDispatchMergeCompleteCycle:
	"""End-to-end integration tests for the dispatch-merge-complete cycle.

	Uses real git repos, real Database, real GreenBranchManager, and a mock
	WorkerBackend that creates real git branches with real commits. Tests the
	critical path that the ContinuousController orchestrates:

	  dispatch -> worker executes -> MC_RESULT parsed -> green branch merge
	  -> handoff ingested -> planner re-plans
	"""

	async def test_two_units_full_cycle(self, tmp_path: Path) -> None:
		"""Two work units go through full dispatch-merge-complete cycle.

		Verifies:
		- Git state: mc/green has all committed files
		- DB state: units are completed, handoffs stored, events recorded
		- Handoff ingestion: planner accumulates discoveries
		- Mission state: get_recent_handoffs returns all handoffs
		"""
		# --- Setup real git repos ---
		source, workspace = _setup_source_repo(tmp_path)
		config = _real_config(source)
		db = Database(":memory:")

		# --- Real GreenBranchManager ---
		gbm = GreenBranchManager(config, db)
		gbm.workspace = str(workspace)

		# --- Mock backend ---
		backend = MockWorkerBackend(source, tmp_path)

		# --- Real planner (used just for ingest_handoff tracking) ---
		planner = ContinuousPlanner(config, db)

		# --- Create mission, plan, epoch in DB ---
		mission = Mission(objective="Add two features", status="running")
		db.insert_mission(mission)

		plan = Plan(objective=mission.objective)
		db.insert_plan(plan)

		epoch = Epoch(mission_id=mission.id, number=1, units_planned=2)
		db.insert_epoch(epoch)

		# --- Define work units ---
		unit_specs = [
			("feature_alpha.py", "# Feature Alpha\ndef run():\n\tpass\n", "Add feature alpha"),
			("feature_beta.py", "# Feature Beta\ndef run():\n\tpass\n", "Add feature beta"),
		]
		units: list[WorkUnit] = []
		for filename, content, title in unit_specs:
			unit = WorkUnit(
				plan_id=plan.id,
				title=title,
				description=f"Create {filename} with implementation",
				files_hint=filename,
				priority=1,
				epoch_id=epoch.id,
			)
			units.append(unit)

		# --- Run each unit through the full dispatch-merge-complete cycle ---
		for unit, (filename, content, _title) in zip(units, unit_specs):
			branch_name = f"mc/unit-{unit.id}"

			# Step 1: Dispatch -- provision workspace
			ws = await backend.provision_workspace(
				unit.id, str(source), config.green_branch.green_branch,
			)

			# Step 2: Worker execution -- create branch + commit
			commit_hash, mc_result = backend.simulate_worker(
				unit.id, branch_name, filename, content,
			)

			# Step 3: Parse MC_RESULT -- create Handoff
			unit.branch_name = branch_name
			unit.status = str(mc_result["status"])
			unit.commit_hash = commit_hash
			unit.started_at = "2025-01-01T00:00:00+00:00"
			unit.finished_at = "2025-01-01T00:01:00+00:00"

			# Store unit in DB first (FK constraint: handoff references work_unit)
			db.insert_work_unit(unit)

			handoff = Handoff(
				work_unit_id=unit.id,
				epoch_id=epoch.id,
				status=str(mc_result["status"]),
				commits=list(mc_result["commits"]),  # type: ignore[arg-type]
				summary=str(mc_result["summary"]),
				files_changed=list(mc_result["files_changed"]),  # type: ignore[arg-type]
				discoveries=list(mc_result["discoveries"]),  # type: ignore[arg-type]
				concerns=list(mc_result["concerns"]),  # type: ignore[arg-type]
			)
			db.insert_handoff(handoff)
			unit.handoff_id = handoff.id
			db.update_work_unit(unit)

			# Step 4: Merge to mc/green
			merge_result = await gbm.merge_unit(ws, branch_name)
			assert merge_result.merged is True, (
				f"Merge failed for '{unit.title}': {merge_result.failure_output}"
			)
			assert merge_result.rebase_ok is True

			# Step 5: Record unit event (as controller would)
			event = UnitEvent(
				mission_id=mission.id,
				epoch_id=epoch.id,
				work_unit_id=unit.id,
				event_type="merged",
			)
			db.insert_unit_event(event)

			# Step 6: Ingest handoff into planner (triggers re-plan on next call)
			planner.ingest_handoff(handoff)

		# ===================================================================
		# VERIFICATION
		# ===================================================================

		# --- Git state: mc/green has all committed files ---
		_run(["git", "checkout", "mc/green"], workspace)
		assert (workspace / "feature_alpha.py").exists(), "feature_alpha.py missing from mc/green"
		assert (workspace / "feature_beta.py").exists(), "feature_beta.py missing from mc/green"
		# Original files still present
		assert (workspace / "README.md").exists()
		assert (workspace / "app.py").exists()
		# Verify file contents
		assert "Feature Alpha" in (workspace / "feature_alpha.py").read_text()
		assert "Feature Beta" in (workspace / "feature_beta.py").read_text()

		# --- DB state: units are completed with correct status ---
		for unit in units:
			stored = db.get_work_unit(unit.id)
			assert stored is not None, f"Unit {unit.id} not found in DB"
			assert stored.status == "completed"
			assert stored.commit_hash is not None
			assert stored.branch_name.startswith("mc/unit-")
			assert stored.handoff_id is not None

		# --- DB state: handoffs are stored correctly ---
		for unit in units:
			stored = db.get_work_unit(unit.id)
			assert stored is not None
			h = db.get_handoff(stored.handoff_id)
			assert h is not None, f"Handoff not found for unit {unit.id}"
			assert h.status == "completed"
			assert len(h.commits) == 1
			assert len(h.files_changed) == 1
			assert len(h.discoveries) == 1

		# --- DB state: mission-level handoff query works ---
		mission_handoffs = db.get_recent_handoffs(mission.id)
		assert len(mission_handoffs) == 2

		# --- DB state: unit events recorded ---
		events = db.get_unit_events_for_mission(mission.id)
		assert len(events) == 2
		assert all(e.event_type == "merged" for e in events)

		# --- Planner state: discoveries ingested ---
		assert len(planner._discoveries) == 2
		assert any("feature_alpha.py" in d for d in planner._discoveries)
		assert any("feature_beta.py" in d for d in planner._discoveries)

		# --- Source repo sync: mc/green in source matches workspace ---
		ws_green = _run(["git", "rev-parse", "mc/green"], workspace)
		src_green = _run(["git", "rev-parse", "mc/green"], source)
		assert ws_green == src_green, "Source repo mc/green out of sync with workspace"

	async def test_three_units_sequential_merge(self, tmp_path: Path) -> None:
		"""Three units merge sequentially -- each rebases on top of the previous.

		This tests the rebase-on-stale-branch scenario: workers branch from
		the same mc/green state, but merges happen serially so later units
		must rebase on top of earlier merged work.
		"""
		source, workspace = _setup_source_repo(tmp_path)
		config = _real_config(source)
		db = Database(":memory:")

		gbm = GreenBranchManager(config, db)
		gbm.workspace = str(workspace)

		backend = MockWorkerBackend(source, tmp_path)
		planner = ContinuousPlanner(config, db)

		mission = Mission(objective="Add three features", status="running")
		db.insert_mission(mission)

		plan = Plan(objective=mission.objective)
		db.insert_plan(plan)

		epoch = Epoch(mission_id=mission.id, number=1, units_planned=3)
		db.insert_epoch(epoch)

		# All workers provision from the SAME mc/green base (simulating
		# parallel dispatch where all workers start from the same snapshot)
		unit_specs = [
			("module_one.py", "# Module One\nclass One:\n\tpass\n", "Add module one"),
			("module_two.py", "# Module Two\nclass Two:\n\tpass\n", "Add module two"),
			("module_three.py", "# Module Three\nclass Three:\n\tpass\n", "Add module three"),
		]

		# Provision ALL workspaces first (parallel dispatch pattern)
		provisioned: list[tuple[WorkUnit, str]] = []
		for filename, content, title in unit_specs:
			unit = WorkUnit(
				plan_id=plan.id,
				title=title,
				description=f"Create {filename}",
				files_hint=filename,
				priority=1,
				epoch_id=epoch.id,
			)
			ws = await backend.provision_workspace(
				unit.id, str(source), config.green_branch.green_branch,
			)
			provisioned.append((unit, ws))

		# Simulate all workers executing (all branch from same base)
		completed: list[tuple[WorkUnit, str, str, dict[str, object]]] = []
		for (unit, ws), (filename, content, _title) in zip(provisioned, unit_specs):
			branch_name = f"mc/unit-{unit.id}"
			commit_hash, mc_result = backend.simulate_worker(
				unit.id, branch_name, filename, content,
			)
			completed.append((unit, ws, branch_name, mc_result))

		# Merge sequentially (as process_completions does)
		green_hashes: list[str] = []
		for unit, ws, branch_name, mc_result in completed:
			unit.branch_name = branch_name
			unit.status = str(mc_result["status"])
			unit.commit_hash = str(mc_result["commits"][0])  # type: ignore[index]
			unit.started_at = "2025-01-01T00:00:00+00:00"
			unit.finished_at = "2025-01-01T00:01:00+00:00"

			# Insert unit first (FK constraint: handoff references work_unit)
			db.insert_work_unit(unit)

			handoff = Handoff(
				work_unit_id=unit.id,
				epoch_id=epoch.id,
				status=str(mc_result["status"]),
				commits=list(mc_result["commits"]),  # type: ignore[arg-type]
				summary=str(mc_result["summary"]),
				files_changed=list(mc_result["files_changed"]),  # type: ignore[arg-type]
				discoveries=list(mc_result["discoveries"]),  # type: ignore[arg-type]
				concerns=[],
			)
			db.insert_handoff(handoff)
			unit.handoff_id = handoff.id
			db.update_work_unit(unit)

			# Merge -- later units rebase on top of earlier merged work
			merge_result = await gbm.merge_unit(ws, branch_name)
			assert merge_result.merged is True, (
				f"Merge failed for '{unit.title}': {merge_result.failure_output}"
			)
			assert merge_result.rebase_ok is True

			event = UnitEvent(
				mission_id=mission.id,
				epoch_id=epoch.id,
				work_unit_id=unit.id,
				event_type="merged",
			)
			db.insert_unit_event(event)
			planner.ingest_handoff(handoff)

			# Track green branch advancing after each merge
			green_hashes.append(
				_run(["git", "rev-parse", "mc/green"], workspace),
			)

		# --- Verify mc/green advanced after each merge ---
		assert len(set(green_hashes)) == 3, "mc/green should have a unique hash after each merge"

		# --- Verify all files exist on mc/green ---
		_run(["git", "checkout", "mc/green"], workspace)
		assert (workspace / "module_one.py").exists()
		assert (workspace / "module_two.py").exists()
		assert (workspace / "module_three.py").exists()
		assert "Module One" in (workspace / "module_one.py").read_text()
		assert "Module Two" in (workspace / "module_two.py").read_text()
		assert "Module Three" in (workspace / "module_three.py").read_text()

		# --- Verify DB state ---
		mission_units = db.get_work_units_for_mission(mission.id)
		assert len(mission_units) == 3
		assert all(u.status == "completed" for u in mission_units)

		mission_handoffs = db.get_recent_handoffs(mission.id)
		assert len(mission_handoffs) == 3

		events = db.get_unit_events_for_mission(mission.id)
		assert len(events) == 3

		# --- Verify planner ingestion ---
		assert len(planner._discoveries) == 3

		# --- Verify git log shows merge commits ---
		log_output = _run(
			["git", "log", "--oneline", "--merges", "mc/green"],
			workspace,
		)
		# Each merge_unit creates a merge commit (--no-ff)
		merge_lines = [line for line in log_output.splitlines() if line.strip()]
		assert len(merge_lines) >= 3, f"Expected 3 merge commits, got: {log_output}"

	async def test_failed_unit_does_not_corrupt_green(self, tmp_path: Path) -> None:
		"""A failed unit followed by a successful one leaves mc/green clean.

		Simulates: unit 1 fails (MC_RESULT status=failed), unit 2 succeeds.
		Only unit 2's changes should appear on mc/green.
		"""
		source, workspace = _setup_source_repo(tmp_path)
		config = _real_config(source)
		db = Database(":memory:")

		gbm = GreenBranchManager(config, db)
		gbm.workspace = str(workspace)

		backend = MockWorkerBackend(source, tmp_path)

		mission = Mission(objective="Mixed results test", status="running")
		db.insert_mission(mission)

		plan = Plan(objective=mission.objective)
		db.insert_plan(plan)

		epoch = Epoch(mission_id=mission.id, number=1, units_planned=2)
		db.insert_epoch(epoch)

		green_before = _run(["git", "rev-parse", "mc/green"], workspace)

		# --- Unit 1: fails (no merge attempt) ---
		failed_unit = WorkUnit(
			plan_id=plan.id,
			title="Broken feature",
			description="This unit fails",
			files_hint="broken.py",
			priority=1,
			epoch_id=epoch.id,
		)
		await backend.provision_workspace(
			failed_unit.id, str(source), config.green_branch.green_branch,
		)
		# Worker executes but reports failure
		branch1 = f"mc/unit-{failed_unit.id}"
		commit_hash1, _ = backend.simulate_worker(
			failed_unit.id, branch1, "broken.py", "# Broken\nraise Exception\n",
		)

		failed_unit.branch_name = branch1
		failed_unit.status = "failed"
		failed_unit.commit_hash = commit_hash1
		failed_unit.started_at = "2025-01-01T00:00:00+00:00"
		failed_unit.finished_at = "2025-01-01T00:01:00+00:00"
		failed_unit.attempt = 1

		db.insert_work_unit(failed_unit)

		failed_handoff = Handoff(
			work_unit_id=failed_unit.id,
			epoch_id=epoch.id,
			status="failed",
			commits=[commit_hash1],
			summary="Build failed",
			files_changed=["broken.py"],
			discoveries=[],
			concerns=["Build error in broken.py"],
		)
		db.insert_handoff(failed_handoff)
		failed_unit.handoff_id = failed_handoff.id
		db.update_work_unit(failed_unit)

		# Controller skips merge for failed units -- mc/green unchanged
		green_after_fail = _run(["git", "rev-parse", "mc/green"], workspace)
		assert green_after_fail == green_before, "mc/green should not change for failed unit"

		# --- Unit 2: succeeds ---
		success_unit = WorkUnit(
			plan_id=plan.id,
			title="Good feature",
			description="This unit succeeds",
			files_hint="good_feature.py",
			priority=1,
			epoch_id=epoch.id,
		)
		ws2 = await backend.provision_workspace(
			success_unit.id, str(source), config.green_branch.green_branch,
		)
		branch2 = f"mc/unit-{success_unit.id}"
		commit_hash2, mc_result2 = backend.simulate_worker(
			success_unit.id, branch2, "good_feature.py", "# Good Feature\ndef good():\n\tpass\n",
		)

		success_unit.branch_name = branch2
		success_unit.status = "completed"
		success_unit.commit_hash = commit_hash2
		success_unit.started_at = "2025-01-01T00:02:00+00:00"
		success_unit.finished_at = "2025-01-01T00:03:00+00:00"

		db.insert_work_unit(success_unit)

		success_handoff = Handoff(
			work_unit_id=success_unit.id,
			epoch_id=epoch.id,
			status="completed",
			commits=[commit_hash2],
			summary="Added good_feature.py",
			files_changed=["good_feature.py"],
			discoveries=["Implemented good_feature.py"],
			concerns=[],
		)
		db.insert_handoff(success_handoff)
		success_unit.handoff_id = success_handoff.id
		db.update_work_unit(success_unit)

		merge_result = await gbm.merge_unit(ws2, branch2)
		assert merge_result.merged is True

		# --- Verify: mc/green has good_feature.py but NOT broken.py ---
		_run(["git", "checkout", "mc/green"], workspace)
		assert (workspace / "good_feature.py").exists()
		assert not (workspace / "broken.py").exists(), "Failed unit's file should not be on mc/green"

		# --- Verify DB state ---
		stored_failed = db.get_work_unit(failed_unit.id)
		assert stored_failed is not None
		assert stored_failed.status == "failed"

		stored_success = db.get_work_unit(success_unit.id)
		assert stored_success is not None
		assert stored_success.status == "completed"

		# Both handoffs stored
		h_failed = db.get_handoff(failed_unit.handoff_id)
		assert h_failed is not None
		assert h_failed.status == "failed"

		h_success = db.get_handoff(success_unit.handoff_id)
		assert h_success is not None
		assert h_success.status == "completed"
