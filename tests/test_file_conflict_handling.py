"""Tests for file conflict handling improvements (Phases 1-4)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mission_control.config import (
	GreenBranchConfig,
	MissionConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.continuous_controller import (
	ContinuousController,
	ContinuousMissionResult,
	WorkerCompletion,
)
from mission_control.continuous_planner import ContinuousPlanner
from mission_control.db import Database
from mission_control.green_branch import GreenBranchManager, UnitMergeResult
from mission_control.models import Epoch, Handoff, Mission, Plan, WorkUnit


def _config() -> MissionConfig:
	mc = MissionConfig()
	mc.target = TargetConfig(
		name="test",
		path="/tmp/test",
		branch="main",
		verification=VerificationConfig(command="pytest -q"),
	)
	mc.green_branch = GreenBranchConfig(
		working_branch="mc/working",
		green_branch="mc/green",
	)
	return mc


def _manager() -> GreenBranchManager:
	config = _config()
	db = Database(":memory:")
	mgr = GreenBranchManager(config, db)
	mgr.workspace = "/tmp/test-workspace"
	return mgr


# ---------------------------------------------------------------------------
# Phase 1: Track actual merged files via git diff
# ---------------------------------------------------------------------------


class TestMergedFilesFromGitDiff:
	"""UnitMergeResult.changed_files is populated from git diff, not files_hint."""

	async def test_changed_files_populated_on_merge(self) -> None:
		"""Successful merge populates changed_files from git diff --name-only."""
		mgr = _manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and args[1] == "--name-only":
				return (True, "src/app.py\nsrc/utils.py\n")
			if args[0] == "rev-parse" and args[1] == "HEAD":
				return (True, "abc123\n")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(True, ""))  # type: ignore[method-assign]
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert result.changed_files == ["src/app.py", "src/utils.py"]

	async def test_changed_files_empty_when_diff_empty(self) -> None:
		"""If git diff returns empty output, changed_files is empty list."""
		mgr = _manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "diff" and args[1] == "--name-only":
				return (True, "")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(True, ""))  # type: ignore[method-assign]
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert result.changed_files == []

	async def test_controller_uses_changed_files_not_files_hint(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Controller tracks actual changed files, not files_hint."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		# Mock green branch manager to return changed_files different from files_hint
		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
				changed_files=["src/app.py"],  # actual diff
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, ""))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
			files_hint="src/app.py, src/index.html",  # declared scope includes index.html
		)
		db.insert_work_unit(unit)

		handoff = Handoff(
			work_unit_id="wu1", round_id="", epoch_id="ep1",
			status="completed", summary="Done",
		)

		completion = WorkerCompletion(
			unit=unit, handoff=handoff, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		# _merged_files should contain only app.py (from git diff), not index.html
		assert "src/app.py" in ctrl._merged_files
		assert "src/index.html" not in ctrl._merged_files

	async def test_controller_falls_back_to_files_hint(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When changed_files is empty, controller falls back to files_hint."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
				changed_files=[],  # empty -- fallback to files_hint
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, ""))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu2", plan_id="p1", title="Task",
			status="completed", commit_hash="def456",
			branch_name="mc/unit-wu2",
			files_hint="src/models.py",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		assert "src/models.py" in ctrl._merged_files


# ---------------------------------------------------------------------------
# Phase 2: Locked files in planner prompt
# ---------------------------------------------------------------------------


class TestGetLockedFiles:
	"""ContinuousController._get_locked_files() returns in-flight + merged files."""

	def test_empty_when_nothing_running(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		locked = ctrl._get_locked_files()
		assert locked == {}

	def test_includes_merged_files(self, config: MissionConfig, db: Database) -> None:
		ctrl = ContinuousController(config, db)
		ctrl._merged_files = {"src/app.py", "src/index.html"}
		locked = ctrl._get_locked_files()
		assert "src/app.py" in locked
		assert "already merged" in locked["src/app.py"]
		assert "src/index.html" in locked
		assert "already merged" in locked["src/index.html"]

	def test_includes_in_flight_files(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(
			id="wu-running", plan_id="p1", title="Build API",
			status="running", files_hint="src/api.py, src/routes.py",
		)
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		locked = ctrl._get_locked_files()

		assert "src/api.py" in locked
		assert any("in-flight" in r for r in locked["src/api.py"])
		assert "src/routes.py" in locked

	def test_combines_in_flight_and_merged(self, config: MissionConfig, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)
		unit = WorkUnit(
			id="wu-r", plan_id="p1", title="Build API",
			status="running", files_hint="src/api.py",
		)
		db.insert_work_unit(unit)

		ctrl = ContinuousController(config, db)
		ctrl._merged_files = {"src/app.py"}
		locked = ctrl._get_locked_files()

		assert "src/api.py" in locked
		assert "src/app.py" in locked


class TestLockedFilesInPlannerPrompt:
	"""Locked files section is injected into the planner prompt."""

	@pytest.mark.asyncio
	async def test_locked_files_passed_to_planner(self) -> None:
		"""ContinuousPlanner forwards locked_files to RecursivePlanner."""
		config = _config()
		db = Database(":memory:")
		planner = ContinuousPlanner(config, db)

		captured_kwargs: dict = {}

		async def mock_plan_round(**kwargs):
			captured_kwargs.update(kwargs)
			plan = Plan(objective="test")
			from mission_control.models import PlanNode
			root = PlanNode(plan_id=plan.id, depth=0, scope="test", node_type="branch")
			root.strategy = "leaves"
			root._child_leaves = []  # type: ignore[attr-defined]
			root.status = "expanded"
			return plan, root

		planner._inner.plan_round = mock_plan_round  # type: ignore[assignment]

		locked = {"src/app.py": ["already merged"], "src/api.py": ["in-flight: Build API"]}
		mission = Mission(id="m1", objective="test")

		await planner.get_next_units(mission, locked_files=locked)

		assert "locked_files" in captured_kwargs
		assert captured_kwargs["locked_files"] == locked

	@pytest.mark.asyncio
	async def test_locked_section_in_prompt(self) -> None:
		"""RecursivePlanner injects ## Locked Files section into the LLM prompt."""
		from mission_control.recursive_planner import PlannerResult, RecursivePlanner

		config = _config()
		db = Database(":memory:")
		rp = RecursivePlanner(config, db)

		captured_prompt = ""

		async def mock_subprocess(prompt: str, node: object) -> PlannerResult:
			nonlocal captured_prompt
			captured_prompt = prompt
			return PlannerResult(type="leaves", units=[])

		rp._run_planner_subprocess = mock_subprocess  # type: ignore[assignment]

		locked = {"src/app.py": ["already merged"], "src/api.py": ["in-flight: Build API"]}
		await rp.plan_round(
			objective="test",
			snapshot_hash="",
			prior_discoveries=[],
			round_number=1,
			locked_files=locked,
		)

		assert "## Locked Files" in captured_prompt
		assert "src/app.py (already merged)" in captured_prompt
		assert "src/api.py (in-flight: Build API)" in captured_prompt
		assert "AUTOMATICALLY DROPPED" in captured_prompt

	@pytest.mark.asyncio
	async def test_no_locked_section_when_empty(self) -> None:
		"""No ## Locked Files section when locked_files is empty."""
		from mission_control.recursive_planner import PlannerResult, RecursivePlanner

		config = _config()
		db = Database(":memory:")
		rp = RecursivePlanner(config, db)

		captured_prompt = ""

		async def mock_subprocess(prompt: str, node: object) -> PlannerResult:
			nonlocal captured_prompt
			captured_prompt = prompt
			return PlannerResult(type="leaves", units=[])

		rp._run_planner_subprocess = mock_subprocess  # type: ignore[assignment]

		await rp.plan_round(
			objective="test",
			snapshot_hash="",
			prior_discoveries=[],
			round_number=1,
			locked_files={},
		)

		assert "## Locked Files" not in captured_prompt


# ---------------------------------------------------------------------------
# Phase 3: Rebase-before-retry
# ---------------------------------------------------------------------------


class TestRebaseRetry:
	"""merge_unit() retries rebase once before failing."""

	async def test_rebase_retry_succeeds(self) -> None:
		"""First rebase fails, retry succeeds -- merge completes."""
		mgr = _manager()
		rebase_count = 0

		async def mock_git(*args: str) -> tuple[bool, str]:
			nonlocal rebase_count
			if args[0] == "rebase" and len(args) > 1 and args[1] == "mc/green":
				rebase_count += 1
				if rebase_count == 1:
					return (False, "CONFLICT")
				return (True, "")
			if args[0] == "rebase" and "--abort" in args:
				return (True, "")
			if args[0] == "diff" and args[1] == "--name-only":
				return (True, "file.py\n")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(True, ""))  # type: ignore[method-assign]
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert result.rebase_ok is True
		assert rebase_count == 2  # first attempt + retry

	async def test_rebase_retry_also_fails(self) -> None:
		"""Both rebase attempts fail -- returns rebase_conflict failure."""
		mgr = _manager()

		async def mock_git(*args: str) -> tuple[bool, str]:
			if args[0] == "rebase" and len(args) > 1 and args[1] == "mc/green":
				return (False, "CONFLICT in file.py")
			if args[0] == "rebase" and "--abort" in args:
				return (True, "")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is False
		assert result.rebase_ok is False
		assert result.failure_stage == "rebase_conflict"
		assert "after retry" in result.failure_output

	async def test_rebase_succeeds_first_time_no_retry(self) -> None:
		"""When rebase succeeds first time, no retry is attempted."""
		mgr = _manager()
		rebase_count = 0

		async def mock_git(*args: str) -> tuple[bool, str]:
			nonlocal rebase_count
			if args[0] == "rebase" and len(args) > 1 and args[1] == "mc/green":
				rebase_count += 1
				return (True, "")
			if args[0] == "diff" and args[1] == "--name-only":
				return (True, "")
			return (True, "")

		mgr._run_git = AsyncMock(side_effect=mock_git)
		mgr._run_command = AsyncMock(return_value=(True, ""))  # type: ignore[method-assign]
		mgr._sync_to_source = AsyncMock()  # type: ignore[method-assign]

		result = await mgr.merge_unit("/tmp/worker", "feat/branch")

		assert result.merged is True
		assert rebase_count == 1


# ---------------------------------------------------------------------------
# Phase 3: Real git rebase retry integration test
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: str | Path) -> str:
	result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
	return result.stdout.strip()


def _setup_source_repo(tmp_path: Path) -> tuple[Path, Path]:
	source = tmp_path / "source.git"
	source.mkdir()
	_run(["git", "init", "--bare"], source)

	setup_clone = tmp_path / "setup-clone"
	_run(["git", "clone", str(source), str(setup_clone)], tmp_path)
	_run(["git", "config", "user.email", "test@test.com"], setup_clone)
	_run(["git", "config", "user.name", "Test"], setup_clone)

	(setup_clone / "README.md").write_text("# Test\n")
	_run(["git", "add", "README.md"], setup_clone)
	_run(["git", "commit", "-m", "Initial commit"], setup_clone)
	_run(["git", "push", "origin", "main"], setup_clone)

	(setup_clone / "app.py").write_text("print('hello')\n")
	_run(["git", "add", "app.py"], setup_clone)
	_run(["git", "commit", "-m", "Add app.py"], setup_clone)
	_run(["git", "push", "origin", "main"], setup_clone)

	main_hash = _run(["git", "rev-parse", "main"], setup_clone)
	for branch in ("mc/green", "mc/working"):
		_run(["git", "branch", branch, main_hash], setup_clone)
		_run(["git", "push", "origin", branch], setup_clone)

	workspace = tmp_path / "workspace"
	_run(["git", "clone", str(source), str(workspace)], tmp_path)
	_run(["git", "config", "user.email", "test@test.com"], workspace)
	_run(["git", "config", "user.name", "Test"], workspace)
	_run(["git", "branch", "mc/green", "origin/mc/green"], workspace)
	_run(["git", "branch", "mc/working", "origin/mc/working"], workspace)

	return source, workspace


class TestChangedFilesRealGit:
	"""Integration: changed_files populated from actual git diff."""

	async def test_changed_files_from_real_merge(self, tmp_path: Path) -> None:
		"""changed_files reflects actual files changed, not declared scope."""
		source, workspace = _setup_source_repo(tmp_path)
		config = MissionConfig()
		config.target = TargetConfig(
			name="test", path=str(source), branch="main",
			verification=VerificationConfig(command="true"),
		)
		config.green_branch = GreenBranchConfig(
			working_branch="mc/working",
			green_branch="mc/green",
			reset_on_init=False,
		)

		db = Database(":memory:")
		mgr = GreenBranchManager(config, db)
		mgr.workspace = str(workspace)

		# Create worker that only touches feature_a.py
		worker = tmp_path / "worker"
		_run(["git", "clone", str(source), str(worker)], tmp_path)
		_run(["git", "config", "user.email", "w@test.com"], worker)
		_run(["git", "config", "user.name", "Worker"], worker)
		_run(["git", "checkout", "-b", "unit/feature-a"], worker)
		(worker / "feature_a.py").write_text("# Feature A\n")
		_run(["git", "add", "feature_a.py"], worker)
		_run(["git", "commit", "-m", "Add feature A"], worker)

		result = await mgr.merge_unit(str(worker), "unit/feature-a")

		assert result.merged is True
		assert "feature_a.py" in result.changed_files
		# Should NOT include files not actually changed
		assert "app.py" not in result.changed_files
		assert "README.md" not in result.changed_files


# ---------------------------------------------------------------------------
# Phase 4: Reconciler sweep
# ---------------------------------------------------------------------------


class TestReconciliationCheck:
	"""GreenBranchManager.run_reconciliation_check() runs verification on green."""

	async def test_reconciliation_check_passes(self) -> None:
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(True, "all tests passed"))  # type: ignore[method-assign]

		ok, output = await mgr.run_reconciliation_check()

		assert ok is True
		assert output == "all tests passed"
		# Should checkout mc/green before running verification
		mgr._run_git.assert_any_call("checkout", "mc/green")

	async def test_reconciliation_check_fails(self) -> None:
		mgr = _manager()
		mgr._run_git = AsyncMock(return_value=(True, ""))
		mgr._run_command = AsyncMock(return_value=(False, "2 tests failed"))  # type: ignore[method-assign]

		ok, output = await mgr.run_reconciliation_check()

		assert ok is False
		assert "failed" in output


class TestReconcilerSweep:
	"""Controller triggers reconciler after merges."""

	@pytest.mark.asyncio
	async def test_reconciler_runs_after_merge(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Reconciler sweep fires after a successful merge."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, "all pass"))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		mock_gbm.run_reconciliation_check.assert_awaited_once()

	@pytest.mark.asyncio
	async def test_reconciler_triggers_fixup_on_failure(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When reconciler verification fails, run_fixup is called."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		from mission_control.green_branch import FixupResult

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(
			return_value=(False, "ImportError: cannot import name 'foo'"),
		)
		mock_gbm.run_fixup = AsyncMock(
			return_value=FixupResult(success=True),
		)
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		unit = WorkUnit(
			id="wu1", plan_id="p1", title="Task",
			status="completed", commit_hash="abc123",
			branch_name="mc/unit-wu1",
		)
		db.insert_work_unit(unit)

		completion = WorkerCompletion(
			unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
		)
		ctrl._completion_queue.put_nowait(completion)
		ctrl.running = False

		await ctrl._process_completions(Mission(id="m1"), result)

		mock_gbm.run_fixup.assert_awaited_once()
		# Fixup should receive the failure output
		call_args = mock_gbm.run_fixup.call_args
		assert "ImportError" in call_args[0][0]

	@pytest.mark.asyncio
	async def test_reconciler_skipped_when_no_new_merges(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Reconciler doesn't re-run if no new merges since last check."""
		db.insert_mission(Mission(id="m1", objective="test"))
		ctrl = ContinuousController(config, db)
		result = ContinuousMissionResult(mission_id="m1")

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock(
			return_value=UnitMergeResult(
				merged=True, rebase_ok=True, verification_passed=True,
			),
		)
		mock_gbm.run_reconciliation_check = AsyncMock(return_value=(True, ""))
		ctrl._green_branch = mock_gbm

		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		plan = Plan(id="p1", objective="test")
		db.insert_plan(plan)

		# Process first merge
		unit1 = WorkUnit(
			id="wu1", plan_id="p1", title="Task 1",
			status="completed", commit_hash="abc", branch_name="mc/unit-wu1",
		)
		db.insert_work_unit(unit1)
		ctrl._completion_queue.put_nowait(
			WorkerCompletion(unit=unit1, handoff=None, workspace="/tmp/ws", epoch=epoch),
		)

		# Process a failed unit (no merge)
		unit2 = WorkUnit(
			id="wu2", plan_id="p1", title="Task 2",
			status="failed",
		)
		db.insert_work_unit(unit2)
		ctrl._completion_queue.put_nowait(
			WorkerCompletion(unit=unit2, handoff=None, workspace="/tmp/ws", epoch=epoch),
		)

		ctrl.running = False
		await ctrl._process_completions(Mission(id="m1"), result)

		# Reconciler should have run once (after first merge) but not again for the failure
		assert mock_gbm.run_reconciliation_check.await_count == 1
