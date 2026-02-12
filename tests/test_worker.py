"""Tests for the worker agent."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from mission_control.backends.base import WorkerBackend, WorkerHandle
from mission_control.config import MissionConfig, TargetConfig, VerificationConfig
from mission_control.db import Database
from mission_control.models import Mission, Plan, Round, Worker, WorkUnit
from mission_control.worker import WorkerAgent, render_mission_worker_prompt, render_worker_prompt


class MockBackend(WorkerBackend):
	"""Minimal mock backend for tests."""

	async def provision_workspace(self, worker_id: str, source_repo: str, base_branch: str) -> str:
		return "/tmp/mock-workspace"

	async def spawn(self, worker_id: str, workspace_path: str, command: list[str], timeout: int) -> WorkerHandle:
		return WorkerHandle(worker_id=worker_id, pid=12345, workspace_path=workspace_path)

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


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


@pytest.fixture()
def config() -> MissionConfig:
	cfg = MissionConfig()
	cfg.target = TargetConfig(
		name="test-proj",
		path="/tmp/test",
		branch="main",
		verification=VerificationConfig(command="pytest -q"),
	)
	return cfg


@pytest.fixture()
def mock_backend() -> MockBackend:
	return MockBackend()


@pytest.fixture()
def worker_and_unit(db: Database) -> tuple[Worker, WorkUnit]:
	# Create mission + round so handoff FK is satisfied
	mission = Mission(id="m1", objective="test")
	db.insert_mission(mission)
	rnd = Round(id="r1", mission_id="m1")
	db.insert_round(rnd)
	db.insert_plan(Plan(id="p1", objective="test", round_id="r1"))
	wu = WorkUnit(id="wu1", plan_id="p1", title="Fix tests", description="Fix failing tests", round_id="r1")
	db.insert_work_unit(wu)
	w = Worker(id="w1", workspace_path="/tmp/clone1")
	db.insert_worker(w)
	return w, wu


class TestRenderWorkerPrompt:
	def test_contains_title(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Fix lint", description="Fix ruff errors")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-abc")
		assert "Fix lint" in prompt
		assert "Fix ruff errors" in prompt

	def test_contains_files_hint(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X", files_hint="src/foo.py,src/bar.py")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-abc")
		assert "src/foo.py,src/bar.py" in prompt

	def test_contains_branch_name(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-abc123")
		assert "mc/unit-abc123" in prompt

	def test_contains_verification_hint(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X", verification_hint="Run test_foo.py specifically")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-x")
		assert "Run test_foo.py specifically" in prompt

	def test_contains_verification_command(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-x")
		assert "pytest -q" in prompt

	def test_contains_target_name(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-x")
		assert "test-proj" in prompt

	def test_default_files_hint(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-x")
		assert "Not specified" in prompt

	def test_per_unit_verification_command_override(self, config: MissionConfig) -> None:
		"""Per-unit verification_command overrides config default."""
		unit = WorkUnit(title="X", verification_command="make test-specific")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-x")
		assert "make test-specific" in prompt
		assert "pytest -q" not in prompt

	def test_per_unit_verification_command_none_uses_config(self, config: MissionConfig) -> None:
		"""When verification_command is None, falls back to config."""
		unit = WorkUnit(title="X")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-x")
		assert "pytest -q" in prompt


class TestRenderMissionWorkerPrompt:
	def test_uses_config_verification_by_default(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Fix bug", description="Fix the thing")
		prompt = render_mission_worker_prompt(unit, config, "/tmp/ws", "mc/unit-x")
		assert "pytest -q" in prompt

	def test_per_unit_verification_override(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Fix bug", verification_command="npm test")
		prompt = render_mission_worker_prompt(unit, config, "/tmp/ws", "mc/unit-x")
		assert "npm test" in prompt
		assert "pytest -q" not in prompt


class TestWorkerAgent:
	async def test_heartbeat_fires(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		w, wu = worker_and_unit
		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=1)

		# Claim the unit manually so heartbeat has something to update
		claimed = db.claim_work_unit(w.id)
		assert claimed is not None

		# Start heartbeat, let it fire once, then cancel
		task = agent._heartbeat_loop()  # noqa: SLF001
		ht = __import__("asyncio").create_task(task)
		await __import__("asyncio").sleep(1.5)
		ht.cancel()
		try:
			await ht
		except __import__("asyncio").CancelledError:
			pass

		# Check heartbeat was updated
		refreshed = db.get_work_unit(claimed.id)
		assert refreshed is not None
		# heartbeat_at should have been updated (it was set during claim, then updated by heartbeat)

	async def test_successful_unit_execution(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		w, _ = worker_and_unit

		# Configure mock backend to return MC_RESULT output
		mc_output = (
			'MC_RESULT:{"status":"completed","commits":["abc123"],'
			'"summary":"Fixed it","files_changed":["foo.py"]}'
		)
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)
		agent.running = True

		# Mock git subprocess (for checkout -b)
		mock_git_proc = AsyncMock()
		mock_git_proc.communicate = AsyncMock(return_value=(b"", b""))
		mock_git_proc.returncode = 0

		with patch("mission_control.worker.asyncio.create_subprocess_exec", return_value=mock_git_proc):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# Check work unit was completed
		unit = db.get_work_unit("wu1")
		assert unit is not None
		assert unit.status == "completed"
		assert unit.commit_hash == "abc123"
		assert unit.output_summary == "Fixed it"

		# Check merge request was created
		mr = db.get_next_merge_request()
		assert mr is not None
		assert mr.work_unit_id == "wu1"
		assert mr.worker_id == "w1"

	async def test_failed_unit_marks_correctly(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		w, _ = worker_and_unit

		# Backend returns "failed" status with error output
		mock_backend.check_status = AsyncMock(return_value="failed")  # type: ignore[method-assign]
		mock_backend.get_output = AsyncMock(return_value="Error: something broke")  # type: ignore[method-assign]

		mock_git_proc = AsyncMock()
		mock_git_proc.communicate = AsyncMock(return_value=(b"", b""))
		mock_git_proc.returncode = 0

		with patch("mission_control.worker.asyncio.create_subprocess_exec", return_value=mock_git_proc):
			agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "failed"
		assert result.attempt == 1
		assert w.units_failed == 1

	async def test_timeout_marks_failed(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		w, _ = worker_and_unit

		# Backend always returns "running" to trigger the deadline timeout
		mock_backend.check_status = AsyncMock(return_value="running")  # type: ignore[method-assign]
		mock_backend.kill = AsyncMock()  # type: ignore[method-assign]

		# Set a very short timeout so it triggers quickly
		config.scheduler.session_timeout = 0

		mock_git_proc = AsyncMock()
		mock_git_proc.communicate = AsyncMock(return_value=(b"", b""))
		mock_git_proc.returncode = 0

		with patch("mission_control.worker.asyncio.create_subprocess_exec", return_value=mock_git_proc):
			agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "failed"
		assert "Timed out" in result.output_summary

	async def test_success_path_cleans_up_workspace(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""After successful unit, workspace checks out base branch but keeps feature branch for merge queue."""
		w, _ = worker_and_unit

		mc_output = (
			'MC_RESULT:{"status":"completed","commits":["abc123"],'
			'"summary":"Fixed it","files_changed":["foo.py"]}'
		)
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		git_calls: list[tuple[str, ...]] = []

		async def tracking_run_git(*args: str, cwd: str) -> bool:
			git_calls.append(args)
			return True

		with patch.object(agent, "_run_git", side_effect=tracking_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# Should checkout base branch but NOT delete feature branch (merge queue needs it)
		checkout_base_calls = [c for c in git_calls if c == ("checkout", "main")]
		branch_delete_calls = [c for c in git_calls if c[0] == "branch" and c[1] == "-D"]
		assert len(checkout_base_calls) >= 1, f"Expected checkout to base branch, got: {git_calls}"
		assert len(branch_delete_calls) == 0, f"Branch should NOT be deleted on success: {git_calls}"


	async def test_checkout_failure_marks_unit_failed(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""If both checkout -b and -B fail, unit is marked failed."""
		w, _ = worker_and_unit

		# Make git checkout always fail
		mock_git_proc = AsyncMock()
		mock_git_proc.communicate = AsyncMock(return_value=(b"error: branch exists", b""))
		mock_git_proc.returncode = 1

		with patch("mission_control.worker.asyncio.create_subprocess_exec", return_value=mock_git_proc):
			agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "failed"
		assert "Failed to create branch" in result.output_summary
		assert w.units_failed == 1

	async def test_large_output_drains_stdout_during_polling(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""WorkerAgent drains stdout during polling to prevent pipe deadlock for >64KB output."""
		w, _ = worker_and_unit

		# Simulate a process that produces >64KB of output
		large_output = "x" * 100_000  # 100KB of output
		mc_line = (
			'MC_RESULT:{"status":"completed","commits":["abc123"],'
			'"summary":"Fixed it","files_changed":["foo.py"]}'
		)
		full_output = large_output + "\n" + mc_line

		# Track get_output calls to verify draining happens during polling
		get_output_call_count = 0
		check_status_call_count = 0

		async def mock_check_status(handle: WorkerHandle) -> str:
			nonlocal check_status_call_count
			check_status_call_count += 1
			# Simulate running for a few iterations, then completed
			if check_status_call_count < 3:
				return "running"
			return "completed"

		async def mock_get_output(handle: WorkerHandle) -> str:
			nonlocal get_output_call_count
			get_output_call_count += 1
			return full_output

		mock_backend.check_status = mock_check_status  # type: ignore[method-assign]
		mock_backend.get_output = mock_get_output  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		mock_git_proc = AsyncMock()
		mock_git_proc.communicate = AsyncMock(return_value=(b"", b""))
		mock_git_proc.returncode = 0

		with patch("mission_control.worker.asyncio.create_subprocess_exec", return_value=mock_git_proc):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# get_output should have been called during polling (at least once while running)
		# plus once after completion = at least 3 calls total (2 running + 1 final)
		assert get_output_call_count >= 3, (
			f"Expected get_output called during polling, got {get_output_call_count} calls"
		)

		# Unit should complete successfully
		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "completed"

	async def test_branch_exists_until_merge_queue_processes(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""After MR submission, feature branch is NOT deleted -- merge queue needs to fetch it."""
		w, _ = worker_and_unit

		mc_output = (
			'MC_RESULT:{"status":"completed","commits":["abc123"],'
			'"summary":"Fixed it","files_changed":["foo.py"]}'
		)
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		git_calls: list[tuple[str, ...]] = []

		async def tracking_run_git(*args: str, cwd: str) -> bool:
			git_calls.append(args)
			return True

		with patch.object(agent, "_run_git", side_effect=tracking_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# MR should be pending (not yet processed by merge queue)
		mr = db.get_next_merge_request()
		assert mr is not None
		assert mr.status == "pending"

		# No branch -D calls should have happened on the success path
		delete_calls = [c for c in git_calls if c[0] == "branch" and c[1] == "-D"]
		assert len(delete_calls) == 0, f"Branch should NOT be deleted before merge queue fetches: {git_calls}"

	async def test_cleanup_deletes_branches_for_processed_mrs(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""_cleanup_merged_branches deletes branches for MRs that have been processed."""
		w, _ = worker_and_unit

		# Insert a processed (merged) MR for this worker
		from mission_control.models import MergeRequest
		mr = MergeRequest(
			id="mr-old",
			work_unit_id="wu1",
			worker_id=w.id,
			branch_name="mc/unit-old",
			commit_hash="def456",
			status="merged",
		)
		db.insert_merge_request(mr)

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		git_calls: list[tuple[str, ...]] = []

		async def tracking_run_git(*args: str, cwd: str) -> bool:
			git_calls.append(args)
			return True

		with patch.object(agent, "_run_git", side_effect=tracking_run_git):
			await agent._cleanup_merged_branches("/tmp/clone1")  # noqa: SLF001

		# Should delete the old branch
		delete_calls = [c for c in git_calls if c[0] == "branch" and c[1] == "-D"]
		assert len(delete_calls) == 1
		assert delete_calls[0] == ("branch", "-D", "mc/unit-old")

	async def test_cleanup_skips_pending_mr_branches(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""_cleanup_merged_branches does NOT delete branches for pending MRs."""
		w, _ = worker_and_unit

		# Insert a pending MR for this worker (merge queue hasn't fetched yet)
		from mission_control.models import MergeRequest
		mr = MergeRequest(
			id="mr-pending",
			work_unit_id="wu1",
			worker_id=w.id,
			branch_name="mc/unit-pending",
			commit_hash="abc123",
			status="pending",
		)
		db.insert_merge_request(mr)

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		git_calls: list[tuple[str, ...]] = []

		async def tracking_run_git(*args: str, cwd: str) -> bool:
			git_calls.append(args)
			return True

		with patch.object(agent, "_run_git", side_effect=tracking_run_git):
			await agent._cleanup_merged_branches("/tmp/clone1")  # noqa: SLF001

		# Should NOT delete branches for pending MRs
		delete_calls = [c for c in git_calls if c[0] == "branch" and c[1] == "-D"]
		assert len(delete_calls) == 0, f"Should not delete pending MR branches: {git_calls}"

	async def test_cleanup_runs_before_new_unit(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""Cleanup of old branches runs at the start of _execute_unit."""
		w, _ = worker_and_unit

		# Insert a processed (merged) MR for this worker
		from mission_control.models import MergeRequest
		mr = MergeRequest(
			id="mr-old",
			work_unit_id="wu1",
			worker_id=w.id,
			branch_name="mc/unit-old",
			commit_hash="def456",
			status="merged",
		)
		db.insert_merge_request(mr)

		mc_output = (
			'MC_RESULT:{"status":"completed","commits":["abc123"],'
			'"summary":"Fixed it","files_changed":["foo.py"]}'
		)
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		git_calls: list[tuple[str, ...]] = []

		async def tracking_run_git(*args: str, cwd: str) -> bool:
			git_calls.append(args)
			return True

		with patch.object(agent, "_run_git", side_effect=tracking_run_git):
			# Need a second work unit for the second execution
			db.insert_plan(Plan(id="p2", objective="test2", round_id="r1"))
			wu2 = WorkUnit(id="wu2", plan_id="p2", title="Fix more", description="Fix more tests", round_id="r1")
			db.insert_work_unit(wu2)
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# The old branch should be cleaned up BEFORE the new checkout -b
		delete_calls = [(i, c) for i, c in enumerate(git_calls) if c[0] == "branch" and c[1] == "-D"]
		checkout_b_calls = [(i, c) for i, c in enumerate(git_calls) if c[0:2] == ("checkout", "-b")]
		assert len(delete_calls) >= 1, f"Expected cleanup of old branch, got: {git_calls}"
		assert delete_calls[0][1] == ("branch", "-D", "mc/unit-old")
		# Cleanup must happen before new branch creation
		if checkout_b_calls:
			assert delete_calls[0][0] < checkout_b_calls[0][0], "Cleanup should run before branch creation"

	async def test_git_failure_logs_output(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""_run_git logs warning with git error output when command fails."""
		w, _ = worker_and_unit

		# Create a process mock that fails with error output
		mock_git_proc = AsyncMock()
		mock_git_proc.communicate = AsyncMock(return_value=(b"fatal: not a git repository", b""))
		mock_git_proc.returncode = 128

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		with patch("mission_control.worker.asyncio.create_subprocess_exec", return_value=mock_git_proc):
			with caplog.at_level(logging.WARNING, logger="mission_control.worker"):
				result = await agent._run_git("checkout", "main", cwd="/tmp/clone1")  # noqa: SLF001

		assert result is False
		assert "fatal: not a git repository" in caplog.text
		assert "rc=128" in caplog.text

	def test_stop(self, db: Database, config: MissionConfig, mock_backend: MockBackend) -> None:
		w = Worker(id="w1", workspace_path="/tmp/clone")
		agent = WorkerAgent(w, db, config, mock_backend)
		assert agent.running is True
		agent.stop()
		assert agent.running is False
