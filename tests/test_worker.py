"""Tests for the worker agent."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from mission_control.backends.base import WorkerBackend, WorkerHandle
from mission_control.config import MissionConfig, ModelsConfig
from mission_control.db import Database
from mission_control.models import Mission, Plan, Round, Worker, WorkUnit
from mission_control.worker import (
	WorkerAgent,
	_sanitize_braces,
	render_architect_prompt,
	render_editor_prompt,
	render_mission_worker_prompt,
	render_worker_prompt,
)


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

	def test_per_unit_verification_command_override(self, config: MissionConfig) -> None:
		"""Per-unit verification_command overrides config default."""
		unit = WorkUnit(title="X", verification_command="make test-specific")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-x")
		assert "make test-specific" in prompt
		assert "pytest -q" not in prompt


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

	def test_mission_state_injected_in_prompt(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Fix bug", description="Fix the thing")
		state = "## Completed\n- [x] abc123 -- Added auth module"
		prompt = render_mission_worker_prompt(unit, config, "/tmp/ws", "mc/unit-x", mission_state=state)
		assert "## Mission State" in prompt
		assert "Added auth module" in prompt

	def test_overlap_warnings_injected_in_prompt(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Fix bug", description="Fix the thing")
		warnings = "- src/auth.py (also targeted by unit abc12345: Add login)"
		prompt = render_mission_worker_prompt(unit, config, "/tmp/ws", "mc/unit-x", overlap_warnings=warnings)
		assert "## File Locking Warnings" in prompt
		assert "src/auth.py" in prompt


class TestConsolidationConstraint:
	def test_mission_prompt_allows_new_files(self, config: MissionConfig) -> None:
		"""Mission worker prompt allows creating new files."""
		unit = WorkUnit(title="Add feature", description="Add something")
		prompt = render_mission_worker_prompt(unit, config, "/tmp/ws", "mc/unit-x")
		assert "creating new files if needed" in prompt

	def test_editor_prompt_allows_new_files(self, config: MissionConfig) -> None:
		"""Editor prompt allows creating new files."""
		unit = WorkUnit(title="Add feature", description="Add something")
		prompt = render_editor_prompt(unit, config, "/tmp/ws", architect_output="Change X")
		assert "creating new files if needed" in prompt


class TestResearchPromptSelection:
	def test_research_unit_uses_research_template(self, config: MissionConfig) -> None:
		"""Research units should use the research prompt template."""
		unit = WorkUnit(title="Investigate API", description="Explore the API surface", unit_type="research")
		prompt = render_mission_worker_prompt(unit, config, "/tmp/ws", "mc/unit-x")
		assert "research agent" in prompt.lower()
		assert "EXPLORATION and DISCOVERY" in prompt
		assert "Do NOT commit code changes" in prompt


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

		# Set max_attempts=1 so the first failure is permanent
		wu = db.get_work_unit("wu1")
		assert wu is not None
		wu.max_attempts = 1
		db.update_work_unit(wu)

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

		# Set max_attempts=1 so the first failure is permanent
		wu = db.get_work_unit("wu1")
		assert wu is not None
		wu.max_attempts = 1
		db.update_work_unit(wu)

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

		# Set max_attempts=1 so the first failure is permanent
		wu = db.get_work_unit("wu1")
		assert wu is not None
		wu.max_attempts = 1
		db.update_work_unit(wu)

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

	async def test_failed_unit_resets_to_pending_if_retries_remain(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""Failed unit with attempt < max_attempts is reset to pending for retry."""
		w, _ = worker_and_unit

		# Default max_attempts=3, so first failure should reset to pending
		mock_backend.check_status = AsyncMock(return_value="failed")  # type: ignore[method-assign]
		mock_backend.get_output = AsyncMock(return_value="Error: transient failure")  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "pending", f"Expected pending for retry, got {result.status}"
		assert result.attempt == 1
		assert result.claimed_at is None
		assert result.started_at is None

	async def test_permanently_failed_unit_stays_failed(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""Failed unit with attempt >= max_attempts stays permanently failed."""
		w, _ = worker_and_unit

		# Set max_attempts=1 so the first failure is permanent
		wu = db.get_work_unit("wu1")
		assert wu is not None
		wu.max_attempts = 1
		db.update_work_unit(wu)

		mock_backend.check_status = AsyncMock(return_value="failed")  # type: ignore[method-assign]
		mock_backend.get_output = AsyncMock(return_value="Error: permanent failure")  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "failed"
		assert result.attempt == 1
		assert result.finished_at is not None

	async def test_completed_no_commits_treated_as_success(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""Completed status with no commits is a no-op success, not a failure."""
		w, _ = worker_and_unit

		# Backend returns completed with no commits
		mc_output = (
			'MC_RESULT:{"status":"completed","commits":[],'
			'"summary":"Task already done, no changes needed","files_changed":[]}'
		)
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "completed", f"No-op should be completed, got {result.status}"
		assert result.finished_at is not None
		assert w.units_completed == 1
		assert w.units_failed == 0

		# No merge request should be created for no-op
		mr = db.get_next_merge_request()
		assert mr is None, "No MR should exist for no-op completion"

	async def test_run_loop_survives_execute_unit_exception(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend, caplog: pytest.LogCaptureFixture,
	) -> None:
		"""WorkerAgent.run() continues looping after _execute_unit raises an unexpected exception."""
		w, _ = worker_and_unit

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		call_count = 0

		async def explode_then_stop(unit: WorkUnit) -> None:
			nonlocal call_count
			call_count += 1
			if call_count == 1:
				raise RuntimeError("unexpected kaboom")
			# On second call, just stop the agent
			agent.running = False

		with (
			patch.object(agent, "_execute_unit", side_effect=explode_then_stop),
			caplog.at_level(logging.ERROR, logger="mission_control.worker"),
		):
			# Insert a second unit so the loop has work after the first failure
			db.insert_plan(Plan(id="p2", objective="test2", round_id="r1"))
			wu2 = WorkUnit(id="wu2", plan_id="p2", title="Second task", description="test", round_id="r1")
			db.insert_work_unit(wu2)

			await agent.run()

		# Worker survived and processed both units
		assert call_count == 2, f"Expected 2 calls to _execute_unit, got {call_count}"
		assert w.status == "idle"
		assert w.current_unit_id is None

		# Exception was logged
		assert "unexpected kaboom" in caplog.text
		assert "Unexpected error executing unit" in caplog.text


class TestModelSelection:
	"""Tests for worker_model override in _execute_unit command construction."""

	async def test_uses_worker_model_when_configured(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""When models.worker_model is set, it overrides scheduler.model."""
		w, _ = worker_and_unit
		config.models = ModelsConfig(worker_model="sonnet")
		config.scheduler.model = "opus"

		mc_output = (
			'MC_RESULT:{"status":"completed","commits":["abc123"],'
			'"summary":"Fixed it","files_changed":["foo.py"]}'
		)
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]
		mock_backend.spawn = AsyncMock(  # type: ignore[method-assign]
			return_value=WorkerHandle(worker_id=w.id, pid=12345, workspace_path="/tmp/mock-workspace"),
		)

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# Verify the command passed to spawn uses worker_model
		spawn_call = mock_backend.spawn.call_args
		cmd = spawn_call.kwargs.get("command") or spawn_call[0][2]
		model_idx = cmd.index("--model")
		assert cmd[model_idx + 1] == "sonnet"

	async def test_falls_back_to_scheduler_model_without_models_config(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""When models config is absent, falls back to scheduler.model."""
		w, _ = worker_and_unit
		config.scheduler.model = "haiku"
		# Default MissionConfig has ModelsConfig with worker_model="opus",
		# so remove models to test the fallback path
		object.__setattr__(config, "models", None)

		mc_output = (
			'MC_RESULT:{"status":"completed","commits":["abc123"],'
			'"summary":"Fixed it","files_changed":["foo.py"]}'
		)
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]
		mock_backend.spawn = AsyncMock(  # type: ignore[method-assign]
			return_value=WorkerHandle(worker_id=w.id, pid=12345, workspace_path="/tmp/mock-workspace"),
		)

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		spawn_call = mock_backend.spawn.call_args
		cmd = spawn_call.kwargs.get("command") or spawn_call[0][2]
		model_idx = cmd.index("--model")
		assert cmd[model_idx + 1] == "haiku"

	async def test_default_models_config_uses_worker_model(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""Default ModelsConfig (worker_model='opus') is used over scheduler.model."""
		w, _ = worker_and_unit
		# Default ModelsConfig has worker_model="opus"
		config.models = ModelsConfig()
		config.scheduler.model = "haiku"

		mc_output = (
			'MC_RESULT:{"status":"completed","commits":["abc123"],'
			'"summary":"Fixed it","files_changed":["foo.py"]}'
		)
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]
		mock_backend.spawn = AsyncMock(  # type: ignore[method-assign]
			return_value=WorkerHandle(worker_id=w.id, pid=12345, workspace_path="/tmp/mock-workspace"),
		)

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		spawn_call = mock_backend.spawn.call_args
		cmd = spawn_call.kwargs.get("command") or spawn_call[0][2]
		model_idx = cmd.index("--model")
		assert cmd[model_idx + 1] == "opus"


class TestArchitectEditorPrompts:
	"""Tests for architect and editor prompt rendering."""

	def test_architect_prompt_contains_analysis_instructions(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Add auth", description="Add authentication module", files_hint="src/auth.py")
		prompt = render_architect_prompt(unit, config, "/tmp/ws")
		assert "architect" in prompt.lower()
		assert "Do NOT write code" in prompt
		assert "Do NOT modify any files" in prompt
		assert "Add auth" in prompt
		assert "src/auth.py" in prompt

	def test_editor_prompt_contains_architect_output(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Add auth", description="Add authentication module", files_hint="src/auth.py")
		architect_analysis = "Modify src/auth.py to add JWT token validation in the login() function."
		prompt = render_editor_prompt(unit, config, "/tmp/ws", architect_output=architect_analysis)
		assert "Architect Analysis" in prompt
		assert architect_analysis in prompt
		assert "Add auth" in prompt
		assert "src/auth.py" in prompt
		assert "Commit when done" in prompt

	def test_architect_prompt_includes_context_blocks(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Fix bug", description="Fix the thing")
		prompt = render_architect_prompt(
			unit, config, "/tmp/ws",
			experience_context="Past: auth was tricky",
			mission_state="## Done\n- login",
			overlap_warnings="- src/auth.py locked",
		)
		assert "## Relevant Past Experiences" in prompt
		assert "## Mission State" in prompt
		assert "## File Locking Warnings" in prompt

	def test_editor_prompt_includes_context_blocks(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Fix bug", description="Fix the thing")
		prompt = render_editor_prompt(
			unit, config, "/tmp/ws", architect_output="Change X in Y",
			experience_context="Past: auth was tricky",
			mission_state="## Done\n- login",
			overlap_warnings="- src/auth.py locked",
		)
		assert "## Relevant Past Experiences" in prompt
		assert "## Mission State" in prompt
		assert "## File Locking Warnings" in prompt


class TestArchitectEditorMode:
	"""Tests for the two-pass architect/editor execution mode."""

	async def test_single_pass_when_mode_disabled(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""When architect_editor_mode is False, only one session is spawned."""
		w, _ = worker_and_unit
		config.models = ModelsConfig(architect_editor_mode=False)

		mc_output = (
			'MC_RESULT:{"status":"completed","commits":["abc123"],'
			'"summary":"Fixed it","files_changed":["foo.py"]}'
		)
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]
		mock_backend.spawn = AsyncMock(  # type: ignore[method-assign]
			return_value=WorkerHandle(worker_id=w.id, pid=12345, workspace_path="/tmp/mock-workspace"),
		)

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# Only one spawn call (single pass)
		assert mock_backend.spawn.call_count == 1

		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "completed"

	async def test_two_pass_when_mode_enabled(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""When architect_editor_mode is True, two sessions are spawned sequentially."""
		w, _ = worker_and_unit
		config.models = ModelsConfig(architect_editor_mode=True)

		architect_output = (
			'Analysis: modify src/foo.py\n'
			'MC_RESULT:{"status":"completed","commits":[],'
			'"summary":"architectural analysis","discoveries":["found pattern"],"files_changed":[]}'
		)
		editor_output = (
			'MC_RESULT:{"status":"completed","commits":["def456"],'
			'"summary":"Implemented changes","files_changed":["src/foo.py"]}'
		)

		spawn_count = 0

		async def mock_spawn(
			worker_id: str, workspace_path: str, command: list[str], timeout: int,
		) -> WorkerHandle:
			nonlocal spawn_count
			spawn_count += 1
			return WorkerHandle(worker_id=worker_id, pid=12345 + spawn_count, workspace_path=workspace_path)

		get_output_count = 0

		async def mock_get_output(handle: WorkerHandle) -> str:
			nonlocal get_output_count
			get_output_count += 1
			# First session (architect) returns analysis, second (editor) returns implementation
			if handle.pid == 12346:
				return architect_output
			return editor_output

		mock_backend.spawn = mock_spawn  # type: ignore[method-assign]
		mock_backend.get_output = mock_get_output  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# Two spawns: architect + editor
		assert spawn_count == 2

		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "completed"
		assert result.commit_hash == "def456"
		assert result.output_summary == "Implemented changes"

	async def test_architect_failure_falls_back_to_single_pass(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""When the architect pass fails, falls back to single-pass and completes."""
		w, _ = worker_and_unit
		config.models = ModelsConfig(architect_editor_mode=True)

		fallback_output = (
			'MC_RESULT:{"status":"completed","commits":["fallback1"],'
			'"summary":"Completed via fallback","files_changed":["src/foo.py"]}'
		)

		spawn_count = 0

		async def mock_spawn(
			worker_id: str, workspace_path: str, command: list[str], timeout: int,
		) -> WorkerHandle:
			nonlocal spawn_count
			spawn_count += 1
			return WorkerHandle(worker_id=worker_id, pid=10000 + spawn_count, workspace_path=workspace_path)

		async def mock_check_status(handle: WorkerHandle) -> str:
			# Architect pass (first spawn) fails, fallback pass (second spawn) completes
			if handle.pid == 10001:
				return "failed"
			return "completed"

		async def mock_get_output(handle: WorkerHandle) -> str:
			if handle.pid == 10001:
				return "Error: analysis failed"
			return fallback_output

		mock_backend.spawn = mock_spawn  # type: ignore[method-assign]
		mock_backend.check_status = mock_check_status  # type: ignore[method-assign]
		mock_backend.get_output = mock_get_output  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# Two spawns: architect (failed) + fallback single-pass
		assert spawn_count == 2

		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "completed"
		assert result.commit_hash == "fallback1"
		assert result.output_summary == "Completed via fallback"

	async def test_architect_output_passed_to_editor_prompt(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""The architect output is included in the editor session's prompt."""
		w, _ = worker_and_unit
		config.models = ModelsConfig(architect_editor_mode=True)

		architect_analysis = "SPECIFIC_ANALYSIS: Change function foo() in src/bar.py to accept an extra param."
		architect_output = (
			f'{architect_analysis}\n'
			'MC_RESULT:{"status":"completed","commits":[],'
			'"summary":"analysis","discoveries":[],"files_changed":[]}'
		)
		editor_output = (
			'MC_RESULT:{"status":"completed","commits":["abc"],'
			'"summary":"done","files_changed":["src/bar.py"]}'
		)

		spawn_calls: list[list[str]] = []

		async def mock_spawn(
			worker_id: str, workspace_path: str, command: list[str], timeout: int,
		) -> WorkerHandle:
			spawn_calls.append(command)
			return WorkerHandle(worker_id=worker_id, pid=10000 + len(spawn_calls), workspace_path=workspace_path)

		async def mock_get_output(handle: WorkerHandle) -> str:
			if handle.pid == 10001:
				return architect_output
			return editor_output

		mock_backend.spawn = mock_spawn  # type: ignore[method-assign]
		mock_backend.get_output = mock_get_output  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		assert len(spawn_calls) == 2
		# First call is architect (should contain "Do NOT write code")
		architect_cmd_prompt = spawn_calls[0][-1]
		assert "Do NOT write code" in architect_cmd_prompt
		# Second call is editor (should contain the architect's output, with braces sanitized)
		editor_cmd_prompt = spawn_calls[1][-1]
		assert "SPECIFIC_ANALYSIS" in editor_cmd_prompt
		assert "Architect Analysis" in editor_cmd_prompt

	async def test_research_units_skip_architect_mode(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""Research units always use single-pass even when architect_editor_mode is True."""
		w, _ = worker_and_unit
		config.models = ModelsConfig(architect_editor_mode=True)

		# Set unit type to research
		wu = db.get_work_unit("wu1")
		assert wu is not None
		wu.unit_type = "research"
		db.update_work_unit(wu)

		mc_output = (
			'MC_RESULT:{"status":"completed","commits":[],'
			'"summary":"Research findings","discoveries":["found X"],"files_changed":[]}'
		)

		spawn_calls: list[list[str]] = []

		async def mock_spawn(
			worker_id: str, workspace_path: str, command: list[str], timeout: int,
		) -> WorkerHandle:
			spawn_calls.append(command)
			return WorkerHandle(worker_id=worker_id, pid=12345, workspace_path=workspace_path)

		mock_backend.spawn = mock_spawn  # type: ignore[method-assign]
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# Only one spawn (single-pass) even though architect_editor_mode=True
		assert len(spawn_calls) == 1
		prompt_text = spawn_calls[0][-1]
		# Should NOT contain architect instructions
		assert "Do NOT write code" not in prompt_text
		# Should use the worker prompt template (not architect)
		assert "MC_RESULT" in prompt_text

	async def test_experiment_units_skip_architect_mode(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""Experiment units always use single-pass even when architect_editor_mode is True."""
		w, _ = worker_and_unit
		config.models = ModelsConfig(architect_editor_mode=True)

		wu = db.get_work_unit("wu1")
		assert wu is not None
		wu.unit_type = "experiment"
		db.update_work_unit(wu)

		mc_output = (
			'MC_RESULT:{"status":"completed","commits":[],'
			'"summary":"Experiment done","discoveries":["found X"],"files_changed":[]}'
		)
		mock_backend.spawn = AsyncMock(  # type: ignore[method-assign]
			return_value=WorkerHandle(worker_id=w.id, pid=12345, workspace_path="/tmp/mock-workspace"),
		)
		mock_backend.get_output = AsyncMock(return_value=mc_output)  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# Only one spawn call (single-pass)
		assert mock_backend.spawn.call_count == 1

	async def test_architect_timeout_falls_back_to_single_pass(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""When architect pass times out, falls back to single-pass."""
		from mission_control.worker import _SpawnError

		w, _ = worker_and_unit
		config.models = ModelsConfig(architect_editor_mode=True)

		fallback_output = (
			'MC_RESULT:{"status":"completed","commits":["timeout_fb"],'
			'"summary":"Done after timeout fallback","files_changed":["a.py"]}'
		)

		spawn_count = 0

		async def mock_spawn(
			worker_id: str, workspace_path: str, command: list[str], timeout: int,
		) -> WorkerHandle:
			nonlocal spawn_count
			spawn_count += 1
			if spawn_count == 1:
				raise _SpawnError("Timed out after 300s")
			return WorkerHandle(worker_id=worker_id, pid=20000, workspace_path=workspace_path)

		mock_backend.spawn = mock_spawn  # type: ignore[method-assign]
		mock_backend.get_output = AsyncMock(return_value=fallback_output)  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		# Architect timed out, then fallback succeeded
		assert spawn_count == 2
		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "completed"
		assert result.commit_hash == "timeout_fb"

	async def test_mc_result_parsed_from_editor_output(
		self, db: Database, config: MissionConfig, worker_and_unit: tuple[Worker, WorkUnit],
		mock_backend: MockBackend,
	) -> None:
		"""MC_RESULT is correctly parsed from the editor (second) pass output."""
		w, _ = worker_and_unit
		config.models = ModelsConfig(architect_editor_mode=True)

		architect_output = (
			'File analysis complete.\n'
			'MC_RESULT:{"status":"completed","commits":[],'
			'"summary":"analysis done","discoveries":["pattern found"],"files_changed":[]}'
		)
		editor_output = (
			'Implemented all changes.\n'
			'MC_RESULT:{"status":"completed","commits":["ed1","ed2"],'
			'"summary":"Added auth and tests","discoveries":["edge case in login"],'
			'"concerns":["needs load testing"],"files_changed":["src/auth.py","tests/test_auth.py"]}'
		)

		spawn_count = 0

		async def mock_spawn(
			worker_id: str, workspace_path: str, command: list[str], timeout: int,
		) -> WorkerHandle:
			nonlocal spawn_count
			spawn_count += 1
			return WorkerHandle(worker_id=worker_id, pid=30000 + spawn_count, workspace_path=workspace_path)

		async def mock_get_output(handle: WorkerHandle) -> str:
			if handle.pid == 30001:
				return architect_output
			return editor_output

		mock_backend.spawn = mock_spawn  # type: ignore[method-assign]
		mock_backend.get_output = mock_get_output  # type: ignore[method-assign]

		agent = WorkerAgent(w, db, config, mock_backend, heartbeat_interval=9999)

		async def ok_run_git(*args: str, cwd: str) -> bool:
			return True

		with patch.object(agent, "_run_git", side_effect=ok_run_git):
			unit = db.claim_work_unit(w.id)
			assert unit is not None
			await agent._execute_unit(unit)  # noqa: SLF001

		result = db.get_work_unit("wu1")
		assert result is not None
		assert result.status == "completed"
		# commit_hash should be from editor output (first commit)
		assert result.commit_hash == "ed1"
		assert result.output_summary == "Added auth and tests"
		# Handoff should exist with editor's data
		assert result.handoff_id is not None


class TestSanitizeBraces:
	def test_no_braces_unchanged(self) -> None:
		assert _sanitize_braces("hello world") == "hello world"

	def test_single_braces_escaped(self) -> None:
		assert _sanitize_braces("use {foo} here") == "use {{foo}} here"

	def test_already_doubled_braces_still_escaped(self) -> None:
		assert _sanitize_braces("{{foo}}") == "{{{{foo}}}}"

	def test_empty_string(self) -> None:
		assert _sanitize_braces("") == ""

	def test_mixed_content(self) -> None:
		assert _sanitize_braces("a {b} c } d { e") == "a {{b}} c }} d {{ e"


class TestBraceSanitizationInPrompts:
	"""Verify that braces in user-provided fields don't crash .format()."""

	def test_render_worker_prompt_with_braces_no_crash(self, config: MissionConfig) -> None:
		"""Braces in user-provided fields must not raise KeyError/ValueError."""
		unit = WorkUnit(
			title="Fix {broken} thing",
			description="Handle dict {key: value}",
			files_hint="src/{utils}.py",
			verification_hint="run {test}",
		)
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-abc", context="ctx {data}")
		assert "Fix" in prompt
		assert "broken" in prompt
		assert "Handle dict" in prompt
		assert "src/" in prompt
		assert "run" in prompt
		assert "ctx" in prompt

	def test_render_mission_worker_prompt_with_braces_no_crash(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Add {feature}", description="Implement {thing}")
		prompt = render_mission_worker_prompt(
			unit, config, "/tmp/ws", "mc/unit-x",
			context="context {data}",
			experience_context="past {exp}",
			mission_state="state {info}",
			overlap_warnings="warn {overlap}",
		)
		assert "Add" in prompt
		assert "feature" in prompt
		assert "Implement" in prompt
		assert "context" in prompt
		assert "past" in prompt
		assert "state" in prompt
		assert "warn" in prompt

	def test_render_architect_prompt_with_braces_no_crash(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Analyze {module}", description="Check {pattern}")
		prompt = render_architect_prompt(
			unit, config, "/tmp/ws",
			context="ctx {x}",
			experience_context="exp {y}",
			mission_state="ms {z}",
			overlap_warnings="ow {w}",
		)
		assert "Analyze" in prompt
		assert "module" in prompt
		assert "Check" in prompt
		assert "ctx" in prompt

	def test_render_editor_prompt_with_braces_no_crash(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Edit {file}", description="Change {func}")
		prompt = render_editor_prompt(
			unit, config, "/tmp/ws",
			architect_output="modify {foo} in bar()",
			context="ctx {a}",
			experience_context="exp {b}",
			mission_state="ms {c}",
			overlap_warnings="ow {d}",
		)
		assert "Edit" in prompt
		assert "file" in prompt
		assert "Change" in prompt
		assert "modify" in prompt
		assert "foo" in prompt
		assert "ctx" in prompt

	def test_research_prompt_with_braces_no_crash(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Research {api}", description="Explore {endpoints}", unit_type="research")
		prompt = render_mission_worker_prompt(
			unit, config, "/tmp/ws", "mc/unit-x",
			mission_state="state with {braces}",
		)
		assert "Research" in prompt
		assert "api" in prompt
		assert "state with" in prompt
		assert "braces" in prompt

	def test_experiment_prompt_with_braces_no_crash(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="Try {approach}", description="Compare {options}", unit_type="experiment")
		prompt = render_mission_worker_prompt(
			unit, config, "/tmp/ws", "mc/unit-x",
			overlap_warnings="file {lock}",
		)
		assert "Try" in prompt
		assert "approach" in prompt
		assert "file" in prompt
		assert "lock" in prompt

	def test_verification_command_with_braces_no_crash(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X", verification_command="make test ARGS={verbose}")
		prompt = render_worker_prompt(unit, config, "/tmp/clone", "mc/unit-x")
		assert "make test ARGS" in prompt
		assert "verbose" in prompt


class TestPipInstallConstraint:
	"""All worker prompt templates forbid pip install to protect symlinked venvs."""

	_PIP_CONSTRAINT = "Do NOT run `pip install`"

	def test_worker_prompt_forbids_pip(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X", description="Y")
		prompt = render_worker_prompt(unit, config, "/tmp/ws", "mc/unit-x")
		assert self._PIP_CONSTRAINT in prompt

	def test_mission_worker_prompt_forbids_pip(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X", description="Y")
		prompt = render_mission_worker_prompt(unit, config, "/tmp/ws", "mc/unit-x")
		assert self._PIP_CONSTRAINT in prompt

	def test_research_prompt_forbids_pip(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X", description="Y", unit_type="research")
		prompt = render_mission_worker_prompt(unit, config, "/tmp/ws", "mc/unit-x")
		assert self._PIP_CONSTRAINT in prompt

	def test_experiment_prompt_forbids_pip(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X", description="Y", unit_type="experiment")
		prompt = render_mission_worker_prompt(unit, config, "/tmp/ws", "mc/unit-x")
		assert self._PIP_CONSTRAINT in prompt

	def test_architect_prompt_forbids_pip(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X", description="Y")
		prompt = render_architect_prompt(unit, config, "/tmp/ws")
		assert self._PIP_CONSTRAINT in prompt

	def test_editor_prompt_forbids_pip(self, config: MissionConfig) -> None:
		unit = WorkUnit(title="X", description="Y")
		prompt = render_editor_prompt(unit, config, "/tmp/ws", architect_output="Do Z")
		assert self._PIP_CONSTRAINT in prompt
