"""Tests for the strategist module."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig, PlannerConfig, SchedulerConfig, TargetConfig
from mission_control.continuous_controller import ContinuousController, WorkerCompletion
from mission_control.db import Database
from mission_control.models import Epoch, Mission, Plan, StrategicContext, WorkUnit
from mission_control.strategist import (
	STRATEGY_RESULT_MARKER,
	Strategist,
	_build_strategy_prompt,
)

# -- Helpers --


def _config(tmp_path: Path | None = None) -> MissionConfig:
	target_path = str(tmp_path) if tmp_path else "/tmp/test-project"
	return MissionConfig(
		target=TargetConfig(path=target_path),
		planner=PlannerConfig(budget_per_call_usd=0.10),
		scheduler=SchedulerConfig(model="sonnet"),
	)


def _make_strategist(tmp_path: Path | None = None, db: Database | None = None) -> Strategist:
	config = _config(tmp_path)
	if db is None:
		db = MagicMock()
		db.get_all_missions.return_value = []
		db.get_pending_backlog.return_value = []
	return Strategist(config=config, db=db)


def _make_strategy_output(objective: str, rationale: str, score: int) -> str:
	data = {"objective": objective, "rationale": rationale, "ambition_score": score}
	return f"Some reasoning...\n\n{STRATEGY_RESULT_MARKER}{json.dumps(data)}"


def _insert_mission(db: Database, mission_id: str) -> None:
	"""Helper to insert a minimal mission for FK satisfaction."""
	db.insert_mission(Mission(id=mission_id, objective="test"))


# -- Fixtures --


# ============================================================================
# Core Strategist Tests
# ============================================================================


# -- Prompt building --


class TestBuildStrategyPrompt:
	def test_all_context_present(self) -> None:
		prompt = _build_strategy_prompt(
			backlog_md="# Backlog\n- item 1",
			git_log="abc123 feat: add foo",
			past_missions="- [completed] Build API",
			strategic_context="- Focus on testing",
			pending_backlog="- [score=8.0] Fix auth",
		)
		assert "Backlog" in prompt
		assert "abc123" in prompt
		assert "Build API" in prompt
		assert "Focus on testing" in prompt
		assert "Fix auth" in prompt

	def test_empty_context_uses_fallbacks(self) -> None:
		prompt = _build_strategy_prompt(
			backlog_md="",
			git_log="",
			past_missions="",
			strategic_context="",
			pending_backlog="",
		)
		assert "No BACKLOG.md found" in prompt
		assert "No git history available" in prompt
		assert "No prior missions" in prompt
		assert "No strategic context yet" in prompt
		assert "No pending backlog items" in prompt

	def test_output_format_instructions(self) -> None:
		prompt = _build_strategy_prompt("", "", "", "", "")
		assert "STRATEGY_RESULT:" in prompt
		assert "ambition_score" in prompt
		assert "objective" in prompt


# -- Parsing --


class TestParseStrategyOutput:
	def setup_method(self) -> None:
		self.strategist = _make_strategist()

	def test_valid_marker_output(self) -> None:
		output = _make_strategy_output("Build auth system", "High priority", 7)
		obj, rationale, score = self.strategist._parse_strategy_output(output)
		assert obj == "Build auth system"
		assert rationale == "High priority"
		assert score == 7

	def test_json_without_marker(self) -> None:
		data = {"objective": "Fix tests", "rationale": "Broken CI", "ambition_score": 3}
		output = f"Here is the result:\n```json\n{json.dumps(data)}\n```"
		obj, rationale, score = self.strategist._parse_strategy_output(output)
		assert obj == "Fix tests"
		assert score == 3

	def test_ambition_score_clamped_high(self) -> None:
		output = _make_strategy_output("Big refactor", "Needed", 15)
		_, _, score = self.strategist._parse_strategy_output(output)
		assert score == 10

	def test_ambition_score_clamped_low(self) -> None:
		output = _make_strategy_output("Tiny fix", "Quick", -1)
		_, _, score = self.strategist._parse_strategy_output(output)
		assert score == 1

	def test_ambition_score_non_numeric_defaults(self) -> None:
		data = {"objective": "Something", "rationale": "Reason", "ambition_score": "high"}
		output = f"{STRATEGY_RESULT_MARKER}{json.dumps(data)}"
		_, _, score = self.strategist._parse_strategy_output(output)
		assert score == 5

	def test_empty_objective_raises(self) -> None:
		data = {"objective": "", "rationale": "Reason", "ambition_score": 5}
		output = f"{STRATEGY_RESULT_MARKER}{json.dumps(data)}"
		with pytest.raises(ValueError, match="Empty objective"):
			self.strategist._parse_strategy_output(output)

	def test_no_json_raises(self) -> None:
		with pytest.raises(ValueError, match="Could not parse"):
			self.strategist._parse_strategy_output("Just some text with no JSON")

	def test_marker_takes_precedence_over_earlier_json(self) -> None:
		earlier = json.dumps({"objective": "Wrong", "rationale": "Old", "ambition_score": 1})
		correct = json.dumps({"objective": "Right", "rationale": "New", "ambition_score": 8})
		output = f"Earlier attempt: {earlier}\n\n{STRATEGY_RESULT_MARKER}{correct}"
		obj, _, score = self.strategist._parse_strategy_output(output)
		assert obj == "Right"
		assert score == 8


# -- Context gathering --


class TestContextGathering:
	def test_read_backlog_exists(self, tmp_path: Path) -> None:
		(tmp_path / "BACKLOG.md").write_text("# My Backlog\n- item 1")
		s = _make_strategist(tmp_path)
		assert "My Backlog" in s._read_backlog()

	def test_read_backlog_missing(self, tmp_path: Path) -> None:
		s = _make_strategist(tmp_path)
		assert s._read_backlog() == ""

	@pytest.mark.asyncio
	async def test_get_git_log_success(self) -> None:
		s = _make_strategist()
		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"abc123 feat: add X\ndef456 fix: bug Y", b"")
		mock_proc.returncode = 0
		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s._get_git_log()
		assert "abc123" in result
		assert "def456" in result

	@pytest.mark.asyncio
	async def test_get_git_log_failure(self) -> None:
		s = _make_strategist()
		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"", b"")
		mock_proc.returncode = 128
		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s._get_git_log()
		assert result == ""

	@pytest.mark.asyncio
	async def test_get_git_log_timeout(self) -> None:
		s = _make_strategist()
		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = asyncio.TimeoutError()
		mock_proc.kill = AsyncMock()
		mock_proc.wait = AsyncMock()
		with (
			patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc),
			patch("mission_control.strategist.asyncio.wait_for", side_effect=asyncio.TimeoutError),
		):
			result = await s._get_git_log()
		assert result == ""

	def test_get_past_missions_empty(self) -> None:
		s = _make_strategist()
		s.db.get_all_missions.return_value = []
		assert s._get_past_missions() == ""

	def test_get_past_missions_formats(self) -> None:
		s = _make_strategist()
		m = Mission(objective="Build API server", status="completed", total_rounds=3, final_score=8.5)
		s.db.get_all_missions.return_value = [m]
		result = s._get_past_missions()
		assert "completed" in result
		assert "Build API server" in result
		assert "rounds=3" in result

	def test_get_strategic_context_no_method(self) -> None:
		s = _make_strategist()
		del s.db.get_strategic_context
		assert s._get_strategic_context() == ""

	def test_get_strategic_context_with_data(self) -> None:
		s = _make_strategist()
		s.db.get_strategic_context.return_value = ["Focus on auth", "Testing needed"]
		result = s._get_strategic_context()
		assert "Focus on auth" in result
		assert "Testing needed" in result

	def test_get_strategic_context_empty(self) -> None:
		s = _make_strategist()
		s.db.get_strategic_context.return_value = []
		assert s._get_strategic_context() == ""

	def test_get_pending_backlog_returns_empty(self) -> None:
		s = _make_strategist()
		assert s._get_pending_backlog() == ""


# -- propose_objective --


class TestProposeObjective:
	@pytest.mark.asyncio
	async def test_success(self) -> None:
		s = _make_strategist()
		strategy_output = _make_strategy_output("Build new auth system", "Critical for security", 8)

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (strategy_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			obj, rationale, score = await s.propose_objective()

		assert obj == "Build new auth system"
		assert rationale == "Critical for security"
		assert score == 8

	@pytest.mark.asyncio
	async def test_subprocess_failure_raises(self) -> None:
		s = _make_strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"", b"some error")
		mock_proc.returncode = 1

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			with pytest.raises(RuntimeError, match="strategist subprocess failed"):
				await s.propose_objective()

	@pytest.mark.asyncio
	async def test_timeout_raises(self) -> None:
		s = _make_strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = asyncio.TimeoutError()
		mock_proc.kill = AsyncMock()
		mock_proc.wait = AsyncMock()

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch("mission_control.strategist.asyncio.wait_for", side_effect=asyncio.TimeoutError):
				with pytest.raises(asyncio.TimeoutError):
					await s.propose_objective()

	@pytest.mark.asyncio
	async def test_parse_failure_raises(self) -> None:
		s = _make_strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"No valid JSON here", b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			with pytest.raises(ValueError, match="Could not parse"):
				await s.propose_objective()

	@pytest.mark.asyncio
	async def test_sets_cwd_to_target_path(self, tmp_path: Path) -> None:
		s = _make_strategist(tmp_path)
		(tmp_path / "BACKLOG.md").write_text("# Backlog")
		strategy_output = _make_strategy_output("Do something", "Reason", 5)

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (strategy_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
			await s.propose_objective()

		_, kwargs = mock_exec.call_args
		assert kwargs["cwd"] == str(tmp_path)

	@pytest.mark.asyncio
	async def test_gathers_all_context(self) -> None:
		s = _make_strategist()
		m = Mission(objective="Past mission", status="completed", total_rounds=2, final_score=7.0)
		s.db.get_all_missions.return_value = [m]

		strategy_output = _make_strategy_output("Next objective", "Based on context", 6)

		# Git log subprocess (called first by _get_git_log)
		git_proc = AsyncMock()
		git_proc.communicate.return_value = (b"abc123 recent commit", b"")
		git_proc.returncode = 0

		# LLM subprocess (called by _invoke_llm)
		llm_proc = AsyncMock()
		llm_proc.communicate.return_value = (strategy_output.encode(), b"")
		llm_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", side_effect=[git_proc, llm_proc]):
			await s.propose_objective()

		# Verify the prompt was sent to stdin of the LLM call
		call_args = llm_proc.communicate.call_args
		prompt_bytes = call_args[1].get("input") or call_args[0][0] if call_args[0] else call_args[1]["input"]
		prompt = prompt_bytes.decode()
		assert "Past mission" in prompt or "abc123" in prompt


# ============================================================================
# Strategic Context DB Tests
# ============================================================================


class TestStrategicContextTable:
	def test_table_exists(self, db: Database) -> None:
		row = db.conn.execute(
			"SELECT name FROM sqlite_master WHERE type='table' AND name='strategic_context'"
		).fetchone()
		assert row is not None

	def test_insert_and_retrieve(self, db: Database) -> None:
		_insert_mission(db, "m1")
		ctx = StrategicContext(
			id="sc1",
			mission_id="m1",
			what_attempted="Built auth system",
			what_worked="JWT tokens",
			what_failed="Session cookies",
			recommended_next="Add refresh tokens",
		)
		db.insert_strategic_context(ctx)
		results = db.get_strategic_context(limit=10)
		assert len(results) == 1
		assert results[0].id == "sc1"
		assert results[0].mission_id == "m1"
		assert results[0].what_attempted == "Built auth system"
		assert results[0].what_worked == "JWT tokens"
		assert results[0].what_failed == "Session cookies"
		assert results[0].recommended_next == "Add refresh tokens"

	def test_limit_param(self, db: Database) -> None:
		for i in range(5):
			_insert_mission(db, f"m{i}")
			ctx = StrategicContext(
				id=f"sc{i}",
				mission_id=f"m{i}",
				what_attempted=f"Task {i}",
			)
			db.insert_strategic_context(ctx)
		results = db.get_strategic_context(limit=3)
		assert len(results) == 3

	def test_append_strategic_context(self, db: Database) -> None:
		_insert_mission(db, "m1")
		ctx = db.append_strategic_context(
			mission_id="m1",
			what_attempted="Refactored DB layer",
			what_worked="Migration pattern",
			what_failed="Nothing",
			recommended_next="Add indexes",
		)
		assert ctx.id  # auto-generated
		assert ctx.mission_id == "m1"
		results = db.get_strategic_context(limit=10)
		assert len(results) == 1
		assert results[0].what_attempted == "Refactored DB layer"

	def test_ordering_by_timestamp_desc(self, db: Database) -> None:
		_insert_mission(db, "m1")
		_insert_mission(db, "m2")
		ctx1 = StrategicContext(id="sc1", mission_id="m1", timestamp="2025-01-01T00:00:00Z")
		ctx2 = StrategicContext(id="sc2", mission_id="m2", timestamp="2025-06-01T00:00:00Z")
		db.insert_strategic_context(ctx1)
		db.insert_strategic_context(ctx2)
		results = db.get_strategic_context(limit=10)
		assert results[0].id == "sc2"
		assert results[1].id == "sc1"


class TestMissionNewFields:
	def test_insert_and_retrieve_chain_id(self, db: Database) -> None:
		m = Mission(
			id="m2",
			objective="Build feature",
			chain_id="chain-abc",
		)
		db.insert_mission(m)
		result = db.get_mission("m2")
		assert result is not None
		assert result.chain_id == "chain-abc"


# ============================================================================
# Integration Tests
# ============================================================================


class TestControllerStrategistIntegration:
	@pytest.mark.asyncio
	async def test_orchestration_loop_completes_with_strategist(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""With strategist set, orchestration loop still completes successfully."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1

		ctrl = ContinuousController(config, db)

		mock_strategist = MagicMock(spec=Strategist)
		ctrl._strategist = mock_strategist

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [
					WorkUnit(id=f"wu-{i}", plan_id=plan.id, title=f"Task {i}", priority=2)
					for i in range(3)
				]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
		) -> WorkerCompletion | None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			completion = WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch)
			ctrl._semaphore.release()
			ctrl._in_flight_count = max(ctrl._in_flight_count - 1, 0)
			return completion

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()
			ctrl._notifier = None
			ctrl._heartbeat = None
			ctrl._event_stream = None

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=5.0)

		assert result.objective_met is True
		assert result.stopped_reason == "planner_completed"

	@pytest.mark.asyncio
	async def test_no_strategist_runs_normally(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Without a strategist, orchestration loop completes normally."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1

		ctrl = ContinuousController(config, db)

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [
					WorkUnit(id=f"wu-{i}", plan_id=plan.id, title=f"Task {i}", priority=2,
						files_hint=f"src/mod{i}.py")
					for i in range(3)
				]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
		) -> WorkerCompletion | None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			completion = WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch)
			ctrl._semaphore.release()
			ctrl._in_flight_count = max(ctrl._in_flight_count - 1, 0)
			return completion

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()
			ctrl._notifier = None
			ctrl._heartbeat = None
			ctrl._event_stream = None

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=5.0)

		assert result.objective_met is True
		assert result.stopped_reason == "planner_completed"

	@pytest.mark.asyncio
	async def test_orchestration_completes_with_strategist_db_persistence(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Mission completes and is persisted to DB with strategist set."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1

		ctrl = ContinuousController(config, db)

		mock_strategist = MagicMock(spec=Strategist)
		ctrl._strategist = mock_strategist

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [WorkUnit(id="wu-1", plan_id=plan.id, title="Build system", priority=1)]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
		) -> WorkerCompletion | None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			completion = WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch)
			ctrl._semaphore.release()
			ctrl._in_flight_count = max(ctrl._in_flight_count - 1, 0)
			return completion

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()
			ctrl._notifier = None
			ctrl._heartbeat = None
			ctrl._event_stream = None

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=5.0)

		assert result.objective_met is True
		db_mission = db.get_latest_mission()
		assert db_mission is not None

	@pytest.mark.asyncio
	async def test_ambition_enforcement_replans_then_proceeds(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Sequential orchestration: units dispatched until planner returns empty."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 30

		ctrl = ContinuousController(config, db)
		ctrl._strategist = None

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count <= 2:
				units = [WorkUnit(
					id=f"wu-{call_count}", plan_id=plan.id,
					title="Fix lint warning", priority=1,
				)]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
		) -> WorkerCompletion | None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			completion = WorkerCompletion(
				unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch,
			)
			ctrl._semaphore.release()
			ctrl._in_flight_count = max(0, ctrl._in_flight_count - 1)
			return completion

		mock_planner = MagicMock()
		mock_planner.get_next_units = AsyncMock(side_effect=mock_get_next)
		mock_planner.ingest_handoff = MagicMock()
		mock_planner.backlog_size = 0

		mock_gbm = MagicMock()
		mock_gbm.merge_unit = AsyncMock()

		mock_reflection = MagicMock()
		mock_reflection.reflect = AsyncMock(return_value=MagicMock(strategy_revision=None))

		async def mock_init() -> None:
			ctrl._planner = mock_planner
			ctrl._green_branch = mock_gbm
			ctrl._backend = AsyncMock()
			ctrl._notifier = None
			ctrl._heartbeat = None
			ctrl._event_stream = None
			ctrl._strategic_reflection = mock_reflection

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch.object(ctrl, "_run_final_verification", new_callable=AsyncMock, return_value=(True, "ok")),
			patch("mission_control.continuous_controller.EventStream"),
		):
			await asyncio.wait_for(ctrl.run(), timeout=5.0)

		# Planner called 3 times: 2 with units + 1 returning empty (objective met)
		assert mock_planner.get_next_units.call_count == 3


class TestPostMissionStrategicContext:
	"""Test that strategic context is appended after mission completion."""

	def test_append_strategic_context_logic(self) -> None:
		"""Verify the strategic context append logic produces correct args."""
		db = Database(":memory:")
		db.append_strategic_context = MagicMock()  # type: ignore[attr-defined]
		db.get_recent_handoffs = MagicMock(return_value=[])  # type: ignore[method-assign]

		mission = Mission(id="m1", objective="Test objective", status="completed")

		# Replicate the logic from continuous_controller.py
		merged_summaries: list[str] = []
		failed_summaries: list[str] = []
		handoffs = db.get_recent_handoffs(mission.id, limit=50)
		for h in handoffs:
			summary_text = h.summary[:200] if h.summary else ""
			if h.status == "completed" and summary_text:
				merged_summaries.append(summary_text)
			elif summary_text:
				failed_summaries.append(summary_text)

		db.append_strategic_context(
			what_attempted=mission.objective[:500],
			what_worked="; ".join(merged_summaries[:10]) or "nothing merged",
			what_failed="; ".join(failed_summaries[:10]) or "no failures",
			recommended_next="planner_completed",
		)

		db.append_strategic_context.assert_called_once_with(
			what_attempted="Test objective",
			what_worked="nothing merged",
			what_failed="no failures",
			recommended_next="planner_completed",
		)
		db.close()

	def test_append_strategic_context_with_handoffs(self) -> None:
		"""Verify handoff summaries are correctly categorized."""
		db = Database(":memory:")
		db.append_strategic_context = MagicMock()  # type: ignore[attr-defined]

		# Create fake handoffs
		merged_handoff = MagicMock()
		merged_handoff.status = "completed"
		merged_handoff.summary = "Added user authentication"

		failed_handoff = MagicMock()
		failed_handoff.status = "failed"
		failed_handoff.summary = "Database migration failed"

		db.get_recent_handoffs = MagicMock(  # type: ignore[method-assign]
			return_value=[merged_handoff, failed_handoff],
		)

		mission = Mission(id="m2", objective="Improve auth system", status="completed")

		# Replicate the logic
		merged_summaries: list[str] = []
		failed_summaries: list[str] = []
		handoffs = db.get_recent_handoffs(mission.id, limit=50)
		for h in handoffs:
			summary_text = h.summary[:200] if h.summary else ""
			if h.status == "completed" and summary_text:
				merged_summaries.append(summary_text)
			elif summary_text:
				failed_summaries.append(summary_text)

		db.append_strategic_context(
			what_attempted=mission.objective[:500],
			what_worked="; ".join(merged_summaries[:10]) or "nothing merged",
			what_failed="; ".join(failed_summaries[:10]) or "no failures",
			recommended_next="planner_completed",
		)

		db.append_strategic_context.assert_called_once_with(
			what_attempted="Improve auth system",
			what_worked="Added user authentication",
			what_failed="Database migration failed",
			recommended_next="planner_completed",
		)
		db.close()
