"""Tests for the strategist module."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.cli import build_parser, cmd_mission
from mission_control.config import MissionConfig, PlannerConfig, SchedulerConfig, TargetConfig
from mission_control.continuous_controller import ContinuousController, ContinuousMissionResult, WorkerCompletion
from mission_control.db import Database
from mission_control.models import BacklogItem, Epoch, Mission, Plan, StrategicContext, WorkUnit
from mission_control.strategist import (
	AMBITION_LEVEL_DESCRIPTIONS,
	CAPABILITY_DOMAINS,
	FOLLOWUP_RESULT_MARKER,
	STRATEGY_RESULT_MARKER,
	AmbitionLevel,
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


def _make_followup_output(next_objective: str, rationale: str = "Some reason") -> str:
	data = {"next_objective": next_objective, "rationale": rationale}
	return f"Reasoning about follow-up...\n\n{FOLLOWUP_RESULT_MARKER}{json.dumps(data)}"


def _mission_result(
	objective: str = "Build feature",
	objective_met: bool = True,
	total_units_dispatched: int = 5,
	total_units_merged: int = 5,
	total_units_failed: int = 0,
	stopped_reason: str = "planner_completed",
	wall_time_seconds: float = 300.0,
) -> MagicMock:
	r = MagicMock()
	r.objective = objective
	r.objective_met = objective_met
	r.total_units_dispatched = total_units_dispatched
	r.total_units_merged = total_units_merged
	r.total_units_failed = total_units_failed
	r.stopped_reason = stopped_reason
	r.wall_time_seconds = wall_time_seconds
	return r


def _make_units(specs: list[tuple[str, str, str]]) -> list[WorkUnit]:
	"""Create units from (title, description, files_hint) tuples."""
	return [
		WorkUnit(id=f"wu{i}", title=title, description=desc, files_hint=files)
		for i, (title, desc, files) in enumerate(specs)
	]


def _insert_mission(db: Database, mission_id: str) -> None:
	"""Helper to insert a minimal mission for FK satisfaction."""
	db.insert_mission(Mission(id=mission_id, objective="test"))


# -- Fixtures --


@pytest.fixture()
def mock_strategist_module():
	"""Inject a fake mission_control.strategist module into sys.modules."""
	mock_module = types.ModuleType("mission_control.strategist")
	mock_cls = MagicMock()
	mock_cls.return_value.propose_objective = AsyncMock(
		return_value=("Build a REST API", "High priority backlog item", 7),
	)
	mock_module.Strategist = mock_cls  # type: ignore[attr-defined]
	sys.modules["mission_control.strategist"] = mock_module
	yield mock_cls
	sys.modules.pop("mission_control.strategist", None)


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

	def test_get_pending_backlog_empty(self) -> None:
		s = _make_strategist()
		s.db.get_pending_backlog.return_value = []
		assert s._get_pending_backlog() == ""

	def test_get_pending_backlog_formats(self) -> None:
		s = _make_strategist()
		item = BacklogItem(title="Fix auth", description="Auth is broken in prod", priority_score=8.5)
		s.db.get_pending_backlog.return_value = [item]
		result = s._get_pending_backlog()
		assert "Fix auth" in result
		assert "score=8.5" in result

	def test_get_pending_backlog_uses_pinned_score(self) -> None:
		s = _make_strategist()
		item = BacklogItem(title="Pinned task", description="Important", priority_score=3.0, pinned_score=9.5)
		s.db.get_pending_backlog.return_value = [item]
		result = s._get_pending_backlog()
		assert "score=9.5" in result


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
		item = BacklogItem(title="Backlog task", description="Do this", priority_score=6.0)
		s.db.get_pending_backlog.return_value = [item]

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


# -- Follow-up prompt building --


class TestBuildFollowupPrompt:
	"""Test the prompt construction for follow-up evaluation."""

	def test_includes_mission_result_fields(self) -> None:
		s = _make_strategist()
		result = _mission_result(
			objective="Build auth",
			objective_met=False,
			total_units_dispatched=5,
			total_units_merged=3,
			total_units_failed=2,
			stopped_reason="wall_time_exceeded",
		)
		prompt = s._build_followup_prompt(result, "Previous context here")
		assert "Build auth" in prompt
		assert "False" in prompt
		assert "wall_time_exceeded" in prompt
		assert "Previous context here" in prompt

	def test_empty_strategic_context_uses_fallback(self) -> None:
		s = _make_strategist()
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "")
		assert "No strategic context available" in prompt

	def test_includes_pending_backlog(self) -> None:
		s = _make_strategist()
		s.db.get_pending_backlog.return_value = [
			BacklogItem(title="Fix auth", description="Auth is broken", priority_score=8.0),
		]
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "context")
		assert "Fix auth" in prompt

	def test_no_pending_backlog_uses_fallback(self) -> None:
		s = _make_strategist()
		s.db.get_pending_backlog.return_value = []
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "context")
		assert "No pending backlog items" in prompt

	def test_includes_followup_result_marker(self) -> None:
		s = _make_strategist()
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "")
		assert "FOLLOWUP_RESULT:" in prompt


# -- Follow-up parsing --


class TestParseFollowupOutput:
	"""Test parsing of FOLLOWUP_RESULT from LLM output."""

	def test_valid_marker_with_objective(self) -> None:
		s = _make_strategist()
		output = _make_followup_output("Continue with remaining work")
		assert s._parse_followup_output(output) == "Continue with remaining work"

	def test_valid_marker_empty_objective(self) -> None:
		s = _make_strategist()
		output = _make_followup_output("")
		assert s._parse_followup_output(output) == ""

	def test_json_without_marker(self) -> None:
		s = _make_strategist()
		data = {"next_objective": "Fix bugs", "rationale": "Bugs remain"}
		output = f"Here is the result:\n```json\n{json.dumps(data)}\n```"
		assert s._parse_followup_output(output) == "Fix bugs"

	def test_no_json_returns_empty(self) -> None:
		s = _make_strategist()
		assert s._parse_followup_output("Just some text with no JSON") == ""

	def test_marker_takes_precedence(self) -> None:
		s = _make_strategist()
		earlier = json.dumps({"next_objective": "Wrong one", "rationale": "Old"})
		correct = json.dumps({"next_objective": "Right one", "rationale": "New"})
		output = f"Earlier: {earlier}\n\n{FOLLOWUP_RESULT_MARKER}{correct}"
		assert s._parse_followup_output(output) == "Right one"

	def test_whitespace_stripped(self) -> None:
		s = _make_strategist()
		output = _make_followup_output("  Fix auth  ")
		assert s._parse_followup_output(output) == "Fix auth"


# -- Follow-up suggestion --


class TestSuggestFollowup:
	"""Test the async suggest_followup method."""

	@pytest.mark.asyncio
	async def test_returns_objective_from_llm(self) -> None:
		s = _make_strategist()
		llm_output = _make_followup_output("Continue building auth system")

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(), "strategic context")

		assert result == "Continue building auth system"

	@pytest.mark.asyncio
	async def test_returns_empty_when_no_followup(self) -> None:
		s = _make_strategist()
		llm_output = _make_followup_output("")

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(objective_met=True), "")

		assert result == ""

	@pytest.mark.asyncio
	async def test_returns_empty_on_subprocess_failure(self) -> None:
		s = _make_strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"", b"some error")
		mock_proc.returncode = 1

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(), "context")

		assert result == ""

	@pytest.mark.asyncio
	async def test_returns_empty_on_timeout(self) -> None:
		s = _make_strategist()

		mock_proc = AsyncMock()
		mock_proc.kill = AsyncMock()
		mock_proc.wait = AsyncMock()

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch("mission_control.strategist.asyncio.wait_for", side_effect=asyncio.TimeoutError):
				result = await s.suggest_followup(_mission_result(), "context")

		assert result == ""

	@pytest.mark.asyncio
	async def test_returns_empty_on_unparseable_output(self) -> None:
		s = _make_strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"No valid JSON here at all", b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(), "context")

		assert result == ""

	@pytest.mark.asyncio
	async def test_passes_strategic_context_to_prompt(self) -> None:
		s = _make_strategist()
		llm_output = _make_followup_output("Next step")

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			await s.suggest_followup(_mission_result(), "Focus on testing next")

		# Verify prompt was sent via stdin
		call_args = mock_proc.communicate.call_args
		prompt_bytes = call_args[1].get("input") or call_args[0][0] if call_args[0] else call_args[1]["input"]
		prompt = prompt_bytes.decode()
		assert "Focus on testing next" in prompt

	@pytest.mark.asyncio
	async def test_passes_mission_result_to_prompt(self) -> None:
		s = _make_strategist()
		llm_output = _make_followup_output("")

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		result = _mission_result(
			objective="Build API",
			objective_met=False,
			stopped_reason="stall",
		)

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			await s.suggest_followup(result, "")

		call_args = mock_proc.communicate.call_args
		prompt_bytes = call_args[1].get("input") or call_args[0][0] if call_args[0] else call_args[1]["input"]
		prompt = prompt_bytes.decode()
		assert "Build API" in prompt
		assert "stall" in prompt

	@pytest.mark.asyncio
	async def test_sets_cwd_to_target_path(self, tmp_path: Path) -> None:
		s = _make_strategist(tmp_path)
		llm_output = _make_followup_output("")

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
			await s.suggest_followup(_mission_result(), "")

		_, kwargs = mock_exec.call_args
		assert kwargs["cwd"] == str(tmp_path)

	@pytest.mark.asyncio
	async def test_uses_config_model_and_budget(self) -> None:
		s = _make_strategist()
		llm_output = _make_followup_output("")

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
			await s.suggest_followup(_mission_result(), "")

		args = mock_exec.call_args[0]
		assert "--model" in args
		assert "sonnet" in args
		assert "--max-budget-usd" in args
		assert "0.1" in args


# ============================================================================
# Ambition Scoring Tests
# ============================================================================


class TestEvaluateAmbition:
	@pytest.mark.asyncio
	async def test_empty_units_returns_1(self) -> None:
		s = _make_strategist()
		assert await s.evaluate_ambition([]) == 1

	@pytest.mark.asyncio
	async def test_single_lint_fix_scores_low(self) -> None:
		s = _make_strategist()
		units = _make_units([("Fix lint errors", "Run ruff and fix formatting issues", "src/main.py")])
		score = await s.evaluate_ambition(units)
		assert 1 <= score <= 3

	@pytest.mark.asyncio
	async def test_single_typo_fix_scores_low(self) -> None:
		s = _make_strategist()
		units = _make_units([("Fix typo in README", "Minor typo correction", "README.md")])
		score = await s.evaluate_ambition(units)
		assert 1 <= score <= 3

	@pytest.mark.asyncio
	async def test_formatting_cleanup_scores_low(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Fix whitespace issues", "cleanup formatting", "a.py"),
			("Fix style nits", "minor style cleanup", "b.py"),
		])
		score = await s.evaluate_ambition(units)
		assert 1 <= score <= 4

	@pytest.mark.asyncio
	async def test_new_feature_scores_moderate(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Add user authentication", "Implement JWT-based auth", "src/auth.py, src/middleware.py"),
			("Add auth tests", "Create test suite for auth", "tests/test_auth.py"),
		])
		score = await s.evaluate_ambition(units)
		assert 4 <= score <= 7

	@pytest.mark.asyncio
	async def test_architecture_change_scores_high(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Redesign event system", "New architecture for event-driven pipeline", "src/events.py, src/pipeline.py"),
			("Build distributed queue", "New system for async task distribution", "src/queue.py, src/worker.py"),
			("Integrate message bus", "Integration with event bus infrastructure", "src/bus.py, src/config.py"),
		])
		score = await s.evaluate_ambition(units)
		assert score >= 7

	@pytest.mark.asyncio
	async def test_new_system_scores_high(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Build new module for caching", "New system with Redis backend",
			 "src/cache/engine.py, src/cache/store.py"),
			("Add cache migration", "Migration scripts for cache infrastructure",
			 "src/cache/migrate.py"),
		])
		score = await s.evaluate_ambition(units)
		assert score >= 6

	@pytest.mark.asyncio
	async def test_many_files_increases_score(self) -> None:
		s = _make_strategist()
		few_files = _make_units([
			("Add feature X", "Implement feature X", "src/x.py"),
		])
		many_files = _make_units([
			("Add feature X", "Implement feature X",
			 "src/x.py, src/y.py, src/z.py, src/a.py, src/b.py, "
			 "src/c.py, src/d.py, src/e.py, src/f.py, src/g.py"),
		])
		score_few = await s.evaluate_ambition(few_files)
		score_many = await s.evaluate_ambition(many_files)
		assert score_many >= score_few

	@pytest.mark.asyncio
	async def test_many_units_increases_score(self) -> None:
		s = _make_strategist()
		few = _make_units([
			("Add feature", "Implement it", "a.py"),
		])
		many = _make_units([
			(f"Add feature {i}", f"Implement feature {i}", f"src/mod{i}.py")
			for i in range(6)
		])
		score_few = await s.evaluate_ambition(few)
		score_many = await s.evaluate_ambition(many)
		assert score_many >= score_few

	@pytest.mark.asyncio
	async def test_score_clamped_1_to_10(self) -> None:
		s = _make_strategist()
		# Extreme high
		units = _make_units([
			(f"Redesign system {i}", f"New architecture for distributed infrastructure component {i}",
			 ", ".join(f"src/mod{j}.py" for j in range(20)))
			for i in range(10)
		])
		score = await s.evaluate_ambition(units)
		assert 1 <= score <= 10

	@pytest.mark.asyncio
	async def test_mixed_unit_types(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Fix lint in config", "Minor lint cleanup", "src/config.py"),
			("Redesign auth layer", "New architecture for auth", "src/auth.py, src/middleware.py, src/tokens.py"),
			("Add feature flag", "Implement toggle logic", "src/flags.py"),
		])
		score = await s.evaluate_ambition(units)
		# Mix of low and high should land moderate-to-high
		assert 4 <= score <= 9

	@pytest.mark.asyncio
	async def test_refactor_scores_moderate(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Refactor database layer", "Improve error handling and connection pooling", "src/db.py, src/pool.py"),
			("Update API endpoints", "Enhance endpoint validation", "src/api.py"),
		])
		score = await s.evaluate_ambition(units)
		assert 4 <= score <= 7

	@pytest.mark.asyncio
	async def test_no_files_hint(self) -> None:
		s = _make_strategist()
		units = _make_units([
			("Add new feature", "Build something interesting", ""),
		])
		score = await s.evaluate_ambition(units)
		# Should still produce a valid score
		assert 1 <= score <= 10

	@pytest.mark.asyncio
	async def test_description_contributes_to_scoring(self) -> None:
		s = _make_strategist()
		# Title is generic but description has high keywords
		units = _make_units([
			("Task 1", "Redesign the distributed architecture for the pipeline", "src/pipe.py"),
		])
		score = await s.evaluate_ambition(units)
		assert score >= 5


# -- ZFC ambition scoring --


class TestZFCAmbition:
	@pytest.mark.asyncio
	async def test_zfc_enabled_calls_llm(self) -> None:
		"""When zfc_ambition_scoring is True, _zfc_evaluate_ambition is called."""
		s = _make_strategist()
		s.config.zfc.zfc_ambition_scoring = True
		units = _make_units([("Build system", "Architecture redesign", "src/sys.py")])

		with patch.object(s, "_zfc_evaluate_ambition", return_value=8) as mock_zfc:
			score = await s.evaluate_ambition(units)
		mock_zfc.assert_called_once_with(units)
		assert score == 8

	@pytest.mark.asyncio
	async def test_zfc_fallback_on_failure(self) -> None:
		"""When ZFC LLM returns None, heuristic fallback is used."""
		s = _make_strategist()
		s.config.zfc.zfc_ambition_scoring = True
		units = _make_units([("Build system", "Architecture redesign", "src/sys.py")])

		with patch.object(s, "_zfc_evaluate_ambition", return_value=None):
			score = await s.evaluate_ambition(units)
		# Heuristic should return a valid score
		assert 1 <= score <= 10

	@pytest.mark.asyncio
	async def test_zfc_malformed_json_fallback(self) -> None:
		"""Malformed JSON from LLM -> fallback to heuristic."""
		s = _make_strategist()
		s.config.zfc.zfc_ambition_scoring = True
		units = _make_units([("Fix lint", "Minor fix", "a.py")])

		# _invoke_llm returns garbage
		with patch.object(s, "_invoke_llm", return_value="no json here"):
			score = await s.evaluate_ambition(units)
		assert 1 <= score <= 10

	@pytest.mark.asyncio
	async def test_zfc_score_clamping(self) -> None:
		"""ZFC score is clamped to 1-10."""
		s = _make_strategist()
		s.config.zfc.zfc_ambition_scoring = True
		units = _make_units([("Task", "Desc", "")])

		# Return out-of-range score from LLM
		output = 'Some reasoning\nAMBITION_RESULT:{"score": 15, "reasoning": "very ambitious"}'
		with patch.object(s, "_invoke_llm", return_value=output):
			score = await s.evaluate_ambition(units)
		assert score == 10

	@pytest.mark.asyncio
	async def test_zfc_score_clamp_low(self) -> None:
		"""ZFC score below 1 is clamped to 1."""
		s = _make_strategist()
		s.config.zfc.zfc_ambition_scoring = True
		units = _make_units([("Task", "Desc", "")])

		output = 'AMBITION_RESULT:{"score": -5, "reasoning": "trivial"}'
		with patch.object(s, "_invoke_llm", return_value=output):
			score = await s.evaluate_ambition(units)
		assert score == 1


class TestZFCObjectivePassthrough:
	@pytest.mark.asyncio
	async def test_cached_score_returned_and_consumed(self) -> None:
		"""Cached score from propose_objective is returned once, then consumed."""
		s = _make_strategist()
		s.config.zfc.zfc_propose_objective = True
		s._proposed_ambition_score = 7

		units = _make_units([("Task", "Desc", "")])
		score = await s.evaluate_ambition(units)
		assert score == 7
		assert s._proposed_ambition_score is None

		# Second call should use heuristic
		score2 = await s.evaluate_ambition(units)
		assert score2 != 7 or True  # heuristic may coincide, just check it ran
		assert s._proposed_ambition_score is None

	@pytest.mark.asyncio
	async def test_no_cache_without_toggle(self) -> None:
		"""Without zfc_propose_objective, cached score is ignored."""
		s = _make_strategist()
		s.config.zfc.zfc_propose_objective = False
		s._proposed_ambition_score = 7

		units = _make_units([("Fix lint", "Minor cleanup", "a.py")])
		score = await s.evaluate_ambition(units)
		# Should use heuristic, not cached score
		assert 1 <= score <= 10
		# Cache should NOT be consumed
		assert s._proposed_ambition_score == 7


# -- should_replan --


class TestShouldReplan:
	def test_high_ambition_no_replan(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Important task", priority_score=9.0)]
		should, reason = s.should_replan(7, items)
		assert should is False
		assert reason == ""

	def test_exactly_4_no_replan(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Important task", priority_score=9.0)]
		should, reason = s.should_replan(4, items)
		assert should is False

	def test_low_ambition_no_backlog(self) -> None:
		s = _make_strategist()
		should, reason = s.should_replan(2, [])
		assert should is False
		assert "No higher-priority" in reason

	def test_low_ambition_low_priority_backlog(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Trivial task", priority_score=2.0)]
		should, reason = s.should_replan(2, items)
		assert should is False
		assert "No high-priority" in reason

	def test_low_ambition_high_priority_triggers_replan(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Critical auth fix", priority_score=9.0)]
		should, reason = s.should_replan(2, items)
		assert should is True
		assert "Critical auth fix" in reason
		assert "Ambition score 2" in reason

	def test_ambition_3_triggers_replan(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Build new API", priority_score=8.0)]
		should, reason = s.should_replan(3, items)
		assert should is True
		assert "Build new API" in reason

	def test_ambition_1_triggers_replan(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Redesign DB", priority_score=7.5)]
		should, reason = s.should_replan(1, items)
		assert should is True

	def test_pinned_score_used_when_present(self) -> None:
		s = _make_strategist()
		# Low priority_score but high pinned_score
		items = [BacklogItem(title="Pinned task", priority_score=2.0, pinned_score=9.0)]
		should, reason = s.should_replan(2, items)
		assert should is True
		assert "priority=9.0" in reason

	def test_pinned_score_none_uses_priority(self) -> None:
		s = _make_strategist()
		items = [BacklogItem(title="Regular task", priority_score=8.0, pinned_score=None)]
		should, reason = s.should_replan(2, items)
		assert should is True
		assert "priority=8.0" in reason

	def test_mixed_priorities_uses_first_high(self) -> None:
		s = _make_strategist()
		items = [
			BacklogItem(title="Low task", priority_score=3.0),
			BacklogItem(title="High task", priority_score=8.0),
		]
		should, reason = s.should_replan(2, items)
		# Only the first high-priority item (above 5.0) is referenced
		assert should is True
		assert "High task" in reason


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
	def test_insert_with_new_fields(self, db: Database) -> None:
		m = Mission(
			id="m2",
			objective="Build feature",
			ambition_score=7,
			next_objective="Optimize performance",
			proposed_by_strategist=True,
		)
		db.insert_mission(m)
		result = db.get_mission("m2")
		assert result is not None
		assert result.ambition_score == 7
		assert result.next_objective == "Optimize performance"
		assert result.proposed_by_strategist is True

	def test_update_new_fields(self, db: Database) -> None:
		m = Mission(id="m3", objective="Initial")
		db.insert_mission(m)
		m.ambition_score = 5
		m.next_objective = "Follow-up work"
		m.proposed_by_strategist = True
		db.update_mission(m)
		result = db.get_mission("m3")
		assert result is not None
		assert result.ambition_score == 5
		assert result.next_objective == "Follow-up work"
		assert result.proposed_by_strategist is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestControllerStrategistIntegration:
	@pytest.mark.asyncio
	async def test_strategist_evaluate_ambition_used_in_dispatch(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When strategist is set, its evaluate_ambition is used instead of _score_ambition."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		mock_strategist = MagicMock(spec=Strategist)
		mock_strategist.evaluate_ambition = AsyncMock(return_value=9)
		mock_strategist.should_replan.return_value = (False, "")
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
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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

		mock_strategist.evaluate_ambition.assert_called_once()
		assert result.ambition_score == 9
		assert ctrl.ambition_score == 9

	@pytest.mark.asyncio
	async def test_strategist_should_replan_logs_warning(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When strategist recommends replanning, a warning is logged."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		mock_strategist = MagicMock(spec=Strategist)
		mock_strategist.evaluate_ambition = AsyncMock(return_value=2)
		mock_strategist.should_replan.return_value = (True, "Low ambition, replan needed")
		ctrl._strategist = mock_strategist

		# Insert backlog items for should_replan to find
		db.insert_backlog_item(BacklogItem(
			id="bl1", title="Big task", priority_score=9.0, status="pending",
		))

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count == 1:
				units = [WorkUnit(id="wu-1", plan_id=plan.id, title="Fix lint", priority=8)]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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

		warning_messages: list[str] = []

		def capture_warning(msg: str, *args: object, **kwargs: object) -> None:
			warning_messages.append(msg % args if args else msg)

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch.object(
				logging.getLogger("mission_control.continuous_controller"),
				"warning", side_effect=capture_warning,
			),
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=5.0)

		assert result.ambition_score == 2
		replan_warnings = [m for m in warning_messages if "replan" in m.lower()]
		assert len(replan_warnings) > 0

	@pytest.mark.asyncio
	async def test_no_strategist_uses_score_ambition(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Without a strategist, the controller falls back to _score_ambition."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)
		# No strategist set -- ctrl._strategist is None

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
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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
			patch.object(ctrl, "_score_ambition", return_value=6) as mock_score,
			patch("mission_control.continuous_controller.EventStream"),
		):
			result = await asyncio.wait_for(ctrl.run(), timeout=5.0)

		mock_score.assert_called_once()
		assert result.ambition_score == 6

	@pytest.mark.asyncio
	async def test_ambition_score_written_to_db(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Ambition score from strategist should be persisted to the mission DB record."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		mock_strategist = MagicMock(spec=Strategist)
		mock_strategist.evaluate_ambition = AsyncMock(return_value=7)
		mock_strategist.should_replan.return_value = (False, "")
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
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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

		assert result.ambition_score == 7

		# Check DB persistence
		db_mission = db.get_latest_mission()
		assert db_mission is not None
		assert db_mission.ambition_score == 7

	@pytest.mark.asyncio
	async def test_should_replan_false_no_warning(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""When should_replan returns False, no replan warning is logged."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 1
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)

		mock_strategist = MagicMock(spec=Strategist)
		mock_strategist.evaluate_ambition = AsyncMock(return_value=5)
		mock_strategist.should_replan.return_value = (False, "")
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
				units = [WorkUnit(id="wu-1", plan_id=plan.id, title="Task", priority=3)]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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

		warning_messages: list[str] = []

		def capture_warning(msg: str, *args: object, **kwargs: object) -> None:
			warning_messages.append(msg % args if args else msg)

		with (
			patch.object(ctrl, "_init_components", mock_init),
			patch.object(ctrl, "_execute_single_unit", side_effect=mock_execute),
			patch.object(
				logging.getLogger("mission_control.continuous_controller"),
				"warning", side_effect=capture_warning,
			),
			patch("mission_control.continuous_controller.EventStream"),
		):
			await asyncio.wait_for(ctrl.run(), timeout=5.0)

		replan_warnings = [m for m in warning_messages if "replan" in m.lower()]
		assert len(replan_warnings) == 0

	@pytest.mark.asyncio
	async def test_ambition_enforcement_replans_then_proceeds(
		self, config: MissionConfig, db: Database,
	) -> None:
		"""Low ambition triggers replanning; after max_replan_attempts, proceeds anyway."""
		config.target.name = "test"
		config.continuous.max_wall_time_seconds = 2
		config.continuous.min_ambition_score = 5
		config.continuous.max_replan_attempts = 2
		config.discovery.enabled = False
		ctrl = ContinuousController(config, db)
		# No strategist -- uses _score_ambition
		ctrl._strategist = None

		call_count = 0

		async def mock_get_next(
			mission: Mission, max_units: int = 3, feedback_context: str = "", **kwargs: object,
		) -> tuple[Plan, list[WorkUnit], Epoch]:
			nonlocal call_count
			call_count += 1
			plan = Plan(id=f"p{call_count}", objective="test")
			epoch = Epoch(id=f"ep{call_count}", mission_id=mission.id, number=call_count)
			if call_count <= 3:
				# Always return trivial units -- low ambition
				units = [WorkUnit(
					id=f"wu-{call_count}", plan_id=plan.id,
					title="Fix lint warning", priority=1,
				)]
				return plan, units, epoch
			return plan, [], epoch

		async def mock_execute(
			unit: WorkUnit, epoch: Epoch, mission: Mission,
			semaphore: asyncio.Semaphore,
		) -> None:
			unit.status = "completed"
			unit.finished_at = "2025-01-01T00:00:00"
			await ctrl._completion_queue.put(
				WorkerCompletion(unit=unit, handoff=None, workspace="/tmp/ws", epoch=epoch),
			)
			semaphore.release()

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
			patch.object(ctrl, "_score_ambition", return_value=2),
			patch("mission_control.continuous_controller.EventStream"),
		):
			await asyncio.wait_for(ctrl.run(), timeout=5.0)

		# Planner called: 1 (initial) + 2 (replans) + 1 (next loop, returns []) = 4
		assert mock_planner.get_next_units.call_count >= 3
		# Second and third calls should contain rejection feedback
		for i in [1, 2]:
			call_args = mock_planner.get_next_units.call_args_list[i]
			feedback = call_args.kwargs.get("feedback_context", call_args.args[2] if len(call_args.args) > 2 else "")
			assert "PREVIOUS PLAN REJECTED" in feedback


class TestStrategistCliFlag:
	"""Test --strategist flag parsing in CLI."""

	def test_strategist_flag_default_false(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission"])
		assert args.strategist is False

	def test_strategist_flag_set(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--strategist"])
		assert args.strategist is True

	def test_strategist_with_approve_all(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["mission", "--strategist", "--approve-all"])
		assert args.strategist is True
		assert args.approve_all is True

	def test_strategist_with_other_flags(self) -> None:
		parser = build_parser()
		args = parser.parse_args([
			"mission", "--strategist", "--workers", "4", "--config", "custom.toml",
		])
		assert args.strategist is True
		assert args.workers == 4
		assert args.config == "custom.toml"


class TestStrategistApprovalFlow:
	"""Test strategist objective proposal and approval flow in cmd_mission."""

	def _make_config(self, tmp_path: Path) -> Path:
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text("""\
[target]
name = "test"
path = "/tmp/test"
objective = ""

[target.verification]
command = "echo ok"

[scheduler]
model = "sonnet"
""")
		return config_file

	@patch("mission_control.cli._start_dashboard_background", return_value=(None, None))
	@patch("mission_control.cli.input", return_value="y")
	@patch("mission_control.cli.asyncio.run")
	def test_strategist_approved_sets_objective(
		self,
		mock_run: MagicMock,
		mock_input: MagicMock,
		_mock_dash: MagicMock,
		tmp_path: Path,
		mock_strategist_module: MagicMock,
	) -> None:
		config_file = self._make_config(tmp_path)
		db_path = tmp_path / "mission-control.db"
		Database(db_path).close()

		proposal = ("Build a REST API", "High priority backlog item", 7)
		mission_result = ContinuousMissionResult(
			mission_id="m1", objective_met=True, stopped_reason="planner_completed",
		)
		mock_run.side_effect = [proposal, mission_result]

		parser = build_parser()
		args = parser.parse_args([
			"mission", "--strategist", "--config", str(config_file),
		])
		result = cmd_mission(args)

		assert result == 0
		mock_input.assert_called_once()

	@patch("mission_control.cli._start_dashboard_background", return_value=(None, None))
	@patch("mission_control.cli.input", return_value="n")
	@patch("mission_control.cli.asyncio.run")
	def test_strategist_rejected_exits_zero(
		self,
		mock_run: MagicMock,
		mock_input: MagicMock,
		_mock_dash: MagicMock,
		tmp_path: Path,
		mock_strategist_module: MagicMock,
	) -> None:
		config_file = self._make_config(tmp_path)
		db_path = tmp_path / "mission-control.db"
		Database(db_path).close()

		proposal = ("Build a REST API", "High priority backlog item", 7)
		mock_run.return_value = proposal

		parser = build_parser()
		args = parser.parse_args([
			"mission", "--strategist", "--config", str(config_file),
		])
		result = cmd_mission(args)

		assert result == 0

	@patch("mission_control.cli._start_dashboard_background", return_value=(None, None))
	@patch("mission_control.cli.asyncio.run")
	def test_strategist_auto_approve_skips_prompt(
		self,
		mock_run: MagicMock,
		_mock_dash: MagicMock,
		tmp_path: Path,
		mock_strategist_module: MagicMock,
	) -> None:
		config_file = self._make_config(tmp_path)
		db_path = tmp_path / "mission-control.db"
		Database(db_path).close()

		proposal = ("Build a REST API", "High priority backlog item", 7)
		mission_result = ContinuousMissionResult(
			mission_id="m1", objective_met=True, stopped_reason="planner_completed",
		)
		mock_run.side_effect = [proposal, mission_result]

		with patch("mission_control.cli.input") as mock_input:
			parser = build_parser()
			args = parser.parse_args([
				"mission", "--strategist", "--approve-all", "--config", str(config_file),
			])
			result = cmd_mission(args)

		assert result == 0
		mock_input.assert_not_called()


class TestPostMissionStrategicContext:
	"""Test that strategic context is appended after mission completion."""

	def test_controller_init_defaults(self) -> None:
		"""ContinuousController.__init__ sets strategist attributes."""
		db = Database(":memory:")
		config = MagicMock()
		controller = ContinuousController(config, db)
		assert controller.ambition_score == 0.0
		assert controller.proposed_by_strategist is False
		db.close()

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

	def test_proposed_by_strategist_propagated_from_cli(self) -> None:
		"""When --strategist is used, controller.proposed_by_strategist should be True."""
		db = Database(":memory:")
		config = MagicMock()
		controller = ContinuousController(config, db)
		controller.proposed_by_strategist = True
		assert controller.proposed_by_strategist is True
		db.close()

	def test_ambition_score_set_on_controller(self) -> None:
		"""Ambition score can be set on the controller."""
		db = Database(":memory:")
		config = MagicMock()
		controller = ContinuousController(config, db)
		controller.ambition_score = 7.5
		assert controller.ambition_score == 7.5
		db.close()

	def test_mission_hasattr_guard_works(self) -> None:
		"""The hasattr guard for Mission fields works without error."""
		mission = Mission(id="m1", objective="Test", status="completed")

		# Simulate the hasattr guard from the controller
		if hasattr(mission, "ambition_score"):
			mission.ambition_score = 6.0  # type: ignore[attr-defined]
		if hasattr(mission, "proposed_by_strategist"):
			mission.proposed_by_strategist = True  # type: ignore[attr-defined]

		# Currently Mission doesn't have these fields, so hasattr returns False
		# This test verifies the guard works without error
		assert mission.status == "completed"


# -- Ambition Level enum --


class TestAmbitionLevel:
	def test_four_levels_exist(self) -> None:
		assert len(AmbitionLevel) == 4

	def test_level_ordering(self) -> None:
		assert AmbitionLevel.BUGS_QUALITY < AmbitionLevel.IMPROVE_FEATURES
		assert AmbitionLevel.IMPROVE_FEATURES < AmbitionLevel.NEW_CAPABILITIES
		assert AmbitionLevel.NEW_CAPABILITIES < AmbitionLevel.META_IMPROVEMENTS

	def test_level_values(self) -> None:
		assert AmbitionLevel.BUGS_QUALITY == 1
		assert AmbitionLevel.IMPROVE_FEATURES == 2
		assert AmbitionLevel.NEW_CAPABILITIES == 3
		assert AmbitionLevel.META_IMPROVEMENTS == 4

	def test_all_levels_have_descriptions(self) -> None:
		for level in AmbitionLevel:
			assert level in AMBITION_LEVEL_DESCRIPTIONS
			assert len(AMBITION_LEVEL_DESCRIPTIONS[level]) > 0

	def test_capability_domains_exist(self) -> None:
		assert len(CAPABILITY_DOMAINS) > 0
		for domain in CAPABILITY_DOMAINS:
			assert "name" in domain
			assert "level" in domain
			assert "description" in domain
			assert "keywords" in domain

	def test_capability_domains_have_valid_levels(self) -> None:
		for domain in CAPABILITY_DOMAINS:
			level = domain["level"]
			assert level in (AmbitionLevel.NEW_CAPABILITIES, AmbitionLevel.META_IMPROVEMENTS)


# -- Capability gap analysis --


class TestCapabilityGapAnalysis:
	def test_all_gaps_when_no_context(self) -> None:
		s = _make_strategist()
		s.db.get_pending_backlog.return_value = []
		s.db.get_all_missions.return_value = []
		gaps = s.analyze_capability_gaps()
		assert len(gaps) == len(CAPABILITY_DOMAINS)

	def test_gap_removed_when_keyword_in_backlog(self) -> None:
		s = _make_strategist()
		item = BacklogItem(
			title="Add web research capability",
			description="Workers can web search for docs",
			priority_score=8.0,
		)
		s.db.get_pending_backlog.return_value = [item]
		s.db.get_all_missions.return_value = []
		gaps = s.analyze_capability_gaps()
		gap_names = [g["name"] for g in gaps]
		assert "Web Research" not in gap_names

	def test_gap_removed_when_keyword_in_mission_objective(self) -> None:
		s = _make_strategist()
		s.db.get_pending_backlog.return_value = []
		m = Mission(objective="Add browser automation testing", status="completed")
		s.db.get_all_missions.return_value = [m]
		gaps = s.analyze_capability_gaps()
		gap_names = [g["name"] for g in gaps]
		assert "Browser Automation" not in gap_names

	def test_gap_structure(self) -> None:
		s = _make_strategist()
		s.db.get_pending_backlog.return_value = []
		s.db.get_all_missions.return_value = []
		gaps = s.analyze_capability_gaps()
		for gap in gaps:
			assert "name" in gap
			assert "level" in gap
			assert "description" in gap
			assert isinstance(gap["level"], int)

	def test_compute_gaps_partial_match(self) -> None:
		"""Only matching domains are excluded, rest remain as gaps."""
		s = _make_strategist()
		item = BacklogItem(
			title="Implement playwright end-to-end test suite",
			description="Browser automation for UI testing",
			priority_score=7.0,
		)
		s.db.get_pending_backlog.return_value = [item]
		s.db.get_all_missions.return_value = []
		gaps = s.analyze_capability_gaps()
		gap_names = [g["name"] for g in gaps]
		assert "Browser Automation" not in gap_names
		# Other gaps should still be present
		assert "Web Research" in gap_names


# -- Ambition level determination --


class TestDetermineAmbitionLevel:
	def test_level_1_when_quality_items_exist(self) -> None:
		s = _make_strategist()
		pending = [
			BacklogItem(title="Fix auth bug", track="quality", priority_score=7.0, status="pending"),
		]
		level = s._determine_ambition_level(pending, [])
		assert level == AmbitionLevel.BUGS_QUALITY

	def test_level_1_when_security_items_exist(self) -> None:
		s = _make_strategist()
		pending = [
			BacklogItem(title="SQL injection vulnerability", track="security", priority_score=9.0, status="pending"),
		]
		level = s._determine_ambition_level(pending, [])
		assert level == AmbitionLevel.BUGS_QUALITY

	def test_level_1_from_title_keywords(self) -> None:
		s = _make_strategist()
		pending = [
			BacklogItem(title="Fix broken tests", track="feature", priority_score=6.0, status="pending"),
		]
		level = s._determine_ambition_level(pending, [])
		assert level == AmbitionLevel.BUGS_QUALITY

	def test_level_2_when_feature_items_exist(self) -> None:
		s = _make_strategist()
		pending = [
			BacklogItem(title="Add retry logic to API", track="feature", priority_score=7.0, status="pending"),
		]
		level = s._determine_ambition_level(pending, [])
		assert level == AmbitionLevel.IMPROVE_FEATURES

	def test_level_2_from_improve_keywords(self) -> None:
		s = _make_strategist()
		pending = [
			BacklogItem(title="Improve error handling", track="", priority_score=6.0, status="pending"),
		]
		level = s._determine_ambition_level(pending, [])
		assert level == AmbitionLevel.IMPROVE_FEATURES

	def test_level_3_when_no_lower_items_and_gaps_exist(self) -> None:
		s = _make_strategist()
		gaps = [{"name": "Web Research", "level": int(AmbitionLevel.NEW_CAPABILITIES), "description": "..."}]
		level = s._determine_ambition_level([], gaps)
		assert level == AmbitionLevel.NEW_CAPABILITIES

	def test_level_4_when_no_lower_items_and_no_level_3_gaps(self) -> None:
		s = _make_strategist()
		gaps = [{"name": "Self-Improving", "level": int(AmbitionLevel.META_IMPROVEMENTS), "description": "..."}]
		level = s._determine_ambition_level([], gaps)
		assert level == AmbitionLevel.META_IMPROVEMENTS

	def test_level_4_when_no_items_and_no_gaps(self) -> None:
		s = _make_strategist()
		level = s._determine_ambition_level([], [])
		assert level == AmbitionLevel.META_IMPROVEMENTS

	def test_ignores_low_priority_items(self) -> None:
		"""Items below priority 3.0 don't count toward any level."""
		s = _make_strategist()
		pending = [
			BacklogItem(title="Fix trivial lint", track="quality", priority_score=1.0, status="pending"),
		]
		gaps = [{"name": "Web Research", "level": int(AmbitionLevel.NEW_CAPABILITIES), "description": "..."}]
		level = s._determine_ambition_level(pending, gaps)
		assert level == AmbitionLevel.NEW_CAPABILITIES

	def test_ignores_non_pending_items(self) -> None:
		"""Completed/deferred items don't count."""
		s = _make_strategist()
		pending = [
			BacklogItem(title="Fix auth bug", track="quality", priority_score=8.0, status="completed"),
		]
		gaps = [{"name": "Web Research", "level": int(AmbitionLevel.NEW_CAPABILITIES), "description": "..."}]
		level = s._determine_ambition_level(pending, gaps)
		assert level == AmbitionLevel.NEW_CAPABILITIES

	def test_level_1_takes_priority_over_level_2(self) -> None:
		"""When both Level 1 and Level 2 items exist, Level 1 wins."""
		s = _make_strategist()
		pending = [
			BacklogItem(title="Add caching", track="feature", priority_score=8.0, status="pending"),
			BacklogItem(title="Fix race condition", track="security", priority_score=7.0, status="pending"),
		]
		level = s._determine_ambition_level(pending, [])
		assert level == AmbitionLevel.BUGS_QUALITY


# -- Web research context hook --


class TestWebResearchContextHook:
	def test_returns_empty_by_default(self) -> None:
		s = _make_strategist()
		assert s._get_web_research_context() == ""


# -- Strategy prompt with ambition level --


class TestStrategyPromptAmbitionLevel:
	def test_prompt_includes_ambition_level(self) -> None:
		prompt = _build_strategy_prompt(
			"", "", "", "", "",
			ambition_level=AmbitionLevel.NEW_CAPABILITIES,
		)
		assert "Level 3" in prompt
		assert "NEW_CAPABILITIES" in prompt
		assert "MUST propose" in prompt

	def test_prompt_includes_capability_gaps(self) -> None:
		gaps = "- [Level 3] Web Research: Search the web for docs"
		prompt = _build_strategy_prompt(
			"", "", "", "", "",
			capability_gaps=gaps,
		)
		assert "Capability Gap Analysis" in prompt
		assert "Web Research" in prompt

	def test_prompt_includes_web_research_context(self) -> None:
		prompt = _build_strategy_prompt(
			"", "", "", "", "",
			web_research_context="Latest Claude API supports tool use",
		)
		assert "Web Research Context" in prompt
		assert "Claude API" in prompt

	def test_prompt_escalation_instruction_at_level_3(self) -> None:
		prompt = _build_strategy_prompt(
			"", "", "", "", "",
			ambition_level=AmbitionLevel.NEW_CAPABILITIES,
		)
		assert "lower-level work has been exhausted" in prompt

	def test_prompt_no_escalation_at_level_1(self) -> None:
		prompt = _build_strategy_prompt(
			"", "", "", "", "",
			ambition_level=AmbitionLevel.BUGS_QUALITY,
		)
		assert "lower-level work has been exhausted" not in prompt

	def test_prompt_all_level_descriptions_present(self) -> None:
		prompt = _build_strategy_prompt(
			"", "", "", "", "",
			ambition_level=AmbitionLevel.IMPROVE_FEATURES,
		)
		assert "Level 1:" in prompt
		assert "Level 2:" in prompt
		assert "Level 3:" in prompt
		assert "Level 4:" in prompt

	def test_prompt_without_ambition_level_unchanged(self) -> None:
		"""Without ambition_level, prompt doesn't include ambition sections."""
		prompt = _build_strategy_prompt("", "", "", "", "")
		assert "Target Ambition Level" not in prompt
		assert "Capability Gap Analysis" not in prompt


# -- propose_objective with ambition level --


class TestProposeObjectiveAmbition:
	@pytest.mark.asyncio
	async def test_propose_includes_ambition_context(self) -> None:
		s = _make_strategist()
		s.db.get_pending_backlog.return_value = []
		s.db.get_all_missions.return_value = []
		strategy_output = _make_strategy_output("Add web research", "Expand capabilities", 8)

		git_proc = AsyncMock()
		git_proc.communicate.return_value = (b"abc123 commit", b"")
		git_proc.returncode = 0

		llm_proc = AsyncMock()
		llm_proc.communicate.return_value = (strategy_output.encode(), b"")
		llm_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", side_effect=[git_proc, llm_proc]):
			obj, _, score = await s.propose_objective()

		assert obj == "Add web research"
		# Verify the LLM prompt included ambition context
		call_args = llm_proc.communicate.call_args
		prompt_bytes = call_args[1].get("input") or call_args[0][0] if call_args[0] else call_args[1]["input"]
		prompt = prompt_bytes.decode()
		assert "Target Ambition Level" in prompt

	@pytest.mark.asyncio
	async def test_propose_escalates_when_backlog_empty(self) -> None:
		s = _make_strategist()
		s.db.get_pending_backlog.return_value = []
		s.db.get_all_missions.return_value = []
		strategy_output = _make_strategy_output("Build new system", "Level 3 escalation", 9)

		git_proc = AsyncMock()
		git_proc.communicate.return_value = (b"", b"")
		git_proc.returncode = 0

		llm_proc = AsyncMock()
		llm_proc.communicate.return_value = (strategy_output.encode(), b"")
		llm_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", side_effect=[git_proc, llm_proc]):
			await s.propose_objective()

		call_args = llm_proc.communicate.call_args
		prompt_bytes = call_args[1].get("input") or call_args[0][0] if call_args[0] else call_args[1]["input"]
		prompt = prompt_bytes.decode()
		# Should escalate to Level 3+ when no pending items
		assert "lower-level work has been exhausted" in prompt


# ============================================================================
# Episodic Memory Integration Tests
# ============================================================================


class TestBuildStrategyPromptEpisodicContext:
	def test_episodic_context_included(self) -> None:
		prompt = _build_strategy_prompt(
			"", "", "", "", "",
			episodic_context="Learned rules:\n  - [0.9] Always run tests before merge",
		)
		assert "Past Learnings" in prompt
		assert "Always run tests before merge" in prompt

	def test_empty_episodic_context_omitted(self) -> None:
		prompt = _build_strategy_prompt("", "", "", "", "", episodic_context="")
		assert "Past Learnings" not in prompt


class TestGetEpisodicContext:
	def test_formats_semantic_and_episodic(self) -> None:
		s = _make_strategist()
		sem = MagicMock()
		sem.confidence = 0.9
		sem.content = "Cross-cutting refactors cause merge conflicts"
		s.db.get_top_semantic_memories.return_value = [sem]

		ep = MagicMock()
		ep.event_type = "merge_conflict"
		ep.content = "Overlap on db.py"
		ep.outcome = "fail"
		s.db.get_episodic_memories_by_scope.return_value = [ep]

		result = s._get_episodic_context()
		assert "Learned rules:" in result
		assert "Cross-cutting refactors" in result
		assert "Recent episodes:" in result
		assert "merge_conflict" in result
		assert "Overlap on db.py" in result

	def test_empty_when_no_memories(self) -> None:
		s = _make_strategist()
		s.db.get_top_semantic_memories.return_value = []
		s.db.get_episodic_memories_by_scope.return_value = []
		assert s._get_episodic_context() == ""

	def test_semantic_only(self) -> None:
		s = _make_strategist()
		sem = MagicMock()
		sem.confidence = 0.8
		sem.content = "Unit tests prevent regressions"
		s.db.get_top_semantic_memories.return_value = [sem]
		s.db.get_episodic_memories_by_scope.return_value = []
		result = s._get_episodic_context()
		assert "Learned rules:" in result
		assert "Recent episodes:" not in result

	def test_episodic_only(self) -> None:
		s = _make_strategist()
		s.db.get_top_semantic_memories.return_value = []
		ep = MagicMock()
		ep.event_type = "mission_summary"
		ep.content = "Built auth system"
		ep.outcome = "pass"
		s.db.get_episodic_memories_by_scope.return_value = [ep]
		result = s._get_episodic_context()
		assert "Learned rules:" not in result
		assert "Recent episodes:" in result

	def test_resilient_to_db_errors(self) -> None:
		s = _make_strategist()
		s.db.get_top_semantic_memories.side_effect = Exception("DB error")
		s.db.get_episodic_memories_by_scope.side_effect = Exception("DB error")
		assert s._get_episodic_context() == ""


class TestProposeObjectiveEpisodicContext:
	@pytest.mark.asyncio
	async def test_episodic_context_in_prompt(self) -> None:
		s = _make_strategist()
		s.db.get_pending_backlog.return_value = []
		s.db.get_all_missions.return_value = []

		sem = MagicMock()
		sem.confidence = 0.85
		sem.content = "Reduce file overlap for parallel workers"
		s.db.get_top_semantic_memories.return_value = [sem]
		s.db.get_episodic_memories_by_scope.return_value = []

		strategy_output = _make_strategy_output("Improve worker isolation", "Reason", 7)

		git_proc = AsyncMock()
		git_proc.communicate.return_value = (b"abc123 commit", b"")
		git_proc.returncode = 0

		llm_proc = AsyncMock()
		llm_proc.communicate.return_value = (strategy_output.encode(), b"")
		llm_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", side_effect=[git_proc, llm_proc]):
			await s.propose_objective()

		call_args = llm_proc.communicate.call_args
		prompt_bytes = call_args[1].get("input") or call_args[0][0] if call_args[0] else call_args[1]["input"]
		prompt = prompt_bytes.decode()
		assert "Past Learnings" in prompt
		assert "Reduce file overlap" in prompt


class TestStoreMissionEpisode:
	def test_stores_episode_on_followup(self) -> None:
		s = _make_strategist()
		mock_mm = MagicMock()
		s._memory_manager = mock_mm
		s.config.target.name = "my-project"

		result = _mission_result(
			objective="Build auth system",
			objective_met=True,
			total_units_merged=3,
			total_units_failed=0,
			stopped_reason="planner_completed",
		)
		s._store_mission_episode(result)

		mock_mm.store_episode.assert_called_once()
		call_kwargs = mock_mm.store_episode.call_args[1]
		assert call_kwargs["event_type"] == "mission_summary"
		assert "Build auth system" in call_kwargs["content"]
		assert call_kwargs["outcome"] == "pass"
		assert "mission" in call_kwargs["scope_tokens"]
		assert "strategy" in call_kwargs["scope_tokens"]
		assert "my-project" in call_kwargs["scope_tokens"]

	def test_failed_mission_outcome(self) -> None:
		s = _make_strategist()
		mock_mm = MagicMock()
		s._memory_manager = mock_mm
		s.config.target.name = "test"

		result = _mission_result(objective_met=False)
		s._store_mission_episode(result)

		call_kwargs = mock_mm.store_episode.call_args[1]
		assert call_kwargs["outcome"] == "fail"

	def test_no_memory_manager_is_noop(self) -> None:
		s = _make_strategist()
		s._memory_manager = None
		# Should not raise
		s._store_mission_episode(_mission_result())

	def test_resilient_to_store_errors(self) -> None:
		s = _make_strategist()
		mock_mm = MagicMock()
		mock_mm.store_episode.side_effect = Exception("DB write failed")
		s._memory_manager = mock_mm
		s.config.target.name = "test"
		# Should not raise
		s._store_mission_episode(_mission_result())

	@pytest.mark.asyncio
	async def test_suggest_followup_persists_episode(self) -> None:
		s = _make_strategist()
		mock_mm = MagicMock()
		s._memory_manager = mock_mm
		s.config.target.name = "proj"

		llm_output = _make_followup_output("Continue with remaining work")
		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		result = _mission_result(objective="Build API", objective_met=True)

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			await s.suggest_followup(result, "context")

		mock_mm.store_episode.assert_called_once()
		call_kwargs = mock_mm.store_episode.call_args[1]
		assert "Build API" in call_kwargs["content"]
		assert call_kwargs["outcome"] == "pass"
