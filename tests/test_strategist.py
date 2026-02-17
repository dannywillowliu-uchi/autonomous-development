"""Tests for the strategist module."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig, PlannerConfig, SchedulerConfig, TargetConfig
from mission_control.models import BacklogItem, Mission
from mission_control.strategist import (
	FOLLOWUP_RESULT_MARKER,
	STRATEGY_RESULT_MARKER,
	Strategist,
	_build_strategy_prompt,
)


def _config(tmp_path: Path | None = None) -> MissionConfig:
	target_path = str(tmp_path) if tmp_path else "/tmp/test-project"
	return MissionConfig(
		target=TargetConfig(path=target_path),
		planner=PlannerConfig(budget_per_call_usd=0.10),
		scheduler=SchedulerConfig(model="sonnet"),
	)


def _strategist(tmp_path: Path | None = None) -> Strategist:
	config = _config(tmp_path)
	db = MagicMock()
	db.get_all_missions.return_value = []
	db.get_pending_backlog.return_value = []
	return Strategist(config=config, db=db)


def _make_strategy_output(objective: str, rationale: str, score: int) -> str:
	data = {"objective": objective, "rationale": rationale, "ambition_score": score}
	return f"Some reasoning...\n\n{STRATEGY_RESULT_MARKER}{json.dumps(data)}"


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
		self.strategist = _strategist()

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
		s = _strategist(tmp_path)
		assert "My Backlog" in s._read_backlog()

	def test_read_backlog_missing(self, tmp_path: Path) -> None:
		s = _strategist(tmp_path)
		assert s._read_backlog() == ""

	def test_get_git_log_success(self) -> None:
		s = _strategist()
		with patch("mission_control.strategist.subprocess.run") as mock_run:
			mock_run.return_value = MagicMock(returncode=0, stdout="abc123 feat: add X\ndef456 fix: bug Y")
			result = s._get_git_log()
		assert "abc123" in result
		assert "def456" in result

	def test_get_git_log_failure(self) -> None:
		s = _strategist()
		with patch("mission_control.strategist.subprocess.run") as mock_run:
			mock_run.return_value = MagicMock(returncode=128, stdout="")
			result = s._get_git_log()
		assert result == ""

	def test_get_git_log_timeout(self) -> None:
		s = _strategist()
		with patch("mission_control.strategist.subprocess.run") as mock_run:
			mock_run.side_effect = TimeoutError()
			result = s._get_git_log()
		assert result == ""

	def test_get_past_missions_empty(self) -> None:
		s = _strategist()
		s.db.get_all_missions.return_value = []
		assert s._get_past_missions() == ""

	def test_get_past_missions_formats(self) -> None:
		s = _strategist()
		m = Mission(objective="Build API server", status="completed", total_rounds=3, final_score=8.5)
		s.db.get_all_missions.return_value = [m]
		result = s._get_past_missions()
		assert "completed" in result
		assert "Build API server" in result
		assert "rounds=3" in result

	def test_get_strategic_context_no_method(self) -> None:
		s = _strategist()
		del s.db.get_strategic_context
		assert s._get_strategic_context() == ""

	def test_get_strategic_context_with_data(self) -> None:
		s = _strategist()
		s.db.get_strategic_context.return_value = ["Focus on auth", "Testing needed"]
		result = s._get_strategic_context()
		assert "Focus on auth" in result
		assert "Testing needed" in result

	def test_get_strategic_context_empty(self) -> None:
		s = _strategist()
		s.db.get_strategic_context.return_value = []
		assert s._get_strategic_context() == ""

	def test_get_pending_backlog_empty(self) -> None:
		s = _strategist()
		s.db.get_pending_backlog.return_value = []
		assert s._get_pending_backlog() == ""

	def test_get_pending_backlog_formats(self) -> None:
		s = _strategist()
		item = BacklogItem(title="Fix auth", description="Auth is broken in prod", priority_score=8.5)
		s.db.get_pending_backlog.return_value = [item]
		result = s._get_pending_backlog()
		assert "Fix auth" in result
		assert "score=8.5" in result

	def test_get_pending_backlog_uses_pinned_score(self) -> None:
		s = _strategist()
		item = BacklogItem(title="Pinned task", description="Important", priority_score=3.0, pinned_score=9.5)
		s.db.get_pending_backlog.return_value = [item]
		result = s._get_pending_backlog()
		assert "score=9.5" in result


# -- propose_objective integration --


class TestProposeObjective:
	@pytest.mark.asyncio
	async def test_success(self) -> None:
		s = _strategist()
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
		s = _strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"", b"some error")
		mock_proc.returncode = 1

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			with pytest.raises(RuntimeError, match="strategist subprocess failed"):
				await s.propose_objective()

	@pytest.mark.asyncio
	async def test_timeout_raises(self) -> None:
		s = _strategist()

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
		s = _strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"No valid JSON here", b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			with pytest.raises(ValueError, match="Could not parse"):
				await s.propose_objective()

	@pytest.mark.asyncio
	async def test_sets_cwd_to_target_path(self, tmp_path: Path) -> None:
		s = _strategist(tmp_path)
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
		s = _strategist()
		m = Mission(objective="Past mission", status="completed", total_rounds=2, final_score=7.0)
		s.db.get_all_missions.return_value = [m]
		item = BacklogItem(title="Backlog task", description="Do this", priority_score=6.0)
		s.db.get_pending_backlog.return_value = [item]

		strategy_output = _make_strategy_output("Next objective", "Based on context", 6)
		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (strategy_output.encode(), b"")
		mock_proc.returncode = 0

		with (
			patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc),
			patch("mission_control.strategist.subprocess.run") as mock_git,
		):
			mock_git.return_value = MagicMock(returncode=0, stdout="abc123 recent commit")
			await s.propose_objective()

		# Verify the prompt was sent to stdin
		call_args = mock_proc.communicate.call_args
		prompt_bytes = call_args[1].get("input") or call_args[0][0] if call_args[0] else call_args[1]["input"]
		prompt = prompt_bytes.decode()
		assert "Past mission" in prompt or "abc123" in prompt


# -- Follow-up suggestion helpers --


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


# -- Follow-up prompt building --


class TestBuildFollowupPrompt:
	"""Test the prompt construction for follow-up evaluation."""

	def test_includes_mission_result_fields(self) -> None:
		s = _strategist()
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
		s = _strategist()
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "")
		assert "No strategic context available" in prompt

	def test_includes_pending_backlog(self) -> None:
		s = _strategist()
		s.db.get_pending_backlog.return_value = [
			BacklogItem(title="Fix auth", description="Auth is broken", priority_score=8.0),
		]
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "context")
		assert "Fix auth" in prompt

	def test_no_pending_backlog_uses_fallback(self) -> None:
		s = _strategist()
		s.db.get_pending_backlog.return_value = []
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "context")
		assert "No pending backlog items" in prompt

	def test_includes_followup_result_marker(self) -> None:
		s = _strategist()
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "")
		assert "FOLLOWUP_RESULT:" in prompt


# -- Follow-up parsing --


class TestParseFollowupOutput:
	"""Test parsing of FOLLOWUP_RESULT from LLM output."""

	def test_valid_marker_with_objective(self) -> None:
		s = _strategist()
		output = _make_followup_output("Continue with remaining work")
		assert s._parse_followup_output(output) == "Continue with remaining work"

	def test_valid_marker_empty_objective(self) -> None:
		s = _strategist()
		output = _make_followup_output("")
		assert s._parse_followup_output(output) == ""

	def test_json_without_marker(self) -> None:
		s = _strategist()
		data = {"next_objective": "Fix bugs", "rationale": "Bugs remain"}
		output = f"Here is the result:\n```json\n{json.dumps(data)}\n```"
		assert s._parse_followup_output(output) == "Fix bugs"

	def test_no_json_returns_empty(self) -> None:
		s = _strategist()
		assert s._parse_followup_output("Just some text with no JSON") == ""

	def test_marker_takes_precedence(self) -> None:
		s = _strategist()
		earlier = json.dumps({"next_objective": "Wrong one", "rationale": "Old"})
		correct = json.dumps({"next_objective": "Right one", "rationale": "New"})
		output = f"Earlier: {earlier}\n\n{FOLLOWUP_RESULT_MARKER}{correct}"
		assert s._parse_followup_output(output) == "Right one"

	def test_whitespace_stripped(self) -> None:
		s = _strategist()
		output = _make_followup_output("  Fix auth  ")
		assert s._parse_followup_output(output) == "Fix auth"


# -- Follow-up suggestion --


class TestSuggestFollowup:
	"""Test the async suggest_followup method."""

	@pytest.mark.asyncio
	async def test_returns_objective_from_llm(self) -> None:
		s = _strategist()
		llm_output = _make_followup_output("Continue building auth system")

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(), "strategic context")

		assert result == "Continue building auth system"

	@pytest.mark.asyncio
	async def test_returns_empty_when_no_followup(self) -> None:
		s = _strategist()
		llm_output = _make_followup_output("")

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(objective_met=True), "")

		assert result == ""

	@pytest.mark.asyncio
	async def test_returns_empty_on_subprocess_failure(self) -> None:
		s = _strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"", b"some error")
		mock_proc.returncode = 1

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(), "context")

		assert result == ""

	@pytest.mark.asyncio
	async def test_returns_empty_on_timeout(self) -> None:
		s = _strategist()

		mock_proc = AsyncMock()
		mock_proc.kill = AsyncMock()
		mock_proc.wait = AsyncMock()

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch("mission_control.strategist.asyncio.wait_for", side_effect=asyncio.TimeoutError):
				result = await s.suggest_followup(_mission_result(), "context")

		assert result == ""

	@pytest.mark.asyncio
	async def test_returns_empty_on_unparseable_output(self) -> None:
		s = _strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"No valid JSON here at all", b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(), "context")

		assert result == ""

	@pytest.mark.asyncio
	async def test_passes_strategic_context_to_prompt(self) -> None:
		s = _strategist()
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
		s = _strategist()
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
		s = _strategist(tmp_path)
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
		s = _strategist()
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
