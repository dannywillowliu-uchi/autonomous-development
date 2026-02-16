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
			with pytest.raises(RuntimeError, match="Strategist subprocess failed"):
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
