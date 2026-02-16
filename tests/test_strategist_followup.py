"""Tests for Strategist.suggest_followup async method."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig, PlannerConfig, SchedulerConfig, TargetConfig
from mission_control.models import BacklogItem
from mission_control.strategist import FOLLOWUP_RESULT_MARKER, Strategist


def _config(tmp_path: Path | None = None) -> MissionConfig:
	target_path = str(tmp_path) if tmp_path else "/tmp/test-project"
	return MissionConfig(
		target=TargetConfig(path=target_path),
		planner=PlannerConfig(budget_per_call_usd=0.10),
		scheduler=SchedulerConfig(model="sonnet"),
	)


def _strategist(tmp_path: Path | None = None) -> tuple[Strategist, MagicMock]:
	config = _config(tmp_path)
	db = MagicMock()
	db.get_all_missions.return_value = []
	db.get_pending_backlog.return_value = []
	return Strategist(config=config, db=db), db


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


class TestBuildFollowupPrompt:
	"""Test the prompt construction for follow-up evaluation."""

	def test_includes_mission_result_fields(self) -> None:
		s, _ = _strategist()
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
		s, _ = _strategist()
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "")
		assert "No strategic context available" in prompt

	def test_includes_pending_backlog(self) -> None:
		s, db = _strategist()
		db.get_pending_backlog.return_value = [
			BacklogItem(title="Fix auth", description="Auth is broken", priority_score=8.0),
		]
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "context")
		assert "Fix auth" in prompt

	def test_no_pending_backlog_uses_fallback(self) -> None:
		s, db = _strategist()
		db.get_pending_backlog.return_value = []
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "context")
		assert "No pending backlog items" in prompt

	def test_includes_followup_result_marker(self) -> None:
		s, _ = _strategist()
		result = _mission_result()
		prompt = s._build_followup_prompt(result, "")
		assert "FOLLOWUP_RESULT:" in prompt


class TestParseFollowupOutput:
	"""Test parsing of FOLLOWUP_RESULT from LLM output."""

	def test_valid_marker_with_objective(self) -> None:
		s, _ = _strategist()
		output = _make_followup_output("Continue with remaining work")
		assert s._parse_followup_output(output) == "Continue with remaining work"

	def test_valid_marker_empty_objective(self) -> None:
		s, _ = _strategist()
		output = _make_followup_output("")
		assert s._parse_followup_output(output) == ""

	def test_json_without_marker(self) -> None:
		s, _ = _strategist()
		data = {"next_objective": "Fix bugs", "rationale": "Bugs remain"}
		output = f"Here is the result:\n```json\n{json.dumps(data)}\n```"
		assert s._parse_followup_output(output) == "Fix bugs"

	def test_no_json_returns_empty(self) -> None:
		s, _ = _strategist()
		assert s._parse_followup_output("Just some text with no JSON") == ""

	def test_marker_takes_precedence(self) -> None:
		s, _ = _strategist()
		earlier = json.dumps({"next_objective": "Wrong one", "rationale": "Old"})
		correct = json.dumps({"next_objective": "Right one", "rationale": "New"})
		output = f"Earlier: {earlier}\n\n{FOLLOWUP_RESULT_MARKER}{correct}"
		assert s._parse_followup_output(output) == "Right one"

	def test_whitespace_stripped(self) -> None:
		s, _ = _strategist()
		output = _make_followup_output("  Fix auth  ")
		assert s._parse_followup_output(output) == "Fix auth"


class TestSuggestFollowup:
	"""Test the async suggest_followup method."""

	@pytest.mark.asyncio
	async def test_returns_objective_from_llm(self) -> None:
		s, _ = _strategist()
		llm_output = _make_followup_output("Continue building auth system")

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(), "strategic context")

		assert result == "Continue building auth system"

	@pytest.mark.asyncio
	async def test_returns_empty_when_no_followup(self) -> None:
		s, _ = _strategist()
		llm_output = _make_followup_output("")

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(objective_met=True), "")

		assert result == ""

	@pytest.mark.asyncio
	async def test_returns_empty_on_subprocess_failure(self) -> None:
		s, _ = _strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"", b"some error")
		mock_proc.returncode = 1

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(), "context")

		assert result == ""

	@pytest.mark.asyncio
	async def test_returns_empty_on_timeout(self) -> None:
		s, _ = _strategist()

		mock_proc = AsyncMock()
		mock_proc.kill = AsyncMock()
		mock_proc.wait = AsyncMock()

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch("mission_control.strategist.asyncio.wait_for", side_effect=asyncio.TimeoutError):
				result = await s.suggest_followup(_mission_result(), "context")

		assert result == ""

	@pytest.mark.asyncio
	async def test_returns_empty_on_unparseable_output(self) -> None:
		s, _ = _strategist()

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"No valid JSON here at all", b"")
		mock_proc.returncode = 0

		with patch("mission_control.strategist.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await s.suggest_followup(_mission_result(), "context")

		assert result == ""

	@pytest.mark.asyncio
	async def test_passes_strategic_context_to_prompt(self) -> None:
		s, _ = _strategist()
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
		s, _ = _strategist()
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
		s, _ = _strategist(tmp_path)
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
		s, _ = _strategist()
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
