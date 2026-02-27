"""Tests for CriticAgent -- prompt building, parsing, context gathering."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.batch_analyzer import BatchSignals
from mission_control.config import DeliberationConfig, MissionConfig, TargetConfig
from mission_control.critic_agent import (
	CRITIC_RESULT_MARKER,
	CriticAgent,
	_build_batch_signals_text,
	_parse_critic_output,
)
from mission_control.models import CriticFinding, Mission


def _config(tmp_path: Path | None = None) -> MissionConfig:
	target_path = str(tmp_path) if tmp_path else "/tmp/test-project"
	cfg = MissionConfig(
		target=TargetConfig(path=target_path, name="test"),
		deliberation=DeliberationConfig(
			critic_budget_usd=1.0, timeout=30,
		),
	)
	return cfg


def _signals(**kwargs) -> BatchSignals:
	defaults = {
		"file_hotspots": [("src/auth.py", 5)],
		"failure_clusters": {"Import error": 3},
		"stalled_areas": ["src/db.py"],
		"effort_distribution": {"src/auth.py": 0.6, "tests/": 0.4},
		"retry_depth": {},
		"knowledge_gaps": [],
	}
	defaults.update(kwargs)
	return BatchSignals(**defaults)


class TestParseCriticOutput:
	def test_valid_marker_output(self) -> None:
		data = {
			"findings": ["uses SQLAlchemy"],
			"risks": ["tight coupling"],
			"gaps": ["unknown schema"],
			"open_questions": ["which ORM?"],
			"verdict": "sufficient",
			"confidence": 0.85,
			"strategy_text": "Use SQLAlchemy with Alembic",
		}
		output = f"Analysis complete.\n\n{CRITIC_RESULT_MARKER}{json.dumps(data)}"
		result = _parse_critic_output(output)
		assert result.findings == ["uses SQLAlchemy"]
		assert result.risks == ["tight coupling"]
		assert result.gaps == ["unknown schema"]
		assert result.verdict == "sufficient"
		assert result.confidence == 0.85
		assert "SQLAlchemy" in result.strategy_text

	def test_json_without_marker(self) -> None:
		data = {"findings": ["f1"], "verdict": "needs_refinement", "confidence": 0.5}
		output = f"Here is the result:\n```json\n{json.dumps(data)}\n```"
		result = _parse_critic_output(output)
		assert result.findings == ["f1"]
		assert result.verdict == "needs_refinement"

	def test_no_json_returns_raw(self) -> None:
		output = "Just plain text analysis with no JSON"
		result = _parse_critic_output(output)
		assert len(result.findings) == 1
		assert "plain text" in result.findings[0]
		assert result.verdict == "needs_refinement"

	def test_empty_output(self) -> None:
		result = _parse_critic_output("")
		assert result.findings == []
		assert result.verdict == "needs_refinement"

	def test_marker_takes_precedence(self) -> None:
		wrong = json.dumps({"findings": ["wrong"], "verdict": "sufficient"})
		right = json.dumps({"findings": ["right"], "verdict": "needs_refinement"})
		output = f"Earlier: {wrong}\n\n{CRITIC_RESULT_MARKER}{right}"
		result = _parse_critic_output(output)
		assert result.findings == ["right"]
		assert result.verdict == "needs_refinement"

	def test_proposed_objective_for_chaining(self) -> None:
		data = {
			"findings": [], "verdict": "sufficient",
			"proposed_objective": "Build auth system",
			"strategy_text": "Next priority",
		}
		output = f"{CRITIC_RESULT_MARKER}{json.dumps(data)}"
		result = _parse_critic_output(output)
		assert result.proposed_objective == "Build auth system"

	def test_partial_fields_default(self) -> None:
		data = {"findings": ["f1"]}
		output = f"{CRITIC_RESULT_MARKER}{json.dumps(data)}"
		result = _parse_critic_output(output)
		assert result.findings == ["f1"]
		assert result.risks == []
		assert result.gaps == []
		assert result.open_questions == []
		assert result.verdict == "needs_refinement"
		assert result.confidence == 0.0


class TestBuildBatchSignalsText:
	def test_formats_hotspots(self) -> None:
		text = _build_batch_signals_text(_signals())
		assert "src/auth.py" in text
		assert "5 touches" in text

	def test_formats_failures(self) -> None:
		text = _build_batch_signals_text(_signals())
		assert "Import error" in text
		assert "3 failures" in text

	def test_empty_signals(self) -> None:
		text = _build_batch_signals_text(BatchSignals())
		assert "(none)" in text


class TestCriticGatherContext:
	def test_gathers_backlog(self, tmp_path: Path) -> None:
		(tmp_path / "BACKLOG.md").write_text("# My Backlog\n- item 1")
		db = MagicMock()
		db.get_all_missions.return_value = []
		db.get_top_semantic_memories.return_value = []
		agent = CriticAgent(_config(tmp_path), db)
		ctx = agent.gather_context()
		assert "My Backlog" in ctx

	def test_no_context_available(self, tmp_path: Path) -> None:
		db = MagicMock()
		db.get_all_missions.return_value = []
		db.get_top_semantic_memories.return_value = []
		db.get_episodic_memories_by_scope.return_value = []
		del db.get_strategic_context
		agent = CriticAgent(_config(tmp_path), db)
		ctx = agent.gather_context()
		assert "No project context available" in ctx

	@pytest.mark.asyncio
	async def test_gather_context_async_includes_git(self, tmp_path: Path) -> None:
		db = MagicMock()
		db.get_all_missions.return_value = []
		db.get_top_semantic_memories.return_value = []
		del db.get_strategic_context
		agent = CriticAgent(_config(tmp_path), db)
		with patch(
			"mission_control.critic_agent.get_git_log",
			new_callable=AsyncMock, return_value="abc123 feat: add X",
		):
			ctx = await agent.gather_context_async()
		assert "abc123" in ctx


class TestCriticResearch:
	@pytest.mark.asyncio
	async def test_research_returns_stub(self, tmp_path: Path) -> None:
		"""research() is now a stub -- returns minimal finding, no LLM call."""
		db = MagicMock()
		agent = CriticAgent(_config(tmp_path), db)

		with patch.object(agent, "_invoke_llm", new_callable=AsyncMock) as mock_llm:
			result = await agent.research("Build API", "context here")

		mock_llm.assert_not_called()
		assert result.verdict == "needs_refinement"
		assert result.confidence == 0.5


class TestCriticReviewPlan:
	@pytest.mark.asyncio
	async def test_review_sufficient(self, tmp_path: Path) -> None:
		db = MagicMock()
		agent = CriticAgent(_config(tmp_path), db)
		from mission_control.models import WorkUnit
		units = [
			WorkUnit(title="Add auth", description="JWT auth", files_hint="src/auth.py", priority=1),
			WorkUnit(title="Add tests", description="Test auth", files_hint="tests/test_auth.py", priority=2),
		]
		prev = CriticFinding(findings=["uses Flask"], risks=["coupling"])
		verdict_data = {"verdict": "sufficient", "confidence": 0.9, "findings": ["plan looks good"]}
		fake_output = f"{CRITIC_RESULT_MARKER}{json.dumps(verdict_data)}"

		with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=fake_output):
			result = await agent.review_plan("Build API", units, prev)

		assert result.verdict == "sufficient"
		assert result.confidence == 0.9

	@pytest.mark.asyncio
	async def test_review_needs_refinement(self, tmp_path: Path) -> None:
		db = MagicMock()
		agent = CriticAgent(_config(tmp_path), db)
		from mission_control.models import WorkUnit
		units = [WorkUnit(title="Fix everything", description="all in one", priority=1)]
		prev = CriticFinding(findings=["complex codebase"])
		verdict_data = {"verdict": "needs_refinement", "gaps": ["too coarse-grained"]}
		fake_output = f"{CRITIC_RESULT_MARKER}{json.dumps(verdict_data)}"

		with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=fake_output):
			result = await agent.review_plan("Build API", units, prev)

		assert result.verdict == "needs_refinement"
		assert "too coarse-grained" in result.gaps

	@pytest.mark.asyncio
	async def test_review_prompt_is_feasibility_focused(self, tmp_path: Path) -> None:
		"""review_plan prompt uses feasibility framing, not strategic."""
		db = MagicMock()
		agent = CriticAgent(_config(tmp_path), db)
		from mission_control.models import WorkUnit
		units = [WorkUnit(title="Task", description="x", files_hint="a.py", priority=1)]
		prev = CriticFinding()
		fake_output = f'{CRITIC_RESULT_MARKER}{json.dumps({"verdict": "sufficient"})}'

		with patch.object(
			agent, "_invoke_llm", new_callable=AsyncMock, return_value=fake_output,
		) as mock_llm:
			await agent.review_plan("Build API", units, prev)

		prompt = mock_llm.call_args[0][0]
		assert "feasibility reviewer" in prompt
		assert "NOT to set strategy" in prompt
		assert "Do NOT propose new strategic direction" in prompt


class TestCriticProposeNext:
	@pytest.mark.asyncio
	async def test_propose_next_success(self, tmp_path: Path) -> None:
		db = MagicMock()
		db.get_all_missions.return_value = []
		db.get_top_semantic_memories.return_value = []
		del db.get_strategic_context
		agent = CriticAgent(_config(tmp_path), db)

		mission = Mission(objective="Build API", status="completed")
		mock_result = MagicMock()
		mock_result.objective_met = True
		mock_result.total_units_merged = 5
		mock_result.total_units_failed = 0
		mock_result.stopped_reason = "planner_completed"

		verdict_data = {
			"proposed_objective": "Add auth system",
			"strategy_text": "Next step",
			"verdict": "sufficient",
		}
		fake_output = f"{CRITIC_RESULT_MARKER}{json.dumps(verdict_data)}"

		with (
			patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=fake_output),
			patch("mission_control.critic_agent.get_git_log", new_callable=AsyncMock, return_value=""),
		):
			result = await agent.propose_next(mission, mock_result, "context")

		assert result.proposed_objective == "Add auth system"
