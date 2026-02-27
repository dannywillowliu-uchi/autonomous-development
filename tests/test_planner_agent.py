"""Tests for PlannerAgent -- critic-enriched planning wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig, PlannerConfig, TargetConfig
from mission_control.models import CriticFinding, Plan, PlanNode, WorkUnit
from mission_control.planner_agent import PlannerAgent, _format_critic_context


def _config(tmp_path: Path | None = None) -> MissionConfig:
	target_path = str(tmp_path) if tmp_path else "/tmp/test-project"
	return MissionConfig(
		target=TargetConfig(path=target_path),
		planner=PlannerConfig(budget_per_call_usd=0.10),
	)


class TestFormatCriticContext:
	def test_full_finding(self) -> None:
		finding = CriticFinding(
			findings=["uses Flask", "has auth module"],
			risks=["tight coupling"],
			gaps=["unknown schema"],
			open_questions=["which ORM?"],
			strategy_text="Use SQLAlchemy with Alembic",
		)
		text = _format_critic_context(finding)
		assert "## Critic Strategy" in text
		assert "SQLAlchemy" in text
		assert "## Key Findings" in text
		assert "uses Flask" in text
		assert "## Risks" in text
		assert "tight coupling" in text
		assert "## Knowledge Gaps" in text
		assert "## Open Questions" in text

	def test_empty_finding(self) -> None:
		finding = CriticFinding()
		text = _format_critic_context(finding)
		assert text == ""

	def test_partial_finding(self) -> None:
		finding = CriticFinding(
			findings=["one finding"],
			strategy_text="Do X",
		)
		text = _format_critic_context(finding)
		assert "## Critic Strategy" in text
		assert "## Key Findings" in text
		assert "## Risks" not in text
		assert "## Knowledge Gaps" not in text


class TestPlannerDecompose:
	@pytest.mark.asyncio
	async def test_decompose_with_critic_findings(self, tmp_path: Path) -> None:
		config = _config(tmp_path)
		db = MagicMock()
		agent = PlannerAgent(config, db)

		# Mock the inner RecursivePlanner
		mock_plan = Plan(id="p1", objective="test")
		mock_root = PlanNode(id="n1", plan_id="p1")
		mock_unit = WorkUnit(
			id="wu1", plan_id="p1", title="Add auth",
			files_hint="src/auth.py", priority=1,
		)
		mock_root._child_leaves = [(mock_root, mock_unit)]  # type: ignore[attr-defined]

		with patch.object(
			agent._inner, "plan_round",
			new_callable=AsyncMock,
			return_value=(mock_plan, mock_root),
		) as mock_plan_round:
			finding = CriticFinding(
				findings=["uses Flask"],
				strategy_text="Use JWT",
			)
			plan, units = await agent.decompose("Build API", finding)

		assert len(units) == 1
		assert units[0].title == "Add auth"
		# Verify critic context was passed to inner planner
		call_kwargs = mock_plan_round.call_args[1]
		assert "JWT" in call_kwargs["feedback_context"]
		assert "uses Flask" in call_kwargs["feedback_context"]

	@pytest.mark.asyncio
	async def test_decompose_empty_finding(self, tmp_path: Path) -> None:
		config = _config(tmp_path)
		db = MagicMock()
		agent = PlannerAgent(config, db)

		mock_plan = Plan(id="p1", objective="test")
		mock_root = PlanNode(id="n1", plan_id="p1")
		mock_unit = WorkUnit(id="wu1", plan_id="p1", title="Task 1")
		mock_root._child_leaves = [(mock_root, mock_unit)]  # type: ignore[attr-defined]

		with patch.object(
			agent._inner, "plan_round",
			new_callable=AsyncMock,
			return_value=(mock_plan, mock_root),
		):
			plan, units = await agent.decompose("Objective", CriticFinding())

		assert len(units) == 1


class TestPlannerRefine:
	@pytest.mark.asyncio
	async def test_refine_includes_previous_units(self, tmp_path: Path) -> None:
		config = _config(tmp_path)
		db = MagicMock()
		agent = PlannerAgent(config, db)

		existing_units = [
			WorkUnit(title="Old task", description="do something", files_hint="src/old.py"),
		]

		mock_plan = Plan(id="p2", objective="test")
		mock_root = PlanNode(id="n2", plan_id="p2")
		mock_unit = WorkUnit(id="wu2", plan_id="p2", title="Refined task")
		mock_root._child_leaves = [(mock_root, mock_unit)]  # type: ignore[attr-defined]

		with patch.object(
			agent._inner, "plan_round",
			new_callable=AsyncMock,
			return_value=(mock_plan, mock_root),
		) as mock_plan_round:
			critic_feedback = CriticFinding(
				gaps=["missing error handling"],
				verdict="needs_refinement",
			)
			plan, units = await agent.refine("Objective", existing_units, critic_feedback)

		assert len(units) == 1
		assert units[0].title == "Refined task"
		# Verify previous units and gaps are in the context
		call_kwargs = mock_plan_round.call_args[1]
		assert "Old task" in call_kwargs["feedback_context"]
		assert "missing error handling" in call_kwargs["feedback_context"]


class TestExtractUnits:
	def test_extract_from_child_leaves(self, tmp_path: Path) -> None:
		agent = PlannerAgent(_config(tmp_path), MagicMock())
		node = PlanNode()
		wu1 = WorkUnit(title="Unit 1")
		wu2 = WorkUnit(title="Unit 2")
		node._child_leaves = [(node, wu1), (node, wu2)]  # type: ignore[attr-defined]
		units = agent._extract_units(node)
		assert len(units) == 2

	def test_extract_from_forced_unit(self, tmp_path: Path) -> None:
		agent = PlannerAgent(_config(tmp_path), MagicMock())
		node = PlanNode()
		wu = WorkUnit(title="Forced")
		node._forced_unit = wu  # type: ignore[attr-defined]
		units = agent._extract_units(node)
		assert len(units) == 1
		assert units[0].title == "Forced"

	def test_extract_from_subdivided(self, tmp_path: Path) -> None:
		agent = PlannerAgent(_config(tmp_path), MagicMock())
		root = PlanNode()
		child = PlanNode()
		wu = WorkUnit(title="Leaf")
		child._child_leaves = [(child, wu)]  # type: ignore[attr-defined]
		root._subdivided_children = [child]  # type: ignore[attr-defined]
		units = agent._extract_units(root)
		assert len(units) == 1
		assert units[0].title == "Leaf"

	def test_extract_empty_node(self, tmp_path: Path) -> None:
		agent = PlannerAgent(_config(tmp_path), MagicMock())
		node = PlanNode()
		units = agent._extract_units(node)
		assert len(units) == 0
