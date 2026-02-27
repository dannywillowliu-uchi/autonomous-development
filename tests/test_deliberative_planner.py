"""Tests for DeliberativePlanner -- dual-agent deliberation loop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import (
	DeliberationConfig,
	MissionConfig,
	PlannerConfig,
	ResearchConfig,
	TargetConfig,
)
from mission_control.db import Database
from mission_control.deliberative_planner import DeliberativePlanner
from mission_control.models import CriticFinding, Epoch, Mission, Plan, WorkUnit


def _config(tmp_path: Path) -> MissionConfig:
	cfg = MissionConfig(
		target=TargetConfig(path=str(tmp_path), name="test"),
		planner=PlannerConfig(budget_per_call_usd=0.10),
		deliberation=DeliberationConfig(
			max_rounds=3, critic_budget_usd=1.0,
			planner_budget_usd=0.5, timeout=30,
		),
		research=ResearchConfig(enabled=False),
	)
	return cfg


def _mission() -> Mission:
	return Mission(id="m1", objective="Build a REST API with auth")


def _sufficient_finding(**kwargs) -> CriticFinding:
	defaults = {
		"findings": ["codebase uses Flask"],
		"risks": ["tight coupling"],
		"gaps": [],
		"open_questions": [],
		"verdict": "sufficient",
		"confidence": 0.9,
		"strategy_text": "Use JWT with Flask",
	}
	defaults.update(kwargs)
	return CriticFinding(**defaults)


def _refinement_finding(**kwargs) -> CriticFinding:
	defaults = {
		"findings": ["plan too coarse"],
		"risks": ["merge conflicts"],
		"gaps": ["missing error handling"],
		"open_questions": [],
		"verdict": "needs_refinement",
		"confidence": 0.5,
		"strategy_text": "Need finer decomposition",
	}
	defaults.update(kwargs)
	return CriticFinding(**defaults)


def _mock_units(count: int = 2) -> list[WorkUnit]:
	return [
		WorkUnit(
			id=f"wu-{i}", plan_id="p1",
			title=f"Task {i}", files_hint=f"src/mod{i}.py",
			priority=i + 1,
		)
		for i in range(count)
	]


class TestDeliberationLoop:
	@pytest.mark.asyncio
	async def test_single_round_sufficient(self, tmp_path: Path) -> None:
		"""Critic approves on first review -> 2 total rounds."""
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_config(tmp_path), db)

		mock_plan = Plan(id="p1", objective="test")
		units = _mock_units(2)

		# Round 1: research -> decompose
		# Round 2: review -> sufficient (no refine needed)
		with (
			patch.object(
				planner._critic, "research",
				new_callable=AsyncMock,
				return_value=_refinement_finding(),
			),
			patch.object(
				planner._critic, "review_plan",
				new_callable=AsyncMock,
				return_value=_sufficient_finding(),
			),
			patch.object(
				planner._planner, "decompose",
				new_callable=AsyncMock,
				return_value=(mock_plan, units),
			),
			patch.object(
				planner._critic, "gather_context_async",
				new_callable=AsyncMock,
				return_value="context",
			),
		):
			plan, result_units, epoch = await planner.get_next_units(
				_mission(), max_units=3,
			)

		assert len(result_units) == 2
		assert isinstance(epoch, Epoch)
		assert epoch.units_planned == 2

	@pytest.mark.asyncio
	async def test_refinement_loop(self, tmp_path: Path) -> None:
		"""Critic rejects first plan, approves second -> 3 rounds."""
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_config(tmp_path), db)

		initial_units = _mock_units(1)
		refined_units = _mock_units(3)
		mock_plan = Plan(id="p1", objective="test")

		# Round 1: research (returns needs_refinement finding)
		# Round 2: review (needs_refinement) -> refine
		# Round 3: review (sufficient) -> done
		review_results = [
			_refinement_finding(),
			_sufficient_finding(),
		]

		with (
			patch.object(
				planner._critic, "research",
				new_callable=AsyncMock,
				return_value=_refinement_finding(),
			),
			patch.object(
				planner._critic, "review_plan",
				new_callable=AsyncMock,
				side_effect=review_results,
			),
			patch.object(
				planner._planner, "decompose",
				new_callable=AsyncMock,
				return_value=(mock_plan, initial_units),
			),
			patch.object(
				planner._planner, "refine",
				new_callable=AsyncMock,
				return_value=(mock_plan, refined_units),
			),
			patch.object(
				planner._critic, "gather_context_async",
				new_callable=AsyncMock,
				return_value="context",
			),
		):
			plan, result_units, epoch = await planner.get_next_units(
				_mission(), max_units=5,
			)

		assert len(result_units) == 3
		assert epoch.units_planned == 3

	@pytest.mark.asyncio
	async def test_max_rounds_cap(self, tmp_path: Path) -> None:
		"""Critic never approves -> stops at max_rounds."""
		cfg = _config(tmp_path)
		cfg.deliberation.max_rounds = 2
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(cfg, db)

		units = _mock_units(2)
		mock_plan = Plan(id="p1", objective="test")

		with (
			patch.object(
				planner._critic, "research",
				new_callable=AsyncMock,
				return_value=_refinement_finding(),
			),
			patch.object(
				planner._critic, "review_plan",
				new_callable=AsyncMock,
				return_value=_refinement_finding(),
			),
			patch.object(
				planner._planner, "decompose",
				new_callable=AsyncMock,
				return_value=(mock_plan, units),
			),
			patch.object(
				planner._planner, "refine",
				new_callable=AsyncMock,
				return_value=(mock_plan, units),
			),
			patch.object(
				planner._critic, "gather_context_async",
				new_callable=AsyncMock,
				return_value="context",
			),
		):
			plan, result_units, epoch = await planner.get_next_units(
				_mission(), max_units=5,
			)

		# Should still return units even if not "sufficient"
		assert len(result_units) == 2

	@pytest.mark.asyncio
	async def test_empty_units_no_review(self, tmp_path: Path) -> None:
		"""Planner returns empty units (objective met) -> no review loop."""
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_config(tmp_path), db)

		mock_plan = Plan(id="p1", objective="test")

		with (
			patch.object(
				planner._critic, "research",
				new_callable=AsyncMock,
				return_value=_sufficient_finding(),
			),
			patch.object(
				planner._critic, "review_plan",
				new_callable=AsyncMock,
			) as mock_review,
			patch.object(
				planner._planner, "decompose",
				new_callable=AsyncMock,
				return_value=(mock_plan, []),
			),
			patch.object(
				planner._critic, "gather_context_async",
				new_callable=AsyncMock,
				return_value="context",
			),
		):
			plan, result_units, epoch = await planner.get_next_units(
				_mission(), max_units=3,
			)

		assert result_units == []
		mock_review.assert_not_called()

	@pytest.mark.asyncio
	async def test_max_units_limit(self, tmp_path: Path) -> None:
		"""Units are capped at max_units."""
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_config(tmp_path), db)

		units = _mock_units(5)
		mock_plan = Plan(id="p1", objective="test")

		with (
			patch.object(
				planner._critic, "research",
				new_callable=AsyncMock,
				return_value=_sufficient_finding(),
			),
			patch.object(
				planner._critic, "review_plan",
				new_callable=AsyncMock,
				return_value=_sufficient_finding(),
			),
			patch.object(
				planner._planner, "decompose",
				new_callable=AsyncMock,
				return_value=(mock_plan, units),
			),
			patch.object(
				planner._critic, "gather_context_async",
				new_callable=AsyncMock,
				return_value="context",
			),
		):
			plan, result_units, epoch = await planner.get_next_units(
				_mission(), max_units=2,
			)

		assert len(result_units) == 2
		assert epoch.units_planned == 5  # planned count before cap


class TestWriteStrategy:
	def test_writes_strategy_file(self, tmp_path: Path) -> None:
		db = Database(":memory:")
		planner = DeliberativePlanner(_config(tmp_path), db)
		planner._write_strategy("# Strategy\nUse JWT")
		path = tmp_path / "MISSION_STRATEGY.md"
		assert path.exists()
		assert "Use JWT" in path.read_text()


class TestStoreKnowledge:
	def test_stores_findings(self, tmp_path: Path) -> None:
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_config(tmp_path), db)
		finding = CriticFinding(
			findings=["finding 1", "finding 2"],
			confidence=0.8,
		)
		planner._store_knowledge(finding, _mission())
		items = db.get_knowledge_for_mission("m1")
		assert len(items) == 2
		assert items[0].source_unit_type == "research"

	def test_empty_findings(self, tmp_path: Path) -> None:
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_config(tmp_path), db)
		planner._store_knowledge(CriticFinding(), _mission())
		items = db.get_knowledge_for_mission("m1")
		assert len(items) == 0


class TestProposeNextObjective:
	@pytest.mark.asyncio
	async def test_propose_next(self, tmp_path: Path) -> None:
		db = Database(":memory:")
		planner = DeliberativePlanner(_config(tmp_path), db)

		mission = Mission(objective="Build API", status="completed")
		mock_result = MagicMock()
		mock_result.objective_met = True

		chaining_finding = CriticFinding(
			proposed_objective="Add authentication",
			strategy_text="Next priority after API is built",
		)

		with (
			patch.object(
				planner._critic, "gather_context_async",
				new_callable=AsyncMock,
				return_value="context",
			),
			patch.object(
				planner._critic, "propose_next",
				new_callable=AsyncMock,
				return_value=chaining_finding,
			),
		):
			objective, rationale = await planner.propose_next_objective(mission, mock_result)

		assert objective == "Add authentication"
		assert "Next priority" in rationale


class TestEpochCounting:
	@pytest.mark.asyncio
	async def test_epoch_increments(self, tmp_path: Path) -> None:
		db = Database(":memory:")
		db.insert_mission(_mission())
		planner = DeliberativePlanner(_config(tmp_path), db)

		mock_plan = Plan(id="p1", objective="test")
		units = _mock_units(1)

		with (
			patch.object(planner._critic, "research", new_callable=AsyncMock, return_value=_sufficient_finding()),
			patch.object(planner._critic, "review_plan", new_callable=AsyncMock, return_value=_sufficient_finding()),
			patch.object(planner._planner, "decompose", new_callable=AsyncMock, return_value=(mock_plan, units)),
			patch.object(planner._critic, "gather_context_async", new_callable=AsyncMock, return_value="ctx"),
		):
			_, _, epoch1 = await planner.get_next_units(_mission())
			_, _, epoch2 = await planner.get_next_units(_mission())

		assert epoch1.number == 1
		assert epoch2.number == 2


class TestBackwardCompat:
	def test_set_strategy(self, tmp_path: Path) -> None:
		db = Database(":memory:")
		planner = DeliberativePlanner(_config(tmp_path), db)
		planner.set_strategy("test strategy")
		assert planner._current_strategy == "test strategy"

	def test_set_causal_context(self, tmp_path: Path) -> None:
		db = Database(":memory:")
		planner = DeliberativePlanner(_config(tmp_path), db)
		planner.set_causal_context("risk text")

	def test_set_project_snapshot(self, tmp_path: Path) -> None:
		db = Database(":memory:")
		planner = DeliberativePlanner(_config(tmp_path), db)
		planner.set_project_snapshot("snapshot text")
