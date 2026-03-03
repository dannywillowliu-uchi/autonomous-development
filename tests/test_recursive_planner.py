"""Tests for recursive_planner.py -- flat LLM-based planner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig, PlannerConfig, SchedulerConfig, TargetConfig
from mission_control.models import Plan, WorkUnit
from mission_control.recursive_planner import (
	PlannerResult,
	RecursivePlanner,
	_is_parse_fallback,
	_parse_planner_output,
	_parse_subprocess_cost,
)


def _config(tmp_path: Path) -> MissionConfig:
	config = MissionConfig(
		planner=PlannerConfig(budget_per_call_usd=0.10),
		scheduler=SchedulerConfig(model="sonnet"),
		target=TargetConfig(path=str(tmp_path), name="test-project"),
	)
	return config


def _planner(tmp_path: Path) -> RecursivePlanner:
	return RecursivePlanner(config=_config(tmp_path), db=MagicMock())


# -- Subprocess cwd tests (documented gotcha) --


class TestSubprocessCwd:
	"""Verify subprocess cwd is set to config.target.resolved_path."""

	@pytest.mark.asyncio
	async def test_subprocess_cwd_is_target_resolved_path(self, tmp_path: Path) -> None:
		"""CRITICAL: cwd must be the target project path, not the scheduler's directory."""
		planner = _planner(tmp_path)

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task", "description": "d", "files_hint": "f.py", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			await planner._run_planner_subprocess("test prompt")

		call_kwargs = mock_exec.call_args[1]
		assert call_kwargs["cwd"] == str(tmp_path)

	@pytest.mark.asyncio
	async def test_subprocess_cwd_uses_expanded_path(self, tmp_path: Path) -> None:
		"""Path with ~ should be expanded via resolved_path."""
		config = MissionConfig(
			planner=PlannerConfig(budget_per_call_usd=0.10),
			scheduler=SchedulerConfig(model="sonnet"),
			target=TargetConfig(path=str(tmp_path), name="test"),
		)
		planner = RecursivePlanner(config=config, db=MagicMock())

		response = json.dumps({"type": "leaves", "units": []})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			await planner._run_planner_subprocess("test prompt")

		call_kwargs = mock_exec.call_args[1]
		cwd = call_kwargs["cwd"]
		assert "~" not in cwd
		assert Path(cwd).is_absolute()

	@pytest.mark.asyncio
	async def test_subprocess_cwd_assertion_on_relative_path(self) -> None:
		"""Relative path in target config triggers AssertionError."""
		config = MissionConfig(
			planner=PlannerConfig(budget_per_call_usd=0.10),
			scheduler=SchedulerConfig(model="sonnet"),
			target=TargetConfig(path="relative/path", name="test"),
		)
		planner = RecursivePlanner(config=config, db=MagicMock())

		with pytest.raises(AssertionError, match="absolute"):
			await planner._run_planner_subprocess("test prompt")


# -- <!-- PLAN --> block regex parsing --


class TestPlanBlockParsing:
	"""Tests for <!-- PLAN --> block extraction and JSON parsing."""

	def test_valid_plan_block_with_json(self) -> None:
		"""Standard <!-- PLAN -->...<!-- /PLAN --> block parses correctly."""
		data = {"type": "leaves", "units": [
			{"title": "Add auth", "description": "JWT auth", "files_hint": "auth.py", "priority": 1},
		]}
		output = f"Some reasoning here.\n\n<!-- PLAN -->{json.dumps(data)}<!-- /PLAN -->\n\nDone."
		result = _parse_planner_output(output)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Add auth"

	def test_plan_block_with_code_fences(self) -> None:
		"""PLAN block containing ```json fences should be stripped."""
		data = {"type": "leaves", "units": [
			{"title": "Fenced task", "description": "ok", "files_hint": "", "priority": 1},
		]}
		output = f"Reasoning.\n\n<!-- PLAN -->\n```json\n{json.dumps(data)}\n```\n<!-- /PLAN -->"
		result = _parse_planner_output(output)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Fenced task"

	def test_plan_block_with_trailing_commas(self) -> None:
		"""PLAN block with trailing commas (common LLM quirk) should parse."""
		raw_json = (
			'{"type":"leaves","units":[{"title":"Comma task",'
			'"description":"ok","files_hint":"","priority":1,},]}'
		)
		output = f"<!-- PLAN -->{raw_json}<!-- /PLAN -->"
		result = _parse_planner_output(output)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Comma task"

	def test_plan_block_with_whitespace(self) -> None:
		"""PLAN block with extra whitespace around tags parses correctly."""
		data = {"type": "leaves", "units": [
			{"title": "Spaced", "description": "ok", "files_hint": "", "priority": 1},
		]}
		output = f"<!--  PLAN  -->\n\n{json.dumps(data)}\n\n<!--  /PLAN  -->"
		result = _parse_planner_output(output)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Spaced"

	def test_plan_block_multiple_units(self) -> None:
		"""PLAN block with multiple units all get parsed."""
		data = {"type": "leaves", "units": [
			{"title": f"Task {i}", "description": f"Desc {i}", "files_hint": f"f{i}.py", "priority": i}
			for i in range(5)
		]}
		output = f"<!-- PLAN -->{json.dumps(data)}<!-- /PLAN -->"
		result = _parse_planner_output(output)
		assert result.type == "leaves"
		assert len(result.units) == 5
		assert result.units[3]["title"] == "Task 3"

	def test_plan_block_takes_priority_over_plan_result(self) -> None:
		"""When both PLAN block and PLAN_RESULT exist, PLAN block wins."""
		plan_data = {"type": "leaves", "units": [
			{"title": "From PLAN block", "description": "ok", "files_hint": "", "priority": 1},
		]}
		result_data = {"type": "leaves", "units": [
			{"title": "From PLAN_RESULT", "description": "ok", "files_hint": "", "priority": 1},
		]}
		output = (
			f"<!-- PLAN -->{json.dumps(plan_data)}<!-- /PLAN -->\n\n"
			f"PLAN_RESULT:{json.dumps(result_data)}"
		)
		result = _parse_planner_output(output)
		assert result.units[0]["title"] == "From PLAN block"


# -- Empty plan handling --


class TestEmptyPlanHandling:
	"""Tests for when planner returns no units (objective already met)."""

	def test_parse_empty_units_list(self) -> None:
		"""PLAN block with empty units list returns empty result."""
		output = '<!-- PLAN -->{"type":"leaves","units":[]}<!-- /PLAN -->'
		result = _parse_planner_output(output)
		assert result.type == "leaves"
		assert result.units == []

	@pytest.mark.asyncio
	async def test_plan_round_empty_units_returns_empty_list(self, tmp_path: Path) -> None:
		"""plan_round with empty units returns Plan with total_units=0 and empty list."""
		planner = _planner(tmp_path)
		empty_result = PlannerResult(type="leaves", units=[])

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=empty_result):
			plan, units, cost = await planner.plan_round("Already completed")

		assert plan.total_units == 0
		assert units == []
		assert isinstance(plan, Plan)

	def test_empty_json_object_defaults_to_empty_units(self) -> None:
		"""PLAN_RESULT:{} returns leaves with empty units."""
		output = "PLAN_RESULT:{}"
		result = _parse_planner_output(output)
		assert result.type == "leaves"
		assert result.units == []


# -- Malformed/missing PLAN block handling --


class TestMalformedPlanHandling:
	"""Tests for graceful degradation on malformed or missing PLAN blocks."""

	def test_malformed_json_in_plan_block_falls_through(self) -> None:
		"""Malformed JSON inside PLAN block falls through to other parsers."""
		output = "<!-- PLAN -->{not valid json at all}<!-- /PLAN -->"
		result = _parse_planner_output(output)
		# Falls through to PLAN_RESULT or bare JSON fallback
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"

	def test_missing_plan_block_with_bare_json(self) -> None:
		"""No PLAN block but valid bare JSON should still parse."""
		data = {"type": "leaves", "units": [
			{"title": "Bare JSON", "description": "ok", "files_hint": "", "priority": 1},
		]}
		output = json.dumps(data)
		result = _parse_planner_output(output)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Bare JSON"

	def test_completely_unparseable_returns_fallback(self) -> None:
		"""Total garbage returns single fallback leaf."""
		result = _parse_planner_output("The quick brown fox jumps over the lazy dog")
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"

	def test_empty_output_returns_fallback(self) -> None:
		"""Empty string returns fallback."""
		result = _parse_planner_output("")
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"

	def test_plan_block_with_incomplete_json(self) -> None:
		"""Truncated JSON inside PLAN block falls through."""
		output = '<!-- PLAN -->{"type":"leaves","units":[{"tit<!-- /PLAN -->'
		result = _parse_planner_output(output)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"

	def test_plan_block_with_array_not_object(self) -> None:
		"""PLAN block containing a JSON array (not object) falls through."""
		output = '<!-- PLAN -->[1, 2, 3]<!-- /PLAN -->'
		result = _parse_planner_output(output)
		# Array is not isinstance(data, dict), so falls through
		assert result.type == "leaves"


# -- Cost tracking --


class TestCostTracking:
	"""Tests for cost parsing and propagation through the planner pipeline."""

	def test_parse_cost_from_explicit_pattern(self) -> None:
		assert _parse_subprocess_cost("Session cost: $1.42\n", 0.10) == 1.42

	def test_parse_cost_from_bare_dollar(self) -> None:
		assert _parse_subprocess_cost("Total: $0.73 done\n", 0.10) == 0.73

	def test_empty_stderr_returns_budget_fallback(self) -> None:
		assert _parse_subprocess_cost("", 0.50) == 0.50

	def test_no_cost_pattern_returns_fallback(self) -> None:
		assert _parse_subprocess_cost("Loading tools...\nReady.\n", 0.25) == 0.25

	@pytest.mark.asyncio
	async def test_run_planner_subprocess_sets_cost_from_stderr(self, tmp_path: Path) -> None:
		"""Cost is parsed from subprocess stderr and set on PlannerResult."""
		planner = _planner(tmp_path)
		response = json.dumps({"type": "leaves", "units": []})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"Session cost: $0.88\n")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		):
			result = await planner._run_planner_subprocess("test")

		assert result.cost_usd == 0.88

	@pytest.mark.asyncio
	async def test_failed_subprocess_uses_budget_as_cost(self, tmp_path: Path) -> None:
		"""When subprocess fails (rc!=0), cost is set to budget_per_call_usd."""
		planner = _planner(tmp_path)
		mock_proc = AsyncMock()
		mock_proc.returncode = 1
		mock_proc.communicate.return_value = (b"", b"Error")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		):
			result = await planner._run_planner_subprocess("test")

		assert result.cost_usd == 0.10  # budget_per_call_usd

	@pytest.mark.asyncio
	async def test_timeout_uses_budget_as_cost(self, tmp_path: Path) -> None:
		"""When subprocess times out, cost is set to budget_per_call_usd."""
		planner = _planner(tmp_path)
		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = asyncio.TimeoutError()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		):
			result = await planner._run_planner_subprocess("test")

		assert result.cost_usd == 0.10

	@pytest.mark.asyncio
	async def test_invoke_planner_llm_accumulates_retry_cost(self, tmp_path: Path) -> None:
		"""When planner retries on parse fallback, costs from both calls accumulate."""
		planner = _planner(tmp_path)
		fallback = PlannerResult(
			type="leaves",
			units=[{"title": "Execute scope", "description": "x", "files_hint": "", "priority": 1}],
			cost_usd=0.05,
		)
		real = PlannerResult(
			type="leaves",
			units=[{"title": "Real task", "description": "ok", "files_hint": "a.py", "priority": 1}],
			cost_usd=0.07,
		)
		with patch.object(
			planner, "_run_planner_subprocess",
			new_callable=AsyncMock,
			side_effect=[fallback, real],
		):
			result = await planner._invoke_planner_llm("objective")

		assert result.cost_usd == pytest.approx(0.12)


# -- plan_round WorkUnit list construction --


class TestPlanRoundWorkUnits:
	"""Tests for plan_round returning correct WorkUnit objects."""

	@pytest.mark.asyncio
	async def test_returns_plan_units_cost_tuple(self, tmp_path: Path) -> None:
		"""plan_round returns (Plan, list[WorkUnit], float)."""
		planner = _planner(tmp_path)
		result = PlannerResult(
			type="leaves",
			units=[{"title": "T", "description": "D", "files_hint": "f.py", "priority": 1}],
			cost_usd=0.20,
		)
		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=result):
			plan, units, cost = await planner.plan_round("Build feature")

		assert isinstance(plan, Plan)
		assert plan.objective == "Build feature"
		assert len(units) == 1
		assert isinstance(units[0], WorkUnit)
		assert cost == 0.20

	@pytest.mark.asyncio
	async def test_work_unit_fields_populated(self, tmp_path: Path) -> None:
		"""All WorkUnit fields from parsed data are correctly set."""
		planner = _planner(tmp_path)
		result = PlannerResult(
			type="leaves",
			units=[{
				"title": "Add tests",
				"description": "Write pytest tests for module",
				"files_hint": "tests/test_mod.py",
				"priority": 3,
				"acceptance_criteria": "pytest -q tests/test_mod.py",
				"specialist": "test-writer",
				"speculation_score": 0.7,
			}],
			cost_usd=0.10,
		)
		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=result):
			_, units, _ = await planner.plan_round("Test objective")

		wu = units[0]
		assert wu.title == "Add tests"
		assert wu.description == "Write pytest tests for module"
		assert wu.files_hint == "tests/test_mod.py"
		assert wu.priority == 3
		assert wu.acceptance_criteria == "pytest -q tests/test_mod.py"
		assert wu.specialist == "test-writer"
		assert wu.speculation_score == 0.7

	@pytest.mark.asyncio
	async def test_depends_on_indices_resolved(self, tmp_path: Path) -> None:
		"""depends_on_indices are resolved to WorkUnit IDs."""
		planner = _planner(tmp_path)
		result = PlannerResult(
			type="leaves",
			units=[
				{"title": "A", "description": "a", "files_hint": "a.py", "priority": 1, "depends_on_indices": []},
				{"title": "B", "description": "b", "files_hint": "b.py", "priority": 2, "depends_on_indices": [0]},
			],
			cost_usd=0.10,
		)
		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=result):
			_, units, _ = await planner.plan_round("Dep test")

		assert units[0].depends_on == ""
		assert units[1].depends_on == units[0].id

	@pytest.mark.asyncio
	async def test_plan_id_consistent_across_units(self, tmp_path: Path) -> None:
		"""All WorkUnits share the same plan_id from the Plan object."""
		planner = _planner(tmp_path)
		result = PlannerResult(
			type="leaves",
			units=[
				{"title": "A", "description": "a", "files_hint": "a.py", "priority": 1},
				{"title": "B", "description": "b", "files_hint": "b.py", "priority": 2},
			],
			cost_usd=0.10,
		)
		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=result):
			plan, units, _ = await planner.plan_round("Multi unit")

		assert all(wu.plan_id == plan.id for wu in units)

	@pytest.mark.asyncio
	async def test_plan_total_units_matches(self, tmp_path: Path) -> None:
		"""plan.total_units matches the number of returned units."""
		planner = _planner(tmp_path)
		result = PlannerResult(
			type="leaves",
			units=[
				{"title": f"T{i}", "description": "d", "files_hint": f"f{i}.py", "priority": i}
				for i in range(4)
			],
			cost_usd=0.10,
		)
		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=result):
			plan, units, _ = await planner.plan_round("Count test")

		assert plan.total_units == 4
		assert len(units) == 4

	@pytest.mark.asyncio
	async def test_missing_optional_fields_default(self, tmp_path: Path) -> None:
		"""WorkUnit fields missing from parsed data get default values."""
		planner = _planner(tmp_path)
		result = PlannerResult(
			type="leaves",
			units=[{"title": "Minimal", "description": "bare"}],
			cost_usd=0.10,
		)
		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=result):
			_, units, _ = await planner.plan_round("Minimal test")

		wu = units[0]
		assert wu.files_hint == ""
		assert wu.priority == 1
		assert wu.acceptance_criteria == ""
		assert wu.specialist == ""
		assert wu.speculation_score == 0.0


# -- _is_parse_fallback helper --


class TestIsParseFallback:
	def test_detects_fallback(self) -> None:
		result = PlannerResult(
			type="leaves",
			units=[{"title": "Execute scope", "description": "x", "files_hint": "", "priority": 1}],
		)
		assert _is_parse_fallback(result) is True

	def test_real_result_not_fallback(self) -> None:
		result = PlannerResult(
			type="leaves",
			units=[{"title": "Add tests", "description": "x", "files_hint": "", "priority": 1}],
		)
		assert _is_parse_fallback(result) is False

	def test_empty_units_not_fallback(self) -> None:
		result = PlannerResult(type="leaves", units=[])
		assert _is_parse_fallback(result) is False

	def test_subdivide_not_fallback(self) -> None:
		result = PlannerResult(type="subdivide", children=[{"scope": "x"}])
		assert _is_parse_fallback(result) is False


# -- Context setters --


class TestContextSetters:
	def test_set_causal_context(self, tmp_path: Path) -> None:
		planner = _planner(tmp_path)
		planner.set_causal_context("risk: high failure rate")
		assert planner._causal_risks == "risk: high failure rate"

	def test_set_project_snapshot(self, tmp_path: Path) -> None:
		planner = _planner(tmp_path)
		planner.set_project_snapshot("src/\ntests/")
		assert planner._project_snapshot == "src/\ntests/"
