"""Tests for the flat planner and DAG planner modules."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig, PlannerConfig, SchedulerConfig
from mission_control.models import WorkUnit
from mission_control.overlap import resolve_file_overlaps, topological_layers
from mission_control.recursive_planner import (
	PlannerResult,
	RecursivePlanner,
	_is_parse_fallback,
	_parse_planner_output,
)


def _config() -> MissionConfig:
	config = MissionConfig(
		planner=PlannerConfig(
			budget_per_call_usd=0.10,
		),
		scheduler=SchedulerConfig(model="sonnet"),
	)
	config.target.path = "/tmp/test-target-project"
	return config


def _planner() -> RecursivePlanner:
	config = _config()
	db = MagicMock()
	return RecursivePlanner(config=config, db=db)


def _wu(title: str = "", files_hint: str = "", priority: int = 1, depends_on: str = "") -> WorkUnit:
	return WorkUnit(
		plan_id="plan-1",
		title=title,
		files_hint=files_hint,
		priority=priority,
		depends_on=depends_on,
	)


# -- _parse_planner_output tests --


class TestParsePlannerOutput:
	def test_valid_subdivide_json(self) -> None:
		raw = json.dumps({
			"type": "subdivide",
			"children": [
				{"scope": "Backend API"},
				{"scope": "Frontend UI"},
			],
		})
		result = _parse_planner_output(raw)
		assert result.type == "subdivide"
		assert len(result.children) == 2
		assert result.children[0]["scope"] == "Backend API"

	def test_valid_leaves_json(self) -> None:
		raw = json.dumps({
			"type": "leaves",
			"units": [
				{"title": "Add auth", "description": "Implement JWT", "files_hint": "auth.py", "priority": 1},
			],
		})
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Add auth"

	def test_invalid_json_returns_fallback(self) -> None:
		result = _parse_planner_output("This is not valid JSON at all")
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"

	def test_plan_result_marker(self) -> None:
		data = {
			"type": "leaves",
			"units": [{"title": "Marker task", "description": "ok", "files_hint": "", "priority": 1}],
		}
		raw = f"Some reasoning.\n\nPLAN_RESULT:{json.dumps(data)}"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Marker task"

	def test_empty_string_returns_fallback(self) -> None:
		result = _parse_planner_output("")
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"

	def test_defaults_to_leaves_type(self) -> None:
		"""When type key is missing, defaults to leaves."""
		raw = json.dumps({"units": [{"title": "Test", "description": "x", "files_hint": "", "priority": 1}]})
		result = _parse_planner_output(raw)
		assert result.type == "leaves"


# -- plan_round tests --


class TestPlanRound:
	@pytest.mark.asyncio
	async def test_plan_round_returns_plan_and_units(self) -> None:
		"""plan_round returns (Plan, list[WorkUnit]) directly."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[
				{"title": "Task A", "description": "Do A", "files_hint": "a.py", "priority": 1},
				{"title": "Task B", "description": "Do B", "files_hint": "b.py", "priority": 2},
			],
		)

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=leaf_result):
			plan, units = await planner.plan_round("Build feature")

		assert plan.objective == "Build feature"
		assert plan.total_units == 2
		assert len(units) == 2
		assert units[0].title == "Task A"
		assert units[1].title == "Task B"
		assert units[0].plan_id == plan.id
		assert units[1].plan_id == plan.id

	@pytest.mark.asyncio
	async def test_plan_round_empty_units(self) -> None:
		"""Empty units list means objective is met."""
		planner = _planner()
		empty_result = PlannerResult(type="leaves", units=[])

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=empty_result):
			plan, units = await planner.plan_round("Already done")

		assert plan.total_units == 0
		assert units == []

	@pytest.mark.asyncio
	async def test_plan_round_resolves_depends_on_indices(self) -> None:
		"""depends_on_indices are resolved to WorkUnit IDs."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[
				{"title": "A", "description": "a", "files_hint": "a.py", "priority": 1, "depends_on_indices": []},
				{"title": "B", "description": "b", "files_hint": "b.py", "priority": 2, "depends_on_indices": [0]},
				{"title": "C", "description": "c", "files_hint": "c.py", "priority": 3, "depends_on_indices": [0, 1]},
			],
		)

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=leaf_result):
			plan, units = await planner.plan_round("Dep chain")

		assert units[0].depends_on == ""
		assert units[1].depends_on == units[0].id
		assert units[2].depends_on == f"{units[0].id},{units[1].id}"

	@pytest.mark.asyncio
	async def test_plan_round_sets_work_unit_fields(self) -> None:
		"""WorkUnit fields are correctly populated from parsed data."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[{
				"title": "Add tests",
				"description": "Write pytest tests",
				"files_hint": "tests/test_foo.py",
				"priority": 2,
				"acceptance_criteria": "pytest -q tests/test_foo.py",
				"specialist": "test-writer",
				"speculation_score": 0.3,
			}],
		)

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=leaf_result):
			plan, units = await planner.plan_round("Add tests")

		wu = units[0]
		assert wu.title == "Add tests"
		assert wu.description == "Write pytest tests"
		assert wu.files_hint == "tests/test_foo.py"
		assert wu.priority == 2
		assert wu.acceptance_criteria == "pytest -q tests/test_foo.py"
		assert wu.specialist == "test-writer"
		assert wu.speculation_score == 0.3

	@pytest.mark.asyncio
	async def test_plan_round_resolves_file_overlaps(self) -> None:
		"""File overlaps between units are resolved with dependency edges."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[
				{"title": "A", "description": "a", "files_hint": "shared.py,a.py", "priority": 1},
				{"title": "B", "description": "b", "files_hint": "shared.py,b.py", "priority": 2},
			],
		)

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=leaf_result):
			plan, units = await planner.plan_round("Overlap test")

		# B should depend on A due to shared.py overlap
		assert units[0].id in units[1].depends_on


# -- _invoke_planner_llm tests --


class TestInvokePlannerLlm:
	@pytest.mark.asyncio
	async def test_llm_failure_returns_fallback(self) -> None:
		"""When subprocess fails, return a single fallback leaf."""
		planner = _planner()

		mock_proc = AsyncMock()
		mock_proc.returncode = 1
		mock_proc.communicate.return_value = (b"", b"Error occurred")

		with patch("mission_control.recursive_planner.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await planner._invoke_planner_llm("Test objective")

		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"

	@pytest.mark.asyncio
	async def test_llm_success_parses_output(self) -> None:
		"""When subprocess succeeds, output is parsed via _parse_planner_output."""
		planner = _planner()

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Parsed task", "description": "From LLM", "files_hint": "x.py", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch("mission_control.recursive_planner.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await planner._invoke_planner_llm("Test objective")

		assert result.type == "leaves"
		assert result.units[0]["title"] == "Parsed task"

	@pytest.mark.asyncio
	async def test_llm_timeout_returns_fallback_and_kills_process(self) -> None:
		"""When subprocess times out, kill the process and return a fallback leaf."""
		planner = _planner()
		planner.config.target.verification.timeout = 10

		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = asyncio.TimeoutError()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch("mission_control.recursive_planner.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await planner._invoke_planner_llm("Test objective")

		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"
		assert mock_proc.kill.call_count == 1
		assert mock_proc.wait.await_count == 1

	@pytest.mark.asyncio
	async def test_llm_uses_stdin_not_shell_interpolation(self) -> None:
		"""Prompt with shell metacharacters is passed via stdin, not shell command."""
		planner = _planner()

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Safe task", "description": "ok", "files_hint": "", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			result = await planner._invoke_planner_llm("obj with $() backticks")

		mock_exec.assert_called_once()
		call_args = mock_exec.call_args
		assert call_args[0][0] == "claude"
		assert call_args[0][1] == "-p"
		assert call_args[1].get("stdin") is not None
		comm_call = mock_proc.communicate.call_args
		assert comm_call[1].get("input") is not None or (comm_call[0] and comm_call[0][0] is not None)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Safe task"


# -- Planner retry logic tests --


class TestPlannerRetry:
	@pytest.mark.asyncio
	async def test_retry_on_parse_fallback_succeeds(self) -> None:
		"""When first call returns unparseable output, retry once and return valid result."""
		planner = _planner()

		valid_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "Real task", "description": "Valid", "files_hint": "x.py", "priority": 1}],
		})

		mock_proc1 = AsyncMock()
		mock_proc1.returncode = 0
		mock_proc1.communicate.return_value = (b"This is not valid JSON at all {{{", b"")

		mock_proc2 = AsyncMock()
		mock_proc2.returncode = 0
		mock_proc2.communicate.return_value = (valid_json.encode(), b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			side_effect=[mock_proc1, mock_proc2],
		) as mock_exec:
			result = await planner._invoke_planner_llm("Test objective")

		assert result.type == "leaves"
		assert result.units[0]["title"] == "Real task"
		assert mock_exec.call_count == 2

	@pytest.mark.asyncio
	async def test_retry_on_parse_fallback_also_fails(self) -> None:
		"""When both calls return unparseable output, return fallback after two attempts."""
		planner = _planner()

		mock_proc1 = AsyncMock()
		mock_proc1.returncode = 0
		mock_proc1.communicate.return_value = (b"garbage output 1", b"")

		mock_proc2 = AsyncMock()
		mock_proc2.returncode = 0
		mock_proc2.communicate.return_value = (b"garbage output 2", b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			side_effect=[mock_proc1, mock_proc2],
		) as mock_exec:
			result = await planner._invoke_planner_llm("Test objective")

		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"
		assert mock_exec.call_count == 2

	@pytest.mark.asyncio
	async def test_no_retry_on_subprocess_failure(self) -> None:
		"""When subprocess fails (returncode != 0), no retry -- return fallback immediately."""
		planner = _planner()

		mock_proc = AsyncMock()
		mock_proc.returncode = 1
		mock_proc.communicate.return_value = (b"", b"Error occurred")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			result = await planner._invoke_planner_llm("Test objective")

		assert result.type == "leaves"
		assert result.units[0]["title"] == "Execute scope"
		assert mock_exec.call_count == 1

	@pytest.mark.asyncio
	async def test_no_retry_on_timeout(self) -> None:
		"""When subprocess times out, no retry -- return fallback immediately."""
		planner = _planner()
		planner.config.target.verification.timeout = 10

		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = asyncio.TimeoutError()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			result = await planner._invoke_planner_llm("Test objective")

		assert result.type == "leaves"
		assert result.units[0]["title"] == "Execute scope"
		assert mock_exec.call_count == 1

	def test_is_parse_fallback_detection(self) -> None:
		"""Unit test for _is_parse_fallback helper."""
		fallback = PlannerResult(
			type="leaves",
			units=[{"title": "Execute scope", "description": "some text", "files_hint": "", "priority": 1}],
		)
		assert _is_parse_fallback(fallback) is True

		valid = PlannerResult(
			type="leaves",
			units=[{"title": "Add tests", "description": "Write tests", "files_hint": "tests/", "priority": 1}],
		)
		assert _is_parse_fallback(valid) is False

		subdivide = PlannerResult(
			type="subdivide",
			children=[{"scope": "Backend"}],
		)
		assert _is_parse_fallback(subdivide) is False

		multi = PlannerResult(
			type="leaves",
			units=[
				{"title": "Execute scope", "description": "x", "files_hint": "", "priority": 1},
				{"title": "Another", "description": "y", "files_hint": "", "priority": 2},
			],
		)
		assert _is_parse_fallback(multi) is False

		empty = PlannerResult(type="leaves", units=[])
		assert _is_parse_fallback(empty) is False


# -- _parse_planner_output edge cases --


class TestParsePlannerOutputEdgeCases:
	def test_plan_result_marker_truncated_json(self) -> None:
		"""PLAN_RESULT with truncated JSON should fall through to fallback."""
		raw = 'Some analysis here.\n\nPLAN_RESULT:{"type":"leaves","units":[{"tit'
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"

	def test_plan_result_marker_empty_json(self) -> None:
		"""PLAN_RESULT:{} -- valid JSON but no type/units. Should return leaves with empty units."""
		raw = "Reasoning text.\n\nPLAN_RESULT:{}"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units == []

	def test_plan_result_marker_with_trailing_text(self) -> None:
		"""PLAN_RESULT:<valid json> followed by trailing prose should still parse."""
		obj = json.dumps({
			"type": "leaves",
			"units": [{"title": "Trailing test", "description": "ok", "files_hint": "", "priority": 1}],
		})
		raw = f"Let me think.\n\nPLAN_RESULT:{obj}\n\nHope that helps! Let me know if you need changes."
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Trailing test"


# -- depends_on_indices resolution tests --


class TestDependsOnIndicesResolution:
	"""Tests for the depends_on_indices -> WorkUnit.depends_on resolution in plan_round."""

	@pytest.mark.asyncio
	async def test_out_of_range_index(self) -> None:
		"""depends_on_indices=[99] with only 3 units -- out-of-range index silently skipped."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[
				{"title": "A", "description": "a", "files_hint": "a.py", "priority": 1, "depends_on_indices": []},
				{"title": "B", "description": "b", "files_hint": "b.py", "priority": 2, "depends_on_indices": []},
				{"title": "C", "description": "c", "files_hint": "c.py", "priority": 3, "depends_on_indices": [99]},
			],
		)

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=leaf_result):
			plan, units = await planner.plan_round("Test deps")

		assert len(units) == 3
		assert units[2].depends_on == ""

	@pytest.mark.asyncio
	async def test_self_reference_index(self) -> None:
		"""depends_on_indices=[0] on unit index 0 -- self-reference should be skipped."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[
				{"title": "A", "description": "a", "files_hint": "a.py", "priority": 1, "depends_on_indices": [0]},
				{"title": "B", "description": "b", "files_hint": "b.py", "priority": 2, "depends_on_indices": []},
			],
		)

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=leaf_result):
			plan, units = await planner.plan_round("Test self-ref")

		assert units[0].depends_on == ""

	@pytest.mark.asyncio
	async def test_non_integer_values(self) -> None:
		"""depends_on_indices=["foo", None, 1.5] -- non-int values should be skipped."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[
				{"title": "A", "description": "a", "files_hint": "a.py", "priority": 1, "depends_on_indices": []},
				{
				"title": "B", "description": "b", "files_hint": "b.py", "priority": 2,
				"depends_on_indices": ["foo", None, 1.5],
			},
			],
		)

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=leaf_result):
			plan, units = await planner.plan_round("Test non-int")

		assert units[1].depends_on == ""

	@pytest.mark.asyncio
	async def test_empty_depends_on_indices(self) -> None:
		"""depends_on_indices=[] -- no depends_on should be set."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[
				{"title": "A", "description": "a", "files_hint": "a.py", "priority": 1, "depends_on_indices": []},
				{"title": "B", "description": "b", "files_hint": "b.py", "priority": 2, "depends_on_indices": []},
			],
		)

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=leaf_result):
			plan, units = await planner.plan_round("Test empty deps")

		for wu in units:
			assert wu.depends_on == ""

	@pytest.mark.asyncio
	async def test_valid_dependency_chain(self) -> None:
		"""3 units: unit[1] depends on unit[0], unit[2] depends on unit[0] and unit[1]."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[
				{"title": "A", "description": "a", "files_hint": "a.py", "priority": 1, "depends_on_indices": []},
				{"title": "B", "description": "b", "files_hint": "b.py", "priority": 2, "depends_on_indices": [0]},
				{"title": "C", "description": "c", "files_hint": "c.py", "priority": 3, "depends_on_indices": [0, 1]},
			],
		)

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=leaf_result):
			plan, units = await planner.plan_round("Test dep chain")

		assert units[0].depends_on == ""
		assert units[1].depends_on == units[0].id
		assert units[2].depends_on == f"{units[0].id},{units[1].id}"

	@pytest.mark.asyncio
	async def test_mixed_valid_and_invalid_indices(self) -> None:
		"""depends_on_indices=[0, 99, -1, 1] -- only valid in-range, non-self indices kept."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[
				{"title": "A", "description": "a", "files_hint": "a.py", "priority": 1, "depends_on_indices": []},
				{"title": "B", "description": "b", "files_hint": "b.py", "priority": 2, "depends_on_indices": []},
				{
				"title": "C", "description": "c", "files_hint": "c.py", "priority": 3,
				"depends_on_indices": [0, 99, -1, 1],
			},
			],
		)

		with patch.object(planner, "_run_planner_subprocess", new_callable=AsyncMock, return_value=leaf_result):
			plan, units = await planner.plan_round("Test mixed deps")

		# Only indices 0 and 1 are valid (99 out of range, -1 negative)
		assert units[2].depends_on == f"{units[0].id},{units[1].id}"


# -- Subprocess cwd assertion tests --


class TestSubprocessCwdAssertion:
	"""Verify that _run_planner_subprocess always uses config.target.resolved_path as cwd."""

	@pytest.mark.asyncio
	async def test_subprocess_cwd_matches_target_resolved_path(self) -> None:
		"""The cwd passed to create_subprocess_exec must equal config.target.resolved_path."""
		planner = _planner()

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
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
		expected_cwd = str(planner.config.target.resolved_path)
		assert call_kwargs["cwd"] == expected_cwd

	@pytest.mark.asyncio
	async def test_subprocess_cwd_rejects_relative_path(self) -> None:
		"""If config.target.resolved_path is somehow relative, the assertion fires."""
		planner = _planner()

		planner.config.target.path = "relative/path"
		assert not planner.config.target.resolved_path.is_absolute()

		with pytest.raises(AssertionError, match="Planner cwd must be absolute"):
			await planner._run_planner_subprocess("test prompt")


# -- <!-- PLAN --> block parsing tests --


class TestPlanBlockParsing:
	"""Tests for the <!-- PLAN --> structured block parsing in _parse_planner_output."""

	def test_valid_plan_block_extraction(self) -> None:
		"""Valid JSON inside <!-- PLAN --> block is extracted correctly."""
		plan_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "Block task", "description": "via block", "files_hint": "x.py", "priority": 1}],
		})
		raw = f"Let me analyze the scope.\n\n<!-- PLAN -->{plan_json}<!-- /PLAN -->"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Block task"

	def test_plan_block_with_subdivide(self) -> None:
		"""<!-- PLAN --> block works with subdivide type."""
		plan_json = json.dumps({
			"type": "subdivide",
			"children": [{"scope": "Backend API"}, {"scope": "Frontend UI"}],
		})
		raw = f"This needs subdivision.\n\n<!-- PLAN -->{plan_json}<!-- /PLAN -->"
		result = _parse_planner_output(raw)
		assert result.type == "subdivide"
		assert len(result.children) == 2
		assert result.children[0]["scope"] == "Backend API"

	def test_missing_plan_block_falls_back_to_plan_result(self) -> None:
		"""When no <!-- PLAN --> block, falls back to PLAN_RESULT marker."""
		plan_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "Legacy marker", "description": "ok", "files_hint": "", "priority": 1}],
		})
		raw = f"Some reasoning.\n\nPLAN_RESULT:{plan_json}"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Legacy marker"

	def test_malformed_json_inside_plan_block(self) -> None:
		"""Malformed JSON in <!-- PLAN --> block falls through to PLAN_RESULT."""
		plan_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "Fallback task", "description": "ok", "files_hint": "", "priority": 1}],
		})
		raw = f'<!-- PLAN -->{{"type":"leaves","units":[{{"tit<!-- /PLAN -->\n\nPLAN_RESULT:{plan_json}'
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Fallback task"

	def test_malformed_json_inside_plan_block_no_fallback(self) -> None:
		"""Malformed JSON in <!-- PLAN --> block with no other parseable content gives fallback leaf."""
		raw = "<!-- PLAN -->not valid json at all<!-- /PLAN -->\n\nSome prose but no JSON."
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"

	def test_multiple_plan_blocks_uses_first(self) -> None:
		"""When multiple <!-- PLAN --> blocks exist, the first one is used."""
		first_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "First block", "description": "correct", "files_hint": "", "priority": 1}],
		})
		second_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "Second block", "description": "wrong", "files_hint": "", "priority": 2}],
		})
		raw = f"<!-- PLAN -->{first_json}<!-- /PLAN -->\n\nRevised:\n<!-- PLAN -->{second_json}<!-- /PLAN -->"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "First block"

	def test_plan_block_with_surrounding_prose(self) -> None:
		"""<!-- PLAN --> block surrounded by analysis prose extracts correctly."""
		plan_json = json.dumps({
			"type": "leaves",
			"units": [
				{"title": "Add auth", "description": "JWT auth", "files_hint": "auth.py", "priority": 1},
				{"title": "Add tests", "description": "test auth", "files_hint": "tests/", "priority": 2},
			],
		})
		raw = (
			"I've analyzed the codebase and here's my assessment:\n\n"
			"The scope involves authentication and testing.\n"
			"The auth module needs JWT support and tests need to be added.\n\n"
			f"<!-- PLAN -->{plan_json}<!-- /PLAN -->\n\n"
			"This plan covers the key areas identified in the analysis."
		)
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 2
		assert result.units[0]["title"] == "Add auth"
		assert result.units[1]["title"] == "Add tests"

	def test_plan_block_takes_precedence_over_plan_result(self) -> None:
		"""<!-- PLAN --> block takes priority over PLAN_RESULT marker."""
		block_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "From block", "description": "correct", "files_hint": "", "priority": 1}],
		})
		marker_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "From marker", "description": "wrong", "files_hint": "", "priority": 1}],
		})
		raw = f"<!-- PLAN -->{block_json}<!-- /PLAN -->\n\nPLAN_RESULT:{marker_json}"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "From block"

	def test_plan_block_takes_precedence_over_bare_json(self) -> None:
		"""<!-- PLAN --> block takes priority over bare JSON in output."""
		block_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "From block", "description": "correct", "files_hint": "", "priority": 1}],
		})
		bare_json = json.dumps({
			"type": "subdivide",
			"children": [{"scope": "Wrong"}],
		})
		raw = f"Analysis: {bare_json}\n\n<!-- PLAN -->{block_json}<!-- /PLAN -->"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "From block"

	def test_plan_block_with_whitespace_in_markers(self) -> None:
		"""<!-- PLAN --> markers with extra whitespace still match."""
		plan_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "Whitespace ok", "description": "ok", "files_hint": "", "priority": 1}],
		})
		raw = f"<!--  PLAN  -->{plan_json}<!--  /PLAN  -->"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Whitespace ok"

	def test_plan_block_with_newlines_inside(self) -> None:
		"""JSON inside <!-- PLAN --> block can span multiple lines."""
		plan_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "Multiline", "description": "ok", "files_hint": "", "priority": 1}],
		}, indent=2)
		raw = f"<!-- PLAN -->\n{plan_json}\n<!-- /PLAN -->"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Multiline"

	def test_existing_plan_result_format_still_works(self) -> None:
		"""Existing PLAN_RESULT format continues to work as fallback."""
		plan_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "Legacy", "description": "still works", "files_hint": "x.py", "priority": 1}],
		})
		raw = f"Analysis here.\n\nPLAN_RESULT:{plan_json}"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Legacy"

	def test_plan_block_empty_units(self) -> None:
		"""<!-- PLAN --> block with empty units (objective met)."""
		plan_json = json.dumps({"type": "leaves", "units": []})
		raw = f"Objective already achieved.\n\n<!-- PLAN -->{plan_json}<!-- /PLAN -->"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units == []

	def test_plan_block_with_code_fences(self) -> None:
		"""<!-- PLAN --> block with markdown code fences around JSON."""
		raw = (
			"Analysis complete.\n\n"
			"<!-- PLAN -->\n"
			"```json\n"
			'{"type":"leaves","units":[{"title":"Task A","description":"do it","files_hint":"a.py","priority":1}]}\n'
			"```\n"
			"<!-- /PLAN -->"
		)
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Task A"

	def test_plan_block_with_code_fences_no_lang(self) -> None:
		"""<!-- PLAN --> block with bare ``` fences (no language tag)."""
		raw = (
			"<!-- PLAN -->\n"
			"```\n"
			'{"type":"leaves","units":[]}\n'
			"```\n"
			"<!-- /PLAN -->"
		)
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units == []

	def test_plan_block_with_trailing_commas(self) -> None:
		"""<!-- PLAN --> block with trailing commas in JSON."""
		raw = (
			'<!-- PLAN -->{"type":"leaves","units":[\n'
			'  {"title":"A","description":"x","files_hint":"","priority":1,},\n'
			']}<!-- /PLAN -->'
		)
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "A"


# -- set_causal_context / set_project_snapshot tests --


class TestCausalContextAndSnapshot:
	def test_set_causal_context(self) -> None:
		"""set_causal_context stores the value in _causal_risks."""
		planner = _planner()
		assert planner._causal_risks == ""
		planner.set_causal_context("model=opus: 9% failure")
		assert planner._causal_risks == "model=opus: 9% failure"

	def test_set_project_snapshot(self) -> None:
		"""set_project_snapshot stores the value in _project_snapshot."""
		planner = _planner()
		assert planner._project_snapshot == ""
		planner.set_project_snapshot("src/ has 20 files")
		assert planner._project_snapshot == "src/ has 20 files"

	@pytest.mark.asyncio
	async def test_causal_risks_included_in_prompt(self) -> None:
		"""When _causal_risks is set, the prompt includes it."""
		planner = _planner()
		planner.set_causal_context("## Causal Risk\nmodel=opus: 9% failure")

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch("mission_control.recursive_planner.asyncio.create_subprocess_exec", return_value=mock_proc):
			await planner._invoke_planner_llm("obj")

		prompt = mock_proc.communicate.call_args[1]["input"].decode()
		assert "model=opus: 9% failure" in prompt

	@pytest.mark.asyncio
	async def test_project_snapshot_included_in_prompt(self) -> None:
		"""When _project_snapshot is set, the prompt includes it."""
		planner = _planner()
		planner.set_project_snapshot("src/ has 20 files")

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch("mission_control.recursive_planner.asyncio.create_subprocess_exec", return_value=mock_proc):
			await planner._invoke_planner_llm("obj")

		prompt = mock_proc.communicate.call_args[1]["input"].decode()
		assert "src/ has 20 files" in prompt
		assert "## Project Structure" in prompt


	@pytest.mark.asyncio
	async def test_ambitious_prompt_with_web_search(self) -> None:
		"""Planner prompt includes ambitious framing and WebSearch instruction."""
		planner = _planner()
		planner.config.target.name = "my-project"

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch("mission_control.recursive_planner.asyncio.create_subprocess_exec", return_value=mock_proc):
			await planner._invoke_planner_llm("Build auth system")

		prompt = mock_proc.communicate.call_args[1]["input"].decode()
		assert "strategic planner for my-project" in prompt
		assert "most impactful work" in prompt
		assert "WebSearch" in prompt
		assert "WebFetch" in prompt
		assert "Think ambitiously" in prompt

	@pytest.mark.asyncio
	async def test_allowed_tools_passed_to_subprocess(self) -> None:
		"""Planner subprocess command includes --allowedTools flags."""
		planner = _planner()

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			await planner._run_planner_subprocess("test prompt")

		call_args = list(mock_exec.call_args[0])
		assert "--allowedTools" in call_args
		tool_indices = [i for i, a in enumerate(call_args) if a == "--allowedTools"]
		tools_passed = [call_args[i + 1] for i in tool_indices]
		assert "WebSearch" in tools_passed
		assert "WebFetch" in tools_passed


# -- Per-component model usage tests --


class TestPerComponentModelUsage:
	"""Tests for config.models.planner_model usage in _run_planner_subprocess."""

	@pytest.mark.asyncio
	async def test_uses_scheduler_model_when_no_models_config(self) -> None:
		"""Without config.models.planner_model, falls back to scheduler.model."""
		planner = _planner()
		planner.config.models.planner_model = ""

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			await planner._run_planner_subprocess("test prompt")

		call_args = mock_exec.call_args[0]
		model_idx = list(call_args).index("--model")
		assert call_args[model_idx + 1] == "sonnet"

	@pytest.mark.asyncio
	async def test_uses_planner_model_from_models_config(self) -> None:
		"""When config.models.planner_model is set, it overrides scheduler.model."""
		planner = _planner()

		class ModelsConfig:
			planner_model = "haiku"
		planner.config.models = ModelsConfig()  # type: ignore[attr-defined]

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			await planner._run_planner_subprocess("test prompt")

		call_args = mock_exec.call_args[0]
		model_idx = list(call_args).index("--model")
		assert call_args[model_idx + 1] == "haiku"

	@pytest.mark.asyncio
	async def test_falls_back_when_planner_model_is_none(self) -> None:
		"""When config.models exists but planner_model is None, falls back to scheduler.model."""
		planner = _planner()

		class ModelsConfig:
			planner_model = None
		planner.config.models = ModelsConfig()  # type: ignore[attr-defined]

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			await planner._run_planner_subprocess("test prompt")

		call_args = mock_exec.call_args[0]
		model_idx = list(call_args).index("--model")
		assert call_args[model_idx + 1] == "sonnet"

	@pytest.mark.asyncio
	async def test_falls_back_when_planner_model_is_empty(self) -> None:
		"""When config.models.planner_model is empty string, falls back to scheduler.model."""
		planner = _planner()

		class ModelsConfig:
			planner_model = ""
		planner.config.models = ModelsConfig()  # type: ignore[attr-defined]

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			await planner._run_planner_subprocess("test prompt")

		call_args = mock_exec.call_args[0]
		model_idx = list(call_args).index("--model")
		assert call_args[model_idx + 1] == "sonnet"


# -- DAG planner: topological layers tests --


class TestTopologicalLayers:
	def test_empty_input(self) -> None:
		assert topological_layers([]) == []

	def test_no_dependencies_single_layer(self) -> None:
		"""All units with no deps form a single parallel layer."""
		a = _wu("A", "a.py")
		b = _wu("B", "b.py")
		c = _wu("C", "c.py")
		layers = topological_layers([a, b, c])
		assert len(layers) == 1
		assert set(u.id for u in layers[0]) == {a.id, b.id, c.id}

	def test_linear_chain(self) -> None:
		"""A -> B -> C forms 3 layers of 1 unit each."""
		a = _wu("A", "a.py")
		b = _wu("B", "b.py", depends_on=a.id)
		c = _wu("C", "c.py", depends_on=b.id)
		layers = topological_layers([a, b, c])
		assert len(layers) == 3
		assert layers[0][0].id == a.id
		assert layers[1][0].id == b.id
		assert layers[2][0].id == c.id

	def test_diamond_dag(self) -> None:
		"""Diamond: A -> (B, C) -> D forms 3 layers."""
		a = _wu("A", "a.py")
		b = _wu("B", "b.py", depends_on=a.id)
		c = _wu("C", "c.py", depends_on=a.id)
		d = _wu("D", "d.py", depends_on=f"{b.id},{c.id}")
		layers = topological_layers([a, b, c, d])
		assert len(layers) == 3
		assert layers[0][0].id == a.id
		layer1_ids = {u.id for u in layers[1]}
		assert layer1_ids == {b.id, c.id}
		assert layers[2][0].id == d.id

	def test_orphan_deps_ignored(self) -> None:
		"""Dependencies on non-existent IDs are silently ignored."""
		a = _wu("A", "a.py", depends_on="nonexistent-id")
		b = _wu("B", "b.py")
		layers = topological_layers([a, b])
		assert len(layers) == 1
		assert len(layers[0]) == 2

	def test_single_unit(self) -> None:
		a = _wu("A", "a.py")
		layers = topological_layers([a])
		assert len(layers) == 1
		assert layers[0][0].id == a.id

	def test_mixed_deps_and_independent(self) -> None:
		"""Mix of dependent and independent units."""
		a = _wu("A", "a.py")
		b = _wu("B", "b.py", depends_on=a.id)
		c = _wu("C", "c.py")
		layers = topological_layers([a, b, c])
		assert len(layers) == 2
		layer0_ids = {u.id for u in layers[0]}
		assert a.id in layer0_ids
		assert c.id in layer0_ids
		assert layers[1][0].id == b.id


# -- Overlap with layers integration tests --


class TestOverlapWithLayers:
	def test_resolve_then_layer(self) -> None:
		"""resolve_file_overlaps + topological_layers integration."""
		a = _wu("A", "shared.py,a.py", priority=1)
		b = _wu("B", "shared.py,b.py", priority=2)
		c = _wu("C", "c.py", priority=3)
		resolve_file_overlaps([a, b, c])
		assert a.id in b.depends_on
		layers = topological_layers([a, b, c])
		assert len(layers) == 2
		layer0_ids = {u.id for u in layers[0]}
		assert a.id in layer0_ids
		assert c.id in layer0_ids
		assert layers[1][0].id == b.id

	def test_diamond_from_overlaps(self) -> None:
		"""File overlaps creating a diamond pattern layer correctly."""
		a = _wu("A", "base.py", priority=1)
		b = _wu("B", "base.py,feature.py", priority=2)
		c = _wu("C", "base.py,util.py", priority=2)
		d = _wu("D", "feature.py,util.py", priority=3)
		resolve_file_overlaps([a, b, c, d])
		layers = topological_layers([a, b, c, d])
		assert layers[0][0].id == a.id
		d_layer = None
		for idx, layer in enumerate(layers):
			if any(u.id == d.id for u in layer):
				d_layer = idx
		assert d_layer is not None and d_layer >= 2
