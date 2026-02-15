"""Tests for the recursive planner module."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig, PlannerConfig, SchedulerConfig
from mission_control.models import Plan, PlanNode
from mission_control.recursive_planner import (
	PlannerResult,
	RecursivePlanner,
	_is_parse_fallback,
	_parse_planner_output,
)


def _config(max_depth: int = 3, max_children: int = 5) -> MissionConfig:
	return MissionConfig(
		planner=PlannerConfig(
			max_depth=max_depth,
			max_children_per_node=max_children,
			budget_per_call_usd=0.10,
		),
		scheduler=SchedulerConfig(model="sonnet"),
	)


def _planner(max_depth: int = 3, max_children: int = 5) -> RecursivePlanner:
	config = _config(max_depth=max_depth, max_children=max_children)
	db = MagicMock()
	return RecursivePlanner(config=config, db=db)


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
		assert result.children[1]["scope"] == "Frontend UI"
		assert result.units == []

	def test_valid_leaves_json(self) -> None:
		raw = json.dumps({
			"type": "leaves",
			"units": [
				{"title": "Add tests", "description": "Write unit tests", "files_hint": "tests/", "priority": 1},
				{"title": "Fix lint", "description": "Run ruff", "files_hint": "src/", "priority": 2},
			],
		})
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 2
		assert result.units[0]["title"] == "Add tests"
		assert result.units[1]["priority"] == 2
		assert result.children == []

	def test_markdown_fenced_json(self) -> None:
		inner = json.dumps({
			"type": "subdivide",
			"children": [{"scope": "Database layer"}],
		})
		raw = f"Here is the plan:\n\n```json\n{inner}\n```\n\nLet me know."
		result = _parse_planner_output(raw)
		assert result.type == "subdivide"
		assert len(result.children) == 1
		assert result.children[0]["scope"] == "Database layer"

	def test_markdown_fenced_without_lang_tag(self) -> None:
		inner = json.dumps({
			"type": "leaves",
			"units": [{"title": "Task A", "description": "do A", "files_hint": "", "priority": 1}],
		})
		raw = f"```\n{inner}\n```"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1

	def test_invalid_json_fallback(self) -> None:
		raw = "This is not valid JSON at all {{{"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"
		assert result.units[0]["priority"] == 1

	def test_invalid_json_preserves_description(self) -> None:
		raw = "Some useful context about the task"
		result = _parse_planner_output(raw)
		assert result.units[0]["description"] == raw

	def test_missing_type_defaults_to_leaves(self) -> None:
		raw = json.dumps({
			"units": [{"title": "Implicit leaves", "description": "x", "files_hint": "", "priority": 1}],
		})
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert len(result.units) == 1

	def test_json_embedded_in_prose(self) -> None:
		"""JSON object surrounded by non-JSON text, no fences."""
		obj = json.dumps({
			"type": "leaves",
			"units": [{"title": "Embedded", "description": "found it", "files_hint": "", "priority": 1}],
		})
		raw = f"I think the best plan is:\n{obj}\nDoes that look good?"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Embedded"

	def test_plan_result_marker(self) -> None:
		"""PLAN_RESULT: marker is the preferred parse method."""
		obj = json.dumps({
			"type": "leaves",
			"units": [{"title": "Via marker", "description": "extracted", "files_hint": "x.py", "priority": 1}],
		})
		raw = f"Let me analyze the scope and decompose it.\n\nPLAN_RESULT:{obj}"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Via marker"

	def test_plan_result_marker_with_subdivide(self) -> None:
		"""PLAN_RESULT: marker works with subdivide type."""
		obj = json.dumps({
			"type": "subdivide",
			"children": [{"scope": "Backend"}, {"scope": "Frontend"}],
		})
		raw = f"This scope spans multiple subsystems.\n\nPLAN_RESULT:{obj}"
		result = _parse_planner_output(raw)
		assert result.type == "subdivide"
		assert len(result.children) == 2
		assert result.children[0]["scope"] == "Backend"

	def test_plan_result_marker_takes_precedence(self) -> None:
		"""When both marker and bare JSON exist, marker wins."""
		bad_json = json.dumps({"type": "subdivide", "children": [{"scope": "Wrong"}]})
		good_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "Right", "description": "correct", "files_hint": "", "priority": 1}],
		})
		raw = f"Analysis: {bad_json}\n\nPLAN_RESULT:{good_json}"
		result = _parse_planner_output(raw)
		assert result.type == "leaves"
		assert result.units[0]["title"] == "Right"


# -- _force_create_leaf tests --


class TestForceCreateLeaf:
	def test_creates_leaf_with_work_unit(self) -> None:
		planner = _planner()
		plan = Plan(objective="Test objective")
		node = PlanNode(plan_id=plan.id, depth=3, scope="Implement feature X", node_type="branch")

		planner._force_create_leaf(node, plan)

		assert node.node_type == "leaf"
		assert node.strategy == "leaves"
		assert node.status == "expanded"
		assert node.work_unit_id is not None
		assert hasattr(node, "_forced_unit")
		wu = node._forced_unit  # type: ignore[attr-defined]
		assert wu.plan_id == plan.id
		assert wu.title == "Implement feature X"
		assert wu.description == "Implement feature X"
		assert wu.priority == 1
		assert wu.plan_node_id == node.id

	def test_truncates_long_scope_in_title(self) -> None:
		planner = _planner()
		plan = Plan(objective="Test")
		long_scope = "A" * 200
		node = PlanNode(plan_id=plan.id, depth=3, scope=long_scope, node_type="branch")

		planner._force_create_leaf(node, plan)

		wu = node._forced_unit  # type: ignore[attr-defined]
		assert len(wu.title) == 120
		assert wu.description == long_scope


# -- _iter_leaves tests --


class TestIterLeaves:
	def test_single_leaf_node(self) -> None:
		planner = _planner()
		leaf = PlanNode(node_type="leaf", scope="Task A")
		leaves = planner._iter_leaves(leaf, {})
		assert len(leaves) == 1
		assert leaves[0] is leaf

	def test_branch_with_child_leaves(self) -> None:
		planner = _planner()
		parent = PlanNode(node_type="branch", scope="Parent")
		child1 = PlanNode(node_type="leaf", scope="Child 1")
		child2 = PlanNode(node_type="leaf", scope="Child 2")
		wu1 = MagicMock()
		wu2 = MagicMock()
		parent._child_leaves = [(child1, wu1), (child2, wu2)]  # type: ignore[attr-defined]

		leaves = planner._iter_leaves(parent, {})
		assert len(leaves) == 2
		assert child1 in leaves
		assert child2 in leaves

	def test_branch_without_children_returns_empty(self) -> None:
		planner = _planner()
		branch = PlanNode(node_type="branch", scope="Empty branch")
		leaves = planner._iter_leaves(branch, {})
		assert leaves == []

	def test_branch_with_subdivided_children(self) -> None:
		"""Subdivided children are traversed recursively."""
		planner = _planner()
		root = PlanNode(node_type="branch", scope="Root")
		child_a = PlanNode(node_type="branch", scope="Child A")
		child_b = PlanNode(node_type="branch", scope="Child B")

		leaf_a = PlanNode(node_type="leaf", scope="Leaf A")
		leaf_b = PlanNode(node_type="leaf", scope="Leaf B")
		wu_a = MagicMock()
		wu_b = MagicMock()

		child_a._child_leaves = [(leaf_a, wu_a)]  # type: ignore[attr-defined]
		child_b._child_leaves = [(leaf_b, wu_b)]  # type: ignore[attr-defined]
		root._subdivided_children = [child_a, child_b]  # type: ignore[attr-defined]

		leaves = planner._iter_leaves(root, {})
		assert len(leaves) == 2
		assert leaf_a in leaves
		assert leaf_b in leaves

	def test_nested_subdivide_recursion(self) -> None:
		"""Deeply nested subdivisions are traversed."""
		planner = _planner()
		root = PlanNode(node_type="branch", scope="Root")
		mid = PlanNode(node_type="branch", scope="Mid")
		bottom = PlanNode(node_type="branch", scope="Bottom")

		leaf = PlanNode(node_type="leaf", scope="Deep leaf")
		wu = MagicMock()

		bottom._child_leaves = [(leaf, wu)]  # type: ignore[attr-defined]
		mid._subdivided_children = [bottom]  # type: ignore[attr-defined]
		root._subdivided_children = [mid]  # type: ignore[attr-defined]

		leaves = planner._iter_leaves(root, {})
		assert len(leaves) == 1
		assert leaves[0] is leaf


# -- plan_round tests --


class TestPlanRound:
	@pytest.mark.asyncio
	async def test_plan_round_with_leaves(self) -> None:
		"""LLM returns leaves directly -- no subdivision."""
		planner = _planner()
		leaf_result = PlannerResult(
			type="leaves",
			units=[
				{"title": "Task A", "description": "Do A", "files_hint": "a.py", "priority": 1},
				{"title": "Task B", "description": "Do B", "files_hint": "b.py", "priority": 2},
			],
		)

		with patch.object(planner, "_invoke_planner_llm", new_callable=AsyncMock, return_value=leaf_result):
			plan, root = await planner.plan_round("Build feature", "abc123", [], 1)

		assert plan.objective == "Build feature"
		assert root.node_type == "branch"
		assert root.strategy == "leaves"
		assert root.status == "expanded"
		assert root.depth == 0
		assert plan.root_node_id == root.id
		# Two leaf units created as _child_leaves
		assert hasattr(root, "_child_leaves")
		assert len(root._child_leaves) == 2  # type: ignore[attr-defined]
		assert plan.total_units == 2

	@pytest.mark.asyncio
	async def test_plan_round_with_subdivide_then_leaves(self) -> None:
		"""LLM subdivides the root, then returns leaves for children."""
		planner = _planner(max_depth=3)

		subdivide_result = PlannerResult(
			type="subdivide",
			children=[
				{"scope": "Backend"},
				{"scope": "Frontend"},
			],
		)
		leaf_result_backend = PlannerResult(
			type="leaves",
			units=[{"title": "API endpoint", "description": "REST", "files_hint": "api.py", "priority": 1}],
		)
		leaf_result_frontend = PlannerResult(
			type="leaves",
			units=[{"title": "UI component", "description": "React", "files_hint": "ui.tsx", "priority": 1}],
		)

		call_count = 0

		async def mock_invoke(node, objective, snapshot_hash, prior_discoveries):
			nonlocal call_count
			call_count += 1
			if node.depth == 0:
				return subdivide_result
			elif node.scope == "Backend":
				return leaf_result_backend
			else:
				return leaf_result_frontend

		with patch.object(planner, "_invoke_planner_llm", side_effect=mock_invoke):
			plan, root = await planner.plan_round("Full stack app", "def456", [], 1)

		assert call_count == 3  # root + 2 children
		assert root.strategy == "subdivide"
		assert root.node_type == "branch"
		assert len(root.children_ids.split(",")) == 2
		# _iter_leaves now recurses through _subdivided_children
		assert plan.total_units == 2
		assert root.status == "expanded"

	@pytest.mark.asyncio
	async def test_plan_round_with_prior_discoveries(self) -> None:
		"""Prior discoveries are passed through to the LLM invocation."""
		planner = _planner()
		discoveries = ["Found legacy API at /v1", "Database uses PostgreSQL"]

		captured_args: list = []

		async def capture_invoke(node, objective, snapshot_hash, prior_discoveries):
			captured_args.append(prior_discoveries)
			return PlannerResult(
				type="leaves",
				units=[{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
			)

		with patch.object(planner, "_invoke_planner_llm", side_effect=capture_invoke):
			await planner.plan_round("Migrate API", "abc", discoveries, 2)

		assert len(captured_args) == 1
		assert captured_args[0] == discoveries


# -- Max depth enforcement --


class TestMaxDepthEnforcement:
	@pytest.mark.asyncio
	async def test_max_depth_forces_leaf(self) -> None:
		"""When a node is at max_depth, expand_node should force a leaf without calling LLM."""
		planner = _planner(max_depth=2)

		plan = Plan(objective="Deep task")
		node = PlanNode(plan_id=plan.id, depth=2, scope="Already at max depth", node_type="branch")

		invoke_mock = AsyncMock()
		with patch.object(planner, "_invoke_planner_llm", invoke_mock):
			await planner.expand_node(node, plan, "Deep task", "hash", [])

		# LLM should NOT have been called
		invoke_mock.assert_not_called()
		# Node should have been forced into a leaf
		assert node.node_type == "leaf"
		assert node.strategy == "leaves"
		assert node.status == "expanded"
		assert node.work_unit_id is not None

	@pytest.mark.asyncio
	async def test_depth_below_max_calls_llm(self) -> None:
		"""When depth < max_depth, LLM should be called normally."""
		planner = _planner(max_depth=3)

		plan = Plan(objective="Normal task")
		node = PlanNode(plan_id=plan.id, depth=1, scope="Not at max", node_type="branch")

		leaf_result = PlannerResult(
			type="leaves",
			units=[{"title": "Work", "description": "Do work", "files_hint": "", "priority": 1}],
		)

		invoke_mock = AsyncMock(return_value=leaf_result)
		with patch.object(planner, "_invoke_planner_llm", invoke_mock):
			await planner.expand_node(node, plan, "Normal task", "hash", [])

		invoke_mock.assert_called_once()
		assert node.status == "expanded"

	@pytest.mark.asyncio
	async def test_subdivide_respects_max_children(self) -> None:
		"""Subdivision should cap children at max_children_per_node."""
		planner = _planner(max_depth=3, max_children=2)

		plan = Plan(objective="Many children")
		node = PlanNode(plan_id=plan.id, depth=0, scope="Root", node_type="branch")

		subdivide_result = PlannerResult(
			type="subdivide",
			children=[
				{"scope": "A"},
				{"scope": "B"},
				{"scope": "C"},  # should be dropped (max_children=2)
				{"scope": "D"},  # should be dropped
			],
		)
		leaf_result = PlannerResult(
			type="leaves",
			units=[{"title": "Task", "description": "x", "files_hint": "", "priority": 1}],
		)

		async def mock_invoke(n, obj, snap, disc):
			if n.depth == 0:
				return subdivide_result
			return leaf_result

		with patch.object(planner, "_invoke_planner_llm", side_effect=mock_invoke):
			await planner.expand_node(node, plan, "Many children", "hash", [])

		# Only 2 children should have been created
		assert len(node.children_ids.split(",")) == 2


# -- _invoke_planner_llm tests --


class TestInvokePlannerLlm:
	@pytest.mark.asyncio
	async def test_llm_failure_returns_fallback(self) -> None:
		"""When subprocess fails, return a single fallback leaf."""
		planner = _planner()
		node = PlanNode(depth=0, scope="Test scope", node_type="branch")

		mock_proc = AsyncMock()
		mock_proc.returncode = 1
		mock_proc.communicate.return_value = (b"", b"Error occurred")

		with patch("mission_control.recursive_planner.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await planner._invoke_planner_llm(node, "obj", "hash", [])

		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"
		assert result.units[0]["description"] == "Test scope"

	@pytest.mark.asyncio
	async def test_llm_success_parses_output(self) -> None:
		"""When subprocess succeeds, output is parsed via _parse_planner_output."""
		planner = _planner()
		node = PlanNode(depth=0, scope="Test scope", node_type="branch")

		response = json.dumps({
			"type": "leaves",
			"units": [{"title": "Parsed task", "description": "From LLM", "files_hint": "x.py", "priority": 1}],
		})
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch("mission_control.recursive_planner.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await planner._invoke_planner_llm(node, "obj", "hash", ["discovery 1"])

		assert result.type == "leaves"
		assert result.units[0]["title"] == "Parsed task"

	@pytest.mark.asyncio
	async def test_llm_timeout_returns_fallback_and_kills_process(self) -> None:
		"""When subprocess times out, kill the process and return a fallback leaf.

		Timeout/infra fallbacks are NOT retried (only parse fallbacks are).
		"""
		planner = _planner()
		planner.config.target.verification.timeout = 10
		node = PlanNode(depth=1, scope="Slow scope", node_type="branch")

		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = asyncio.TimeoutError()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch("mission_control.recursive_planner.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await planner._invoke_planner_llm(node, "obj", "hash", [])

		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"
		assert result.units[0]["description"] == "Slow scope"
		assert mock_proc.kill.call_count == 1
		assert mock_proc.wait.await_count == 1

	@pytest.mark.asyncio
	async def test_llm_uses_stdin_not_shell_interpolation(self) -> None:
		"""Prompt with shell metacharacters is passed via stdin, not shell command."""
		planner = _planner()
		node = PlanNode(depth=0, scope='$(echo INJECTED) && `rm -rf /`', node_type="branch")

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
			result = await planner._invoke_planner_llm(node, "obj with $() backticks", "hash", [])

		# Verify create_subprocess_exec was called (not shell)
		mock_exec.assert_called_once()
		call_args = mock_exec.call_args
		# First positional args should be the command parts
		assert call_args[0][0] == "claude"
		assert call_args[0][1] == "-p"
		# Prompt should be passed via stdin, not as a command argument
		assert call_args[1].get("stdin") is not None
		# communicate should have been called with input= containing the prompt
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
		node = PlanNode(depth=0, scope="Test scope", node_type="branch")

		valid_json = json.dumps({
			"type": "leaves",
			"units": [{"title": "Real task", "description": "Valid", "files_hint": "x.py", "priority": 1}],
		})

		# First call: unparseable output (returncode=0 but garbage text)
		mock_proc1 = AsyncMock()
		mock_proc1.returncode = 0
		mock_proc1.communicate.return_value = (b"This is not valid JSON at all {{{", b"")

		# Second call: valid PLAN_RESULT
		mock_proc2 = AsyncMock()
		mock_proc2.returncode = 0
		mock_proc2.communicate.return_value = (valid_json.encode(), b"")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			side_effect=[mock_proc1, mock_proc2],
		) as mock_exec:
			result = await planner._invoke_planner_llm(node, "obj", "hash", [])

		assert result.type == "leaves"
		assert result.units[0]["title"] == "Real task"
		assert mock_exec.call_count == 2

	@pytest.mark.asyncio
	async def test_retry_on_parse_fallback_also_fails(self) -> None:
		"""When both calls return unparseable output, return fallback after two attempts."""
		planner = _planner()
		node = PlanNode(depth=0, scope="Test scope", node_type="branch")

		# Both calls: unparseable output
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
			result = await planner._invoke_planner_llm(node, "obj", "hash", [])

		assert result.type == "leaves"
		assert len(result.units) == 1
		assert result.units[0]["title"] == "Execute scope"
		assert mock_exec.call_count == 2

	@pytest.mark.asyncio
	async def test_no_retry_on_subprocess_failure(self) -> None:
		"""When subprocess fails (returncode != 0), no retry -- return fallback immediately."""
		planner = _planner()
		node = PlanNode(depth=0, scope="Test scope", node_type="branch")

		mock_proc = AsyncMock()
		mock_proc.returncode = 1
		mock_proc.communicate.return_value = (b"", b"Error occurred")

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			result = await planner._invoke_planner_llm(node, "obj", "hash", [])

		assert result.type == "leaves"
		assert result.units[0]["title"] == "Execute scope"
		assert mock_exec.call_count == 1

	@pytest.mark.asyncio
	async def test_no_retry_on_timeout(self) -> None:
		"""When subprocess times out, no retry -- return fallback immediately."""
		planner = _planner()
		planner.config.target.verification.timeout = 10
		node = PlanNode(depth=0, scope="Slow scope", node_type="branch")

		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = asyncio.TimeoutError()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch(
			"mission_control.recursive_planner.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		) as mock_exec:
			result = await planner._invoke_planner_llm(node, "obj", "hash", [])

		assert result.type == "leaves"
		assert result.units[0]["title"] == "Execute scope"
		assert mock_exec.call_count == 1

	def test_is_parse_fallback_detection(self) -> None:
		"""Unit test for _is_parse_fallback helper."""
		# Positive: matches the fallback pattern from _parse_planner_output
		fallback = PlannerResult(
			type="leaves",
			units=[{"title": "Execute scope", "description": "some text", "files_hint": "", "priority": 1}],
		)
		assert _is_parse_fallback(fallback) is True

		# Negative: valid leaves result with a real title
		valid = PlannerResult(
			type="leaves",
			units=[{"title": "Add tests", "description": "Write tests", "files_hint": "tests/", "priority": 1}],
		)
		assert _is_parse_fallback(valid) is False

		# Negative: subdivide result
		subdivide = PlannerResult(
			type="subdivide",
			children=[{"scope": "Backend"}],
		)
		assert _is_parse_fallback(subdivide) is False

		# Negative: multiple units (even if first is "Execute scope")
		multi = PlannerResult(
			type="leaves",
			units=[
				{"title": "Execute scope", "description": "x", "files_hint": "", "priority": 1},
				{"title": "Another", "description": "y", "files_hint": "", "priority": 2},
			],
		)
		assert _is_parse_fallback(multi) is False

		# Negative: empty units
		empty = PlannerResult(type="leaves", units=[])
		assert _is_parse_fallback(empty) is False
