"""Recursive planner that decomposes objectives into parallel work units."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from mission_control.config import MissionConfig, claude_subprocess_env
from mission_control.db import Database
from mission_control.json_utils import extract_json_from_text
from mission_control.models import Plan, PlanNode, WorkUnit

log = logging.getLogger(__name__)


@dataclass
class PlannerResult:
	type: str = ""  # "subdivide" or "leaves"
	children: list[dict[str, Any]] = field(default_factory=list)
	units: list[dict[str, Any]] = field(default_factory=list)


def _is_parse_fallback(result: PlannerResult) -> bool:
	"""Detect when _parse_planner_output returned the fallback single leaf."""
	return (
		result.type == "leaves"
		and len(result.units) == 1
		and result.units[0].get("title") == "Execute scope"
	)


def _parse_planner_output(output: str) -> PlannerResult:
	"""Extract JSON from PLAN_RESULT marker in planner output, with fallbacks."""
	data = None

	# 1. Try PLAN_RESULT: marker (preferred -- matches session.py MC_RESULT pattern)
	marker = "PLAN_RESULT:"
	idx = output.rfind(marker)
	if idx != -1:
		remainder = output[idx + len(marker):]
		data = extract_json_from_text(remainder)

	# 2. Fallback: try parsing the whole output (backward compatibility)
	if not isinstance(data, dict):
		data = extract_json_from_text(output)

	if not isinstance(data, dict):
		log.warning("Failed to parse planner output, falling back to single leaf")
		fallback = {"title": "Execute scope", "description": output[:500], "files_hint": "", "priority": 1}
		return PlannerResult(type="leaves", units=[fallback])

	result = PlannerResult(type=data.get("type", "leaves"))
	if result.type == "subdivide":
		result.children = data.get("children", [])
	else:
		result.units = data.get("units", [])
	return result


def _is_parse_fallback(result: PlannerResult) -> bool:
	"""Return True if result is the fallback produced by _parse_planner_output on unparseable input."""
	return (
		result.type == "leaves"
		and len(result.units) == 1
		and result.units[0].get("title") == "Execute scope"
	)


class RecursivePlanner:
	"""Recursively decomposes objectives into a tree of plan nodes and leaf work units."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db

	async def plan_round(
		self,
		objective: str,
		snapshot_hash: str,
		prior_discoveries: list[str],
		round_number: int,
		feedback_context: str = "",
	) -> tuple[Plan, PlanNode]:
		plan = Plan(objective=objective)
		root = PlanNode(
			plan_id=plan.id,
			depth=0,
			scope=objective,
			node_type="branch",
		)
		plan.root_node_id = root.id
		self._feedback_context = feedback_context

		await self.expand_node(root, plan, objective, snapshot_hash, prior_discoveries)

		# Count leaf work units
		leaf_count = sum(1 for _ in self._iter_leaves(root, {}))
		plan.total_units = leaf_count

		return plan, root

	async def expand_node(
		self,
		node: PlanNode,
		plan: Plan,
		objective: str,
		snapshot_hash: str,
		prior_discoveries: list[str],
	) -> None:
		if node.depth >= self.config.planner.max_depth:
			self._force_create_leaf(node, plan)
			return

		node.status = "expanding"
		result = await self._invoke_planner_llm(node, objective, snapshot_hash, prior_discoveries)

		if result.type == "subdivide" and result.children:
			node.strategy = "subdivide"
			node.node_type = "branch"
			child_ids: list[str] = []
			child_nodes: list[PlanNode] = []

			for child_data in result.children[: self.config.planner.max_children_per_node]:
				child = PlanNode(
					plan_id=plan.id,
					parent_id=node.id,
					depth=node.depth + 1,
					scope=child_data.get("scope", ""),
					node_type="branch",
				)
				child_ids.append(child.id)
				child_nodes.append(child)
				await self.expand_node(child, plan, objective, snapshot_hash, prior_discoveries)

			node.children_ids = ",".join(child_ids)
			node._subdivided_children = child_nodes  # type: ignore[attr-defined]
		else:
			node.strategy = "leaves"
			node.node_type = "branch"
			leaf_pairs: list[tuple[PlanNode, WorkUnit]] = []
			for unit_data in result.units:
				wu = WorkUnit(
					plan_id=plan.id,
					title=unit_data.get("title", ""),
					description=unit_data.get("description", ""),
					files_hint=unit_data.get("files_hint", ""),
					priority=unit_data.get("priority", 1),
					plan_node_id=node.id,
				)
				leaf = PlanNode(
					plan_id=plan.id,
					parent_id=node.id,
					depth=node.depth + 1,
					scope=wu.title,
					strategy="leaves",
					node_type="leaf",
					work_unit_id=wu.id,
				)
				leaf.status = "expanded"
				leaf_pairs.append((leaf, wu))

			# Resolve depends_on_indices to actual WorkUnit IDs
			wu_ids = [wu.id for _, wu in leaf_pairs]
			for i, unit_data in enumerate(result.units):
				if i >= len(leaf_pairs):
					break
				dep_indices = unit_data.get("depends_on_indices", [])
				if isinstance(dep_indices, list):
					dep_ids: list[str] = []
					for idx in dep_indices:
						if isinstance(idx, int) and 0 <= idx < len(wu_ids) and idx != i:
							dep_ids.append(wu_ids[idx])
					if dep_ids:
						leaf_pairs[i][1].depends_on = ",".join(dep_ids)

			# Store leaf references (collected during persist)
			node._child_leaves = leaf_pairs  # type: ignore[attr-defined]

		node.status = "expanded"

	def _build_retry_prompt(self, node: PlanNode, objective: str) -> str:
		"""Build a simplified prompt that strongly emphasizes the PLAN_RESULT JSON format."""
		max_children = self.config.planner.max_children_per_node
		return f"""You are a planner. Your ONLY job is to output a valid PLAN_RESULT JSON line.

Objective: {objective}
Scope: {node.scope}

You MUST output EXACTLY one line starting with PLAN_RESULT: followed by valid JSON.

Option A - subdivide into sub-scopes (max {max_children}):
PLAN_RESULT:{{"type":"subdivide","children":[{{"scope":"sub-scope description"}}]}}

Option B - leaf tasks:
PLAN_RESULT:{{"type":"leaves","units":[{{"title":"task","description":"do X","files_hint":"f.py","priority":1}}]}}

Option C - nothing to do:
PLAN_RESULT:{{"type":"leaves","units":[]}}

Output ONLY the PLAN_RESULT line. No explanation. No reasoning. Just the JSON."""

	async def _invoke_planner_llm(
		self,
		node: PlanNode,
		objective: str,
		snapshot_hash: str,
		prior_discoveries: list[str],
	) -> PlannerResult:
		discoveries_text = "\n".join(f"- {d}" for d in prior_discoveries) if prior_discoveries else "None"
		max_depth = self.config.planner.max_depth
		max_children = self.config.planner.max_children_per_node

		feedback_text = getattr(self, "_feedback_context", "")
		feedback_section = ""
		if feedback_text and node.depth == 0:
			feedback_section = f"\n## Past Round Performance\n{feedback_text}\n"

		prompt = f"""You are a recursive planner decomposing work for parallel execution.

## Objective
{objective}

## Your Scope
{node.scope}

## Depth: {node.depth}/{max_depth}

## Prior Discoveries
{discoveries_text}
{feedback_section}
## Heuristics
- SUBDIVIDE when: scope spans multiple unrelated subsystems, >5 files across different directories
- PRODUCE LEAVES when: scope is focused, 1-3 concrete tasks can handle it
- Max {max_children} children per subdivision
- NEVER let sibling leaves touch the same file. Merge them or add depends_on_indices.
- Read MISSION_STATE.md in the project root to see what's already been completed.
- If Past Round Performance lists already-modified files, do NOT target those files again.
- If the objective has been fully achieved based on MISSION_STATE.md and past discoveries, return EMPTY units:
  PLAN_RESULT:{{"type":"leaves","units":[]}}

## Output Format
You may explain your reasoning, but you MUST end your response with a PLAN_RESULT line.

For subdivision:
PLAN_RESULT:{{"type": "subdivide", "children": [{{"scope": "description of sub-scope"}}, ...]}}

For leaf tasks:
PLAN_RESULT:{{"type":"leaves","units":[{{"title":"...","description":"...","files_hint":"...","priority":1,"depends_on_indices":[]}}]}}

IMPORTANT: The PLAN_RESULT line must be the LAST line of your output. Put all reasoning BEFORE it."""

		result = await self._run_planner_subprocess(prompt, node)

		if _is_parse_fallback(result) and not getattr(node, "_planner_retried", False):
			node._planner_retried = True  # type: ignore[attr-defined]
			log.warning("Planner parse fallback at depth %d, retrying with simplified prompt", node.depth)
			retry_prompt = self._build_retry_prompt(node, objective)
			result = await self._run_planner_subprocess(retry_prompt, node)
			if _is_parse_fallback(result):
				log.warning("Planner retry also returned fallback at depth %d", node.depth)

		return result

	async def _run_planner_subprocess(self, prompt: str, node: PlanNode) -> PlannerResult:
		"""Run the planner LLM subprocess and parse its output."""
		budget = self.config.planner.budget_per_call_usd
		model = self.config.scheduler.model
		timeout = self.config.target.verification.timeout

		log.info("Invoking planner LLM at depth %d for scope: %s", node.depth, node.scope[:80])

		try:
			proc = await asyncio.create_subprocess_exec(
				"claude", "-p",
				"--output-format", "text",
				"--max-budget-usd", str(budget),
				"--model", model,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(),
				cwd=str(self.config.target.resolved_path),
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
				timeout=timeout,
			)
			output = stdout.decode() if stdout else ""
		except asyncio.TimeoutError:
			log.error("Planner LLM timed out after %ds at depth %d", timeout, node.depth)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			fallback = {"title": "Execute scope", "description": node.scope, "files_hint": "", "priority": 1}
			return PlannerResult(type="leaves", units=[fallback])

		if proc.returncode != 0:
			log.warning("Planner LLM failed (rc=%d): %s", proc.returncode, stderr.decode()[:200])
			fallback = {"title": "Execute scope", "description": node.scope, "files_hint": "", "priority": 1}
			return PlannerResult(type="leaves", units=[fallback])

		result = _parse_planner_output(output)

		# Retry once if the LLM returned unparseable output
		if _is_parse_fallback(result):
			log.info("Parse fallback at depth %d, retrying once", node.depth)
			try:
				proc2 = await asyncio.create_subprocess_exec(
					"claude", "-p",
					"--output-format", "text",
					"--max-budget-usd", str(budget),
					"--model", model,
					stdin=asyncio.subprocess.PIPE,
					stdout=asyncio.subprocess.PIPE,
					stderr=asyncio.subprocess.PIPE,
					env=claude_subprocess_env(),
					cwd=str(self.config.target.resolved_path),
				)
				stdout2, stderr2 = await asyncio.wait_for(
					proc2.communicate(input=prompt.encode()),
					timeout=timeout,
				)
				output2 = stdout2.decode() if stdout2 else ""
			except (asyncio.TimeoutError, OSError):
				log.warning("Retry attempt failed, using original fallback")
				return result

			if proc2.returncode != 0:
				return result

			result = _parse_planner_output(output2)

		return result

	def _force_create_leaf(self, node: PlanNode, plan: Plan) -> None:
		"""Convert a node at max depth into a leaf with a single work unit."""
		node.node_type = "leaf"
		node.strategy = "leaves"
		node.status = "expanded"
		wu = WorkUnit(
			plan_id=plan.id,
			title=node.scope[:120],
			description=node.scope,
			priority=1,
			plan_node_id=node.id,
		)
		node.work_unit_id = wu.id
		node._forced_unit = wu  # type: ignore[attr-defined]

	def _iter_leaves(self, node: PlanNode, _seen: dict[str, Any]) -> list[PlanNode]:
		"""Recursively collect leaf nodes from the in-memory tree."""
		leaves: list[PlanNode] = []
		if node.node_type == "leaf":
			leaves.append(node)
		if hasattr(node, "_child_leaves"):
			for leaf, _ in node._child_leaves:
				leaves.append(leaf)
		if hasattr(node, "_subdivided_children"):
			for child in node._subdivided_children:
				leaves.extend(self._iter_leaves(child, _seen))
		return leaves
