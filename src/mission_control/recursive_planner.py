"""Recursive planner that decomposes objectives into parallel work units."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import Plan, PlanNode, WorkUnit

log = logging.getLogger(__name__)


@dataclass
class PlannerResult:
	type: str = ""  # "subdivide" or "leaves"
	children: list[dict[str, Any]] = field(default_factory=list)
	units: list[dict[str, Any]] = field(default_factory=list)


def _parse_planner_output(output: str) -> PlannerResult:
	"""Extract JSON from planner LLM output, with fallback for malformed responses."""
	# Try to find JSON in markdown fences first
	fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", output, re.DOTALL)
	text = fence_match.group(1).strip() if fence_match else output.strip()

	# Try to find a JSON object in the remaining text
	brace_match = re.search(r"\{.*\}", text, re.DOTALL)
	if brace_match:
		text = brace_match.group(0)

	try:
		data = json.loads(text)
	except (json.JSONDecodeError, ValueError):
		log.warning("Failed to parse planner output, falling back to single leaf")
		fallback = {"title": "Execute scope", "description": output[:500], "files_hint": "", "priority": 1}
		return PlannerResult(type="leaves", units=[fallback])

	result = PlannerResult(type=data.get("type", "leaves"))
	if result.type == "subdivide":
		result.children = data.get("children", [])
	else:
		result.units = data.get("units", [])
	return result


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
	) -> tuple[Plan, PlanNode]:
		plan = Plan(objective=objective)
		root = PlanNode(
			plan_id=plan.id,
			depth=0,
			scope=objective,
			node_type="branch",
		)
		plan.root_node_id = root.id

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

			for child_data in result.children[: self.config.planner.max_children_per_node]:
				child = PlanNode(
					plan_id=plan.id,
					parent_id=node.id,
					depth=node.depth + 1,
					scope=child_data.get("scope", ""),
					node_type="branch",
				)
				child_ids.append(child.id)
				await self.expand_node(child, plan, objective, snapshot_hash, prior_discoveries)

			node.children_ids = ",".join(child_ids)
		else:
			node.strategy = "leaves"
			node.node_type = "branch"
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
				# Store leaf reference (collected during persist)
				if not hasattr(node, "_child_leaves"):
					node._child_leaves = []  # type: ignore[attr-defined]
				node._child_leaves.append((leaf, wu))  # type: ignore[attr-defined]

		node.status = "expanded"

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
		budget = self.config.planner.budget_per_call_usd
		model = self.config.scheduler.model

		prompt = f"""You are a recursive planner decomposing work for parallel execution.

## Objective
{objective}

## Your Scope
{node.scope}

## Depth: {node.depth}/{max_depth}

## Prior Discoveries
{discoveries_text}

## Heuristics
- SUBDIVIDE when: scope spans multiple unrelated subsystems, >5 files across different directories
- PRODUCE LEAVES when: scope is focused, 1-3 concrete tasks can handle it
- Max {max_children} children per subdivision

## Output Format
Return JSON:
For subdivision:
{{"type": "subdivide", "children": [{{"scope": "description of sub-scope"}}, ...]}}

For leaf tasks:
{{"type": "leaves", "units": [{{"title": "...", "description": "...", "files_hint": "...", "priority": 1}}, ...]}}"""

		cmd = f'claude -p --output-format text --max-budget-usd {budget} --model {model} "{prompt}"'
		log.info("Invoking planner LLM at depth %d for scope: %s", node.depth, node.scope[:80])

		proc = await asyncio.create_subprocess_shell(
			cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, stderr = await proc.communicate()
		output = stdout.decode() if stdout else ""

		if proc.returncode != 0:
			log.warning("Planner LLM failed (rc=%d): %s", proc.returncode, stderr.decode()[:200])
			fallback = {"title": "Execute scope", "description": node.scope, "files_hint": "", "priority": 1}
			return PlannerResult(type="leaves", units=[fallback])

		return _parse_planner_output(output)

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
		# Children tracked by expand_node are reached via _child_leaves or recursion
		return leaves
