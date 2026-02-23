"""Recursive planner that decomposes objectives into parallel work units."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from mission_control.config import MissionConfig, claude_subprocess_env
from mission_control.db import Database
from mission_control.json_utils import extract_json_from_text
from mission_control.models import Plan, PlanNode, WorkUnit
from mission_control.overlap import resolve_file_overlaps

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


_PLAN_BLOCK_RE = re.compile(r"<!--\s*PLAN\s*-->(.*?)<!--\s*/PLAN\s*-->", re.DOTALL)


def _parse_planner_output(output: str) -> PlannerResult:
	"""Extract JSON from structured plan blocks or PLAN_RESULT marker, with fallbacks.

	Parse priority:
	1. <!-- PLAN -->...<!-- /PLAN --> block (new structured format)
	2. PLAN_RESULT: marker (legacy preferred format)
	3. Bare JSON extraction (backward compatibility)
	4. Single-leaf fallback
	"""
	data = None

	# 1. Try <!-- PLAN --> block (use first match if multiple)
	plan_match = _PLAN_BLOCK_RE.search(output)
	if plan_match:
		block_content = plan_match.group(1).strip()
		# Strip markdown code fences that LLMs often add
		block_content = re.sub(r"^```(?:json)?\s*\n?", "", block_content)
		block_content = re.sub(r"\n?```\s*$", "", block_content)
		block_content = block_content.strip()
		try:
			data = json.loads(block_content)
		except (json.JSONDecodeError, ValueError):
			# Try stripping trailing commas (common LLM quirk)
			cleaned = re.sub(r",\s*([}\]])", r"\1", block_content)
			try:
				data = json.loads(cleaned)
			except (json.JSONDecodeError, ValueError):
				log.warning(
					"Malformed JSON inside <!-- PLAN --> block, falling through to PLAN_RESULT. "
					"Content (first 300 chars): %s",
					block_content[:300],
				)
				data = None

	# 2. Try PLAN_RESULT: marker (matches session.py MC_RESULT pattern)
	if not isinstance(data, dict):
		marker = "PLAN_RESULT:"
		idx = output.rfind(marker)
		if idx != -1:
			remainder = output[idx + len(marker):]
			data = extract_json_from_text(remainder)

	# 3. Fallback: try parsing the whole output (backward compatibility)
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



class RecursivePlanner:
	"""Recursively decomposes objectives into a tree of plan nodes and leaf work units."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db
		self._causal_risks: str = ""
		self._project_snapshot: str = ""
		self._feedback_context: str = ""
		self._locked_files: dict[str, list[str]] = {}
		max_concurrent = getattr(config.planner, "max_concurrent_expansions", 4)
		self._semaphore = asyncio.Semaphore(max_concurrent)

	def set_causal_context(self, risks: str) -> None:
		"""Set causal risk factors to include in the planner prompt."""
		self._causal_risks = risks

	def set_project_snapshot(self, snapshot: str) -> None:
		"""Set project structure snapshot to include in the planner prompt."""
		self._project_snapshot = snapshot

	async def plan_round(
		self,
		objective: str,
		snapshot_hash: str,
		prior_discoveries: list[str],
		round_number: int,
		feedback_context: str = "",
		locked_files: dict[str, list[str]] | None = None,
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
		self._locked_files = locked_files or {}

		await self.expand_node(root, plan, objective, snapshot_hash, prior_discoveries)

		# Count leaf work units
		leaf_count = sum(1 for _ in self._iter_leaves(root, {}))
		plan.total_units = leaf_count

		return plan, root

	async def _bounded_expand(
		self,
		node: PlanNode,
		plan: Plan,
		objective: str,
		snapshot_hash: str,
		prior_discoveries: list[str],
	) -> None:
		"""Expand a node while respecting the concurrency semaphore."""
		async with self._semaphore:
			await self.expand_node(node, plan, objective, snapshot_hash, prior_discoveries)

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
					parent_ids=node.id,
					depth=node.depth + 1,
					scope=child_data.get("scope", ""),
					node_type="branch",
				)
				child_ids.append(child.id)
				child_nodes.append(child)

			async with asyncio.TaskGroup() as tg:
				for child in child_nodes:
					tg.create_task(self._bounded_expand(child, plan, objective, snapshot_hash, prior_discoveries))

			node.children_ids = ",".join(child_ids)
			node._subdivided_children = child_nodes  # type: ignore[attr-defined]

			# Cross-branch overlap resolution: collect all leaf units across
			# children and inject depends_on edges for shared files
			all_leaves: list[WorkUnit] = []
			for child in child_nodes:
				all_leaves.extend(self._collect_work_units(child))
			if len(all_leaves) > 1:
				resolve_file_overlaps(all_leaves)
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
					acceptance_criteria=unit_data.get("acceptance_criteria", ""),
					specialist=unit_data.get("specialist", ""),
					speculation_score=float(unit_data.get("speculation_score", 0.0)),
				)
				leaf = PlanNode(
					plan_id=plan.id,
					parent_id=node.id,
					parent_ids=node.id,
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
		"""Build a simplified prompt that strongly emphasizes the <!-- PLAN --> block format."""
		max_children = self.config.planner.max_children_per_node
		return f"""You are a planner. Your ONLY job is to output valid JSON inside a <!-- PLAN --> block.

Objective: {objective}
Scope: {node.scope}

You MUST output EXACTLY one <!-- PLAN --> block containing valid JSON.

Option A - subdivide into sub-scopes (max {max_children}):
<!-- PLAN -->{{"type":"subdivide","children":[{{"scope":"sub-scope description"}}]}}<!-- /PLAN -->

Option B - leaf tasks:
<!-- PLAN -->{{"type":"leaves","units":[
  {{"title":"task","description":"do X","files_hint":"f.py",
    "priority":1,"acceptance_criteria":"testable condition","specialist":"",
    "speculation_score":0.0}}
]}}<!-- /PLAN -->

Option C - nothing to do:
<!-- PLAN -->{{"type":"leaves","units":[]}}<!-- /PLAN -->

Output ONLY the <!-- PLAN --> block. No explanation. No reasoning. Just the block."""

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

		feedback_text = self._feedback_context
		feedback_section = ""
		if feedback_text and node.depth == 0:
			feedback_section = f"\n## Past Round Performance\n{feedback_text}\n"

		locked_files = self._locked_files
		locked_section = ""
		if locked_files and node.depth == 0:
			lines = []
			for fpath, reasons in sorted(locked_files.items()):
				reason_str = "; ".join(reasons)
				lines.append(f"- {fpath} ({reason_str})")
			locked_section = (
				"\n## Locked Files (DO NOT target these)\n"
				"The following files are currently being worked on or have already been merged.\n"
				"Creating units that target these files will be REJECTED by the dispatcher.\n"
				+ "\n".join(lines) + "\n"
			)

		causal_section = ""
		if node.depth == 0 and self._causal_risks:
			causal_section = f"\n{self._causal_risks}\n"

		snapshot_section = ""
		if node.depth == 0 and self._project_snapshot:
			snapshot_section = f"\n## Project Structure\n{self._project_snapshot}\n"

		prompt = f"""You are a recursive planner decomposing work for parallel execution.

## Objective
{objective}

## Your Scope
{node.scope}

## Depth: {node.depth}/{max_depth}

## Prior Discoveries
{discoveries_text}
{feedback_section}{locked_section}{causal_section}{snapshot_section}
## Heuristics
- SUBDIVIDE when: scope spans multiple unrelated subsystems, >5 files across different directories
- PRODUCE LEAVES when: scope is focused, 1-3 concrete tasks can handle it
- Max {max_children} children per subdivision
- NEVER let sibling leaves touch the same file. Merge them or add depends_on_indices.
- Read MISSION_STATE.md in the project root to see what's already been completed.
- If Past Round Performance lists already-modified files, do NOT target those files again.
- Units targeting locked files will be AUTOMATICALLY DROPPED. This is enforced, not optional.
- If the objective has been fully achieved based on MISSION_STATE.md and past discoveries, return EMPTY units:
  <!-- PLAN -->{{"type":"leaves","units":[]}}<!-- /PLAN -->

## Output Format
Reason in prose first, then emit your plan inside a <!-- PLAN --> block.

For subdivision:
<!-- PLAN -->{{"type": "subdivide", "children": [{{"scope": "description of sub-scope"}}, ...]}}<!-- /PLAN -->

For leaf tasks:
<!-- PLAN -->{{"type":"leaves","units":[
  {{"title":"...","description":"...","files_hint":"...","priority":1,
    "depends_on_indices":[],"acceptance_criteria":"testable done condition",
    "specialist":"test-writer|refactorer|debugger|",
    "speculation_score":0.0}}
]}}<!-- /PLAN -->

Specialist field: optionally assign a specialist role to each unit.
- "test-writer": for units focused on adding or fixing tests
- "refactorer": for units focused on code cleanup or restructuring
- "debugger": for units focused on fixing bugs or failures
- "" (empty): general-purpose worker (default)

speculation_score: 0.0-1.0, how uncertain the right approach is.
- Set >= 0.7 when: multiple valid approaches, vague requirements, risky refactoring
- Set 0.0 for straightforward, well-defined tasks

IMPORTANT: Put all reasoning BEFORE the <!-- PLAN --> block. The block must contain valid JSON only."""

		result = await self._run_planner_subprocess(prompt, node)

		is_retryable = _is_parse_fallback(result) and not getattr(result, "_infra_fallback", False)
		if is_retryable and not getattr(node, "_planner_retried", False):
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
		models = getattr(self.config, "models", None)
		model = getattr(models, "planner_model", None) or self.config.scheduler.model
		timeout = self.config.target.verification.timeout

		# CRITICAL: cwd must be the target project path, not the scheduler's own directory.
		# Without this, the planner LLM sees the scheduler's file tree and generates
		# work units targeting scheduler files instead of the target project.
		# See CLAUDE.md Gotchas section.
		cwd = self.config.target.resolved_path
		assert cwd.is_absolute(), f"Planner cwd must be absolute, got: {cwd}"

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
				cwd=str(cwd),
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
				timeout=timeout,
			)
			output = stdout.decode() if stdout else ""
			stderr_text = stderr.decode() if stderr else ""
			if stderr_text:
				log.debug("Planner stderr (depth %d): %s", node.depth, stderr_text[:500])
			log.info("Planner LLM output (depth %d, %d chars): %s", node.depth, len(output), output[:1000])
		except asyncio.TimeoutError:
			log.error("Planner LLM timed out after %ds at depth %d", timeout, node.depth)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			fallback = {"title": "Execute scope", "description": node.scope, "files_hint": "", "priority": 1}
			result = PlannerResult(type="leaves", units=[fallback])
			result._infra_fallback = True  # type: ignore[attr-defined]
			return result

		if proc.returncode != 0:
			log.warning("Planner LLM failed (rc=%d): %s", proc.returncode, stderr.decode()[:200])
			fallback = {"title": "Execute scope", "description": node.scope, "files_hint": "", "priority": 1}
			result = PlannerResult(type="leaves", units=[fallback])
			result._infra_fallback = True  # type: ignore[attr-defined]
			return result

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

	def _collect_work_units(self, node: PlanNode) -> list[WorkUnit]:
		"""Recursively collect WorkUnit objects from the in-memory tree."""
		units: list[WorkUnit] = []
		if hasattr(node, "_forced_unit"):
			units.append(node._forced_unit)
		if hasattr(node, "_child_leaves"):
			for _, wu in node._child_leaves:
				units.append(wu)
		if hasattr(node, "_subdivided_children"):
			for child in node._subdivided_children:
				units.extend(self._collect_work_units(child))
		return units

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
