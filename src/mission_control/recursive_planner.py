"""Flat planner that decomposes objectives into parallel work units via a single LLM call."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from mission_control.config import MissionConfig, build_claude_cmd, claude_subprocess_env
from mission_control.db import Database
from mission_control.json_utils import extract_json_from_text
from mission_control.models import Plan, WorkUnit
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
	"""Flat planner: decomposes objectives into work units via a single LLM call."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db
		self._causal_risks: str = ""
		self._project_snapshot: str = ""
		self._feedback_context: str = ""
		self._locked_files: dict[str, list[str]] = {}

	def set_causal_context(self, risks: str) -> None:
		"""Set causal risk factors to include in the planner prompt."""
		self._causal_risks = risks

	def set_project_snapshot(self, snapshot: str) -> None:
		"""Set project structure snapshot to include in the planner prompt."""
		self._project_snapshot = snapshot

	async def plan_round(
		self,
		objective: str,
		round_number: int = 1,
		feedback_context: str = "",
		locked_files: dict[str, list[str]] | None = None,
	) -> tuple[Plan, list[WorkUnit]]:
		plan = Plan(objective=objective)
		self._feedback_context = feedback_context
		self._locked_files = locked_files or {}

		result = await self._invoke_planner_llm(objective)

		# Build WorkUnit objects from parsed units
		units: list[WorkUnit] = []
		for unit_data in result.units:
			wu = WorkUnit(
				plan_id=plan.id,
				title=unit_data.get("title", ""),
				description=unit_data.get("description", ""),
				files_hint=unit_data.get("files_hint", ""),
				priority=unit_data.get("priority", 1),
				acceptance_criteria=unit_data.get("acceptance_criteria", ""),
				specialist=unit_data.get("specialist", ""),
				speculation_score=float(unit_data.get("speculation_score", 0.0)),
			)
			units.append(wu)

		# Resolve depends_on_indices to actual WorkUnit IDs
		wu_ids = [wu.id for wu in units]
		for i, unit_data in enumerate(result.units):
			if i >= len(units):
				break
			dep_indices = unit_data.get("depends_on_indices", [])
			if isinstance(dep_indices, list):
				dep_ids: list[str] = []
				for idx in dep_indices:
					if isinstance(idx, int) and 0 <= idx < len(wu_ids) and idx != i:
						dep_ids.append(wu_ids[idx])
				if dep_ids:
					units[i].depends_on = ",".join(dep_ids)

		# Resolve file overlaps across all units
		if len(units) > 1:
			resolve_file_overlaps(units)

		plan.total_units = len(units)

		return plan, units

	def _build_retry_prompt(self, objective: str, scope: str) -> str:
		"""Build a simplified prompt that strongly emphasizes the <!-- PLAN --> block format."""
		return f"""You are a planner. Your ONLY job is to output valid JSON inside a <!-- PLAN --> block.

Objective: {objective}
Scope: {scope}

You MUST output EXACTLY one <!-- PLAN --> block containing valid JSON.

Produce a flat list of concrete tasks:
<!-- PLAN -->{{"type":"leaves","units":[
  {{"title":"task","description":"do X","files_hint":"f.py",
    "priority":1,"acceptance_criteria":"shell command(s) that exit 0 when done","specialist":"",
    "speculation_score":0.0}}
]}}<!-- /PLAN -->

If nothing to do:
<!-- PLAN -->{{"type":"leaves","units":[]}}<!-- /PLAN -->

Output ONLY the <!-- PLAN --> block. No explanation. No reasoning. Just the block."""

	async def _invoke_planner_llm(
		self,
		objective: str,
	) -> PlannerResult:
		feedback_text = self._feedback_context
		feedback_section = ""
		if feedback_text:
			feedback_section = f"\n## Past Round Performance\n{feedback_text}\n"

		locked_files = self._locked_files
		locked_section = ""
		if locked_files:
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
		if self._causal_risks:
			causal_section = f"\n{self._causal_risks}\n"

		snapshot_section = ""
		if self._project_snapshot:
			snapshot_section = f"\n## Project Structure\n{self._project_snapshot}\n"

		target_name = self.config.target.name or "this project"

		prompt = f"""You are a strategic planner for {target_name}. \
Your job is to propose the most impactful work possible.

## Objective
{objective}

## Tools Available
You have WebSearch and WebFetch tools. Use them to:
- Research external tools, frameworks, and libraries that could help
- Look up best practices and state-of-the-art approaches
- Find Claude Code plugins, MCP servers, or agentic tooling
- Check documentation for APIs or libraries relevant to the objective
{feedback_section}{locked_section}{causal_section}{snapshot_section}
## Planning Philosophy
- Think ambitiously. The best plan integrates external capabilities, not just internal fixes.
- If the objective mentions tooling or capability, research what exists before proposing to build from scratch.
- Each unit should be completable by one worker in one session.
- NEVER let sibling tasks touch the same file. Merge them or add depends_on_indices.
- Read MISSION_STATE.md in the project root to see what's already been completed.
- If Past Round Performance lists already-modified files, do NOT target those files again.
- Units targeting locked files will be AUTOMATICALLY DROPPED. This is enforced, not optional.
- If the objective has been fully achieved based on MISSION_STATE.md and past discoveries, return EMPTY units:
  <!-- PLAN -->{{"type":"leaves","units":[]}}<!-- /PLAN -->

## Output Format
Reason in prose first (include your research findings), then emit your plan inside a <!-- PLAN --> block.

<!-- PLAN -->{{"type":"leaves","units":[
  {{"title":"...","description":"...","files_hint":"...","priority":1,
    "depends_on_indices":[],"acceptance_criteria":"shell command(s) that exit 0 when done, separated by &&",
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

		result = await self._run_planner_subprocess(prompt)

		is_retryable = _is_parse_fallback(result) and not getattr(result, "_infra_fallback", False)
		if is_retryable and not getattr(self, "_planner_retried", False):
			self._planner_retried = True
			log.warning("Planner parse fallback, retrying with simplified prompt")
			retry_prompt = self._build_retry_prompt(objective, objective)
			result = await self._run_planner_subprocess(retry_prompt)
			if _is_parse_fallback(result):
				log.warning("Planner retry also returned fallback")
		# Reset retry flag for next invocation
		self._planner_retried = False

		return result

	async def _run_planner_subprocess(self, prompt: str) -> PlannerResult:
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

		log.info("Invoking planner LLM for objective: %s", prompt[:80])

		allowed_tools = self.config.planner.allowed_tools or None
		cmd = build_claude_cmd(self.config, model=model, budget=budget, allowed_tools=allowed_tools)
		try:
			proc = await asyncio.create_subprocess_exec(
				*cmd,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(self.config),
				cwd=str(cwd),
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
				timeout=timeout,
			)
			output = stdout.decode() if stdout else ""
			stderr_text = stderr.decode() if stderr else ""
			if stderr_text:
				log.debug("Planner stderr: %s", stderr_text[:500])
			log.info("Planner LLM output (%d chars): %s", len(output), output[:1000])
		except asyncio.TimeoutError:
			log.error("Planner LLM timed out after %ds", timeout)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			fallback = {"title": "Execute scope", "description": prompt[:500], "files_hint": "", "priority": 1}
			result = PlannerResult(type="leaves", units=[fallback])
			result._infra_fallback = True  # type: ignore[attr-defined]
			return result

		if proc.returncode != 0:
			log.warning("Planner LLM failed (rc=%d): %s", proc.returncode, stderr.decode()[:200])
			fallback = {"title": "Execute scope", "description": prompt[:500], "files_hint": "", "priority": 1}
			result = PlannerResult(type="leaves", units=[fallback])
			result._infra_fallback = True  # type: ignore[attr-defined]
			return result

		return _parse_planner_output(output)
