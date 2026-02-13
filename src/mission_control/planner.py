"""Planner -- decompose an objective into parallel WorkUnits via Claude."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.json_utils import extract_json_from_text
from mission_control.models import Plan, Snapshot, WorkUnit, _new_id

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a task planner for an autonomous development system.

## Objective
{objective}

## Project Health
- Tests: {test_passed}/{test_total} passing ({test_failed} failed)
- Lint errors: {lint_errors}
- Type errors: {type_errors}

## Discovered Issues
{discovered_issues}

## File Tree
{file_tree}

## Instructions
Decompose the objective into independent work units that can be executed in parallel
by separate Claude Code agents. Each work unit should be self-contained and modify
a small set of files.

Output a JSON array of work units:
```json
[
  {{
    "title": "Short descriptive title",
    "description": "Detailed task description with acceptance criteria",
    "files_hint": "comma,separated,file,paths",
    "verification_hint": "What to verify after this unit",
    "priority": 1,
    "depends_on_indices": []
  }}
]
```

Rules:
- Each unit should touch as few files as possible
- Use depends_on_indices to reference other units by their array index (0-based)
- Priority 1 = most important, higher = less important
- Be specific about which files to modify
- Include verification criteria for each unit
- CRITICAL: No two units should create or modify the same file. If multiple
  tasks need the same file, consolidate into one unit or use depends_on_indices
  so the later unit builds on the earlier one's changes.
"""

async def create_plan(config: MissionConfig, snapshot: Snapshot, db: Database) -> Plan:
	"""Run planner agent and create a Plan with WorkUnits."""
	cwd = str(config.target.resolved_path)

	file_tree = await _get_file_tree(cwd, max_chars=config.planner.max_file_tree_chars)

	# Build discovered issues summary from snapshot
	issues: list[str] = []
	if snapshot.test_failed > 0:
		issues.append(f"- {snapshot.test_failed} failing test(s)")
	if snapshot.lint_errors > 0:
		issues.append(f"- {snapshot.lint_errors} lint error(s)")
	if snapshot.type_errors > 0:
		issues.append(f"- {snapshot.type_errors} type error(s)")
	if snapshot.security_findings > 0:
		issues.append(f"- {snapshot.security_findings} security finding(s)")
	discovered_issues = "\n".join(issues) if issues else "None"

	prompt = PLANNER_PROMPT.format(
		objective=config.target.objective,
		test_passed=snapshot.test_passed,
		test_total=snapshot.test_total,
		test_failed=snapshot.test_failed,
		lint_errors=snapshot.lint_errors,
		type_errors=snapshot.type_errors,
		discovered_issues=discovered_issues,
		file_tree=file_tree,
	)

	cmd = [
		"claude",
		"-p",
		"--output-format", "text",
		"--max-budget-usd", str(config.planner.budget_per_call_usd),
		prompt,
	]

	timeout = config.target.verification.timeout

	try:
		proc = await asyncio.create_subprocess_exec(
			*cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
			cwd=cwd,
		)
		stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
		output = stdout_bytes.decode("utf-8", errors="replace")
	except asyncio.TimeoutError:
		logger.error("Planner subprocess timed out after %ds", timeout)
		try:
			proc.kill()
			await proc.wait()
		except ProcessLookupError:
			pass
		output = ""
	except OSError as exc:
		logger.error("Planner subprocess failed: %s", exc)
		output = ""

	plan = Plan(
		objective=config.target.objective,
		status="active",
		raw_planner_output=output,
	)

	units = _parse_plan_output(output, plan.id)
	plan.total_units = len(units)

	db.insert_plan(plan)
	for unit in units:
		db.insert_work_unit(unit)

	return plan


def _parse_plan_output(output: str, plan_id: str) -> list[WorkUnit]:
	"""Parse planner JSON output into WorkUnit objects.

	Handles:
	- JSON array in output (may be surrounded by markdown fences)
	- depends_on_indices -> depends_on (comma-separated IDs)
	- Graceful fallback on malformed JSON
	"""
	if not output.strip():
		return []

	raw_units = extract_json_from_text(output, expect_array=True)

	if not isinstance(raw_units, list):
		return []

	# Generate IDs upfront so we can resolve dependency indices
	unit_ids = [_new_id() for _ in raw_units]

	units: list[WorkUnit] = []
	for i, raw in enumerate(raw_units):
		if not isinstance(raw, dict):
			continue

		# Resolve depends_on_indices to comma-separated IDs
		dep_indices = raw.get("depends_on_indices", [])
		dep_ids: list[str] = []
		if isinstance(dep_indices, list):
			for idx in dep_indices:
				if isinstance(idx, int) and 0 <= idx < len(unit_ids) and idx != i:
					dep_ids.append(unit_ids[idx])

		units.append(WorkUnit(
			id=unit_ids[i],
			plan_id=plan_id,
			title=str(raw.get("title", "")),
			description=str(raw.get("description", "")),
			files_hint=str(raw.get("files_hint", "")),
			verification_hint=str(raw.get("verification_hint", "")),
			priority=int(raw.get("priority", 1)),
			depends_on=",".join(dep_ids),
		))

	return units


async def _get_file_tree(cwd: str | Path, max_depth: int = 3, max_chars: int = 2000) -> str:
	"""Get a truncated file tree of the project."""
	try:
		proc = await asyncio.create_subprocess_exec(
			"find", ".", "-maxdepth", str(max_depth),
			"-not", "-path", "./.git/*",
			"-not", "-path", "./.git",
			"-not", "-path", "./__pycache__/*",
			"-not", "-path", "*/__pycache__/*",
			"-not", "-path", "./.venv/*",
			"-not", "-path", "./node_modules/*",
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.DEVNULL,
			cwd=str(cwd),
		)
		stdout_bytes, _ = await proc.communicate()
		tree = stdout_bytes.decode("utf-8", errors="replace")
	except OSError:
		return "(file tree unavailable)"

	if len(tree) > max_chars:
		tree = tree[:max_chars] + "\n... (truncated)"
	return tree
