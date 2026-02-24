"""LLM-based diff reviewer -- scores merged work units on quality dimensions."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from mission_control.config import MissionConfig, ReviewConfig, claude_subprocess_env
from mission_control.json_utils import extract_json_from_text
from mission_control.models import UnitReview, WorkUnit

logger = logging.getLogger(__name__)

REVIEW_RESULT_MARKER = "REVIEW_RESULT:"


def _build_review_prompt(unit: WorkUnit, diff: str, objective: str, project_snapshot: str = "") -> str:
	criteria_section = ""
	criteria_scoring = ""
	criteria_output = ""
	if unit.acceptance_criteria:
		criteria_section = f"\nAcceptance Criteria: {unit.acceptance_criteria}\n"
		criteria_scoring = '\n4. **Criteria Met** (1-10): How well does the diff satisfy the acceptance criteria above?'
		criteria_output = ', "criteria_met": 7'
	snapshot_section = f"\n## Project Structure\n{project_snapshot}\n\n" if project_snapshot else ""
	return f"""You are a code reviewer evaluating a merged work unit's diff.

## Mission Objective
{objective}

## Work Unit
Title: {unit.title}
Description: {unit.description}
{criteria_section}
{snapshot_section}## Git Diff
```
{diff[:8000]}
```

## Instructions

Score each dimension 1-10:
1. **Alignment** (1-10): How well does this diff advance the mission objective?
2. **Approach** (1-10): Is the implementation approach clean, idiomatic, and maintainable?
3. **Test Quality** (1-10): Are the tests meaningful and covering real behavior (not trivial)?{criteria_scoring}

Then provide a 1-2 sentence rationale summarizing the overall quality.

## Output Format

You MUST end your response with a REVIEW_RESULT line:

REVIEW_RESULT:{{"alignment": 7, "approach": 8, "test_quality": 6{criteria_output}, "rationale": "summary"}}

IMPORTANT: The REVIEW_RESULT line must be the LAST line of your output."""


def _parse_review_output(output: str) -> dict[str, Any] | None:
	"""Parse REVIEW_RESULT from LLM output. Returns parsed dict or None."""
	idx = output.rfind(REVIEW_RESULT_MARKER)
	data = None
	if idx != -1:
		remainder = output[idx + len(REVIEW_RESULT_MARKER):]
		data = extract_json_from_text(remainder)

		# Fallback: single-line regex (matches MC_RESULT pattern in session.py)
		if not isinstance(data, dict):
			match = re.search(r"\{.*\}", remainder.split("\n")[0])
			if match:
				try:
					raw = json.loads(match.group(0))
					if isinstance(raw, dict):
						data = raw
				except json.JSONDecodeError:
					pass

	if not isinstance(data, dict):
		data = extract_json_from_text(output)

	if not isinstance(data, dict):
		return None

	return data


def _clamp_score(value: Any) -> int:
	"""Clamp a score value to 1-10 range."""
	try:
		return max(1, min(10, int(value)))
	except (TypeError, ValueError):
		return 5


class DiffReviewer:
	"""Reviews merged diffs via Claude subprocess."""

	def __init__(self, config: MissionConfig) -> None:
		self._config = config
		self._review_config: ReviewConfig = config.review

	async def review_unit(
		self,
		unit: WorkUnit,
		diff: str,
		objective: str,
		mission_id: str,
		epoch_id: str,
		project_snapshot: str = "",
	) -> UnitReview | None:
		"""Review a merged unit's diff. Returns UnitReview on success, None on failure."""
		if not self._review_config.enabled:
			return None

		if not diff or not diff.strip():
			logger.debug("Skipping review for unit %s: empty diff", unit.id)
			return None

		prompt = _build_review_prompt(unit, diff, objective, project_snapshot=project_snapshot)
		model = self._review_config.model

		try:
			proc = await asyncio.create_subprocess_exec(
				"claude", "--print", "--output-format", "text",
				"--model", model,
				"--max-turns", "1",
				"-p", prompt,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(),
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(),
				timeout=120,
			)
			output = stdout.decode() if stdout else ""
		except asyncio.TimeoutError:
			logger.warning("Review timed out for unit %s", unit.id)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			return None
		except Exception as exc:
			logger.warning("Review failed for unit %s: %s", unit.id, exc)
			return None

		if proc.returncode != 0:
			logger.warning("Review subprocess failed for unit %s (rc=%d)", unit.id, proc.returncode)
			return None

		logger.debug("Raw review output for %s (len=%d): %.500s", unit.id, len(output), output)
		data = _parse_review_output(output)
		if data is None:
			logger.warning("Could not parse REVIEW_RESULT for unit %s", unit.id)
			return None

		alignment = _clamp_score(data.get("alignment", 5))
		approach = _clamp_score(data.get("approach", 5))
		test_quality = _clamp_score(data.get("test_quality", 5))
		criteria_met = 0
		scores = [alignment, approach, test_quality]
		if unit.acceptance_criteria and "criteria_met" in data:
			criteria_met = _clamp_score(data["criteria_met"])
			scores.append(criteria_met)
		avg = round(sum(scores) / len(scores), 1)
		rationale = str(data.get("rationale", ""))[:500]

		return UnitReview(
			work_unit_id=unit.id,
			mission_id=mission_id,
			epoch_id=epoch_id,
			alignment_score=alignment,
			approach_score=approach,
			test_score=test_quality,
			criteria_met_score=criteria_met,
			avg_score=avg,
			rationale=rationale,
			model=model,
		)
