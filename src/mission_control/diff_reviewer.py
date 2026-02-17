"""LLM-based diff reviewer -- scores merged work units on quality dimensions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mission_control.config import MissionConfig, ReviewConfig, claude_subprocess_env
from mission_control.json_utils import extract_json_from_text
from mission_control.models import UnitReview, WorkUnit

logger = logging.getLogger(__name__)

REVIEW_RESULT_MARKER = "REVIEW_RESULT:"


def _build_review_prompt(unit: WorkUnit, diff: str, objective: str) -> str:
	return f"""You are a code reviewer evaluating a merged work unit's diff.

## Mission Objective
{objective}

## Work Unit
Title: {unit.title}
Description: {unit.description}

## Git Diff
```
{diff[:8000]}
```

## Instructions

Score each dimension 1-10:
1. **Alignment** (1-10): How well does this diff advance the mission objective?
2. **Approach** (1-10): Is the implementation approach clean, idiomatic, and maintainable?
3. **Test Quality** (1-10): Are the tests meaningful and covering real behavior (not trivial)?

Then provide a 1-2 sentence rationale summarizing the overall quality.

## Output Format

You MUST end your response with a REVIEW_RESULT line:

REVIEW_RESULT:{{"alignment": 7, "approach": 8, "test_quality": 6, "rationale": "Clean but shallow tests"}}

IMPORTANT: The REVIEW_RESULT line must be the LAST line of your output."""


def _parse_review_output(output: str) -> dict[str, Any] | None:
	"""Parse REVIEW_RESULT from LLM output. Returns parsed dict or None."""
	idx = output.rfind(REVIEW_RESULT_MARKER)
	data = None
	if idx != -1:
		remainder = output[idx + len(REVIEW_RESULT_MARKER):]
		data = extract_json_from_text(remainder)

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
	) -> UnitReview | None:
		"""Review a merged unit's diff. Returns UnitReview on success, None on failure."""
		if not self._review_config.enabled:
			return None

		if not diff or not diff.strip():
			logger.debug("Skipping review for unit %s: empty diff", unit.id)
			return None

		prompt = _build_review_prompt(unit, diff, objective)
		model = self._review_config.model
		budget = self._review_config.budget_per_review_usd

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
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
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

		data = _parse_review_output(output)
		if data is None:
			logger.warning("Could not parse REVIEW_RESULT for unit %s", unit.id)
			return None

		alignment = _clamp_score(data.get("alignment", 5))
		approach = _clamp_score(data.get("approach", 5))
		test_quality = _clamp_score(data.get("test_quality", 5))
		avg = round((alignment + approach + test_quality) / 3.0, 1)
		rationale = str(data.get("rationale", ""))[:500]

		return UnitReview(
			work_unit_id=unit.id,
			mission_id=mission_id,
			epoch_id=epoch_id,
			alignment_score=alignment,
			approach_score=approach,
			test_score=test_quality,
			avg_score=avg,
			rationale=rationale,
			model=model,
		)
