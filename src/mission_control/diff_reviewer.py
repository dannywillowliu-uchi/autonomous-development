"""LLM-based diff reviewer -- scores merged work units on quality dimensions."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from mission_control.config import MissionConfig, ReviewConfig, build_claude_cmd, claude_subprocess_env
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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_REVIEW_KEYS = {"alignment", "approach", "test_quality"}
_MARKER_RE = re.compile(
	r"[*`_~]*REVIEW[_\s-]*RESULT[*`_~]*\s*[:]\s*",
	re.IGNORECASE,
)


def _strip_ansi(text: str) -> str:
	return _ANSI_RE.sub("", text)


def _is_review_dict(d: Any) -> bool:
	if not isinstance(d, dict):
		return False
	return bool(_REVIEW_KEYS & set(d.keys()))


def _extract_scores_from_prose(text: str) -> dict[str, Any] | None:
	"""Last-resort: pull individual scores from natural language output."""
	scores: dict[str, Any] = {}
	patterns = {
		"alignment": r"[Aa]lignment[^0-9]*(\d{1,2})",
		"approach": r"[Aa]pproach[^0-9]*(\d{1,2})",
		"test_quality": r"[Tt]est[_ ]?[Qq]uality[^0-9]*(\d{1,2})",
		"criteria_met": r"[Cc]riteria[_ ]?[Mm]et[^0-9]*(\d{1,2})",
	}
	for key, pattern in patterns.items():
		m = re.search(pattern, text)
		if m:
			scores[key] = int(m.group(1))
	if not (_REVIEW_KEYS & set(scores.keys())):
		return None
	rationale_m = re.search(r"[Rr]ationale[:\s]*[\"']?(.+?)[\"']?\s*$", text, re.MULTILINE)
	if rationale_m:
		scores.setdefault("rationale", rationale_m.group(1).strip())
	return scores


def _parse_review_output(output: str) -> dict[str, Any] | None:
	"""Parse REVIEW_RESULT from LLM output. Returns parsed dict or None."""
	if not output or not output.strip():
		return None

	cleaned = _strip_ansi(output)

	# Strategy 1: regex-based marker search (case-insensitive, markdown-tolerant)
	data = None
	marker_match = _MARKER_RE.search(cleaned)
	if marker_match:
		remainder = cleaned[marker_match.end():]
		data = extract_json_from_text(remainder)
		if not _is_review_dict(data):
			match = re.search(r"\{.*\}", remainder.split("\n")[0])
			if match:
				try:
					raw = json.loads(match.group(0))
					if isinstance(raw, dict):
						data = raw
				except json.JSONDecodeError:
					pass

	# Strategy 2: exact marker (legacy path)
	if not _is_review_dict(data):
		idx = cleaned.rfind(REVIEW_RESULT_MARKER)
		if idx != -1:
			remainder = cleaned[idx + len(REVIEW_RESULT_MARKER):]
			data = extract_json_from_text(remainder)

	# Strategy 3: find any JSON object in the output that has review keys
	if not _is_review_dict(data):
		data = extract_json_from_text(cleaned)
		if not _is_review_dict(data):
			for line in reversed(cleaned.splitlines()):
				line = line.strip()
				if not line or "{" not in line:
					continue
				candidate = extract_json_from_text(line)
				if _is_review_dict(candidate):
					data = candidate
					break

	# Strategy 4: extract individual scores from prose
	if not _is_review_dict(data):
		data = _extract_scores_from_prose(cleaned)

	if not _is_review_dict(data):
		logger.warning(
			"All parse strategies failed. Output length=%d, last 300 chars: %.300s",
			len(cleaned), cleaned[-300:] if cleaned else "(empty)",
		)
		return None

	return data


def _clamp_score(value: Any) -> int:
	"""Clamp a score value to 1-10 range."""
	try:
		return max(1, min(10, int(value)))
	except (TypeError, ValueError):
		return 5


_DIMENSIONS = ("alignment", "approach", "test_quality")
_DIMENSION_ATTRS = {"alignment": "alignment_score", "approach": "approach_score", "test_quality": "test_score"}


def _dimension_averages(reviews: list[UnitReview]) -> dict[str, float]:
	"""Compute per-dimension averages across a list of reviews."""
	if not reviews:
		return {d: 0.0 for d in _DIMENSIONS}
	return {
		d: sum(getattr(r, _DIMENSION_ATTRS[d]) for r in reviews) / len(reviews)
		for d in _DIMENSIONS
	}


def compute_review_trend(reviews: list[UnitReview], window: int = 3) -> dict:
	"""Compute quality trend by comparing the last *window* reviews to the overall average.

	Returns dict with keys: overall_avg, recent_avg, trend, worst_dimension.
	"""
	if not reviews:
		return {"overall_avg": 0.0, "recent_avg": 0.0, "trend": "stable", "worst_dimension": "alignment"}

	overall = _dimension_averages(reviews)
	overall_avg = round(sum(overall.values()) / len(overall), 2)

	recent = reviews[-window:]
	recent_avgs = _dimension_averages(recent)
	recent_avg = round(sum(recent_avgs.values()) / len(recent_avgs), 2)

	if recent_avg > overall_avg + 0.5:
		trend = "improving"
	elif recent_avg < overall_avg - 0.5:
		trend = "declining"
	else:
		trend = "stable"

	worst_dimension = min(recent_avgs, key=lambda d: recent_avgs[d])

	return {
		"overall_avg": overall_avg,
		"recent_avg": recent_avg,
		"trend": trend,
		"worst_dimension": worst_dimension,
	}


def is_quality_declining(reviews: list[UnitReview], window: int = 3, threshold: float = 1.5) -> bool:
	"""Return True if the recent window average is more than *threshold* below the overall average."""
	if not reviews:
		return False
	trend = compute_review_trend(reviews, window=window)
	return trend["overall_avg"] - trend["recent_avg"] > threshold


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

		cmd = build_claude_cmd(self._config, model=model, max_turns=1, prompt=prompt)
		try:
			proc = await asyncio.create_subprocess_exec(
				*cmd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(self._config),
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

		if not output or not output.strip():
			stderr_text = stderr.decode() if stderr else ""
			logger.warning(
				"Review returned empty stdout for unit %s (rc=%d, stderr_len=%d): %.200s",
				unit.id, proc.returncode, len(stderr_text), stderr_text[:200],
			)
			return None

		logger.debug("Raw review output for %s (len=%d): %.500s", unit.id, len(output), output)
		data = _parse_review_output(output)
		if data is None:
			logger.warning(
				"Could not parse REVIEW_RESULT for unit %s (output_len=%d, last_200=%.200s)",
				unit.id, len(output), output[-200:],
			)
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
