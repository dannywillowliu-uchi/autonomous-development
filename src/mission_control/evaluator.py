"""Objective evaluator -- spawns a lightweight Claude session to score progress."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from mission_control.config import MissionConfig

log = logging.getLogger(__name__)

EVALUATOR_PROMPT = """\
You are evaluating progress toward a development objective.

## Objective
{objective}

## Current State (commit: {snapshot_hash})

## Round Summary
{round_summary}

## Instructions
Score progress from 0.0 to 1.0 where:
- 0.0 = no progress
- 0.5 = significant progress, major work remaining
- 0.9 = nearly complete, minor polish needed
- 1.0 = fully complete

Output JSON only:
{{"score": 0.X, "met": true/false, "reasoning": "why this score", "remaining": ["task1", "task2"]}}

Set "met" to true ONLY when score >= 0.9 AND the objective is genuinely satisfied.\
"""


@dataclass
class ObjectiveEvaluation:
	score: float = 0.0
	met: bool = False
	reasoning: str = ""
	remaining: list[str] = field(default_factory=list)


def _parse_evaluation(text: str) -> ObjectiveEvaluation:
	"""Extract JSON from Claude output and build an ObjectiveEvaluation."""
	# Strip markdown fences if present
	cleaned = re.sub(r"```(?:json)?\s*", "", text)
	cleaned = re.sub(r"```", "", cleaned)
	cleaned = cleaned.strip()

	try:
		data = json.loads(cleaned)
	except json.JSONDecodeError:
		# Try to find JSON object in the text
		match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
		if not match:
			log.warning("Failed to parse evaluation response")
			return ObjectiveEvaluation()
		try:
			data = json.loads(match.group())
		except json.JSONDecodeError:
			log.warning("Failed to parse evaluation JSON from response")
			return ObjectiveEvaluation()

	score = float(data.get("score", 0.0))
	score = max(0.0, min(1.0, score))

	return ObjectiveEvaluation(
		score=score,
		met=bool(data.get("met", False)),
		reasoning=str(data.get("reasoning", "")),
		remaining=list(data.get("remaining", [])),
	)


async def evaluate_objective(
	config: MissionConfig,
	snapshot_hash: str,
	round_summary: str,
	objective: str,
) -> ObjectiveEvaluation:
	"""Spawn a lightweight Claude session to evaluate objective progress."""
	prompt = EVALUATOR_PROMPT.format(
		objective=objective,
		snapshot_hash=snapshot_hash,
		round_summary=round_summary,
	)

	cmd = [
		"claude", "-p",
		"--output-format", "text",
		"--max-budget-usd", "0.50",
		"--model", config.scheduler.model,
		prompt,
	]

	log.info("Evaluating objective (commit %s)", snapshot_hash[:8])

	try:
		proc = await asyncio.create_subprocess_exec(
			*cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, stderr = await proc.communicate()

		if proc.returncode != 0:
			log.warning(
				"Evaluator exited with code %d: %s",
				proc.returncode,
				stderr.decode().strip(),
			)
			return ObjectiveEvaluation()

		output = stdout.decode().strip()
		if not output:
			log.warning("Evaluator returned empty output")
			return ObjectiveEvaluation()

		evaluation = _parse_evaluation(output)
		log.info(
			"Evaluation: score=%.2f met=%s reasoning=%s",
			evaluation.score,
			evaluation.met,
			evaluation.reasoning[:80],
		)
		return evaluation

	except (OSError, FileNotFoundError) as exc:
		log.error("Failed to run evaluator: %s", exc)
		return ObjectiveEvaluation()
