"""LLM-based evaluator for intelligence findings using claude --print."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess as _subprocess
from pathlib import Path

from autodev.intelligence.evaluator import evaluate_findings, generate_proposals
from autodev.intelligence.models import AdaptationProposal, Finding
from autodev.intelligence.utils import find_claude_binary

logger = logging.getLogger(__name__)

EVALUATOR_PROMPT = """\
You are evaluating intelligence findings for an autonomous development system.
Read the program document below, then evaluate each finding.

<program>
{program}
</program>

<findings>
{findings}
</findings>

For each finding, decide: integrate or skip.
Return a JSON array wrapped in ```json fences:
[
  {{
    "finding_id": "...",
    "decision": "integrate" | "skip",
    "reasoning": "1-2 sentences",
    "proposed_action": "What to implement",
    "target_modules": ["file1.py", "file2.py"]
  }}
]
"""


def _extract_json_array(text: str) -> list[dict]:
	"""Extract a JSON array from LLM output, handling markdown fences and preamble."""
	fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
	if fence_match:
		candidate = fence_match.group(1).strip()
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, list):
				return parsed
		except json.JSONDecodeError:
			pass

	bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
	if bracket_match:
		try:
			parsed = json.loads(bracket_match.group(0))
			if isinstance(parsed, list):
				return parsed
		except json.JSONDecodeError:
			pass

	raise ValueError("No valid JSON array found in LLM output")


def _build_enriched_program(program_md: str, project_path: Path) -> str:
	"""Enrich program.md with CLAUDE.md architecture section and recent git log."""
	enriched = program_md

	claude_md_path = project_path / "CLAUDE.md"
	if claude_md_path.exists():
		claude_text = claude_md_path.read_text()
		arch_match = re.search(
			r"(## Architecture\s*\n.*?)(?=\n## |\Z)", claude_text, re.DOTALL
		)
		if arch_match:
			enriched += "\n\n## Current Architecture (from CLAUDE.md)\n"
			enriched += arch_match.group(1).strip()

	try:
		result = _subprocess.run(
			["git", "log", "--oneline", "-20"],
			capture_output=True, text=True, cwd=str(project_path), timeout=5,
		)
		if result.returncode == 0 and result.stdout.strip():
			enriched += "\n\n## Recent Commits\n```\n" + result.stdout.strip() + "\n```"
	except Exception:
		pass

	return enriched


async def evaluate_findings_llm(
	findings: list[Finding],
	project_path: Path,
	program_path: Path | None = None,
) -> list[AdaptationProposal]:
	"""Evaluate findings using claude --print, falling back to keyword evaluator."""
	if program_path is None:
		program_path = project_path / "docs" / "program.md"

	try:
		program_md = program_path.read_text()
	except OSError:
		logger.warning("Cannot read program.md at %s, falling back to keyword evaluator", program_path)
		return generate_proposals(evaluate_findings(findings))

	enriched = _build_enriched_program(program_md, project_path)

	findings_data = [
		{
			"id": f.id,
			"source": f.source,
			"title": f.title,
			"url": f.url,
			"summary": f.summary,
			"relevance_score": f.relevance_score,
		}
		for f in findings
	]

	prompt = EVALUATOR_PROMPT.format(
		program=enriched,
		findings=json.dumps(findings_data, indent=2),
	)

	claude_bin = find_claude_binary()

	try:
		proc = await asyncio.create_subprocess_exec(
			claude_bin, "--print", "-p", prompt,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
			cwd=str(project_path),
		)
		stdout, stderr = await proc.communicate()

		if proc.returncode != 0:
			logger.warning("claude --print failed (exit %d): %s", proc.returncode, stderr.decode()[:200])
			return generate_proposals(evaluate_findings(findings))

		output = stdout.decode()
		decisions = _extract_json_array(output)

	except Exception as exc:
		logger.warning("LLM evaluator failed: %s", exc)
		return generate_proposals(evaluate_findings(findings))

	finding_map = {f.id: f for f in findings}

	proposals: list[AdaptationProposal] = []
	for decision in decisions:
		if decision.get("decision") != "integrate":
			continue

		finding_id = decision.get("finding_id", "")
		finding = finding_map.get(finding_id)
		title = finding.title if finding else decision.get("proposed_action", "Unknown")

		proposals.append(AdaptationProposal(
			finding_id=finding_id,
			title=f"Adapt: {title}",
			description=decision.get("reasoning", ""),
			proposal_type="integration",
			target_modules=decision.get("target_modules", []),
			priority=2,
		))

	return proposals
