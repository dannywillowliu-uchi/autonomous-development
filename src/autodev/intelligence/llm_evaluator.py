"""LLM-powered evaluator for intelligence findings.

Uses Claude via subprocess to make binary integrate/skip decisions
guided by the program.md steering document and enriched project context.
Falls back to the keyword evaluator on failure."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from autodev.intelligence.models import AdaptationProposal, Finding
from autodev.intelligence.utils import find_claude_binary

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are evaluating intelligence findings for an autonomous development system.
Read the program document below, then evaluate each finding.

<program>
{enriched_program}
</program>

<findings>
{findings_json}
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


async def _build_enriched_program(
	project_path: Path,
	program_path: Path,
) -> str:
	"""Build enriched program context from program.md + architecture + git log."""
	parts: list[str] = []

	if program_path.exists():
		parts.append(program_path.read_text())
	else:
		logger.warning("program.md not found at %s", program_path)

	claude_md = project_path / "CLAUDE.md"
	if claude_md.exists():
		text = claude_md.read_text()
		arch_match = re.search(r"(## Architecture.*?)(?=\n## |\Z)", text, re.DOTALL)
		if arch_match:
			parts.append("\n## Current Architecture\n" + arch_match.group(1))

	try:
		proc = await asyncio.create_subprocess_exec(
			"git", "log", "--oneline", "-20",
			cwd=str(project_path),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, _ = await proc.communicate()
		if proc.returncode == 0 and stdout.strip():
			parts.append("\n## Recent Activity\n" + stdout.decode().strip())
	except Exception:
		logger.debug("Could not read git log", exc_info=True)

	return "\n\n".join(parts)


def _decisions_to_proposals(
	decisions: list[dict],
	findings_by_id: dict[str, Finding],
) -> list[AdaptationProposal]:
	"""Convert LLM integrate decisions to AdaptationProposal objects."""
	proposals: list[AdaptationProposal] = []
	for dec in decisions:
		if dec.get("decision") != "integrate":
			continue
		finding_id = dec.get("finding_id", "")
		finding = findings_by_id.get(finding_id)
		title = dec.get("proposed_action", "")
		if finding:
			title = title or f"Adapt: {finding.title}"
		proposals.append(AdaptationProposal(
			finding_id=finding_id,
			title=title,
			description=dec.get("reasoning", ""),
			proposal_type="integration",
			target_modules=dec.get("target_modules", []),
			priority=2,
			effort_estimate="medium",
			risk_level="low",
		))
	return proposals


async def evaluate_findings(
	findings: list[Finding],
	project_path: Path,
	program_path: Path | None = None,
) -> list[AdaptationProposal]:
	"""Evaluate findings using LLM judgment guided by program.md.

	Args:
		findings: Raw findings from all scanners.
		project_path: Path to the autodev project root.
		program_path: Path to program.md (defaults to project_path/docs/program.md).

	Returns:
		List of AdaptationProposal for findings the LLM decided to integrate.
	"""
	if not findings:
		return []

	if program_path is None:
		program_path = project_path / "docs" / "program.md"

	enriched_program = await _build_enriched_program(project_path, program_path)

	findings_dicts = []
	findings_by_id: dict[str, Finding] = {}
	for f in findings:
		findings_by_id[f.id] = f
		findings_dicts.append({
			"id": f.id,
			"source": f.source,
			"title": f.title,
			"url": f.url,
			"summary": f.summary,
		})
	findings_json = json.dumps(findings_dicts, indent=2)

	prompt = _PROMPT_TEMPLATE.format(
		enriched_program=enriched_program,
		findings_json=findings_json,
	)

	try:
		claude_bin = find_claude_binary()
		proc = await asyncio.create_subprocess_exec(
			claude_bin, "--print", "-p", prompt,
			cwd=str(project_path),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, stderr = await proc.communicate()

		if proc.returncode != 0:
			raise RuntimeError(
				f"claude subprocess failed (rc={proc.returncode}): {stderr.decode()[:200]}"
			)

		raw_output = stdout.decode()
		decisions = _extract_json_array(raw_output)
		return _decisions_to_proposals(decisions, findings_by_id)

	except Exception:
		logger.warning(
			"LLM evaluator failed, falling back to keyword evaluator", exc_info=True
		)
		from autodev.intelligence.evaluator import evaluate_findings as kw_evaluate
		from autodev.intelligence.evaluator import generate_proposals
		evaluated = kw_evaluate(findings)
		return generate_proposals(evaluated)
