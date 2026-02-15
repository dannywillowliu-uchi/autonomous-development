"""Auto-discovery engine -- analyzes a target codebase and generates improvement objectives."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from mission_control.config import MissionConfig, claude_subprocess_env
from mission_control.db import Database
from mission_control.json_utils import extract_json_from_text
from mission_control.models import DiscoveryItem, DiscoveryResult

logger = logging.getLogger(__name__)


def _compute_priority(impact: int, effort: int) -> float:
	"""Compute priority score: impact * (11 - effort) / 10."""
	impact = max(1, min(10, impact))
	effort = max(1, min(10, effort))
	return round(impact * (11 - effort) / 10, 1)


class DiscoveryEngine:
	"""Analyzes a target codebase and generates improvement objectives."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db

	async def discover(self) -> tuple[DiscoveryResult, list[DiscoveryItem]]:
		"""Run three-track discovery on the target codebase."""
		prompt = self._build_discovery_prompt()
		output, error_type, error_detail = await self._run_discovery_subprocess(prompt)
		result, items = self._parse_discovery_output(output)
		result.error_type = error_type
		result.error_detail = error_detail
		self.db.insert_discovery_result(result, items)
		return result, items

	def _build_discovery_prompt(self) -> str:
		"""Build prompt with three-track analysis instructions."""
		dc = self.config.discovery
		target = self.config.target
		tracks = dc.tracks
		max_per_track = dc.max_items_per_track

		# Get past discovery titles to avoid repetition
		past_titles = self.db.get_past_discovery_titles(limit=50)
		past_section = ""
		if past_titles:
			titles_list = "\n".join(f"- {t}" for t in past_titles)
			past_section = (
				"\n## Previously Discovered (DO NOT repeat these)\n"
				f"{titles_list}\n"
			)

		# Get recent mission handoff summaries for context
		handoff_section = ""
		try:
			latest_mission = self.db.get_latest_mission()
			if latest_mission:
				handoffs = self.db.get_recent_handoffs(latest_mission.id, limit=20)
				if handoffs:
					completed = [h for h in handoffs if h.status == "completed"]
					if completed:
						summaries = "\n".join(
							f"- {h.summary[:100]}" for h in completed[:10]
						)
						handoff_section = (
							"\n## Recently Completed Work\n"
							f"{summaries}\n"
						)
		except Exception:
			pass

		track_instructions = []
		if "feature" in tracks:
			track_instructions.append(
				"**Track A - Features**: New capabilities, missing functionality, "
				"UX improvements, API additions. Focus on what would make the "
				"project more useful or complete."
			)
		if "quality" in tracks:
			track_instructions.append(
				"**Track B - Code Quality**: Refactoring opportunities, test "
				"coverage gaps, performance improvements, code smells, dead code, "
				"missing error handling, documentation gaps."
			)
		if "security" in tracks:
			track_instructions.append(
				"**Track C - Security/Reliability**: Security vulnerabilities, "
				"input validation gaps, error handling weaknesses, race conditions, "
				"dependency issues, logging gaps."
			)

		tracks_text = "\n".join(track_instructions)

		prompt = f"""You are a senior software architect analyzing the codebase at the current directory.

Your job is to discover concrete, actionable improvements across these tracks:

{tracks_text}

For each track, identify up to {max_per_track} improvements.
Each item must be specific and scoped enough for a single developer to complete in 1-4 hours.
{past_section}{handoff_section}
## Output Format

You MUST output a JSON block with a DISCOVERY_RESULT: marker. Example:

DISCOVERY_RESULT:
```json
{{
  "items": [
    {{
      "track": "feature",
      "title": "Add retry logic to API client",
      "description": "The HTTP client in src/api.py has no retry logic. Add backoff for transient failures.",
      "rationale": "API calls fail silently on network issues.",
      "files_hint": "src/api.py, tests/test_api.py",
      "impact": 7,
      "effort": 3
    }}
  ]
}}
```

Field definitions:
- track: one of {json.dumps(tracks)}
- title: short actionable title (imperative form)
- description: detailed explanation of what to change and how
- rationale: why this matters
- files_hint: comma-separated file paths most relevant
- impact: 1-10 (10 = highest impact on project quality/usefulness)
- effort: 1-10 (10 = most effort required)

## Rules
1. Be SPECIFIC -- reference actual files, functions, and patterns you find
2. Do NOT suggest items that duplicate previously discovered work
3. Focus on improvements with highest impact-to-effort ratio
4. Each item should be independently implementable
5. Read the codebase thoroughly before proposing items

Project: {target.name}
Target path: {target.resolved_path}
"""
		return prompt

	@staticmethod
	def _classify_error(stderr_text: str) -> str:
		"""Classify subprocess error from stderr content."""
		lower = stderr_text.lower()
		if re.search(r"timeout", lower):
			return "timeout"
		if re.search(r"budget", lower):
			return "budget_exceeded"
		if re.search(r"permission", lower):
			return "permission_denied"
		if re.search(r"corrupt|workspace", lower):
			return "workspace_corruption"
		return "unknown"

	async def _run_discovery_subprocess(self, prompt: str) -> tuple[str, str, str]:
		"""Spawn `claude -p` in target repo dir, return (output, error_type, error_detail)."""
		dc = self.config.discovery
		budget = dc.budget_per_call_usd
		model = dc.model
		target_path = str(self.config.target.resolved_path)

		cmd = ["claude", "-p", "--output-format", "text", "--max-budget-usd", str(budget), "--model", model]

		logger.info(
			"Running discovery on %s (model=%s, budget=$%.2f)",
			target_path, model, budget,
		)

		proc = None
		try:
			proc = await asyncio.create_subprocess_exec(
				*cmd,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(),
				cwd=target_path,
			)
			stdout, stderr = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()),
				timeout=300,
			)
			output = stdout.decode() if stdout else ""
			stderr_text = stderr.decode() if stderr else ""
		except asyncio.TimeoutError:
			logger.error(
				"Discovery subprocess timed out after 300s | cmd=%s cwd=%s",
				cmd, target_path,
			)
			if proc is not None:
				try:
					proc.kill()
					await proc.wait()
				except ProcessLookupError:
					pass
			return "", "timeout", "subprocess timed out after 300s"

		if proc.returncode != 0:
			error_type = self._classify_error(stderr_text)
			error_detail = stderr_text[:500]
			logger.warning(
				"Discovery subprocess failed | cmd=%s cwd=%s rc=%d stderr=%s",
				cmd, target_path, proc.returncode, error_detail,
			)
			return output, error_type, error_detail

		return output, "", ""

	def _parse_discovery_output(self, output: str) -> tuple[DiscoveryResult, list[DiscoveryItem]]:
		"""Extract DISCOVERY_RESULT: JSON from output."""
		dc = self.config.discovery
		target_path = str(self.config.target.resolved_path)

		result = DiscoveryResult(
			target_path=target_path,
			raw_output=output,
			model=dc.model,
		)

		if not output:
			return result, []

		# Extract JSON from DISCOVERY_RESULT: marker
		data: dict[str, Any] | None = None
		marker = "DISCOVERY_RESULT:"
		idx = output.rfind(marker)
		if idx != -1:
			remainder = output[idx + len(marker):]
			data = extract_json_from_text(remainder)

		# Fallback: try the whole output
		if not isinstance(data, dict):
			data = extract_json_from_text(output)

		if not isinstance(data, dict):
			logger.warning("Failed to parse discovery output")
			return result, []

		raw_items = data.get("items", [])
		if not isinstance(raw_items, list):
			return result, []

		items: list[DiscoveryItem] = []
		for raw in raw_items:
			if not isinstance(raw, dict):
				continue
			track = str(raw.get("track", ""))
			if track not in dc.tracks:
				continue

			impact = int(raw.get("impact", 5))
			effort = int(raw.get("effort", 5))
			priority = _compute_priority(impact, effort)

			if priority < dc.min_priority_score:
				continue

			item = DiscoveryItem(
				track=track,
				title=str(raw.get("title", "")),
				description=str(raw.get("description", "")),
				rationale=str(raw.get("rationale", "")),
				files_hint=str(raw.get("files_hint", "")),
				impact=impact,
				effort=effort,
				priority_score=priority,
			)
			items.append(item)

		# Sort by priority score descending
		items.sort(key=lambda i: i.priority_score, reverse=True)

		result.item_count = len(items)
		return result, items

	def compose_objective(self, items: list[DiscoveryItem]) -> str:
		"""Turn approved discovery items into a mission objective string."""
		if not items:
			return ""

		# Group by track
		by_track: dict[str, list[DiscoveryItem]] = {}
		for item in items:
			by_track.setdefault(item.track, []).append(item)

		lines = ["Implement the following improvements:\n"]

		track_labels = {
			"feature": "Features",
			"quality": "Code Quality",
			"security": "Security/Reliability",
		}

		for track, track_items in by_track.items():
			label = track_labels.get(track, track.title())
			lines.append(f"## {label}")
			for item in track_items:
				lines.append(f"- {item.title}: {item.description}")
				if item.files_hint:
					lines.append(f"  Files: {item.files_hint}")
			lines.append("")

		return "\n".join(lines)
