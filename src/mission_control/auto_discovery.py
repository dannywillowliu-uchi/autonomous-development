"""Auto-discovery engine -- multi-stage pipeline for codebase analysis and improvement proposals.

Stages:
1. Analyze -- deep codebase analysis, outputs structured gaps
2. Research -- web search for best practices per gap category (parallel, optional)
3. Synthesize -- combine analysis + research into evidence-based proposals
"""

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
from mission_control.models import BacklogItem, DiscoveryResult

logger = logging.getLogger(__name__)


def _compute_priority(impact: int, effort: int) -> float:
	"""Compute priority score: impact * (11 - effort) / 10."""
	impact = max(1, min(10, impact))
	effort = max(1, min(10, effort))
	return round(impact * (11 - effort) / 10, 1)


@dataclass
class AnalysisOutput:
	"""Structured output from the analyze stage."""

	architecture: str = ""
	patterns: list[str] = field(default_factory=list)
	gaps: list[dict[str, str]] = field(default_factory=list)
	raw: str = ""


@dataclass
class ResearchOutput:
	"""Structured output from the research stage."""

	findings: list[dict[str, str]] = field(default_factory=list)
	raw: str = ""


class DiscoveryEngine:
	"""Multi-stage discovery pipeline for codebase improvement proposals."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db

	# -- Public API (unchanged signature) --

	async def discover(self) -> tuple[DiscoveryResult, list[BacklogItem]]:
		"""Run multi-stage discovery pipeline on the target codebase.

		Stage 1 (analyze) -> if research_enabled: Stage 2 (research) -> Stage 3 (synthesize)
		"""
		dc = self.config.discovery

		# Stage 1: Analyze
		analysis = await self._stage_analyze()
		if analysis is None:
			result = DiscoveryResult(
				target_path=str(self.config.target.resolved_path),
				model=dc.model,
				error_type="analyze_failed",
				error_detail="Stage 1 (analyze) failed to produce output",
			)
			self.db.insert_discovery_result(result, [])
			return result, []

		# Stage 2: Research (optional)
		research: ResearchOutput | None = None
		if dc.research_enabled and analysis.gaps:
			research = await self._stage_research(analysis)
			if research is None:
				logger.warning("Stage 2 (research) failed; proceeding without research context")

		# Stage 3: Synthesize
		output, error_type, error_detail = await self._stage_synthesize(analysis, research)
		result, items = self._parse_discovery_output(output)
		result.error_type = error_type
		result.error_detail = error_detail
		self.db.insert_discovery_result(result, items)
		self._insert_items_to_backlog(items)
		return result, items

	def _insert_items_to_backlog(self, items: list[BacklogItem]) -> None:
		"""Insert BacklogItems, skipping duplicate titles."""
		if not items:
			return
		existing_titles: set[str] = {
			row["title"]
			for row in self.db.conn.execute("SELECT title FROM backlog_items").fetchall()
		}
		for item in items:
			if item.title in existing_titles:
				logger.debug("Skipping duplicate backlog item: %s", item.title)
				continue
			self.db.insert_backlog_item(item)
			existing_titles.add(item.title)

	# -- Stage 1: Analyze --

	async def _stage_analyze(self) -> AnalysisOutput | None:
		"""Run deep codebase analysis, returning structured gaps."""
		prompt = self._build_analyze_prompt()
		output, error_type, _ = await self._run_stage_subprocess(
			prompt, model=self.config.discovery.model, stage_name="analyze",
		)
		if error_type:
			return None
		return self._parse_analysis_output(output)

	def _build_analyze_prompt(self) -> str:
		"""Build prompt for the analysis stage."""
		target = self.config.target
		categories = (
			"testing, error_handling, performance, security, "
			"documentation, architecture, code_quality, observability"
		)
		return f"""You are a senior software architect performing a deep analysis \
of the codebase at the current directory.

Analyze the codebase thoroughly and identify:
1. **Architecture**: How the project is structured (modules, layers, patterns)
2. **Patterns**: Key design patterns and conventions used
3. **Gaps**: Areas where the codebase falls short of best practices

For each gap, classify it into one of these categories: {categories}.

You MUST output a JSON block with an ANALYSIS_RESULT: marker:

ANALYSIS_RESULT:
```json
{{
  "architecture": "Brief description of overall architecture",
  "patterns": ["pattern1", "pattern2"],
  "gaps": [
    {{
      "category": "testing",
      "description": "Missing integration tests for API endpoints",
      "files": "src/api.py, src/routes.py",
      "severity": "high"
    }}
  ]
}}
```

Be specific -- reference actual files, functions, and line numbers you find.

Project: {target.name}
Target path: {target.resolved_path}
"""

	def _parse_analysis_output(self, output: str) -> AnalysisOutput:
		"""Extract ANALYSIS_RESULT: JSON from output."""
		if not output:
			return AnalysisOutput(raw=output)

		data: dict[str, Any] | None = None
		marker = "ANALYSIS_RESULT:"
		idx = output.rfind(marker)
		if idx != -1:
			remainder = output[idx + len(marker):]
			data = extract_json_from_text(remainder)

		if not isinstance(data, dict):
			data = extract_json_from_text(output)

		if not isinstance(data, dict):
			logger.warning("Failed to parse analysis output; returning empty gaps")
			return AnalysisOutput(raw=output)

		gaps = data.get("gaps", [])
		if not isinstance(gaps, list):
			gaps = []

		return AnalysisOutput(
			architecture=str(data.get("architecture", "")),
			patterns=[str(p) for p in data.get("patterns", []) if isinstance(p, str)],
			gaps=[g for g in gaps if isinstance(g, dict)],
			raw=output,
		)

	# -- Stage 2: Research --

	async def _stage_research(self, analysis: AnalysisOutput) -> ResearchOutput | None:
		"""Research best practices for each gap category via parallel web searches."""
		dc = self.config.discovery

		# Group gaps by category
		categories: dict[str, list[dict[str, str]]] = {}
		for gap in analysis.gaps:
			cat = gap.get("category", "general")
			categories.setdefault(cat, []).append(gap)

		# Spawn parallel research queries (one per category, capped)
		category_list = list(categories.keys())[:dc.research_parallel_queries]

		async def _research_one(category: str, gaps: list[dict[str, str]]) -> dict[str, str] | None:
			gap_descriptions = "\n".join(
				f"- {g.get('description', 'unknown')}" for g in gaps
			)
			project_name = self.config.target.name
			prompt = f"""You are a software engineering researcher. \
Search the web for best practices related to the following gaps \
in a {project_name} codebase.

Category: {category}
Gaps found:
{gap_descriptions}

Use WebSearch to find:
1. Current best practices for addressing these gaps
2. Examples from well-known open source projects
3. Specific libraries or tools that could help

You MUST output a JSON block with a RESEARCH_RESULT: marker:

RESEARCH_RESULT:
```json
{{
  "gap_category": "{category}",
  "best_practices": "Summary of best practices found",
  "examples": "Specific examples from OSS projects",
  "sources": "URLs of key references"
}}
```
"""
			output, error_type, _ = await self._run_stage_subprocess(
				prompt, model=dc.research_model, stage_name=f"research-{category}",
				enable_web=True,
			)
			if error_type:
				return None
			return self._parse_research_result(output, category)

		tasks = [
			_research_one(cat, categories[cat])
			for cat in category_list
		]
		results = await asyncio.gather(*tasks, return_exceptions=True)

		findings: list[dict[str, str]] = []
		raw_parts: list[str] = []
		for r in results:
			if isinstance(r, dict):
				findings.append(r)
				raw_parts.append(str(r))
			elif isinstance(r, Exception):
				logger.warning("Research query failed: %s", r)

		if not findings:
			return None

		return ResearchOutput(findings=findings, raw="\n---\n".join(raw_parts))

	def _parse_research_result(self, output: str, category: str) -> dict[str, str] | None:
		"""Extract RESEARCH_RESULT: JSON from a single research subprocess output."""
		if not output:
			return None

		data: dict[str, Any] | None = None
		marker = "RESEARCH_RESULT:"
		idx = output.rfind(marker)
		if idx != -1:
			remainder = output[idx + len(marker):]
			data = extract_json_from_text(remainder)

		if not isinstance(data, dict):
			data = extract_json_from_text(output)

		if not isinstance(data, dict):
			return None

		return {
			"gap_category": str(data.get("gap_category", category)),
			"best_practices": str(data.get("best_practices", "")),
			"examples": str(data.get("examples", "")),
			"sources": str(data.get("sources", "")),
		}

	# -- Stage 3: Synthesize --

	async def _stage_synthesize(
		self,
		analysis: AnalysisOutput,
		research: ResearchOutput | None,
	) -> tuple[str, str, str]:
		"""Combine analysis + research into final discovery proposals."""
		prompt = self._build_synthesize_prompt(analysis, research)
		return await self._run_stage_subprocess(
			prompt, model=self.config.discovery.model, stage_name="synthesize",
		)

	def _build_synthesize_prompt(
		self,
		analysis: AnalysisOutput,
		research: ResearchOutput | None,
	) -> str:
		"""Build the synthesis prompt combining analysis gaps and research findings."""
		dc = self.config.discovery
		target = self.config.target
		tracks = dc.tracks
		max_per_track = dc.max_items_per_track

		# Build context sections
		handoff_section = self._build_handoff_section()
		past_section = self._build_past_section()
		track_text = self._format_track_instructions(tracks)

		# Analysis context
		analysis_section = ""
		if analysis.gaps:
			gaps_json = json.dumps(analysis.gaps, indent=2)
			analysis_section = f"""
## Codebase Analysis Results

Architecture: {analysis.architecture}
Patterns found: {', '.join(analysis.patterns) if analysis.patterns else 'none identified'}

Gaps identified:
```json
{gaps_json}
```
"""

		# Research context (when available)
		research_section = ""
		if research and research.findings:
			findings_json = json.dumps(research.findings, indent=2)
			research_section = f"""
## Research Findings (Best Practices)

The following best practices were found through web research:
```json
{findings_json}
```

Use these findings to inform your recommendations. Cite specific practices and examples where relevant.
"""

		return f"""You are a senior software architect synthesizing improvement \
proposals for the codebase at the current directory.
{analysis_section}{research_section}
Your job is to produce concrete, actionable improvements across these tracks:

{track_text}

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

	# -- Shared helpers --

	def _build_handoff_section(self) -> str:
		"""Build recently completed work context for prompts."""
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
						return (
							"\n## Recently Completed Work\n"
							f"{summaries}\n"
						)
		except Exception:
			pass
		return ""

	def _build_past_section(self) -> str:
		"""Build past discovery titles section for deduplication."""
		past_titles = self.db.get_past_discovery_titles(limit=50)
		if past_titles:
			titles_list = "\n".join(f"- {t}" for t in past_titles)
			return (
				"\n## Previously Discovered (DO NOT repeat these)\n"
				f"{titles_list}\n"
			)
		return ""

	def _format_track_instructions(self, tracks: list[str]) -> str:
		"""Format track description text."""
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
		return "\n".join(track_instructions)

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

	async def _run_stage_subprocess(
		self,
		prompt: str,
		*,
		model: str,
		stage_name: str,
		enable_web: bool = False,
	) -> tuple[str, str, str]:
		"""Spawn `claude -p` in target repo dir, return (output, error_type, error_detail).

		Args:
			prompt: The prompt text to send to claude.
			model: Model identifier (e.g. "opus", "sonnet").
			stage_name: Name for logging (e.g. "analyze", "research-testing").
			enable_web: If True, adds --permission-mode bypassPermissions for web tools.
		"""
		dc = self.config.discovery
		budget = dc.budget_per_call_usd
		target_path = str(self.config.target.resolved_path)

		cmd = [
			"claude", "-p", "--output-format", "text",
			"--max-budget-usd", str(budget), "--model", model,
		]
		if enable_web:
			cmd.extend(["--permission-mode", "bypassPermissions"])

		logger.info(
			"Running stage '%s' on %s (model=%s, budget=$%.2f, web=%s)",
			stage_name, target_path, model, budget, enable_web,
		)

		proc = None
		discovery_timeout = self.config.scheduler.session_timeout
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
				timeout=discovery_timeout,
			)
			output = stdout.decode() if stdout else ""
			stderr_text = stderr.decode() if stderr else ""
		except asyncio.TimeoutError:
			logger.error(
				"Stage '%s' timed out after %ds | cwd=%s",
				stage_name, discovery_timeout, target_path,
			)
			if proc is not None:
				try:
					proc.kill()
					await proc.wait()
				except ProcessLookupError:
					pass
			return "", "timeout", f"stage '{stage_name}' timed out after {discovery_timeout}s"

		if proc.returncode != 0:
			error_type = self._classify_error(stderr_text)
			error_detail = stderr_text[:500]
			logger.warning(
				"Stage '%s' failed | rc=%d stderr=%s",
				stage_name, proc.returncode, error_detail,
			)
			return output, error_type, error_detail

		return output, "", ""

	def _parse_discovery_output(self, output: str) -> tuple[DiscoveryResult, list[BacklogItem]]:
		"""Extract DISCOVERY_RESULT: JSON from output and return BacklogItems directly."""
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

		items: list[BacklogItem] = []
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

			item = BacklogItem(
				track=track,
				title=str(raw.get("title", "")),
				description=str(raw.get("description", "")),
				impact=impact,
				effort=effort,
				priority_score=priority,
			)
			items.append(item)

		# Sort by priority score descending
		items.sort(key=lambda i: i.priority_score, reverse=True)

		result.item_count = len(items)
		return result, items

	def compose_objective(self, items: list[BacklogItem]) -> str:
		"""Turn approved discovery items into a mission objective string."""
		if not items:
			return ""

		# Group by track
		by_track: dict[str, list[BacklogItem]] = {}
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
			lines.append("")

		return "\n".join(lines)
