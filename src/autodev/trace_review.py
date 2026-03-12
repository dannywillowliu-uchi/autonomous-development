"""Trace analyzer for autonomous development agent runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from autodev.db import Database


ERROR_PATTERNS_RE = [
	(re.compile(r"Traceback \(most recent call last\)"), "Traceback"),
	(re.compile(r"Error:\s*(.{1,80})"), "Error"),
	(re.compile(r"FAILED\b"), "FAILED"),
	(re.compile(r"AssertionError"), "AssertionError"),
	(re.compile(r"ModuleNotFoundError"), "ModuleNotFoundError"),
	(re.compile(r"ImportError"), "ImportError"),
	(re.compile(r"SyntaxError"), "SyntaxError"),
	(re.compile(r"TimeoutError"), "TimeoutError"),
	(re.compile(r"PermissionError"), "PermissionError"),
	(re.compile(r"merge conflict", re.IGNORECASE), "merge conflict"),
]


@dataclass
class RunAnalysis:
	run_id: str
	total_agents: int
	success_rate: float
	total_cost_usd: float
	total_duration_s: float
	file_hotspots: list[tuple[str, int]]  # (file, touch_count)
	error_patterns: list[tuple[str, int]]  # (pattern, occurrence_count)
	wasted_agents: list[str]  # agent names that produced no useful output
	recommendations: list[str]


@dataclass
class HistoryAnalysis:
	runs_analyzed: int
	overall_success_rate: float
	cost_trend: list[float]
	recurring_failures: list[tuple[str, int]]
	improvement_velocity: float
	top_recommendations: list[str]


def _extract_errors(output_tail: str) -> list[str]:
	"""Extract error pattern names from output text."""
	found: list[str] = []
	for pattern, name in ERROR_PATTERNS_RE:
		if pattern.search(output_tail):
			found.append(name)
	return found


def _compute_file_hotspots(traces: list[dict]) -> list[tuple[str, int]]:
	"""Count files touched by multiple agents."""
	counter: Counter[str] = Counter()
	for trace in traces:
		files = trace.get("files_changed", [])
		if isinstance(files, str):
			files = json.loads(files)
		for f in files:
			counter[f] += 1
	return [(f, count) for f, count in counter.most_common() if count > 1]


def _find_wasted_agents(traces: list[dict]) -> list[str]:
	"""Agents with non-zero exit code and no files changed."""
	wasted: list[str] = []
	for trace in traces:
		files = trace.get("files_changed", [])
		if isinstance(files, str):
			files = json.loads(files)
		exit_code = trace.get("exit_code")
		if exit_code is not None and exit_code != 0 and not files:
			wasted.append(trace["agent_name"])
	return wasted


def _generate_recommendations(
	file_hotspots: list[tuple[str, int]],
	error_patterns: list[tuple[str, int]],
	wasted_agents: list[str],
	success_rate: float,
) -> list[str]:
	"""Generate actionable recommendations from analysis patterns."""
	recs: list[str] = []
	for f, count in file_hotspots:
		recs.append(f"File {f} was touched by {count} agents -- serialize work on it")
	if wasted_agents:
		recs.append(
			f"{len(wasted_agents)} agent(s) produced no useful output -- "
			"review task decomposition for clarity"
		)
	for pattern, count in error_patterns:
		if count >= 3:
			recs.append(f"'{pattern}' occurred {count} times -- investigate root cause")
	if success_rate < 0.5:
		recs.append("Success rate below 50% -- consider smaller, more focused tasks")
	return recs


class TraceAnalyzer:
	"""Review agent traces and surface patterns."""

	def __init__(self, db: Database, project_path: Path):
		self._db = db
		self._project_path = project_path

	async def analyze_run(self, run_id: str) -> RunAnalysis:
		"""Analyze traces from a single run."""
		traces = self._db.get_agent_traces(run_id=run_id)

		if not traces:
			return RunAnalysis(
				run_id=run_id,
				total_agents=0,
				success_rate=0.0,
				total_cost_usd=0.0,
				total_duration_s=0.0,
				file_hotspots=[],
				error_patterns=[],
				wasted_agents=[],
				recommendations=[],
			)

		total = len(traces)
		successes = sum(1 for t in traces if t.get("exit_code") == 0)
		success_rate = successes / total if total else 0.0
		total_cost = sum(t.get("cost_usd", 0.0) for t in traces)
		total_duration = sum(t.get("duration_s", 0.0) for t in traces)

		# Error pattern aggregation
		error_counter: Counter[str] = Counter()
		for trace in traces:
			for err in _extract_errors(trace.get("output_tail", "")):
				error_counter[err] += 1
		error_patterns = error_counter.most_common()

		file_hotspots = _compute_file_hotspots(traces)
		wasted_agents = _find_wasted_agents(traces)
		recommendations = _generate_recommendations(
			file_hotspots, error_patterns, wasted_agents, success_rate,
		)

		return RunAnalysis(
			run_id=run_id,
			total_agents=total,
			success_rate=success_rate,
			total_cost_usd=total_cost,
			total_duration_s=total_duration,
			file_hotspots=file_hotspots,
			error_patterns=error_patterns,
			wasted_agents=wasted_agents,
			recommendations=recommendations,
		)

	async def analyze_history(self, last_n_runs: int = 10) -> HistoryAnalysis:
		"""Analyze patterns across multiple runs."""
		all_traces = self._db.get_agent_traces(limit=last_n_runs * 50)

		run_ids: list[str] = []
		seen: set[str] = set()
		for t in all_traces:
			rid = t["run_id"]
			if rid not in seen:
				seen.add(rid)
				run_ids.append(rid)
		run_ids = run_ids[:last_n_runs]

		if not run_ids:
			return HistoryAnalysis(
				runs_analyzed=0,
				overall_success_rate=0.0,
				cost_trend=[],
				recurring_failures=[],
				improvement_velocity=0.0,
				top_recommendations=[],
			)

		run_analyses: list[RunAnalysis] = []
		for rid in run_ids:
			analysis = await self.analyze_run(rid)
			run_analyses.append(analysis)

		total_agents = sum(a.total_agents for a in run_analyses)
		total_successes = sum(
			int(a.success_rate * a.total_agents) for a in run_analyses
		)
		overall_success_rate = total_successes / total_agents if total_agents else 0.0
		cost_trend = [a.total_cost_usd for a in run_analyses]

		# Aggregate error patterns across runs
		failure_counter: Counter[str] = Counter()
		for a in run_analyses:
			for pattern, count in a.error_patterns:
				failure_counter[pattern] += count
		recurring_failures = failure_counter.most_common()

		# Improvement velocity: change in success rate from first to last run
		if len(run_analyses) >= 2:
			improvement_velocity = (
				run_analyses[-1].success_rate - run_analyses[0].success_rate
			)
		else:
			improvement_velocity = 0.0

		# Top recommendations from most recent runs
		rec_counter: Counter[str] = Counter()
		for a in run_analyses:
			for rec in a.recommendations:
				rec_counter[rec] += 1
		top_recommendations = [rec for rec, _ in rec_counter.most_common(5)]

		return HistoryAnalysis(
			runs_analyzed=len(run_analyses),
			overall_success_rate=overall_success_rate,
			cost_trend=cost_trend,
			recurring_failures=recurring_failures,
			improvement_velocity=improvement_velocity,
			top_recommendations=top_recommendations,
		)

	async def generate_report(self, analysis: RunAnalysis | HistoryAnalysis) -> str:
		"""Generate structured markdown report."""
		if isinstance(analysis, RunAnalysis):
			return self._run_report(analysis)
		return self._history_report(analysis)

	def _run_report(self, a: RunAnalysis) -> str:
		lines = [
			f"# Trace Review: {a.run_id}",
			"",
			f"**Agents**: {a.total_agents} | "
			f"**Success rate**: {a.success_rate:.0%} | "
			f"**Cost**: ${a.total_cost_usd:.2f} | "
			f"**Duration**: {a.total_duration_s:.0f}s",
			"",
		]
		if a.file_hotspots:
			lines.append("## File Hotspots")
			lines.append("")
			lines.append("| File | Agents |")
			lines.append("|------|--------|")
			for f, count in a.file_hotspots:
				lines.append(f"| {f} | {count} |")
			lines.append("")
		if a.error_patterns:
			lines.append("## Error Patterns")
			lines.append("")
			lines.append("| Pattern | Count |")
			lines.append("|---------|-------|")
			for pattern, count in a.error_patterns:
				lines.append(f"| {pattern} | {count} |")
			lines.append("")
		if a.wasted_agents:
			lines.append("## Wasted Agents")
			lines.append("")
			for name in a.wasted_agents:
				lines.append(f"- {name}")
			lines.append("")
		if a.recommendations:
			lines.append("## Recommendations")
			lines.append("")
			for rec in a.recommendations:
				lines.append(f"- {rec}")
			lines.append("")
		return "\n".join(lines)

	def _history_report(self, a: HistoryAnalysis) -> str:
		lines = [
			"# Trace Review: History",
			"",
			f"**Runs analyzed**: {a.runs_analyzed} | "
			f"**Overall success rate**: {a.overall_success_rate:.0%} | "
			f"**Improvement velocity**: {a.improvement_velocity:+.0%}",
			"",
		]
		if a.cost_trend:
			lines.append("## Cost Trend")
			lines.append("")
			for i, cost in enumerate(a.cost_trend):
				lines.append(f"- Run {i + 1}: ${cost:.2f}")
			lines.append("")
		if a.recurring_failures:
			lines.append("## Recurring Failures")
			lines.append("")
			lines.append("| Pattern | Total Count |")
			lines.append("|---------|-------------|")
			for pattern, count in a.recurring_failures:
				lines.append(f"| {pattern} | {count} |")
			lines.append("")
		if a.top_recommendations:
			lines.append("## Top Recommendations")
			lines.append("")
			for rec in a.top_recommendations:
				lines.append(f"- {rec}")
			lines.append("")
		return "\n".join(lines)

	async def llm_review_traces(self, run_id: str) -> str:
		"""Build an LLM prompt for deeper trace analysis. Returns the formatted prompt."""
		traces = self._db.get_agent_traces(run_id=run_id)
		if not traces:
			return "No traces found for this run."

		summaries: list[str] = []
		for t in traces:
			files = t.get("files_changed", [])
			if isinstance(files, str):
				files = json.loads(files)
			summary = (
				f"Agent: {t['agent_name']}\n"
				f"Task: {t['task_title']}\n"
				f"Duration: {t['duration_s']:.0f}s | Exit: {t['exit_code']} | "
				f"Cost: ${t['cost_usd']:.2f}\n"
				f"Files: {', '.join(files) if files else 'none'}\n"
				f"Output tail:\n{t['output_tail'][-500:]}\n"
			)
			summaries.append(summary)

		prompt = (
			"Review these agent execution traces from an autonomous dev swarm.\n"
			"Identify: coordination failures, wasted work, recurring errors, "
			"agents that struggled unnecessarily, patterns the planner should "
			"learn from. Be specific with file names and error messages.\n\n"
			+ "\n---\n".join(summaries)
		)
		return prompt
