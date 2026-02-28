"""Orchestrates intelligence scanning across all sources."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx

from mission_control.intelligence.evaluator import evaluate_findings, generate_proposals
from mission_control.intelligence.models import AdaptationProposal, Finding
from mission_control.intelligence.sources import scan_arxiv, scan_github, scan_hackernews


@dataclass
class IntelReport:
	"""Aggregated results from a full intelligence scan."""

	findings: list[Finding] = field(default_factory=list)
	proposals: list[AdaptationProposal] = field(default_factory=list)
	timestamp: str = ""
	sources_scanned: list[str] = field(default_factory=list)
	scan_duration_seconds: float = 0.0


async def run_scan(threshold: float = 0.3) -> IntelReport:
	"""Run all intelligence scanners concurrently and return an aggregated report."""
	from datetime import datetime, timezone

	start = time.monotonic()
	async with httpx.AsyncClient(timeout=30.0) as client:
		hn_results, gh_results, arxiv_results = await asyncio.gather(
			scan_hackernews(client),
			scan_github(client),
			scan_arxiv(client),
			return_exceptions=True,
		)

	# Collect findings, skipping any scanner that raised
	all_findings: list[Finding] = []
	sources: list[str] = []
	for name, result in [("hackernews", hn_results), ("github", gh_results), ("arxiv", arxiv_results)]:
		if isinstance(result, BaseException):
			continue
		all_findings.extend(result)
		sources.append(name)

	evaluated = evaluate_findings(all_findings)
	proposals = generate_proposals(evaluated, threshold=threshold)

	return IntelReport(
		findings=evaluated,
		proposals=proposals,
		timestamp=datetime.now(timezone.utc).isoformat(),
		sources_scanned=sources,
		scan_duration_seconds=round(time.monotonic() - start, 2),
	)


class IntelScanner:
	"""Thin OOP wrapper around run_scan."""

	def __init__(self, threshold: float = 0.3) -> None:
		self.threshold = threshold

	async def scan(self) -> IntelReport:
		return await run_scan(threshold=self.threshold)
