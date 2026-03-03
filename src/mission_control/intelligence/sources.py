"""Async scanners for external intelligence sources."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

import httpx

from mission_control.intelligence.models import Finding

logger = logging.getLogger(__name__)


@dataclass
class ScanCache:
	"""Cached state from a previous incremental scan."""

	last_scan_time: float = 0.0
	findings: list[Finding] = field(default_factory=list)
	source_timestamps: dict[str, float] = field(default_factory=dict)


SourceScanner = Callable[[httpx.AsyncClient], Coroutine[Any, Any, list[Finding]]]

# Registry mapping source name to its scanner function
SOURCE_SCANNERS: dict[str, SourceScanner] = {}


def _register_scanner(name: str) -> Callable[[SourceScanner], SourceScanner]:
	def decorator(fn: SourceScanner) -> SourceScanner:
		SOURCE_SCANNERS[name] = fn
		return fn
	return decorator


def _finding_key(f: Finding) -> tuple[str, str]:
	"""Deduplication key for a finding: (title, url)."""
	return (f.title, f.url)


def _serialize_cache(cache: ScanCache) -> dict[str, Any]:
	return {
		"last_scan_time": cache.last_scan_time,
		"findings": [asdict(f) for f in cache.findings],
		"source_timestamps": cache.source_timestamps,
	}


def _deserialize_cache(data: dict[str, Any]) -> ScanCache:
	findings = [Finding(**f) for f in data.get("findings", [])]
	return ScanCache(
		last_scan_time=data.get("last_scan_time", 0.0),
		findings=findings,
		source_timestamps=data.get("source_timestamps", {}),
	)


class IncrementalScanner:
	"""Wraps source scanners with TTL-based incremental scanning."""

	def __init__(self, ttl_seconds: float = 21600.0) -> None:
		self.ttl_seconds = ttl_seconds
		self.cache = ScanCache()

	def save_cache(self, path: Path) -> None:
		"""Persist cache to a JSON file."""
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(_serialize_cache(self.cache), indent=2))

	def load_cache(self, path: Path) -> None:
		"""Load cache from a JSON file. No-op if file doesn't exist."""
		if not path.exists():
			return
		data = json.loads(path.read_text())
		self.cache = _deserialize_cache(data)

	def _expired_sources(self, now: float) -> list[str]:
		"""Return source names whose TTL has expired."""
		expired = []
		for name in SOURCE_SCANNERS:
			last = self.cache.source_timestamps.get(name, 0.0)
			if (now - last) >= self.ttl_seconds:
				expired.append(name)
		return expired

	async def scan(self, client: httpx.AsyncClient | None = None) -> list[Finding]:
		"""Run incremental scan: only fetch from sources whose TTL expired."""
		now = time.time()
		expired = self._expired_sources(now)

		if not expired:
			return list(self.cache.findings)

		own_client = client is None
		if own_client:
			client = httpx.AsyncClient(timeout=30.0)

		try:
			tasks = {name: SOURCE_SCANNERS[name](client) for name in expired}
			results = await asyncio.gather(*tasks.values(), return_exceptions=True)

			new_findings: list[Finding] = []
			for name, result in zip(tasks.keys(), results):
				if isinstance(result, BaseException):
					logger.warning("Scanner %s failed: %s", name, result)
					continue
				new_findings.extend(result)
				self.cache.source_timestamps[name] = now
		finally:
			if own_client:
				await client.aclose()

		# Merge: keep existing findings from non-expired sources, add new ones, deduplicate
		seen: set[tuple[str, str]] = set()
		merged: list[Finding] = []
		for f in new_findings:
			key = _finding_key(f)
			if key not in seen:
				seen.add(key)
				merged.append(f)
		for f in self.cache.findings:
			key = _finding_key(f)
			if key not in seen:
				seen.add(key)
				merged.append(f)

		self.cache.findings = merged
		self.cache.last_scan_time = now
		return merged

HN_KEYWORDS = [
	"AI agent",
	"MCP server",
	"Claude Code",
	"autonomous coding",
	"tool use LLM",
	"agentic framework",
]

GITHUB_QUERY = "AI agent OR MCP server OR autonomous coding"
ARXIV_QUERY = "cat:cs.AI AND (agent OR multi-agent OR autonomous)"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


@_register_scanner("hackernews")
async def scan_hackernews(client: httpx.AsyncClient | None = None) -> list[Finding]:
	"""Scan Hacker News Algolia API for relevant stories."""
	own_client = client is None
	if own_client:
		client = httpx.AsyncClient(timeout=30.0)

	findings: list[Finding] = []
	seen_ids: set[str] = set()
	try:
		for keyword in HN_KEYWORDS:
			try:
				resp = await client.get(
					"https://hn.algolia.com/api/v1/search",
					params={"query": keyword, "tags": "story", "hitsPerPage": 10},
				)
				resp.raise_for_status()
				data = resp.json()
				for hit in data.get("hits", []):
					object_id = hit.get("objectID", "")
					if object_id in seen_ids:
						continue
					seen_ids.add(object_id)
					created_at = hit.get("created_at", "")
					findings.append(Finding(
						source="hackernews",
						title=hit.get("title", ""),
						url=hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}",
						summary=f"HN story with {hit.get('points', 0)} points, {hit.get('num_comments', 0)} comments",
						published_at=created_at if created_at else datetime.now(timezone.utc).isoformat(),
						raw_data=hit,
						relevance_score=min(1.0, (hit.get("points", 0) or 0) / 500),
					))
			except httpx.HTTPError as exc:
				logger.warning("HN search failed for keyword %r: %s", keyword, exc)
	finally:
		if own_client:
			await client.aclose()

	return findings


@_register_scanner("github")
async def scan_github(client: httpx.AsyncClient | None = None) -> list[Finding]:
	"""Scan GitHub search API for relevant repositories."""
	own_client = client is None
	if own_client:
		client = httpx.AsyncClient(timeout=30.0)

	findings: list[Finding] = []
	try:
		resp = await client.get(
			"https://api.github.com/search/repositories",
			params={
				"q": f"{GITHUB_QUERY} stars:>50 pushed:>2025-01-01",
				"sort": "updated",
				"order": "desc",
				"per_page": 30,
			},
			headers={"Accept": "application/vnd.github.v3+json"},
		)
		resp.raise_for_status()
		data = resp.json()
		for repo in data.get("items", []):
			pushed_at = repo.get("pushed_at", "")
			findings.append(Finding(
				source="github",
				title=repo.get("full_name", ""),
				url=repo.get("html_url", ""),
				summary=repo.get("description", "") or "",
				published_at=pushed_at if pushed_at else datetime.now(timezone.utc).isoformat(),
				raw_data=repo,
				relevance_score=min(1.0, (repo.get("stargazers_count", 0) or 0) / 10000),
			))
	except httpx.HTTPError as exc:
		logger.warning("GitHub search failed: %s", exc)
	finally:
		if own_client:
			await client.aclose()

	return findings


@_register_scanner("arxiv")
async def scan_arxiv(client: httpx.AsyncClient | None = None) -> list[Finding]:
	"""Scan arXiv API for relevant AI/agent papers."""
	own_client = client is None
	if own_client:
		client = httpx.AsyncClient(timeout=30.0)

	findings: list[Finding] = []
	try:
		await asyncio.sleep(1)  # Rate limit courtesy for arXiv
		resp = await client.get(
			"http://export.arxiv.org/api/query",
			params={
				"search_query": ARXIV_QUERY,
				"start": 0,
				"max_results": 20,
				"sortBy": "submittedDate",
				"sortOrder": "descending",
			},
		)
		resp.raise_for_status()
		root = ET.fromstring(resp.text)
		for entry in root.findall("atom:entry", ARXIV_NS):
			title_el = entry.find("atom:title", ARXIV_NS)
			summary_el = entry.find("atom:summary", ARXIV_NS)
			published_el = entry.find("atom:published", ARXIV_NS)
			link_el = entry.find("atom:id", ARXIV_NS)

			title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
			summary = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
			published = (published_el.text or "").strip() if published_el is not None else ""
			url = (link_el.text or "").strip() if link_el is not None else ""

			findings.append(Finding(
				source="arxiv",
				title=title,
				url=url,
				summary=summary[:300],
				published_at=published if published else datetime.now(timezone.utc).isoformat(),
				raw_data={"title": title, "summary": summary, "url": url},
				relevance_score=0.5,
			))
	except (httpx.HTTPError, ET.ParseError) as exc:
		logger.warning("arXiv search failed: %s", exc)
	finally:
		if own_client:
			await client.aclose()

	return findings


async def scan_incremental(
	cache_path: Path,
	ttl_seconds: float = 21600.0,
	client: httpx.AsyncClient | None = None,
) -> list[Finding]:
	"""Load cache, scan only expired sources, merge results, save cache.

	Returns the full merged list of findings.
	"""
	scanner = IncrementalScanner(ttl_seconds=ttl_seconds)
	scanner.load_cache(cache_path)
	findings = await scanner.scan(client=client)
	scanner.save_cache(cache_path)
	return findings
