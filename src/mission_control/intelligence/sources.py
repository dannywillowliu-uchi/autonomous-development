"""Async scanners for external intelligence sources."""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from mission_control.intelligence.models import Finding

logger = logging.getLogger(__name__)

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
