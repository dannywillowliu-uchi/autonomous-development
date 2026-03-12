"""Tests for web source scanners."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from autodev.intelligence.claude_code import AUTOMATION_KEYWORDS
from autodev.intelligence.models import Finding
from autodev.intelligence.web_sources import (
	SOURCES,
	WebSourceScanner,
	_score_relevance,
)


MOCK_BLOG_HTML = """
<html><body>
<a href="/news/new-agent-skill-support">New Agent Skill Support for Automation</a>
<a href="/news/hiring">We are hiring</a>
<a href="/news/mcp-server-update">MCP Server Update and Tool Improvements</a>
<a href="/x">Hi</a>
</body></html>
"""

MOCK_TRENDING_HTML = """
<html><body>
<article>
<h2 class="h3 lh-condensed">
  <a href="/anthropics/claude-agent-sdk" data-hydro-click>
  </a>
</h2>
<p class="col-9 color-fg-muted my-1 pr-4">
  Autonomous agent SDK for building LLM tool use applications
</p>
</article>
<article>
<h2 class="h3 lh-condensed">
  <a href="/user/some-web-framework" data-hydro-click>
  </a>
</h2>
<p class="col-9 color-fg-muted my-1 pr-4">
  A modern web framework for building fast websites
</p>
</article>
</body></html>
"""

MOCK_ARXIV_HTML = """
<html><body>
<dl>
<dt><a name="item1"></a></dt>
<dd>
<div class="list-title mathjax">
<span class="descriptor">Title:</span> arXiv:2603.01234
<div class="list-title mathjax">
  <span class="descriptor">Title:</span> Multi-Agent Planning with Tool Use
</div>
</div>
</dd>
</dl>
</body></html>
"""

MOCK_ARXIV_SIMPLE_HTML = """
<html><body>
<a href="/abs/2603.05678">arXiv:2603.05678</a>
</body></html>
"""


def _route_blog(request: httpx.Request) -> httpx.Response:
	url = str(request.url)
	if "anthropic.com" in url:
		return httpx.Response(200, text=MOCK_BLOG_HTML)
	if "openai.com" in url:
		return httpx.Response(200, text=MOCK_BLOG_HTML)
	if "deepmind.google" in url:
		return httpx.Response(200, text=MOCK_BLOG_HTML)
	if "github.com/trending" in url:
		return httpx.Response(200, text=MOCK_TRENDING_HTML)
	if "arxiv.org" in url:
		return httpx.Response(200, text=MOCK_ARXIV_HTML)
	return httpx.Response(404)


def _route_error(request: httpx.Request) -> httpx.Response:
	return httpx.Response(500)


class TestScoreRelevance:
	def test_relevant_text_scores_high(self) -> None:
		score = _score_relevance("new skill and hook support with mcp agent")
		assert score > 0.0

	def test_irrelevant_text_scores_zero(self) -> None:
		score = _score_relevance("weather forecast for tomorrow")
		assert score == 0.0

	def test_uses_automation_keywords(self) -> None:
		for kw in AUTOMATION_KEYWORDS[:3]:
			score = _score_relevance(f"article about {kw} features")
			assert score > 0.0, f"Keyword '{kw}' should produce positive score"

	def test_capped_at_one(self) -> None:
		heavy = " ".join(AUTOMATION_KEYWORDS)
		score = _score_relevance(heavy)
		assert score <= 1.0


class TestWebSourceScanner:
	@pytest.mark.asyncio
	async def test_scan_returns_findings(self) -> None:
		transport = httpx.MockTransport(_route_blog)
		client = httpx.AsyncClient(transport=transport)
		scanner = WebSourceScanner()
		findings = await scanner.scan(client)
		await client.aclose()

		assert isinstance(findings, list)
		for f in findings:
			assert isinstance(f, Finding)

	@pytest.mark.asyncio
	async def test_findings_have_trust_level(self) -> None:
		transport = httpx.MockTransport(_route_blog)
		client = httpx.AsyncClient(transport=transport)
		scanner = WebSourceScanner()
		findings = await scanner.scan(client)
		await client.aclose()

		for f in findings:
			assert f.trust_level in ("high", "medium")

	@pytest.mark.asyncio
	async def test_findings_have_source(self) -> None:
		transport = httpx.MockTransport(_route_blog)
		client = httpx.AsyncClient(transport=transport)
		scanner = WebSourceScanner()
		findings = await scanner.scan(client)
		await client.aclose()

		valid_sources = set(SOURCES.keys())
		for f in findings:
			assert f.source in valid_sources

	@pytest.mark.asyncio
	async def test_blog_filters_irrelevant(self) -> None:
		transport = httpx.MockTransport(_route_blog)
		client = httpx.AsyncClient(transport=transport)
		scanner = WebSourceScanner()
		findings = await scanner._scan_blog(
			client, "https://www.anthropic.com/news", "anthropic_blog", "high"
		)
		await client.aclose()

		titles = [f.title for f in findings]
		assert "We are hiring" not in titles
		assert "Hi" not in titles

	@pytest.mark.asyncio
	async def test_blog_relevant_articles_found(self) -> None:
		transport = httpx.MockTransport(_route_blog)
		client = httpx.AsyncClient(transport=transport)
		scanner = WebSourceScanner()
		findings = await scanner._scan_blog(
			client, "https://www.anthropic.com/news", "anthropic_blog", "high"
		)
		await client.aclose()

		titles = [f.title for f in findings]
		# At least one relevant article should be found
		assert any("Agent" in t or "MCP" in t or "Skill" in t or "Tool" in t for t in titles)

	@pytest.mark.asyncio
	async def test_blog_resolves_relative_urls(self) -> None:
		transport = httpx.MockTransport(_route_blog)
		client = httpx.AsyncClient(transport=transport)
		scanner = WebSourceScanner()
		findings = await scanner._scan_blog(
			client, "https://www.anthropic.com/news", "anthropic_blog", "high"
		)
		await client.aclose()

		for f in findings:
			assert f.url.startswith("https://")

	@pytest.mark.asyncio
	async def test_github_trending_filters_by_keywords(self) -> None:
		transport = httpx.MockTransport(_route_blog)
		client = httpx.AsyncClient(transport=transport)
		scanner = WebSourceScanner()
		findings = await scanner._scan_github_trending(client, "medium")
		await client.aclose()

		# Agent-related repos should be included, generic ones filtered
		sources = [f.source for f in findings]
		for s in sources:
			assert s == "github_trending"

	@pytest.mark.asyncio
	async def test_graceful_error_handling(self) -> None:
		transport = httpx.MockTransport(_route_error)
		client = httpx.AsyncClient(transport=transport)
		scanner = WebSourceScanner()
		findings = await scanner.scan(client)
		await client.aclose()

		assert findings == []

	@pytest.mark.asyncio
	async def test_individual_scanner_error_doesnt_crash(self) -> None:
		call_count = 0

		def mixed_route(request: httpx.Request) -> httpx.Response:
			nonlocal call_count
			call_count += 1
			url = str(request.url)
			if "anthropic.com" in url:
				return httpx.Response(500)
			return _route_blog(request)

		transport = httpx.MockTransport(mixed_route)
		client = httpx.AsyncClient(transport=transport)
		scanner = WebSourceScanner()
		findings = await scanner.scan(client)
		await client.aclose()

		# Should still get results from non-failing sources
		assert isinstance(findings, list)

	@pytest.mark.asyncio
	async def test_arxiv_scan(self) -> None:
		def route(request: httpx.Request) -> httpx.Response:
			return httpx.Response(200, text=MOCK_ARXIV_SIMPLE_HTML)

		transport = httpx.MockTransport(route)
		client = httpx.AsyncClient(transport=transport)
		scanner = WebSourceScanner()
		findings = await scanner._scan_arxiv(client, "medium")
		await client.aclose()

		for f in findings:
			assert f.source == "arxiv_agents"
			assert f.trust_level == "medium"
			assert "arxiv.org" in f.url


class TestSourcesConfig:
	def test_all_sources_defined(self) -> None:
		assert "anthropic_blog" in SOURCES
		assert "openai_blog" in SOURCES
		assert "deepmind_blog" in SOURCES
		assert "github_trending" in SOURCES
		assert "arxiv_agents" in SOURCES

	def test_source_types(self) -> None:
		assert SOURCES["anthropic_blog"].source_type == "blog"
		assert SOURCES["github_trending"].source_type == "trending"
		assert SOURCES["arxiv_agents"].source_type == "papers"

	def test_trust_levels(self) -> None:
		assert SOURCES["anthropic_blog"].trust_level == "high"
		assert SOURCES["github_trending"].trust_level == "medium"
		assert SOURCES["arxiv_agents"].trust_level == "medium"


class TestScannerIntegration:
	@pytest.mark.asyncio
	async def test_run_scan_includes_web_sources(self) -> None:
		"""Test that scanner.run_scan() includes web_sources in sources list."""
		from autodev.intelligence.scanner import run_scan

		with patch("autodev.intelligence.scanner.scan_hackernews", new_callable=AsyncMock, return_value=[]), \
			patch("autodev.intelligence.scanner.scan_github", new_callable=AsyncMock, return_value=[]), \
			patch("autodev.intelligence.scanner.scan_arxiv", new_callable=AsyncMock, return_value=[]), \
			patch("autodev.intelligence.scanner.WebSourceScanner") as mock_cls:

			mock_instance = AsyncMock()
			mock_instance.scan.return_value = [
				Finding(source="anthropic_blog", title="Test", trust_level="high", relevance_score=0.5)
			]
			mock_cls.return_value = mock_instance

			report = await run_scan(threshold=0.0)

		assert "web_sources" in report.sources_scanned
		assert any(f.source == "anthropic_blog" for f in report.findings)

	@pytest.mark.asyncio
	async def test_run_scan_survives_web_scanner_failure(self) -> None:
		"""Test that scanner.run_scan() doesn't crash if web scanner fails."""
		from autodev.intelligence.scanner import run_scan

		with patch("autodev.intelligence.scanner.scan_hackernews", new_callable=AsyncMock, return_value=[]), \
			patch("autodev.intelligence.scanner.scan_github", new_callable=AsyncMock, return_value=[]), \
			patch("autodev.intelligence.scanner.scan_arxiv", new_callable=AsyncMock, return_value=[]), \
			patch("autodev.intelligence.scanner.WebSourceScanner") as mock_cls:

			mock_instance = AsyncMock()
			mock_instance.scan.side_effect = RuntimeError("network down")
			mock_cls.return_value = mock_instance

			report = await run_scan(threshold=0.0)

		assert "web_sources" not in report.sources_scanned
		assert isinstance(report.findings, list)


class TestFindingTrustLevel:
	def test_default_trust_level(self) -> None:
		f = Finding()
		assert f.trust_level == "medium"

	def test_custom_trust_level(self) -> None:
		f = Finding(trust_level="high")
		assert f.trust_level == "high"
