"""Tests for incremental scanning with TTL caching in sources.py."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from mission_control.intelligence.models import Finding
from mission_control.intelligence.sources import (
	IncrementalScanner,
	ScanCache,
	_deserialize_cache,
	_finding_key,
	_serialize_cache,
	scan_incremental,
)


def _mock_hn_response() -> httpx.Response:
	data = {
		"hits": [
			{
				"objectID": "1001",
				"title": "MCP server for coding agents",
				"url": "https://example.com/mcp",
				"points": 100,
				"num_comments": 30,
				"created_at": "2026-01-15T12:00:00Z",
			},
		],
	}
	return httpx.Response(200, json=data)


def _mock_gh_response() -> httpx.Response:
	data = {
		"items": [
			{
				"full_name": "example/agent-toolkit",
				"html_url": "https://github.com/example/agent-toolkit",
				"description": "Agent toolkit",
				"stargazers_count": 200,
				"pushed_at": "2026-01-15T08:00:00Z",
			},
		],
	}
	return httpx.Response(200, json=data)


def _mock_arxiv_response() -> httpx.Response:
	xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Multi-Agent Systems Survey</title>
    <summary>Survey of multi-agent systems.</summary>
    <published>2026-01-10T00:00:00Z</published>
    <id>http://arxiv.org/abs/2601.00001v1</id>
  </entry>
</feed>"""
	return httpx.Response(200, text=xml)


def _route_mock(request: httpx.Request) -> httpx.Response:
	url = str(request.url)
	if "hn.algolia.com" in url:
		return _mock_hn_response()
	if "api.github.com" in url:
		return _mock_gh_response()
	if "arxiv.org" in url:
		return _mock_arxiv_response()
	return httpx.Response(404)


def _make_finding(title: str = "Test", url: str = "https://example.com", source: str = "test") -> Finding:
	return Finding(source=source, title=title, url=url, summary="summary")


class TestCacheSerialization:
	"""Cache save/load roundtrip tests."""

	def test_roundtrip_empty_cache(self, tmp_path: Path) -> None:
		cache = ScanCache()
		path = tmp_path / "cache.json"

		scanner = IncrementalScanner()
		scanner.cache = cache
		scanner.save_cache(path)

		scanner2 = IncrementalScanner()
		scanner2.load_cache(path)

		assert scanner2.cache.last_scan_time == 0.0
		assert scanner2.cache.findings == []
		assert scanner2.cache.source_timestamps == {}

	def test_roundtrip_with_findings(self, tmp_path: Path) -> None:
		finding = _make_finding("My Title", "https://example.com/1", "hackernews")
		cache = ScanCache(
			last_scan_time=1000.0,
			findings=[finding],
			source_timestamps={"hackernews": 1000.0, "github": 900.0},
		)
		path = tmp_path / "subdir" / "cache.json"

		scanner = IncrementalScanner()
		scanner.cache = cache
		scanner.save_cache(path)

		scanner2 = IncrementalScanner()
		scanner2.load_cache(path)

		assert scanner2.cache.last_scan_time == 1000.0
		assert len(scanner2.cache.findings) == 1
		assert scanner2.cache.findings[0].title == "My Title"
		assert scanner2.cache.findings[0].url == "https://example.com/1"
		assert scanner2.cache.source_timestamps["hackernews"] == 1000.0
		assert scanner2.cache.source_timestamps["github"] == 900.0

	def test_load_missing_file_is_noop(self, tmp_path: Path) -> None:
		scanner = IncrementalScanner()
		scanner.load_cache(tmp_path / "nonexistent.json")
		assert scanner.cache.findings == []
		assert scanner.cache.last_scan_time == 0.0

	def test_serialize_deserialize_functions(self) -> None:
		finding = _make_finding("F1", "https://f1.com")
		cache = ScanCache(last_scan_time=42.0, findings=[finding], source_timestamps={"x": 42.0})
		data = _serialize_cache(cache)
		restored = _deserialize_cache(data)
		assert restored.last_scan_time == 42.0
		assert len(restored.findings) == 1
		assert restored.findings[0].title == "F1"
		assert restored.source_timestamps == {"x": 42.0}

	def test_cache_creates_parent_dirs(self, tmp_path: Path) -> None:
		path = tmp_path / "a" / "b" / "c" / "cache.json"
		scanner = IncrementalScanner()
		scanner.save_cache(path)
		assert path.exists()


class TestTTLExpiry:
	"""TTL expiry logic tests."""

	def test_all_sources_expired_when_no_cache(self) -> None:
		scanner = IncrementalScanner(ttl_seconds=3600.0)
		now = time.time()
		expired = scanner._expired_sources(now)
		assert "hackernews" in expired
		assert "github" in expired
		assert "arxiv" in expired

	def test_source_not_expired_within_ttl(self) -> None:
		scanner = IncrementalScanner(ttl_seconds=3600.0)
		now = time.time()
		scanner.cache.source_timestamps = {
			"hackernews": now - 100,
			"github": now - 100,
			"arxiv": now - 100,
		}
		expired = scanner._expired_sources(now)
		assert expired == []

	def test_only_expired_sources_returned(self) -> None:
		scanner = IncrementalScanner(ttl_seconds=3600.0)
		now = time.time()
		scanner.cache.source_timestamps = {
			"hackernews": now - 100,
			"github": now - 5000,
			"arxiv": now - 100,
		}
		expired = scanner._expired_sources(now)
		assert expired == ["github"]

	def test_exact_ttl_boundary_triggers_expiry(self) -> None:
		scanner = IncrementalScanner(ttl_seconds=3600.0)
		now = time.time()
		scanner.cache.source_timestamps = {"hackernews": now - 3600.0}
		expired = scanner._expired_sources(now)
		assert "hackernews" in expired


class TestDeduplication:
	"""Finding deduplication tests."""

	def test_finding_key_uses_title_and_url(self) -> None:
		f = _make_finding("Title A", "https://a.com")
		assert _finding_key(f) == ("Title A", "https://a.com")

	@pytest.mark.asyncio
	async def test_duplicate_findings_merged(self) -> None:
		scanner = IncrementalScanner(ttl_seconds=0.0)
		scanner.cache.findings = [
			_make_finding("MCP server for coding agents", "https://example.com/mcp", "hackernews"),
		]
		transport = httpx.MockTransport(_route_mock)
		client = httpx.AsyncClient(transport=transport)
		results = await scanner.scan(client=client)
		await client.aclose()

		titles_urls = [(f.title, f.url) for f in results]
		assert len(titles_urls) == len(set(titles_urls))

	@pytest.mark.asyncio
	async def test_new_findings_take_precedence(self) -> None:
		scanner = IncrementalScanner(ttl_seconds=0.0)
		old = _make_finding("MCP server for coding agents", "https://example.com/mcp", "hackernews")
		old.summary = "old summary"
		scanner.cache.findings = [old]

		transport = httpx.MockTransport(_route_mock)
		client = httpx.AsyncClient(transport=transport)
		results = await scanner.scan(client=client)
		await client.aclose()

		matched = [f for f in results if f.title == "MCP server for coding agents"]
		assert len(matched) == 1
		# New finding overwrites old
		assert matched[0].summary != "old summary"


class TestIncrementalMerge:
	"""Incremental merge behavior tests."""

	@pytest.mark.asyncio
	async def test_fresh_scan_fetches_all_sources(self) -> None:
		scanner = IncrementalScanner(ttl_seconds=3600.0)
		transport = httpx.MockTransport(_route_mock)
		client = httpx.AsyncClient(transport=transport)
		results = await scanner.scan(client=client)
		await client.aclose()

		sources_in_results = {f.source for f in results}
		assert "hackernews" in sources_in_results
		assert "github" in sources_in_results
		assert "arxiv" in sources_in_results
		assert len(results) >= 3

	@pytest.mark.asyncio
	async def test_cached_findings_preserved_for_fresh_sources(self) -> None:
		scanner = IncrementalScanner(ttl_seconds=3600.0)
		now = time.time()
		# HN and arxiv are fresh, github is expired
		scanner.cache.source_timestamps = {
			"hackernews": now,
			"github": now - 5000,
			"arxiv": now,
		}
		cached_hn = _make_finding("Cached HN", "https://cached-hn.com", "hackernews")
		scanner.cache.findings = [cached_hn]

		transport = httpx.MockTransport(_route_mock)
		client = httpx.AsyncClient(transport=transport)

		with patch("mission_control.intelligence.sources.time.time", return_value=now):
			results = await scanner.scan(client=client)
		await client.aclose()

		# Cached HN finding should still be present
		assert any(f.title == "Cached HN" for f in results)
		# Fresh github finding should also be present
		assert any(f.source == "github" for f in results)

	@pytest.mark.asyncio
	async def test_no_scan_when_all_fresh(self) -> None:
		scanner = IncrementalScanner(ttl_seconds=3600.0)
		now = time.time()
		scanner.cache.source_timestamps = {
			"hackernews": now,
			"github": now,
			"arxiv": now,
		}
		cached = _make_finding("Cached", "https://cached.com", "test")
		scanner.cache.findings = [cached]

		with patch("mission_control.intelligence.sources.time.time", return_value=now):
			results = await scanner.scan()

		assert len(results) == 1
		assert results[0].title == "Cached"

	@pytest.mark.asyncio
	async def test_timestamps_updated_after_scan(self) -> None:
		scanner = IncrementalScanner(ttl_seconds=0.0)
		transport = httpx.MockTransport(_route_mock)
		client = httpx.AsyncClient(transport=transport)

		before = time.time()
		await scanner.scan(client=client)
		await client.aclose()

		assert scanner.cache.source_timestamps["hackernews"] >= before
		assert scanner.cache.source_timestamps["github"] >= before
		assert scanner.cache.source_timestamps["arxiv"] >= before
		assert scanner.cache.last_scan_time >= before


class TestScanIncremental:
	"""Tests for the scan_incremental convenience function."""

	@pytest.mark.asyncio
	async def test_fresh_scan_no_cache_file(self, tmp_path: Path) -> None:
		cache_path = tmp_path / ".cache" / "intel_scan.json"
		transport = httpx.MockTransport(_route_mock)
		client = httpx.AsyncClient(transport=transport)

		results = await scan_incremental(cache_path, ttl_seconds=3600.0, client=client)
		await client.aclose()

		assert len(results) >= 3
		assert cache_path.exists()
		data = json.loads(cache_path.read_text())
		assert len(data["findings"]) >= 3

	@pytest.mark.asyncio
	async def test_second_scan_uses_cache(self, tmp_path: Path) -> None:
		cache_path = tmp_path / "cache.json"
		transport = httpx.MockTransport(_route_mock)

		client1 = httpx.AsyncClient(transport=transport)
		results1 = await scan_incremental(cache_path, ttl_seconds=99999.0, client=client1)
		await client1.aclose()

		# Second scan with high TTL should return same results without hitting network
		client2 = httpx.AsyncClient(transport=transport)
		results2 = await scan_incremental(cache_path, ttl_seconds=99999.0, client=client2)
		await client2.aclose()

		assert len(results1) == len(results2)
		titles1 = {f.title for f in results1}
		titles2 = {f.title for f in results2}
		assert titles1 == titles2
