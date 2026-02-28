"""Smoke tests for the intelligence subsystem and CLI integration."""

from __future__ import annotations

import json
from dataclasses import asdict
from unittest.mock import patch

import httpx
import pytest

from mission_control.cli import build_parser, cmd_intel
from mission_control.intelligence import IntelReport, run_scan


def _mock_hn_response() -> httpx.Response:
	"""Fake HN Algolia response with one relevant hit."""
	data = {
		"hits": [
			{
				"objectID": "1001",
				"title": "New MCP server for autonomous coding agents",
				"url": "https://example.com/mcp-server",
				"points": 250,
				"num_comments": 80,
				"created_at": "2026-01-15T12:00:00Z",
			},
			{
				"objectID": "1002",
				"title": "Multi-agent coordination framework released",
				"url": "https://example.com/multi-agent",
				"points": 120,
				"num_comments": 40,
				"created_at": "2026-01-14T10:00:00Z",
			},
		],
	}
	return httpx.Response(200, json=data)


def _mock_gh_response() -> httpx.Response:
	"""Fake GitHub search response."""
	data = {
		"items": [
			{
				"full_name": "example/ai-agent-toolkit",
				"html_url": "https://github.com/example/ai-agent-toolkit",
				"description": "Claude Code tool use and MCP integration toolkit",
				"stargazers_count": 500,
				"pushed_at": "2026-01-15T08:00:00Z",
			},
		],
	}
	return httpx.Response(200, json=data)


def _mock_arxiv_response() -> httpx.Response:
	"""Fake arXiv Atom feed response."""
	xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Multi-Agent Autonomous Coding Systems: A Survey</title>
    <summary>We survey recent advances in multi-agent systems for autonomous software development.</summary>
    <published>2026-01-10T00:00:00Z</published>
    <id>http://arxiv.org/abs/2601.00001v1</id>
  </entry>
</feed>"""
	return httpx.Response(200, text=xml)


def _route_mock_response(request: httpx.Request) -> httpx.Response:
	"""Route mock responses based on URL host."""
	url = str(request.url)
	if "hn.algolia.com" in url:
		return _mock_hn_response()
	elif "api.github.com" in url:
		return _mock_gh_response()
	elif "arxiv.org" in url:
		return _mock_arxiv_response()
	return httpx.Response(404)


@pytest.mark.asyncio
async def test_run_scan_returns_report() -> None:
	"""run_scan returns an IntelReport with findings from all sources."""
	transport = httpx.MockTransport(_route_mock_response)
	mock_client = httpx.AsyncClient(transport=transport)

	with patch("mission_control.intelligence.scanner.httpx.AsyncClient") as mock_cls:
		mock_cls.return_value = mock_client
		report = await run_scan(threshold=0.3)

	assert isinstance(report, IntelReport)
	assert len(report.findings) > 0
	assert report.timestamp != ""
	assert report.scan_duration_seconds >= 0
	assert len(report.sources_scanned) > 0

	# Check serialization roundtrip
	data = asdict(report)
	assert "findings" in data
	assert "proposals" in data


@pytest.mark.asyncio
async def test_threshold_filters_proposals() -> None:
	"""Higher threshold should produce fewer or equal proposals."""
	transport = httpx.MockTransport(_route_mock_response)
	mock_client = httpx.AsyncClient(transport=transport)

	with patch("mission_control.intelligence.scanner.httpx.AsyncClient") as mock_cls:
		mock_cls.return_value = mock_client
		report_low = await run_scan(threshold=0.0)

	mock_client2 = httpx.AsyncClient(transport=transport)
	with patch("mission_control.intelligence.scanner.httpx.AsyncClient") as mock_cls:
		mock_cls.return_value = mock_client2
		report_high = await run_scan(threshold=999.0)

	assert len(report_low.proposals) >= len(report_high.proposals)
	assert len(report_high.proposals) == 0


def test_cmd_intel_json(capsys: pytest.CaptureFixture[str]) -> None:
	"""mc intel --json outputs valid JSON with expected keys."""
	transport = httpx.MockTransport(_route_mock_response)
	mock_client = httpx.AsyncClient(transport=transport)

	parser = build_parser()
	args = parser.parse_args(["intel", "--json"])

	with patch("mission_control.intelligence.scanner.httpx.AsyncClient") as mock_cls:
		mock_cls.return_value = mock_client
		result = cmd_intel(args)

	assert result == 0
	captured = capsys.readouterr()
	data = json.loads(captured.out)
	assert "findings" in data
	assert "proposals" in data
	assert "sources_scanned" in data
	assert isinstance(data["findings"], list)
	assert len(data["findings"]) > 0


def test_cmd_intel_table(capsys: pytest.CaptureFixture[str]) -> None:
	"""mc intel (no --json) outputs human-readable table."""
	transport = httpx.MockTransport(_route_mock_response)
	mock_client = httpx.AsyncClient(transport=transport)

	parser = build_parser()
	args = parser.parse_args(["intel"])

	with patch("mission_control.intelligence.scanner.httpx.AsyncClient") as mock_cls:
		mock_cls.return_value = mock_client
		result = cmd_intel(args)

	assert result == 0
	captured = capsys.readouterr()
	assert "Intel Report" in captured.out
	assert "Findings:" in captured.out


def test_cmd_intel_threshold_flag(capsys: pytest.CaptureFixture[str]) -> None:
	"""--threshold flag is passed through to run_scan."""
	transport = httpx.MockTransport(_route_mock_response)
	mock_client = httpx.AsyncClient(transport=transport)

	parser = build_parser()
	args = parser.parse_args(["intel", "--json", "--threshold", "999"])

	with patch("mission_control.intelligence.scanner.httpx.AsyncClient") as mock_cls:
		mock_cls.return_value = mock_client
		result = cmd_intel(args)

	assert result == 0
	captured = capsys.readouterr()
	data = json.loads(captured.out)
	assert data["proposals"] == []
