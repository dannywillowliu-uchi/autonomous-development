"""Tests for the auto-update pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from autodev.auto_update import AutoUpdatePipeline
from autodev.cli import build_parser, cmd_auto_update
from autodev.config import MissionConfig, NotificationConfig, TelegramConfig
from autodev.db import Database
from autodev.intelligence.models import AdaptationProposal, Finding

# -- Fixtures --


@pytest.fixture
def db() -> Database:
	return Database(":memory:")


@pytest.fixture
def config() -> MissionConfig:
	"""Minimal MissionConfig with Telegram settings."""
	cfg = MissionConfig()
	cfg.notifications = NotificationConfig(
		telegram=TelegramConfig(
			bot_token="123:FAKE",
			chat_id="456",
		),
	)
	return cfg


def _make_proposal(
	risk: str = "low",
	title: str = "Test proposal",
	proposal_id: str = "prop_001",
) -> AdaptationProposal:
	return AdaptationProposal(
		id=proposal_id,
		finding_id="find_001",
		title=title,
		description="A test proposal for auto-update",
		proposal_type="integration",
		target_modules=["config.py"],
		priority=3,
		effort_estimate="small",
		risk_level=risk,
	)


def _make_finding(title: str = "MCP server improvement") -> Finding:
	return Finding(
		source="hackernews",
		title=title,
		url="https://example.com",
		summary="New MCP server pattern for autonomous coding agents",
		relevance_score=1.5,
	)


def _mock_ratchet():
	"""Create a mock GitRatchet that returns a fake tag."""
	mock = MagicMock()
	mock.checkpoint = AsyncMock(return_value="autodev/pre-test")
	mock.rollback = AsyncMock()
	mock.verify_and_decide = AsyncMock(return_value=True)
	return mock


def _patch_ratchet():
	"""Context manager to patch GitRatchet in auto_update."""
	return patch("autodev.ratchet.GitRatchet", return_value=_mock_ratchet())


def _mock_scan_response(request: httpx.Request) -> httpx.Response:
	"""Mock HTTP transport for intel scanners."""
	url = str(request.url)
	if "hn.algolia.com" in url:
		return httpx.Response(200, json={
			"hits": [{
				"objectID": "9001",
				"title": "New MCP server for autonomous coding",
				"url": "https://example.com/mcp",
				"points": 200,
				"num_comments": 50,
				"created_at": "2026-03-10T12:00:00Z",
			}],
		})
	if "api.github.com" in url:
		if "anthropics/claude-code/releases" in url:
			return httpx.Response(200, json=[])
		return httpx.Response(200, json={
			"items": [{
				"full_name": "example/mcp-toolkit",
				"html_url": "https://github.com/example/mcp-toolkit",
				"description": "MCP server toolkit for agent coordination",
				"stargazers_count": 300,
				"pushed_at": "2026-03-10T08:00:00Z",
			}],
		})
	if "arxiv.org" in url:
		return httpx.Response(200, text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>')
	return httpx.Response(404)


# -- DB table tests --


def test_applied_proposals_table_exists(db: Database) -> None:
	"""The applied_proposals table is created during initialization."""
	row = db.conn.execute(
		"SELECT name FROM sqlite_master WHERE type='table' AND name='applied_proposals'"
	).fetchone()
	assert row is not None


def test_record_and_check_applied_proposal(db: Database) -> None:
	"""record_applied_proposal stores a row and is_proposal_applied returns True."""
	assert not db.is_proposal_applied("Test Proposal")

	row_id = db.record_applied_proposal(
		proposal_id="prop_001",
		finding_title="Test Proposal",
		mission_id="mission_abc",
		status="launched",
		objective="Do the thing",
	)
	assert row_id

	assert db.is_proposal_applied("Test Proposal")
	assert not db.is_proposal_applied("Nonexistent Proposal")


def test_get_applied_proposals(db: Database) -> None:
	"""get_applied_proposals returns recorded proposals."""
	db.record_applied_proposal("p1", "Title 1", "m1", "launched", "obj1")
	db.record_applied_proposal("p2", "Title 2", "m2", "launched", "obj2")

	results = db.get_applied_proposals()
	assert len(results) == 2
	ids = {r["proposal_id"] for r in results}
	assert ids == {"p1", "p2"}


# -- Pipeline tests --


@pytest.mark.asyncio
async def test_pipeline_dry_run(config: MissionConfig, db: Database) -> None:
	"""Dry run returns proposals with action='dry_run' and doesn't write to DB."""
	pipeline = AutoUpdatePipeline(config, db)

	transport = httpx.MockTransport(_mock_scan_response)
	mock_client = httpx.AsyncClient(transport=transport)

	with patch("autodev.intelligence.scanner.httpx.AsyncClient") as mock_cls:
		mock_cls.return_value = mock_client
		results = await pipeline.run(dry_run=True, threshold=0.0)

	assert len(results) > 0
	for r in results:
		assert r.action == "dry_run"

	# No proposals should be recorded in DB
	assert db.get_applied_proposals() == []


@pytest.mark.asyncio
async def test_pipeline_low_risk_auto_launches(config: MissionConfig, db: Database) -> None:
	"""Low-risk proposals are auto-launched without approval."""
	pipeline = AutoUpdatePipeline(config, db)

	transport = httpx.MockTransport(_mock_scan_response)
	mock_client = httpx.AsyncClient(transport=transport)

	with (
		patch("autodev.intelligence.scanner.httpx.AsyncClient") as mock_cls,
		_patch_ratchet(),
	):
		mock_cls.return_value = mock_client
		results = await pipeline.run(threshold=0.0)

	launched = [r for r in results if r.action == "launched"]
	# All proposals from the mock data should be low-risk
	assert len(launched) > 0

	# Verify recorded in DB
	applied = db.get_applied_proposals()
	assert len(applied) == len(launched)


@pytest.mark.asyncio
async def test_pipeline_skips_already_applied(config: MissionConfig, db: Database) -> None:
	"""Proposals that are already in the DB are filtered out."""
	pipeline = AutoUpdatePipeline(config, db)

	transport = httpx.MockTransport(_mock_scan_response)

	# First run: process proposals
	mock_client = httpx.AsyncClient(transport=transport)
	with (
		patch("autodev.intelligence.scanner.httpx.AsyncClient") as mock_cls,
		_patch_ratchet(),
	):
		mock_cls.return_value = mock_client
		first_results = await pipeline.run(threshold=0.0)

	assert len(first_results) > 0

	# Second run: same proposals should be skipped
	mock_client2 = httpx.AsyncClient(transport=transport)
	with (
		patch("autodev.intelligence.scanner.httpx.AsyncClient") as mock_cls,
		_patch_ratchet(),
	):
		mock_cls.return_value = mock_client2
		second_results = await pipeline.run(threshold=0.0)

	assert len(second_results) == 0


@pytest.mark.asyncio
async def test_pipeline_high_risk_no_telegram(db: Database) -> None:
	"""High-risk proposals are skipped when Telegram is not configured."""
	config = MissionConfig()
	config.notifications = NotificationConfig(telegram=TelegramConfig())
	pipeline = AutoUpdatePipeline(config, db)

	# Create a mock report with a high-risk proposal
	high_risk = _make_proposal(risk="high", proposal_id="hr_001")
	mock_report = MagicMock()
	mock_report.findings = [_make_finding()]
	mock_report.proposals = [high_risk]

	with patch("autodev.auto_update.run_scan", return_value=mock_report):
		results = await pipeline.run()

	assert len(results) == 1
	assert results[0].action == "skipped"
	assert results[0].risk_level == "high"


@pytest.mark.asyncio
async def test_pipeline_high_risk_approve_all(config: MissionConfig, db: Database) -> None:
	"""--approve-all flag bypasses Telegram approval for high-risk proposals."""
	pipeline = AutoUpdatePipeline(config, db)

	high_risk = _make_proposal(risk="high", proposal_id="hr_002")
	mock_report = MagicMock()
	mock_report.findings = [_make_finding()]
	mock_report.proposals = [high_risk]

	with patch("autodev.auto_update.run_scan", return_value=mock_report):
		results = await pipeline.run(approve_all=True)

	assert len(results) == 1
	assert results[0].action == "launched"
	assert results[0].risk_level == "high"
	assert results[0].mission_id != ""


@pytest.mark.asyncio
async def test_pipeline_high_risk_approved_via_telegram(config: MissionConfig, db: Database) -> None:
	"""High-risk proposals approved via Telegram get launched."""
	pipeline = AutoUpdatePipeline(config, db)

	high_risk = _make_proposal(risk="high", proposal_id="hr_003")
	mock_report = MagicMock()
	mock_report.findings = [_make_finding()]
	mock_report.proposals = [high_risk]

	mock_notifier = AsyncMock()
	mock_notifier.request_approval = AsyncMock(return_value=True)
	mock_notifier.close = AsyncMock()

	with (
		patch("autodev.auto_update.run_scan", return_value=mock_report),
		patch("autodev.auto_update.TelegramNotifier", return_value=mock_notifier),
	):
		results = await pipeline.run()

	assert len(results) == 1
	assert results[0].action == "approved"
	assert results[0].mission_id != ""
	mock_notifier.request_approval.assert_called_once()
	mock_notifier.close.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_high_risk_rejected_via_telegram(config: MissionConfig, db: Database) -> None:
	"""High-risk proposals rejected via Telegram are not launched."""
	pipeline = AutoUpdatePipeline(config, db)

	high_risk = _make_proposal(risk="high", proposal_id="hr_004")
	mock_report = MagicMock()
	mock_report.findings = [_make_finding()]
	mock_report.proposals = [high_risk]

	mock_notifier = AsyncMock()
	mock_notifier.request_approval = AsyncMock(return_value=False)
	mock_notifier.close = AsyncMock()

	with (
		patch("autodev.auto_update.run_scan", return_value=mock_report),
		patch("autodev.auto_update.TelegramNotifier", return_value=mock_notifier),
	):
		results = await pipeline.run()

	assert len(results) == 1
	assert results[0].action == "rejected"
	assert results[0].mission_id == ""


def test_generate_objective(config: MissionConfig, db: Database) -> None:
	"""_generate_objective produces a well-formed objective string."""
	pipeline = AutoUpdatePipeline(config, db)
	proposal = _make_proposal()
	objective = pipeline._generate_objective(proposal)

	assert "[AUTO-UPDATE]" in objective
	assert proposal.title in objective
	assert "config.py" in objective
	assert "All tests must pass" in objective


# -- CLI tests --


def test_cli_auto_update_parser() -> None:
	"""auto-update subcommand parses --dry-run and --approve-all flags."""
	parser = build_parser()

	args = parser.parse_args(["auto-update", "--dry-run"])
	assert args.command == "auto-update"
	assert args.dry_run is True
	assert args.approve_all is False

	args2 = parser.parse_args(["auto-update", "--approve-all", "--threshold", "0.5"])
	assert args2.approve_all is True
	assert args2.threshold == 0.5


def test_cmd_auto_update_dry_run(capsys: pytest.CaptureFixture[str], db: Database) -> None:
	"""cmd_auto_update in dry-run mode prints proposals."""
	parser = build_parser()
	args = parser.parse_args(["auto-update", "--config", "autodev.toml", "--dry-run"])

	transport = httpx.MockTransport(_mock_scan_response)
	mock_client = httpx.AsyncClient(transport=transport)

	mock_config = MissionConfig()

	with (
		patch("autodev.cli.load_config", return_value=mock_config),
		patch("autodev.cli._get_db_path", return_value=":memory:"),
		patch("autodev.intelligence.scanner.httpx.AsyncClient") as mock_cls,
	):
		mock_cls.return_value = mock_client
		result = cmd_auto_update(args)

	assert result == 0
	captured = capsys.readouterr()
	assert "dry_run" in captured.out or "No new proposals" in captured.out


# -- Rate limiting tests --


def test_rate_limit_under_limit(config: MissionConfig, db: Database) -> None:
	"""_check_rate_limit returns True when under the daily limit."""
	pipeline = AutoUpdatePipeline(config, db, max_daily_modifications=5)
	assert pipeline._check_rate_limit() is True


def test_rate_limit_at_limit(config: MissionConfig, db: Database) -> None:
	"""_check_rate_limit returns False when at or over the daily limit."""
	pipeline = AutoUpdatePipeline(config, db, max_daily_modifications=2)
	# Record 2 proposals today
	db.record_applied_proposal("p1", "Title 1", "m1", "launched", "obj1")
	db.record_applied_proposal("p2", "Title 2", "m2", "launched", "obj2")
	assert pipeline._check_rate_limit() is False


@pytest.mark.asyncio
async def test_rate_limit_blocks_run(config: MissionConfig, db: Database) -> None:
	"""run() returns empty list when rate limit is exceeded."""
	pipeline = AutoUpdatePipeline(config, db, max_daily_modifications=1)
	db.record_applied_proposal("p1", "Title 1", "m1", "launched", "obj1")

	results = await pipeline.run()
	assert results == []


def test_count_proposals_applied_since(db: Database) -> None:
	"""count_proposals_applied_since returns correct count."""
	db.record_applied_proposal("p1", "T1", "m1", "launched", "o1")
	db.record_applied_proposal("p2", "T2", "m2", "launched", "o2")

	today = datetime.now(timezone.utc).date().isoformat()
	assert db.count_proposals_applied_since(today) == 2
	assert db.count_proposals_applied_since("2099-01-01") == 0


# -- Diff review tests --


@pytest.mark.asyncio
async def test_review_modification_approved(config: MissionConfig, db: Database) -> None:
	"""_review_modification returns approved when LLM approves."""
	pipeline = AutoUpdatePipeline(config, db)

	mock_diff_proc = AsyncMock()
	mock_diff_proc.communicate = AsyncMock(return_value=(b"+ new line\n- old line\n", b""))

	review_json = json.dumps({"approved": True, "concerns": [], "summary": "Looks good"})
	mock_review_proc = AsyncMock()
	mock_review_proc.communicate = AsyncMock(return_value=(review_json.encode(), b""))

	call_count = 0

	async def mock_create_subprocess(*args, **kwargs):
		nonlocal call_count
		call_count += 1
		if call_count == 1:
			return mock_diff_proc
		return mock_review_proc

	with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
		result = await pipeline._review_modification("autodev/pre-test")

	assert result["approved"] is True
	assert result["summary"] == "Looks good"


@pytest.mark.asyncio
async def test_review_modification_rejected(config: MissionConfig, db: Database) -> None:
	"""_review_modification returns rejected when LLM flags concerns."""
	pipeline = AutoUpdatePipeline(config, db)

	mock_diff_proc = AsyncMock()
	mock_diff_proc.communicate = AsyncMock(return_value=(b"+ dangerous code\n", b""))

	review_json = json.dumps({
		"approved": False,
		"concerns": ["Security issue"],
		"summary": "Rejected due to security concern",
	})
	mock_review_proc = AsyncMock()
	mock_review_proc.communicate = AsyncMock(return_value=(review_json.encode(), b""))

	call_count = 0

	async def mock_create_subprocess(*args, **kwargs):
		nonlocal call_count
		call_count += 1
		if call_count == 1:
			return mock_diff_proc
		return mock_review_proc

	with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
		result = await pipeline._review_modification("autodev/pre-test")

	assert result["approved"] is False
	assert "Security issue" in result["concerns"]


@pytest.mark.asyncio
async def test_review_modification_llm_failure(config: MissionConfig, db: Database) -> None:
	"""_review_modification defaults to approved when LLM fails."""
	pipeline = AutoUpdatePipeline(config, db)

	with patch("asyncio.create_subprocess_exec", side_effect=Exception("LLM unavailable")):
		result = await pipeline._review_modification("autodev/pre-test")

	assert result["approved"] is True
	assert result["summary"] == "Review unavailable"
