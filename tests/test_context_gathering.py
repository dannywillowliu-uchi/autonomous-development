"""Tests for context_gathering.py -- all 7 public functions."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig, TargetConfig
from mission_control.context_gathering import (
	VerificationFailureSummary,
	format_verification_failures,
	get_episodic_context,
	get_git_log,
	get_human_preferences,
	get_intel_context,
	get_past_missions,
	get_strategic_context,
	get_verification_failures,
	read_backlog,
)
from mission_control.models import (
	EpisodicMemory,
	Mission,
	SemanticMemory,
	TrajectoryRating,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mission(
	mid: str = "m1",
	objective: str = "Build feature X",
	status: str = "completed",
	total_rounds: int = 3,
	final_score: float = 0.8,
	stopped_reason: str = "",
) -> Mission:
	return Mission(
		id=mid,
		objective=objective,
		status=status,
		total_rounds=total_rounds,
		final_score=final_score,
		stopped_reason=stopped_reason,
	)


def _rating(mission_id: str = "m1", rating: int = 8) -> TrajectoryRating:
	return TrajectoryRating(mission_id=mission_id, rating=rating)


# ---------------------------------------------------------------------------
# read_backlog
# ---------------------------------------------------------------------------


class TestReadBacklog:
	def test_existing_backlog(self, tmp_path: Path) -> None:
		(tmp_path / "BACKLOG.md").write_text("## Phase 1\n- item")
		cfg = MissionConfig(target=TargetConfig(path=str(tmp_path), name="t"))
		assert read_backlog(cfg) == "## Phase 1\n- item"

	def test_missing_backlog_returns_empty(self, tmp_path: Path) -> None:
		cfg = MissionConfig(target=TargetConfig(path=str(tmp_path), name="t"))
		assert read_backlog(cfg) == ""


# ---------------------------------------------------------------------------
# get_git_log
# ---------------------------------------------------------------------------


class TestGetGitLog:
	@pytest.mark.asyncio()
	async def test_successful_git_log(self, tmp_path: Path) -> None:
		cfg = MissionConfig(target=TargetConfig(path=str(tmp_path), name="t"))
		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate = AsyncMock(return_value=(b"abc123 first\ndef456 second\n", b""))

		with patch("mission_control.context_gathering.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await get_git_log(cfg, limit=5)

		assert "abc123 first" in result
		assert "def456 second" in result

	@pytest.mark.asyncio()
	async def test_timeout_returns_empty(self, tmp_path: Path) -> None:
		cfg = MissionConfig(target=TargetConfig(path=str(tmp_path), name="t"))

		async def _hang() -> tuple[bytes, bytes]:
			await asyncio.sleep(999)
			return (b"", b"")

		mock_proc = AsyncMock()
		mock_proc.communicate = _hang

		with patch("mission_control.context_gathering.asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch("mission_control.context_gathering.asyncio.wait_for", side_effect=asyncio.TimeoutError):
				result = await get_git_log(cfg)

		assert result == ""

	@pytest.mark.asyncio()
	async def test_nonzero_exit_returns_empty(self, tmp_path: Path) -> None:
		cfg = MissionConfig(target=TargetConfig(path=str(tmp_path), name="t"))
		mock_proc = AsyncMock()
		mock_proc.returncode = 128
		mock_proc.communicate = AsyncMock(return_value=(b"fatal: not a repo", b""))

		with patch("mission_control.context_gathering.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await get_git_log(cfg)

		assert result == ""


# ---------------------------------------------------------------------------
# get_past_missions
# ---------------------------------------------------------------------------


class TestGetPastMissions:
	def test_multiple_missions(self) -> None:
		db = MagicMock()
		db.get_all_missions.return_value = [
			_mission("m1", "Build auth system", "completed", 5, 0.9),
			_mission("m2", "Fix login bug", "failed", 2, 0.3, stopped_reason="timeout"),
		]
		db.get_trajectory_ratings_for_mission.return_value = []

		result = get_past_missions(db, limit=10)
		assert "[completed] Build auth system" in result
		assert "[failed] Fix login bug" in result
		assert "reason=timeout" in result
		assert "reason=n/a" in result

	def test_with_trajectory_ratings(self) -> None:
		db = MagicMock()
		db.get_all_missions.return_value = [_mission("m1")]
		db.get_trajectory_ratings_for_mission.return_value = [_rating("m1", 9)]

		result = get_past_missions(db)
		assert "human_rating=9/10" in result

	def test_empty_missions_list(self) -> None:
		db = MagicMock()
		db.get_all_missions.return_value = []

		assert get_past_missions(db) == ""


# ---------------------------------------------------------------------------
# get_strategic_context
# ---------------------------------------------------------------------------


class TestGetStrategicContext:
	def test_with_entries(self) -> None:
		db = MagicMock()
		db.get_strategic_context.return_value = ["entry one", "entry two"]

		result = get_strategic_context(db)
		assert "- entry one" in result
		assert "- entry two" in result

	def test_db_lacks_method(self) -> None:
		"""hasattr guard returns empty when DB has no get_strategic_context."""
		db = MagicMock(spec=[])  # spec=[] means no attributes at all
		result = get_strategic_context(db)
		assert result == ""

	def test_empty_entries_returns_empty(self) -> None:
		db = MagicMock()
		db.get_strategic_context.return_value = []
		assert get_strategic_context(db) == ""


# ---------------------------------------------------------------------------
# get_episodic_context
# ---------------------------------------------------------------------------


class TestGetEpisodicContext:
	def test_semantic_and_episodic_memories(self) -> None:
		db = MagicMock()
		db.get_top_semantic_memories.return_value = [
			SemanticMemory(content="Always run tests", confidence=0.95),
		]
		db.get_episodic_memories_by_scope.return_value = [
			EpisodicMemory(event_type="merge_success", content="Merged PR #1", outcome="green"),
		]

		result = get_episodic_context(db)
		assert "Learned rules:" in result
		assert "[0.9] Always run tests" in result
		assert "Recent episodes:" in result
		assert "[merge_success] Merged PR #1 -> green" in result

	def test_exception_resilience_semantic(self) -> None:
		"""Semantic failure doesn't prevent episodic from loading."""
		db = MagicMock()
		db.get_top_semantic_memories.side_effect = RuntimeError("db locked")
		db.get_episodic_memories_by_scope.return_value = [
			EpisodicMemory(event_type="test_failure", content="flaky test", outcome="retry"),
		]

		result = get_episodic_context(db)
		assert "Learned rules:" not in result
		assert "[test_failure] flaky test -> retry" in result

	def test_exception_resilience_episodic(self) -> None:
		"""Episodic failure doesn't prevent semantic from loading."""
		db = MagicMock()
		db.get_top_semantic_memories.return_value = [
			SemanticMemory(content="Keep it simple", confidence=0.8),
		]
		db.get_episodic_memories_by_scope.side_effect = RuntimeError("corrupt")

		result = get_episodic_context(db)
		assert "Learned rules:" in result
		assert "Recent episodes:" not in result

	def test_both_empty(self) -> None:
		db = MagicMock()
		db.get_top_semantic_memories.return_value = []
		db.get_episodic_memories_by_scope.return_value = []

		assert get_episodic_context(db) == ""


# ---------------------------------------------------------------------------
# get_human_preferences
# ---------------------------------------------------------------------------


class TestGetHumanPreferences:
	def test_high_and_low_rated_formatting(self) -> None:
		db = MagicMock()
		db.get_all_missions.return_value = [
			_mission("m1", "Great feature"),
			_mission("m2", "Bad refactor"),
			_mission("m3", "OK work"),
		]
		db.get_trajectory_ratings_for_mission.side_effect = lambda mid: {
			"m1": [_rating("m1", 9)],
			"m2": [_rating("m2", 2)],
			"m3": [_rating("m3", 5)],
		}[mid]

		result = get_human_preferences(db)
		assert "Highly-rated:" in result
		assert '"Great feature" (9/10)' in result
		assert "Low-rated:" in result
		assert '"Bad refactor" (2/10)' in result
		# Mid-range (5) should not appear in either category
		assert "OK work" not in result
		assert "Prefer work similar to highly-rated missions." in result

	def test_no_ratings_returns_empty(self) -> None:
		db = MagicMock()
		db.get_all_missions.return_value = [_mission("m1")]
		db.get_trajectory_ratings_for_mission.return_value = []

		assert get_human_preferences(db) == ""

	def test_exception_returns_empty(self) -> None:
		db = MagicMock()
		db.get_all_missions.side_effect = RuntimeError("boom")

		assert get_human_preferences(db) == ""


# ---------------------------------------------------------------------------
# get_intel_context
# ---------------------------------------------------------------------------


class TestGetIntelContext:
	@pytest.mark.asyncio()
	async def test_cache_hit(self, tmp_path: Path) -> None:
		"""Fresh cache is used without re-scanning."""
		cfg = MissionConfig(target=TargetConfig(path=str(tmp_path), name="t"))
		cache_dir = tmp_path / ".cache"
		cache_dir.mkdir()

		cached = {
			"cached_at": datetime.now(timezone.utc).isoformat(),
			"findings": [],
			"proposals": [
				{
					"id": "p1",
					"finding_id": "f1",
					"title": "Adopt async DB",
					"description": "Switch to async driver",
					"proposal_type": "integration",
					"target_modules": [],
					"priority": 1,
					"effort_estimate": "medium",
				},
			],
			"timestamp": "",
			"sources_scanned": [],
			"scan_duration_seconds": 0.0,
		}
		(cache_dir / "intel_report.json").write_text(json.dumps(cached))

		result = await get_intel_context(cfg, ttl_hours=6.0)
		assert "Ecosystem Intelligence" in result
		assert "Adopt async DB" in result

	@pytest.mark.asyncio()
	async def test_cache_miss_triggers_scan(self, tmp_path: Path) -> None:
		"""No cache file -> run_scan is called and cache is written."""
		cfg = MissionConfig(target=TargetConfig(path=str(tmp_path), name="t"))

		from mission_control.intelligence.models import AdaptationProposal
		from mission_control.intelligence.scanner import IntelReport

		mock_report = IntelReport(
			findings=[],
			proposals=[
				AdaptationProposal(title="Use caching", description="Add Redis", priority=2),
			],
			timestamp="",
			sources_scanned=[],
			scan_duration_seconds=1.0,
		)

		with patch("mission_control.intelligence.run_scan", new_callable=AsyncMock, return_value=mock_report):
			result = await get_intel_context(cfg)

		assert "Use caching" in result
		# Cache file should be written
		cache_path = tmp_path / ".cache" / "intel_report.json"
		assert cache_path.exists()

	@pytest.mark.asyncio()
	async def test_stale_cache_rescans(self, tmp_path: Path) -> None:
		"""Expired cache triggers a fresh scan."""
		cfg = MissionConfig(target=TargetConfig(path=str(tmp_path), name="t"))
		cache_dir = tmp_path / ".cache"
		cache_dir.mkdir()

		old_time = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
		cached = {
			"cached_at": old_time,
			"findings": [],
			"proposals": [],
			"timestamp": "",
			"sources_scanned": [],
			"scan_duration_seconds": 0.0,
		}
		(cache_dir / "intel_report.json").write_text(json.dumps(cached))

		from mission_control.intelligence.models import AdaptationProposal
		from mission_control.intelligence.scanner import IntelReport

		fresh_report = IntelReport(
			findings=[],
			proposals=[AdaptationProposal(title="Fresh insight", description="New", priority=1)],
			timestamp="",
			sources_scanned=[],
			scan_duration_seconds=0.5,
		)

		with patch("mission_control.intelligence.run_scan", new_callable=AsyncMock, return_value=fresh_report):
			result = await get_intel_context(cfg, ttl_hours=6.0)

		assert "Fresh insight" in result

	@pytest.mark.asyncio()
	async def test_exception_returns_empty(self, tmp_path: Path) -> None:
		"""Any exception in get_intel_context returns empty string."""
		cfg = MissionConfig(target=TargetConfig(path=str(tmp_path), name="t"))

		with patch(
			"mission_control.intelligence.run_scan",
			new_callable=AsyncMock,
			side_effect=RuntimeError("network down"),
		):
			result = await get_intel_context(cfg)

		assert result == ""


# ---------------------------------------------------------------------------
# get_verification_failures / format_verification_failures
# ---------------------------------------------------------------------------

_FAILING_VERIFICATION_OUTPUT = (
	"============================= test session starts ==============================\n"
	"collected 42 items\n"
	"\n"
	"FAILED tests/test_auth.py::test_login - AssertionError: expected 200 got 401\n"
	"FAILED tests/test_db.py::test_connection_pool - TimeoutError: pool exhausted\n"
	"================ 2 failed, 40 passed in 3.21s ================================\n"
)


class TestGetVerificationFailures:
	def test_failing_verification(self, tmp_path: Path) -> None:
		"""Parses FAILED lines from verification_output."""
		report = {
			"verification_passed": False,
			"verification_output": _FAILING_VERIFICATION_OUTPUT,
		}
		(tmp_path / "mission_report.json").write_text(json.dumps(report))

		db = MagicMock()
		summary = get_verification_failures(db, tmp_path)

		assert not summary.passed
		assert len(summary.failures) == 2
		assert summary.failures[0] == (
			"tests/test_auth.py::test_login",
			"AssertionError: expected 200 got 401",
		)
		assert summary.failures[1] == (
			"tests/test_db.py::test_connection_pool",
			"TimeoutError: pool exhausted",
		)

	def test_passing_verification(self, tmp_path: Path) -> None:
		"""When verification passed, returns empty summary."""
		report = {
			"verification_passed": True,
			"verification_output": "42 passed in 1.00s",
		}
		(tmp_path / "mission_report.json").write_text(json.dumps(report))

		db = MagicMock()
		summary = get_verification_failures(db, tmp_path)

		assert summary.passed
		assert summary.failures == []

	def test_missing_report_file(self, tmp_path: Path) -> None:
		"""When mission_report.json doesn't exist, returns empty summary."""
		db = MagicMock()
		summary = get_verification_failures(db, tmp_path)

		assert summary.passed
		assert summary.failures == []


class TestFormatVerificationFailures:
	def test_format_with_failures(self) -> None:
		summary = VerificationFailureSummary(
			passed=False,
			failures=[
				("tests/test_auth.py::test_login", "AssertionError: expected 200 got 401"),
				("tests/test_db.py::test_pool", "TimeoutError: pool exhausted"),
			],
		)
		result = format_verification_failures(summary)

		assert "### Previous Verification Failures" in result
		assert "- `tests/test_auth.py::test_login`: AssertionError: expected 200 got 401" in result
		assert "- `tests/test_db.py::test_pool`: TimeoutError: pool exhausted" in result

	def test_format_passing_returns_empty(self) -> None:
		summary = VerificationFailureSummary(passed=True, failures=[])
		assert format_verification_failures(summary) == ""

	def test_format_failed_but_no_parsed_failures(self) -> None:
		"""Failed verification but no FAILED lines matched (e.g. syntax error)."""
		summary = VerificationFailureSummary(passed=False, failures=[])
		assert format_verification_failures(summary) == ""
