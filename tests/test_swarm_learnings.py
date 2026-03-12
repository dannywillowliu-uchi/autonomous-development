"""Tests for swarm learnings -- persistent memory across runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodev.swarm.learnings import HEADER, LEARNINGS_FILE, SwarmLearnings, _score_learning


@pytest.fixture
def learnings(tmp_path: Path) -> SwarmLearnings:
	"""Create a SwarmLearnings instance in a temp directory."""
	return SwarmLearnings(tmp_path)


@pytest.fixture
def learnings_path(tmp_path: Path) -> Path:
	return tmp_path / LEARNINGS_FILE


def test_add_discovery(learnings: SwarmLearnings, learnings_path: Path) -> None:
	text = "Found a bug in src/autodev/controller.py that causes race conditions"
	result = learnings.add_discovery("test-agent", text)
	assert result is True
	content = learnings_path.read_text()
	assert "## Discovery" in content
	assert "**Source:** test-agent" in content
	assert "bug in src/autodev/controller.py" in content


def test_add_successful_approach(learnings: SwarmLearnings, learnings_path: Path) -> None:
	text = "Must use retry logic with backoff in src/auth/login.py to fix the flaky test"
	result = learnings.add_successful_approach("Fix login bug", text)
	assert result is True
	content = learnings_path.read_text()
	assert "## What Worked" in content
	assert "**Task:** Fix login bug" in content
	assert "Must use retry logic with backoff in src/auth/login.py" in content


def test_add_failed_approach(learnings: SwarmLearnings, learnings_path: Path) -> None:
	text = (
		"Bug in src/db/pool.py -- connection pool exhausted under load,"
		" must increase max_connections"
	)
	result = learnings.add_failed_approach("Refactor DB layer", text, attempt=3)
	assert result is True
	content = learnings_path.read_text()
	assert "## What Failed" in content
	assert "**Task:** Refactor DB layer (attempt 3)" in content
	assert "connection pool exhausted under load" in content


def test_add_stagnation_insight(learnings: SwarmLearnings, learnings_path: Path) -> None:
	result = learnings.add_stagnation_insight(
		"flat_tests", "Switched from unit to integration testing approach",
	)
	assert result is True
	content = learnings_path.read_text()
	assert "## Stagnation Pivot" in content
	assert "**Metric:** flat_tests" in content
	assert "Switched from unit to integration testing approach" in content


def test_deduplication(learnings: SwarmLearnings, learnings_path: Path) -> None:
	text = "Bug: database indexes are missing on user_id column in src/models/user.py"
	learnings.add_discovery("agent-1", text)
	first_content = learnings_path.read_text()
	result = learnings.add_discovery("agent-2", text)
	assert result is False
	assert learnings_path.read_text() == first_content


def test_fuzzy_dedup(learnings: SwarmLearnings, learnings_path: Path) -> None:
	"""Similar entries where first 80 chars match should be deduplicated."""
	long_prefix = "Bug in src/autodev/foo.py: must fix " + "A" * 50
	learnings.add_discovery("agent-1", f"{long_prefix} -- ending one")
	first_content = learnings_path.read_text()
	result = learnings.add_discovery("agent-2", f"{long_prefix} -- ending two")
	assert result is False
	assert learnings_path.read_text() == first_content


def test_different_entries_not_deduped(learnings: SwarmLearnings, learnings_path: Path) -> None:
	learnings.add_discovery(
		"agent-1", "Bug in src/cache/redis.py: must invalidate cache on write to avoid stale reads",
	)
	learnings.add_discovery(
		"agent-2", "Fix for src/logs/handler.py: never use blocking IO in the hot path",
	)
	content = learnings_path.read_text()
	assert "must invalidate cache on write" in content
	assert "never use blocking IO" in content


def test_line_count_bounding(tmp_path: Path) -> None:
	"""File should be trimmed to ~200 lines when it grows too large."""
	sl = SwarmLearnings(tmp_path)
	# Add many entries to exceed 200 lines
	for i in range(100):
		text = f"Bug in src/mod_{i}/handler.py: must fix race condition " + "x" * 30
		sl.add_discovery(f"agent-{i}", text)
	content = (tmp_path / LEARNINGS_FILE).read_text()
	line_count = len(content.split("\n"))
	# Should be trimmed: header lines + ~150 entry lines
	assert line_count <= 210, f"File has {line_count} lines, expected <= 210"


def test_get_for_planner(learnings: SwarmLearnings) -> None:
	learnings.add_discovery(
		"agent-1", "Bug in src/autodev/planner.py: must handle empty LLM responses",
	)
	learnings.add_successful_approach(
		"Task B", "Fix in src/autodev/worker.py: never retry on auth failures",
	)
	result = learnings.get_for_planner()
	assert "Accumulated Learnings" in result
	assert "must handle empty LLM responses" in result
	assert "never retry on auth failures" in result


def test_get_for_planner_respects_max_lines(tmp_path: Path) -> None:
	sl = SwarmLearnings(tmp_path)
	for i in range(30):
		text = f"Bug in src/module_{i}/handler.py: must fix race condition number {i}"
		sl.add_discovery(f"agent-{i}", text)
	result = sl.get_for_planner(max_lines=10)
	lines = result.strip().split("\n")
	# Header line + blank + 10 content lines
	assert len(lines) <= 12


def test_empty_file(tmp_path: Path) -> None:
	"""get_for_planner on a fresh file returns empty string."""
	sl = SwarmLearnings(tmp_path)
	assert sl.get_for_planner() == ""


def test_corrupted_file(tmp_path: Path) -> None:
	"""Handles file with no valid sections gracefully."""
	path = tmp_path / LEARNINGS_FILE
	path.write_text("some garbage\nno headers\nrandom stuff\n")
	sl = SwarmLearnings(tmp_path)
	# Should not crash, get_for_planner returns empty for no entries
	assert sl.get_for_planner() == ""
	# Should still be able to add entries
	text = "Bug in src/autodev/parser.py: must never ignore parse errors"
	result = sl.add_discovery("agent", text)
	assert result is True
	assert "must never ignore parse errors" in path.read_text()


def test_timestamps(learnings: SwarmLearnings, learnings_path: Path) -> None:
	"""Entries should have ISO-format timestamps."""
	text = "Bug in src/autodev/session.py: must fix timeout handling"
	learnings.add_discovery("agent", text)
	content = learnings_path.read_text()
	# Timestamps are in YYYY-MM-DD HH:MM format
	import re
	assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", content)


def test_file_created_on_init(tmp_path: Path) -> None:
	"""Learnings file should be created on initialization."""
	SwarmLearnings(tmp_path)
	path = tmp_path / LEARNINGS_FILE
	assert path.exists()
	assert path.read_text() == HEADER


def test_strips_whitespace_in_entries(learnings: SwarmLearnings, learnings_path: Path) -> None:
	"""Entry text should be stripped of leading/trailing whitespace."""
	text = "  Bug in src/autodev/worker.py: must fix race condition  \n\n"
	learnings.add_discovery("agent", text)
	content = learnings_path.read_text()
	assert "  Bug in src/autodev/worker.py" not in content
	assert "Bug in src/autodev/worker.py" in content


# --- Quality scoring tests ---


def test_score_learning_with_file_path() -> None:
	"""Text mentioning a file path scores high."""
	text = "Found a race condition in src/autodev/controller.py when spawning workers"
	score = _score_learning(text)
	assert score >= 1.0


def test_score_learning_with_function_name() -> None:
	"""Text mentioning a function name scores high."""
	text = "The parse_cost() function returns incorrect values with unicode input"
	score = _score_learning(text)
	assert score >= 1.0


def test_score_learning_with_actionable_keyword() -> None:
	"""Text with actionable keywords like 'bug' or 'fix' scores high."""
	text = "There is a bug in the connection pool that causes intermittent timeouts"
	score = _score_learning(text)
	assert score >= 0.5
	text2 = "Must never call the cleanup handler while the lock is still held"
	score2 = _score_learning(text2)
	assert score2 >= 0.5


def test_score_learning_short_generic() -> None:
	"""Short text with no specifics scores low."""
	score = _score_learning("it works now")
	assert score < 0.5


def test_score_learning_generic_completion() -> None:
	"""Generic completion phrases score low."""
	score = _score_learning("All tests pass, completed successfully")
	assert score < 0.5


# --- Quality filter integration tests ---


def test_quality_filter_rejects_generic(learnings: SwarmLearnings) -> None:
	"""add_discovery with generic text returns False."""
	result = learnings.add_discovery("agent", "done")
	assert result is False


def test_quality_filter_accepts_specific(learnings: SwarmLearnings) -> None:
	"""add_discovery with file paths and actionable content returns True."""
	text = "Bug in src/autodev/planner.py: must handle empty LLM responses"
	result = learnings.add_discovery("agent", text)
	assert result is True


# --- Hash-based dedup tests ---


def test_hash_dedup_prevents_duplicate(learnings: SwarmLearnings) -> None:
	"""Same text added twice -- second returns False via hash dedup."""
	text = "Bug in src/autodev/worker.py: must never retry on auth failures"
	first = learnings.add_discovery("agent-1", text)
	assert first is True
	second = learnings.add_discovery("agent-2", text)
	assert second is False


def test_hash_dedup_different_text(learnings: SwarmLearnings) -> None:
	"""Different texts both get added successfully."""
	r1 = learnings.add_discovery(
		"agent-1", "Bug in src/autodev/db.py: must close WAL checkpoint before migration",
	)
	r2 = learnings.add_discovery(
		"agent-2", "Gotcha in src/autodev/green_branch.py: never force-push to shared branch",
	)
	assert r1 is True
	assert r2 is True
