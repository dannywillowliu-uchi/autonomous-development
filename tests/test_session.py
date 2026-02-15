"""Tests for session spawning."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import (
	MissionConfig,
	SchedulerConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.models import MCResultSchema, Snapshot, TaskRecord
from mission_control.session import (
	build_branch_name,
	parse_mc_result,
	render_prompt,
	spawn_session,
	validate_mc_result,
)


def _config() -> MissionConfig:
	return MissionConfig(
		target=TargetConfig(
			name="test-proj",
			path="/tmp/test",
			branch="main",
			objective="Build something",
			verification=VerificationConfig(command="pytest -q"),
		),
		scheduler=SchedulerConfig(model="sonnet"),
	)


def _snapshot() -> Snapshot:
	return Snapshot(test_total=10, test_passed=8, test_failed=2, lint_errors=3, type_errors=1)


def _task(desc: str = "Fix the failing tests") -> TaskRecord:
	return TaskRecord(source="test_failure", description=desc, priority=2)


class TestRenderPrompt:
	def test_contains_task(self) -> None:
		prompt = render_prompt(_task(), _snapshot(), _config(), "mc/session-abc")
		assert "Fix the failing tests" in prompt

	def test_contains_stats(self) -> None:
		prompt = render_prompt(_task(), _snapshot(), _config(), "mc/session-abc")
		assert "8/10 passing" in prompt
		assert "Lint errors: 3" in prompt
		assert "Type errors: 1" in prompt

	def test_contains_branch(self) -> None:
		prompt = render_prompt(_task(), _snapshot(), _config(), "mc/session-abc")
		assert "mc/session-abc" in prompt

	def test_contains_verification_command(self) -> None:
		prompt = render_prompt(_task(), _snapshot(), _config(), "mc/session-abc")
		assert "pytest -q" in prompt

	def test_contains_context(self) -> None:
		prompt = render_prompt(_task(), _snapshot(), _config(), "mc/session-abc", context="Previous session failed")
		assert "Previous session failed" in prompt

	def test_default_context(self) -> None:
		prompt = render_prompt(_task(), _snapshot(), _config(), "mc/session-abc")
		assert "No additional context" in prompt

	def test_contains_target_name(self) -> None:
		prompt = render_prompt(_task(), _snapshot(), _config(), "mc/session-abc")
		assert "test-proj" in prompt


class TestParseMcResult:
	def test_valid_result(self) -> None:
		output = (
			"Some output\n"
			'MC_RESULT:{"status":"completed","commits":["abc123"],'
			'"summary":"Fixed tests","files_changed":["src/foo.py"]}'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["commits"] == ["abc123"]
		assert result["summary"] == "Fixed tests"

	def test_no_result(self) -> None:
		output = "Just some regular output\nNo structured result here"
		result = parse_mc_result(output)
		assert result is None

	def test_malformed_json(self) -> None:
		output = "MC_RESULT:{bad json}"
		result = parse_mc_result(output)
		assert result is None

	def test_result_in_middle(self) -> None:
		output = 'line 1\nMC_RESULT:{"status":"failed","commits":[],"summary":"Could not fix"}\nline 3'
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "failed"

	def test_empty_output(self) -> None:
		result = parse_mc_result("")
		assert result is None

	def test_multiline_json(self) -> None:
		"""MC_RESULT with pretty-printed multiline JSON should parse correctly."""
		output = (
			"Some output\n"
			"MC_RESULT:{\n"
			'  "status": "completed",\n'
			'  "commits": ["abc123"],\n'
			'  "summary": "Fixed the thing",\n'
			'  "files_changed": ["src/foo.py"]\n'
			"}\n"
			"More output after"
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["commits"] == ["abc123"]

	def test_multiline_json_no_trailing_content(self) -> None:
		"""MC_RESULT with multiline JSON at end of output."""
		output = (
			"Working on task...\n"
			"MC_RESULT:{\n"
			'  "status": "failed",\n'
			'  "commits": [],\n'
			'  "summary": "Could not fix"\n'
			"}"
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "failed"

	def test_uses_last_mc_result(self) -> None:
		"""When multiple MC_RESULT markers exist, use the last one."""
		output = (
			'MC_RESULT:{"status":"failed","commits":[],"summary":"first attempt"}\n'
			"Retrying...\n"
			'MC_RESULT:{"status":"completed","commits":["def456"],"summary":"second attempt"}'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["summary"] == "second attempt"


class TestBranchName:
	def test_format(self) -> None:
		assert build_branch_name("abc123") == "mc/session-abc123"

	def test_unique(self) -> None:
		assert build_branch_name("a") != build_branch_name("b")


class TestSpawnSessionTimeout:
	@pytest.mark.asyncio
	async def test_timeout_kills_subprocess(self) -> None:
		"""When session times out, the subprocess should be killed."""
		config = MissionConfig(
			target=TargetConfig(
				name="test-proj",
				path="/tmp/test",
				branch="main",
				objective="Build something",
				verification=VerificationConfig(command="pytest -q"),
			),
			scheduler=SchedulerConfig(model="sonnet", session_timeout=5),
		)
		task = TaskRecord(source="test", description="Do work", priority=1)
		snapshot = Snapshot(test_total=10, test_passed=10)

		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = asyncio.TimeoutError()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with (
			patch("mission_control.session.asyncio.create_subprocess_exec", return_value=mock_proc),
			patch("mission_control.session.create_branch", return_value=True),
		):
			session = await spawn_session(task, snapshot, config)

		assert session.status == "failed"
		assert "timed out" in session.output_summary
		mock_proc.kill.assert_called_once()
		mock_proc.wait.assert_awaited_once()


class TestMCResultSchemaValidation:
	"""Tests for MCResultSchema and validate_mc_result degraded parsing."""

	def test_valid_input_passes(self) -> None:
		"""Fully valid MC_RESULT dict passes validation unchanged."""
		raw = {
			"status": "completed",
			"commits": ["abc123"],
			"summary": "Did the thing",
			"files_changed": ["src/foo.py"],
			"discoveries": ["found a bug"],
			"concerns": ["might break later"],
		}
		result = validate_mc_result(raw)
		assert result["status"] == "completed"
		assert result["commits"] == ["abc123"]
		assert result["summary"] == "Did the thing"
		assert result["files_changed"] == ["src/foo.py"]
		assert result["discoveries"] == ["found a bug"]
		assert result["concerns"] == ["might break later"]

	def test_missing_optional_fields_get_defaults(self) -> None:
		"""Missing optional fields (discoveries, concerns) default to empty lists."""
		raw = {
			"status": "completed",
			"commits": ["abc123"],
			"summary": "Did it",
			"files_changed": ["src/foo.py"],
		}
		result = validate_mc_result(raw)
		assert result["discoveries"] == []
		assert result["concerns"] == []

	def test_extra_fields_are_ignored(self) -> None:
		"""Extra fields not in the schema are stripped out."""
		raw = {
			"status": "completed",
			"commits": ["abc123"],
			"summary": "Did it",
			"files_changed": ["src/foo.py"],
			"extra_field": "should be ignored",
			"another": 42,
		}
		result = validate_mc_result(raw)
		assert "extra_field" not in result
		assert "another" not in result
		assert result["status"] == "completed"

	def test_wrong_status_type_returns_degraded(self, caplog: pytest.LogCaptureFixture) -> None:
		"""Invalid status value triggers degraded parsing with warning."""
		raw = {
			"status": "unknown_status",
			"commits": ["abc123"],
			"summary": "Did it",
			"files_changed": ["src/foo.py"],
		}
		with caplog.at_level(logging.WARNING, logger="mission_control.session"):
			result = validate_mc_result(raw)
		assert result["status"] == "failed"  # degraded default
		assert result["commits"] == ["abc123"]  # valid field preserved
		assert result["summary"] == "Did it"  # valid field preserved
		assert "schema validation failed" in caplog.text.lower()

	def test_wrong_commits_type_returns_degraded(self, caplog: pytest.LogCaptureFixture) -> None:
		"""Non-list commits triggers degraded parsing, valid fields preserved."""
		raw = {
			"status": "completed",
			"commits": "not-a-list",
			"summary": "Did it",
			"files_changed": ["src/foo.py"],
		}
		with caplog.at_level(logging.WARNING, logger="mission_control.session"):
			result = validate_mc_result(raw)
		assert result["status"] == "completed"  # valid field preserved
		assert result["commits"] == []  # invalid field gets default
		assert "schema validation failed" in caplog.text.lower()

	def test_completely_invalid_input_returns_degraded(self, caplog: pytest.LogCaptureFixture) -> None:
		"""Completely invalid dict returns all defaults with warning."""
		raw = {"garbage": True, "number": 42}
		with caplog.at_level(logging.WARNING, logger="mission_control.session"):
			result = validate_mc_result(raw)
		assert result["status"] == "failed"
		assert result["commits"] == []
		assert result["summary"] == ""
		assert result["files_changed"] == []
		assert result["discoveries"] == []
		assert result["concerns"] == []
		assert "schema validation failed" in caplog.text.lower()

	def test_pydantic_model_directly(self) -> None:
		"""MCResultSchema can be instantiated directly with valid data."""
		schema = MCResultSchema(
			status="blocked",
			commits=[],
			summary="Blocked on dependency",
			files_changed=[],
		)
		assert schema.status == "blocked"
		assert schema.discoveries == []
		assert schema.concerns == []

	def test_parse_mc_result_integration(self) -> None:
		"""parse_mc_result returns validated data for valid MC_RESULT output."""
		output = (
			'MC_RESULT:{"status":"completed","commits":["abc"],'
			'"summary":"done","files_changed":["f.py"]}'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["discoveries"] == []  # default filled in by schema

	def test_parse_mc_result_degraded_integration(self, caplog: pytest.LogCaptureFixture) -> None:
		"""parse_mc_result returns degraded result for invalid status."""
		output = (
			'MC_RESULT:{"status":"invalid","commits":["abc"],'
			'"summary":"done","files_changed":["f.py"]}'
		)
		with caplog.at_level(logging.WARNING, logger="mission_control.session"):
			result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "failed"  # degraded default
		assert result["commits"] == ["abc"]  # valid field preserved
