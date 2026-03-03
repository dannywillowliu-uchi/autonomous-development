"""Tests for session spawning."""

from __future__ import annotations

import logging

import pytest

from mission_control.session import (
	build_branch_name,
	parse_mc_result,
	validate_mc_result,
)


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

	def test_status_alias_success(self) -> None:
		"""Status 'success' normalizes to 'completed'."""
		raw = {
			"status": "success",
			"commits": [],
			"summary": "done",
			"files_changed": [],
		}
		result = validate_mc_result(raw)
		assert result["status"] == "completed"

	def test_status_alias_failure(self) -> None:
		"""Status 'failure' normalizes to 'failed'."""
		raw = {
			"status": "failure",
			"commits": [],
			"summary": "oops",
			"files_changed": [],
		}
		result = validate_mc_result(raw)
		assert result["status"] == "failed"

	def test_status_alias_error(self) -> None:
		"""Status 'error' normalizes to 'failed'."""
		raw = {
			"status": "error",
			"commits": [],
			"summary": "oops",
			"files_changed": [],
		}
		result = validate_mc_result(raw)
		assert result["status"] == "failed"

	def test_files_modified_alias(self) -> None:
		"""'files_modified' is normalized to 'files_changed'."""
		raw = {
			"status": "completed",
			"commits": ["abc"],
			"summary": "done",
			"files_modified": ["src/a.py", "src/b.py"],
		}
		result = validate_mc_result(raw)
		assert result["files_changed"] == ["src/a.py", "src/b.py"]

	def test_files_modified_does_not_override_files_changed(self) -> None:
		"""'files_modified' is ignored when 'files_changed' is already present."""
		raw = {
			"status": "completed",
			"commits": [],
			"summary": "done",
			"files_changed": ["real.py"],
			"files_modified": ["ignored.py"],
		}
		result = validate_mc_result(raw)
		assert result["files_changed"] == ["real.py"]

	def test_degraded_invalid_commits_type(self, caplog: pytest.LogCaptureFixture) -> None:
		"""Invalid type for commits triggers degraded parsing; valid fields preserved."""
		raw = {
			"status": "completed",
			"commits": "not-a-list",
			"summary": "done",
			"files_changed": ["f.py"],
		}
		with caplog.at_level(logging.WARNING, logger="mission_control.session"):
			result = validate_mc_result(raw)
		assert result["status"] == "completed"
		assert result["commits"] == []  # degraded default
		assert result["summary"] == "done"

	def test_degraded_invalid_files_changed_type(self, caplog: pytest.LogCaptureFixture) -> None:
		"""Invalid files_changed triggers degraded parsing; other fields preserved."""
		raw = {
			"status": "completed",
			"commits": ["abc"],
			"summary": "done",
			"files_changed": 123,
		}
		with caplog.at_level(logging.WARNING, logger="mission_control.session"):
			result = validate_mc_result(raw)
		assert result["status"] == "completed"
		assert result["commits"] == ["abc"]
		assert result["files_changed"] == []  # degraded default


class TestParseMcResultEdgeCases:
	"""Edge-case tests for parse_mc_result."""

	def test_multiline_json_with_nested_braces(self) -> None:
		"""Multiline JSON with nested objects/arrays parses correctly."""
		output = (
			"log line 1\n"
			'MC_RESULT:{\n'
			'  "status": "completed",\n'
			'  "commits": ["abc123", "def456"],\n'
			'  "summary": "Refactored {braces} in code",\n'
			'  "files_changed": ["src/foo.py"],\n'
			'  "discoveries": ["found {nested} pattern"],\n'
			'  "concerns": []\n'
			"}\n"
			"trailing output"
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["commits"] == ["abc123", "def456"]
		assert result["summary"] == "Refactored {braces} in code"
		assert result["discoveries"] == ["found {nested} pattern"]

	def test_multiple_markers_picks_last(self) -> None:
		"""When output has multiple MC_RESULT markers, the last one wins."""
		output = (
			'MC_RESULT:{"status":"failed","commits":[],"summary":"attempt 1","files_changed":[]}\n'
			"Retrying after error...\n"
			'MC_RESULT:{"status":"blocked","commits":[],"summary":"attempt 2","files_changed":[]}\n'
			"One more try...\n"
			'MC_RESULT:{"status":"completed","commits":["final"],"summary":"attempt 3","files_changed":["ok.py"]}'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["summary"] == "attempt 3"
		assert result["commits"] == ["final"]

	def test_malformed_json_returns_none(self) -> None:
		"""Malformed JSON after MC_RESULT marker returns None."""
		output = 'MC_RESULT:{status: not valid json, missing quotes}'
		result = parse_mc_result(output)
		assert result is None

	def test_truncated_json_returns_none(self) -> None:
		"""Truncated JSON (no closing brace) returns None."""
		output = 'MC_RESULT:{"status":"completed","commits":["abc"],"summary":"tru'
		result = parse_mc_result(output)
		assert result is None

	def test_empty_json_object_returns_degraded(self, caplog: pytest.LogCaptureFixture) -> None:
		"""Empty JSON object after MC_RESULT triggers degraded parsing."""
		output = "MC_RESULT:{}"
		with caplog.at_level(logging.WARNING, logger="mission_control.session"):
			result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "failed"
		assert result["commits"] == []
		assert result["summary"] == ""

	def test_status_alias_via_parse(self) -> None:
		"""Status aliases are normalized through the full parse_mc_result path."""
		output = (
			'MC_RESULT:{"status":"success","commits":["abc"],'
			'"summary":"done","files_changed":["f.py"]}'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"

	def test_files_modified_alias_via_parse(self) -> None:
		"""files_modified alias is normalized through the full parse_mc_result path."""
		output = (
			'MC_RESULT:{"status":"completed","commits":[],'
			'"summary":"done","files_modified":["a.py","b.py"]}'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["files_changed"] == ["a.py", "b.py"]

	def test_mc_result_with_surrounding_noise(self) -> None:
		"""MC_RESULT embedded in noisy output with ANSI codes and garbage."""
		output = (
			"\x1b[32mDone!\x1b[0m\n"
			"Some random log output here\n"
			'MC_RESULT:{"status":"completed","commits":["xyz"],'
			'"summary":"worked","files_changed":["main.py"]}\n'
			"\x1b[0mEnd of session"
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["commits"] == ["xyz"]
