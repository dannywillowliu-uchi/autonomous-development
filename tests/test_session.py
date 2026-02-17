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
