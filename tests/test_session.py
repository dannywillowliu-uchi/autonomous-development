"""Tests for session spawning."""

from __future__ import annotations

import json
import logging

import pytest

from autodev.session import (
	build_branch_name,
	extract_fallback_handoff,
	extract_text_from_stream_json,
	parse_mc_result,
	validate_mc_result,
)


class TestParseMcResult:
	def test_valid_result(self) -> None:
		output = (
			"Some output\n"
			'AD_RESULT:{"status":"completed","commits":["abc123"],'
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
		"""AD_RESULT with pretty-printed multiline JSON should parse correctly."""
		output = (
			"Some output\n"
			"AD_RESULT:{\n"
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
		"""When multiple AD_RESULT markers exist, use the last one."""
		output = (
			'AD_RESULT:{"status":"failed","commits":[],"summary":"first attempt"}\n'
			"Retrying...\n"
			'AD_RESULT:{"status":"completed","commits":["def456"],"summary":"second attempt"}'
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
		"""Fully valid AD_RESULT dict passes validation unchanged."""
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
		with caplog.at_level(logging.WARNING, logger="autodev.session"):
			result = validate_mc_result(raw)
		assert result["status"] == "failed"  # degraded default
		assert result["commits"] == ["abc123"]  # valid field preserved
		assert result["summary"] == "Did it"  # valid field preserved
		assert "schema validation failed" in caplog.text.lower()

	def test_completely_invalid_input_returns_degraded(self, caplog: pytest.LogCaptureFixture) -> None:
		"""Completely invalid dict returns all defaults with warning."""
		raw = {"garbage": True, "number": 42}
		with caplog.at_level(logging.WARNING, logger="autodev.session"):
			result = validate_mc_result(raw)
		assert result["status"] == "failed"
		assert result["commits"] == []
		assert result["summary"] == ""
		assert result["files_changed"] == []
		assert result["discoveries"] == []
		assert result["concerns"] == []
		assert "schema validation failed" in caplog.text.lower()

	def test_parse_mc_result_integration(self) -> None:
		"""parse_mc_result returns validated data for valid AD_RESULT output."""
		output = (
			'AD_RESULT:{"status":"completed","commits":["abc"],'
			'"summary":"done","files_changed":["f.py"]}'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["discoveries"] == []  # default filled in by schema

	def test_parse_mc_result_degraded_integration(self, caplog: pytest.LogCaptureFixture) -> None:
		"""parse_mc_result returns degraded result for invalid status."""
		output = (
			'AD_RESULT:{"status":"invalid","commits":["abc"],'
			'"summary":"done","files_changed":["f.py"]}'
		)
		with caplog.at_level(logging.WARNING, logger="autodev.session"):
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
		with caplog.at_level(logging.WARNING, logger="autodev.session"):
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
		with caplog.at_level(logging.WARNING, logger="autodev.session"):
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
			'AD_RESULT:{\n'
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
		"""When output has multiple AD_RESULT markers, the last one wins."""
		output = (
			'AD_RESULT:{"status":"failed","commits":[],"summary":"attempt 1","files_changed":[]}\n'
			"Retrying after error...\n"
			'AD_RESULT:{"status":"blocked","commits":[],"summary":"attempt 2","files_changed":[]}\n'
			"One more try...\n"
			'AD_RESULT:{"status":"completed","commits":["final"],"summary":"attempt 3","files_changed":["ok.py"]}'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["summary"] == "attempt 3"
		assert result["commits"] == ["final"]

	def test_malformed_json_returns_none(self) -> None:
		"""Malformed JSON after AD_RESULT marker returns None."""
		output = 'AD_RESULT:{status: not valid json, missing quotes}'
		result = parse_mc_result(output)
		assert result is None

	def test_truncated_json_returns_none(self) -> None:
		"""Truncated JSON (no closing brace) returns None."""
		output = 'AD_RESULT:{"status":"completed","commits":["abc"],"summary":"tru'
		result = parse_mc_result(output)
		assert result is None

	def test_empty_json_object_returns_degraded(self, caplog: pytest.LogCaptureFixture) -> None:
		"""Empty JSON object after AD_RESULT triggers degraded parsing."""
		output = "AD_RESULT:{}"
		with caplog.at_level(logging.WARNING, logger="autodev.session"):
			result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "failed"
		assert result["commits"] == []
		assert result["summary"] == ""

	def test_status_alias_via_parse(self) -> None:
		"""Status aliases are normalized through the full parse_mc_result path."""
		output = (
			'AD_RESULT:{"status":"success","commits":["abc"],'
			'"summary":"done","files_changed":["f.py"]}'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"

	def test_files_modified_alias_via_parse(self) -> None:
		"""files_modified alias is normalized through the full parse_mc_result path."""
		output = (
			'AD_RESULT:{"status":"completed","commits":[],'
			'"summary":"done","files_modified":["a.py","b.py"]}'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["files_changed"] == ["a.py", "b.py"]

	def test_mc_result_with_surrounding_noise(self) -> None:
		"""AD_RESULT embedded in noisy output with ANSI codes and garbage."""
		output = (
			"\x1b[32mDone!\x1b[0m\n"
			"Some random log output here\n"
			'AD_RESULT:{"status":"completed","commits":["xyz"],'
			'"summary":"worked","files_changed":["main.py"]}\n'
			"\x1b[0mEnd of session"
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["commits"] == ["xyz"]


class TestExtractFallbackHandoff:
	"""Tests for extract_fallback_handoff() -- recovers data when AD_RESULT missing."""

	def test_no_mc_result_with_commits(self) -> None:
		"""Extract commits and changed files from git output."""
		output = (
			"Working on the task...\n"
			"[mc/unit-abc123 f3dea4e] feat: add fallback extraction\n"
			" src/session.py | 42 +++++++++\n"
			" tests/test_session.py | 18 ++++\n"
			" 2 files changed, 60 insertions(+)\n"
			"commit f3dea4e26a9e0b1c2d3e4f5a6b7c8d9e0f1a2b3c\n"
			"Done.\n"
		)
		result = extract_fallback_handoff(output, exit_code=0)
		assert result["status"] == "completed"
		assert "f3dea4e26a9e0b1c2d3e4f5a6b7c8d9e0f1a2b3c" in result["commits"]
		assert "f3dea4e" in result["commits"]
		assert "src/session.py" in result["files_changed"]
		assert "tests/test_session.py" in result["files_changed"]
		assert result["discoveries"] == []
		assert len(result["concerns"]) == 1

	def test_no_mc_result_with_failures_syntax_error(self) -> None:
		"""Classify as failed when output contains SyntaxError."""
		output = (
			"Running tests...\n"
			"  File \"src/foo.py\", line 10\n"
			"    def broken(\n"
			"              ^\n"
			"SyntaxError: unexpected EOF while parsing\n"
		)
		result = extract_fallback_handoff(output, exit_code=1)
		assert result["status"] == "failed"
		assert result["commits"] == []
		assert "[Failed]" in result["summary"]

	def test_no_mc_result_with_failures_test_failure(self) -> None:
		"""Classify as failed when output contains FAILED test markers."""
		output = (
			"=== test session starts ===\n"
			"tests/test_foo.py::test_bar FAILED\n"
			"AssertionError: expected 1, got 2\n"
			"1 failed, 3 passed\n"
		)
		result = extract_fallback_handoff(output, exit_code=1)
		assert result["status"] == "failed"

	def test_no_mc_result_with_failures_import_error(self) -> None:
		"""Classify as failed when output contains ImportError."""
		output = "ModuleNotFoundError: No module named 'nonexistent'\n"
		result = extract_fallback_handoff(output, exit_code=1)
		assert result["status"] == "failed"

	def test_no_mc_result_with_failures_merge_conflict(self) -> None:
		"""Classify as failed on merge conflict markers."""
		output = "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\n"
		result = extract_fallback_handoff(output, exit_code=1)
		assert result["status"] == "failed"

	def test_no_mc_result_with_failures_timeout(self) -> None:
		"""Classify as failed on timeout signatures."""
		output = "Process timed out after 300s\n"
		result = extract_fallback_handoff(output, exit_code=None)
		assert result["status"] == "failed"

	def test_empty_output(self) -> None:
		"""Empty output returns failed with no commits or files."""
		result = extract_fallback_handoff("", exit_code=None)
		assert result["status"] == "failed"
		assert result["commits"] == []
		assert result["files_changed"] == []
		assert result["summary"] == "No output captured"

	def test_empty_output_with_zero_exit(self) -> None:
		"""Empty output with exit_code=0 still returns completed."""
		result = extract_fallback_handoff("", exit_code=0)
		assert result["status"] == "completed"
		assert result["summary"] == "No output captured"

	def test_exit_code_zero_overrides_no_errors(self) -> None:
		"""Exit code 0 classifies as completed even with ambiguous output."""
		output = "Some random output\nAll done.\n"
		result = extract_fallback_handoff(output, exit_code=0)
		assert result["status"] == "completed"
		assert "[Fallback]" in result["summary"]

	def test_nonzero_exit_code_with_no_patterns(self) -> None:
		"""Non-zero exit code with no recognized error patterns → failed."""
		output = "Something went wrong but no recognizable pattern.\n"
		result = extract_fallback_handoff(output, exit_code=1)
		assert result["status"] == "failed"

	def test_multiple_commits_deduplicated(self) -> None:
		"""Same commit hash appearing multiple times is deduplicated."""
		output = (
			"commit abc1234567890\n"
			"Author: test\n"
			"commit abc1234567890\n"
		)
		result = extract_fallback_handoff(output, exit_code=0)
		assert result["commits"] == ["abc1234567890"]

	def test_diff_stat_extraction(self) -> None:
		"""Multiple file paths from diff --stat are all extracted."""
		output = (
			" src/models.py        | 15 +++++++++------\n"
			" src/config.py        |  3 +--\n"
			" tests/test_models.py | 22 ++++++++++++++++++++++\n"
			" 3 files changed, 28 insertions(+), 8 deletions(-)\n"
		)
		result = extract_fallback_handoff(output, exit_code=0)
		assert "src/models.py" in result["files_changed"]
		assert "src/config.py" in result["files_changed"]
		assert "tests/test_models.py" in result["files_changed"]
		assert len(result["files_changed"]) == 3

	def test_returns_all_handoff_keys(self) -> None:
		"""Result dict has all keys expected by Handoff model."""
		result = extract_fallback_handoff("output", exit_code=0)
		for key in ("status", "commits", "summary", "files_changed", "discoveries", "concerns"):
			assert key in result

	def test_lint_error_detected(self) -> None:
		"""Ruff lint errors in output classify as failed."""
		output = (
			"ruff check src/\n"
			"src/foo.py:10:1: F401 `os` imported but unused\n"
			"Found 1 error.\n"
		)
		result = extract_fallback_handoff(output, exit_code=1)
		assert result["status"] == "failed"


class TestExtractTextFromStreamJson:
	"""Tests for extract_text_from_stream_json -- extracts plain text from NDJSON output."""

	def test_result_event(self) -> None:
		"""Extracts text from a stream-json 'result' event."""
		ad_result = 'AD_RESULT:{"status":"completed","commits":[],"summary":"done","files_changed":[]}'
		result_event = json.dumps({"type": "result", "result": ad_result})
		output = f'{{"type":"system","subtype":"init"}}\n{result_event}\n'
		text = extract_text_from_stream_json(output)
		assert "AD_RESULT:" in text
		assert '"status":"completed"' in text

	def test_assistant_text_blocks(self) -> None:
		"""Extracts text from assistant message text blocks."""
		ad_json = '{"status":"completed","commits":[],"summary":"done","files_changed":[]}'
		event = json.dumps({
			"type": "assistant",
			"message": {
				"content": [
					{"type": "text", "text": "Working on the task..."},
					{"type": "text", "text": f"AD_RESULT:{ad_json}"},
				],
			},
		})
		text = extract_text_from_stream_json(event)
		assert "AD_RESULT:" in text
		assert "Working on the task" in text

	def test_result_event_preferred_over_assistant(self) -> None:
		"""Result event text is preferred over assistant text blocks."""
		assistant = json.dumps({
			"type": "assistant",
			"message": {"content": [{"type": "text", "text": "intermediate text"}]},
		})
		result = json.dumps({"type": "result", "result": "final result text"})
		output = f"{assistant}\n{result}\n"
		text = extract_text_from_stream_json(output)
		assert text == "final result text"

	def test_trace_file_prefixes_stripped(self) -> None:
		"""[OUT] prefixes from trace files are stripped before parsing."""
		event = json.dumps({"type": "result", "result": "AD_RESULT:{}"})
		output = f"[OUT] {event}\n"
		text = extract_text_from_stream_json(output)
		assert "AD_RESULT:" in text

	def test_non_json_lines_ignored(self) -> None:
		"""Non-JSON lines (headers, garbage) are silently skipped."""
		output = (
			"# Agent: test-agent\n"
			"# Format: stream-json\n"
			"\n"
			'{"type":"result","result":"hello"}\n'
			"not json at all\n"
		)
		text = extract_text_from_stream_json(output)
		assert text == "hello"

	def test_empty_output(self) -> None:
		"""Empty or whitespace output returns empty string."""
		assert extract_text_from_stream_json("") == ""
		assert extract_text_from_stream_json("  \n\n  ") == ""

	def test_no_text_events(self) -> None:
		"""Output with only system events (no text) returns empty string."""
		output = '{"type":"system","subtype":"init"}\n{"type":"system","subtype":"hook_started"}\n'
		text = extract_text_from_stream_json(output)
		assert text == ""


class TestParseMcResultStreamJson:
	"""Tests for parse_mc_result with stream-json (NDJSON) formatted output.

	This is the critical bug scenario: when --output-format stream-json is used,
	AD_RESULT markers are embedded inside JSON string fields where quotes are
	escaped as \\". The brace-counting parser fails because it enters a "string"
	state and never exits (all \\" look escaped). parse_mc_result must fall back
	to extracting text from the stream-json events.
	"""

	def test_ad_result_in_result_event(self) -> None:
		"""AD_RESULT in a stream-json 'result' event is extracted correctly."""
		ad_json = '{"status":"completed","commits":["abc"],"summary":"done","files_changed":["f.py"]}'
		ad_text = f"AD_RESULT:{ad_json}"
		result_event = json.dumps({"type": "result", "result": ad_text})
		output = (
			'{"type":"system","subtype":"init"}\n'
			f"{result_event}\n"
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["commits"] == ["abc"]
		assert result["summary"] == "done"

	def test_ad_result_in_assistant_text(self) -> None:
		"""AD_RESULT in assistant message text is extracted from stream-json."""
		ad_json = '{"status":"completed","commits":[],"summary":"fixed it","files_changed":["a.py"]}'
		event = json.dumps({
			"type": "assistant",
			"message": {
				"content": [{"type": "text", "text": f"AD_RESULT:{ad_json}"}],
			},
		})
		output = f"{event}\n"
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["summary"] == "fixed it"

	def test_stream_json_with_only_hooks_returns_none(self) -> None:
		"""Stream-json output with only hook events (no text) returns None."""
		output = (
			'{"type":"system","subtype":"hook_started","hook_name":"SessionStart:startup"}\n'
			'{"type":"system","subtype":"hook_response","exit_code":0}\n'
		)
		result = parse_mc_result(output)
		assert result is None

	def test_stream_json_with_trace_prefixes(self) -> None:
		"""Stream-json output with [OUT] trace prefixes is handled."""
		ad_json = '{"status":"blocked","commits":[],"summary":"stuck","files_changed":[],"concerns":["blocked"]}'
		result_event = json.dumps({"type": "result", "result": f"AD_RESULT:{ad_json}"})
		output = (
			"[OUT] " + '{"type":"system","subtype":"init"}\n'
			"[OUT] " + f"{result_event}\n"
			"[ERR] some stderr\n"
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "blocked"

	def test_plain_text_still_works(self) -> None:
		"""Plain text output (non-NDJSON) still works after adding stream-json fallback."""
		output = (
			"Working on task...\n"
			"Done.\n"
			'AD_RESULT:{"status":"completed","commits":["xyz"],"summary":"all good","files_changed":["b.py"]}\n'
		)
		result = parse_mc_result(output)
		assert result is not None
		assert result["status"] == "completed"
		assert result["commits"] == ["xyz"]
