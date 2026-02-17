"""Tests for the LLM diff reviewer -- prompt building and result parsing."""

from __future__ import annotations

from mission_control.diff_reviewer import (
	_build_review_prompt,
	_clamp_score,
	_parse_review_output,
)
from mission_control.models import WorkUnit


class TestBuildReviewPrompt:
	def test_contains_unit_info(self) -> None:
		unit = WorkUnit(title="Add auth", description="Implement JWT auth")
		prompt = _build_review_prompt(unit, "diff content here", "Build auth system")
		assert "Add auth" in prompt
		assert "Implement JWT auth" in prompt
		assert "Build auth system" in prompt
		assert "diff content here" in prompt

	def test_truncates_long_diff(self) -> None:
		unit = WorkUnit(title="Big change")
		long_diff = "x" * 20000
		prompt = _build_review_prompt(unit, long_diff, "objective")
		# Diff should be truncated to 8000 chars
		assert len(prompt) < 20000


class TestParseReviewOutput:
	def test_valid_result(self) -> None:
		output = """Some reasoning here.

REVIEW_RESULT:{"alignment": 7, "approach": 8, "test_quality": 6, "rationale": "Clean but shallow tests"}"""
		data = _parse_review_output(output)
		assert data is not None
		assert data["alignment"] == 7
		assert data["approach"] == 8
		assert data["test_quality"] == 6
		assert "shallow" in data["rationale"]

	def test_malformed_json(self) -> None:
		output = "REVIEW_RESULT:{not valid json}"
		data = _parse_review_output(output)
		assert data is None

	def test_missing_marker(self) -> None:
		output = "No marker here, just text."
		data = _parse_review_output(output)
		assert data is None

	def test_json_without_marker(self) -> None:
		output = '{"alignment": 5, "approach": 5, "test_quality": 5, "rationale": "ok"}'
		data = _parse_review_output(output)
		assert data is not None
		assert data["alignment"] == 5

	def test_partial_fields(self) -> None:
		output = 'REVIEW_RESULT:{"alignment": 9}'
		data = _parse_review_output(output)
		assert data is not None
		assert data["alignment"] == 9

	def test_marker_with_extra_text_after(self) -> None:
		output = """Analysis complete.
REVIEW_RESULT:{"alignment": 8, "approach": 7, "test_quality": 9, "rationale": "Excellent"}
"""
		data = _parse_review_output(output)
		assert data is not None
		assert data["alignment"] == 8


class TestClampScore:
	def test_normal_range(self) -> None:
		assert _clamp_score(5) == 5
		assert _clamp_score(1) == 1
		assert _clamp_score(10) == 10

	def test_below_minimum(self) -> None:
		assert _clamp_score(0) == 1
		assert _clamp_score(-5) == 1

	def test_above_maximum(self) -> None:
		assert _clamp_score(15) == 10
		assert _clamp_score(100) == 10

	def test_string_input(self) -> None:
		assert _clamp_score("7") == 7

	def test_invalid_input(self) -> None:
		assert _clamp_score("not a number") == 5
		assert _clamp_score(None) == 5

	def test_float_input(self) -> None:
		assert _clamp_score(7.8) == 7
