"""Tests for the LLM diff reviewer -- prompt building and result parsing."""

from __future__ import annotations

from mission_control.diff_reviewer import (
	_build_review_prompt,
	_clamp_score,
	_parse_review_output,
	compute_review_trend,
	is_quality_declining,
)
from mission_control.models import UnitReview, WorkUnit


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

	def test_regex_fallback_when_extract_json_fails(self) -> None:
		"""When extract_json_from_text fails, regex fallback should parse single-line JSON."""
		# Simulate output where the JSON is on the same line as the marker
		# but wrapped in a way that extract_json_from_text can't handle
		output = 'REVIEW_RESULT:{"alignment": 6, "approach": 7, "test_quality": 5, "rationale": "ok"}'
		data = _parse_review_output(output)
		assert data is not None
		assert data["alignment"] == 6
		assert data["approach"] == 7
		assert data["test_quality"] == 5

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


def _review(alignment: int = 5, approach: int = 5, test_quality: int = 5) -> UnitReview:
	"""Helper to build a UnitReview with specific dimension scores."""
	return UnitReview(
		alignment_score=alignment,
		approach_score=approach,
		test_score=test_quality,
		avg_score=round((alignment + approach + test_quality) / 3, 1),
	)


class TestComputeReviewTrend:
	def test_empty_reviews(self) -> None:
		result = compute_review_trend([])
		assert result["overall_avg"] == 0.0
		assert result["recent_avg"] == 0.0
		assert result["trend"] == "stable"
		assert result["worst_dimension"] == "alignment"

	def test_single_review(self) -> None:
		reviews = [_review(7, 8, 6)]
		result = compute_review_trend(reviews, window=3)
		# Single review: overall == recent, so trend is stable
		assert result["overall_avg"] == result["recent_avg"]
		assert result["trend"] == "stable"
		assert result["worst_dimension"] == "test_quality"

	def test_stable_quality(self) -> None:
		reviews = [_review(7, 7, 7) for _ in range(6)]
		result = compute_review_trend(reviews, window=3)
		assert result["trend"] == "stable"
		assert result["overall_avg"] == result["recent_avg"]

	def test_declining_quality(self) -> None:
		# First 4 reviews are strong, last 3 are weak
		reviews = [_review(9, 9, 9)] * 4 + [_review(3, 3, 3)] * 3
		result = compute_review_trend(reviews, window=3)
		assert result["trend"] == "declining"
		assert result["recent_avg"] < result["overall_avg"]

	def test_improving_quality(self) -> None:
		# First 4 reviews are weak, last 3 are strong
		reviews = [_review(3, 3, 3)] * 4 + [_review(9, 9, 9)] * 3
		result = compute_review_trend(reviews, window=3)
		assert result["trend"] == "improving"
		assert result["recent_avg"] > result["overall_avg"]

	def test_worst_dimension_identification(self) -> None:
		# Recent reviews have low test_quality
		reviews = [_review(8, 8, 2)] * 3
		result = compute_review_trend(reviews, window=3)
		assert result["worst_dimension"] == "test_quality"

		# Recent reviews have low approach
		reviews = [_review(8, 2, 8)] * 3
		result = compute_review_trend(reviews, window=3)
		assert result["worst_dimension"] == "approach"

		# Recent reviews have low alignment
		reviews = [_review(2, 8, 8)] * 3
		result = compute_review_trend(reviews, window=3)
		assert result["worst_dimension"] == "alignment"


class TestIsQualityDeclining:
	def test_empty_reviews(self) -> None:
		assert is_quality_declining([]) is False

	def test_stable_quality_not_declining(self) -> None:
		reviews = [_review(7, 7, 7)] * 6
		assert is_quality_declining(reviews) is False

	def test_detects_decline(self) -> None:
		reviews = [_review(9, 9, 9)] * 5 + [_review(3, 3, 3)] * 3
		assert is_quality_declining(reviews, window=3, threshold=1.5) is True

	def test_improving_not_declining(self) -> None:
		reviews = [_review(3, 3, 3)] * 5 + [_review(9, 9, 9)] * 3
		assert is_quality_declining(reviews) is False

	def test_single_review_not_declining(self) -> None:
		reviews = [_review(5, 5, 5)]
		assert is_quality_declining(reviews) is False

	def test_threshold_sensitivity(self) -> None:
		# Small drop: declining at low threshold, not at high
		reviews = [_review(8, 8, 8)] * 5 + [_review(6, 6, 6)] * 3
		assert is_quality_declining(reviews, threshold=0.5) is True
		assert is_quality_declining(reviews, threshold=3.0) is False
