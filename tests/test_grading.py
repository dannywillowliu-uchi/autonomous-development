"""Tests for decomposition grading formula and edge cases."""

from __future__ import annotations

from mission_control.grading import compute_decomposition_grade, format_decomposition_feedback
from mission_control.models import UnitReview, WorkUnit


def _make_unit(
	uid: str = "u1",
	status: str = "completed",
	attempt: int = 1,
	files_hint: str = "",
) -> WorkUnit:
	return WorkUnit(
		id=uid, plan_id="p1", title=f"Unit {uid}",
		status=status, attempt=attempt, files_hint=files_hint,
	)


def _make_review(avg: float = 7.0) -> UnitReview:
	return UnitReview(avg_score=avg, alignment_score=7, approach_score=7, test_score=7)


class TestComputeDecompositionGrade:
	def test_empty_units(self) -> None:
		grade = compute_decomposition_grade([], [])
		assert grade.unit_count == 0
		assert grade.composite_score == 0.0

	def test_perfect_score(self) -> None:
		"""All completed, no retries, no overlap, perfect reviews."""
		units = [
			_make_unit("u1", files_hint="a.py"),
			_make_unit("u2", files_hint="b.py"),
		]
		reviews = [_make_review(10.0), _make_review(10.0)]
		grade = compute_decomposition_grade(units, reviews)
		assert grade.completion_rate == 1.0
		assert grade.retry_rate == 0.0
		assert grade.overlap_rate == 0.0
		assert grade.avg_review_score == 10.0
		# 0.30*1.0 + 0.25*1.0 + 0.25*1.0 + 0.20*1.0 = 1.0
		assert grade.composite_score == 1.0

	def test_all_retries(self) -> None:
		"""All units needed retries."""
		units = [
			_make_unit("u1", attempt=2),
			_make_unit("u2", attempt=3),
		]
		grade = compute_decomposition_grade(units, [])
		assert grade.retry_rate == 1.0

	def test_all_failed(self) -> None:
		"""All units failed."""
		units = [
			_make_unit("u1", status="failed"),
			_make_unit("u2", status="failed"),
		]
		grade = compute_decomposition_grade(units, [])
		assert grade.completion_rate == 0.0

	def test_file_overlap(self) -> None:
		"""Units sharing files should produce overlap_rate > 0."""
		units = [
			_make_unit("u1", files_hint="shared.py,a.py"),
			_make_unit("u2", files_hint="shared.py,b.py"),
			_make_unit("u3", files_hint="c.py"),
		]
		grade = compute_decomposition_grade(units, [])
		# u1 overlaps with u2 -> overlap_count = 1 (u1 triggers break)
		# overlap_rate = 1/3
		assert grade.overlap_rate > 0.0
		assert grade.overlap_rate < 1.0

	def test_no_reviews_uses_neutral_default(self) -> None:
		"""Without reviews, avg_review defaults to 5.0."""
		units = [_make_unit("u1")]
		grade = compute_decomposition_grade(units, [])
		assert grade.avg_review_score == 5.0

	def test_mixed_scenario(self) -> None:
		"""Realistic mixed scenario."""
		units = [
			_make_unit("u1", files_hint="models.py"),
			_make_unit("u2", status="failed", attempt=2, files_hint="db.py"),
			_make_unit("u3", files_hint="models.py,api.py"),  # overlaps with u1
		]
		reviews = [_make_review(8.0)]
		grade = compute_decomposition_grade(units, reviews)
		assert grade.unit_count == 3
		assert grade.completion_rate > 0  # u1 and u3 completed
		assert grade.retry_rate > 0  # u2 retried
		assert grade.overlap_rate > 0  # u1 and u3 share models.py
		assert 0.0 < grade.composite_score < 1.0

	def test_stores_metadata(self) -> None:
		units = [_make_unit("u1")]
		grade = compute_decomposition_grade(
			units, [], plan_id="p1", epoch_id="ep1", mission_id="m1",
		)
		assert grade.plan_id == "p1"
		assert grade.epoch_id == "ep1"
		assert grade.mission_id == "m1"


class TestFormatDecompositionFeedback:
	def test_formats_correctly(self) -> None:
		from mission_control.models import DecompositionGrade
		grade = DecompositionGrade(
			avg_review_score=7.2, retry_rate=0.2,
			overlap_rate=0.1, completion_rate=1.0,
			composite_score=0.72,
		)
		text = format_decomposition_feedback(grade)
		assert "Composite: 0.72" in text
		assert "Reviews: 7.2/10" in text
		assert "Retry: 20%" in text

	def test_identifies_weakest_area(self) -> None:
		from mission_control.models import DecompositionGrade
		grade = DecompositionGrade(
			avg_review_score=9.0, retry_rate=0.8,
			overlap_rate=0.0, completion_rate=1.0,
			composite_score=0.5,
		)
		text = format_decomposition_feedback(grade)
		assert "retry_rate" in text
		assert "decompose units more independently" in text
