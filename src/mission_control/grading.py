"""Algorithmic grading of planner decomposition quality (no LLM)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mission_control.constants import GRADING_WEIGHTS
from mission_control.models import DecompositionGrade, UnitReview, WorkUnit

if TYPE_CHECKING:
	pass

logger = logging.getLogger(__name__)


def compute_decomposition_grade(
	units: list[WorkUnit],
	reviews: list[UnitReview],
	plan_id: str = "",
	epoch_id: str = "",
	mission_id: str = "",
) -> DecompositionGrade:
	"""Compute a decomposition quality grade for an epoch's units.

	Weights are defined in constants.GRADING_WEIGHTS.
	"""
	unit_count = len(units)
	if unit_count == 0:
		return DecompositionGrade(
			plan_id=plan_id,
			epoch_id=epoch_id,
			mission_id=mission_id,
			unit_count=0,
			composite_score=0.0,
		)

	# Completion rate
	completed = sum(1 for u in units if u.status == "completed")
	completion_rate = completed / unit_count

	# Retry rate: units with attempt > 1 (needed a retry)
	retried = sum(1 for u in units if u.attempt > 1)
	retry_rate = retried / unit_count

	# Overlap rate: fraction of units that share files_hint with another unit
	file_sets: list[set[str]] = []
	for u in units:
		files = {f.strip() for f in u.files_hint.split(",") if f.strip()} if u.files_hint else set()
		file_sets.append(files)

	overlap_count = 0
	for i in range(len(file_sets)):
		for j in range(i + 1, len(file_sets)):
			if file_sets[i] & file_sets[j]:
				overlap_count += 1
				break  # count each unit at most once
	overlap_rate = overlap_count / unit_count if unit_count > 0 else 0.0

	# Average review score (normalized to 0-1 by dividing by 10)
	if reviews:
		avg_review = sum(r.avg_score for r in reviews) / len(reviews)
	else:
		avg_review = 5.0  # neutral default when no reviews available
	avg_review_normalized = avg_review / 10.0

	# Composite score
	w_review, w_retry, w_overlap, w_completion = GRADING_WEIGHTS
	composite = (
		w_review * avg_review_normalized
		+ w_retry * (1.0 - retry_rate)
		+ w_overlap * (1.0 - overlap_rate)
		+ w_completion * completion_rate
	)
	composite = round(composite, 3)

	return DecompositionGrade(
		plan_id=plan_id,
		epoch_id=epoch_id,
		mission_id=mission_id,
		avg_review_score=round(avg_review, 1),
		retry_rate=round(retry_rate, 3),
		overlap_rate=round(overlap_rate, 3),
		completion_rate=round(completion_rate, 3),
		composite_score=composite,
		unit_count=unit_count,
	)


def format_decomposition_feedback(grade: DecompositionGrade) -> str:
	"""Format a DecompositionGrade into planner-readable feedback."""
	lines = [
		"## Decomposition Quality (Previous Epoch)",
		f"Composite: {grade.composite_score:.2f} | "
		f"Reviews: {grade.avg_review_score:.1f}/10 | "
		f"Retry: {grade.retry_rate:.0%} | "
		f"Overlap: {grade.overlap_rate:.0%} | "
		f"Completion: {grade.completion_rate:.0%}",
	]

	# Identify weakest area
	areas = {
		"retry_rate": grade.retry_rate,
		"overlap_rate": grade.overlap_rate,
		"completion_rate": 1.0 - grade.completion_rate,
		"review_scores": 1.0 - (grade.avg_review_score / 10.0),
	}
	weakest = max(areas, key=areas.get)  # type: ignore[arg-type]
	advice = {
		"retry_rate": "decompose units more independently to reduce retries",
		"overlap_rate": "reduce file overlap between units to avoid conflicts",
		"completion_rate": "plan smaller, more achievable units",
		"review_scores": "focus on meaningful implementations with substantive tests",
	}
	if areas[weakest] > 0.1:
		lines.append(f"Weakest area: {weakest} -- {advice[weakest]}")

	return "\n".join(lines)
