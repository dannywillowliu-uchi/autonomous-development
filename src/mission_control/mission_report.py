"""Post-mission report generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mission_control.config import MissionConfig
from mission_control.continuous_controller import ContinuousMissionResult
from mission_control.db import Database
from mission_control.models import Mission

logger = logging.getLogger(__name__)


def generate_mission_report(
	result: ContinuousMissionResult,
	mission: Mission,
	db: Database,
	config: MissionConfig,
) -> dict[str, Any]:
	"""Build a structured mission report and write it to disk.

	Returns the report dict for programmatic use / testing.
	"""
	# Collect timeline events
	events = db.get_unit_events_for_mission(mission.id, limit=10_000)
	timeline = [
		{
			"timestamp": e.timestamp,
			"event_type": e.event_type,
			"work_unit_id": e.work_unit_id,
			"details": e.details,
		}
		for e in events
	]
	timeline.sort(key=lambda e: e["timestamp"])

	# Collect files_changed from handoffs
	handoffs = db.get_recent_handoffs(mission.id, limit=10_000)
	all_files: set[str] = set()
	for h in handoffs:
		all_files.update(h.files_changed)

	# Collect quality data
	reviews = db.get_unit_reviews_for_mission(mission.id)
	review_data = [
		{
			"work_unit_id": r.work_unit_id,
			"alignment": r.alignment_score,
			"approach": r.approach_score,
			"tests": r.test_score,
			"avg": r.avg_score,
			"rationale": r.rationale,
		}
		for r in reviews
	]

	ratings = db.get_trajectory_ratings_for_mission(mission.id)
	rating_data = [
		{"rating": r.rating, "feedback": r.feedback, "timestamp": r.timestamp}
		for r in ratings
	]

	grades = db.get_decomposition_grades_for_mission(mission.id)
	grade_data = [
		{
			"epoch_id": g.epoch_id,
			"composite_score": g.composite_score,
			"avg_review_score": g.avg_review_score,
			"retry_rate": g.retry_rate,
			"overlap_rate": g.overlap_rate,
			"completion_rate": g.completion_rate,
			"unit_count": g.unit_count,
		}
		for g in grades
	]

	report: dict[str, Any] = {
		"objective": mission.objective,
		"outcome": {
			"objective_met": result.objective_met,
			"stopped_reason": result.stopped_reason,
		},
		"mission_id": mission.id,
		"units_dispatched": result.total_units_dispatched,
		"units_merged": result.total_units_merged,
		"units_failed": result.total_units_failed,
		"wall_time_seconds": result.wall_time_seconds,
		"total_cost_usd": mission.total_cost_usd,
		"files_changed": sorted(all_files),
		"verification_passed": result.final_verification_passed,
		"verification_output": result.final_verification_output,
		"backlog_item_ids": result.backlog_item_ids or [],
		"unit_reviews": review_data,
		"trajectory_ratings": rating_data,
		"decomposition_grades": grade_data,
		"timeline": timeline,
	}

	# Write to disk
	report_path = Path(config.target.resolved_path) / "mission_report.json"
	try:
		report_path.write_text(json.dumps(report, indent=2) + "\n")
		logger.info("Mission report written to %s", report_path)
	except OSError as exc:
		logger.error("Failed to write mission report: %s", exc)

	return report
