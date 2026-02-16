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
		try:
			files = json.loads(h.files_changed) if h.files_changed else []
		except (json.JSONDecodeError, TypeError):
			files = []
		if isinstance(files, list):
			all_files.update(f for f in files if isinstance(f, str))

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
