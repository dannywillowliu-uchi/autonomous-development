"""Planner context and mission state formatting -- extracted from continuous_controller."""

from __future__ import annotations

import logging
from pathlib import Path

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.grading import compute_decomposition_grade, format_decomposition_feedback
from mission_control.models import Mission

logger = logging.getLogger(__name__)


def build_planner_context(db: Database, mission_id: str) -> str:
	"""Build planner context from recent handoff summaries."""
	try:
		handoffs = db.get_recent_handoffs(mission_id, limit=15)
	except Exception as exc:
		logger.error("Failed to get recent handoffs: %s", exc)
		return ""

	if not handoffs:
		return ""

	lines = ["## Recent Handoff Summaries"]
	merged_count = 0
	failed_count = 0

	for h in reversed(handoffs):  # oldest first
		status_label = h.status or "unknown"
		lines.append(f"\n### Unit {h.work_unit_id[:8]} ({status_label})")
		if h.summary:
			lines.append(f"Summary: {h.summary}")

		discoveries = h.discoveries
		if discoveries:
			lines.append("Discoveries:")
			for d in discoveries[:5]:
				lines.append(f"  - {d}")

		concerns = h.concerns
		if concerns:
			lines.append("Concerns:")
			for c in concerns[:5]:
				lines.append(f"  - {c}")

		if status_label == "completed":
			merged_count += 1
		else:
			failed_count += 1

	lines.append(
		f"\nMerge stats: {merged_count} merged, {failed_count} failed "
		f"(of last {len(handoffs)} units)",
	)

	# Add explicit completed + in-flight unit lists so the planner doesn't re-plan them
	try:
		all_units = db.get_work_units_for_mission(mission_id)
		completed_units = [u for u in all_units if u.status == "completed"]
		if completed_units:
			lines.append("\n## Completed Work (DO NOT re-plan these)")
			for u in completed_units:
				files_part = f" (files: {u.files_hint})" if u.files_hint else ""
				lines.append(f"- {u.id[:8]}: \"{u.title}\"{files_part}")

		running_units = [u for u in all_units if u.status == "running"]
		if running_units:
			lines.append("\n## In-Flight Units (currently being worked on -- DO NOT duplicate)")
			for u in running_units:
				files_part = f" (files: {u.files_hint})" if u.files_hint else ""
				lines.append(f"- {u.id[:8]}: \"{u.title}\"{files_part}")

		if completed_units or running_units:
			lines.append(
				"\nIMPORTANT: Do NOT create units that duplicate completed or in-flight work above."
			)
	except Exception:
		pass

	# Append quality review aggregates
	try:
		reviews = db.get_unit_reviews_for_mission(mission_id)
		if reviews:
			# Get reviews from the most recent epoch
			recent = reviews[-10:]
			avg_align = sum(r.alignment_score for r in recent) / len(recent)
			avg_approach = sum(r.approach_score for r in recent) / len(recent)
			avg_tests = sum(r.test_score for r in recent) / len(recent)
			lines.append(
				f"\n## Quality Review Scores (last {len(recent)} units)"
			)
			lines.append(
				f"Average: alignment={avg_align:.1f}, "
				f"approach={avg_approach:.1f}, tests={avg_tests:.1f}"
			)
			# Identify weakest area
			scores = {
				"objective alignment": avg_align,
				"approach quality": avg_approach,
				"test meaningfulness": avg_tests,
			}
			weakest = min(scores, key=scores.get)  # type: ignore[arg-type]
			if scores[weakest] < 7.0:
				lines.append(
					f"Weak area: {weakest} -- prioritize improvement here"
				)
	except Exception:
		pass

	# Compute decomposition grade for the most recent completed epoch
	try:
		epochs = db.get_epochs_for_mission(mission_id)
		if epochs:
			latest_epoch = epochs[-1]
			all_units = db.get_work_units_for_mission(mission_id)
			epoch_units = [u for u in all_units if u.epoch_id == latest_epoch.id]
			resolved = all(
				u.status in ("completed", "failed") for u in epoch_units
			)
			if epoch_units and resolved:
				epoch_reviews = db.get_unit_reviews_for_mission(mission_id)
				epoch_reviews = [
					r for r in epoch_reviews if r.epoch_id == latest_epoch.id
				]
				grade = compute_decomposition_grade(
					epoch_units, epoch_reviews,
					epoch_id=latest_epoch.id, mission_id=mission_id,
				)
				# Store grade if not already persisted
				existing = db.get_decomposition_grades_for_mission(
					mission_id,
				)
				if not any(g.epoch_id == latest_epoch.id for g in existing):
					db.insert_decomposition_grade(grade)
				lines.append("")
				lines.append(format_decomposition_feedback(grade))
	except Exception:
		pass

	return "\n".join(lines)


def update_mission_state(
	db: Database,
	mission: Mission,
	config: MissionConfig,
	state_changelog: list[str] | None = None,
	degradation_status: dict | None = None,
) -> None:
	"""Write MISSION_STATE.md in the target repo as a living checklist.

	The planner reads this file to understand what's already been
	completed, avoiding duplicate work and naturally narrowing scope.
	"""
	target_path = config.target.resolved_path
	state_path = Path(target_path) / "MISSION_STATE.md"

	try:
		handoffs = db.get_recent_handoffs(mission.id, limit=50)
	except Exception:
		handoffs = []

	lines = [
		"# Mission State",
		f"Objective: {mission.objective}",
		"",
	]

	completed: list[str] = []
	failed: list[str] = []
	all_files: set[str] = set()

	for h in reversed(handoffs):  # oldest first
		files = h.files_changed
		file_str = ", ".join(files[:5]) if files else ""
		for f in files:
			all_files.add(f)

		summary = h.summary[:100] if h.summary else ""

		# Look up work unit for timestamp
		try:
			wu = db.get_work_unit(h.work_unit_id)
			timestamp = wu.finished_at if wu and wu.finished_at else ""
		except Exception:
			timestamp = ""

		if h.status == "completed":
			ts_part = f" ({timestamp})" if timestamp else ""
			completed.append(
				f"- [x] {h.work_unit_id[:8]}{ts_part} -- {summary}"
				+ (f" (files: {file_str})" if file_str else ""),
			)
		else:
			concerns = h.concerns
			detail = concerns[-1][:100] if concerns else "unknown"
			ts_part = f" ({timestamp})" if timestamp else ""
			failed.append(f"- [ ] {h.work_unit_id[:8]}{ts_part} -- {detail}")

	if completed:
		lines.append("## Completed")
		lines.extend(completed)
		lines.append("")

	if failed:
		lines.append("## Failed")
		lines.extend(failed)
		lines.append("")

	# In-flight units section so the planner doesn't duplicate running work
	try:
		all_units = db.get_work_units_for_mission(mission.id)
		running_units = [u for u in all_units if u.status == "running"]
		if running_units:
			lines.append("## In-Flight (DO NOT duplicate)")
			for u in running_units:
				files_part = f" (files: {u.files_hint})" if u.files_hint else ""
				lines.append(f"- [ ] {u.id[:8]} -- {u.title}{files_part}")
			lines.append("")
	except Exception:
		pass

	if all_files:
		lines.append("## Files Modified")
		lines.append(", ".join(sorted(all_files)))
		lines.append("")

	# Quality Reviews section
	try:
		reviews = db.get_unit_reviews_for_mission(mission.id)
		if reviews:
			lines.append("## Quality Reviews")
			for r in reviews[-20:]:  # show last 20
				# Look up unit title
				try:
					wu = db.get_work_unit(r.work_unit_id)
					title = wu.title[:40] if wu else r.work_unit_id[:8]
				except Exception:
					title = r.work_unit_id[:8]
				rationale_short = r.rationale[:80] if r.rationale else ""
				lines.append(
					f"- {r.work_unit_id[:8]} ({title}): "
					f"alignment={r.alignment_score} approach={r.approach_score} "
					f"tests={r.test_score} avg={r.avg_score}"
				)
				if rationale_short:
					lines.append(f"  \"{rationale_short}\"")
			lines.append("")
	except Exception:
		pass

	if degradation_status:
		lines.append("## System Health")
		lines.append(f"Degradation level: {degradation_status.get('level', 'FULL_CAPACITY')}")
		db_errs = degradation_status.get("db_errors", 0)
		if db_errs:
			lines.append(f"DB errors: {db_errs}")
		cr = degradation_status.get("conflict_rate", 0.0)
		if cr > 0:
			lines.append(f"Merge conflict rate: {cr:.0%}")
		vf = degradation_status.get("verification_failures", 0)
		if vf:
			lines.append(f"Verification failures: {vf}")
		lines.append("")

	lines.extend([
		"## Remaining",
		"The planner should focus on what hasn't been done yet.",
		"Do NOT re-target files in the 'Files Modified' list unless fixing a failure.",
	])

	if state_changelog:
		lines.append("")
		lines.append("## Changelog")
		lines.extend(state_changelog)

	try:
		state_path.write_text("\n".join(lines) + "\n")
	except OSError as exc:
		logger.warning("Could not write MISSION_STATE.md: %s", exc)
