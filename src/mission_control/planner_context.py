"""Planner context and mission state formatting -- extracted from continuous_controller."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from mission_control.batch_analyzer import BatchAnalyzer, format_cost_trend
from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import Mission

logger = logging.getLogger(__name__)


def build_planner_context(db: Database, mission_id: str, core_test_context: str = "") -> str:
	"""Build minimal planner context: cross-mission learnings + recent failures.

	The planner reads MISSION_STATE.md and MISSION_STRATEGY.md from disk
	for per-mission state. This function provides only what can't be found
	on disk: semantic memories from past missions and recent failure reasons.
	"""
	lines: list[str] = []

	# Recent failures with actionable detail (last 3)
	try:
		handoffs = db.get_recent_handoffs(mission_id, limit=10)
		failed = [h for h in handoffs if h.status != "completed"]
		if failed:
			lines.append("## Recent Failures")
			for h in failed[-3:]:
				concerns = h.concerns
				detail = concerns[-1][:300] if concerns else "unknown"
				lines.append(f"- {h.work_unit_id[:8]}: {detail}")
	except Exception:
		pass

	# Inject learned rules from semantic memory (cross-mission)
	try:
		semantic_memories = db.get_top_semantic_memories(limit=5)
		if semantic_memories:
			lines.append("\n## Learned Rules (from past missions)")
			for sm in semantic_memories:
				conf = f" (confidence: {sm.confidence:.1f})" if sm.confidence < 1.0 else ""
				lines.append(f"- {sm.content}{conf}")
	except Exception:
		pass

	# Epoch cost trend (last 5 epochs)
	try:
		analyzer = BatchAnalyzer(db)
		summaries = analyzer.get_epoch_cost_summary(mission_id)
		trend = format_cost_trend(summaries)
		if trend:
			lines.append("\n## Cost Trend (recent epochs)")
			lines.append(trend)
	except Exception:
		pass

	if core_test_context:
		lines.append("")
		lines.append(core_test_context)

	return "\n".join(lines)


def update_mission_state(
	db: Database,
	mission: Mission,
	config: MissionConfig,
	state_changelog: list[str] | None = None,
	degradation_status: dict | None = None,
	strategy: str = "",
	reflection: object | None = None,
	core_test_results: str | None = None,
) -> None:
	"""Write MISSION_STATE.md as a fixed-size summary (not a growing log).

	The planner reads this file to understand current progress, active issues,
	and patterns. The file stays roughly constant size regardless of mission length.
	"""
	target_path = config.target.resolved_path
	state_path = Path(target_path) / "MISSION_STATE.md"

	try:
		units = db.get_work_units_for_mission(mission.id)
	except Exception:
		units = []
	try:
		handoffs = db.get_recent_handoffs(mission.id, limit=50)
	except Exception:
		handoffs = []
	try:
		epochs = db.get_epochs_for_mission(mission.id)
	except Exception:
		epochs = []

	completed_count = sum(1 for u in units if u.status == "completed")
	failed_count = sum(1 for u in units if u.status == "failed")
	epoch_number = len(epochs)

	# Find the last completed unit
	last_completed = ""
	for u in reversed(units):
		if u.status == "completed":
			last_completed = f'"{u.title[:80]}"'
			if u.finished_at:
				last_completed += f" ({u.finished_at})"
			break

	lines = [
		"# Mission State",
		f"Objective: {mission.objective}",
		"",
		"## Progress",
		f"{completed_count} tasks complete, {failed_count} failed. Epoch {epoch_number}.",
	]
	if last_completed:
		lines.append(f"Last completed: {last_completed}")
	lines.append("")

	# Strategy summary (first 500 chars from MISSION_STRATEGY.md)
	if strategy:
		lines.append("## Strategy")
		lines.append(strategy[:500])
		lines.append("")

	# Active issues (last 3 failed units with concern strings)
	failed_handoffs = [h for h in handoffs if h.status != "completed"]
	if failed_handoffs:
		lines.append("## Active Issues")
		for h in failed_handoffs[-3:]:
			concerns = h.concerns
			detail = concerns[-1][:200] if concerns else "unknown"
			lines.append(f"- {h.work_unit_id[:8]}: {detail}")
		lines.append("")

	# Key decisions (knowledge items of type design)
	try:
		knowledge = db.get_knowledge_for_mission(mission.id)
		design_items = [k for k in knowledge if k.source_unit_type == "design"]
		if design_items:
			lines.append("## Key Decisions")
			for k in design_items[-5:]:
				lines.append(f"- {k.title}: {k.content[:200]}")
			lines.append("")
	except Exception:
		pass

	# Patterns from reflection
	if reflection is not None:
		patterns = getattr(reflection, "patterns", [])
		tensions = getattr(reflection, "tensions", [])
		open_qs = getattr(reflection, "open_questions", [])

		if patterns or tensions:
			lines.append("## Patterns (from reflection)")
			for p in patterns[:5]:
				lines.append(f"- {p}")
			if tensions:
				lines.append("Tensions:")
				for t in tensions[:3]:
					lines.append(f"- {t}")
			lines.append("")

		if open_qs:
			lines.append("## Open Questions")
			for q in open_qs[:5]:
				lines.append(f"- {q}")
			lines.append("")

	# Low-confidence knowledge items as open questions
	try:
		if knowledge:
			low_conf = [k for k in knowledge if k.confidence < 0.7]
			if low_conf:
				if not (reflection and getattr(reflection, "open_questions", [])):
					lines.append("## Open Questions")
				for k in low_conf[-3:]:
					lines.append(f"- [{k.confidence:.1f}] {k.title}: {k.content[:150]}")
				lines.append("")
	except Exception:
		pass

	# Files modified (grouped by directory)
	all_files: set[str] = set()
	for h in handoffs:
		for f in (h.files_changed or []):
			all_files.add(f)

	if all_files:
		lines.append("## Files Modified")
		# Group by directory
		by_dir: dict[str, list[str]] = defaultdict(list)
		for f in sorted(all_files):
			parts = f.rsplit("/", 1)
			dir_name = parts[0] if len(parts) > 1 else "."
			by_dir[dir_name].append(f)
		for dir_name in sorted(by_dir):
			files = by_dir[dir_name]
			lines.append(f"- {dir_name}/: {', '.join(Path(f).name for f in files)}")
		lines.append("")

	if core_test_results:
		lines.append(core_test_results)

	try:
		state_path.write_text("\n".join(lines) + "\n")
	except OSError as exc:
		logger.warning("Could not write MISSION_STATE.md: %s", exc)
