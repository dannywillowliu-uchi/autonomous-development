"""Shared context-gathering functions for planning agents.

Extracted from strategist.py so both the critic agent and any planner
can access project context (backlog, git history, past missions, etc.)
without duplicating code.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from datetime import datetime, timezone

from mission_control.config import MissionConfig
from mission_control.db import Database

log = logging.getLogger(__name__)


def read_backlog(config: MissionConfig) -> str:
	"""Read BACKLOG.md from the target project root."""
	backlog_path = config.target.resolved_path / "BACKLOG.md"
	try:
		return backlog_path.read_text()
	except FileNotFoundError:
		log.info("No BACKLOG.md found at %s", backlog_path)
		return ""


async def get_git_log(config: MissionConfig, limit: int = 20) -> str:
	"""Get recent git log from the target project."""
	try:
		proc = await asyncio.create_subprocess_exec(
			"git", "log", "--oneline", f"-{limit}",
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
			cwd=str(config.target.resolved_path),
		)
		stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
		output = stdout.decode() if stdout else ""
		return output.strip() if proc.returncode == 0 else ""
	except (asyncio.TimeoutError, FileNotFoundError, OSError):
		log.info("Could not read git log")
		return ""


def get_past_missions(db: Database, limit: int = 10) -> str:
	"""Format recent missions as context text."""
	missions = db.get_all_missions(limit=limit)
	if not missions:
		return ""
	lines = []
	for m in missions:
		rating_str = ""
		try:
			ratings = db.get_trajectory_ratings_for_mission(m.id)
			if ratings:
				rating_str = f", human_rating={ratings[0].rating}/10"
		except Exception:
			pass
		lines.append(
			f"- [{m.status}] {m.objective[:120]} "
			f"(rounds={m.total_rounds}, score={m.final_score}, "
			f"reason={m.stopped_reason or 'n/a'}{rating_str})"
		)
	return "\n".join(lines)


def get_strategic_context(db: Database, limit: int = 10) -> str:
	"""Get rolling strategic context entries from DB."""
	if not hasattr(db, "get_strategic_context"):
		return ""
	try:
		entries = db.get_strategic_context(limit=limit)
		if not entries:
			return ""
		return "\n".join(f"- {e}" for e in entries)
	except Exception:
		log.debug("get_strategic_context not available", exc_info=True)
		return ""


def get_episodic_context(db: Database) -> str:
	"""Load episodic + semantic memories and format as context."""
	lines: list[str] = []
	try:
		semantic = db.get_top_semantic_memories(limit=5)
		if semantic:
			lines.append("Learned rules:")
			for sm in semantic:
				lines.append(f"  - [{sm.confidence:.1f}] {sm.content}")
	except Exception:
		log.debug("Failed to load semantic memories", exc_info=True)

	try:
		episodes = db.get_episodic_memories_by_scope(["mission", "strategy"], limit=10)
		if episodes:
			lines.append("Recent episodes:")
			for ep in episodes:
				lines.append(f"  - [{ep.event_type}] {ep.content} -> {ep.outcome}")
	except Exception:
		log.debug("Failed to load episodic memories", exc_info=True)

	return "\n".join(lines)


def get_human_preferences(db: Database) -> str:
	"""Build human preference signals from trajectory ratings."""
	try:
		missions = db.get_all_missions(limit=20)
		high_rated: list[str] = []
		low_rated: list[str] = []
		for m in missions:
			ratings = db.get_trajectory_ratings_for_mission(m.id)
			if not ratings:
				continue
			r = ratings[0]
			label = f"\"{m.objective[:80]}\" ({r.rating}/10)"
			if r.rating >= 7:
				high_rated.append(label)
			elif r.rating <= 4:
				low_rated.append(label)
		if not high_rated and not low_rated:
			return ""
		lines = []
		if high_rated:
			lines.append("Highly-rated: " + ", ".join(high_rated[:5]))
		if low_rated:
			lines.append("Low-rated: " + ", ".join(low_rated[:5]))
		lines.append("Prefer work similar to highly-rated missions.")
		return "\n".join(lines)
	except Exception:
		return ""


async def get_intel_context(config: MissionConfig, ttl_hours: float = 6.0) -> str:
	"""Return markdown summary of top intel proposals, with file-based TTL cache."""
	try:
		from mission_control.intelligence import AdaptationProposal, IntelReport, run_scan
		from mission_control.intelligence.models import Finding

		cache_path = config.target.resolved_path / ".cache" / "intel_report.json"

		report: IntelReport | None = None

		# Try loading from cache
		if cache_path.exists():
			try:
				data = json.loads(cache_path.read_text())
				cached_at = datetime.fromisoformat(data["cached_at"])
				age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
				if age_hours < ttl_hours:
					report = IntelReport(
						findings=[Finding(**f) for f in data.get("findings", [])],
						proposals=[AdaptationProposal(**p) for p in data.get("proposals", [])],
						timestamp=data.get("timestamp", ""),
						sources_scanned=data.get("sources_scanned", []),
						scan_duration_seconds=data.get("scan_duration_seconds", 0.0),
					)
			except (json.JSONDecodeError, KeyError, TypeError, ValueError):
				log.debug("Intel cache invalid, will re-scan", exc_info=True)

		# Cache miss or stale -- run a fresh scan
		if report is None:
			report = await run_scan(threshold=0.3)
			cache_path.parent.mkdir(parents=True, exist_ok=True)
			payload = dataclasses.asdict(report)
			payload["cached_at"] = datetime.now(timezone.utc).isoformat()
			cache_path.write_text(json.dumps(payload))

		# Format top 5 proposals by priority (1 = highest)
		top = sorted(report.proposals, key=lambda p: p.priority)[:5]
		if not top:
			return ""

		lines = [f"{i}. **{p.title}** -- {p.description}" for i, p in enumerate(top, 1)]
		return "### Ecosystem Intelligence\n" + "\n".join(lines)
	except Exception:
		log.debug("get_intel_context failed", exc_info=True)
		return ""
