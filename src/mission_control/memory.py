"""Context loading for Claude Code sessions."""

from __future__ import annotations

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import Session, TaskRecord
from mission_control.reviewer import ReviewVerdict

CONTEXT_BUDGET = 16000  # ~4000 tokens


def load_context(
	task: TaskRecord,
	db: Database,
	config: MissionConfig,
) -> str:
	"""Assemble context for a session within the token budget."""
	sections: list[str] = []
	budget = CONTEXT_BUDGET

	# 1. Session history (last N as one-liners)
	history = _format_session_history(db.get_recent_sessions(10))
	if history:
		sections.append(f"### Recent Sessions\n{history}")
		budget -= len(history)

	# 2. Recent decisions
	decisions = db.get_recent_decisions(5)
	if decisions:
		dec_text = "\n".join(f"- {d.decision}: {d.rationale}" for d in decisions)
		if len(dec_text) < budget:
			sections.append(f"### Recent Decisions\n{dec_text}")
			budget -= len(dec_text)

	# 3. Project CLAUDE.md (truncated)
	claude_md = _read_project_claude_md(config)
	if claude_md:
		truncated = claude_md[:min(2000, budget)]
		sections.append(f"### Project Instructions\n{truncated}")

	return "\n\n".join(sections) if sections else ""


def _format_session_history(sessions: list[Session]) -> str:
	"""Format sessions as one-liners."""
	if not sessions:
		return ""
	lines = []
	for s in sessions:
		status_icon = {"completed": "+", "failed": "x", "reverted": "!"}.get(s.status, "?")
		lines.append(f"[{status_icon}] {s.id}: {s.task_description} -> {s.status} ({s.output_summary[:80]})")
	return "\n".join(lines)


def summarize_session(session: Session, verdict: ReviewVerdict) -> str:
	"""Create a one-paragraph summary for future context."""
	parts = [f"Session {session.id} ({session.task_description}):"]
	parts.append(f"Verdict: {verdict.verdict}.")
	if verdict.improvements:
		parts.append(f"Improved: {', '.join(verdict.improvements)}.")
	if verdict.regressions:
		parts.append(f"Regressed: {', '.join(verdict.regressions)}.")
	if session.output_summary:
		parts.append(f"Output: {session.output_summary[:200]}")
	return " ".join(parts)


def compress_history(sessions: list[Session], max_chars: int = 4000) -> str:
	"""Compress session history to fit within budget."""
	if not sessions:
		return ""

	lines: list[str] = []
	total = 0
	for s in sessions:
		line = f"{s.id}: {s.task_description} -> {s.status}"
		if total + len(line) > max_chars:
			lines.append(f"... and {len(sessions) - len(lines)} more sessions")
			break
		lines.append(line)
		total += len(line) + 1

	return "\n".join(lines)


def _read_project_claude_md(config: MissionConfig) -> str:
	"""Read the target project's CLAUDE.md if it exists."""
	claude_md_path = config.target.resolved_path / "CLAUDE.md"
	try:
		return claude_md_path.read_text()
	except (FileNotFoundError, PermissionError):
		return ""
