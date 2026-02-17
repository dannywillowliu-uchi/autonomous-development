"""Context loading for Claude Code sessions."""

from __future__ import annotations

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import ContextItem, Session, TaskRecord, WorkUnit
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


def load_context_for_work_unit(
	unit: WorkUnit,
	db: Database,
	config: MissionConfig,
) -> str:
	"""Assemble context for a parallel worker session.

	Includes plan context, sibling unit status, and project instructions.
	"""
	sections: list[str] = []
	budget = CONTEXT_BUDGET

	# 1. Plan objective
	plan = db.get_plan(unit.plan_id) if unit.plan_id else None
	if plan:
		obj_text = f"Objective: {plan.objective}"
		sections.append(f"### Plan\n{obj_text}")
		budget -= len(obj_text)

	# 2. Sibling unit status (what else is being worked on)
	if unit.plan_id:
		siblings = db.get_work_units_for_plan(unit.plan_id)
		sibling_lines = []
		for s in siblings:
			if s.id == unit.id:
				continue
			line = f"- [{s.status}] {s.title}"
			if s.output_summary:
				line += f" ({s.output_summary[:60]})"
			sibling_lines.append(line)
		if sibling_lines:
			sib_text = "\n".join(sibling_lines)
			if len(sib_text) < budget:
				sections.append(f"### Sibling Units\n{sib_text}")
				budget -= len(sib_text)

	# 3. Context items from prior workers
	ctx_section = inject_context_items(unit, db, budget)
	if ctx_section:
		sections.append(ctx_section)
		budget -= len(ctx_section)

	# 4. Project CLAUDE.md (truncated)
	claude_md = _read_project_claude_md(config)
	if claude_md:
		truncated = claude_md[:min(2000, budget)]
		sections.append(f"### Project Instructions\n{truncated}")

	return "\n\n".join(sections) if sections else ""


def load_context_for_mission_worker(
	unit: WorkUnit,
	config: MissionConfig,
	db: Database | None = None,
) -> str:
	"""Assemble minimal context for a mission mode worker.

	Fresh-start pattern: workers get ONLY the work unit description + project
	CLAUDE.md + relevant context items. No session history, no sibling status,
	no prior decisions. This reduces context drift and keeps workers focused.
	"""
	sections: list[str] = []
	budget = CONTEXT_BUDGET

	# 1. Context items from prior workers (if db available)
	if db is not None:
		ctx_section = inject_context_items(unit, db, min(2000, budget))
		if ctx_section:
			sections.append(ctx_section)
			budget -= len(ctx_section)

	# 2. Project CLAUDE.md (truncated to budget)
	claude_md = _read_project_claude_md(config)
	if claude_md:
		truncated = claude_md[:min(4000, budget)]
		sections.append(f"### Project Instructions\n{truncated}")

	return "\n\n".join(sections) if sections else ""


def extract_scope_tokens(unit: WorkUnit) -> list[str]:
	"""Extract scope tokens from a work unit's files_hint and title for matching."""
	tokens: list[str] = []
	if unit.files_hint:
		for path in unit.files_hint.split(","):
			path = path.strip()
			if path:
				tokens.append(path)
				parts = path.rsplit("/", 1)
				if len(parts) == 2:
					tokens.append(parts[1])
	if unit.title:
		tokens.append(unit.title)
	return tokens


def format_context_items(items: list[ContextItem]) -> str:
	"""Format context items as a text block for injection into worker prompts."""
	if not items:
		return ""
	lines: list[str] = []
	for item in items:
		conf = f" (confidence: {item.confidence:.1f})" if item.confidence < 1.0 else ""
		lines.append(f"- [{item.item_type}] {item.content}{conf}")
	return "\n".join(lines)


def inject_context_items(
	unit: WorkUnit,
	db: Database,
	budget: int,
	min_confidence: float = 0.5,
) -> str:
	"""Build a context section from relevant ContextItems for a work unit.

	Finds items whose scope overlaps with the unit's files_hint and title,
	formats them, and returns a section string that fits within the budget.
	"""
	tokens = extract_scope_tokens(unit)
	if not tokens:
		return ""
	items = db.get_context_items_by_scope_overlap(tokens, min_confidence=min_confidence)
	if not items:
		return ""
	formatted = format_context_items(items)
	if len(formatted) > budget:
		formatted = formatted[:budget]
	return f"### Context from Prior Workers\n{formatted}"


def _read_project_claude_md(config: MissionConfig) -> str:
	"""Read the target project's CLAUDE.md if it exists."""
	claude_md_path = config.target.resolved_path / "CLAUDE.md"
	try:
		return claude_md_path.read_text()
	except (FileNotFoundError, PermissionError):
		return ""
