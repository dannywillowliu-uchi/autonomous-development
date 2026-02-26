"""Context loading for Claude Code sessions."""

from __future__ import annotations

import asyncio
import logging

from mission_control.config import EpisodicMemoryConfig, MissionConfig, claude_subprocess_env
from mission_control.db import Database
from mission_control.models import ContextItem, EpisodicMemory, SemanticMemory, Session, WorkUnit, _new_id, _now_iso

logger = logging.getLogger(__name__)

CONTEXT_BUDGET = 16000  # ~4000 tokens


def _format_session_history(sessions: list[Session]) -> str:
	"""Format sessions as one-liners."""
	if not sessions:
		return ""
	lines = []
	for s in sessions:
		status_icon = {"completed": "+", "failed": "x", "reverted": "!"}.get(s.status, "?")
		lines.append(f"[{status_icon}] {s.id}: {s.task_description} -> {s.status} ({s.output_summary[:80]})")
	return "\n".join(lines)


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


class MemoryManager:
	"""Three-layer memory system: episodic storage, retrieval with access tracking, and semantic distillation."""

	def __init__(self, db: Database, config: EpisodicMemoryConfig) -> None:
		self.db = db
		self.config = config

	def store_episode(
		self,
		event_type: str,
		content: str,
		outcome: str,
		scope_tokens: list[str],
	) -> EpisodicMemory:
		"""Create and persist an episodic memory."""
		mem = EpisodicMemory(
			id=_new_id(),
			event_type=event_type,
			content=content,
			outcome=outcome,
			scope_tokens=",".join(scope_tokens),
			ttl_days=self.config.default_ttl_days,
			created_at=_now_iso(),
			last_accessed=_now_iso(),
		)
		self.db.insert_episodic_memory(mem)
		return mem

	def retrieve_relevant(
		self, query_tokens: list[str], limit: int = 10,
	) -> list[EpisodicMemory]:
		"""Retrieve relevant episodes and bump access counters."""
		episodes = self.db.get_episodic_memories_by_scope(query_tokens, limit=limit)
		now = _now_iso()
		for ep in episodes:
			ep.access_count += 1
			ep.last_accessed = now
			self.db.update_episodic_memory(ep)
		return episodes

	async def distill_to_semantic(
		self, episodes: list[EpisodicMemory],
	) -> SemanticMemory | None:
		"""Use LLM to distill a generalized rule from multiple episodes.

		Skips if fewer than min_episodes_for_distill episodes provided.
		Confidence is the average of source episode confidences.
		"""
		if len(episodes) < self.config.min_episodes_for_distill:
			return None

		episode_texts = []
		for ep in episodes:
			episode_texts.append(
				f"[{ep.event_type}] {ep.content} -> {ep.outcome}"
			)
		episodes_block = "\n".join(episode_texts)

		prompt = (
			f"You are analyzing patterns from past development events. "
			f"Here are {len(episodes)} episodes:\n\n{episodes_block}\n\n"
			f"Extract ONE concise, actionable rule or pattern that generalizes "
			f"across these episodes. Output ONLY the rule, nothing else."
		)

		try:
			proc = await asyncio.create_subprocess_exec(
				"claude", "-p", "--model", self.config.distill_model,
				"--output-format", "text",
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(),
			)
			stdout, _ = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=120)
			rule_content = stdout.decode().strip()
		except (asyncio.TimeoutError, OSError) as exc:
			logger.warning("Distillation LLM call failed: %s", exc)
			return None

		if not rule_content:
			return None

		avg_confidence = sum(ep.confidence for ep in episodes) / len(episodes)
		source_ids = ",".join(ep.id for ep in episodes)

		semantic = SemanticMemory(
			id=_new_id(),
			content=rule_content,
			source_episode_ids=source_ids,
			confidence=avg_confidence,
			created_at=_now_iso(),
		)
		self.db.insert_semantic_memory(semantic)
		logger.info("Distilled semantic memory: %s", rule_content[:80])
		return semantic

	def decay_tick(self) -> tuple[int, int]:
		"""Decrement TTL for all episodes; extend frequently accessed ones; evict expired.

		Returns (evicted_count, extended_count).
		"""
		all_episodes = self.db.get_all_episodic_memories()
		evicted = 0
		extended = 0

		for ep in all_episodes:
			# Extend TTL for frequently accessed episodes
			if ep.access_count >= 3:
				ep.ttl_days += self.config.access_boost_days
				ep.access_count = 0  # reset counter after boost
				extended += 1

			# Decay
			ep.ttl_days -= 1

			if ep.ttl_days <= 0:
				self.db.delete_episodic_memory(ep.id)
				evicted += 1
			else:
				self.db.update_episodic_memory(ep)

		return evicted, extended

	def get_promote_candidates(self) -> list[EpisodicMemory]:
		"""Get episodes with high confidence nearing expiration -- candidates for distillation."""
		all_episodes = self.db.get_all_episodic_memories()
		return [
			ep for ep in all_episodes
			if ep.confidence >= 0.7 and ep.ttl_days <= 3
		]
