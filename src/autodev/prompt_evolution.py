"""Prompt evolution engine -- UCB1-based A/B testing of prompt variants with LLM-driven mutation."""

from __future__ import annotations

import asyncio
import logging
import math
from uuid import uuid4

from autodev.config import MissionConfig, PromptEvolutionConfig, claude_subprocess_env
from autodev.db import Database
from autodev.models import PromptOutcome, PromptVariant, _now_iso

logger = logging.getLogger(__name__)


class PromptEvolutionEngine:
	"""UCB1 multi-armed bandit for prompt variant selection and mutation."""

	def __init__(
		self,
		db: Database,
		config: PromptEvolutionConfig,
		mission_config: MissionConfig | None = None,
	) -> None:
		self.db = db
		self.config = config
		self.mission_config = mission_config

	def record_outcome(self, variant_id: str, outcome: str, context: str = "") -> None:
		"""Record a pass/fail outcome for a variant and recompute win_rate from DB counts."""
		variant = self.db.get_prompt_variant(variant_id)
		if variant is None:
			logger.warning("record_outcome: variant %s not found", variant_id)
			return

		self.db.insert_prompt_outcome(PromptOutcome(
			variant_id=variant_id,
			outcome=outcome,
			context=context,
			recorded_at=_now_iso(),
		))

		counts = self.db.count_prompt_outcomes(variant_id)
		total = sum(counts.values())
		pass_count = counts.get("pass", 0)
		variant.win_rate = pass_count / total if total > 0 else 0.0
		variant.sample_count = total
		self.db.update_prompt_variant(variant)

	def select_variant(self, component: str) -> PromptVariant | None:
		"""Select a variant for the given component using UCB1.

		Unseen variants (sample_count=0) are always selected first.
		Returns None if no variants exist for the component.
		"""
		variants = self.db.get_prompt_variants_for_component(component)
		if not variants:
			return None

		total_samples = sum(v.sample_count for v in variants)
		if total_samples == 0:
			return variants[0]

		c = self.config.exploration_factor
		best_variant = None
		best_score = -1.0

		for v in variants:
			if v.sample_count == 0:
				return v
			exploration = c * math.sqrt(math.log(total_samples) / v.sample_count)
			score = v.win_rate + exploration
			if score > best_score:
				best_score = score
				best_variant = v

		return best_variant

	async def propose_mutation(
		self, component: str, failure_traces: list[str],
	) -> PromptVariant | None:
		"""Use LLM to propose a mutated variant based on failure traces.

		Skips if the best parent has fewer than min_samples_before_mutation samples.
		"""
		variants = self.db.get_prompt_variants_for_component(component)
		if not variants:
			return None

		parent = variants[0]  # highest win_rate (ordered DESC from DB)
		if parent.sample_count < self.config.min_samples_before_mutation:
			return None

		traces_text = "\n---\n".join(failure_traces[:5])
		prompt = (
			f"You are a prompt engineering assistant. The following prompt variant "
			f"for the '{component}' component has a win rate of {parent.win_rate:.2f} "
			f"over {parent.sample_count} samples.\n\n"
			f"Current prompt:\n{parent.content}\n\n"
			f"Recent failure traces:\n{traces_text}\n\n"
			f"Propose an improved version of the prompt that addresses the failure patterns. "
			f"Output ONLY the improved prompt text, nothing else."
		)

		try:
			from autodev.intelligence.utils import find_claude_binary
			proc = await asyncio.create_subprocess_exec(
				find_claude_binary(), "-p", "--model", self.config.mutation_model,
				"--output-format", "text",
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(self.mission_config),
			)
			stdout, _ = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=120)
			mutated_content = stdout.decode().strip()
		except (asyncio.TimeoutError, OSError) as exc:
			logger.warning("Mutation LLM call failed: %s", exc)
			return None

		if not mutated_content:
			return None

		new_variant = PromptVariant(
			id=uuid4().hex[:12],
			component=component,
			variant_id=f"{component}-{uuid4().hex[:8]}",
			content=mutated_content,
			created_at=_now_iso(),
			parent_variant_id=parent.variant_id,
		)
		self.db.insert_prompt_variant(new_variant)
		logger.info(
			"Proposed mutation %s from parent %s",
			new_variant.variant_id, parent.variant_id,
		)
		return new_variant
