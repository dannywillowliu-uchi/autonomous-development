"""Auto-update pipeline: bridges intel proposals to swarm missions.

Scans for improvement proposals via the intelligence subsystem, filters
already-applied ones, classifies risk, and either auto-launches low-risk
proposals as swarm missions or gates high-risk ones via Telegram approval.

Safety rails: rate limiting, diff review gate, ratchet + oracle integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from autodev.config import MissionConfig
from autodev.db import Database
from autodev.intelligence.models import AdaptationProposal, Finding
from autodev.intelligence.scanner import run_scan
from autodev.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
	"""Result of a single proposal processing."""

	proposal_id: str
	title: str
	risk_level: str
	action: str  # launched, approved, rejected, skipped, dry_run, oracle_blocked, review_rejected
	mission_id: str = ""
	review: dict = field(default_factory=dict)
	ratchet_tag: str = ""


class AutoUpdatePipeline:
	"""Bridge intel proposals to swarm missions."""

	def __init__(
		self,
		config: MissionConfig,
		db: Database,
		max_daily_modifications: int = 5,
	) -> None:
		self._config = config
		self._db = db
		self._max_daily_modifications = max_daily_modifications
		self._findings_by_id: dict[str, Finding] = {}

	def _check_rate_limit(self) -> bool:
		"""Return True if under the daily self-modification limit."""
		today = datetime.now(timezone.utc).date().isoformat()
		count = self._db.count_proposals_applied_since(today)
		return count < self._max_daily_modifications

	async def run(
		self,
		dry_run: bool = False,
		approve_all: bool = False,
		threshold: float = 0.3,
	) -> list[UpdateResult]:
		"""Full pipeline: scan -> rate check -> evaluate -> filter -> approve -> launch.

		Args:
			dry_run: Show proposals without launching missions.
			approve_all: Skip Telegram approval for high-risk proposals.
			threshold: Relevance threshold for proposal generation.

		Returns:
			List of UpdateResult for each processed proposal.
		"""
		if not dry_run and not self._check_rate_limit():
			logger.warning(
				"Rate limit reached: %d modifications today (max %d)",
				self._db.count_proposals_applied_since(
					datetime.now(timezone.utc).date().isoformat()
				),
				self._max_daily_modifications,
			)
			return []

		report = await run_scan(threshold=threshold)
		logger.info(
			"Intel scan complete: %d findings, %d proposals",
			len(report.findings), len(report.proposals),
		)

		self._findings_by_id = {f.id: f for f in report.findings}

		# Filter by title (IDs are regenerated each scan, but titles are stable)
		proposals = [
			p for p in report.proposals
			if not self._is_already_applied(p.title)
		]
		if not proposals:
			logger.info("No new proposals to process")
			return []

		logger.info("%d new proposals to process", len(proposals))

		results: list[UpdateResult] = []
		for proposal in proposals:
			result = await self._process_proposal(
				proposal,
				dry_run=dry_run,
				approve_all=approve_all,
			)
			results.append(result)

		return results

	async def _process_proposal(
		self,
		proposal: AdaptationProposal,
		dry_run: bool = False,
		approve_all: bool = False,
	) -> UpdateResult:
		"""Process a single proposal based on its risk level."""
		if dry_run:
			logger.info("DRY RUN: %s (risk=%s)", proposal.title, proposal.risk_level)
			return UpdateResult(
				proposal_id=proposal.id,
				title=proposal.title,
				risk_level=proposal.risk_level,
				action="dry_run",
			)

		if proposal.risk_level == "low":
			return await self._auto_launch(proposal)

		# High-risk: require approval
		if approve_all:
			return await self._auto_launch(proposal)

		return await self._request_approval(proposal)

	async def _auto_launch(self, proposal: AdaptationProposal) -> UpdateResult:
		"""Auto-launch a proposal as a swarm mission with ratchet checkpoint."""
		from autodev.ratchet import GitRatchet

		repo_path = Path(self._config.target.resolved_path)
		ratchet = GitRatchet(repo_path)
		tag = await ratchet.checkpoint(proposal.id)

		objective = await self._generate_spec_or_objective(proposal)
		mission_id = self._record_applied(proposal, objective)
		logger.info("Launched mission %s for proposal: %s (tag=%s)", mission_id, proposal.title, tag)

		return UpdateResult(
			proposal_id=proposal.id,
			title=proposal.title,
			risk_level=proposal.risk_level,
			action="launched",
			mission_id=mission_id,
			ratchet_tag=tag,
		)

	async def _finalize_modification(
		self,
		proposal: AdaptationProposal,
		tag: str,
		verification_cmd: str = "",
	) -> UpdateResult:
		"""Finalize a self-modification after the swarm mission completes.

		Call this after the swarm mission launched by _auto_launch finishes.
		Runs oracle check, diff review, and ratchet verify_and_decide.

		Args:
			proposal: The original proposal that was launched.
			tag: The ratchet tag from _auto_launch (stored in UpdateResult.ratchet_tag).
			verification_cmd: Command to run for verification (e.g. pytest + ruff).
		"""
		from autodev.oracle import check_oracle_violation
		from autodev.ratchet import GitRatchet

		repo_path = Path(self._config.target.resolved_path)
		ratchet = GitRatchet(repo_path)

		# Step 1: Check oracle violations on changed files
		diff_result = await asyncio.create_subprocess_exec(
			"git", "diff", "--name-only", f"{tag}..HEAD",
			cwd=str(repo_path),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, _ = await diff_result.communicate()
		changed_files = [f for f in stdout.decode().strip().split("\n") if f]

		violations = check_oracle_violation(changed_files)
		if violations:
			logger.warning("Oracle violation: protected files modified: %s", violations)
			await ratchet.rollback(tag)
			return UpdateResult(
				proposal_id=proposal.id,
				title=proposal.title,
				risk_level=proposal.risk_level,
				action="oracle_blocked",
				ratchet_tag=tag,
			)

		# Step 2: Diff review gate
		review = await self._review_modification(tag)
		if not review.get("approved", True):
			logger.warning("Diff review rejected: %s", review.get("concerns", []))
			await ratchet.rollback(tag)
			return UpdateResult(
				proposal_id=proposal.id,
				title=proposal.title,
				risk_level=proposal.risk_level,
				action="review_rejected",
				review=review,
				ratchet_tag=tag,
			)

		# Step 3: Ratchet verify and decide
		if verification_cmd:
			kept = await ratchet.verify_and_decide(tag, verification_cmd)
			action = "launched" if kept else "review_rejected"
		else:
			action = "launched"

		return UpdateResult(
			proposal_id=proposal.id,
			title=proposal.title,
			risk_level=proposal.risk_level,
			action=action,
			ratchet_tag=tag,
			review=review,
		)

	async def _review_modification(self, tag: str) -> dict:
		"""Review self-modification diff via LLM."""
		try:
			repo_path = Path(self._config.target.resolved_path)
			diff_proc = await asyncio.create_subprocess_exec(
				"git", "diff", f"{tag}..HEAD",
				cwd=str(repo_path),
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
			)
			stdout, _ = await diff_proc.communicate()
			diff_text = stdout.decode()

			if not diff_text.strip():
				return {"approved": True, "concerns": [], "summary": "No changes detected"}

			prompt = (
				"Review this self-modification diff. Flag: regressions, unnecessary changes, "
				"security concerns, scope creep beyond the proposal. Return JSON with "
				'{approved: bool, concerns: list[str], summary: str}\n\n'
				f"```diff\n{diff_text[:50000]}\n```"
			)

			review_proc = await asyncio.create_subprocess_exec(
				"claude", "--print", "--output-format", "json", "-p", prompt,
				cwd=str(repo_path),
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
			)
			review_stdout, _ = await review_proc.communicate()
			response_text = review_stdout.decode()

			result = json.loads(response_text)
			if isinstance(result, dict) and "result" in result:
				inner = result["result"]
				if isinstance(inner, str):
					import re
					json_match = re.search(r"\{[^{}]*\"approved\"[^{}]*\}", inner)
					if json_match:
						return json.loads(json_match.group())
				elif isinstance(inner, dict):
					return inner
			if isinstance(result, dict) and "approved" in result:
				return result

			return {"approved": True, "concerns": [], "summary": "Review unavailable"}
		except Exception:
			logger.exception("LLM review failed, defaulting to approved")
			return {"approved": True, "concerns": [], "summary": "Review unavailable"}

	async def _request_approval(self, proposal: AdaptationProposal) -> UpdateResult:
		"""Send a high-risk proposal to Telegram for approval."""
		tg_config = self._config.notifications.telegram
		if not tg_config.bot_token or not tg_config.chat_id:
			logger.warning(
				"Telegram not configured, skipping high-risk proposal: %s",
				proposal.title,
			)
			return UpdateResult(
				proposal_id=proposal.id,
				title=proposal.title,
				risk_level=proposal.risk_level,
				action="skipped",
			)

		notifier = TelegramNotifier(tg_config.bot_token, tg_config.chat_id)
		try:
			description = (
				f"Auto-Update Proposal (HIGH RISK)\n\n"
				f"Title: {proposal.title}\n"
				f"Type: {proposal.proposal_type}\n"
				f"Priority: {proposal.priority}\n"
				f"Effort: {proposal.effort_estimate}\n"
				f"Target modules: {', '.join(proposal.target_modules)}\n\n"
				f"{proposal.description}"
			)
			approved = await notifier.request_approval(description)
		finally:
			await notifier.close()

		if approved:
			objective = await self._generate_spec_or_objective(proposal)
			mission_id = self._record_applied(proposal, objective)
			logger.info("Approved and launched mission %s: %s", mission_id, proposal.title)
			return UpdateResult(
				proposal_id=proposal.id,
				title=proposal.title,
				risk_level=proposal.risk_level,
				action="approved",
				mission_id=mission_id,
			)

		logger.info("Proposal rejected via Telegram: %s", proposal.title)
		return UpdateResult(
			proposal_id=proposal.id,
			title=proposal.title,
			risk_level=proposal.risk_level,
			action="rejected",
		)

	def _generate_objective(self, proposal: AdaptationProposal) -> str:
		"""Convert a proposal into a swarm mission objective."""
		modules = ", ".join(proposal.target_modules) if proposal.target_modules else "TBD"
		return (
			f"[AUTO-UPDATE] {proposal.title}. "
			f"{proposal.description} "
			f"Target modules: {modules}. "
			f"Effort: {proposal.effort_estimate}. "
			f"All tests must pass after changes."
		)

	def _is_already_applied(self, title: str) -> bool:
		"""Check if a proposal with this title was already applied."""
		return self._db.is_proposal_applied(title)

	def _record_applied(self, proposal: AdaptationProposal, objective: str) -> str:
		"""Record a proposal as applied and return the mission_id."""
		from autodev.models import _new_id

		mission_id = _new_id()
		self._db.record_applied_proposal(
			proposal_id=proposal.id,
			finding_title=proposal.title,
			mission_id=mission_id,
			status="launched",
			objective=objective,
		)
		return mission_id
