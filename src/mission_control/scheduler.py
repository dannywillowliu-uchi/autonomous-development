"""Main scheduler loop -- wire everything together."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.discovery import discover_from_snapshot
from mission_control.memory import load_context, summarize_session
from mission_control.reviewer import ReviewVerdict, review_session
from mission_control.session import delete_branch, merge_branch, spawn_session
from mission_control.state import snapshot_project_health

logger = logging.getLogger(__name__)


@dataclass
class SchedulerReport:
	"""Summary of a scheduler run."""

	sessions_run: int = 0
	sessions_helped: int = 0
	sessions_hurt: int = 0
	sessions_neutral: int = 0
	total_cost_usd: float = 0.0
	verdicts: list[ReviewVerdict] = field(default_factory=list)
	stopped_reason: str = ""


class Scheduler:
	"""Main scheduler that orchestrates autonomous development sessions."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db
		self.running = True

	async def run(self, max_sessions: int | None = None) -> SchedulerReport:
		"""Run the scheduler loop.

		Args:
			max_sessions: Maximum sessions to run (None = use config default).
		"""
		if max_sessions is None:
			max_sessions = self.config.scheduler.max_sessions_per_run

		report = SchedulerReport()

		while self.running and report.sessions_run < max_sessions:
			# Take health snapshot
			before = await snapshot_project_health(self.config)

			# Get previous snapshot BEFORE inserting the new one,
			# otherwise get_latest_snapshot returns the just-inserted before
			previous = self.db.get_latest_snapshot()
			self.db.insert_snapshot(before)

			# Discover work
			recent = self.db.get_recent_sessions(10)
			tasks = discover_from_snapshot(before, self.config, recent, previous)

			if not tasks:
				report.stopped_reason = "no_work"
				logger.info("No work discovered, stopping")
				break

			task = tasks[0]
			logger.info("Working on: %s (priority %d)", task.description, task.priority)

			# Load context
			context = load_context(task, self.db, self.config)

			# Spawn session
			try:
				session = await spawn_session(task, before, self.config, context)
			except (OSError, FileNotFoundError) as exc:
				logger.error("Failed to spawn session: %s", exc)
				report.stopped_reason = "spawn_error"
				break

			# Take after snapshot
			after = await snapshot_project_health(self.config)
			self.db.insert_snapshot(after)

			# Review
			verdict = review_session(
				before, after, session,
				auto_merge=self.config.scheduler.git.auto_merge,
			)
			logger.info("Verdict: %s", verdict.verdict)

			# Act on verdict
			cwd = str(self.config.target.resolved_path)
			if verdict.should_revert:
				reverted = await delete_branch(session.branch_name, self.config.target.branch, cwd)
				if reverted:
					session.status = "reverted"
				else:
					logger.error("Failed to delete branch %s, attempting checkout recovery", session.branch_name)
					session.status = "revert_failed"
					# Attempt to at least get back to the base branch
					recover = await asyncio.create_subprocess_exec(
						"git", "checkout", self.config.target.branch,
						cwd=cwd,
						stdout=asyncio.subprocess.PIPE,
						stderr=asyncio.subprocess.STDOUT,
					)
					await recover.communicate()
			elif verdict.should_merge:
				merged = await merge_branch(session.branch_name, self.config.target.branch, cwd)
				if not merged:
					logger.error("Failed to merge branch %s into %s", session.branch_name, self.config.target.branch)
					session.status = "merge_failed"

			# Update session summary
			session.output_summary = summarize_session(session, verdict)

			# Persist session (snapshots already inserted above)
			self.db.insert_session(session)

			# Update report
			report.sessions_run += 1
			report.verdicts.append(verdict)
			if session.cost_usd:
				report.total_cost_usd += session.cost_usd

			if verdict.verdict == "helped":
				report.sessions_helped += 1
			elif verdict.verdict == "hurt":
				report.sessions_hurt += 1
			else:
				report.sessions_neutral += 1

			# Cooldown
			if self.running and report.sessions_run < max_sessions:
				await asyncio.sleep(self.config.scheduler.cooldown)

		if not report.stopped_reason:
			report.stopped_reason = "max_sessions" if report.sessions_run >= max_sessions else "stopped"

		return report

	def stop(self) -> None:
		"""Signal the scheduler to stop after the current session."""
		self.running = False
