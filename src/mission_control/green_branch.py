"""Green Branch Pattern for mission-control.

Manages mc/working and mc/green branches. Workers merge into mc/working
without verification; a fixup agent promotes mc/working to mc/green when
verification passes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from mission_control.config import MissionConfig
from mission_control.db import Database

logger = logging.getLogger(__name__)


@dataclass
class FixupResult:
	promoted: bool = False
	fixup_attempts: int = 0
	failure_output: str = ""


class GreenBranchManager:
	"""Manages the mc/working and mc/green branch lifecycle."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db
		self.workspace: str = ""
		self._merge_lock = asyncio.Lock()

	async def initialize(self, workspace: str) -> None:
		"""Create mc/working and mc/green branches if they don't exist."""
		self.workspace = workspace
		base = self.config.target.branch
		gb = self.config.green_branch

		# Ensure we're on the base branch first
		await self._run_git("checkout", base)

		for branch in (gb.working_branch, gb.green_branch):
			ok, _ = await self._run_git("rev-parse", "--verify", branch)
			if not ok:
				logger.info("Creating branch %s from %s", branch, base)
				await self._run_git("branch", branch, base)

	async def merge_to_working(self, worker_workspace: str, branch_name: str) -> bool:
		"""Merge a worker branch into mc/working. Serialized via lock."""
		async with self._merge_lock:
			gb = self.config.green_branch
			remote_name = f"worker-{branch_name}"

			# Add worker workspace as a remote (ignore error if already exists)
			await self._run_git("remote", "add", remote_name, worker_workspace)
			ok, _ = await self._run_git("fetch", remote_name, branch_name)
			if not ok:
				await self._run_git("remote", "remove", remote_name)
				return False

			await self._run_git("checkout", gb.working_branch)
			ok, output = await self._run_git(
				"merge", "--no-ff", f"{remote_name}/{branch_name}",
				"-m", f"Merge {branch_name} into {gb.working_branch}",
			)

			# Clean up remote
			await self._run_git("remote", "remove", remote_name)

			if not ok:
				logger.warning("Merge conflict for %s: %s", branch_name, output)
				await self._run_git("merge", "--abort")
				return False

			return True

	async def run_fixup(self) -> FixupResult:
		"""Run verification on mc/working; promote to mc/green if passing.

		Each fixup attempt saves the pre-fixup state. If the fixup agent crashes
		or makes things worse, the working branch is restored to the pre-fixup
		state before the next attempt.
		"""
		gb = self.config.green_branch
		verify_cmd = self.config.target.verification.command

		await self._run_git("checkout", gb.working_branch)

		# Run verification
		ok, output = await self._run_command(verify_cmd)
		if ok:
			await self._run_git("checkout", gb.green_branch)
			merge_ok, merge_out = await self._run_git("merge", "--ff-only", gb.working_branch)
			if not merge_ok:
				logger.error(
					"Failed to fast-forward %s to %s: %s",
					gb.green_branch, gb.working_branch, merge_out,
				)
				return FixupResult(promoted=False, failure_output="ff-only merge failed")
			logger.info("Promoted %s to %s (clean pass)", gb.working_branch, gb.green_branch)
			return FixupResult(promoted=True)

		# Verification failed -- spawn fixup agent
		max_attempts = gb.fixup_max_attempts
		for attempt in range(1, max_attempts + 1):
			logger.info("Fixup attempt %d/%d", attempt, max_attempts)

			# Save pre-fixup state so we can restore on failure
			_, pre_fixup_hash = await self._run_git("rev-parse", "HEAD")
			pre_fixup_hash = pre_fixup_hash.strip()

			prompt = (
				"You are a fixup agent. ONLY fix these specific verification "
				"failures. No features, no refactoring. "
				f"Failures:\n{output}"
			)

			fixup_budget = self.config.scheduler.budget.fixup_budget_usd

			try:
				fixup_ok, _ = await self._run_claude(prompt, fixup_budget)
			except Exception as exc:
				logger.warning("Fixup agent crashed on attempt %d: %s", attempt, exc)
				await self._restore_to(pre_fixup_hash)
				continue

			if not fixup_ok:
				logger.warning("Fixup agent failed on attempt %d, restoring state", attempt)
				await self._restore_to(pre_fixup_hash)
				continue

			# Re-run verification
			ok, output = await self._run_command(verify_cmd)
			if ok:
				await self._run_git("checkout", gb.green_branch)
				merge_ok, merge_out = await self._run_git("merge", "--ff-only", gb.working_branch)
				if not merge_ok:
					logger.error(
						"Failed to fast-forward %s to %s: %s",
						gb.green_branch, gb.working_branch, merge_out,
					)
					return FixupResult(
						promoted=False, fixup_attempts=attempt,
						failure_output="ff-only merge failed",
					)
				logger.info(
					"Promoted %s to %s after %d fixup attempt(s)",
					gb.working_branch, gb.green_branch, attempt,
				)
				return FixupResult(promoted=True, fixup_attempts=attempt)

			# Verification still failing -- restore to pre-fixup state
			logger.warning("Verification still failing after attempt %d, restoring state", attempt)
			await self._restore_to(pre_fixup_hash)

		return FixupResult(
			promoted=False,
			fixup_attempts=max_attempts,
			failure_output=output,
		)

	async def _restore_to(self, commit_hash: str) -> None:
		"""Restore working branch to a specific commit, discarding changes."""
		await self._run_git("reset", "--hard", commit_hash)
		await self._run_git("clean", "-fd")

	async def get_green_hash(self) -> str:
		"""Return the current commit hash of mc/green."""
		gb = self.config.green_branch
		ok, output = await self._run_git("rev-parse", gb.green_branch)
		if not ok:
			return ""
		return output.strip()

	async def _run_git(self, *args: str) -> tuple[bool, str]:
		"""Run a git command in self.workspace."""
		proc = await asyncio.create_subprocess_exec(
			"git", *args,
			cwd=self.workspace,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await proc.communicate()
		output = stdout.decode() if stdout else ""
		return (proc.returncode == 0, output)

	async def _run_claude(self, prompt: str, budget: float) -> tuple[bool, str]:
		"""Run a Claude session with the prompt passed via stdin to avoid injection."""
		proc = await asyncio.create_subprocess_exec(
			"claude", "-p",
			"--permission-mode", "bypassPermissions",
			"--max-budget-usd", str(budget),
			cwd=self.workspace,
			stdin=asyncio.subprocess.PIPE,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await proc.communicate(input=prompt.encode())
		output = stdout.decode(errors="replace") if stdout else ""
		return (proc.returncode == 0, output)

	async def _run_command(self, cmd: str) -> tuple[bool, str]:
		"""Run a shell command in self.workspace."""
		proc = await asyncio.create_subprocess_shell(
			cmd,
			cwd=self.workspace,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await proc.communicate()
		output = stdout.decode() if stdout else ""
		return (proc.returncode == 0, output)
