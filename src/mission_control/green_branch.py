"""Green Branch Pattern for mission-control.

Manages mc/working and mc/green branches. Workers merge into mc/working
without verification; a fixup agent promotes mc/working to mc/green when
verification passes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from mission_control.config import MissionConfig, claude_subprocess_env
from mission_control.db import Database

logger = logging.getLogger(__name__)


@dataclass
class FixupResult:
	promoted: bool = False
	fixup_attempts: int = 0
	failure_output: str = ""


@dataclass
class UnitMergeResult:
	"""Result of verify-and-merge for a single work unit."""

	merged: bool = False
	rebase_ok: bool = True
	verification_passed: bool = False
	failure_output: str = ""


class GreenBranchManager:
	"""Manages the mc/working and mc/green branch lifecycle."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db
		self.workspace: str = ""
		self._merge_lock = asyncio.Lock()

	async def initialize(self, workspace: str) -> None:
		"""Create mc/working and mc/green branches if they don't exist.

		If reset_on_init is True (default), existing branches are reset to the
		latest base branch HEAD to prevent divergence across missions.

		Branches are also created in the source repo so that worker clones
		(which clone from origin = source repo) can check them out.
		"""
		self.workspace = workspace
		base = self.config.target.branch
		gb = self.config.green_branch
		source_repo = self.config.target.path

		# Ensure branches exist in SOURCE repo first (worker clones need them)
		for branch in (gb.working_branch, gb.green_branch):
			ok, _ = await self._run_git_in(source_repo, "rev-parse", "--verify", branch)
			if not ok:
				logger.info("Creating branch %s in source repo from %s", branch, base)
				await self._run_git_in(source_repo, "branch", branch, base)
			elif gb.reset_on_init:
				logger.info("Resetting branch %s in source repo to %s", branch, base)
				await self._run_git_in(
					source_repo, "update-ref", f"refs/heads/{branch}", base,
				)

		# Now set up workspace clone
		await self._run_git("fetch", "origin")
		await self._run_git("checkout", base)

		for branch in (gb.working_branch, gb.green_branch):
			ok, _ = await self._run_git("rev-parse", "--verify", branch)
			if not ok:
				await self._run_git("branch", branch, f"origin/{branch}")
			elif gb.reset_on_init:
				await self._run_git("update-ref", f"refs/heads/{branch}", base)

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

	async def merge_unit(
		self,
		worker_workspace: str,
		branch_name: str,
	) -> UnitMergeResult:
		"""Merge a unit branch directly to mc/green without verification.

		Flow:
		1. Fetch the unit branch from the worker workspace
		2. Create a temp branch from mc/green, merge unit into it
		3. If merge conflict -> fail
		4. Fast-forward mc/green to the merge commit
		5. If auto_push -> push mc/green to main
		"""
		async with self._merge_lock:
			gb = self.config.green_branch
			remote_name = f"worker-{branch_name}"
			temp_branch = f"mc/merge-{branch_name}"

			# Fetch the unit branch from worker workspace
			await self._run_git("remote", "add", remote_name, worker_workspace)
			ok, _ = await self._run_git("fetch", remote_name, branch_name)
			if not ok:
				await self._run_git("remote", "remove", remote_name)
				return UnitMergeResult(failure_output="Failed to fetch unit branch")

			try:
				# Create temp branch from mc/green
				await self._run_git("checkout", gb.green_branch)
				await self._run_git("branch", "-D", temp_branch)  # clean up stale
				await self._run_git("checkout", "-b", temp_branch)

				# Merge unit branch into temp
				ok, output = await self._run_git(
					"merge", "--no-ff", f"{remote_name}/{branch_name}",
					"-m", f"Merge {branch_name} into {gb.green_branch}",
				)
				if not ok:
					logger.warning("Merge conflict for %s: %s", branch_name, output)
					await self._run_git("merge", "--abort")
					return UnitMergeResult(
						rebase_ok=False,
						failure_output=f"Merge conflict: {output[:500]}",
					)

				# Fast-forward mc/green to the merge commit
				await self._run_git("checkout", gb.green_branch)
				merge_ok, merge_out = await self._run_git(
					"merge", "--ff-only", temp_branch,
				)
				if not merge_ok:
					logger.error(
						"Failed to ff mc/green to merge temp: %s", merge_out,
					)
					return UnitMergeResult(
						failure_output="ff-only merge failed",
					)

				logger.info("Merged %s directly into %s", branch_name, gb.green_branch)

				await self._sync_to_source()

				# Auto-push if configured
				if gb.auto_push:
					await self.push_green_to_main()

				return UnitMergeResult(
					merged=True,
					rebase_ok=True,
					verification_passed=True,
				)
			finally:
				# Clean up remote and temp branch
				await self._run_git("checkout", gb.green_branch)
				await self._run_git("branch", "-D", temp_branch)
				await self._run_git("remote", "remove", remote_name)

	async def verify_and_merge_unit(
		self,
		worker_workspace: str,
		branch_name: str,
	) -> UnitMergeResult:
		"""Verify a unit branch in isolation, then merge directly to mc/green.

		This is the core of "no centralized fixup gate" -- each unit is verified
		independently before being accepted into the green branch.

		Flow:
		1. Fetch the unit branch from the worker workspace
		2. Create a temp branch from mc/green, merge unit into it
		3. If merge conflict -> fail
		4. Run verification on the merged state
		5. If verification fails -> fail
		6. Fast-forward mc/green to the merge commit
		7. If auto_push -> push mc/green to main
		"""
		async with self._merge_lock:
			gb = self.config.green_branch
			verify_cmd = self.config.target.verification.command
			remote_name = f"worker-{branch_name}"
			temp_branch = f"mc/verify-{branch_name}"

			# Fetch the unit branch from worker workspace
			await self._run_git("remote", "add", remote_name, worker_workspace)
			ok, _ = await self._run_git("fetch", remote_name, branch_name)
			if not ok:
				await self._run_git("remote", "remove", remote_name)
				return UnitMergeResult(failure_output="Failed to fetch unit branch")

			try:
				# Create temp branch from mc/green
				await self._run_git("checkout", gb.green_branch)
				await self._run_git("branch", "-D", temp_branch)  # clean up stale
				await self._run_git("checkout", "-b", temp_branch)

				# Merge unit branch into temp
				ok, output = await self._run_git(
					"merge", "--no-ff", f"{remote_name}/{branch_name}",
					"-m", f"Merge {branch_name} into {gb.green_branch}",
				)
				if not ok:
					logger.warning("Merge conflict for %s: %s", branch_name, output)
					await self._run_git("merge", "--abort")
					return UnitMergeResult(
						rebase_ok=False,
						failure_output=f"Merge conflict: {output[:500]}",
					)

				# Run verification on merged state
				ok, output = await self._run_command(verify_cmd)
				if not ok:
					logger.warning(
						"Verification failed for %s: %s",
						branch_name, output[:200],
					)
					return UnitMergeResult(
						rebase_ok=True,
						failure_output=output,
					)

				# Verification passed -- fast-forward mc/green
				await self._run_git("checkout", gb.green_branch)
				merge_ok, merge_out = await self._run_git(
					"merge", "--ff-only", temp_branch,
				)
				if not merge_ok:
					logger.error(
						"Failed to ff mc/green to verified temp: %s", merge_out,
					)
					return UnitMergeResult(
						verification_passed=True,
						failure_output="ff-only merge failed",
					)

				logger.info("Merged %s directly into %s", branch_name, gb.green_branch)

				await self._sync_to_source()

				# Auto-push if configured
				if gb.auto_push:
					await self.push_green_to_main()

				return UnitMergeResult(
					merged=True,
					rebase_ok=True,
					verification_passed=True,
				)
			finally:
				# Clean up remote and temp branch
				await self._run_git("checkout", gb.green_branch)
				await self._run_git("branch", "-D", temp_branch)
				await self._run_git("remote", "remove", remote_name)

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
			await self._sync_to_source()
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
				await self._sync_to_source()
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

	async def _sync_to_source(self) -> None:
		"""Sync mc/green and mc/working refs from workspace clone to source repo.

		After promotion in the workspace clone, the source repo's refs are stale.
		Workers provision from the source repo, so they'd start from the old base
		unless we push the updated refs back.
		"""
		source_repo = self.config.target.path
		gb = self.config.green_branch
		for branch in (gb.green_branch, gb.working_branch):
			ok, output = await self._run_git_in(
				source_repo, "fetch", self.workspace, f"{branch}:{branch}",
			)
			if not ok:
				logger.warning("Failed to sync %s to source repo: %s", branch, output)

	async def push_green_to_main(self) -> bool:
		"""Merge mc/green into the push branch and push to origin.

		Runs in the SOURCE repo (config.target.path), not the workspace clone,
		because the clone's origin points back to the source repo, not GitHub.
		"""
		gb = self.config.green_branch
		if not gb.auto_push:
			return False

		source_repo = self.config.target.path
		push_branch = gb.push_branch

		# Fetch latest mc/green from workspace into source repo
		await self._run_git_in(source_repo, "fetch", self.workspace, gb.green_branch)

		await self._run_git_in(source_repo, "checkout", push_branch)

		# Pull remote first to avoid non-fast-forward
		await self._run_git_in(source_repo, "pull", "--rebase", "origin", push_branch)

		ok, output = await self._run_git_in(source_repo, "merge", "--ff-only", "FETCH_HEAD")
		if not ok:
			logger.error("Failed to ff-merge mc/green into %s: %s", push_branch, output)
			return False

		ok, output = await self._run_git_in(source_repo, "push", "origin", push_branch)
		if not ok:
			logger.error("Failed to push %s: %s", push_branch, output)
			return False

		logger.info("Pushed mc/green to origin/%s", push_branch)
		return True

	async def get_green_hash(self) -> str:
		"""Return the current commit hash of mc/green."""
		gb = self.config.green_branch
		ok, output = await self._run_git("rev-parse", gb.green_branch)
		if not ok:
			return ""
		return output.strip()

	async def _run_git_in(self, cwd: str, *args: str) -> tuple[bool, str]:
		"""Run a git command in an arbitrary directory."""
		proc = await asyncio.create_subprocess_exec(
			"git", *args,
			cwd=cwd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await proc.communicate()
		output = stdout.decode() if stdout else ""
		return (proc.returncode == 0, output)

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
		timeout = self.config.scheduler.session_timeout
		proc = await asyncio.create_subprocess_exec(
			"claude", "-p",
			"--permission-mode", "bypassPermissions",
			"--model", self.config.scheduler.model,
			"--max-budget-usd", str(budget),
			cwd=self.workspace,
			stdin=asyncio.subprocess.PIPE,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
			env=claude_subprocess_env(),
		)
		try:
			stdout, _ = await asyncio.wait_for(
				proc.communicate(input=prompt.encode()), timeout=timeout,
			)
		except asyncio.TimeoutError:
			logger.warning("Claude session timed out after %ds", timeout)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			return (False, f"Claude session timed out after {timeout}s")
		output = stdout.decode(errors="replace") if stdout else ""
		return (proc.returncode == 0, output)

	async def _run_command(self, cmd: str) -> tuple[bool, str]:
		"""Run a shell command in self.workspace."""
		timeout = self.config.target.verification.timeout
		proc = await asyncio.create_subprocess_shell(
			cmd,
			cwd=self.workspace,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		try:
			stdout, _ = await asyncio.wait_for(
				proc.communicate(), timeout=timeout,
			)
		except asyncio.TimeoutError:
			logger.warning("Command timed out after %ds: %s", timeout, cmd)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			return (False, f"Command timed out after {timeout}s")
		output = stdout.decode() if stdout else ""
		return (proc.returncode == 0, output)
