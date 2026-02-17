"""Green Branch Pattern for mission-control.

Manages mc/green branch. Workers produce unit branches that merge directly
into mc/green. Verification runs once at mission end.
"""

from __future__ import annotations

import asyncio
import logging
import random
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.state import _parse_pytest, _parse_ruff

logger = logging.getLogger(__name__)

FIXUP_PROMPTS = [
	"Fix the failing tests by modifying the implementation code. Do NOT change any test files.",
	"Fix by adjusting the test expectations to match the current implementation behavior.",
	"Fix by refactoring the surrounding code to make both tests and implementation consistent.",
]


@dataclass
class FixupCandidate:
	"""Result of a single fixup candidate attempt."""

	branch: str = ""
	verification_passed: bool = False
	tests_passed: int = 0
	lint_errors: int = 0
	diff_lines: int = 0


@dataclass
class FixupResult:
	"""Result of the N-of-M fixup selection process."""

	success: bool = False
	winner: FixupCandidate | None = None
	candidates: list[FixupCandidate] = field(default_factory=list)


@dataclass
class UnitMergeResult:
	"""Result of merging a single work unit into mc/green."""

	merged: bool = False
	rebase_ok: bool = True
	verification_passed: bool = False
	failure_output: str = ""
	failure_stage: str = ""


class GreenBranchManager:
	"""Manages the mc/green branch lifecycle."""

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

		# Run setup command if configured (e.g., npm install)
		setup_cmd = self.config.target.verification.setup_command
		if setup_cmd:
			setup_timeout = self.config.target.verification.setup_timeout
			logger.info("Running workspace setup: %s", setup_cmd)
			proc = await asyncio.create_subprocess_exec(
				*shlex.split(setup_cmd),
				cwd=self.workspace,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			try:
				stdout, _ = await asyncio.wait_for(
					proc.communicate(), timeout=setup_timeout,
				)
			except asyncio.TimeoutError:
				try:
					proc.kill()
					await proc.wait()
				except ProcessLookupError:
					pass
				raise RuntimeError(
					f"Workspace setup timed out after {setup_timeout}s: {setup_cmd}"
				)
			if proc.returncode != 0:
				output = stdout.decode() if stdout else ""
				raise RuntimeError(
					f"Workspace setup failed (exit {proc.returncode}): {output[:500]}"
				)

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
				return UnitMergeResult(failure_output="Failed to fetch unit branch", failure_stage="fetch")

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
						failure_stage="merge_conflict",
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
						failure_stage="fast_forward",
					)

				logger.info("Merged %s directly into %s", branch_name, gb.green_branch)

				# Commit MISSION_STATE.md into mc/green if it exists
				state_path = Path(self.config.target.resolved_path) / "MISSION_STATE.md"
				if state_path.exists():
					try:
						content = state_path.read_text()
						await self.commit_state_file(content)
					except Exception as exc:
						logger.warning("Failed to commit MISSION_STATE.md: %s", exc)

				await self._sync_to_source()

				# Auto-push if configured
				if gb.auto_push:
					push_ok = await self.push_green_to_main()
					# Deploy after push if configured
					deploy = self.config.deploy
					if push_ok and deploy.enabled and deploy.on_auto_push:
						deploy_ok, deploy_out = await self.run_deploy()
						if not deploy_ok:
							logger.warning("Post-push deploy failed: %s", deploy_out[:200])

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

	async def run_fixup(self, failure_output: str) -> FixupResult:
		"""Run N fixup candidates in parallel and select the best one.

		Each candidate gets a different prompt approach and runs on a temporary
		branch. Verification is run on each. The best candidate (most tests
		passing, fewest lint errors, smallest diff on tie) is merged into mc/green.
		"""
		gb = self.config.green_branch
		n = gb.fixup_candidates
		prompts = FIXUP_PROMPTS[:n]
		# Pad with the first prompt if N > len(FIXUP_PROMPTS)
		while len(prompts) < n:
			prompts.append(FIXUP_PROMPTS[0])

		green_ref = gb.green_branch

		# Run all candidates concurrently
		tasks = [
			self._run_fixup_candidate(i, prompts[i], failure_output, green_ref)
			for i in range(n)
		]
		candidates = await asyncio.gather(*tasks)

		result = FixupResult(candidates=list(candidates))

		# Find the best candidate
		passing = [c for c in candidates if c.verification_passed]
		if not passing:
			logger.warning("All %d fixup candidates failed verification", n)
			return result

		# Sort by: most tests passed (desc), fewest lint errors (asc), smallest diff (asc)
		passing.sort(key=lambda c: (-c.tests_passed, c.lint_errors, c.diff_lines))
		winner = passing[0]
		result.winner = winner
		result.success = True

		# Merge the winner branch into mc/green
		async with self._merge_lock:
			await self._run_git("checkout", green_ref)
			ok, output = await self._run_git(
				"merge", "--ff-only", winner.branch,
			)
			if not ok:
				# Try non-ff merge as fallback
				ok, output = await self._run_git(
					"merge", "--no-ff", winner.branch,
					"-m", f"Merge fixup candidate {winner.branch}",
				)
			if not ok:
				logger.error("Failed to merge winning fixup branch %s: %s", winner.branch, output)
				result.success = False
				result.winner = None
				return result

			logger.info(
				"Fixup succeeded: branch=%s tests_passed=%d lint_errors=%d diff_lines=%d",
				winner.branch, winner.tests_passed, winner.lint_errors, winner.diff_lines,
			)

		# Cleanup all candidate branches
		for c in candidates:
			await self._run_git("branch", "-D", c.branch)

		return result

	async def _run_fixup_candidate(
		self,
		index: int,
		prompt: str,
		failure_output: str,
		green_ref: str,
	) -> FixupCandidate:
		"""Run a single fixup candidate on a temporary branch."""
		branch = f"mc/fixup-candidate-{index}"
		candidate = FixupCandidate(branch=branch)

		# Create candidate branch from mc/green
		await self._run_git("branch", "-D", branch)  # cleanup stale
		await self._run_git("checkout", green_ref)
		ok, _ = await self._run_git("checkout", "-b", branch)
		if not ok:
			logger.warning("Failed to create fixup branch %s", branch)
			return candidate

		# Run the fixup session (Claude Code subprocess)
		full_prompt = (
			f"{prompt}\n\n"
			f"## Verification Failure\n{failure_output}\n\n"
			f"## Verification Command\n{self.config.target.verification.command}\n\n"
			f"Run the verification command after making changes. "
			f"Commit your fix if verification passes."
		)

		ok, output = await self._run_fixup_session(full_prompt)

		# Run verification on the candidate branch
		verify_ok, verify_output = await self._run_command(
			self.config.target.verification.command,
		)
		candidate.verification_passed = verify_ok

		# Parse test and lint results
		pytest_data = _parse_pytest(verify_output)
		ruff_data = _parse_ruff(verify_output)
		candidate.tests_passed = pytest_data["test_passed"]
		candidate.lint_errors = ruff_data["lint_errors"]

		# Measure diff size
		diff_ok, diff_output = await self._run_git(
			"diff", "--stat", green_ref, branch,
		)
		if diff_ok:
			candidate.diff_lines = self._count_diff_lines(diff_output)

		# Return to green branch
		await self._run_git("checkout", green_ref)

		return candidate

	def _get_fixup_model(self) -> str:
		"""Resolve the model for fixup sessions.

		Uses config.models.fixup_model if available, falls back to scheduler.model.
		"""
		models = getattr(self.config, "models", None)
		if models is not None:
			fixup_model = getattr(models, "fixup_model", "")
			if fixup_model:
				return fixup_model
		return self.config.scheduler.model

	async def _run_fixup_session(self, prompt: str) -> tuple[bool, str]:
		"""Spawn a Claude Code subprocess for fixup.

		Returns (success, output) tuple.
		"""
		model = self._get_fixup_model()
		cmd = [
			"claude", "--print", "--output-format", "text",
			"--model", model,
			"--max-turns", "5",
			"-p", prompt,
		]
		return await self._run_command(cmd)

	@staticmethod
	def _count_diff_lines(diff_stat_output: str) -> int:
		"""Count total insertions + deletions from git diff --stat output.

		The last line of git diff --stat looks like:
		  3 files changed, 10 insertions(+), 5 deletions(-)
		"""
		for line in reversed(diff_stat_output.splitlines()):
			if "changed" in line:
				total = 0
				parts = line.split(",")
				for part in parts:
					part = part.strip()
					if "insertion" in part or "deletion" in part:
						digits = ""
						for ch in part:
							if ch.isdigit():
								digits += ch
						if digits:
							total += int(digits)
				return total
		return 0

	async def commit_state_file(self, content: str) -> bool:
		"""Write MISSION_STATE.md to the green branch workspace, stage, and commit it."""
		workspace_path = Path(self.workspace) / "MISSION_STATE.md"
		workspace_path.write_text(content)
		await self._run_git("add", "MISSION_STATE.md")
		ok, output = await self._run_git("commit", "-m", "Update MISSION_STATE.md")
		if not ok:
			if "nothing to commit" in output:
				return True
			logger.warning("Failed to commit MISSION_STATE.md: %s", output)
			return False
		return True

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
		green_ref = "refs/mc/green-push"

		# Fetch mc/green into a named ref (FETCH_HEAD gets overwritten by pull)
		ok, output = await self._run_git_in(
			source_repo, "fetch", self.workspace,
			f"{gb.green_branch}:{green_ref}",
		)
		if not ok:
			logger.error("Failed to fetch mc/green: %s", output)
			return False

		# Stash any dirty state (e.g. MISSION_STATE.md) before checkout
		stashed = False
		ok_stash, stash_out = await self._run_git_in(
			source_repo, "stash", "--include-untracked",
		)
		if ok_stash and "No local changes" not in stash_out:
			stashed = True

		try:
			ok, output = await self._run_git_in(source_repo, "checkout", push_branch)
			if not ok:
				logger.error("Failed to checkout %s: %s", push_branch, output)
				return False

			# Pull remote first to avoid non-fast-forward
			await self._run_git_in(source_repo, "pull", "--rebase", "origin", push_branch)

			ok, output = await self._run_git_in(source_repo, "merge", "--ff-only", green_ref)
			if not ok:
				logger.error("Failed to ff-merge mc/green into %s: %s", push_branch, output)
				return False

			ok, output = await self._run_git_in(source_repo, "push", "origin", push_branch)
			if not ok:
				logger.error("Failed to push %s: %s", push_branch, output)
				return False

			logger.info("Pushed mc/green to origin/%s", push_branch)
			return True
		finally:
			if stashed:
				await self._run_git_in(source_repo, "stash", "pop")

	async def run_deploy(self) -> tuple[bool, str]:
		"""Run the configured deploy command and optional health check.

		Returns (success, output) tuple.
		"""
		deploy = self.config.deploy
		if not deploy.command:
			return (False, "No deploy command configured")

		source_repo = self.config.target.path
		logger.info("Running deploy: %s", deploy.command)

		proc = await asyncio.create_subprocess_exec(
			*shlex.split(deploy.command),
			cwd=source_repo,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		try:
			stdout, _ = await asyncio.wait_for(
				proc.communicate(), timeout=deploy.timeout,
			)
		except asyncio.TimeoutError:
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			return (False, f"Deploy timed out after {deploy.timeout}s")

		output = stdout.decode() if stdout else ""
		if proc.returncode != 0:
			return (False, f"Deploy failed (exit {proc.returncode}): {output[:500]}")

		logger.info("Deploy command succeeded")

		# Health check if URL is configured
		if deploy.health_check_url:
			healthy = await self._poll_health_check(
				deploy.health_check_url, deploy.health_check_timeout,
			)
			if not healthy:
				return (False, f"Health check failed: {deploy.health_check_url}")
			logger.info("Health check passed: %s", deploy.health_check_url)

		return (True, output)

	async def _poll_health_check(self, url: str, timeout: int) -> bool:
		"""Poll a URL with HTTP GET until 200 or timeout."""
		loop = asyncio.get_event_loop()
		deadline = loop.time() + timeout
		async with httpx.AsyncClient() as client:
			while True:
				remaining = deadline - loop.time()
				if remaining <= 0:
					return False
				try:
					resp = await client.get(url, timeout=min(remaining, 10.0))
					if resp.status_code == 200:
						return True
				except (httpx.HTTPError, OSError):
					pass
				# Check if we have time for another poll
				remaining = deadline - loop.time()
				if remaining <= 0:
					return False
				await asyncio.sleep(random.uniform(3.0, 7.0))

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

	async def _run_command(self, cmd: str | list[str]) -> tuple[bool, str]:
		"""Run a command in self.workspace using subprocess exec."""
		timeout = self.config.target.verification.timeout
		args = shlex.split(cmd) if isinstance(cmd, str) else cmd
		proc = await asyncio.create_subprocess_exec(
			*args,
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
