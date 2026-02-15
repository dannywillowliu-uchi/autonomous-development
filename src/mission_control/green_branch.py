"""Green Branch Pattern for mission-control.

Manages mc/green branch. Workers produce unit branches that merge directly
into mc/green. Verification runs once at mission end.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from mission_control.config import MissionConfig
from mission_control.db import Database

logger = logging.getLogger(__name__)


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
			proc = await asyncio.create_subprocess_shell(
				setup_cmd,
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

		await self._run_git_in(source_repo, "checkout", push_branch)

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

	async def run_deploy(self) -> tuple[bool, str]:
		"""Run the configured deploy command and optional health check.

		Returns (success, output) tuple.
		"""
		deploy = self.config.deploy
		if not deploy.command:
			return (False, "No deploy command configured")

		source_repo = self.config.target.path
		logger.info("Running deploy: %s", deploy.command)

		proc = await asyncio.create_subprocess_shell(
			deploy.command,
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
		import urllib.request

		deadline = asyncio.get_event_loop().time() + timeout
		while asyncio.get_event_loop().time() < deadline:
			try:
				req = urllib.request.Request(url, method="GET")
				resp = await asyncio.get_event_loop().run_in_executor(
					None, lambda: urllib.request.urlopen(req, timeout=10),  # noqa: S310
				)
				if resp.status == 200:
					return True
			except Exception:
				pass
			await asyncio.sleep(5)
		return False

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
