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
from mission_control.models import VerificationNodeKind, VerificationReport
from mission_control.state import run_verification_nodes

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
	failed_kinds: list[str] = field(default_factory=list)


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
	merge_commit_hash: str = ""
	changed_files: list[str] = field(default_factory=list)
	sync_ok: bool = True
	verification_report: VerificationReport | None = None
	hitl_decision: str = ""


class GreenBranchManager:
	"""Manages the mc/green branch lifecycle."""

	def __init__(self, config: MissionConfig, db: Database) -> None:
		self.config = config
		self.db = db
		self.workspace: str = ""
		self._merge_lock = asyncio.Lock()
		self._hitl_gate: object | None = None  # ApprovalGate, lazily typed to avoid circular import

	def configure_hitl(self, gate: object) -> None:
		"""Set the HITL approval gate for push and large merge checks."""
		self._hitl_gate = gate

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
		"""Merge a unit branch into mc/green with smoke-test verification.

		Flow:
		1. Fetch the unit branch from the worker workspace
		2. Create a temp branch from mc/green, merge unit into it
		3. If merge conflict -> fail
		4. Run verification command on the merged result
		5. If verification fails -> abort
		6. Fast-forward mc/green to the merge commit
		7. If auto_push -> push mc/green to main
		"""
		async with self._merge_lock:
			gb = self.config.green_branch
			remote_name = f"worker-{branch_name}"
			temp_branch = f"mc/merge-{branch_name}"
			rebase_branch = f"mc/rebase-{branch_name}"

			# Fetch the unit branch from worker workspace
			await self._run_git("remote", "add", remote_name, worker_workspace)
			ok, _ = await self._run_git("fetch", remote_name, branch_name)
			if not ok:
				await self._run_git("remote", "remove", remote_name)
				return UnitMergeResult(failure_output="Failed to fetch unit branch", failure_stage="fetch")

			try:
				# Rebase unit branch onto current mc/green
				await self._run_git("branch", "-D", rebase_branch)  # clean stale
				await self._run_git("checkout", "-b", rebase_branch, f"{remote_name}/{branch_name}")

				rebase_ok, rebase_output = await self._run_git("rebase", gb.green_branch)
				if not rebase_ok:
					# Retry rebase once: green may have advanced since we fetched
					logger.info("Rebase conflict for %s, retrying after brief wait", branch_name)
					await self._run_git("rebase", "--abort")
					await asyncio.sleep(2)
					# Re-checkout the unit branch and retry rebase onto latest green
					await self._run_git("checkout", rebase_branch)
					rebase_ok, rebase_output = await self._run_git("rebase", gb.green_branch)
					if not rebase_ok:
						logger.warning("Rebase retry also failed for %s: %s", branch_name, rebase_output)
						await self._run_git("rebase", "--abort")
						return UnitMergeResult(
							rebase_ok=False,
							failure_output=f"Rebase conflict (after retry): {rebase_output[:500]}",
							failure_stage="rebase_conflict",
						)
					logger.info("Rebase retry succeeded for %s", branch_name)
				logger.info("Rebased %s onto %s", branch_name, gb.green_branch)

				# Create temp branch from mc/green
				await self._run_git("checkout", gb.green_branch)
				await self._run_git("branch", "-D", temp_branch)  # clean up stale
				await self._run_git("checkout", "-b", temp_branch)

				# Merge unit branch into temp
				ok, output = await self._run_git(
					"merge", "--no-ff", rebase_branch,
					"-m", f"Merge {branch_name} into {gb.green_branch}",
				)
				if not ok:
					logger.warning("Merge conflict for %s: %s", branch_name, output)
					await self._run_git("merge", "--abort")
					return UnitMergeResult(
						failure_output=f"Merge conflict: {output[:500]}",
						failure_stage="merge_conflict",
					)

				# Run smoke-test verification before promoting to mc/green
				report = await self._run_verification()
				if not report.overall_passed:
					logger.warning("Verification failed for %s: %s", branch_name, report.raw_output[-500:])
					await self._run_git("checkout", gb.green_branch)
					return UnitMergeResult(
						verification_passed=False,
						failure_output=report.raw_output,
						failure_stage="verification",
						verification_report=report,
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

				# Capture the merge commit hash and actual changed files
				_, merge_hash = await self._run_git("rev-parse", "HEAD")
				merge_commit_hash = merge_hash.strip()

				# Get actual files changed via git diff (not files_hint)
				_, diff_output = await self._run_git(
					"diff", "--name-only", "HEAD~1", "HEAD",
				)
				changed_files = [
					f.strip() for f in diff_output.splitlines() if f.strip()
				]

				logger.info("Merged %s directly into %s", branch_name, gb.green_branch)

				hitl_decision = ""

				# Gate A: Large merge HITL check
				large_gate = self.config.hitl.large_merge_gate
				if large_gate.enabled and self._hitl_gate is not None:
					# Compute lines changed
					_, stat_output = await self._run_git(
						"diff", "--stat", "HEAD~1", "HEAD",
					)
					lines_changed = self._count_diff_lines(stat_output)
					files_count = len(changed_files)

					if (
						lines_changed >= large_gate.large_merge_threshold_lines
						or files_count >= large_gate.large_merge_threshold_files
					):
						from mission_control.hitl import ApprovalRequest
						req = ApprovalRequest(
							gate_type="large_merge",
							timeout_seconds=large_gate.timeout_seconds,
							timeout_action=large_gate.timeout_action,
							context={
								"branch": branch_name,
								"lines_changed": lines_changed,
								"files_changed": files_count,
								"changed_files": changed_files[:20],
							},
						)
						approved = await self._hitl_gate.request_approval(req)
						if not approved:
							hitl_decision = "large_merge_denied"
							logger.info("HITL denied large merge for %s -- reverting", branch_name)
							await self._run_git("reset", "--hard", "HEAD~1")
							return UnitMergeResult(
								failure_stage="hitl_large_merge_denied",
								failure_output=f"HITL denied: {lines_changed} lines, {files_count} files",
								hitl_decision=hitl_decision,
							)
						hitl_decision = "large_merge_approved"

				# Commit MISSION_STATE.md into mc/green if it exists
				state_path = Path(self.config.target.resolved_path) / "MISSION_STATE.md"
				if state_path.exists():
					try:
						content = state_path.read_text()
						await self.commit_state_file(content)
					except Exception as exc:
						logger.warning("Failed to commit MISSION_STATE.md: %s", exc)

				sync_ok = await self._sync_to_source()

				# Auto-push if configured
				if gb.auto_push:
					# Gate B: Push HITL check
					push_gate = self.config.hitl.push_gate
					push_approved = True
					if push_gate.enabled and self._hitl_gate is not None:
						from mission_control.hitl import ApprovalRequest
						req = ApprovalRequest(
							gate_type="push",
							timeout_seconds=push_gate.timeout_seconds,
							timeout_action=push_gate.timeout_action,
							context={
								"branch": branch_name,
								"push_branch": gb.push_branch,
								"merge_commit": merge_commit_hash,
								"files_changed": len(changed_files),
							},
						)
						push_approved = await self._hitl_gate.request_approval(req)
						if not push_approved:
							hitl_decision = hitl_decision or "push_denied"
							logger.info("HITL denied push for %s -- merge stays, push skipped", branch_name)

					if push_approved:
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
					merge_commit_hash=merge_commit_hash,
					changed_files=changed_files,
					sync_ok=sync_ok,
					verification_report=report,
					hitl_decision=hitl_decision,
				)
			finally:
				# Clean up remote, temp branch, and rebase branch
				await self._run_git("checkout", gb.green_branch)
				await self._run_git("branch", "-D", temp_branch)
				await self._run_git("branch", "-D", rebase_branch)
				await self._run_git("remote", "remove", remote_name)

	async def _zfc_generate_fixup_strategies(self, failure_output: str, n: int) -> list[str] | None:
		"""Generate N fixup strategies via LLM. Returns list of strategy strings or None."""
		from mission_control.config import claude_subprocess_env

		prompt = (
			f"You are a code fixup strategist. Given a test/verification failure, "
			f"generate exactly {n} distinct strategies to fix the issue.\n\n"
			f"## Failure Output\n{failure_output[:2000]}\n\n"
			f"Each strategy should be a short instruction paragraph.\n\n"
			f"You MUST end your response with:\n"
			f'FIXUP_STRATEGIES:{{"strategies": ["strategy1", "strategy2", ...]}}'
		)

		zfc = self.config.zfc
		model = zfc.model or self.config.scheduler.model
		timeout = zfc.llm_timeout

		try:
			proc = await asyncio.create_subprocess_exec(
				"claude", "--print", "--output-format", "text",
				"--model", model,
				"--max-turns", "1",
				"-p", prompt,
				cwd=self.workspace,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				env=claude_subprocess_env(),
			)
			stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
			output = stdout.decode() if stdout else ""
		except asyncio.TimeoutError:
			logger.warning("ZFC fixup strategy generation timed out after %ds", timeout)
			try:
				proc.kill()
				await proc.wait()
			except ProcessLookupError:
				pass
			return None
		except (FileNotFoundError, OSError) as exc:
			logger.warning("ZFC fixup strategy generation failed: %s", exc)
			return None

		if proc.returncode != 0:
			return None

		marker = "FIXUP_STRATEGIES:"
		idx = output.rfind(marker)
		if idx == -1:
			logger.warning("No FIXUP_STRATEGIES marker in ZFC output")
			return None

		from mission_control.json_utils import extract_json_from_text
		remainder = output[idx + len(marker):]
		data = extract_json_from_text(remainder)
		if not isinstance(data, dict):
			return None

		strategies = data.get("strategies")
		if not isinstance(strategies, list) or not strategies:
			return None

		return [str(s) for s in strategies[:n]]

	async def run_fixup(self, failure_output: str) -> FixupResult:
		"""Run N fixup candidates in parallel and select the best one.

		Each candidate gets a different prompt approach and runs on a temporary
		branch. Verification is run on each. The best candidate (most tests
		passing, fewest lint errors, smallest diff on tie) is merged into mc/green.
		"""
		gb = self.config.green_branch
		n = gb.fixup_candidates

		# ZFC: try LLM-generated fixup strategies
		prompts = None
		if self.config.zfc.zfc_fixup_prompts:
			prompts = await self._zfc_generate_fixup_strategies(failure_output, n)
		if prompts is None:
			prompts = list(FIXUP_PROMPTS[:n])
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
		report = await self._run_verification()
		candidate.verification_passed = report.overall_passed
		candidate.failed_kinds = [k.value for k in report.failed_kinds()]

		# Extract test and lint metrics from report
		for r in report.results:
			if r.kind == VerificationNodeKind.PYTEST:
				candidate.tests_passed = r.metrics.get("test_passed", 0)
			elif r.kind == VerificationNodeKind.RUFF:
				candidate.lint_errors = r.metrics.get("lint_errors", 0)

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

	async def _sync_to_source(self) -> bool:
		"""Sync mc/green and mc/working refs from workspace clone to source repo.

		After promotion in the workspace clone, the source repo's refs are stale.
		Workers provision from the source repo, so they'd start from the old base
		unless we push the updated refs back.

		Returns True if all syncs succeeded.
		"""
		source_repo = self.config.target.path
		gb = self.config.green_branch
		all_ok = True
		for branch in (gb.green_branch, gb.working_branch):
			ok, output = await self._run_git_in(
				source_repo, "fetch", self.workspace, f"{branch}:{branch}",
			)
			if not ok:
				logger.warning("Failed to sync %s to source repo: %s", branch, output)
				all_ok = False
		return all_ok

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

	async def run_reconciliation_check(self) -> tuple[bool, str]:
		"""Run verification on the current green branch state.

		Used as a reconciler sweep after merging multiple units to catch
		cross-unit integration issues (import conflicts, duplicate definitions).
		"""
		gb = self.config.green_branch
		await self._run_git("checkout", gb.green_branch)
		report = await self._run_verification()
		return (report.overall_passed, report.raw_output)

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

	async def _run_verification(self) -> VerificationReport:
		"""Run verification nodes and return a structured report.

		Uses self._run_command for single-command fallback (backward compat
		with tests that mock _run_command). Delegates to run_verification_nodes
		only when explicit nodes are configured.
		"""
		from mission_control.state import _build_result_from_single_command

		await self._reanchor_editable_install()

		nodes = self.config.target.verification.nodes
		if nodes:
			return await run_verification_nodes(self.config, self.workspace)
		# Single-command fallback using self._run_command
		ok, output = await self._run_command(self.config.target.verification.command)
		return _build_result_from_single_command(output, 0 if ok else 1)

	async def _reanchor_editable_install(self) -> None:
		"""Re-anchor the editable install to the source repo before verification.

		Worker pool clones share the source .venv via symlink. If a worker runs
		`pip install -e .` in its clone, the editable path gets hijacked to point
		at the clone instead of the source repo. This causes verification to import
		stale code from the clone rather than the merged code in the source repo.
		"""
		venv_python = Path(self.workspace) / ".venv" / "bin" / "python"
		if not venv_python.exists():
			return
		proc = await asyncio.create_subprocess_exec(
			"uv", "pip", "install", "-e", ".",
			"--python", str(venv_python),
			cwd=self.workspace,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		stdout, _ = await proc.communicate()
		if proc.returncode != 0:
			output = stdout.decode(errors="replace") if stdout else ""
			logger.warning("Failed to re-anchor editable install: %s", output[:300])

	async def _run_command(self, cmd: str | list[str]) -> tuple[bool, str]:
		"""Run a command in self.workspace using shell for string commands."""
		timeout = self.config.target.verification.timeout
		if isinstance(cmd, str):
			proc = await asyncio.create_subprocess_shell(
				cmd,
				cwd=self.workspace,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
		else:
			proc = await asyncio.create_subprocess_exec(
				*cmd,
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
