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
from mission_control.trace_log import TraceEvent, TraceLogger

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

	def __init__(self, config: MissionConfig, db: Database, trace_logger: TraceLogger | None = None) -> None:
		self.config = config
		self.db = db
		self.workspace: str = ""
		self._merge_lock = asyncio.Lock()
		self._hitl_gate: object | None = None  # ApprovalGate, lazily typed to avoid circular import
		self._merges_since_push: int = 0
		self._trace_logger = trace_logger

	def _trace(self, event_type: str, **details: object) -> None:
		"""Emit a trace event if a logger is configured."""
		if self._trace_logger is None:
			return
		self._trace_logger.write(TraceEvent(event_type=event_type, details=details))

	def configure_hitl(self, gate: object) -> None:
		"""Set the HITL approval gate for push and large merge checks."""
		self._hitl_gate = gate

	async def maybe_push(self, force: bool = False) -> bool:
		"""Push if merges since last push >= push_batch_size or force=True.

		Returns True if a push was performed.
		"""
		gb = self.config.green_branch
		if not gb.auto_push:
			return False
		if force or self._merges_since_push >= gb.push_batch_size:
			result = await self.push_green_to_main()
			if result:
				self._merges_since_push = 0
			return result
		return False

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

		# Flush unpushed mc/green work before resetting
		if gb.reset_on_init and gb.auto_push:
			green_exists, _ = await self._run_git_in(
				source_repo, "rev-parse", "--verify", gb.green_branch,
			)
			if green_exists:
				# Check if mc/green has commits ahead of base (unpushed work)
				ok, ahead = await self._run_git_in(
					source_repo, "rev-list", "--count", f"{base}..{gb.green_branch}",
				)
				if ok and ahead.strip() not in ("", "0"):
					logger.warning(
						"mc/green has %s unpushed commits, pushing before reset",
						ahead.strip(),
					)
					self.workspace = workspace
					await self.push_green_to_main()

		# Ensure branches exist in SOURCE repo first (worker clones need them)
		for branch in (gb.working_branch, gb.green_branch):
			ok, _ = await self._run_git_in(source_repo, "rev-parse", "--verify", branch)
			if not ok:
				logger.info("Creating branch %s in source repo from %s", branch, base)
				await self._run_git_in(source_repo, "branch", branch, base)
				self._trace("branch_created", branch=branch, base=base, repo="source")
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
				self._trace("branch_created", branch=branch, base=f"origin/{branch}", repo="workspace")
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
				env=self._workspace_env(),
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

	_OPTIMISTIC_RETRIES: int = 3

	async def merge_unit(
		self,
		worker_workspace: str,
		branch_name: str,
		acceptance_criteria: str = "",
	) -> UnitMergeResult:
		"""Merge a unit branch into mc/green.

		Optimistic concurrency flow:
		  1. LOCK: fetch + rebase + merge (fast ~2s). Record merge commit hash. UNLOCK.
		  2. UNLOCKED: run verification + acceptance criteria (slow 30-300s).
		  3. LOCK: if passed, check HEAD unchanged, sync+push. If HEAD advanced,
		     re-rebase and retry (up to 3 times). If failed, rollback via git revert.
		"""
		gb = self.config.green_branch

		# --- Phase 1: fast git ops under lock ---
		merge_result = await self._merge_git_ops(worker_workspace, branch_name)
		if merge_result is not None:
			# Early exit: fetch or merge failed
			return merge_result

		# Capture merge commit hash (set by _merge_git_ops on the branch)
		_, merge_hash_out = await self._run_git("rev-parse", gb.green_branch)
		merge_commit_hash = merge_hash_out.strip()

		_, diff_output = await self._run_git(
			"diff", "--name-only", f"{merge_commit_hash}~1", merge_commit_hash,
		)
		changed_files = [
			f.strip() for f in diff_output.splitlines() if f.strip()
		]

		# --- Phase 2: slow verification OUTSIDE lock ---
		verification_report = None
		if self.config.continuous.verify_before_merge:
			verification_report = await self._run_verification()
			if not verification_report.overall_passed:
				logger.warning("Pre-merge verification failed for %s, rolling back", branch_name)
				await self._rollback_merge(merge_commit_hash, branch_name)
				return UnitMergeResult(
					merged=False,
					verification_passed=False,
					failure_output=verification_report.raw_output[:2000],
					failure_stage="pre_merge_verification",
					verification_report=verification_report,
				)

		if acceptance_criteria:
			ac_passed, ac_output = await self._run_acceptance_criteria(acceptance_criteria)
			if not ac_passed:
				logger.warning("Acceptance criteria failed for %s, rolling back", branch_name)
				await self._rollback_merge(merge_commit_hash, branch_name)
				return UnitMergeResult(
					merged=False,
					verification_passed=False,
					failure_output=ac_output[:2000],
					failure_stage="acceptance_criteria",
					verification_report=verification_report,
				)

		# --- Phase 3: finalize under lock (sync + push) ---
		async with self._merge_lock:
			_, current_head_out = await self._run_git("rev-parse", gb.green_branch)
			current_head = current_head_out.strip()

			if current_head != merge_commit_hash:
				# HEAD advanced while we verified -- another merge landed.
				# Our merge commit is still in history, so sync+push is safe;
				# the verification already confirmed our code is green.
				logger.info(
					"HEAD advanced during verification for %s (%s -> %s), "
					"proceeding with sync (merge still in history)",
					branch_name, merge_commit_hash[:8], current_head[:8],
				)

			sync_ok = await self._sync_to_source()

			if gb.auto_push:
				self._merges_since_push += 1
				pushed = await self.maybe_push()

				if pushed and self.config.deploy.on_auto_push:
					await self.run_deploy()

		return UnitMergeResult(
			merged=True,
			rebase_ok=True,
			verification_passed=True,
			merge_commit_hash=merge_commit_hash,
			changed_files=changed_files,
			sync_ok=sync_ok,
			verification_report=verification_report,
		)

	async def merge_batch(
		self,
		units: list[tuple[str, str, str]],
	) -> list[UnitMergeResult]:
		"""Merge multiple units speculatively as a batch (bors-ng pattern).

		units: list of (worker_workspace, branch_name, acceptance_criteria)

		Flow:
		  1. Create speculative branch from mc/green
		  2. Rebase + merge ALL units onto speculative branch (under lock)
		  3. Run verification ONCE on the combined result (outside lock)
		  4. If passes: fast-forward mc/green to speculative branch, return all success
		  5. If fails: bisect to find offending unit(s)

		Falls back to individual merge_unit() for single-item lists.
		"""
		if len(units) <= 1:
			if not units:
				return []
			ws, branch, ac = units[0]
			return [await self.merge_unit(ws, branch, acceptance_criteria=ac)]

		gb = self.config.green_branch
		spec_branch = "mc/speculative-batch"

		# Phase 1: merge all units onto speculative branch under lock
		merge_hashes: list[str] = []
		results: list[UnitMergeResult] = []
		failed_indices: list[int] = []

		async with self._merge_lock:
			# Create speculative branch from current mc/green
			await self._run_git("branch", "-D", spec_branch)
			await self._run_git("checkout", gb.green_branch)
			await self._run_git("checkout", "-b", spec_branch)

			for i, (ws, branch, _ac) in enumerate(units):
				remote_name = f"worker-{branch}"
				await self._run_git("remote", "add", remote_name, ws)
				ok, _ = await self._run_git("fetch", remote_name, branch)
				if not ok:
					await self._run_git("remote", "remove", remote_name)
					results.append(UnitMergeResult(
						failure_output="Failed to fetch unit branch",
						failure_stage="fetch",
					))
					failed_indices.append(i)
					continue

				rebase_branch = f"mc/rebase-{branch}"
				await self._run_git("branch", "-D", rebase_branch)
				await self._run_git("branch", rebase_branch, f"{remote_name}/{branch}")
				rebase_ok, rebase_out = await self._run_git(
					"rebase", spec_branch, rebase_branch,
				)
				if not rebase_ok:
					await self._run_git("rebase", "--abort")
					await self._run_git("checkout", spec_branch)
					await self._run_git("branch", "-D", rebase_branch)
					await self._run_git("remote", "remove", remote_name)
					results.append(UnitMergeResult(
						rebase_ok=False,
						failure_output=f"Rebase conflict: {rebase_out[:500]}",
						failure_stage="merge_conflict",
					))
					failed_indices.append(i)
					continue

				await self._run_git("checkout", spec_branch)
				ok, output = await self._run_git(
					"merge", "--no-ff", rebase_branch,
					"-m", f"Merge {branch} (rebased) into {spec_branch}",
				)
				await self._run_git("branch", "-D", rebase_branch)
				await self._run_git("remote", "remove", remote_name)
				if not ok:
					await self._run_git("merge", "--abort")
					results.append(UnitMergeResult(
						failure_output=f"Merge failed after rebase: {output[:500]}",
						failure_stage="merge_conflict",
					))
					failed_indices.append(i)
					continue

				_, hash_out = await self._run_git("rev-parse", "HEAD")
				merge_hashes.append(hash_out.strip())
				results.append(UnitMergeResult())  # placeholder

			# Return to green
			await self._run_git("checkout", gb.green_branch)

		# If all units failed during merge phase, return early
		mergeable_indices = [i for i in range(len(units)) if i not in failed_indices]
		if not mergeable_indices:
			return results

		# Phase 2: run verification ONCE on the combined speculative branch (outside lock)
		if self.config.continuous.verify_before_merge:
			# Checkout spec branch for verification
			await self._run_git("checkout", spec_branch)
			verification_report = await self._run_verification()
			await self._run_git("checkout", gb.green_branch)

			if not verification_report.overall_passed:
				# Bisect to find the offending unit(s)
				logger.warning(
					"Batch verification failed, bisecting %d units",
					len(mergeable_indices),
				)
				bisect_results = await self._bisect_batch(
					[units[i] for i in mergeable_indices],
				)
				# Map bisection results back to original indices
				for j, idx in enumerate(mergeable_indices):
					results[idx] = bisect_results[j]
				# Clean up speculative branch
				await self._run_git("branch", "-D", spec_branch)
				return results
		else:
			verification_report = None

		# Phase 3: fast-forward mc/green to speculative branch (under lock)
		async with self._merge_lock:
			await self._run_git("checkout", gb.green_branch)
			ok, output = await self._run_git("merge", "--ff-only", spec_branch)
			if not ok:
				# Fall back to regular merge
				ok, output = await self._run_git(
					"merge", "--no-ff", spec_branch,
					"-m", f"Merge speculative batch into {gb.green_branch}",
				)
			if not ok:
				logger.error("Failed to merge speculative batch into mc/green: %s", output)
				await self._run_git("branch", "-D", spec_branch)
				# Fall back to individual merges
				return await self._fallback_individual_merge(units)

			sync_ok = await self._sync_to_source()

			if gb.auto_push:
				self._merges_since_push += len(mergeable_indices)
				pushed = await self.maybe_push()
				if pushed and self.config.deploy.on_auto_push:
					await self.run_deploy()

		# Clean up
		await self._run_git("branch", "-D", spec_branch)

		# Fill in success results for all mergeable units
		hash_idx = 0
		for i in mergeable_indices:
			results[i] = UnitMergeResult(
				merged=True,
				rebase_ok=True,
				verification_passed=True,
				merge_commit_hash=merge_hashes[hash_idx] if hash_idx < len(merge_hashes) else "",
				sync_ok=sync_ok,
				verification_report=verification_report,
			)
			hash_idx += 1

		return results

	async def _bisect_batch(
		self,
		units: list[tuple[str, str, str]],
	) -> list[UnitMergeResult]:
		"""Bisect a failing batch to find offending unit(s).

		Recursively splits the batch in half and tests each half until
		individual offenders are identified.
		"""
		if len(units) <= 1:
			# Base case: single unit -- test it individually
			if not units:
				return []
			ws, branch, ac = units[0]
			return [await self.merge_unit(ws, branch, acceptance_criteria=ac)]

		mid = len(units) // 2
		left = units[:mid]
		right = units[mid:]

		# Test left half as a batch (recursive)
		left_results = await self._test_half_batch(left)
		right_results = await self._test_half_batch(right)

		return left_results + right_results

	async def _test_half_batch(
		self,
		units: list[tuple[str, str, str]],
	) -> list[UnitMergeResult]:
		"""Test a half-batch by merging onto a temp branch and verifying."""
		if len(units) <= 1:
			if not units:
				return []
			ws, branch, ac = units[0]
			return [await self.merge_unit(ws, branch, acceptance_criteria=ac)]

		gb = self.config.green_branch
		temp_branch = f"mc/bisect-{id(units) % 10000}"

		# Merge all units onto temp branch
		async with self._merge_lock:
			await self._run_git("branch", "-D", temp_branch)
			await self._run_git("checkout", gb.green_branch)
			await self._run_git("checkout", "-b", temp_branch)

			all_ok = True
			for ws, branch, _ac in units:
				remote_name = f"worker-{branch}"
				await self._run_git("remote", "add", remote_name, ws)
				ok, _ = await self._run_git("fetch", remote_name, branch)
				if not ok:
					await self._run_git("remote", "remove", remote_name)
					all_ok = False
					break

				rebase_branch = f"mc/rebase-{branch}"
				await self._run_git("branch", "-D", rebase_branch)
				await self._run_git("branch", rebase_branch, f"{remote_name}/{branch}")
				rebase_ok, _ = await self._run_git("rebase", temp_branch, rebase_branch)
				if not rebase_ok:
					await self._run_git("rebase", "--abort")
					await self._run_git("checkout", temp_branch)
					await self._run_git("branch", "-D", rebase_branch)
					await self._run_git("remote", "remove", remote_name)
					all_ok = False
					break

				await self._run_git("checkout", temp_branch)
				ok, _ = await self._run_git(
					"merge", "--no-ff", rebase_branch,
					"-m", f"Merge {branch} (rebased) into {temp_branch}",
				)
				await self._run_git("branch", "-D", rebase_branch)
				await self._run_git("remote", "remove", remote_name)
				if not ok:
					await self._run_git("merge", "--abort")
					all_ok = False
					break

			await self._run_git("checkout", gb.green_branch)

		if not all_ok:
			await self._run_git("branch", "-D", temp_branch)
			# Merge conflict in this half -- bisect further
			return await self._bisect_batch(units)

		# Verify the half-batch
		await self._run_git("checkout", temp_branch)
		report = await self._run_verification()
		await self._run_git("checkout", gb.green_branch)
		await self._run_git("branch", "-D", temp_branch)

		if report.overall_passed:
			# This half is clean -- merge individually (they all pass)
			individual_results = []
			for ws, branch, ac in units:
				individual_results.append(
					await self.merge_unit(ws, branch, acceptance_criteria=ac)
				)
			return individual_results
		else:
			# This half has the offender -- bisect further
			return await self._bisect_batch(units)

	async def _fallback_individual_merge(
		self,
		units: list[tuple[str, str, str]],
	) -> list[UnitMergeResult]:
		"""Fall back to merging each unit individually."""
		results = []
		for ws, branch, ac in units:
			results.append(
				await self.merge_unit(ws, branch, acceptance_criteria=ac)
			)
		return results

	async def _merge_git_ops(
		self,
		worker_workspace: str,
		branch_name: str,
	) -> UnitMergeResult | None:
		"""Phase 1: fast git operations under _merge_lock.

		Returns None on success (merge committed), or a UnitMergeResult on failure.
		"""
		gb = self.config.green_branch
		remote_name = f"worker-{branch_name}"

		async with self._merge_lock:
			await self._run_git("remote", "add", remote_name, worker_workspace)
			ok, _ = await self._run_git("fetch", remote_name, branch_name)
			if not ok:
				await self._run_git("remote", "remove", remote_name)
				return UnitMergeResult(failure_output="Failed to fetch unit branch", failure_stage="fetch")

			try:
				await self._run_git("checkout", gb.green_branch)
				await self._run_git("reset", "--hard", "HEAD")
				await self._run_git("clean", "-fd", "-e", ".venv")

				rebase_branch = f"mc/rebase-{branch_name}"
				await self._run_git("branch", "-D", rebase_branch)
				await self._run_git("branch", rebase_branch, f"{remote_name}/{branch_name}")
				self._trace("rebase_attempted", branch=branch_name, onto=gb.green_branch)
				rebase_ok, rebase_out = await self._run_git(
					"rebase", gb.green_branch, rebase_branch,
				)
				self._trace(
					"rebase_result", branch=branch_name,
					success=rebase_ok, output=rebase_out[:500],
				)
				if not rebase_ok:
					await self._run_git("rebase", "--abort")
					await self._run_git("checkout", gb.green_branch)
					await self._run_git("branch", "-D", rebase_branch)
					logger.warning("Rebase failed for %s: %s", branch_name, rebase_out)
					return UnitMergeResult(
						rebase_ok=False,
						failure_output=f"Rebase conflict: {rebase_out[:500]}",
						failure_stage="merge_conflict",
					)

				await self._run_git("checkout", gb.green_branch)
				ok, output = await self._run_git(
					"merge", "--no-ff", rebase_branch,
					"-m", f"Merge {branch_name} (rebased) into {gb.green_branch}",
				)
				await self._run_git("branch", "-D", rebase_branch)
				if not ok:
					await self._run_git("merge", "--abort")
					logger.warning("Merge failed after rebase for %s: %s", branch_name, output)
					return UnitMergeResult(
						failure_output=f"Merge failed after rebase: {output[:500]}",
						failure_stage="merge_conflict",
					)

				logger.info("Merged %s into %s", branch_name, gb.green_branch)
			finally:
				await self._run_git("checkout", gb.green_branch)
				await self._run_git("remote", "remove", remote_name)

		# Lock released -- merge commit is on mc/green
		return None

	async def _rollback_merge(self, merge_commit_hash: str, branch_name: str) -> None:
		"""Rollback a merge commit using git revert (safe when HEAD has advanced)."""
		async with self._merge_lock:
			gb = self.config.green_branch
			await self._run_git("checkout", gb.green_branch)
			ok, output = await self._run_git(
				"revert", "--no-edit", "-m", "1", merge_commit_hash,
			)
			if not ok:
				logger.error(
					"Failed to revert merge %s for %s: %s",
					merge_commit_hash[:8], branch_name, output,
				)
			else:
				logger.info(
					"Reverted merge %s for %s (verification failed)",
					merge_commit_hash[:8], branch_name,
				)

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
			# Force-update (+) because workspace mc/green always has the latest merges
			ok, output = await self._run_git_in(
				source_repo, "fetch", self.workspace, f"+{branch}:{branch}",
			)
			if not ok:
				logger.warning("Failed to sync %s to source repo: %s", branch, output)
				all_ok = False
			else:
				self._trace("git_push", branch=branch, target="source")
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
		# Force-update (+) because mc/green is reset at each mission start
		ok, output = await self._run_git_in(
			source_repo, "fetch", self.workspace,
			f"+{gb.green_branch}:{green_ref}",
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

			# Try ff-only first; fall back to regular merge if main has diverged
			ok, output = await self._run_git_in(source_repo, "merge", "--ff-only", green_ref)
			if not ok:
				logger.warning("ff-merge failed, trying regular merge: %s", output)
				ok, output = await self._run_git_in(
					source_repo, "merge", "--no-edit", green_ref,
				)
				if not ok:
					logger.error("Failed to merge mc/green into %s: %s", push_branch, output)
					return False

			ok, output = await self._run_git_in(source_repo, "push", "origin", push_branch)
			if not ok:
				logger.error("Failed to push %s: %s", push_branch, output)
				return False

			self._trace("git_push", branch=push_branch, target="origin")
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

	def _workspace_env(self) -> dict[str, str]:
		"""Build env with .venv/bin on PATH and PYTHONPATH pointing to workspace src/.

		PYTHONPATH ensures imports resolve to the merged code on mc/green rather
		than wherever the editable .pth file happens to point (which workers
		can corrupt by running `pip install -e .` in their clones).
		"""
		import os
		env = os.environ.copy()
		venv_bin = str(Path(self.workspace) / ".venv" / "bin")
		workspace_src = str(Path(self.workspace) / "src")
		env["PATH"] = venv_bin + ":" + env.get("PATH", "")
		env["VIRTUAL_ENV"] = str(Path(self.workspace) / ".venv")
		env["PYTHONPATH"] = workspace_src
		return env

	async def _run_acceptance_criteria(self, criteria: str, timeout: int = 120) -> tuple[bool, str]:
		"""Run acceptance criteria shell command(s) in the workspace.

		Returns (passed, output) tuple. Criteria are expected to be shell
		commands that exit 0 on success.
		"""
		try:
			proc = await asyncio.create_subprocess_shell(
				criteria,
				cwd=self.workspace,
				env=self._workspace_env(),
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
			try:
				stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
			except asyncio.TimeoutError:
				logger.warning("Acceptance criteria timed out after %ds: %s", timeout, criteria[:100])
				try:
					proc.kill()
					await proc.wait()
				except ProcessLookupError:
					pass
				return (False, f"Acceptance criteria timed out after {timeout}s")
			output = stdout.decode() if stdout else ""
			return (proc.returncode == 0, output)
		except (FileNotFoundError, OSError) as exc:
			return (False, f"Acceptance criteria execution error: {exc}")

	async def _run_verification(self) -> VerificationReport:
		"""Run verification nodes and return a structured report.

		Uses self._run_command for single-command fallback (backward compat
		with tests that mock _run_command). Delegates to run_verification_nodes
		only when explicit nodes are configured.
		"""
		from mission_control.state import _build_result_from_single_command

		nodes = self.config.target.verification.nodes
		if nodes:
			return await run_verification_nodes(self.config, self.workspace)
		# Single-command fallback using self._run_command
		ok, output = await self._run_command(self.config.target.verification.command)
		return _build_result_from_single_command(output, 0 if ok else 1)

	async def _run_command(self, cmd: str | list[str]) -> tuple[bool, str]:
		"""Run a command in self.workspace using shell for string commands."""
		timeout = self.config.target.verification.timeout
		env = self._workspace_env()
		if isinstance(cmd, str):
			proc = await asyncio.create_subprocess_shell(
				cmd,
				cwd=self.workspace,
				env=env,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.STDOUT,
			)
		else:
			proc = await asyncio.create_subprocess_exec(
				*cmd,
				cwd=self.workspace,
				env=env,
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
