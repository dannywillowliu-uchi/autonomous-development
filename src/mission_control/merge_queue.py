"""Sequential merge queue -- fetch, rebase, verify, merge or reject."""

from __future__ import annotations

import asyncio
import logging

from mission_control.config import MissionConfig
from mission_control.db import Database
from mission_control.models import MergeRequest, Session, _now_iso
from mission_control.reviewer import review_session
from mission_control.state import snapshot_project_health

logger = logging.getLogger(__name__)

POLL_INTERVAL = 5  # seconds between queue checks


class MergeQueue:
	def __init__(self, config: MissionConfig, db: Database, workspace: str) -> None:
		self.config = config
		self.db = db
		self.workspace = workspace  # dedicated clone for merge operations
		self.running = True

	async def run(self) -> None:
		"""Main loop: process merge requests one at a time."""
		while self.running:
			mr = await self.db.locked_call("get_next_merge_request")
			if mr is None:
				await asyncio.sleep(POLL_INTERVAL)
				continue
			await self._process_merge_request(mr)

	async def _process_merge_request(self, mr: MergeRequest) -> None:
		"""Process a single merge request: fetch, rebase, verify, merge/reject."""
		mr.status = "verifying"
		await self.db.locked_call("update_merge_request", mr)

		# 1. Fetch the branch from worker clone
		fetch_ok = await self._fetch_worker_branch(mr)
		if not fetch_ok:
			mr.status = "conflict"
			mr.rejection_reason = "Failed to fetch branch from worker"
			await self.db.locked_call("update_merge_request", mr)
			await self._release_unit_for_retry(mr)
			return

		# 2. Rebase onto base branch
		rebase_ok = await self._rebase_onto_base(mr)
		if not rebase_ok:
			mr.status = "conflict"
			mr.rejection_reason = "Rebase conflict"
			mr.rebase_attempts += 1
			await self.db.locked_call("update_merge_request", mr)
			await self._release_unit_for_retry(mr)
			return

		# 3. Take before snapshot on base branch (workspace is on feature branch after rebase)
		await self._run_git("checkout", self.config.target.branch)
		before = await snapshot_project_health(self.config, cwd=self.workspace)

		# 4. Checkout the rebased feature branch and take after snapshot
		await self._run_git("checkout", mr.branch_name)
		after = await snapshot_project_health(self.config, cwd=self.workspace)

		# 5. Review using existing reviewer
		dummy_session = Session(id=mr.work_unit_id, branch_name=mr.branch_name)
		verdict = review_session(before, after, dummy_session, auto_merge=True)

		# 6. Act on verdict
		if verdict.verdict == "helped" or verdict.verdict == "neutral":
			# Merge into base
			merge_ok = await self._merge_into_base(mr)
			if merge_ok:
				mr.status = "merged"
				mr.merged_at = _now_iso()
				mr.verified_at = _now_iso()
				# Update the work unit
				unit = await self.db.locked_call("get_work_unit", mr.work_unit_id)
				if unit:
					unit.status = "completed"
					unit.finished_at = _now_iso()
					await self.db.locked_call("update_work_unit", unit)
			else:
				mr.status = "conflict"
				mr.rejection_reason = "Merge failed"
				await self._release_unit_for_retry(mr)
		else:
			mr.status = "rejected"
			mr.rejection_reason = f"Verification failed: {verdict.summary}"
			mr.verified_at = _now_iso()
			await self._release_unit_for_retry(mr)
			# Reset workspace to base branch
			await self._run_git("checkout", self.config.target.branch)
			await self._run_git("reset", "--hard", f"origin/{self.config.target.branch}")

		await self.db.locked_call("update_merge_request", mr)

	async def _fetch_worker_branch(self, mr: MergeRequest) -> bool:
		"""Fetch branch from worker clone via filesystem path."""
		unit = await self.db.locked_call("get_work_unit", mr.work_unit_id)
		if not unit or not unit.worker_id:
			return False
		worker = await self.db.locked_call("get_worker", unit.worker_id)
		if not worker:
			return False

		remote_name = f"worker-{unit.worker_id[:8]}"
		# Add remote (ignore error if already exists)
		await self._run_git("remote", "add", remote_name, worker.workspace_path)
		# Fetch the branch
		fetch_ok = await self._run_git("fetch", remote_name, mr.branch_name)
		if fetch_ok:
			# Create a local branch from FETCH_HEAD before removing the remote,
			# otherwise the fetched ref is lost when the remote is deleted
			await self._run_git("branch", "-f", mr.branch_name, "FETCH_HEAD")
		# Clean up remote to prevent accumulation
		await self._run_git("remote", "remove", remote_name)
		return fetch_ok

	async def _rebase_onto_base(self, mr: MergeRequest) -> bool:
		"""Rebase the branch onto the current local base branch.

		Uses the local base branch ref (not origin/) because prior MRs may have
		been merged locally but not yet pushed to origin.
		"""
		await self._run_git("checkout", mr.branch_name)
		ok = await self._run_git("rebase", self.config.target.branch)
		if not ok:
			await self._run_git("rebase", "--abort")
			# Return to base branch so workspace is clean for the next MR
			await self._run_git("checkout", self.config.target.branch)
		return ok

	async def _merge_into_base(self, mr: MergeRequest) -> bool:
		"""Fast-forward merge the branch into base."""
		await self._run_git("checkout", self.config.target.branch)
		return await self._run_git("merge", "--no-ff", mr.branch_name, "-m", f"Merge {mr.branch_name}")

	async def _release_unit_for_retry(self, mr: MergeRequest) -> None:
		"""Release the work unit back to pending for retry if under max_attempts."""
		unit = await self.db.locked_call("get_work_unit", mr.work_unit_id)
		if unit and unit.attempt < unit.max_attempts:
			unit.status = "pending"
			unit.worker_id = None
			unit.claimed_at = None
			unit.heartbeat_at = None
			unit.attempt += 1
			await self.db.locked_call("update_work_unit", unit)
		elif unit:
			unit.status = "failed"
			unit.finished_at = _now_iso()
			await self.db.locked_call("update_work_unit", unit)

	async def _run_git(self, *args: str) -> bool:
		"""Run a git command in the merge workspace."""
		proc = await asyncio.create_subprocess_exec(
			"git", *args,
			cwd=self.workspace,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		await proc.communicate()
		return proc.returncode == 0

	def stop(self) -> None:
		self.running = False
