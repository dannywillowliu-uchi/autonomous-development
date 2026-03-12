"""Git ratchet for safe self-modification checkpoints."""

import asyncio
import datetime
from pathlib import Path


class GitRatchet:
	"""Commit/reset ratchet for safe self-modification."""

	def __init__(self, repo_path: Path):
		self._repo = repo_path

	async def checkpoint(self, proposal_id: str) -> str:
		"""Tag current HEAD as pre-modification checkpoint. Returns tag name."""
		tag = f"autodev/pre-{proposal_id}"
		proc = await asyncio.create_subprocess_exec(
			"git", "tag", tag, "HEAD",
			cwd=str(self._repo),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		_, stderr = await proc.communicate()
		if proc.returncode != 0:
			raise RuntimeError(f"git tag failed: {stderr.decode().strip()}")
		return tag

	async def verify_and_decide(self, tag: str, verification_cmd: str) -> bool:
		"""Run verification. If passes, keep. If fails, rollback to tag."""
		proc = await asyncio.create_subprocess_shell(
			verification_cmd,
			cwd=str(self._repo),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		await proc.communicate()
		if proc.returncode == 0:
			return True
		await self.rollback(tag)
		return False

	async def rollback(self, tag: str) -> None:
		"""Hard reset to the checkpoint tag."""
		proc = await asyncio.create_subprocess_exec(
			"git", "reset", "--hard", tag,
			cwd=str(self._repo),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		_, stderr = await proc.communicate()
		if proc.returncode != 0:
			raise RuntimeError(f"git reset failed: {stderr.decode().strip()}")

	def append_experiment_log(
		self, commit: str, tests_before: int, tests_after: int,
		outcome: str, title: str, duration: float, cost: float,
	) -> None:
		"""Append a row to .autodev-experiments.tsv"""
		tsv_path = self._repo / ".autodev-experiments.tsv"
		header = "commit\ttests_before\ttests_after\toutcome\tproposal_title\tduration_s\tcost_usd\ttimestamp\n"
		if not tsv_path.exists():
			tsv_path.write_text(header)
		timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
		row = f"{commit}\t{tests_before}\t{tests_after}\t{outcome}\t{title}\t{duration:.1f}\t{cost:.4f}\t{timestamp}\n"
		with open(tsv_path, "a") as f:
			f.write(row)
