"""Multi-contributor coordination protocol for autonomous development.

Git-native protocol for multiple contributors to pool agents on the same
project. The git repo is the coordination layer -- no central server needed.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClaimInfo:
	user: str
	claimed_at: str
	status: str  # 'claimed', 'completed', 'abandoned'


@dataclass
class ContributorInfo:
	username: str
	joined_at: str
	agent_count: int = 0
	proposals_completed: int = 0


class ContributorProtocol:
	"""Git-native multi-contributor coordination."""

	CLAIMS_FILE = ".autodev-claims.json"
	REGISTRY_FILE = ".autodev-contributor-registry.json"
	LEARNINGS_FILE = ".autodev-swarm-learnings.md"
	EXPERIMENT_LOG = ".autodev-experiments.tsv"

	def __init__(self, repo_path: Path, username: str):
		self._repo = repo_path
		self._username = username

	async def _run_git(self, *args: str) -> tuple[int, str, str]:
		"""Run a git command and return (returncode, stdout, stderr)."""
		proc = await asyncio.create_subprocess_exec(
			"git", *args,
			cwd=str(self._repo),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, stderr = await proc.communicate()
		return proc.returncode, stdout.decode(), stderr.decode()

	def _read_json(self, filename: str) -> dict:
		"""Read a JSON file from the repo, returning empty dict if missing."""
		path = self._repo / filename
		if not path.exists():
			return {}
		try:
			return json.loads(path.read_text())
		except (json.JSONDecodeError, OSError) as exc:
			logger.warning("Failed to read %s: %s", filename, exc)
			return {}

	def _write_json(self, filename: str, data: dict) -> None:
		"""Write a JSON file to the repo."""
		path = self._repo / filename
		path.write_text(json.dumps(data, indent="\t") + "\n")

	async def register(self) -> ContributorInfo:
		"""Register this contributor in the shared registry."""
		await self._run_git("pull", "--rebase")

		registry = self._read_json(self.REGISTRY_FILE)
		now = datetime.now(timezone.utc).isoformat()

		if self._username in registry:
			info = ContributorInfo(
				username=self._username,
				joined_at=registry[self._username]["joined_at"],
				agent_count=registry[self._username].get("agent_count", 0),
				proposals_completed=registry[self._username].get("proposals_completed", 0),
			)
			logger.info("Contributor %s already registered", self._username)
			return info

		info = ContributorInfo(username=self._username, joined_at=now)
		registry[self._username] = asdict(info)
		self._write_json(self.REGISTRY_FILE, registry)

		await self._run_git("add", self.REGISTRY_FILE)
		await self._run_git("commit", "-m", f"contrib: register {self._username}")
		rc, _, stderr = await self._run_git("push")
		if rc != 0:
			logger.warning("Push failed during register: %s", stderr)

		return info

	async def list_proposals(self) -> list[dict]:
		"""List available proposals (unclaimed or abandoned)."""
		await self._run_git("pull", "--rebase")

		claims = self._read_json(self.CLAIMS_FILE)
		available = []
		for proposal_id, claim_data in claims.items():
			status = claim_data.get("status", "")
			if status in ("abandoned", ""):
				available.append({"proposal_id": proposal_id, **claim_data})
		return available

	async def claim_proposal(self, proposal_id: str) -> bool:
		"""Claim a proposal using git-based optimistic locking."""
		rc, _, _ = await self._run_git("pull", "--rebase")
		if rc != 0:
			logger.warning("Git pull failed before claim attempt")

		claims = self._read_json(self.CLAIMS_FILE)

		if proposal_id in claims:
			existing = claims[proposal_id]
			if existing.get("status") == "claimed" and existing.get("user") != self._username:
				logger.info("Proposal %s already claimed by %s", proposal_id, existing["user"])
				return False

		now = datetime.now(timezone.utc).isoformat()
		claims[proposal_id] = asdict(ClaimInfo(
			user=self._username,
			claimed_at=now,
			status="claimed",
		))
		self._write_json(self.CLAIMS_FILE, claims)

		await self._run_git("add", self.CLAIMS_FILE)
		await self._run_git("commit", "-m", f"contrib: claim {proposal_id} by {self._username}")

		rc, _, stderr = await self._run_git("push")
		if rc != 0:
			logger.info("Push failed (conflict), claim lost: %s", stderr)
			await self._run_git("reset", "--hard", "HEAD~1")
			return False

		return True

	async def publish_result(self, proposal_id: str, result: dict) -> None:
		"""Publish experiment result to shared log."""
		await self._run_git("pull", "--rebase")

		log_path = self._repo / self.EXPERIMENT_LOG
		if not log_path.exists():
			header = "commit\ttests_before\ttests_after\toutcome\tproposal_title\tduration_s\tcost_usd\ttimestamp\n"
			log_path.write_text(header)

		row = "\t".join(str(result.get(col, "")) for col in [
			"commit", "tests_before", "tests_after", "outcome",
			"proposal_title", "duration_s", "cost_usd", "timestamp",
		])
		with open(log_path, "a") as f:
			f.write(row + "\n")

		claims = self._read_json(self.CLAIMS_FILE)
		if proposal_id in claims:
			claims[proposal_id]["status"] = "completed"
			self._write_json(self.CLAIMS_FILE, claims)

		await self._run_git("add", self.EXPERIMENT_LOG, self.CLAIMS_FILE)
		await self._run_git("commit", "-m", f"contrib: publish result for {proposal_id}")
		rc, _, stderr = await self._run_git("push")
		if rc != 0:
			logger.warning("Push failed after publish: %s", stderr)

	async def sync_learnings(self) -> str:
		"""Pull latest shared learnings."""
		await self._run_git("pull", "--rebase")

		learnings_path = self._repo / self.LEARNINGS_FILE
		if not learnings_path.exists():
			return ""

		return learnings_path.read_text()
