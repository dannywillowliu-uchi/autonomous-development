"""Mission launcher -- spawn and manage mc mission subprocesses."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from mission_control.db import Database
from mission_control.models import (
	Campaign,
	CampaignObjective,
	Mission,
	Signal,
	StrategicContext,
	_new_id,
	_now_iso,
)
from mission_control.registry import ProjectInfo, ProjectRegistry

logger = logging.getLogger(__name__)


class MissionLauncher:
	"""Launch and control missions as background subprocesses."""

	def __init__(self, registry: ProjectRegistry) -> None:
		self.registry = registry

	def launch(
		self,
		project_name: str,
		config_overrides: dict[str, str] | None = None,
	) -> int:
		"""Launch a mission for the given project. Returns the PID.

		Spawns `mc mission --config <path>` as a detached subprocess.
		Stores the PID in the registry.
		"""
		project = self.registry.get_project(project_name)
		if project is None:
			raise ValueError(f"Project '{project_name}' is not registered")

		if self.is_running(project_name):
			raise RuntimeError(f"Project '{project_name}' already has a running mission")

		config_path = project.config_path
		if not Path(config_path).exists():
			raise FileNotFoundError(f"Config not found: {config_path}")

		cmd = [sys.executable, "-m", "mission_control.cli", "mission", "--config", config_path]

		# Apply config overrides as CLI args
		if config_overrides:
			if "mode" in config_overrides:
				cmd.extend(["--mode", str(config_overrides["mode"])])
			if "max_rounds" in config_overrides:
				cmd.extend(["--max-rounds", str(config_overrides["max_rounds"])])
			if "workers" in config_overrides:
				cmd.extend(["--workers", str(config_overrides["workers"])])

		# Spawn detached process
		proc = subprocess.Popen(
			cmd,
			start_new_session=True,
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
		)

		self.registry.update_pid(project_name, proc.pid)
		logger.info("Launched mission for %s (PID %d)", project_name, proc.pid)
		return proc.pid

	def is_running(self, project_name: str) -> bool:
		"""Check if the project has a live mission process."""
		project = self.registry.get_project(project_name)
		if project is None or project.active_pid is None:
			return False

		if _pid_alive(project.active_pid):
			return True

		# Stale PID -- clean up
		self.registry.update_pid(project_name, None)
		return False

	def stop(self, project_name: str) -> bool:
		"""Send a stop signal to a running mission via the DB.

		Returns True if the signal was inserted.
		"""
		project = self.registry.get_project(project_name)
		if project is None:
			raise ValueError(f"Project '{project_name}' is not registered")

		db_path = Path(project.db_path)
		if not db_path.exists():
			return False

		db = Database(db_path)
		try:
			mission = db.get_latest_mission()
			if mission is None or mission.status != "running":
				return False

			signal = Signal(
				id=_new_id(),
				mission_id=mission.id,
				signal_type="stop",
				created_at=_now_iso(),
			)
			db.insert_signal(signal)
			logger.info("Sent stop signal for project %s mission %s", project_name, mission.id)
			return True
		finally:
			db.close()

	def retry_unit(self, project_name: str, unit_id: str) -> bool:
		"""Send a retry signal for a specific work unit."""
		project = self.registry.get_project(project_name)
		if project is None:
			raise ValueError(f"Project '{project_name}' is not registered")

		db_path = Path(project.db_path)
		if not db_path.exists():
			return False

		db = Database(db_path)
		try:
			mission = db.get_latest_mission()
			if mission is None:
				return False

			signal = Signal(
				id=_new_id(),
				mission_id=mission.id,
				signal_type="retry_unit",
				payload=unit_id,
				created_at=_now_iso(),
			)
			db.insert_signal(signal)
			return True
		finally:
			db.close()

	def adjust(self, project_name: str, params: dict[str, int | float | str]) -> bool:
		"""Send an adjust signal to modify runtime parameters."""
		import json

		project = self.registry.get_project(project_name)
		if project is None:
			raise ValueError(f"Project '{project_name}' is not registered")

		db_path = Path(project.db_path)
		if not db_path.exists():
			return False

		db = Database(db_path)
		try:
			mission = db.get_latest_mission()
			if mission is None or mission.status != "running":
				return False

			signal = Signal(
				id=_new_id(),
				mission_id=mission.id,
				signal_type="adjust",
				payload=json.dumps(params),
				created_at=_now_iso(),
			)
			db.insert_signal(signal)
			return True
		finally:
			db.close()

	def run_campaign(self, project_name: str, campaign: Campaign) -> Campaign:
		"""Execute a campaign's objectives in dependency order.

		Launches missions sequentially respecting dependency DAG. On failure,
		marks all transitively-dependent objectives as 'skipped'. Shares
		strategic context between missions via the project DB.
		"""
		project = self.registry.get_project(project_name)
		if project is None:
			raise ValueError(f"Project '{project_name}' is not registered")

		campaign.status = "running"
		campaign.started_at = _now_iso()

		n = len(campaign.objectives)
		completed: set[int] = set()

		# Build reverse dependency map for skip propagation
		dependents: dict[int, list[int]] = {i: [] for i in range(n)}
		for i, obj in enumerate(campaign.objectives):
			for dep_idx in obj.depends_on_indices:
				if 0 <= dep_idx < n:
					dependents[dep_idx].append(i)

		def _skip_transitively(idx: int) -> None:
			"""Mark idx and all transitive dependents as skipped."""
			stack = [idx]
			while stack:
				cur = stack.pop()
				if campaign.objectives[cur].status in ("skipped", "completed", "failed"):
					continue
				campaign.objectives[cur].status = "skipped"
				stack.extend(dependents.get(cur, []))

		# Topological execution: keep running until nothing is runnable
		while True:
			runnable: list[int] = []
			for i, obj in enumerate(campaign.objectives):
				if obj.status != "pending":
					continue
				# All deps must be completed
				if all(d in completed for d in obj.depends_on_indices):
					runnable.append(i)

			if not runnable:
				break

			# Execute runnable objectives sequentially
			for idx in runnable:
				obj = campaign.objectives[idx]
				obj.status = "running"

				try:
					pid = self.launch(project_name, config_overrides=obj.config_overrides or None)

					# Wait for process to finish
					self._wait_for_pid(pid)

					# Clean up PID so next launch isn't blocked
					self.registry.update_pid(project_name, None)

					# Check mission outcome from DB
					mission = self._get_latest_mission(project)
					succeeded = mission is not None and mission.status == "completed"
					if succeeded:
						obj.status = "completed"
						obj.mission_id = mission.id
						completed.add(idx)
						self._share_strategic_context(project, obj, idx)
					else:
						obj.status = "failed"
						for dep_idx in dependents.get(idx, []):
							_skip_transitively(dep_idx)
				except Exception:
					logger.exception("Campaign objective %d failed to launch", idx)
					obj.status = "failed"
					self.registry.update_pid(project_name, None)
					for dep_idx in dependents.get(idx, []):
						_skip_transitively(dep_idx)

		# Determine overall campaign status
		statuses = {obj.status for obj in campaign.objectives}
		if all(s == "completed" for s in statuses):
			campaign.status = "completed"
		elif "failed" in statuses:
			campaign.status = "failed"
		else:
			campaign.status = "completed"  # all completed or skipped

		campaign.finished_at = _now_iso()
		return campaign

	def _wait_for_pid(self, pid: int, poll_interval: float = 0.5) -> None:
		"""Poll until a PID is no longer alive."""
		while _pid_alive(pid):
			time.sleep(poll_interval)

	def _get_latest_mission(self, project: ProjectInfo) -> Mission | None:
		"""Get the latest mission from the project DB."""
		db_path = Path(project.db_path)
		if not db_path.exists():
			return None

		db = Database(db_path)
		try:
			return db.get_latest_mission()
		finally:
			db.close()

	def _share_strategic_context(
		self,
		project: ProjectInfo,
		obj: CampaignObjective,
		idx: int,
	) -> None:
		"""Write strategic context for a completed objective so subsequent missions can read it."""
		db_path = Path(project.db_path)
		if not db_path.exists():
			return

		db = Database(db_path)
		try:
			ctx = StrategicContext(
				mission_id=obj.mission_id or "",
				what_attempted=f"Campaign objective {idx}: {obj.objective}",
				what_worked=obj.objective,
				what_failed="",
				recommended_next="Continue with dependent campaign objectives",
			)
			db.insert_strategic_context(ctx)
		finally:
			db.close()

	@staticmethod
	def parse_campaign_file(path: Path) -> Campaign:
		"""Parse a TOML campaign definition file into a Campaign.

		Expected format:
			[campaign]
			name = "my-campaign"

			[[campaign.objectives]]
			objective = "refactor auth module"
			depends_on = []

			[[campaign.objectives]]
			objective = "add OAuth support"
			depends_on = [0]
			[campaign.objectives.config_overrides]
			mode = "continuous"
		"""
		with open(path, "rb") as f:
			data = tomllib.load(f)

		campaign_data = data.get("campaign", {})
		name = campaign_data.get("name", path.stem)
		raw_objectives = campaign_data.get("objectives", [])

		objectives: list[CampaignObjective] = []
		for raw in raw_objectives:
			objectives.append(
				CampaignObjective(
					objective=raw.get("objective", ""),
					depends_on_indices=raw.get("depends_on", []),
					config_overrides={
						str(k): str(v) for k, v in raw.get("config_overrides", {}).items()
					},
				)
			)

		return Campaign(name=name, objectives=objectives)


def _pid_alive(pid: int) -> bool:
	"""Check if a PID is alive."""
	try:
		os.kill(pid, 0)
		return True
	except (OSError, ProcessLookupError):
		return False
