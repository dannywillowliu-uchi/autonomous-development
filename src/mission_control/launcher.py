"""Mission launcher -- spawn and manage mc mission subprocesses."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from mission_control.db import Database
from mission_control.models import Signal, _new_id, _now_iso
from mission_control.registry import ProjectRegistry

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


def _pid_alive(pid: int) -> bool:
	"""Check if a PID is alive."""
	try:
		os.kill(pid, 0)
		return True
	except (OSError, ProcessLookupError):
		return False
