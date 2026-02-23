"""Central project registry at ~/.mission-control/registry.db."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from mission_control.models import _new_id, _now_iso

DEFAULT_REGISTRY_DIR = Path.home() / ".mission-control"
DEFAULT_REGISTRY_DB = DEFAULT_REGISTRY_DIR / "registry.db"

REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
	id TEXT PRIMARY KEY,
	name TEXT NOT NULL UNIQUE,
	config_path TEXT NOT NULL,
	db_path TEXT NOT NULL DEFAULT '',
	description TEXT NOT NULL DEFAULT '',
	registered_at TEXT NOT NULL,
	active_pid INTEGER
);
"""


@dataclass
class ProjectInfo:
	"""A registered project in the central registry."""

	id: str = field(default_factory=_new_id)
	name: str = ""
	config_path: str = ""
	db_path: str = ""
	description: str = ""
	registered_at: str = field(default_factory=_now_iso)
	active_pid: int | None = None


@dataclass
class ProjectStatus:
	"""Project info enriched with latest mission status."""

	project: ProjectInfo
	mission_status: str = "idle"  # idle/running/completed/failed/stalled/stopped
	mission_objective: str = ""
	mission_score: float = 0.0
	mission_rounds: int = 0
	mission_cost: float = 0.0


class ProjectRegistry:
	"""Central registry mapping project names to config/DB paths.

	Stores data in ~/.mission-control/registry.db.
	"""

	def __init__(
		self,
		db_path: str | Path | None = None,
		allowed_bases: list[Path] | None = None,
	) -> None:
		if db_path is None:
			db_path = DEFAULT_REGISTRY_DB
		self._db_path = Path(db_path)
		self._db_path.parent.mkdir(parents=True, exist_ok=True)
		self._allowed_bases = allowed_bases if allowed_bases is not None else [Path.home()]
		self.conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
		self.conn.row_factory = sqlite3.Row
		self.conn.execute("PRAGMA journal_mode=WAL")
		self.conn.execute("PRAGMA foreign_keys=ON")
		self.conn.executescript(REGISTRY_SCHEMA)

	def close(self) -> None:
		self.conn.close()

	def register(
		self,
		name: str,
		config_path: str,
		db_path: str = "",
		description: str = "",
	) -> ProjectInfo:
		"""Register a project. Derives db_path from config_path if not provided."""
		from mission_control.path_security import validate_config_path

		validate_config_path(config_path, self._allowed_bases)
		config = Path(config_path).resolve()
		if not db_path:
			db_path = str(config.parent / "mission-control.db")

		project = ProjectInfo(
			name=name,
			config_path=str(config),
			db_path=db_path,
			description=description,
		)
		try:
			self.conn.execute(
				"""INSERT INTO projects
				(id, name, config_path, db_path, description, registered_at, active_pid)
				VALUES (?, ?, ?, ?, ?, ?, ?)""",
				(
					project.id, project.name, project.config_path,
					project.db_path, project.description,
					project.registered_at, project.active_pid,
				),
			)
			self.conn.commit()
		except sqlite3.IntegrityError as exc:
			if "UNIQUE" in str(exc):
				raise ValueError(f"Project '{name}' is already registered") from exc
			raise
		return project

	def unregister(self, name: str) -> bool:
		"""Remove a project from the registry. Returns True if found."""
		cursor = self.conn.execute(
			"DELETE FROM projects WHERE name=?", (name,),
		)
		self.conn.commit()
		return cursor.rowcount > 0

	def list_projects(self) -> list[ProjectInfo]:
		"""List all registered projects."""
		rows = self.conn.execute(
			"SELECT * FROM projects ORDER BY name ASC",
		).fetchall()
		return [self._row_to_project(r) for r in rows]

	def get_project(self, name: str) -> ProjectInfo | None:
		"""Get a single project by name."""
		row = self.conn.execute(
			"SELECT * FROM projects WHERE name=?", (name,),
		).fetchone()
		if row is None:
			return None
		return self._row_to_project(row)

	def update_pid(self, name: str, pid: int | None) -> None:
		"""Update the active PID for a project."""
		self.conn.execute(
			"UPDATE projects SET active_pid=? WHERE name=?",
			(pid, name),
		)
		self.conn.commit()

	def get_project_status(self, name: str) -> ProjectStatus | None:
		"""Get project info enriched with latest mission status from project DB."""
		project = self.get_project(name)
		if project is None:
			return None

		status = ProjectStatus(project=project)

		# Try to read latest mission from project's own DB
		project_db_path = Path(project.db_path)
		if project_db_path.exists():
			try:
				from mission_control.db import Database
				db = Database(project_db_path)
				try:
					mission = db.get_latest_mission()
					if mission:
						status.mission_status = mission.status
						status.mission_objective = mission.objective
						status.mission_score = mission.final_score
						status.mission_rounds = mission.total_rounds
						status.mission_cost = mission.total_cost_usd
				finally:
					db.close()
			except Exception:
				pass  # DB may be locked or corrupted

		return status

	@staticmethod
	def _row_to_project(row: sqlite3.Row) -> ProjectInfo:
		return ProjectInfo(
			id=row["id"],
			name=row["name"],
			config_path=row["config_path"],
			db_path=row["db_path"],
			description=row["description"],
			registered_at=row["registered_at"],
			active_pid=row["active_pid"],
		)
