"""Operational health checks for mission-control installations."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def _check_entry(name: str, status: str, details: list[str], remediation: str = "") -> dict[str, Any]:
	entry: dict[str, Any] = {"name": name, "status": status, "details": details}
	if remediation:
		entry["remediation"] = remediation
	return entry


def _check_db_health(db_path: Path) -> dict[str, Any]:
	"""Check database health: stale WAL/SHM files, orphaned running units, DB size."""
	if not db_path.exists():
		return _check_entry("Database", "WARN", ["Database file not found"], "Run 'mc mission' to create the database")

	details: list[str] = []
	status = "OK"
	remediation = ""

	# DB file size
	size_bytes = db_path.stat().st_size
	if size_bytes < 1024:
		details.append(f"Size: {size_bytes} B")
	elif size_bytes < 1024 * 1024:
		details.append(f"Size: {size_bytes / 1024:.1f} KB")
	else:
		details.append(f"Size: {size_bytes / (1024 * 1024):.1f} MB")

	# Stale WAL/SHM files
	wal_path = db_path.parent / (db_path.name + "-wal")
	shm_path = db_path.parent / (db_path.name + "-shm")
	stale_files = []
	for p in (wal_path, shm_path):
		if p.exists() and p.stat().st_size > 0:
			stale_files.append(p.name)
	if stale_files:
		status = "WARN"
		details.append(f"Stale WAL/SHM files: {', '.join(stale_files)}")
		remediation = (
			"Open and close the database to checkpoint WAL, "
			"or delete stale -shm/-wal files if no process is using the DB"
		)

	# Orphaned running work units
	try:
		from mission_control.db import Database

		with Database(db_path) as db:
			row = db.conn.execute(
				"SELECT COUNT(*) AS cnt FROM work_units WHERE status='running'"
			).fetchone()
			orphan_count = int(row["cnt"]) if row else 0
			if orphan_count > 0:
				status = "WARN" if status == "OK" else status
				details.append(f"Orphaned running work units: {orphan_count}")
				if not remediation:
					remediation = "Investigate orphaned units -- they may be from a crashed mission"
			else:
				details.append("No orphaned running work units")
	except Exception as exc:
		status = "ERROR"
		details.append(f"Failed to query database: {exc}")
		remediation = "Check database integrity"

	return _check_entry("Database", status, details, remediation)


def _check_workspace_health(config_path: Path) -> dict[str, Any]:
	"""Check workspace pool: leftover worker dirs, broken .venv symlinks, editable install."""
	details: list[str] = []
	status = "OK"
	remediation = ""

	try:
		from mission_control.config import load_config

		config = load_config(config_path)
	except Exception:
		return _check_entry("Workspace Pool", "WARN", ["Could not load config to check workspace pool"])

	pool_dir_str = config.scheduler.parallel.pool_dir
	if not pool_dir_str:
		return _check_entry("Workspace Pool", "OK", ["No pool_dir configured (using default)"])

	pool_path = Path(os.path.expanduser(pool_dir_str))
	if not pool_path.exists():
		return _check_entry("Workspace Pool", "OK", ["Pool directory does not exist yet"])

	# Leftover worker-* directories
	worker_dirs = [p for p in pool_path.iterdir() if p.is_dir() and p.name.startswith("worker-")]
	if worker_dirs:
		status = "WARN"
		details.append(f"Leftover worker directories: {len(worker_dirs)}")
		remediation = "Remove leftover worker-* directories in the pool, or run a mission to clean them up"
	else:
		details.append("No leftover worker directories")

	# Check .venv symlinks in worker dirs
	broken_venvs = []
	for wd in worker_dirs:
		venv = wd / ".venv"
		if venv.is_symlink() and not venv.resolve().exists():
			broken_venvs.append(wd.name)
	if broken_venvs:
		status = "WARN"
		details.append(f"Broken .venv symlinks in: {', '.join(broken_venvs[:5])}")
		if not remediation:
			remediation = "Re-create .venv symlinks or remove stale worker directories"

	# Check editable install .pth files
	for wd in worker_dirs:
		venv = wd / ".venv"
		if not venv.is_dir():
			continue
		site_packages = list(venv.glob("lib/python*/site-packages"))
		for sp in site_packages:
			pth_files = list(sp.glob("*.pth"))
			for pth in pth_files:
				try:
					content = pth.read_text().strip()
					if content and not Path(content).exists():
						status = "WARN"
						details.append(f"Broken .pth in {wd.name}: points to {content}")
						if not remediation:
							remediation = "Fix editable install .pth files or re-run setup"
				except OSError:
					pass

	if not details:
		details.append("Workspace pool healthy")

	return _check_entry("Workspace Pool", status, details, remediation)


def _check_config_health(config_path: Path) -> dict[str, Any]:
	"""Check config: loads without error, verification command is executable."""
	details: list[str] = []
	status = "OK"
	remediation = ""

	if not config_path.exists():
		return _check_entry(
			"Config", "ERROR",
			[f"Config file not found: {config_path}"],
			"Run 'mc init' to create a config",
		)

	try:
		from mission_control.config import load_config

		config = load_config(config_path)
		details.append(f"Config loaded: {config_path.name}")
	except Exception as exc:
		return _check_entry("Config", "ERROR", [f"Config load failed: {exc}"], "Fix TOML syntax in config file")

	# Check verification command is executable
	verify_cmd = config.target.verification.command
	if verify_cmd:
		first_token = verify_cmd.split()[0]
		if shutil.which(first_token) is not None:
			details.append(f"Verification command executable: {first_token}")
		else:
			status = "WARN"
			details.append(f"Verification command not found: {first_token}")
			remediation = f"Install or add '{first_token}' to PATH"
	else:
		details.append("No verification command configured")

	return _check_entry("Config", status, details, remediation)


def _check_cost_summary(db_path: Path) -> dict[str, Any]:
	"""Summarize total cost across recent missions."""
	if not db_path.exists():
		return _check_entry("Cost Summary", "OK", ["No database -- no cost data"])

	try:
		from mission_control.db import Database

		with Database(db_path) as db:
			missions = db.get_all_missions(limit=10)
			if not missions:
				return _check_entry("Cost Summary", "OK", ["No missions found"])

			total_cost = sum(m.total_cost_usd for m in missions)
			details = [
				f"Recent missions: {len(missions)}",
				f"Total cost: ${total_cost:.2f}",
			]
			if missions:
				latest = missions[0]
				details.append(f"Latest: {latest.status} (${latest.total_cost_usd:.2f})")
			return _check_entry("Cost Summary", "OK", details)
	except Exception as exc:
		return _check_entry("Cost Summary", "WARN", [f"Failed to query costs: {exc}"])


def run_diagnose(config_path: Path, db_path: Path) -> dict[str, Any]:
	"""Run all health checks and return a structured report."""
	checks = [
		_check_db_health(db_path),
		_check_workspace_health(config_path),
		_check_config_health(config_path),
		_check_cost_summary(db_path),
	]
	return {"checks": checks}
