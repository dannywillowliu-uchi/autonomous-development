"""Project health snapshots -- run verification and parse output."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from typing import Any

from mission_control.config import MissionConfig
from mission_control.models import Snapshot, SnapshotDelta


def _parse_pytest(output: str) -> dict[str, int]:
	"""Parse pytest output for pass/fail counts."""
	total = 0
	passed = 0
	failed = 0

	# Match "X passed", "X failed", "X error" patterns from summary line
	passed_m = re.search(r"(\d+) passed", output)
	failed_m = re.search(r"(\d+) failed", output)
	error_m = re.search(r"(\d+) error", output)

	if passed_m:
		passed = int(passed_m.group(1))
	if failed_m:
		failed = int(failed_m.group(1))

	errors = int(error_m.group(1)) if error_m else 0
	failed += errors
	total = passed + failed

	return {"test_total": total, "test_passed": passed, "test_failed": failed}


def _parse_ruff(output: str) -> dict[str, int]:
	"""Parse ruff output -- count error lines."""
	if not output.strip() or "All checks passed" in output:
		return {"lint_errors": 0}

	# Count lines matching ruff error format: "file.py:line:col: CODE message"
	error_lines = [line for line in output.strip().splitlines() if re.match(r".+:\d+:\d+:", line)]
	return {"lint_errors": len(error_lines)}


def _parse_mypy(output: str) -> dict[str, int]:
	"""Parse mypy output -- count lines matching mypy error format.

	Uses anchored regex to avoid false positives from pytest tracebacks
	that also contain 'error:'.
	"""
	if "Success" in output:
		return {"type_errors": 0}
	# Match mypy-specific format: file.py:line: error: message
	error_count = sum(
		1 for line in output.splitlines()
		if re.match(r"\S+\.py:\d+: error:", line)
	)
	return {"type_errors": error_count}


def _parse_bandit(output: str) -> dict[str, int]:
	"""Parse bandit output for security findings."""
	if "No issues identified" in output:
		return {"security_findings": 0}
	# Count ">> Issue:" lines
	findings = sum(1 for line in output.splitlines() if ">> Issue:" in line)
	return {"security_findings": findings}


async def _run_command(cmd: str, cwd: str, timeout: int = 300) -> dict[str, Any]:
	"""Run a shell command and capture output."""
	try:
		proc = await asyncio.create_subprocess_exec(
			*shlex.split(cmd),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
			cwd=cwd,
		)
		stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
		return {
			"output": stdout.decode("utf-8", errors="replace"),
			"returncode": proc.returncode,
		}
	except asyncio.TimeoutError:
		try:
			proc.kill()
			await proc.wait()
		except ProcessLookupError:
			pass
		return {"output": f"Command timed out after {timeout}s", "returncode": -1}
	except FileNotFoundError:
		return {"output": f"Command not found: {cmd}", "returncode": -1}


async def snapshot_project_health(config: MissionConfig, cwd: str | None = None) -> Snapshot:
	"""Take a project health snapshot by running verification commands."""
	cwd = cwd or str(config.target.resolved_path)
	timeout = config.target.verification.timeout

	# Run the project's configured verification command
	cmd = config.target.verification.command
	result = await _run_command(cmd, cwd, timeout)
	output = result["output"]

	pytest_data = _parse_pytest(output)
	ruff_data = _parse_ruff(output)
	mypy_data = _parse_mypy(output)
	bandit_data = _parse_bandit(output)

	max_raw_chars = config.scheduler.raw_output_max_chars

	raw = {
		"verification": output[-max_raw_chars:],
	}

	return Snapshot(
		test_total=pytest_data["test_total"],
		test_passed=pytest_data["test_passed"],
		test_failed=pytest_data["test_failed"],
		lint_errors=ruff_data["lint_errors"],
		type_errors=mypy_data["type_errors"],
		security_findings=bandit_data["security_findings"],
		raw_output=json.dumps(raw),
	)


def compare_snapshots(before: Snapshot, after: Snapshot) -> SnapshotDelta:
	"""Compare two snapshots to determine what changed."""
	return SnapshotDelta(
		tests_added=after.test_total - before.test_total,
		tests_fixed=max(0, after.test_passed - before.test_passed),
		tests_broken=max(0, after.test_failed - before.test_failed),
		lint_delta=after.lint_errors - before.lint_errors,
		type_delta=after.type_errors - before.type_errors,
		security_delta=after.security_findings - before.security_findings,
	)
