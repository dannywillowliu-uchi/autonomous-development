"""Project health snapshots -- run verification and parse output."""

from __future__ import annotations

import asyncio
import re
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
	total = passed + failed + errors

	return {"test_total": total, "test_passed": passed, "test_failed": failed}


def _parse_ruff(output: str) -> dict[str, int]:
	"""Parse ruff output -- count error lines."""
	if not output.strip() or "All checks passed" in output:
		return {"lint_errors": 0}

	# Count lines matching ruff error format: "file.py:line:col: CODE message"
	error_lines = [line for line in output.strip().splitlines() if re.match(r".+:\d+:\d+:", line)]
	return {"lint_errors": len(error_lines)}


def _parse_mypy(output: str) -> dict[str, int]:
	"""Parse mypy output -- count 'error:' lines."""
	if "Success" in output:
		return {"type_errors": 0}
	error_count = sum(1 for line in output.splitlines() if "error:" in line)
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
		proc = await asyncio.create_subprocess_shell(
			cmd,
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
		return {"output": f"Command timed out after {timeout}s", "returncode": -1}
	except FileNotFoundError:
		return {"output": f"Command not found: {cmd}", "returncode": -1}


async def snapshot_project_health(config: MissionConfig) -> Snapshot:
	"""Take a project health snapshot by running verification commands."""
	cwd = str(config.target.resolved_path)
	timeout = config.target.verification.timeout

	# Run pytest, ruff, mypy separately for parsing
	pytest_result = await _run_command("uv run pytest -q 2>&1", cwd, timeout)
	ruff_result = await _run_command("uv run ruff check src/ 2>&1", cwd, timeout)
	mypy_result = await _run_command("uv run mypy src/ --ignore-missing-imports 2>&1", cwd, timeout)

	pytest_data = _parse_pytest(pytest_result["output"])
	ruff_data = _parse_ruff(ruff_result["output"])
	mypy_data = _parse_mypy(mypy_result["output"])

	raw = {
		"pytest": pytest_result["output"][-2000:],
		"ruff": ruff_result["output"][-2000:],
		"mypy": mypy_result["output"][-2000:],
	}

	import json

	return Snapshot(
		test_total=pytest_data["test_total"],
		test_passed=pytest_data["test_passed"],
		test_failed=pytest_data["test_failed"],
		lint_errors=ruff_data["lint_errors"],
		type_errors=mypy_data["type_errors"],
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
