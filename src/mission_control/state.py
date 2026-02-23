"""Project health snapshots -- run verification and parse output."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import time
from typing import Any

from mission_control.config import MissionConfig
from mission_control.models import (
	Snapshot,
	SnapshotDelta,
	VerificationNodeKind,
	VerificationReport,
	VerificationResult,
)


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


_KIND_PARSER_MAP: dict[str, Any] = {
	"pytest": _parse_pytest,
	"ruff": _parse_ruff,
	"mypy": _parse_mypy,
	"bandit": _parse_bandit,
}


def _tool_detected(kind: str, output: str) -> bool:
	"""Check if a verification tool actually produced recognizable output."""
	if kind == "pytest":
		return bool(re.search(r"\d+ passed|\d+ failed|\d+ error|no tests ran", output))
	if kind == "ruff":
		return (
			"All checks passed" in output
			or bool(re.search(r"Found \d+ errors?", output))
			or bool(re.search(r"\S+:\d+:\d+: [A-Z]\d+", output))
		)
	if kind == "mypy":
		return "Success" in output or bool(re.search(r"\S+\.py:\d+: error:", output))
	if kind == "bandit":
		return "No issues identified" in output or ">> Issue:" in output or "Run started" in output
	return False


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


def _build_result_from_single_command(output: str, returncode: int) -> VerificationReport:
	"""Build a VerificationReport from a single combined verification command (backward compat).

	Detects which tools actually produced output via tool-specific markers.
	Tools that never ran are marked skipped=True, passed=False so downstream
	scoring distinguishes 'never ran' from 'ran and failed'.
	"""
	pytest_data = _parse_pytest(output)
	ruff_data = _parse_ruff(output)
	mypy_data = _parse_mypy(output)
	bandit_data = _parse_bandit(output)

	pytest_ran = _tool_detected("pytest", output)
	ruff_ran = _tool_detected("ruff", output)
	mypy_ran = _tool_detected("mypy", output)
	bandit_ran = _tool_detected("bandit", output)

	# Only mark undetected tools as skipped when at least one tool IS detected.
	# If no tools are detected (e.g. custom script or empty output), fall back
	# to trusting the returncode for all nodes (backward compat).
	any_detected = pytest_ran or ruff_ran or mypy_ran or bandit_ran

	results: list[VerificationResult] = []

	# Pytest node
	pytest_skipped = not pytest_ran and any_detected
	test_passed = pytest_data["test_failed"] == 0 and pytest_data["test_total"] > 0
	results.append(VerificationResult(
		kind=VerificationNodeKind.PYTEST,
		passed=False if pytest_skipped else (test_passed or (pytest_data["test_total"] == 0 and returncode == 0)),
		skipped=pytest_skipped,
		exit_code=returncode,
		output=output,
		metrics=pytest_data,
	))

	# Ruff node
	ruff_skipped = not ruff_ran and any_detected
	results.append(VerificationResult(
		kind=VerificationNodeKind.RUFF,
		passed=False if ruff_skipped else ruff_data["lint_errors"] == 0,
		skipped=ruff_skipped,
		exit_code=0 if ruff_data["lint_errors"] == 0 else 1,
		output=output,
		metrics=ruff_data,
	))

	# Mypy node
	mypy_skipped = not mypy_ran and any_detected
	results.append(VerificationResult(
		kind=VerificationNodeKind.MYPY,
		passed=False if mypy_skipped else mypy_data["type_errors"] == 0,
		skipped=mypy_skipped,
		exit_code=0 if mypy_data["type_errors"] == 0 else 1,
		output=output,
		metrics=mypy_data,
	))

	# Bandit node
	bandit_skipped = not bandit_ran and any_detected
	results.append(VerificationResult(
		kind=VerificationNodeKind.BANDIT,
		passed=False if bandit_skipped else bandit_data["security_findings"] == 0,
		skipped=bandit_skipped,
		exit_code=0 if bandit_data["security_findings"] == 0 else 1,
		output=output,
		metrics=bandit_data,
	))

	# When the overall command succeeded, mark non-skipped nodes as passed
	if returncode == 0:
		for r in results:
			if not r.skipped:
				r.passed = True

	return VerificationReport(results=results, raw_output=output)


async def _run_single_node(
	kind_str: str,
	command: str,
	cwd: str,
	timeout: int,
	required: bool,
	weight: float,
) -> VerificationResult:
	"""Run a single verification node command and parse its output."""
	try:
		kind = VerificationNodeKind(kind_str)
	except ValueError:
		kind = VerificationNodeKind.CUSTOM
	parser = _KIND_PARSER_MAP.get(kind_str)

	t0 = time.monotonic()
	result = await _run_command(command, cwd, timeout)
	duration = time.monotonic() - t0

	output = result["output"]
	returncode = result["returncode"]
	metrics: dict[str, int] = parser(output) if parser else {}
	passed = returncode == 0

	return VerificationResult(
		kind=kind,
		passed=passed,
		exit_code=returncode,
		output=output,
		metrics=metrics,
		duration_seconds=duration,
		required=required,
		weight=weight,
	)


async def run_verification_nodes(config: MissionConfig, cwd: str) -> VerificationReport:
	"""Run verification as typed nodes and return a structured report.

	If config.target.verification.nodes is empty, falls back to single-command
	mode for backward compatibility.
	"""
	verification = config.target.verification
	nodes = verification.nodes

	if not nodes:
		# Backward compat: single command mode
		result = await _run_command(verification.command, cwd, verification.timeout)
		return _build_result_from_single_command(result["output"], result["returncode"])

	# Multi-node mode: run required nodes sequentially, optional can overlap
	required_nodes = [n for n in nodes if n.required]
	optional_nodes = [n for n in nodes if not n.required]

	results: list[VerificationResult] = []

	# Run required nodes sequentially (order matters)
	for node in required_nodes:
		vr = await _run_single_node(
			node.kind, node.command, cwd, node.timeout, node.required, node.weight,
		)
		results.append(vr)

	# Run optional nodes concurrently
	if optional_nodes:
		optional_tasks = [
			_run_single_node(
				n.kind, n.command, cwd, n.timeout, n.required, n.weight,
			)
			for n in optional_nodes
		]
		optional_results = await asyncio.gather(*optional_tasks)
		results.extend(optional_results)

	raw_output = "\n".join(r.output for r in results)
	return VerificationReport(results=results, raw_output=raw_output)


async def snapshot_project_health(config: MissionConfig, cwd: str | None = None) -> Snapshot:
	"""Take a project health snapshot by running verification commands."""
	cwd = cwd or str(config.target.resolved_path)

	report = await run_verification_nodes(config, cwd)

	# Aggregate metrics from all results
	pytest_data: dict[str, int] = {"test_total": 0, "test_passed": 0, "test_failed": 0}
	ruff_data: dict[str, int] = {"lint_errors": 0}
	mypy_data: dict[str, int] = {"type_errors": 0}
	bandit_data: dict[str, int] = {"security_findings": 0}

	for r in report.results:
		if r.kind == VerificationNodeKind.PYTEST:
			pytest_data = r.metrics or pytest_data
		elif r.kind == VerificationNodeKind.RUFF:
			ruff_data = r.metrics or ruff_data
		elif r.kind == VerificationNodeKind.MYPY:
			mypy_data = r.metrics or mypy_data
		elif r.kind == VerificationNodeKind.BANDIT:
			bandit_data = r.metrics or bandit_data

	max_raw_chars = config.scheduler.raw_output_max_chars
	raw = {"verification": report.raw_output[-max_raw_chars:]}

	return Snapshot(
		test_total=pytest_data.get("test_total", 0),
		test_passed=pytest_data.get("test_passed", 0),
		test_failed=pytest_data.get("test_failed", 0),
		lint_errors=ruff_data.get("lint_errors", 0),
		type_errors=mypy_data.get("type_errors", 0),
		security_findings=bandit_data.get("security_findings", 0),
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
