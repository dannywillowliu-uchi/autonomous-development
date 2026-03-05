"""Core test feedback integration for mission-control.

Runs a project-defined core test suite after each epoch and feeds
pass/fail/regression data back to the planner for correctness-driven
prioritization. The test runner and results format are project-specific;
this module defines the contract (JSON schema) and integration points.

Expected results.json schema from the runner:
{
  "summary": {"total": N, "passed": N, "failed": N, "skipped": N},
  "tests": {"test_name": {"status": "PASS|FAIL|SKIP", "category": "...", "error_msg": "...", "diagnostic": "...", "exit_code": N}},
  "deltas": {"newly_passing": [...], "newly_failing": [...], "newly_compiling": [...]},
  "skip_analysis": {"pattern": {"count": N, "examples": [...]}}
}
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class CoreTestFailure:
	test_name: str
	category: str
	error_type: str  # "wrong_output" | "crash" | "timeout" | "compile_error"
	detail: str


@dataclass
class SkipPattern:
	pattern: str
	count: int
	examples: list[str] = field(default_factory=list)


@dataclass
class CoreTestResults:
	total: int = 0
	passed: int = 0
	failed: int = 0
	skipped: int = 0
	pass_rate: float = 0.0
	newly_passing: list[str] = field(default_factory=list)
	newly_failing: list[str] = field(default_factory=list)
	newly_compiling: list[str] = field(default_factory=list)
	failures: list[CoreTestFailure] = field(default_factory=list)
	skip_patterns: list[SkipPattern] = field(default_factory=list)
	delta_passed: int = 0
	delta_failed: int = 0


async def run_core_tests(
	project_path: str,
	runner_command: str,
	baseline_path: str = "",
	timeout: int = 120,
) -> CoreTestResults | None:
	"""Run the project's core test runner, return parsed results.

	The runner_command is executed as a shell command with cwd=project_path.
	It must produce a results.json in the same directory as the runner script
	(or alongside the baseline). The runner decides its own invocation style
	(python, bash, compiled binary, etc.).

	Args:
		project_path: Path to the target project root.
		runner_command: Shell command to run the test suite (executed in project_path).
		baseline_path: Path to baseline JSON relative to project_path (for delta computation).
		timeout: Max seconds to wait for the runner.

	Returns:
		CoreTestResults if runner succeeded, None if not found or errored.
	"""
	project = Path(project_path)

	try:
		proc = await asyncio.create_subprocess_shell(
			runner_command,
			cwd=project_path,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

		if proc.returncode != 0:
			log.warning("Core test runner exited with code %d: %s", proc.returncode, stderr.decode()[:500])

		if stdout:
			log.info("Core tests: %s", stdout.decode().strip())

	except asyncio.TimeoutError:
		log.warning("Core test runner timed out after %ds", timeout)
		return None
	except Exception as exc:
		log.warning("Core test runner failed: %s", exc)
		return None

	# Find results.json: look next to baseline first, then project root
	results_path = None
	if baseline_path:
		candidate = project / Path(baseline_path).parent / "results.json"
		if candidate.exists():
			results_path = candidate
	if results_path is None:
		candidate = project / "results.json"
		if candidate.exists():
			results_path = candidate

	if results_path is None:
		log.warning("Core test results.json not found in %s", project_path)
		return None

	try:
		data = json.loads(results_path.read_text())
	except (json.JSONDecodeError, OSError) as exc:
		log.warning("Failed to parse core test results: %s", exc)
		return None

	return _parse_results(data)


def _parse_results(data: dict) -> CoreTestResults:
	"""Parse the standardized results.json into CoreTestResults."""
	summary = data.get("summary", {})
	tests = data.get("tests", {})
	deltas = data.get("deltas", {})
	skip_analysis_raw = data.get("skip_analysis", {})

	total = summary.get("total", 0)
	passed = summary.get("passed", 0)

	failures = []
	for name, info in tests.items():
		if info.get("status") == "FAIL":
			error_msg = info.get("error_msg", "")
			diagnostic = info.get("diagnostic", "")
			if "timeout" in error_msg.lower():
				error_type = "timeout"
			elif "signal" in error_msg.lower() or "crash" in error_msg.lower():
				error_type = "crash"
			else:
				error_type = "wrong_output"
			# Use diagnostic for richer detail if available
			detail = diagnostic if diagnostic else error_msg
			failures.append(CoreTestFailure(
				test_name=name,
				category=info.get("category", "unknown"),
				error_type=error_type,
				detail=detail[:300],
			))

	# Parse skip analysis patterns (sorted by count descending)
	skip_patterns = []
	for pattern, info in sorted(skip_analysis_raw.items(), key=lambda x: -x[1].get("count", 0)):
		skip_patterns.append(SkipPattern(
			pattern=pattern,
			count=info.get("count", 0),
			examples=info.get("examples", [])[:5],
		))

	return CoreTestResults(
		total=total,
		passed=passed,
		failed=summary.get("failed", 0),
		skipped=summary.get("skipped", 0),
		pass_rate=round(passed / total * 100, 1) if total > 0 else 0.0,
		newly_passing=deltas.get("newly_passing", []),
		newly_failing=deltas.get("newly_failing", []),
		newly_compiling=deltas.get("newly_compiling", []),
		failures=failures,
		skip_patterns=skip_patterns,
		delta_passed=len(deltas.get("newly_passing", [])),
		delta_failed=len(deltas.get("newly_failing", [])),
	)


def format_for_planner(results: CoreTestResults) -> str:
	"""Format results as markdown for MISSION_STATE.md injection.

	Provides the planner with:
	1. Score and deltas (what changed)
	2. Full failure diagnostics (why each test fails)
	3. Skip analysis (what missing features block the most tests)
	4. Root-cause tracing instructions (how to investigate)
	"""
	lines = []
	lines.append("## Core Test Results")
	lines.append("")
	lines.append(f"**Score: PASS: {results.passed}/{results.total} | FAIL: {results.failed}/{results.total} | SKIP: {results.skipped}/{results.total}** ({results.pass_rate}% pass rate)")
	lines.append("")

	# Delta section
	if results.newly_passing or results.newly_failing or results.newly_compiling:
		lines.append("### Delta vs Baseline")
		if results.newly_passing:
			lines.append(f"- **+{len(results.newly_passing)} newly passing**: {', '.join(results.newly_passing[:10])}")
		if results.newly_failing:
			lines.append(f"- **-{len(results.newly_failing)} REGRESSIONS**: {', '.join(results.newly_failing[:10])}")
		if results.newly_compiling:
			lines.append(f"- +{len(results.newly_compiling)} newly compiling: {', '.join(results.newly_compiling[:10])}")
		lines.append("")

	# Full failure diagnostics — the planner needs to understand WHY each test fails
	if results.failures:
		lines.append("### Failure Diagnostics")
		lines.append("Each failing test is listed with its error type and diagnostic detail.")
		lines.append("Use this to identify the root cause in the compiler/codebase.")
		lines.append("")
		by_cat: dict[str, list[CoreTestFailure]] = {}
		for f in results.failures:
			by_cat.setdefault(f.category, []).append(f)

		for cat, fails in sorted(by_cat.items(), key=lambda x: -len(x[1])):
			lines.append(f"**{cat}** ({len(fails)} failures)")
			for f in fails:
				lines.append(f"- `{f.test_name}` [{f.error_type}]: {f.detail}")
			lines.append("")

	# Skip analysis — shows what features would unlock the most tests
	if results.skip_patterns:
		lines.append("### Skip Analysis (Missing Features)")
		lines.append("Tests that cannot compile yet, grouped by error pattern.")
		lines.append("Fixing a common pattern unlocks many tests at once.")
		lines.append("")
		for sp in results.skip_patterns[:15]:
			examples_str = ", ".join(sp.examples[:3])
			lines.append(f"- **{sp.pattern}** ({sp.count} tests) e.g. {examples_str}")
		lines.append("")

	# Root-cause tracing instructions
	lines.append("### How to Use These Results")
	lines.append("1. **FAIL tests are the top priority.** Read each test's source file to understand what it exercises, then trace the failure back to the responsible component.")
	lines.append("2. **Regressions are urgent.** If a previously-passing test now fails, the most recent changes likely caused it. Revert or fix immediately.")
	lines.append("3. **Skip patterns show high-leverage features.** Implementing a missing feature that appears in many skip patterns unlocks those tests for validation.")
	lines.append("4. **Do not add new features** that aren't related to fixing existing FAIL tests or unblocking high-count SKIP patterns.")
	if results.newly_failing:
		lines.append(f"**URGENT: {len(results.newly_failing)} regressions detected -- fix these before any other work.**")
	lines.append("")

	return "\n".join(lines)


def format_for_feedback_context(results: CoreTestResults) -> str:
	"""Compact format for build_planner_context() feedback string."""
	lines = []
	lines.append(f"Core Tests: {results.passed}/{results.total} passing ({results.pass_rate}%)")
	if results.newly_passing:
		lines.append(f"  +{len(results.newly_passing)} newly passing")
	if results.newly_failing:
		lines.append(f"  -{len(results.newly_failing)} REGRESSIONS: {', '.join(results.newly_failing[:5])}")

	if results.failures:
		by_cat: dict[str, int] = {}
		for f in results.failures:
			by_cat[f.category] = by_cat.get(f.category, 0) + 1
		top = sorted(by_cat.items(), key=lambda x: -x[1])[:3]
		cats = ", ".join(f"{cat}({n})" for cat, n in top)
		lines.append(f"  Top failure categories: {cats}")
		# Include brief failure details so planner sees root causes
		for f in results.failures[:5]:
			lines.append(f"  FAIL {f.test_name}: {f.detail[:80]}")

	if results.skip_patterns:
		top_skips = results.skip_patterns[:3]
		skip_summary = ", ".join(f"{sp.pattern}({sp.count})" for sp in top_skips)
		lines.append(f"  Top skip reasons: {skip_summary}")

	return "\n".join(lines)


def store_core_test_experience(db, results: CoreTestResults, epoch_id: str, units: list) -> None:
	"""Store core test deltas as experiences for cross-mission learning.

	Positive reward for improvements, negative for regressions.

	Args:
		db: Database instance with insert_experience()
		results: CoreTestResults from this epoch
		epoch_id: Current epoch ID
		units: List of completed work units this epoch
	"""
	import uuid
	from datetime import datetime, timezone

	if not results or (results.delta_passed == 0 and results.delta_failed == 0):
		return

	unit_titles = [getattr(u, "title", str(u)) for u in units[:5]]
	unit_ids = [getattr(u, "id", "") for u in units[:1]]
	files_hints = [getattr(u, "files_hint", "") for u in units if getattr(u, "files_hint", "")]

	if results.delta_passed > 0:
		from mission_control.models import Experience
		exp = Experience(
			id=str(uuid.uuid4()),
			round_id=epoch_id,
			work_unit_id=unit_ids[0] if unit_ids else epoch_id,
			timestamp=datetime.now(timezone.utc).isoformat(),
			title=f"Core tests +{results.delta_passed}: {', '.join(unit_titles[:3])}",
			scope="core-tests-improvement",
			files_hint=", ".join(files_hints[:5]),
			status="completed",
			summary=f"Core test pass rate improved by {results.delta_passed} tests after this epoch.",
			files_changed=json.dumps(files_hints[:10]),
			discoveries=json.dumps([f"Newly passing: {t}" for t in results.newly_passing[:10]]),
			concerns=json.dumps([]),
			reward=results.delta_passed * 0.5,
		)
		try:
			db.insert_experience(exp)
			log.info("Stored positive core test experience: +%d tests", results.delta_passed)
		except Exception as exc:
			log.warning("Failed to store core test experience: %s", exc)

	if results.delta_failed > 0:
		from mission_control.models import Experience
		exp = Experience(
			id=str(uuid.uuid4()),
			round_id=epoch_id,
			work_unit_id=unit_ids[0] if unit_ids else epoch_id,
			timestamp=datetime.now(timezone.utc).isoformat(),
			title=f"Core tests -{results.delta_failed}: REGRESSIONS from {', '.join(unit_titles[:3])}",
			scope="core-tests-regression",
			files_hint=", ".join(files_hints[:5]),
			status="failed",
			summary=f"Core test regressions: {results.delta_failed} tests broke after this epoch.",
			files_changed=json.dumps(files_hints[:10]),
			discoveries=json.dumps([]),
			concerns=json.dumps([f"Regression: {t}" for t in results.newly_failing[:10]]),
			reward=-results.delta_failed * 1.0,
		)
		try:
			db.insert_experience(exp)
			log.info("Stored negative core test experience: -%d regressions", results.delta_failed)
		except Exception as exc:
			log.warning("Failed to store core test regression experience: %s", exc)
