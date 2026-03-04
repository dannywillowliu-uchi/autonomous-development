"""Torture test feedback integration for mission-control.

Runs the GCC torture test suite on the target project and feeds
results back to the planner for correctness-driven prioritization.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TortureFailure:
	test_name: str
	category: str
	error_type: str  # "wrong_output" | "crash" | "timeout" | "compile_error"
	detail: str


@dataclass
class TortureResults:
	total: int = 0
	passed: int = 0
	failed: int = 0
	skipped: int = 0
	pass_rate: float = 0.0
	newly_passing: list[str] = field(default_factory=list)
	newly_failing: list[str] = field(default_factory=list)
	newly_compiling: list[str] = field(default_factory=list)
	failures: list[TortureFailure] = field(default_factory=list)
	delta_passed: int = 0
	delta_failed: int = 0


async def run_torture_suite(project_path: str, python_cmd: str = ".venv/bin/python") -> TortureResults | None:
	"""Run the torture test runner on the project, return parsed results.

	Args:
		project_path: Path to the C compiler project root
		python_cmd: Python command to use (relative to project_path)

	Returns:
		TortureResults if runner succeeded, None if runner not found or errored.
	"""
	runner_path = Path(project_path) / "tests" / "torture" / "runner.py"
	if not runner_path.exists():
		log.info("Torture runner not found at %s, skipping", runner_path)
		return None

	results_path = Path(project_path) / "tests" / "torture" / "results.json"
	baseline_path = Path(project_path) / "tests" / "torture" / "baseline.json"
	state_path = Path(project_path) / "TORTURE_RESULTS.md"

	cmd = [
		python_cmd, str(runner_path),
		"--baseline", str(baseline_path),
		"--output", str(results_path),
		"--state-file", str(state_path),
	]

	env_additions = {"PYTHONPATH": str(Path(project_path) / "src")}

	try:
		import os
		env = {**os.environ, **env_additions}
		proc = await asyncio.create_subprocess_exec(
			*cmd,
			cwd=project_path,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
			env=env,
		)
		stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

		if proc.returncode != 0:
			log.warning("Torture runner exited with code %d: %s", proc.returncode, stderr.decode()[:500])
			# Runner should always exit 0, but parse results anyway if file exists

		if stdout:
			log.info("Torture runner: %s", stdout.decode().strip())

	except asyncio.TimeoutError:
		log.warning("Torture runner timed out after 120s")
		return None
	except FileNotFoundError:
		log.warning("Python command not found: %s", python_cmd)
		return None
	except Exception as exc:
		log.warning("Torture runner failed: %s", exc)
		return None

	# Parse results
	if not results_path.exists():
		log.warning("Torture results not written to %s", results_path)
		return None

	try:
		data = json.loads(results_path.read_text())
	except (json.JSONDecodeError, OSError) as exc:
		log.warning("Failed to parse torture results: %s", exc)
		return None

	summary = data.get("summary", {})
	tests = data.get("tests", {})
	deltas = data.get("deltas", {})

	total = summary.get("total", 0)
	passed = summary.get("passed", 0)

	failures = []
	for name, info in tests.items():
		if info.get("status") == "FAIL":
			error_msg = info.get("error_msg", "")
			# Determine error type from message
			if "timeout" in error_msg.lower():
				error_type = "timeout"
			elif "signal" in error_msg.lower() or "crash" in error_msg.lower():
				error_type = "crash"
			else:
				error_type = "wrong_output"
			failures.append(TortureFailure(
				test_name=name,
				category=info.get("category", "unknown"),
				error_type=error_type,
				detail=error_msg[:200],
			))

	return TortureResults(
		total=total,
		passed=passed,
		failed=summary.get("failed", 0),
		skipped=summary.get("skipped", 0),
		pass_rate=round(passed / total * 100, 1) if total > 0 else 0.0,
		newly_passing=deltas.get("newly_passing", []),
		newly_failing=deltas.get("newly_failing", []),
		newly_compiling=deltas.get("newly_compiling", []),
		failures=failures,
		delta_passed=len(deltas.get("newly_passing", [])),
		delta_failed=len(deltas.get("newly_failing", [])),
	)


def format_for_planner(results: TortureResults) -> str:
	"""Format results as markdown for MISSION_STATE.md injection."""
	lines = []
	lines.append("## Torture Test Results")
	lines.append("")
	lines.append(f"**Score: PASS: {results.passed}/{results.total} | FAIL: {results.failed}/{results.total} | SKIP: {results.skipped}/{results.total}** ({results.pass_rate}% pass rate)")
	lines.append("")

	# Deltas
	if results.newly_passing or results.newly_failing or results.newly_compiling:
		lines.append("### Delta vs Baseline")
		if results.newly_passing:
			lines.append(f"- **+{len(results.newly_passing)} newly passing**: {', '.join(results.newly_passing[:10])}")
		if results.newly_failing:
			lines.append(f"- **-{len(results.newly_failing)} REGRESSIONS**: {', '.join(results.newly_failing[:10])}")
		if results.newly_compiling:
			lines.append(f"- +{len(results.newly_compiling)} newly compiling: {', '.join(results.newly_compiling[:10])}")
		lines.append("")

	# Failures grouped by category
	if results.failures:
		lines.append("### Failures by Category")
		by_cat: dict[str, list[TortureFailure]] = {}
		for f in results.failures:
			by_cat.setdefault(f.category, []).append(f)

		# Sort categories by number of failures (most first)
		for cat, fails in sorted(by_cat.items(), key=lambda x: -len(x[1])):
			lines.append(f"\n**{cat}** ({len(fails)} failures)")
			for f in fails[:10]:  # Cap at 10 per category to avoid huge output
				lines.append(f"- `{f.test_name}`: {f.error_type} -- {f.detail[:100]}")
		lines.append("")

	# Priority instruction
	lines.append("### Priority Instruction")
	lines.append("PRIORITIZE fixing tests in the FAIL categories over adding new features.")
	lines.append("Tests in SKIP will naturally become testable as features are added.")
	if results.newly_failing:
		lines.append(f"**URGENT: {len(results.newly_failing)} regressions detected -- fix these first.**")
	lines.append("")

	return "\n".join(lines)


def format_for_feedback_context(results: TortureResults) -> str:
	"""Compact format for build_planner_context() feedback string."""
	lines = []
	lines.append(f"Torture Tests: {results.passed}/{results.total} passing ({results.pass_rate}%)")
	if results.newly_passing:
		lines.append(f"  +{len(results.newly_passing)} newly passing")
	if results.newly_failing:
		lines.append(f"  -{len(results.newly_failing)} REGRESSIONS: {', '.join(results.newly_failing[:5])}")

	# Top failure categories
	if results.failures:
		by_cat: dict[str, int] = {}
		for f in results.failures:
			by_cat[f.category] = by_cat.get(f.category, 0) + 1
		top = sorted(by_cat.items(), key=lambda x: -x[1])[:3]
		cats = ", ".join(f"{cat}({n})" for cat, n in top)
		lines.append(f"  Top failure categories: {cats}")

	return "\n".join(lines)


def store_torture_experience(db, results: TortureResults, epoch_id: str, units: list) -> None:
	"""Store torture deltas as experiences for cross-mission learning.

	If pass rate improved: high reward experience linking unit titles to improvement.
	If regression: negative reward experience with specific test names.

	Args:
		db: Database instance with insert_experience()
		results: TortureResults from this epoch
		epoch_id: Current epoch ID
		units: List of completed work units this epoch
	"""
	import uuid
	from datetime import datetime, timezone

	if not results or (results.delta_passed == 0 and results.delta_failed == 0):
		return  # No change, nothing to record

	# Collect unit info for attribution
	unit_titles = [getattr(u, "title", str(u)) for u in units[:5]]
	unit_ids = [getattr(u, "id", "") for u in units[:1]]  # Use first unit as FK
	files_hints = [getattr(u, "files_hint", "") for u in units if getattr(u, "files_hint", "")]

	# Positive experience: tests improved
	if results.delta_passed > 0:
		from mission_control.models import Experience
		exp = Experience(
			id=str(uuid.uuid4()),
			round_id=epoch_id,
			work_unit_id=unit_ids[0] if unit_ids else epoch_id,
			timestamp=datetime.now(timezone.utc).isoformat(),
			title=f"Torture +{results.delta_passed}: {', '.join(unit_titles[:3])}",
			scope=f"torture-improvement",
			files_hint=", ".join(files_hints[:5]),
			status="completed",
			summary=f"Torture test pass rate improved by {results.delta_passed} tests after this epoch's work.",
			files_changed=json.dumps(files_hints[:10]),
			discoveries=json.dumps([f"Newly passing: {t}" for t in results.newly_passing[:10]]),
			concerns=json.dumps([]),
			reward=results.delta_passed * 0.5,
		)
		try:
			db.insert_experience(exp)
			log.info("Stored positive torture experience: +%d tests", results.delta_passed)
		except Exception as exc:
			log.warning("Failed to store torture experience: %s", exc)

	# Negative experience: regressions
	if results.delta_failed > 0:
		from mission_control.models import Experience
		exp = Experience(
			id=str(uuid.uuid4()),
			round_id=epoch_id,
			work_unit_id=unit_ids[0] if unit_ids else epoch_id,
			timestamp=datetime.now(timezone.utc).isoformat(),
			title=f"Torture -{results.delta_failed}: REGRESSIONS from {', '.join(unit_titles[:3])}",
			scope=f"torture-regression",
			files_hint=", ".join(files_hints[:5]),
			status="failed",
			summary=f"Torture test regressions: {results.delta_failed} tests broke after this epoch.",
			files_changed=json.dumps(files_hints[:10]),
			discoveries=json.dumps([]),
			concerns=json.dumps([f"Regression: {t}" for t in results.newly_failing[:10]]),
			reward=-results.delta_failed * 1.0,
		)
		try:
			db.insert_experience(exp)
			log.info("Stored negative torture experience: -%d regressions", results.delta_failed)
		except Exception as exc:
			log.warning("Failed to store torture regression experience: %s", exc)
