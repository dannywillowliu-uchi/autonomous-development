"""Swarm performance metrics tracking and trend analysis."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, fields
from pathlib import Path

METRICS_FILE = ".autodev-metrics.tsv"
EXPERIMENTS_FILE = ".autodev-experiments.tsv"


@dataclass
class SwarmMetrics:
	"""Metrics from a single swarm run."""
	run_id: str
	timestamp: str
	test_count: int
	test_pass_rate: float
	total_cost_usd: float
	cost_per_task: float
	agent_success_rate: float
	total_duration_s: float
	tasks_completed: int
	tasks_failed: int


_FIELDS = [f.name for f in fields(SwarmMetrics)]


class MetricsTracker:
	"""Track and analyze swarm performance over time."""

	def __init__(self, project_path: Path):
		self._project_path = project_path
		self._metrics_file = project_path / METRICS_FILE

	def record_run(self, metrics: SwarmMetrics) -> None:
		"""Append metrics for a completed run to the TSV file."""
		write_header = not self._metrics_file.exists() or self._metrics_file.stat().st_size == 0
		with open(self._metrics_file, "a", newline="") as f:
			writer = csv.writer(f, delimiter="\t")
			if write_header:
				writer.writerow(_FIELDS)
			writer.writerow([getattr(metrics, name) for name in _FIELDS])

	def _read_rows(self, last_n: int = 0) -> list[SwarmMetrics]:
		"""Read metrics rows from TSV. If last_n > 0, return only last N rows."""
		if not self._metrics_file.exists():
			return []
		text = self._metrics_file.read_text()
		reader = csv.DictReader(io.StringIO(text), delimiter="\t")
		rows = []
		for row in reader:
			rows.append(SwarmMetrics(
				run_id=row["run_id"],
				timestamp=row["timestamp"],
				test_count=int(row["test_count"]),
				test_pass_rate=float(row["test_pass_rate"]),
				total_cost_usd=float(row["total_cost_usd"]),
				cost_per_task=float(row["cost_per_task"]),
				agent_success_rate=float(row["agent_success_rate"]),
				total_duration_s=float(row["total_duration_s"]),
				tasks_completed=int(row["tasks_completed"]),
				tasks_failed=int(row["tasks_failed"]),
			))
		if last_n > 0:
			rows = rows[-last_n:]
		return rows

	def get_trend(self, last_n: int = 20) -> dict:
		"""Analyze trends across recent runs."""
		rows = self._read_rows(last_n)
		if len(rows) < 2:
			return {"error": "insufficient_data", "rows": len(rows)}

		mid = len(rows) // 2
		first_half = rows[:mid]
		second_half = rows[mid:]

		def _avg(items: list[SwarmMetrics], attr: str) -> float:
			vals = [getattr(m, attr) for m in items]
			return sum(vals) / len(vals) if vals else 0.0

		def _trend(attr: str) -> str:
			avg_first = _avg(first_half, attr)
			avg_second = _avg(second_half, attr)
			if avg_first == 0:
				return "stable" if avg_second == 0 else "increasing"
			ratio = avg_second / avg_first
			if ratio > 1.05:
				return "increasing"
			elif ratio < 0.95:
				return "decreasing"
			return "stable"

		velocity = (rows[-1].test_count - rows[0].test_count) / len(rows)

		best = max(rows, key=lambda m: m.test_pass_rate)
		worst = min(rows, key=lambda m: m.test_pass_rate)

		return {
			"test_count_trend": _trend("test_count"),
			"cost_trend": _trend("total_cost_usd"),
			"success_rate_trend": _trend("agent_success_rate"),
			"improvement_velocity": velocity,
			"best_run": best,
			"worst_run": worst,
			"total_runs": len(rows),
		}

	def correlate_with_modifications(self) -> list[dict]:
		"""Correlate metrics changes with self-modifications."""
		experiments_path = self._project_path / EXPERIMENTS_FILE
		if not experiments_path.exists():
			return []

		metrics_rows = self._read_rows()
		if not metrics_rows:
			return []

		# Build timestamp -> metrics lookup
		metrics_by_time = {m.timestamp: m for m in metrics_rows}
		timestamps = [m.timestamp for m in metrics_rows]

		exp_text = experiments_path.read_text()
		reader = csv.DictReader(io.StringIO(exp_text), delimiter="\t")

		results = []
		for exp in reader:
			exp_ts = exp.get("timestamp", "")
			title = exp.get("proposal_title", "unknown")

			# Find the closest metrics before and after this experiment
			before = None
			after = None
			for ts in timestamps:
				if ts < exp_ts:
					before = metrics_by_time[ts]
				elif ts >= exp_ts and after is None:
					after = metrics_by_time[ts]

			if before and after:
				delta_tests = after.test_count - before.test_count
				delta_cost = after.total_cost_usd - before.total_cost_usd
				delta_success = after.agent_success_rate - before.agent_success_rate

				if delta_tests > 0 and delta_success >= 0:
					verdict = "positive"
				elif delta_tests < 0 or delta_success < -0.1:
					verdict = "negative"
				else:
					verdict = "neutral"

				results.append({
					"proposal_title": title,
					"metric_delta": {
						"test_count": delta_tests,
						"total_cost_usd": round(delta_cost, 4),
						"agent_success_rate": round(delta_success, 4),
					},
					"verdict": verdict,
				})

		return results
