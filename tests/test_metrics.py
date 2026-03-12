"""Tests for swarm metrics tracking and trend analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from autodev.metrics import EXPERIMENTS_FILE, METRICS_FILE, MetricsTracker, SwarmMetrics


def _make_metrics(
	run_id: str = "run-1",
	timestamp: str = "2026-01-01T00:00:00",
	test_count: int = 50,
	test_pass_rate: float = 0.9,
	total_cost_usd: float = 1.0,
	cost_per_task: float = 0.1,
	agent_success_rate: float = 0.8,
	total_duration_s: float = 300.0,
	tasks_completed: int = 8,
	tasks_failed: int = 2,
) -> SwarmMetrics:
	return SwarmMetrics(
		run_id=run_id,
		timestamp=timestamp,
		test_count=test_count,
		test_pass_rate=test_pass_rate,
		total_cost_usd=total_cost_usd,
		cost_per_task=cost_per_task,
		agent_success_rate=agent_success_rate,
		total_duration_s=total_duration_s,
		tasks_completed=tasks_completed,
		tasks_failed=tasks_failed,
	)


@pytest.fixture
def tracker(tmp_path: Path) -> MetricsTracker:
	return MetricsTracker(tmp_path)


@pytest.fixture
def tsv_path(tmp_path: Path) -> Path:
	return tmp_path / METRICS_FILE


class TestSwarmMetrics:
	def test_dataclass_creation(self) -> None:
		m = _make_metrics()
		assert m.run_id == "run-1"
		assert m.test_count == 50
		assert m.test_pass_rate == 0.9
		assert m.tasks_completed == 8
		assert m.tasks_failed == 2

	def test_defaults_override(self) -> None:
		m = _make_metrics(run_id="custom", test_count=100)
		assert m.run_id == "custom"
		assert m.test_count == 100


class TestRecordRun:
	def test_creates_tsv_with_header(self, tracker: MetricsTracker, tsv_path: Path) -> None:
		tracker.record_run(_make_metrics())
		content = tsv_path.read_text()
		lines = content.strip().split("\n")
		assert len(lines) == 2
		assert lines[0].startswith("run_id\t")
		assert "run-1" in lines[1]

	def test_appends_without_duplicate_header(self, tracker: MetricsTracker, tsv_path: Path) -> None:
		tracker.record_run(_make_metrics(run_id="run-1"))
		tracker.record_run(_make_metrics(run_id="run-2"))
		content = tsv_path.read_text()
		lines = content.strip().split("\n")
		assert len(lines) == 3
		assert lines[0].startswith("run_id\t")
		assert "run-1" in lines[1]
		assert "run-2" in lines[2]

	def test_header_count_is_one(self, tracker: MetricsTracker, tsv_path: Path) -> None:
		for i in range(5):
			tracker.record_run(_make_metrics(run_id=f"run-{i}"))
		content = tsv_path.read_text()
		header_count = content.count("run_id\t")
		assert header_count == 1

	def test_roundtrip_values(self, tracker: MetricsTracker) -> None:
		original = _make_metrics(total_cost_usd=3.14159, agent_success_rate=0.667)
		tracker.record_run(original)
		rows = tracker._read_rows()
		assert len(rows) == 1
		assert rows[0].total_cost_usd == pytest.approx(3.14159)
		assert rows[0].agent_success_rate == pytest.approx(0.667)


class TestGetTrend:
	def test_insufficient_data_zero_rows(self, tracker: MetricsTracker) -> None:
		result = tracker.get_trend()
		assert result["error"] == "insufficient_data"
		assert result["rows"] == 0

	def test_insufficient_data_one_row(self, tracker: MetricsTracker) -> None:
		tracker.record_run(_make_metrics())
		result = tracker.get_trend()
		assert result["error"] == "insufficient_data"
		assert result["rows"] == 1

	def test_increasing_trend(self, tracker: MetricsTracker) -> None:
		for i in range(10):
			tracker.record_run(_make_metrics(
				run_id=f"run-{i}",
				test_count=50 + i * 10,
				total_cost_usd=1.0,
				agent_success_rate=0.5 + i * 0.05,
			))
		result = tracker.get_trend(last_n=10)
		assert result["test_count_trend"] == "increasing"
		assert result["success_rate_trend"] == "increasing"

	def test_decreasing_trend(self, tracker: MetricsTracker) -> None:
		for i in range(10):
			tracker.record_run(_make_metrics(
				run_id=f"run-{i}",
				test_count=100 - i * 10,
				total_cost_usd=5.0 - i * 0.4,
				agent_success_rate=0.9 - i * 0.08,
			))
		result = tracker.get_trend(last_n=10)
		assert result["test_count_trend"] == "decreasing"
		assert result["success_rate_trend"] == "decreasing"

	def test_stable_trend(self, tracker: MetricsTracker) -> None:
		for i in range(10):
			tracker.record_run(_make_metrics(
				run_id=f"run-{i}",
				test_count=50,
				total_cost_usd=1.0,
				agent_success_rate=0.8,
			))
		result = tracker.get_trend(last_n=10)
		assert result["test_count_trend"] == "stable"
		assert result["cost_trend"] == "stable"
		assert result["success_rate_trend"] == "stable"

	def test_improvement_velocity(self, tracker: MetricsTracker) -> None:
		for i in range(5):
			tracker.record_run(_make_metrics(run_id=f"run-{i}", test_count=10 + i * 5))
		result = tracker.get_trend(last_n=5)
		# (30 - 10) / 5 = 4.0
		assert result["improvement_velocity"] == pytest.approx(4.0)

	def test_best_and_worst_run(self, tracker: MetricsTracker) -> None:
		tracker.record_run(_make_metrics(run_id="bad", test_pass_rate=0.3))
		tracker.record_run(_make_metrics(run_id="good", test_pass_rate=0.99))
		tracker.record_run(_make_metrics(run_id="ok", test_pass_rate=0.7))
		result = tracker.get_trend(last_n=3)
		assert result["best_run"].run_id == "good"
		assert result["worst_run"].run_id == "bad"


class TestCorrelateWithModifications:
	def test_no_experiments_file(self, tracker: MetricsTracker) -> None:
		result = tracker.correlate_with_modifications()
		assert result == []

	def test_no_metrics(self, tracker: MetricsTracker, tmp_path: Path) -> None:
		exp_path = tmp_path / EXPERIMENTS_FILE
		exp_path.write_text(
			"commit\ttests_before\ttests_after\toutcome\tproposal_title\tduration_s\tcost_usd\ttimestamp\n"
			"abc123\t50\t55\tpassed\tAdd caching\t120.0\t0.50\t2026-01-02T00:00:00\n"
		)
		result = tracker.correlate_with_modifications()
		assert result == []

	def test_positive_correlation(self, tracker: MetricsTracker, tmp_path: Path) -> None:
		# Record metrics before and after experiment
		tracker.record_run(_make_metrics(
			run_id="before", timestamp="2026-01-01T00:00:00",
			test_count=50, agent_success_rate=0.7,
		))
		tracker.record_run(_make_metrics(
			run_id="after", timestamp="2026-01-03T00:00:00",
			test_count=60, agent_success_rate=0.8,
		))

		exp_path = tmp_path / EXPERIMENTS_FILE
		exp_path.write_text(
			"commit\ttests_before\ttests_after\toutcome\tproposal_title\tduration_s\tcost_usd\ttimestamp\n"
			"abc123\t50\t60\tpassed\tAdd caching\t120.0\t0.50\t2026-01-02T00:00:00\n"
		)

		result = tracker.correlate_with_modifications()
		assert len(result) == 1
		assert result[0]["proposal_title"] == "Add caching"
		assert result[0]["verdict"] == "positive"
		assert result[0]["metric_delta"]["test_count"] == 10

	def test_negative_correlation(self, tracker: MetricsTracker, tmp_path: Path) -> None:
		tracker.record_run(_make_metrics(
			run_id="before", timestamp="2026-01-01T00:00:00",
			test_count=60, agent_success_rate=0.9,
		))
		tracker.record_run(_make_metrics(
			run_id="after", timestamp="2026-01-03T00:00:00",
			test_count=50, agent_success_rate=0.5,
		))

		exp_path = tmp_path / EXPERIMENTS_FILE
		exp_path.write_text(
			"commit\ttests_before\ttests_after\toutcome\tproposal_title\tduration_s\tcost_usd\ttimestamp\n"
			"abc123\t60\t50\tfailed\tBad refactor\t200.0\t1.00\t2026-01-02T00:00:00\n"
		)

		result = tracker.correlate_with_modifications()
		assert len(result) == 1
		assert result[0]["verdict"] == "negative"
