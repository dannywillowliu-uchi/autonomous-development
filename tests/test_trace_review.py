"""Tests for trace_review module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from autodev.trace_review import (
	HistoryAnalysis,
	RunAnalysis,
	TraceAnalyzer,
	_compute_file_hotspots,
	_extract_errors,
	_find_wasted_agents,
)


def _make_trace(
	agent_name: str = "agent-1",
	run_id: str = "run-1",
	exit_code: int = 0,
	cost_usd: float = 0.5,
	duration_s: float = 120.0,
	files_changed: list[str] | None = None,
	output_tail: str = "",
	task_title: str = "test task",
) -> dict[str, Any]:
	return {
		"id": f"id-{agent_name}",
		"run_id": run_id,
		"agent_name": agent_name,
		"agent_id": f"aid-{agent_name}",
		"task_id": f"tid-{agent_name}",
		"task_title": task_title,
		"started_at": "2026-03-12T00:00:00",
		"ended_at": "2026-03-12T00:02:00",
		"duration_s": duration_s,
		"exit_code": exit_code,
		"cost_usd": cost_usd,
		"files_changed": files_changed or [],
		"trace_path": "",
		"output_tail": output_tail,
	}


def _mock_db(traces_by_run: dict[str, list[dict]] | None = None) -> MagicMock:
	db = MagicMock()
	traces_by_run = traces_by_run or {}

	def get_traces(run_id: str | None = None, limit: int = 50) -> list[dict]:
		if run_id:
			return traces_by_run.get(run_id, [])
		all_traces: list[dict] = []
		for traces in traces_by_run.values():
			all_traces.extend(traces)
		return all_traces[:limit]

	db.get_agent_traces = get_traces
	return db


@pytest.mark.asyncio
async def test_analyze_run_empty():
	db = _mock_db()
	analyzer = TraceAnalyzer(db, Path("/tmp/project"))
	result = await analyzer.analyze_run("nonexistent")

	assert result.run_id == "nonexistent"
	assert result.total_agents == 0
	assert result.success_rate == 0.0
	assert result.total_cost_usd == 0.0
	assert result.total_duration_s == 0.0
	assert result.file_hotspots == []
	assert result.error_patterns == []
	assert result.wasted_agents == []
	assert result.recommendations == []


@pytest.mark.asyncio
async def test_analyze_run_with_traces():
	traces = [
		_make_trace("agent-1", exit_code=0, cost_usd=1.0, duration_s=60.0,
			files_changed=["src/a.py"]),
		_make_trace("agent-2", exit_code=0, cost_usd=2.0, duration_s=120.0,
			files_changed=["src/b.py"]),
		_make_trace("agent-3", exit_code=1, cost_usd=0.5, duration_s=30.0,
			output_tail="Error: something broke"),
	]
	db = _mock_db({"run-1": traces})
	analyzer = TraceAnalyzer(db, Path("/tmp/project"))
	result = await analyzer.analyze_run("run-1")

	assert result.total_agents == 3
	assert abs(result.success_rate - 2.0 / 3.0) < 0.01
	assert result.total_cost_usd == 3.5
	assert result.total_duration_s == 210.0


@pytest.mark.asyncio
async def test_file_hotspot_detection():
	traces = [
		_make_trace("agent-1", files_changed=["src/shared.py", "src/a.py"]),
		_make_trace("agent-2", files_changed=["src/shared.py", "src/b.py"]),
		_make_trace("agent-3", files_changed=["src/shared.py"]),
	]
	db = _mock_db({"run-1": traces})
	analyzer = TraceAnalyzer(db, Path("/tmp/project"))
	result = await analyzer.analyze_run("run-1")

	assert len(result.file_hotspots) >= 1
	hotspot_files = {f for f, _ in result.file_hotspots}
	assert "src/shared.py" in hotspot_files
	hotspot_count = dict(result.file_hotspots)
	assert hotspot_count["src/shared.py"] == 3
	assert any("src/shared.py" in r and "3 agents" in r for r in result.recommendations)


@pytest.mark.asyncio
async def test_error_pattern_extraction():
	traces = [
		_make_trace("agent-1", exit_code=1,
			output_tail="Traceback (most recent call last):\n  File 'x.py'\nImportError: no module"),
		_make_trace("agent-2", exit_code=1,
			output_tail="FAILED tests/test_foo.py::test_bar\nError: assertion failed"),
		_make_trace("agent-3", exit_code=1,
			output_tail="Traceback (most recent call last):\n  SyntaxError: bad syntax"),
	]
	db = _mock_db({"run-1": traces})
	analyzer = TraceAnalyzer(db, Path("/tmp/project"))
	result = await analyzer.analyze_run("run-1")

	pattern_names = {p for p, _ in result.error_patterns}
	assert "Traceback" in pattern_names
	pattern_counts = dict(result.error_patterns)
	assert pattern_counts["Traceback"] == 2


@pytest.mark.asyncio
async def test_wasted_agent_detection():
	traces = [
		_make_trace("good-agent", exit_code=0, files_changed=["src/a.py"]),
		_make_trace("wasted-agent", exit_code=1, files_changed=[]),
		_make_trace("partial-agent", exit_code=1, files_changed=["src/b.py"]),
	]
	db = _mock_db({"run-1": traces})
	analyzer = TraceAnalyzer(db, Path("/tmp/project"))
	result = await analyzer.analyze_run("run-1")

	assert "wasted-agent" in result.wasted_agents
	assert "good-agent" not in result.wasted_agents
	assert "partial-agent" not in result.wasted_agents


@pytest.mark.asyncio
async def test_generate_report_run():
	analysis = RunAnalysis(
		run_id="run-1",
		total_agents=5,
		success_rate=0.8,
		total_cost_usd=10.5,
		total_duration_s=600.0,
		file_hotspots=[("src/shared.py", 3)],
		error_patterns=[("Traceback", 2)],
		wasted_agents=["agent-4"],
		recommendations=["File src/shared.py was touched by 3 agents -- serialize work on it"],
	)
	db = _mock_db()
	analyzer = TraceAnalyzer(db, Path("/tmp/project"))
	report = await analyzer.generate_report(analysis)

	assert "# Trace Review: run-1" in report
	assert "80%" in report
	assert "$10.50" in report
	assert "src/shared.py" in report
	assert "Traceback" in report
	assert "agent-4" in report
	assert "Recommendations" in report


@pytest.mark.asyncio
async def test_generate_report_history():
	analysis = HistoryAnalysis(
		runs_analyzed=3,
		overall_success_rate=0.75,
		cost_trend=[5.0, 8.0, 6.0],
		recurring_failures=[("Traceback", 10)],
		improvement_velocity=0.1,
		top_recommendations=["Reduce parallel agents on shared files"],
	)
	db = _mock_db()
	analyzer = TraceAnalyzer(db, Path("/tmp/project"))
	report = await analyzer.generate_report(analysis)

	assert "# Trace Review: History" in report
	assert "75%" in report
	assert "+10%" in report
	assert "Traceback" in report
	assert "Reduce parallel agents" in report


@pytest.mark.asyncio
async def test_analyze_history():
	run1_traces = [
		_make_trace("a1", run_id="run-1", exit_code=0, cost_usd=2.0),
		_make_trace("a2", run_id="run-1", exit_code=1, cost_usd=1.0),
	]
	run2_traces = [
		_make_trace("a3", run_id="run-2", exit_code=0, cost_usd=3.0),
		_make_trace("a4", run_id="run-2", exit_code=0, cost_usd=2.0),
	]
	db = _mock_db({"run-1": run1_traces, "run-2": run2_traces})
	analyzer = TraceAnalyzer(db, Path("/tmp/project"))
	result = await analyzer.analyze_history(last_n_runs=10)

	assert result.runs_analyzed == 2
	assert result.overall_success_rate > 0.0
	assert len(result.cost_trend) == 2


def test_extract_errors_basic():
	text = "Traceback (most recent call last):\n  File 'x.py'\nImportError: no module"
	errors = _extract_errors(text)
	assert "Traceback" in errors
	assert "ImportError" in errors


def test_compute_file_hotspots():
	traces = [
		{"files_changed": ["a.py", "b.py"]},
		{"files_changed": ["a.py", "c.py"]},
		{"files_changed": ["d.py"]},
	]
	hotspots = _compute_file_hotspots(traces)
	hotspot_dict = dict(hotspots)
	assert hotspot_dict["a.py"] == 2
	assert "d.py" not in hotspot_dict


def test_find_wasted_agents():
	traces = [
		{"agent_name": "good", "exit_code": 0, "files_changed": ["a.py"]},
		{"agent_name": "wasted", "exit_code": 1, "files_changed": []},
		{"agent_name": "ok-fail", "exit_code": 1, "files_changed": ["b.py"]},
	]
	wasted = _find_wasted_agents(traces)
	assert wasted == ["wasted"]


@pytest.mark.asyncio
async def test_llm_review_traces():
	traces = [
		_make_trace("agent-1", output_tail="some output here", task_title="fix bug"),
	]
	db = _mock_db({"run-1": traces})
	analyzer = TraceAnalyzer(db, Path("/tmp/project"))
	prompt = await analyzer.llm_review_traces("run-1")

	assert "agent-1" in prompt
	assert "fix bug" in prompt
	assert "coordination failures" in prompt


@pytest.mark.asyncio
async def test_llm_review_traces_empty():
	db = _mock_db()
	analyzer = TraceAnalyzer(db, Path("/tmp/project"))
	result = await analyzer.llm_review_traces("nonexistent")
	assert "No traces found" in result
