"""Tests for metrics collection and structured logging."""

from __future__ import annotations

import json
import logging

from mission_control.metrics import (
	MissionMetrics,
	RoundMetrics,
	Timer,
	_JsonFormatter,
	setup_logging,
)


class TestRoundMetrics:
	def test_defaults(self) -> None:
		m = RoundMetrics()
		assert m.round_number == 0
		assert m.total_duration_s == 0.0
		assert m.completion_rate == 0.0

	def test_completion_rate(self) -> None:
		m = RoundMetrics(total_units=10, completed_units=7, failed_units=3)
		assert m.completion_rate == 0.7

	def test_completion_rate_zero_units(self) -> None:
		m = RoundMetrics(total_units=0)
		assert m.completion_rate == 0.0

	def test_to_dict(self) -> None:
		m = RoundMetrics(
			round_number=1,
			planning_duration_s=1.234,
			execution_duration_s=10.567,
			total_duration_s=15.0,
			total_units=5,
			completed_units=4,
			failed_units=1,
			fixup_attempts=2,
			fixup_promoted=True,
			objective_score=0.85,
		)
		d = m.to_dict()
		assert d["round_number"] == 1
		assert d["planning_duration_s"] == 1.23
		assert d["completion_rate"] == 0.8
		assert d["fixup_promoted"] is True
		assert d["objective_score"] == 0.85


class TestMissionMetrics:
	def test_defaults(self) -> None:
		m = MissionMetrics()
		assert m.total_rounds == 0
		assert m.avg_round_duration_s == 0.0
		assert m.total_completed_units == 0
		assert m.total_failed_units == 0

	def test_add_round(self) -> None:
		m = MissionMetrics()
		r1 = RoundMetrics(round_number=1, total_duration_s=10.0, completed_units=3, failed_units=1)
		r2 = RoundMetrics(round_number=2, total_duration_s=20.0, completed_units=5, failed_units=0)
		m.add_round(r1)
		m.add_round(r2)
		assert m.total_rounds == 2
		assert m.avg_round_duration_s == 15.0
		assert m.total_completed_units == 8
		assert m.total_failed_units == 1

	def test_to_json(self) -> None:
		m = MissionMetrics(total_duration_s=30.0, final_score=0.9, objective_met=True)
		r = RoundMetrics(round_number=1, total_duration_s=30.0, total_units=3, completed_units=3)
		m.add_round(r)
		result = m.to_json()
		data = json.loads(result)
		assert data["total_rounds"] == 1
		assert data["final_score"] == 0.9
		assert data["objective_met"] is True
		assert len(data["rounds"]) == 1

	def test_to_dict_structure(self) -> None:
		m = MissionMetrics()
		d = m.to_dict()
		assert "total_rounds" in d
		assert "total_duration_s" in d
		assert "avg_round_duration_s" in d
		assert "rounds" in d
		assert isinstance(d["rounds"], list)


class TestTimer:
	def test_timer_records_elapsed(self) -> None:
		import time
		with Timer() as t:
			time.sleep(0.01)
		assert t.elapsed > 0.0
		assert t.elapsed < 1.0

	def test_timer_defaults(self) -> None:
		t = Timer()
		assert t.elapsed == 0.0


class TestJsonFormatter:
	def test_format_basic_record(self) -> None:
		formatter = _JsonFormatter()
		record = logging.LogRecord(
			name="test",
			level=logging.INFO,
			pathname="",
			lineno=0,
			msg="Hello %s",
			args=("world",),
			exc_info=None,
		)
		output = formatter.format(record)
		data = json.loads(output)
		assert data["level"] == "INFO"
		assert data["logger"] == "test"
		assert data["msg"] == "Hello world"
		assert "ts" in data


class TestSetupLogging:
	def test_setup_creates_handler(self) -> None:
		# Remove existing handlers
		root = logging.getLogger("mission_control")
		root.handlers.clear()

		setup_logging(level="DEBUG")

		assert len(root.handlers) == 1
		assert root.level == logging.DEBUG

		# Cleanup
		root.handlers.clear()

	def test_setup_json_format(self) -> None:
		root = logging.getLogger("mission_control")
		root.handlers.clear()

		setup_logging(level="INFO", json_format=True)

		assert len(root.handlers) == 1
		assert isinstance(root.handlers[0].formatter, _JsonFormatter)

		# Cleanup
		root.handlers.clear()

	def test_setup_idempotent(self) -> None:
		"""Calling setup_logging twice doesn't add duplicate handlers."""
		root = logging.getLogger("mission_control")
		root.handlers.clear()

		setup_logging(level="INFO")
		setup_logging(level="INFO")

		assert len(root.handlers) == 1

		# Cleanup
		root.handlers.clear()
