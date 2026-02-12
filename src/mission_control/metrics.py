"""Metrics collection for mission-control rounds and sessions."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
	"""Metrics collected during a single round."""

	round_number: int = 0
	planning_duration_s: float = 0.0
	execution_duration_s: float = 0.0
	fixup_duration_s: float = 0.0
	evaluation_duration_s: float = 0.0
	total_duration_s: float = 0.0
	total_units: int = 0
	completed_units: int = 0
	failed_units: int = 0
	fixup_attempts: int = 0
	fixup_promoted: bool = False
	objective_score: float = 0.0

	@property
	def completion_rate(self) -> float:
		"""Fraction of units that completed successfully."""
		if self.total_units == 0:
			return 0.0
		return self.completed_units / self.total_units

	def to_dict(self) -> dict[str, object]:
		return {
			"round_number": self.round_number,
			"planning_duration_s": round(self.planning_duration_s, 2),
			"execution_duration_s": round(self.execution_duration_s, 2),
			"fixup_duration_s": round(self.fixup_duration_s, 2),
			"evaluation_duration_s": round(self.evaluation_duration_s, 2),
			"total_duration_s": round(self.total_duration_s, 2),
			"total_units": self.total_units,
			"completed_units": self.completed_units,
			"failed_units": self.failed_units,
			"completion_rate": round(self.completion_rate, 3),
			"fixup_attempts": self.fixup_attempts,
			"fixup_promoted": self.fixup_promoted,
			"objective_score": round(self.objective_score, 3),
		}


@dataclass
class MissionMetrics:
	"""Aggregate metrics for a full mission run."""

	total_rounds: int = 0
	total_duration_s: float = 0.0
	final_score: float = 0.0
	objective_met: bool = False
	round_metrics: list[RoundMetrics] = field(default_factory=list)

	@property
	def avg_round_duration_s(self) -> float:
		if not self.round_metrics:
			return 0.0
		return sum(r.total_duration_s for r in self.round_metrics) / len(self.round_metrics)

	@property
	def total_completed_units(self) -> int:
		return sum(r.completed_units for r in self.round_metrics)

	@property
	def total_failed_units(self) -> int:
		return sum(r.failed_units for r in self.round_metrics)

	def add_round(self, metrics: RoundMetrics) -> None:
		self.round_metrics.append(metrics)
		self.total_rounds = len(self.round_metrics)

	def to_dict(self) -> dict[str, object]:
		return {
			"total_rounds": self.total_rounds,
			"total_duration_s": round(self.total_duration_s, 2),
			"avg_round_duration_s": round(self.avg_round_duration_s, 2),
			"final_score": round(self.final_score, 3),
			"objective_met": self.objective_met,
			"total_completed_units": self.total_completed_units,
			"total_failed_units": self.total_failed_units,
			"rounds": [r.to_dict() for r in self.round_metrics],
		}

	def to_json(self) -> str:
		return json.dumps(self.to_dict(), indent=2)


class Timer:
	"""Context manager for timing operations."""

	def __init__(self) -> None:
		self._start: float = 0.0
		self.elapsed: float = 0.0

	def __enter__(self) -> "Timer":
		self._start = time.monotonic()
		return self

	def __exit__(self, *args: object) -> None:
		self.elapsed = time.monotonic() - self._start


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
	"""Configure logging with optional JSON output format.

	Args:
		level: Log level (DEBUG, INFO, WARNING, ERROR).
		json_format: If True, emit structured JSON log lines.
	"""
	root = logging.getLogger("mission_control")
	root.setLevel(getattr(logging, level.upper(), logging.INFO))

	if root.handlers:
		return

	handler = logging.StreamHandler()

	if json_format:
		handler.setFormatter(_JsonFormatter())
	else:
		handler.setFormatter(logging.Formatter(
			"%(asctime)s [%(levelname)s] %(name)s: %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S",
		))

	root.addHandler(handler)


class _JsonFormatter(logging.Formatter):
	"""Emit log records as JSON lines."""

	def format(self, record: logging.LogRecord) -> str:
		data = {
			"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
			"level": record.levelname,
			"logger": record.name,
			"msg": record.getMessage(),
		}
		if record.exc_info and record.exc_info[1]:
			data["exception"] = str(record.exc_info[1])
		return json.dumps(data)
