"""Tiered graceful degradation state machine.

Replaces the ad-hoc _db_degraded flag with formal degradation levels
that control worker counts, merge behavior, and stop conditions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DegradationLevel(IntEnum):
	FULL_CAPACITY = 0
	REDUCED_WORKERS = 1
	DB_DEGRADED = 2
	MODEL_FALLBACK = 3
	READ_ONLY = 4
	SAFE_STOP = 5


@dataclass
class DegradationTransition:
	from_level: DegradationLevel
	to_level: DegradationLevel
	trigger: str
	timestamp: float = field(default_factory=time.monotonic)


class DegradationManager:
	"""State machine managing system degradation levels.

	Monitors DB errors, merge conflicts, budget usage, rate limits,
	and verification failures to escalate or recover degradation level.
	"""

	def __init__(
		self,
		config: Any = None,
		on_transition: Callable[[DegradationTransition], None] | None = None,
	) -> None:
		self._level = DegradationLevel.FULL_CAPACITY
		self._on_transition = on_transition
		self._transitions: list[DegradationTransition] = []

		# Config defaults (overridden by DegradationConfig if provided)
		self._budget_fraction_threshold: float = 0.8
		self._conflict_rate_threshold: float = 0.5
		self._reduced_worker_fraction: float = 0.5
		self._db_error_threshold: int = 5
		self._rate_limit_window_seconds: float = 60.0
		self._rate_limit_threshold: int = 3
		self._verification_failure_threshold: int = 3
		self._safe_stop_timeout_seconds: float = 300.0
		self._recovery_success_threshold: int = 3

		self._cost_per_merge_threshold: float = 5.0
		self._budget_pace_threshold: float = 1.5

		if config is not None:
			cfg = config
			self._budget_fraction_threshold = getattr(cfg, "budget_fraction_threshold", self._budget_fraction_threshold)
			self._conflict_rate_threshold = getattr(cfg, "conflict_rate_threshold", self._conflict_rate_threshold)
			self._reduced_worker_fraction = getattr(cfg, "reduced_worker_fraction", self._reduced_worker_fraction)
			self._db_error_threshold = getattr(cfg, "db_error_threshold", self._db_error_threshold)
			self._rate_limit_window_seconds = getattr(cfg, "rate_limit_window_seconds", self._rate_limit_window_seconds)
			self._rate_limit_threshold = getattr(cfg, "rate_limit_threshold", self._rate_limit_threshold)
			self._verification_failure_threshold = getattr(
				cfg, "verification_failure_threshold", self._verification_failure_threshold,
			)
			self._safe_stop_timeout_seconds = getattr(
				cfg, "safe_stop_timeout_seconds", self._safe_stop_timeout_seconds,
			)
			self._recovery_success_threshold = getattr(
				cfg, "recovery_success_threshold", self._recovery_success_threshold,
			)
			self._cost_per_merge_threshold = getattr(
				cfg, "cost_per_merge_threshold", self._cost_per_merge_threshold,
			)

		# Counters
		self._db_error_count: int = 0
		self._merge_attempts: int = 0
		self._merge_conflicts: int = 0
		self._rate_limit_timestamps: list[float] = []
		self._verification_failure_count: int = 0
		self._consecutive_successes: int = 0

		# Cost tracking
		self._unit_costs: list[tuple[str, float, bool]] = []  # (unit_id, cost_usd, merged)
		self._total_spent: float = 0.0
		self._total_budget: float = 0.0
		self._last_budget_pace: float = 0.0

	def _transition(self, new_level: DegradationLevel, trigger: str) -> None:
		if new_level == self._level:
			return
		old_level = self._level
		transition = DegradationTransition(
			from_level=old_level,
			to_level=new_level,
			trigger=trigger,
		)
		self._level = new_level
		self._transitions.append(transition)
		logger.warning(
			"Degradation transition: %s -> %s (trigger: %s)",
			old_level.name, new_level.name, trigger,
		)
		if self._on_transition:
			self._on_transition(transition)

	def _escalate_to(self, target: DegradationLevel, trigger: str) -> None:
		"""Escalate to target level only if it's worse than current."""
		if target > self._level:
			self._transition(target, trigger)

	# -- DB errors --

	def record_db_error(self) -> None:
		self._db_error_count += 1
		self._consecutive_successes = 0
		if self._db_error_count >= self._db_error_threshold:
			self._escalate_to(DegradationLevel.DB_DEGRADED, "db_errors")

	def record_db_success(self) -> None:
		if self._db_error_count > 0:
			self._db_error_count = max(0, self._db_error_count - 1)
		self._consecutive_successes += 1
		recovery = self._consecutive_successes >= self._recovery_success_threshold
		if self._level == DegradationLevel.DB_DEGRADED and recovery:
			self._transition(DegradationLevel.REDUCED_WORKERS, "db_recovery")
			self._db_error_count = 0

	# -- Merge conflicts --

	def record_merge_attempt(self, conflict: bool) -> None:
		self._merge_attempts += 1
		if conflict:
			self._merge_conflicts += 1
		if self._merge_attempts >= 4:
			rate = self._merge_conflicts / self._merge_attempts
			if rate > self._conflict_rate_threshold:
				self._escalate_to(DegradationLevel.REDUCED_WORKERS, "conflict_rate")

	# -- Budget --

	def check_budget_fraction(self, spent: float, budget: float) -> None:
		self._total_spent = spent
		self._total_budget = budget
		if budget <= 0:
			return
		fraction = spent / budget
		if fraction > self._budget_fraction_threshold:
			self._escalate_to(DegradationLevel.REDUCED_WORKERS, "budget_pressure")

	# -- Rate limits --

	def record_rate_limit(self) -> None:
		now = time.monotonic()
		self._rate_limit_timestamps.append(now)
		self._consecutive_successes = 0
		# Trim old entries outside window
		cutoff = now - self._rate_limit_window_seconds
		self._rate_limit_timestamps = [t for t in self._rate_limit_timestamps if t >= cutoff]
		if len(self._rate_limit_timestamps) >= self._rate_limit_threshold:
			self._escalate_to(DegradationLevel.MODEL_FALLBACK, "rate_limits")

	# -- Verification failures --

	def record_verification_failure(self) -> None:
		self._verification_failure_count += 1
		self._consecutive_successes = 0
		if self._verification_failure_count >= self._verification_failure_threshold:
			self._escalate_to(DegradationLevel.READ_ONLY, "verification_failures")

	def record_verification_success(self) -> None:
		if self._verification_failure_count > 0:
			self._verification_failure_count = max(0, self._verification_failure_count - 1)
		self._consecutive_successes += 1
		recovery = self._consecutive_successes >= self._recovery_success_threshold
		if self._level == DegradationLevel.READ_ONLY and recovery:
			self._transition(DegradationLevel.MODEL_FALLBACK, "verification_recovery")
			self._verification_failure_count = 0

	# -- Safe stop --

	def check_in_flight_drained(self, count: int) -> None:
		if self._level == DegradationLevel.READ_ONLY and count == 0:
			self._transition(DegradationLevel.SAFE_STOP, "in_flight_drained")

	# -- General recovery --

	def record_general_success(self) -> None:
		self._consecutive_successes += 1
		recovery = self._consecutive_successes >= self._recovery_success_threshold
		if self._level == DegradationLevel.REDUCED_WORKERS and recovery:
			self._transition(DegradationLevel.FULL_CAPACITY, "general_recovery")

	# -- Cost tracking --

	def record_unit_cost(self, unit_id: str, cost_usd: float, merged: bool) -> None:
		"""Track per-unit cost and whether it merged successfully."""
		self._unit_costs.append((unit_id, cost_usd, merged))
		if self.should_reduce_workers_for_cost():
			self._escalate_to(DegradationLevel.REDUCED_WORKERS, "cost_efficiency")

	def cost_per_merge(self) -> float | None:
		"""Average cost of successfully merged units. None if no merges yet."""
		merged = [cost for _, cost, m in self._unit_costs if m]
		if not merged:
			return None
		return sum(merged) / len(merged)

	def budget_pace(self, elapsed_seconds: float, wall_time_limit: float) -> float:
		"""Ratio of budget consumption rate to time consumption rate.

		Values >1.0 mean overspending relative to elapsed time.
		Returns 0.0 if wall_time_limit or total_budget is zero.
		"""
		if wall_time_limit <= 0 or self._total_budget <= 0 or elapsed_seconds <= 0:
			return 0.0
		budget_fraction = self._total_spent / self._total_budget
		time_fraction = elapsed_seconds / wall_time_limit
		pace = budget_fraction / time_fraction
		self._last_budget_pace = pace
		return pace

	def should_reduce_workers_for_cost(self) -> bool:
		"""True when cost_per_merge exceeds threshold OR budget pace > 1.5."""
		cpm = self.cost_per_merge()
		if cpm is not None and cpm > self._cost_per_merge_threshold:
			return True
		if self._last_budget_pace > self._budget_pace_threshold:
			return True
		return False

	# -- Worker count --

	def get_effective_worker_count(self, configured: int) -> int:
		if self._level >= DegradationLevel.READ_ONLY:
			return 0
		if self._level >= DegradationLevel.REDUCED_WORKERS:
			return max(1, int(configured * self._reduced_worker_fraction))
		return configured

	# -- Properties --

	@property
	def is_db_degraded(self) -> bool:
		return self._level >= DegradationLevel.DB_DEGRADED

	@property
	def is_read_only(self) -> bool:
		return self._level >= DegradationLevel.READ_ONLY

	@property
	def should_stop(self) -> bool:
		return self._level >= DegradationLevel.SAFE_STOP

	@property
	def level(self) -> DegradationLevel:
		return self._level

	@property
	def level_name(self) -> str:
		return self._level.name

	def get_status_dict(self) -> dict[str, Any]:
		return {
			"level": self._level.name,
			"level_value": int(self._level),
			"db_errors": self._db_error_count,
			"merge_attempts": self._merge_attempts,
			"merge_conflicts": self._merge_conflicts,
			"conflict_rate": (self._merge_conflicts / self._merge_attempts) if self._merge_attempts > 0 else 0.0,
			"rate_limit_count": len(self._rate_limit_timestamps),
			"verification_failures": self._verification_failure_count,
			"consecutive_successes": self._consecutive_successes,
			"cost_per_merge": self.cost_per_merge(),
			"unit_costs_recorded": len(self._unit_costs),
			"budget_pace": self._last_budget_pace,
		}
