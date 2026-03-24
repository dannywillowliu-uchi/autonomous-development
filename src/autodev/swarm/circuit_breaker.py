"""Swarm-level circuit breaker for failure cascade prevention.

Implements the standard circuit breaker pattern (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
with both global and per-task failure tracking. When tripped, blocks new agent spawns
and optionally notifies the planner via a callback.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SwarmCircuitBreakerState(Enum):
	CLOSED = "closed"
	OPEN = "open"
	HALF_OPEN = "half_open"


@dataclass
class SwarmCircuitBreakerConfig:
	max_consecutive_failures: int = 3
	cooldown_seconds: float = 180.0
	max_failures_per_window: int = 5
	window_seconds: float = 300.0
	half_open_max_probes: int = 1


class SwarmCircuitBreaker:
	"""Global swarm circuit breaker with per-task tracking and half-open probing.

	State machine:
	  CLOSED  -- normal operation, spawns allowed
	  OPEN    -- tripped, spawns blocked until cooldown expires
	  HALF_OPEN -- cooldown expired, one probe spawn allowed to test recovery
	"""

	def __init__(
		self,
		config: SwarmCircuitBreakerConfig | None = None,
		on_trip: Callable[[str], None] | None = None,
	) -> None:
		self._config = config or SwarmCircuitBreakerConfig()
		self._state = SwarmCircuitBreakerState.CLOSED
		self._consecutive_failures = 0
		self._failure_timestamps: list[float] = []
		self._tripped_at: float = 0.0
		self._trip_reason: str = ""
		self._half_open_probes: int = 0
		self._on_trip = on_trip
		self._task_failures: dict[str, int] = {}

	def record_success(self, task_id: str | None = None) -> None:
		"""Record a successful agent completion. Resets consecutive counter.

		In HALF_OPEN state, a success transitions back to CLOSED.
		"""
		self._consecutive_failures = 0
		if task_id:
			self._task_failures.pop(task_id, None)
		if self._state == SwarmCircuitBreakerState.HALF_OPEN:
			logger.info("Swarm circuit breaker: HALF_OPEN -> CLOSED (probe succeeded)")
			self._state = SwarmCircuitBreakerState.CLOSED
			self._trip_reason = ""
			self._half_open_probes = 0

	def record_failure(self, task_id: str | None = None) -> None:
		"""Record an agent failure. May trip the breaker.

		In HALF_OPEN state, a failure transitions back to OPEN.
		"""
		now = time.monotonic()
		self._consecutive_failures += 1
		self._failure_timestamps.append(now)
		cutoff = now - self._config.window_seconds
		self._failure_timestamps = [t for t in self._failure_timestamps if t > cutoff]

		if task_id:
			self._task_failures[task_id] = self._task_failures.get(task_id, 0) + 1

		if self._state == SwarmCircuitBreakerState.HALF_OPEN:
			self._half_open_probes = 0
			self._trip(f"probe failed (consecutive: {self._consecutive_failures})")
		elif self._consecutive_failures >= self._config.max_consecutive_failures:
			self._trip(f"{self._consecutive_failures} consecutive agent failures")
		elif len(self._failure_timestamps) >= self._config.max_failures_per_window:
			self._trip(f"{len(self._failure_timestamps)} failures in {self._config.window_seconds:.0f}s window")

	def can_spawn(self) -> tuple[bool, str]:
		"""Check whether a new agent spawn is allowed.

		Returns (allowed, reason). Reason is empty when allowed.
		Transitions OPEN -> HALF_OPEN when cooldown expires.
		"""
		if self._state == SwarmCircuitBreakerState.CLOSED:
			return True, ""

		if self._state == SwarmCircuitBreakerState.OPEN:
			elapsed = time.monotonic() - self._tripped_at
			if elapsed >= self._config.cooldown_seconds:
				logger.info("Swarm circuit breaker: OPEN -> HALF_OPEN (cooldown expired)")
				self._state = SwarmCircuitBreakerState.HALF_OPEN
				self._half_open_probes = 0
				self._consecutive_failures = 0
				self._trip_reason = ""
				self._half_open_probes += 1
				return True, ""
			remaining = self._config.cooldown_seconds - elapsed
			return False, f"Circuit breaker tripped ({self._trip_reason}), {remaining:.0f}s remaining"

		# HALF_OPEN: allow up to max_probes
		if self._half_open_probes < self._config.half_open_max_probes:
			self._half_open_probes += 1
			return True, ""
		return False, "Circuit breaker half-open, probe in flight"

	def _trip(self, reason: str) -> None:
		was_open = self._state == SwarmCircuitBreakerState.OPEN
		self._state = SwarmCircuitBreakerState.OPEN
		self._tripped_at = time.monotonic()
		self._trip_reason = reason
		if not was_open:
			logger.warning("Swarm circuit breaker TRIPPED: %s", reason)
			if self._on_trip:
				self._on_trip(reason)

	@property
	def is_tripped(self) -> bool:
		"""True when in OPEN state (fully blocking spawns)."""
		return self._state == SwarmCircuitBreakerState.OPEN

	@property
	def state(self) -> SwarmCircuitBreakerState:
		return self._state

	@property
	def trip_reason(self) -> str:
		return self._trip_reason

	@property
	def consecutive_failures(self) -> int:
		return self._consecutive_failures

	def task_failure_count(self, task_id: str) -> int:
		"""Return consecutive failure count for a specific task."""
		return self._task_failures.get(task_id, 0)

	def reset(self) -> None:
		"""Force-reset the breaker to CLOSED state."""
		logger.info("Swarm circuit breaker: force reset to CLOSED")
		self._state = SwarmCircuitBreakerState.CLOSED
		self._consecutive_failures = 0
		self._failure_timestamps.clear()
		self._tripped_at = 0.0
		self._trip_reason = ""
		self._half_open_probes = 0
		self._task_failures.clear()

	def get_summary(self) -> dict[str, Any]:
		"""Return a snapshot of the breaker state for planner context."""
		return {
			"state": self._state.value,
			"consecutive_failures": self._consecutive_failures,
			"trip_reason": self._trip_reason,
			"task_failures": dict(self._task_failures),
		}
