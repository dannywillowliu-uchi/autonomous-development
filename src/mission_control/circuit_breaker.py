"""Per-workspace circuit breaker for failure isolation.

Tracks failure rates per workspace and prevents degraded workspaces
from consuming budget. States follow the standard circuit breaker
pattern: CLOSED -> OPEN -> HALF_OPEN -> CLOSED/OPEN.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
	CLOSED = "closed"
	OPEN = "open"
	HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
	"""Circuit breaker for a single workspace."""

	workspace_id: str
	_state: CircuitBreakerState = CircuitBreakerState.CLOSED
	failure_count: int = 0
	success_count: int = 0
	last_failure_at: float = 0.0
	opened_at: float = 0.0
	max_failures: int = 3
	cooldown_seconds: float = 120.0
	_original_cooldown: float = 0.0
	half_open_max_probes: int = 1
	_half_open_probes: int = 0

	def __post_init__(self) -> None:
		if self._original_cooldown == 0.0:
			self._original_cooldown = self.cooldown_seconds

	@property
	def state(self) -> str:
		"""Return current state as 'closed', 'open', or 'half_open'."""
		return self._state.value


class CircuitBreakerManager:
	"""Manages circuit breakers for multiple workspaces."""

	def __init__(
		self,
		max_failures: int = 3,
		cooldown_seconds: float = 120.0,
		on_state_change: Callable[[str, str, str], None] | None = None,
		auto_recovery_enabled: bool = True,
	) -> None:
		self._max_failures = max_failures
		self._cooldown_seconds = cooldown_seconds
		self._breakers: dict[str, CircuitBreaker] = {}
		self._on_state_change = on_state_change
		self.auto_recovery_enabled = auto_recovery_enabled

	def _get_or_create(self, workspace_id: str) -> CircuitBreaker:
		if workspace_id not in self._breakers:
			self._breakers[workspace_id] = CircuitBreaker(
				workspace_id=workspace_id,
				max_failures=self._max_failures,
				cooldown_seconds=self._cooldown_seconds,
			)
		return self._breakers[workspace_id]

	def record_success(self, workspace_id: str) -> None:
		"""Record a successful unit execution for this workspace."""
		cb = self._get_or_create(workspace_id)
		cb.success_count += 1
		cb.failure_count = 0

		if cb._state == CircuitBreakerState.HALF_OPEN:
			logger.info(
				"Circuit breaker %s: HALF_OPEN -> CLOSED (success)",
				workspace_id,
			)
			cb._state = CircuitBreakerState.CLOSED
			cb._half_open_probes = 0
			cb.cooldown_seconds = cb._original_cooldown
			if self._on_state_change:
				self._on_state_change(workspace_id, "half_open", "closed")

	def record_failure(self, workspace_id: str) -> None:
		"""Record a failed unit execution for this workspace."""
		cb = self._get_or_create(workspace_id)
		cb.failure_count += 1
		cb.last_failure_at = time.monotonic()

		if cb._state == CircuitBreakerState.HALF_OPEN:
			# Progressive backoff: double cooldown, cap at 10x original
			cb.cooldown_seconds = min(
				cb.cooldown_seconds * 2,
				cb._original_cooldown * 10,
			)
			logger.warning(
				"Circuit breaker %s: HALF_OPEN -> OPEN (probe failed, cooldown=%.0fs)",
				workspace_id, cb.cooldown_seconds,
			)
			cb._state = CircuitBreakerState.OPEN
			cb.opened_at = time.monotonic()
			cb._half_open_probes = 0
			if self._on_state_change:
				self._on_state_change(workspace_id, "half_open", "open")
		elif cb._state == CircuitBreakerState.CLOSED and cb.failure_count >= cb.max_failures:
			logger.warning(
				"Circuit breaker %s: CLOSED -> OPEN (%d consecutive failures)",
				workspace_id, cb.failure_count,
			)
			cb._state = CircuitBreakerState.OPEN
			cb.opened_at = time.monotonic()
			if self._on_state_change:
				self._on_state_change(workspace_id, "closed", "open")

	def can_dispatch(self, workspace_id: str) -> bool:
		"""Check if a workspace is available for dispatch.

		Returns True for CLOSED, checks cooldown for OPEN (transitioning
		to HALF_OPEN if expired and auto_recovery_enabled), and allows
		one probe for HALF_OPEN.
		"""
		cb = self._get_or_create(workspace_id)

		if cb._state == CircuitBreakerState.CLOSED:
			return True

		if cb._state == CircuitBreakerState.OPEN:
			if not self.auto_recovery_enabled:
				return False
			elapsed = time.monotonic() - cb.opened_at
			if elapsed >= cb.cooldown_seconds:
				logger.info(
					"Circuit breaker %s: OPEN -> HALF_OPEN (cooldown expired after %.0fs)",
					workspace_id, elapsed,
				)
				cb._state = CircuitBreakerState.HALF_OPEN
				cb._half_open_probes = 0
				if self._on_state_change:
					self._on_state_change(workspace_id, "open", "half_open")
				# Allow this probe
				cb._half_open_probes += 1
				return True
			return False

		# HALF_OPEN: allow up to max_probes
		if cb._half_open_probes < cb.half_open_max_probes:
			cb._half_open_probes += 1
			return True
		return False

	def all_open(self) -> bool:
		"""Return True when ALL tracked workspaces are OPEN (stall condition)."""
		if not self._breakers:
			return False
		return all(
			cb._state == CircuitBreakerState.OPEN
			for cb in self._breakers.values()
		)

	def get_state(self, workspace_id: str) -> CircuitBreakerState:
		"""Get the current state enum for a workspace."""
		cb = self._get_or_create(workspace_id)
		return cb._state

	def reset(self, workspace_id: str) -> None:
		"""Force a workspace back to CLOSED state."""
		cb = self._get_or_create(workspace_id)
		cb._state = CircuitBreakerState.CLOSED
		cb.failure_count = 0
		cb._half_open_probes = 0
		cb.cooldown_seconds = cb._original_cooldown
		logger.info("Circuit breaker %s: force reset to CLOSED", workspace_id)

	def get_summary(self) -> dict[str, int]:
		"""Return counts per circuit breaker state."""
		counts: dict[str, int] = {"closed": 0, "open": 0, "half_open": 0}
		for cb in self._breakers.values():
			counts[cb._state.value] += 1
		counts["total"] = len(self._breakers)
		return counts

	def get_open_workspaces(self) -> dict[str, int]:
		"""Return mapping of OPEN workspace IDs to their failure counts."""
		return {
			cb.workspace_id: cb.failure_count
			for cb in self._breakers.values()
			if cb._state == CircuitBreakerState.OPEN
		}
