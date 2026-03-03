"""Tests for per-workspace circuit breaker."""

from __future__ import annotations

import time

from mission_control.circuit_breaker import (
	CircuitBreaker,
	CircuitBreakerManager,
	CircuitBreakerState,
)


class TestCircuitBreakerState:
	def test_closed_to_open_at_threshold(self) -> None:
		"""CLOSED -> OPEN when failure count reaches max_failures."""
		mgr = CircuitBreakerManager(max_failures=3, cooldown_seconds=60)
		ws = "/tmp/ws-1"
		assert mgr.get_state(ws) == CircuitBreakerState.CLOSED
		mgr.record_failure(ws)
		mgr.record_failure(ws)
		assert mgr.get_state(ws) == CircuitBreakerState.CLOSED
		mgr.record_failure(ws)
		assert mgr.get_state(ws) == CircuitBreakerState.OPEN

	def test_open_to_half_open_after_cooldown(self) -> None:
		"""OPEN -> HALF_OPEN when cooldown expires (via can_dispatch)."""
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=0.01)
		ws = "/tmp/ws-1"
		mgr.record_failure(ws)
		assert mgr.get_state(ws) == CircuitBreakerState.OPEN
		time.sleep(0.02)
		assert mgr.can_dispatch(ws) is True
		assert mgr.get_state(ws) == CircuitBreakerState.HALF_OPEN

	def test_half_open_to_closed_on_success(self) -> None:
		"""HALF_OPEN -> CLOSED on success."""
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=0.01)
		ws = "/tmp/ws-1"
		mgr.record_failure(ws)
		time.sleep(0.02)
		mgr.can_dispatch(ws)  # transition to HALF_OPEN
		mgr.record_success(ws)
		assert mgr.get_state(ws) == CircuitBreakerState.CLOSED

	def test_half_open_to_open_on_failure(self) -> None:
		"""HALF_OPEN -> OPEN on failure."""
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=0.01)
		ws = "/tmp/ws-1"
		mgr.record_failure(ws)
		time.sleep(0.02)
		mgr.can_dispatch(ws)  # transition to HALF_OPEN
		mgr.record_failure(ws)
		assert mgr.get_state(ws) == CircuitBreakerState.OPEN

	def test_half_open_failure_doubles_cooldown(self) -> None:
		"""HALF_OPEN -> OPEN doubles the cooldown (progressive backoff)."""
		original_cooldown = 0.01
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=original_cooldown)
		ws = "/tmp/ws-1"

		# Trip to OPEN
		mgr.record_failure(ws)
		time.sleep(0.02)
		mgr.can_dispatch(ws)  # -> HALF_OPEN

		# Probe fails -> OPEN with doubled cooldown
		mgr.record_failure(ws)
		cb = mgr._breakers[ws]
		assert cb.cooldown_seconds == original_cooldown * 2

	def test_cooldown_cap_at_10x(self) -> None:
		"""Progressive backoff caps at 10x original cooldown."""
		original_cooldown = 0.01
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=original_cooldown)
		ws = "/tmp/ws-1"

		# Cycle through HALF_OPEN failures to escalate backoff past 10x
		for _ in range(20):
			mgr.record_failure(ws)
			cb = mgr._breakers[ws]
			# Force cooldown expiry for next cycle
			cb.opened_at = time.monotonic() - cb.cooldown_seconds - 1
			mgr.can_dispatch(ws)  # -> HALF_OPEN

		# After the last failure, cooldown should be capped
		mgr.record_failure(ws)
		cb = mgr._breakers[ws]
		assert cb.cooldown_seconds == original_cooldown * 10

	def test_success_resets_cooldown(self) -> None:
		"""HALF_OPEN -> CLOSED resets cooldown to original value."""
		original_cooldown = 0.01
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=original_cooldown)
		ws = "/tmp/ws-1"

		# Trip and fail probe to double cooldown
		mgr.record_failure(ws)
		time.sleep(0.02)
		mgr.can_dispatch(ws)  # -> HALF_OPEN
		mgr.record_failure(ws)  # -> OPEN, cooldown doubled

		cb = mgr._breakers[ws]
		assert cb.cooldown_seconds == original_cooldown * 2

		# Now succeed on next probe
		cb.opened_at = time.monotonic() - cb.cooldown_seconds - 1
		mgr.can_dispatch(ws)  # -> HALF_OPEN
		mgr.record_success(ws)  # -> CLOSED

		assert mgr.get_state(ws) == CircuitBreakerState.CLOSED
		assert cb.cooldown_seconds == original_cooldown

	def test_success_resets_failure_count(self) -> None:
		"""Success resets failure counter so threshold resets."""
		mgr = CircuitBreakerManager(max_failures=3, cooldown_seconds=60)
		ws = "/tmp/ws-1"
		mgr.record_failure(ws)
		mgr.record_failure(ws)
		mgr.record_success(ws)
		# Counter reset -- need 3 more failures to trip
		mgr.record_failure(ws)
		mgr.record_failure(ws)
		assert mgr.get_state(ws) == CircuitBreakerState.CLOSED
		mgr.record_failure(ws)
		assert mgr.get_state(ws) == CircuitBreakerState.OPEN


class TestCircuitBreakerManager:
	def test_multi_workspace_tracking(self) -> None:
		"""Separate breakers per workspace."""
		mgr = CircuitBreakerManager(max_failures=2, cooldown_seconds=60)
		mgr.record_failure("/ws/a")
		mgr.record_failure("/ws/a")
		assert mgr.get_state("/ws/a") == CircuitBreakerState.OPEN
		assert mgr.get_state("/ws/b") == CircuitBreakerState.CLOSED

	def test_all_open_detection(self) -> None:
		"""all_open returns True only when ALL tracked workspaces are OPEN."""
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=60)
		mgr.record_failure("/ws/a")
		assert mgr.all_open() is True  # only one workspace, and it's open
		# Add a healthy workspace
		mgr.record_success("/ws/b")
		assert mgr.all_open() is False  # /ws/b is CLOSED
		mgr.record_failure("/ws/b")
		assert mgr.all_open() is True  # both open now

	def test_all_open_empty(self) -> None:
		"""all_open returns False when no workspaces are tracked."""
		mgr = CircuitBreakerManager()
		assert mgr.all_open() is False

	def test_reset(self) -> None:
		"""reset() forces workspace back to CLOSED."""
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=60)
		mgr.record_failure("/ws/a")
		assert mgr.get_state("/ws/a") == CircuitBreakerState.OPEN
		mgr.reset("/ws/a")
		assert mgr.get_state("/ws/a") == CircuitBreakerState.CLOSED

	def test_reset_restores_original_cooldown(self) -> None:
		"""reset() restores cooldown to original value."""
		original = 0.01
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=original)
		ws = "/ws/a"
		# Trip and fail probe to escalate cooldown
		mgr.record_failure(ws)
		cb = mgr._breakers[ws]
		cb.opened_at = time.monotonic() - original - 1
		mgr.can_dispatch(ws)
		mgr.record_failure(ws)
		assert cb.cooldown_seconds == original * 2

		mgr.reset(ws)
		assert cb.cooldown_seconds == original

	def test_get_summary_empty(self) -> None:
		"""get_summary returns all zeros when no workspaces tracked."""
		mgr = CircuitBreakerManager()
		assert mgr.get_summary() == {"closed": 0, "open": 0, "half_open": 0, "total": 0}

	def test_get_summary_mixed_states(self) -> None:
		"""get_summary counts breakers in each state."""
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=0.01)
		# /ws/a -> CLOSED (success keeps it closed)
		mgr.record_success("/ws/a")
		# /ws/b -> OPEN (one failure trips it)
		mgr.record_failure("/ws/b")
		# /ws/c -> HALF_OPEN (fail then cooldown)
		mgr.record_failure("/ws/c")
		time.sleep(0.02)
		mgr.can_dispatch("/ws/c")  # transitions OPEN -> HALF_OPEN
		summary = mgr.get_summary()
		assert summary == {"closed": 1, "open": 1, "half_open": 1, "total": 3}

	def test_get_open_workspaces(self) -> None:
		"""get_open_workspaces returns OPEN workspaces with failure counts."""
		mgr = CircuitBreakerManager(max_failures=2, cooldown_seconds=60)
		mgr.record_failure("/ws/a")
		mgr.record_failure("/ws/a")
		mgr.record_success("/ws/b")
		result = mgr.get_open_workspaces()
		assert "/ws/a" in result
		assert "/ws/b" not in result

	def test_auto_recovery_enabled_default_true(self) -> None:
		"""auto_recovery_enabled defaults to True."""
		mgr = CircuitBreakerManager()
		assert mgr.auto_recovery_enabled is True

	def test_auto_recovery_disabled_blocks_half_open(self) -> None:
		"""When auto_recovery_enabled=False, OPEN never transitions to HALF_OPEN."""
		mgr = CircuitBreakerManager(
			max_failures=1, cooldown_seconds=0.01, auto_recovery_enabled=False,
		)
		ws = "/tmp/ws-1"
		mgr.record_failure(ws)
		assert mgr.get_state(ws) == CircuitBreakerState.OPEN
		time.sleep(0.02)
		# Cooldown expired, but auto_recovery disabled
		assert mgr.can_dispatch(ws) is False
		assert mgr.get_state(ws) == CircuitBreakerState.OPEN

	def test_auto_recovery_toggle_at_runtime(self) -> None:
		"""auto_recovery_enabled can be toggled at runtime."""
		mgr = CircuitBreakerManager(
			max_failures=1, cooldown_seconds=0.01, auto_recovery_enabled=False,
		)
		ws = "/tmp/ws-1"
		mgr.record_failure(ws)
		time.sleep(0.02)
		assert mgr.can_dispatch(ws) is False  # blocked

		mgr.auto_recovery_enabled = True
		assert mgr.can_dispatch(ws) is True  # now allowed
		assert mgr.get_state(ws) == CircuitBreakerState.HALF_OPEN


class TestCanDispatch:
	def test_closed_returns_true(self) -> None:
		mgr = CircuitBreakerManager(max_failures=3, cooldown_seconds=60)
		assert mgr.can_dispatch("/ws/a") is True

	def test_open_within_cooldown_returns_false(self) -> None:
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=9999)
		mgr.record_failure("/ws/a")
		assert mgr.can_dispatch("/ws/a") is False

	def test_open_past_cooldown_returns_true(self) -> None:
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=0.01)
		mgr.record_failure("/ws/a")
		time.sleep(0.02)
		assert mgr.can_dispatch("/ws/a") is True

	def test_half_open_probe_limit(self) -> None:
		"""HALF_OPEN allows only max_probes (default 1) before blocking."""
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=0.01)
		mgr.record_failure("/ws/a")
		time.sleep(0.02)
		# First call transitions to HALF_OPEN and allows probe
		assert mgr.can_dispatch("/ws/a") is True
		# Second call in HALF_OPEN -- probe already in flight
		assert mgr.can_dispatch("/ws/a") is False


class TestOnStateChangeCallback:
	def test_closed_to_open_fires_callback(self) -> None:
		"""Callback fires on CLOSED -> OPEN transition."""
		transitions: list[tuple[str, str, str]] = []
		mgr = CircuitBreakerManager(
			max_failures=2, cooldown_seconds=60,
			on_state_change=lambda ws, old, new: transitions.append((ws, old, new)),
		)
		mgr.record_failure("/ws/a")
		assert transitions == []
		mgr.record_failure("/ws/a")
		assert transitions == [("/ws/a", "closed", "open")]

	def test_half_open_to_open_fires_callback(self) -> None:
		"""Callback fires on HALF_OPEN -> OPEN transition."""
		transitions: list[tuple[str, str, str]] = []
		mgr = CircuitBreakerManager(
			max_failures=1, cooldown_seconds=0.01,
			on_state_change=lambda ws, old, new: transitions.append((ws, old, new)),
		)
		mgr.record_failure("/ws/a")
		transitions.clear()
		time.sleep(0.02)
		mgr.can_dispatch("/ws/a")  # OPEN -> HALF_OPEN
		transitions.clear()
		mgr.record_failure("/ws/a")  # HALF_OPEN -> OPEN
		assert transitions == [("/ws/a", "half_open", "open")]

	def test_open_to_half_open_fires_callback(self) -> None:
		"""Callback fires on OPEN -> HALF_OPEN transition."""
		transitions: list[tuple[str, str, str]] = []
		mgr = CircuitBreakerManager(
			max_failures=1, cooldown_seconds=0.01,
			on_state_change=lambda ws, old, new: transitions.append((ws, old, new)),
		)
		mgr.record_failure("/ws/a")
		transitions.clear()
		time.sleep(0.02)
		mgr.can_dispatch("/ws/a")
		assert transitions == [("/ws/a", "open", "half_open")]

	def test_half_open_to_closed_fires_callback(self) -> None:
		"""Callback fires on HALF_OPEN -> CLOSED transition."""
		transitions: list[tuple[str, str, str]] = []
		mgr = CircuitBreakerManager(
			max_failures=1, cooldown_seconds=0.01,
			on_state_change=lambda ws, old, new: transitions.append((ws, old, new)),
		)
		mgr.record_failure("/ws/a")
		time.sleep(0.02)
		mgr.can_dispatch("/ws/a")  # -> HALF_OPEN
		transitions.clear()
		mgr.record_success("/ws/a")  # -> CLOSED
		assert transitions == [("/ws/a", "half_open", "closed")]

	def test_no_callback_when_none(self) -> None:
		"""No error when on_state_change is None (default)."""
		mgr = CircuitBreakerManager(max_failures=1, cooldown_seconds=60)
		mgr.record_failure("/ws/a")
		assert mgr.get_state("/ws/a") == CircuitBreakerState.OPEN


class TestCircuitBreakerDataclass:
	def test_defaults(self) -> None:
		cb = CircuitBreaker(workspace_id="/ws/test")
		assert cb._state == CircuitBreakerState.CLOSED
		assert cb.failure_count == 0
		assert cb.success_count == 0
		assert cb.max_failures == 3
		assert cb.cooldown_seconds == 120.0
		assert cb.half_open_max_probes == 1

	def test_state_property_returns_string(self) -> None:
		"""The state property returns a string value."""
		cb = CircuitBreaker(workspace_id="/ws/test")
		assert cb.state == "closed"
		cb._state = CircuitBreakerState.OPEN
		assert cb.state == "open"
		cb._state = CircuitBreakerState.HALF_OPEN
		assert cb.state == "half_open"

	def test_original_cooldown_tracked(self) -> None:
		"""_original_cooldown is set from cooldown_seconds on init."""
		cb = CircuitBreaker(workspace_id="/ws/test", cooldown_seconds=60.0)
		assert cb._original_cooldown == 60.0
