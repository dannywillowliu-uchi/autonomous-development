"""Tests for swarm-level circuit breaker."""

import time

from autodev.swarm.circuit_breaker import (
	SwarmCircuitBreaker,
	SwarmCircuitBreakerConfig,
	SwarmCircuitBreakerState,
)

# --- Existing tests (backward compat) ---


def test_consecutive_failures_trip():
	cb = SwarmCircuitBreaker()
	for _ in range(3):
		cb.record_failure()
	allowed, reason = cb.can_spawn()
	assert not allowed
	assert "consecutive" in reason
	assert cb.is_tripped


def test_success_resets_consecutive():
	cb = SwarmCircuitBreaker()
	cb.record_failure()
	cb.record_failure()
	cb.record_success()
	cb.record_failure()
	cb.record_failure()
	assert not cb.is_tripped
	allowed, _ = cb.can_spawn()
	assert allowed


def test_cooldown_allows_retry(monkeypatch):
	fake_time = [1000.0]
	monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

	cb = SwarmCircuitBreaker(SwarmCircuitBreakerConfig(cooldown_seconds=60.0))
	for _ in range(3):
		cb.record_failure()
	assert cb.is_tripped

	fake_time[0] = 1061.0
	allowed, reason = cb.can_spawn()
	assert allowed
	assert reason == ""
	assert not cb.is_tripped


def test_window_failures_trip(monkeypatch):
	fake_time = [1000.0]
	monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

	config = SwarmCircuitBreakerConfig(
		max_consecutive_failures=10,
		max_failures_per_window=3,
		window_seconds=10.0,
	)
	cb = SwarmCircuitBreaker(config)

	cb.record_failure()
	fake_time[0] = 1001.0
	cb.record_success()
	cb.record_failure()
	fake_time[0] = 1002.0
	cb.record_success()
	cb.record_failure()

	assert cb.is_tripped
	assert "failures in 10s window" in cb.trip_reason


def test_window_expiry(monkeypatch):
	fake_time = [1000.0]
	monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

	config = SwarmCircuitBreakerConfig(
		max_consecutive_failures=10,
		max_failures_per_window=3,
		window_seconds=10.0,
	)
	cb = SwarmCircuitBreaker(config)

	cb.record_failure()
	fake_time[0] = 1001.0
	cb.record_success()
	cb.record_failure()

	fake_time[0] = 1020.0
	cb.record_success()
	cb.record_failure()

	assert not cb.is_tripped


# --- State machine tests ---


def test_state_starts_closed():
	cb = SwarmCircuitBreaker()
	assert cb.state == SwarmCircuitBreakerState.CLOSED
	assert not cb.is_tripped


def test_trip_transitions_to_open():
	cb = SwarmCircuitBreaker()
	for _ in range(3):
		cb.record_failure()
	assert cb.state == SwarmCircuitBreakerState.OPEN


def test_cooldown_transitions_to_half_open(monkeypatch):
	fake_time = [1000.0]
	monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

	cb = SwarmCircuitBreaker(SwarmCircuitBreakerConfig(cooldown_seconds=60.0))
	for _ in range(3):
		cb.record_failure()
	assert cb.state == SwarmCircuitBreakerState.OPEN

	fake_time[0] = 1061.0
	allowed, _ = cb.can_spawn()
	assert allowed
	assert cb.state == SwarmCircuitBreakerState.HALF_OPEN


def test_half_open_probe_success_closes(monkeypatch):
	"""Successful probe in HALF_OPEN transitions to CLOSED."""
	fake_time = [1000.0]
	monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

	cb = SwarmCircuitBreaker(SwarmCircuitBreakerConfig(cooldown_seconds=60.0))
	for _ in range(3):
		cb.record_failure()

	fake_time[0] = 1061.0
	cb.can_spawn()  # transitions to HALF_OPEN
	assert cb.state == SwarmCircuitBreakerState.HALF_OPEN

	cb.record_success()
	assert cb.state == SwarmCircuitBreakerState.CLOSED
	assert not cb.is_tripped
	assert cb.trip_reason == ""


def test_half_open_probe_failure_reopens(monkeypatch):
	"""Failed probe in HALF_OPEN transitions back to OPEN."""
	fake_time = [1000.0]
	monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

	cb = SwarmCircuitBreaker(SwarmCircuitBreakerConfig(cooldown_seconds=60.0))
	for _ in range(3):
		cb.record_failure()

	fake_time[0] = 1061.0
	cb.can_spawn()  # transitions to HALF_OPEN
	assert cb.state == SwarmCircuitBreakerState.HALF_OPEN

	cb.record_failure()
	assert cb.state == SwarmCircuitBreakerState.OPEN
	assert cb.is_tripped
	assert "probe failed" in cb.trip_reason


def test_half_open_probe_limit(monkeypatch):
	"""HALF_OPEN allows only max_probes (default 1) before blocking."""
	fake_time = [1000.0]
	monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

	cb = SwarmCircuitBreaker(SwarmCircuitBreakerConfig(cooldown_seconds=60.0))
	for _ in range(3):
		cb.record_failure()

	fake_time[0] = 1061.0
	allowed1, _ = cb.can_spawn()  # transitions to HALF_OPEN, first probe
	assert allowed1

	allowed2, reason2 = cb.can_spawn()  # second probe, should be blocked
	assert not allowed2
	assert "probe in flight" in reason2


def test_half_open_multiple_probes(monkeypatch):
	"""With half_open_max_probes=2, two probes are allowed."""
	fake_time = [1000.0]
	monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

	config = SwarmCircuitBreakerConfig(cooldown_seconds=60.0, half_open_max_probes=2)
	cb = SwarmCircuitBreaker(config)
	for _ in range(3):
		cb.record_failure()

	fake_time[0] = 1061.0
	allowed1, _ = cb.can_spawn()
	assert allowed1
	allowed2, _ = cb.can_spawn()
	assert allowed2
	allowed3, reason3 = cb.can_spawn()
	assert not allowed3
	assert "probe in flight" in reason3


# --- Per-task failure tracking ---


def test_per_task_failure_tracking():
	cb = SwarmCircuitBreaker(SwarmCircuitBreakerConfig(max_consecutive_failures=10))
	cb.record_failure(task_id="task-A")
	cb.record_failure(task_id="task-A")
	cb.record_failure(task_id="task-B")

	assert cb.task_failure_count("task-A") == 2
	assert cb.task_failure_count("task-B") == 1
	assert cb.task_failure_count("task-C") == 0


def test_per_task_success_clears():
	cb = SwarmCircuitBreaker(SwarmCircuitBreakerConfig(max_consecutive_failures=10))
	cb.record_failure(task_id="task-A")
	cb.record_failure(task_id="task-A")
	cb.record_success(task_id="task-A")

	assert cb.task_failure_count("task-A") == 0


def test_per_task_independent_of_global():
	"""Per-task counters are independent of global consecutive counter."""
	cb = SwarmCircuitBreaker(SwarmCircuitBreakerConfig(max_consecutive_failures=10))
	cb.record_failure(task_id="task-A")
	cb.record_success(task_id="task-B")  # resets global consecutive, not task-A

	assert cb.task_failure_count("task-A") == 1
	assert cb.consecutive_failures == 0


# --- on_trip callback ---


def test_on_trip_callback():
	reasons = []
	cb = SwarmCircuitBreaker(on_trip=lambda r: reasons.append(r))
	for _ in range(3):
		cb.record_failure()

	assert len(reasons) == 1
	assert "consecutive" in reasons[0]


def test_on_trip_not_called_when_already_open():
	"""Callback fires only on CLOSED->OPEN transition, not repeated failures while open."""
	reasons = []
	cb = SwarmCircuitBreaker(on_trip=lambda r: reasons.append(r))
	for _ in range(5):
		cb.record_failure()

	assert len(reasons) == 1


def test_on_trip_callback_on_probe_failure(monkeypatch):
	"""Callback fires when a probe failure re-opens the breaker."""
	fake_time = [1000.0]
	monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

	reasons = []
	cb = SwarmCircuitBreaker(
		SwarmCircuitBreakerConfig(cooldown_seconds=60.0),
		on_trip=lambda r: reasons.append(r),
	)
	for _ in range(3):
		cb.record_failure()
	assert len(reasons) == 1

	fake_time[0] = 1061.0
	cb.can_spawn()  # -> HALF_OPEN

	cb.record_failure()  # probe fails -> OPEN
	assert len(reasons) == 2
	assert "probe failed" in reasons[1]


# --- reset and summary ---


def test_reset():
	cb = SwarmCircuitBreaker()
	for _ in range(3):
		cb.record_failure(task_id="task-A")
	assert cb.is_tripped

	cb.reset()
	assert cb.state == SwarmCircuitBreakerState.CLOSED
	assert not cb.is_tripped
	assert cb.consecutive_failures == 0
	assert cb.task_failure_count("task-A") == 0
	assert cb.trip_reason == ""


def test_get_summary():
	cb = SwarmCircuitBreaker()
	cb.record_failure(task_id="task-A")
	cb.record_failure(task_id="task-A")

	summary = cb.get_summary()
	assert summary["state"] == "closed"
	assert summary["consecutive_failures"] == 2
	assert summary["task_failures"] == {"task-A": 2}
	assert summary["trip_reason"] == ""


def test_get_summary_when_tripped():
	cb = SwarmCircuitBreaker()
	for _ in range(3):
		cb.record_failure(task_id="task-X")

	summary = cb.get_summary()
	assert summary["state"] == "open"
	assert "consecutive" in summary["trip_reason"]
	assert summary["task_failures"] == {"task-X": 3}
