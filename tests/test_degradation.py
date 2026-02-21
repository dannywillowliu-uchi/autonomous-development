"""Tests for the tiered degradation state machine."""

from __future__ import annotations

import time

from mission_control.config import DegradationConfig
from mission_control.degradation import (
	DegradationLevel,
	DegradationManager,
	DegradationTransition,
)


class TestDegradationLevel:
	def test_level_ordering(self) -> None:
		assert DegradationLevel.FULL_CAPACITY < DegradationLevel.REDUCED_WORKERS
		assert DegradationLevel.REDUCED_WORKERS < DegradationLevel.DB_DEGRADED
		assert DegradationLevel.DB_DEGRADED < DegradationLevel.MODEL_FALLBACK
		assert DegradationLevel.MODEL_FALLBACK < DegradationLevel.READ_ONLY
		assert DegradationLevel.READ_ONLY < DegradationLevel.SAFE_STOP

	def test_level_comparison(self) -> None:
		assert DegradationLevel.FULL_CAPACITY == 0
		assert DegradationLevel.SAFE_STOP == 5
		assert DegradationLevel.DB_DEGRADED > DegradationLevel.FULL_CAPACITY

	def test_level_names(self) -> None:
		assert DegradationLevel.FULL_CAPACITY.name == "FULL_CAPACITY"
		assert DegradationLevel.SAFE_STOP.name == "SAFE_STOP"


class TestDegradationManagerInit:
	def test_default_level(self) -> None:
		mgr = DegradationManager()
		assert mgr.level == DegradationLevel.FULL_CAPACITY
		assert mgr.level_name == "FULL_CAPACITY"

	def test_config_overrides(self) -> None:
		cfg = DegradationConfig(db_error_threshold=10, recovery_success_threshold=5)
		mgr = DegradationManager(config=cfg)
		assert mgr._db_error_threshold == 10
		assert mgr._recovery_success_threshold == 5

	def test_default_config_values(self) -> None:
		mgr = DegradationManager()
		assert mgr._budget_fraction_threshold == 0.8
		assert mgr._conflict_rate_threshold == 0.5
		assert mgr._reduced_worker_fraction == 0.5
		assert mgr._db_error_threshold == 5
		assert mgr._rate_limit_threshold == 3
		assert mgr._verification_failure_threshold == 3
		assert mgr._recovery_success_threshold == 3


class TestDbErrors:
	def test_db_error_threshold_triggers_degraded(self) -> None:
		mgr = DegradationManager(config=DegradationConfig(db_error_threshold=3))
		mgr.record_db_error()
		mgr.record_db_error()
		assert mgr.level == DegradationLevel.FULL_CAPACITY
		mgr.record_db_error()
		assert mgr.level == DegradationLevel.DB_DEGRADED
		assert mgr.is_db_degraded is True

	def test_db_error_below_threshold(self) -> None:
		mgr = DegradationManager(config=DegradationConfig(db_error_threshold=5))
		for _ in range(4):
			mgr.record_db_error()
		assert mgr.is_db_degraded is False

	def test_db_recovery_needs_consecutive_successes(self) -> None:
		cfg = DegradationConfig(db_error_threshold=2, recovery_success_threshold=3)
		mgr = DegradationManager(config=cfg)
		mgr.record_db_error()
		mgr.record_db_error()
		assert mgr.level == DegradationLevel.DB_DEGRADED

		mgr.record_db_success()
		mgr.record_db_success()
		assert mgr.level == DegradationLevel.DB_DEGRADED  # not enough yet
		mgr.record_db_success()
		assert mgr.level == DegradationLevel.REDUCED_WORKERS  # recovered one level

	def test_is_db_degraded_at_higher_levels(self) -> None:
		mgr = DegradationManager()
		mgr._transition(DegradationLevel.MODEL_FALLBACK, "test")
		assert mgr.is_db_degraded is True  # MODEL_FALLBACK > DB_DEGRADED


class TestMergeConflicts:
	def test_conflict_rate_triggers_reduced_workers(self) -> None:
		cfg = DegradationConfig(conflict_rate_threshold=0.5)
		mgr = DegradationManager(config=cfg)
		# 3 conflicts out of 4 = 75% > 50%
		mgr.record_merge_attempt(conflict=True)
		mgr.record_merge_attempt(conflict=True)
		mgr.record_merge_attempt(conflict=True)
		assert mgr.level == DegradationLevel.FULL_CAPACITY  # need min 4
		mgr.record_merge_attempt(conflict=False)
		assert mgr.level == DegradationLevel.REDUCED_WORKERS

	def test_low_conflict_rate_no_escalation(self) -> None:
		mgr = DegradationManager(config=DegradationConfig(conflict_rate_threshold=0.5))
		mgr.record_merge_attempt(conflict=True)
		mgr.record_merge_attempt(conflict=False)
		mgr.record_merge_attempt(conflict=False)
		mgr.record_merge_attempt(conflict=False)
		assert mgr.level == DegradationLevel.FULL_CAPACITY

	def test_below_minimum_attempts(self) -> None:
		mgr = DegradationManager()
		mgr.record_merge_attempt(conflict=True)
		mgr.record_merge_attempt(conflict=True)
		mgr.record_merge_attempt(conflict=True)
		# Only 3 attempts, need 4 minimum
		assert mgr.level == DegradationLevel.FULL_CAPACITY


class TestBudget:
	def test_budget_pressure_triggers_reduced_workers(self) -> None:
		cfg = DegradationConfig(budget_fraction_threshold=0.8)
		mgr = DegradationManager(config=cfg)
		mgr.check_budget_fraction(spent=85.0, budget=100.0)
		assert mgr.level == DegradationLevel.REDUCED_WORKERS

	def test_budget_below_threshold(self) -> None:
		mgr = DegradationManager(config=DegradationConfig(budget_fraction_threshold=0.8))
		mgr.check_budget_fraction(spent=70.0, budget=100.0)
		assert mgr.level == DegradationLevel.FULL_CAPACITY

	def test_zero_budget_ignored(self) -> None:
		mgr = DegradationManager()
		mgr.check_budget_fraction(spent=100.0, budget=0.0)
		assert mgr.level == DegradationLevel.FULL_CAPACITY


class TestRateLimits:
	def test_rate_limit_triggers_model_fallback(self) -> None:
		cfg = DegradationConfig(rate_limit_threshold=3, rate_limit_window_seconds=60.0)
		mgr = DegradationManager(config=cfg)
		mgr.record_rate_limit()
		mgr.record_rate_limit()
		assert mgr.level == DegradationLevel.FULL_CAPACITY
		mgr.record_rate_limit()
		assert mgr.level == DegradationLevel.MODEL_FALLBACK

	def test_old_rate_limits_expire(self) -> None:
		cfg = DegradationConfig(
			rate_limit_threshold=3,
			rate_limit_window_seconds=0.01,
		)
		mgr = DegradationManager(config=cfg)
		mgr.record_rate_limit()
		mgr.record_rate_limit()
		time.sleep(0.02)
		mgr.record_rate_limit()
		# Old ones expired, only 1 in window
		assert mgr.level == DegradationLevel.FULL_CAPACITY


class TestVerificationFailures:
	def test_verification_failures_trigger_read_only(self) -> None:
		cfg = DegradationConfig(verification_failure_threshold=3)
		mgr = DegradationManager(config=cfg)
		mgr.record_verification_failure()
		mgr.record_verification_failure()
		assert mgr.level == DegradationLevel.FULL_CAPACITY
		mgr.record_verification_failure()
		assert mgr.level == DegradationLevel.READ_ONLY
		assert mgr.is_read_only is True

	def test_verification_recovery(self) -> None:
		cfg = DegradationConfig(
			verification_failure_threshold=2,
			recovery_success_threshold=2,
		)
		mgr = DegradationManager(config=cfg)
		mgr.record_verification_failure()
		mgr.record_verification_failure()
		assert mgr.level == DegradationLevel.READ_ONLY

		mgr.record_verification_success()
		mgr.record_verification_success()
		assert mgr.level == DegradationLevel.MODEL_FALLBACK


class TestInFlightDrain:
	def test_drain_triggers_safe_stop(self) -> None:
		cfg = DegradationConfig(verification_failure_threshold=1)
		mgr = DegradationManager(config=cfg)
		mgr.record_verification_failure()
		assert mgr.level == DegradationLevel.READ_ONLY

		mgr.check_in_flight_drained(count=5)
		assert mgr.level == DegradationLevel.READ_ONLY  # still in-flight

		mgr.check_in_flight_drained(count=0)
		assert mgr.level == DegradationLevel.SAFE_STOP
		assert mgr.should_stop is True

	def test_drain_only_from_read_only(self) -> None:
		mgr = DegradationManager()
		mgr.check_in_flight_drained(count=0)
		assert mgr.level == DegradationLevel.FULL_CAPACITY  # no effect


class TestGeneralRecovery:
	def test_recovery_from_reduced_workers(self) -> None:
		cfg = DegradationConfig(
			conflict_rate_threshold=0.5,
			recovery_success_threshold=3,
		)
		mgr = DegradationManager(config=cfg)
		# Trigger REDUCED_WORKERS via conflicts
		for _ in range(4):
			mgr.record_merge_attempt(conflict=True)
		assert mgr.level == DegradationLevel.REDUCED_WORKERS

		mgr.record_general_success()
		mgr.record_general_success()
		assert mgr.level == DegradationLevel.REDUCED_WORKERS
		mgr.record_general_success()
		assert mgr.level == DegradationLevel.FULL_CAPACITY

	def test_general_success_no_effect_at_full(self) -> None:
		mgr = DegradationManager()
		mgr.record_general_success()
		assert mgr.level == DegradationLevel.FULL_CAPACITY


class TestEscalation:
	def test_escalation_only_goes_up(self) -> None:
		mgr = DegradationManager()
		mgr._transition(DegradationLevel.MODEL_FALLBACK, "test")
		# Trying to escalate to REDUCED_WORKERS (lower) should do nothing
		mgr._escalate_to(DegradationLevel.REDUCED_WORKERS, "test")
		assert mgr.level == DegradationLevel.MODEL_FALLBACK

	def test_no_transition_to_same_level(self) -> None:
		mgr = DegradationManager()
		transitions: list[DegradationTransition] = []
		mgr._on_transition = lambda t: transitions.append(t)
		mgr._transition(DegradationLevel.FULL_CAPACITY, "noop")
		assert len(transitions) == 0


class TestEffectiveWorkerCount:
	def test_full_capacity(self) -> None:
		mgr = DegradationManager()
		assert mgr.get_effective_worker_count(4) == 4

	def test_reduced_workers(self) -> None:
		cfg = DegradationConfig(reduced_worker_fraction=0.5)
		mgr = DegradationManager(config=cfg)
		mgr._transition(DegradationLevel.REDUCED_WORKERS, "test")
		assert mgr.get_effective_worker_count(4) == 2

	def test_reduced_workers_minimum_one(self) -> None:
		cfg = DegradationConfig(reduced_worker_fraction=0.1)
		mgr = DegradationManager(config=cfg)
		mgr._transition(DegradationLevel.REDUCED_WORKERS, "test")
		assert mgr.get_effective_worker_count(1) == 1

	def test_db_degraded_still_reduced(self) -> None:
		mgr = DegradationManager()
		mgr._transition(DegradationLevel.DB_DEGRADED, "test")
		assert mgr.get_effective_worker_count(4) == 2  # default 0.5 fraction

	def test_read_only_returns_zero(self) -> None:
		mgr = DegradationManager()
		mgr._transition(DegradationLevel.READ_ONLY, "test")
		assert mgr.get_effective_worker_count(4) == 0

	def test_safe_stop_returns_zero(self) -> None:
		mgr = DegradationManager()
		mgr._transition(DegradationLevel.SAFE_STOP, "test")
		assert mgr.get_effective_worker_count(4) == 0


class TestTransitionCallback:
	def test_callback_fires(self) -> None:
		transitions: list[DegradationTransition] = []
		mgr = DegradationManager(on_transition=lambda t: transitions.append(t))
		cfg = DegradationConfig(db_error_threshold=1)
		mgr._db_error_threshold = cfg.db_error_threshold
		mgr.record_db_error()
		assert len(transitions) == 1
		assert transitions[0].from_level == DegradationLevel.FULL_CAPACITY
		assert transitions[0].to_level == DegradationLevel.DB_DEGRADED
		assert transitions[0].trigger == "db_errors"

	def test_callback_not_called_without_transition(self) -> None:
		call_count = 0

		def cb(t: DegradationTransition) -> None:
			nonlocal call_count
			call_count += 1

		mgr = DegradationManager(on_transition=cb)
		mgr.record_db_error()  # 1 of 5, no transition
		assert call_count == 0


class TestStatusDict:
	def test_serialization(self) -> None:
		mgr = DegradationManager()
		status = mgr.get_status_dict()
		assert status["level"] == "FULL_CAPACITY"
		assert status["level_value"] == 0
		assert status["db_errors"] == 0
		assert status["merge_attempts"] == 0
		assert status["merge_conflicts"] == 0
		assert status["conflict_rate"] == 0.0
		assert status["rate_limit_count"] == 0
		assert status["verification_failures"] == 0
		assert status["consecutive_successes"] == 0

	def test_serialization_with_data(self) -> None:
		cfg = DegradationConfig(db_error_threshold=2)
		mgr = DegradationManager(config=cfg)
		mgr.record_db_error()
		mgr.record_db_error()
		mgr.record_merge_attempt(conflict=True)
		status = mgr.get_status_dict()
		assert status["level"] == "DB_DEGRADED"
		assert status["db_errors"] == 2
		assert status["merge_attempts"] == 1
		assert status["merge_conflicts"] == 1
		assert status["conflict_rate"] == 1.0


class TestProperties:
	def test_is_db_degraded_false_at_lower(self) -> None:
		mgr = DegradationManager()
		assert mgr.is_db_degraded is False
		mgr._transition(DegradationLevel.REDUCED_WORKERS, "test")
		assert mgr.is_db_degraded is False

	def test_is_read_only_false_below(self) -> None:
		mgr = DegradationManager()
		mgr._transition(DegradationLevel.MODEL_FALLBACK, "test")
		assert mgr.is_read_only is False

	def test_should_stop_only_at_safe_stop(self) -> None:
		mgr = DegradationManager()
		mgr._transition(DegradationLevel.READ_ONLY, "test")
		assert mgr.should_stop is False
		mgr._transition(DegradationLevel.SAFE_STOP, "test")
		assert mgr.should_stop is True


class TestConfigTomlParsing:
	def test_default_values(self) -> None:
		cfg = DegradationConfig()
		assert cfg.budget_fraction_threshold == 0.8
		assert cfg.conflict_rate_threshold == 0.5
		assert cfg.reduced_worker_fraction == 0.5
		assert cfg.db_error_threshold == 5
		assert cfg.rate_limit_window_seconds == 60.0
		assert cfg.rate_limit_threshold == 3
		assert cfg.verification_failure_threshold == 3
		assert cfg.safe_stop_timeout_seconds == 300.0
		assert cfg.recovery_success_threshold == 3

	def test_custom_values(self) -> None:
		cfg = DegradationConfig(
			budget_fraction_threshold=0.9,
			db_error_threshold=10,
			recovery_success_threshold=5,
		)
		assert cfg.budget_fraction_threshold == 0.9
		assert cfg.db_error_threshold == 10
		assert cfg.recovery_success_threshold == 5


class TestBackwardCompat:
	def test_is_db_degraded_matches_old_semantics(self) -> None:
		"""is_db_degraded should be True when level >= DB_DEGRADED, matching old _db_degraded flag."""
		mgr = DegradationManager(config=DegradationConfig(db_error_threshold=3))
		assert mgr.is_db_degraded is False
		mgr.record_db_error()
		mgr.record_db_error()
		assert mgr.is_db_degraded is False
		mgr.record_db_error()
		assert mgr.is_db_degraded is True
