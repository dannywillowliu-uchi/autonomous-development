"""Tests for centralized constants module."""

from __future__ import annotations

import math

from mission_control.constants import DEFAULT_LIMITS, EVALUATOR_WEIGHTS, GRADING_WEIGHTS


class TestWeightTuples:
	def test_evaluator_weights_sum_to_one(self) -> None:
		assert math.isclose(sum(EVALUATOR_WEIGHTS), 1.0)

	def test_grading_weights_sum_to_one(self) -> None:
		assert math.isclose(sum(GRADING_WEIGHTS), 1.0)

	def test_evaluator_weights_length(self) -> None:
		assert len(EVALUATOR_WEIGHTS) == 4

	def test_grading_weights_length(self) -> None:
		assert len(GRADING_WEIGHTS) == 4

	def test_all_weights_positive(self) -> None:
		for w in EVALUATOR_WEIGHTS:
			assert w > 0.0
		for w in GRADING_WEIGHTS:
			assert w > 0.0


class TestDefaultLimits:
	def test_expected_keys_present(self) -> None:
		expected = {
			"max_sessions_per_run",
			"max_rounds",
			"max_output_mb",
			"max_retries",
			"verification_timeout",
			"session_timeout",
		}
		assert expected == set(DEFAULT_LIMITS.keys())

	def test_all_values_positive(self) -> None:
		for key, val in DEFAULT_LIMITS.items():
			assert val > 0, f"{key} should be positive"
