"""Centralized scoring weights and default limits."""

from __future__ import annotations

# Evaluator round-scoring weights (test_improvement, lint_improvement, completion_rate, no_regression)
EVALUATOR_WEIGHTS: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)

# Decomposition grading weights (avg_review, retry_rate, overlap_rate, completion_rate)
GRADING_WEIGHTS: tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20)

# Common default limits used across the codebase
DEFAULT_LIMITS: dict[str, int] = {
	"max_sessions_per_run": 10,
	"max_rounds": 20,
	"max_output_mb": 50,
	"max_retries": 3,
	"verification_timeout": 300,
	"session_timeout": 2700,
}
