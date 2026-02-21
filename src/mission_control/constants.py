"""Centralized scoring weights and default limits."""

from __future__ import annotations

# -- Unit event types --

UNIT_EVENT_DISPATCHED = "dispatched"
UNIT_EVENT_CLAIMED = "claimed"
UNIT_EVENT_RUNNING = "running"
UNIT_EVENT_COMPLETED = "completed"
UNIT_EVENT_FAILED = "failed"
UNIT_EVENT_MERGED = "merged"
UNIT_EVENT_MERGE_FAILED = "merge_failed"
UNIT_EVENT_REJECTED = "rejected"
UNIT_EVENT_RETRY_QUEUED = "retry_queued"
UNIT_EVENT_RESEARCH_COMPLETED = "research_completed"
UNIT_EVENT_EXPERIMENT_COMPLETED = "experiment_completed"
UNIT_EVENT_DEGRADATION_TRANSITION = "degradation_transition"

UNIT_EVENT_TYPES: frozenset[str] = frozenset({
	UNIT_EVENT_DISPATCHED,
	UNIT_EVENT_CLAIMED,
	UNIT_EVENT_RUNNING,
	UNIT_EVENT_COMPLETED,
	UNIT_EVENT_FAILED,
	UNIT_EVENT_MERGED,
	UNIT_EVENT_MERGE_FAILED,
	UNIT_EVENT_REJECTED,
	UNIT_EVENT_RETRY_QUEUED,
	UNIT_EVENT_RESEARCH_COMPLETED,
	UNIT_EVENT_EXPERIMENT_COMPLETED,
})

EVENT_TO_STATUS: dict[str, str] = {
	UNIT_EVENT_DISPATCHED: "pending",
	UNIT_EVENT_CLAIMED: "claimed",
	UNIT_EVENT_RUNNING: "running",
	UNIT_EVENT_COMPLETED: "completed",
	UNIT_EVENT_FAILED: "failed",
	UNIT_EVENT_MERGED: "completed",
	UNIT_EVENT_MERGE_FAILED: "failed",
	UNIT_EVENT_REJECTED: "failed",
	UNIT_EVENT_RETRY_QUEUED: "pending",
	UNIT_EVENT_RESEARCH_COMPLETED: "completed",
	UNIT_EVENT_EXPERIMENT_COMPLETED: "completed",
}

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
