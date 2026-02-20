"""Data models for mission-control state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class MCResultSchema(BaseModel, extra="ignore"):
	"""Pydantic schema for validating MC_RESULT JSON from worker output."""

	status: Literal["completed", "failed", "blocked"]
	commits: list[str]
	summary: str
	files_changed: list[str]
	discoveries: list[str] = []
	concerns: list[str] = []


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
	return uuid4().hex[:12]


@dataclass
class Session:
	"""A single Claude Code session run."""

	id: str = field(default_factory=_new_id)
	target_name: str = ""
	task_description: str = ""
	status: str = "pending"  # pending/running/completed/failed/reverted
	branch_name: str = ""
	started_at: str = field(default_factory=_now_iso)
	finished_at: str | None = None
	exit_code: int | None = None
	commit_hash: str | None = None
	cost_usd: float | None = None
	output_summary: str = ""


@dataclass
class Snapshot:
	"""Project health snapshot at a point in time."""

	id: str = field(default_factory=_new_id)
	session_id: str | None = None
	taken_at: str = field(default_factory=_now_iso)
	test_total: int = 0
	test_passed: int = 0
	test_failed: int = 0
	lint_errors: int = 0
	type_errors: int = 0
	security_findings: int = 0
	raw_output: str = ""



@dataclass
class Decision:
	"""A decision logged during a session."""

	id: str = field(default_factory=_new_id)
	session_id: str = ""
	decision: str = ""
	rationale: str = ""
	timestamp: str = field(default_factory=_now_iso)


@dataclass
class SnapshotDelta:
	"""Difference between two snapshots."""

	tests_added: int = 0
	tests_fixed: int = 0
	tests_broken: int = 0
	lint_delta: int = 0
	type_delta: int = 0
	security_delta: int = 0

	@property
	def improved(self) -> bool:
		return (
			(self.tests_fixed > 0 or self.lint_delta < 0 or self.type_delta < 0 or self.security_delta < 0)
			and self.tests_broken == 0
			and self.security_delta <= 0
		)

	@property
	def regressed(self) -> bool:
		return self.tests_broken > 0 or self.security_delta > 0


# -- Parallel mode models --


@dataclass
class Plan:
	"""A decomposed objective for parallel execution."""

	id: str = field(default_factory=_new_id)
	objective: str = ""
	status: str = "pending"  # pending/active/completed/failed
	created_at: str = field(default_factory=_now_iso)
	finished_at: str | None = None
	raw_planner_output: str = ""
	total_units: int = 0
	completed_units: int = 0
	failed_units: int = 0
	round_id: str | None = None  # link to Round for mission mode
	root_node_id: str | None = None  # root PlanNode for recursive planning


@dataclass
class WorkUnit:
	"""A single work item within a Plan, claimable by a worker."""

	id: str = field(default_factory=_new_id)
	plan_id: str = ""
	title: str = ""
	description: str = ""
	files_hint: str = ""  # comma-separated paths this unit likely touches
	verification_hint: str = ""  # specific verification focus
	priority: int = 1  # 1=highest
	status: str = "pending"  # pending/claimed/running/completed/failed/blocked
	worker_id: str | None = None
	round_id: str | None = None  # link to Round for mission mode
	plan_node_id: str | None = None  # leaf PlanNode that produced this unit
	handoff_id: str | None = None  # structured handoff from worker
	depends_on: str = ""  # comma-separated WorkUnit IDs
	branch_name: str = ""
	claimed_at: str | None = None
	heartbeat_at: str | None = None
	started_at: str | None = None
	finished_at: str | None = None
	exit_code: int | None = None
	commit_hash: str | None = None
	output_summary: str = ""
	attempt: int = 0
	max_attempts: int = 3
	unit_type: str = "implementation"  # implementation/research
	experiment_mode: bool = False
	timeout: int | None = None  # per-unit timeout override (seconds)
	verification_command: str | None = None  # per-unit verification override
	epoch_id: str | None = None  # continuous mode epoch
	specialist: str = ""  # specialist template name (e.g. "test-writer")
	acceptance_criteria: str = ""  # what must be true for this unit to be "done"
	specialist: str = ""  # specialist type: test-writer, refactorer, debugger, or empty for general
	input_tokens: int = 0
	output_tokens: int = 0
	cost_usd: float = 0.0


@dataclass
class Worker:
	"""A parallel worker agent and its workspace."""

	id: str = field(default_factory=_new_id)
	workspace_path: str = ""
	status: str = "idle"  # idle/working/dead
	current_unit_id: str | None = None
	pid: int | None = None
	started_at: str = field(default_factory=_now_iso)
	last_heartbeat: str = field(default_factory=_now_iso)
	units_completed: int = 0
	units_failed: int = 0
	total_cost_usd: float = 0.0
	backend_type: str = "local"  # local/ssh/container
	backend_metadata: str = ""  # JSON blob for backend-specific data


@dataclass
class MergeRequest:
	"""A request to merge a completed work unit into the base branch."""

	id: str = field(default_factory=_new_id)
	work_unit_id: str = ""
	worker_id: str = ""
	branch_name: str = ""
	commit_hash: str = ""
	status: str = "pending"  # pending/verifying/merged/rejected/conflict
	position: int = 0
	created_at: str = field(default_factory=_now_iso)
	verified_at: str | None = None
	merged_at: str | None = None
	rejection_reason: str = ""
	rebase_attempts: int = 0


# -- Mission mode models --


@dataclass
class Mission:
	"""A continuous development mission toward a single objective."""

	id: str = field(default_factory=_new_id)
	objective: str = ""
	status: str = "pending"  # pending/running/completed/failed/stalled/stopped
	started_at: str = field(default_factory=_now_iso)
	finished_at: str | None = None
	total_rounds: int = 0
	total_cost_usd: float = 0.0
	final_score: float = 0.0
	stopped_reason: str = ""
	ambition_score: int = 0
	next_objective: str = ""
	proposed_by_strategist: bool = False


@dataclass
class Round:
	"""A single plan-execute-evaluate cycle within a mission."""

	id: str = field(default_factory=_new_id)
	mission_id: str = ""
	number: int = 0
	status: str = "pending"  # pending/planning/executing/evaluating/completed/failed
	started_at: str = field(default_factory=_now_iso)
	finished_at: str | None = None
	snapshot_hash: str = ""  # git commit hash of mc/green at round start
	plan_id: str | None = None
	objective_score: float = 0.0
	objective_met: bool = False
	total_units: int = 0
	completed_units: int = 0
	failed_units: int = 0
	cost_usd: float = 0.0
	discoveries: str = ""  # JSON array of discovery strings from workers


@dataclass
class PlanNode:
	"""A node in a recursive plan tree (branch or leaf)."""

	id: str = field(default_factory=_new_id)
	plan_id: str = ""
	parent_id: str | None = None
	depth: int = 0
	scope: str = ""  # what this node is responsible for
	strategy: str = ""  # subdivide/leaves -- what the planner decided
	status: str = "pending"  # pending/expanding/expanded/failed
	node_type: str = "branch"  # branch/leaf
	work_unit_id: str | None = None  # set if node_type == "leaf"
	children_ids: str = ""  # comma-separated child PlanNode IDs


class Handoff(BaseModel):
	"""Structured output from a worker after executing a work unit."""

	id: str = Field(default_factory=_new_id)
	work_unit_id: str = ""
	round_id: str = ""
	status: str = ""  # completed/failed/blocked
	commits: list[str] = Field(default_factory=list)
	summary: str = ""
	discoveries: list[str] = Field(default_factory=list)
	concerns: list[str] = Field(default_factory=list)
	files_changed: list[str] = Field(default_factory=list)
	epoch_id: str | None = None  # continuous mode epoch


# -- Feedback models --


@dataclass
class Reflection:
	"""Post-round structured reflection grounded in objective metrics."""

	id: str = field(default_factory=_new_id)
	mission_id: str = ""
	round_id: str = ""
	round_number: int = 0
	timestamp: str = field(default_factory=_now_iso)
	# Objective metrics (from Snapshot/SnapshotDelta)
	tests_before: int = 0
	tests_after: int = 0
	tests_delta: int = 0
	lint_delta: int = 0
	type_delta: int = 0
	# Round performance
	objective_score: float = 0.0
	score_delta: float = 0.0  # vs previous round
	units_planned: int = 0
	units_completed: int = 0
	units_failed: int = 0
	completion_rate: float = 0.0
	# Planning metrics
	plan_depth: int = 0
	plan_strategy: str = ""  # "subdivide" or "leaves" at root
	# Merge/fixup metrics
	fixup_promoted: bool = False
	fixup_attempts: int = 0
	merge_conflicts: int = 0
	# Discoveries
	discoveries_count: int = 0
	# Continuous mode
	epoch_id: str | None = None


@dataclass
class Reward:
	"""Composite reward score grounded in objective signals."""

	id: str = field(default_factory=_new_id)
	round_id: str = ""
	mission_id: str = ""
	timestamp: str = field(default_factory=_now_iso)
	reward: float = 0.0
	# Components (all derived from objective data)
	verification_improvement: float = 0.0  # SnapshotDelta: did quality improve?
	completion_rate: float = 0.0  # units_completed / units_planned
	score_progress: float = 0.0  # objective score delta
	fixup_efficiency: float = 0.0  # promoted on first attempt?
	no_regression: float = 0.0  # no tests broken, no security added
	# Continuous mode
	epoch_id: str | None = None


@dataclass
class Signal:
	"""Cross-process signal for controlling a running mission."""

	id: str = field(default_factory=_new_id)
	mission_id: str = ""
	signal_type: str = ""  # stop/retry_unit/adjust
	payload: str = ""  # JSON: unit_id for retry, params for adjust
	status: str = "pending"  # pending/acknowledged/expired
	created_at: str = field(default_factory=_now_iso)
	acknowledged_at: str | None = None


@dataclass
class Experience:
	"""Successful approach indexed by task keywords for retrieval."""

	id: str = field(default_factory=_new_id)
	round_id: str = ""
	work_unit_id: str = ""
	timestamp: str = field(default_factory=_now_iso)
	# Task description
	title: str = ""
	scope: str = ""
	files_hint: str = ""
	# Outcome
	status: str = ""  # completed/failed
	summary: str = ""
	files_changed: str = ""  # JSON array
	# Approach data
	discoveries: str = ""  # JSON array
	concerns: str = ""  # JSON array
	# Reward
	reward: float = 0.0
	# Continuous mode
	epoch_id: str | None = None


# -- Continuous mode models --


@dataclass
class Epoch:
	"""A grouping of consecutive units between planner invocations."""

	id: str = field(default_factory=_new_id)
	mission_id: str = ""
	number: int = 0
	started_at: str = field(default_factory=_now_iso)
	finished_at: str | None = None
	units_planned: int = 0
	units_completed: int = 0
	units_failed: int = 0
	score_at_start: float = 0.0
	score_at_end: float = 0.0


@dataclass
class UnitEvent:
	"""A timeline event for a single unit in continuous mode."""

	id: str = field(default_factory=_new_id)
	mission_id: str = ""
	epoch_id: str = ""
	work_unit_id: str = ""
	event_type: str = ""  # dispatched/completed/failed/merged/rejected
	timestamp: str = field(default_factory=_now_iso)
	score_after: float = 0.0
	details: str = ""  # JSON blob for extra info
	input_tokens: int = 0
	output_tokens: int = 0


# -- Strategic context models --


@dataclass
class StrategicContext:
	"""Rolling strategic context that persists across missions."""

	id: str = field(default_factory=_new_id)
	mission_id: str = ""
	timestamp: str = field(default_factory=_now_iso)
	what_attempted: str = ""
	what_worked: str = ""
	what_failed: str = ""
	recommended_next: str = ""


# -- Experiment models --


@dataclass
class ExperimentResult:
	"""Result of an experiment-mode work unit comparing multiple approaches."""

	id: str = field(default_factory=_new_id)
	work_unit_id: str = ""
	epoch_id: str | None = None
	mission_id: str = ""
	timestamp: str = field(default_factory=_now_iso)
	approach_count: int = 2
	comparison_report: str = ""  # JSON blob
	recommended_approach: str = ""
	created_at: str = field(default_factory=_now_iso)


# -- Discovery models --


@dataclass
class BacklogItem:
	"""A persistent backlog item for cross-mission task tracking."""

	id: str = field(default_factory=_new_id)
	title: str = ""
	description: str = ""
	priority_score: float = 0.0
	impact: int = 5
	effort: int = 5
	track: str = ""  # feature/quality/security
	status: str = "pending"  # pending/in_progress/completed/deferred/rejected
	source_mission_id: str = ""
	created_at: str = field(default_factory=_now_iso)
	updated_at: str = field(default_factory=_now_iso)
	attempt_count: int = 0
	last_failure_reason: str = ""
	pinned_score: float | None = None
	depends_on: str = ""  # comma-separated BacklogItem IDs
	tags: str = ""  # comma-separated tags
	acceptance_criteria: str = ""  # what must be true for this item to be "done"


@dataclass
class UnitReview:
	"""LLM review of a merged work unit's code changes."""

	id: str = field(default_factory=_new_id)
	work_unit_id: str = ""
	mission_id: str = ""
	epoch_id: str = ""
	timestamp: str = field(default_factory=_now_iso)
	alignment_score: int = 0  # 1-10: how well the diff aligns with the objective
	approach_score: int = 0  # 1-10: quality of the implementation approach
	test_score: int = 0  # 1-10: meaningfulness of the tests
	criteria_met_score: int = 0  # 1-10: how well acceptance criteria were met
	avg_score: float = 0.0
	rationale: str = ""
	model: str = ""
	cost_usd: float = 0.0


@dataclass
class TrajectoryRating:
	"""Human rating of a mission's overall trajectory."""

	id: str = field(default_factory=_new_id)
	mission_id: str = ""
	rating: int = 0  # 1-10
	feedback: str = ""
	timestamp: str = field(default_factory=_now_iso)


@dataclass
class DecompositionGrade:
	"""Algorithmic grading of a planner's decomposition quality."""

	id: str = field(default_factory=_new_id)
	plan_id: str = ""
	epoch_id: str = ""
	mission_id: str = ""
	timestamp: str = field(default_factory=_now_iso)
	avg_review_score: float = 0.0
	retry_rate: float = 0.0
	overlap_rate: float = 0.0
	completion_rate: float = 0.0
	composite_score: float = 0.0
	unit_count: int = 0


class ContextScope:
	"""Constants for context item scoping levels."""

	MISSION = "mission"
	ROUND = "round"
	UNIT = "unit"


@dataclass
class ContextItem:
	"""A typed context item produced by workers as discoveries.

	Used to selectively inject relevant context into subsequent worker prompts
	based on scope overlap with the work unit being dispatched.
	"""

	id: str = field(default_factory=_new_id)
	item_type: str = ""  # architectural, convention, gotcha, dependency, api, pattern
	scope: str = ""  # comma-separated file paths or module names this applies to
	content: str = ""
	source_unit_id: str = ""  # work unit that produced this context
	round_id: str = ""
	mission_id: str = ""  # mission this context belongs to
	confidence: float = 1.0  # 0.0-1.0, how confident the worker was
	scope_level: str = ""  # mission/round/unit -- filtering granularity
	created_at: str = field(default_factory=_now_iso)


@dataclass
class DiscoveryItem:
	"""A single improvement discovered during codebase analysis."""

	id: str = field(default_factory=_new_id)
	discovery_id: str = ""  # link to parent DiscoveryResult
	track: str = ""  # feature, quality, or security
	title: str = ""
	description: str = ""
	rationale: str = ""
	files_hint: str = ""
	impact: int = 5
	effort: int = 5
	priority_score: float = 0.0  # impact * (11 - effort) / 10
	status: str = "proposed"  # proposed/approved/rejected/completed


@dataclass
class DiscoveryResult:
	"""Result of a discovery run analyzing a target codebase."""

	id: str = field(default_factory=_new_id)
	target_path: str = ""
	timestamp: str = field(default_factory=_now_iso)
	raw_output: str = ""
	model: str = ""
	item_count: int = 0
	error_type: str = ""  # timeout/budget_exceeded/permission_denied/workspace_corruption/unknown
	error_detail: str = ""

