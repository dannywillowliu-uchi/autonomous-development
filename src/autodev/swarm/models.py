"""Data models for swarm coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from autodev.swarm.capabilities import CapabilityManifest
from uuid import uuid4


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
	return uuid4().hex[:12]


# -- Planner Decisions --


class DecisionType(str, Enum):
	"""Actions the planner can take."""

	SPAWN = "spawn"
	KILL = "kill"
	REDIRECT = "redirect"
	CREATE_TASK = "create_task"
	ADJUST = "adjust"
	WAIT = "wait"
	ESCALATE = "escalate"
	CREATE_SKILL = "create_skill"
	CREATE_HOOK = "create_hook"
	REGISTER_MCP = "register_mcp"
	CREATE_AGENT_DEF = "create_agent_def"
	USE_SKILL = "use_skill"


@dataclass
class PlannerDecision:
	"""A single structured decision from the planner."""

	type: DecisionType
	payload: dict[str, Any] = field(default_factory=dict)
	reasoning: str = ""
	priority: int = 0  # higher = more urgent


# -- Agent State --


class AgentRole(str, Enum):
	"""Roles an agent can fill in the swarm."""

	IMPLEMENTER = "implementer"
	RESEARCHER = "researcher"
	TESTER = "tester"
	REVIEWER = "reviewer"
	DEBUGGER = "debugger"
	DESIGNER = "designer"
	GENERAL = "general"


class AgentStatus(str, Enum):
	"""Lifecycle states for a swarm agent."""

	SPAWNING = "spawning"
	IDLE = "idle"
	WORKING = "working"
	BLOCKED = "blocked"
	SHUTTING_DOWN = "shutting_down"
	DEAD = "dead"


@dataclass
class SwarmAgent:
	"""Tracks a single agent in the swarm."""

	id: str = field(default_factory=_new_id)
	name: str = ""
	role: AgentRole = AgentRole.GENERAL
	status: AgentStatus = AgentStatus.SPAWNING
	current_task_id: str | None = None
	capabilities: list[str] = field(default_factory=list)
	spawned_at: str = field(default_factory=_now_iso)
	last_heartbeat: str = field(default_factory=_now_iso)
	messages_sent: int = 0
	messages_received: int = 0
	tasks_completed: int = 0
	tasks_failed: int = 0
	death_time: float | None = None


# -- Task Pool --


class TaskPriority(int, Enum):
	"""Task priority levels."""

	LOW = 0
	NORMAL = 1
	HIGH = 2
	CRITICAL = 3


class TaskStatus(str, Enum):
	"""Task lifecycle states."""

	PENDING = "pending"
	CLAIMED = "claimed"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"
	BLOCKED = "blocked"
	CANCELLED = "cancelled"


@dataclass
class SwarmTask:
	"""A task in the shared task pool."""

	id: str = field(default_factory=_new_id)
	title: str = ""
	description: str = ""
	priority: TaskPriority = TaskPriority.NORMAL
	status: TaskStatus = TaskStatus.PENDING
	claimed_by: str | None = None
	claimed_at: str | None = None
	depends_on: list[str] = field(default_factory=list)
	files_hint: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=_now_iso)
	completed_at: str | None = None
	result_summary: str = ""
	attempt_count: int = 0
	max_attempts: int = 3


# -- Swarm State (snapshot for planner) --


@dataclass
class StagnationSignal:
	"""Indicates the swarm is not making progress."""

	metric: str = ""  # e.g. "test_pass_count", "merge_count"
	value_history: list[float] = field(default_factory=list)  # last N values
	cycles_stagnant: int = 0
	suggested_pivot: str = ""


@dataclass
class SwarmState:
	"""Complete swarm state snapshot for a planner cycle."""

	mission_objective: str = ""
	agents: list[SwarmAgent] = field(default_factory=list)
	tasks: list[SwarmTask] = field(default_factory=list)
	recent_completions: list[dict[str, Any]] = field(default_factory=list)
	recent_failures: list[dict[str, Any]] = field(default_factory=list)
	recent_discoveries: list[str] = field(default_factory=list)
	available_skills: list[str] = field(default_factory=list)
	available_tools: list[str] = field(default_factory=list)
	stagnation_signals: list[StagnationSignal] = field(default_factory=list)
	core_test_results: dict[str, Any] = field(default_factory=dict)
	cycle_number: int = 0
	total_cost_usd: float = 0.0
	wall_time_seconds: float = 0.0
	files_in_flight: list[str] = field(default_factory=list)
	capabilities: CapabilityManifest | None = None
	dead_agent_history: list[SwarmAgent] = field(default_factory=list)
	recent_file_changes: dict[str, list[str]] = field(default_factory=dict)
	agent_costs: dict[str, float] = field(default_factory=dict)
