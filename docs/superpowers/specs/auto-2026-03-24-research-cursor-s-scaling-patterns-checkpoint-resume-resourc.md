Looks like the write permission for the specs directory isn't being granted. Here's the complete spec -- I'll output it directly:

---

# Spec: Cursor Scaling Patterns -- Checkpoint/Resume, Resource Budgeting, Failure Cascades

**Source:** Cursor blog, "Scaling long-running autonomous coding" (Jan 14, 2026)
**Target modules:** `src/autodev/swarm/controller.py`, `src/autodev/swarm/stagnation.py`, `src/autodev/swarm/planner.py`
**Priority:** 2 | **Effort:** medium | **Risk:** low

---

## Problem Statement

The autodev swarm controller runs multiple Claude Code agents in parallel, coordinated by a driving planner. While the architecture is sound, several scaling pain points emerge in long-running sessions (hours to days):

1. **No checkpoint/resume for the swarm itself.** If the planner process crashes or is killed, all in-flight state is lost. Agents may still be running as orphan subprocesses, and the planner has no way to reconnect. The only recovery path is restarting from scratch, wasting both time and credits.

2. **Resource budgeting is coarse.** `SwarmConfig.max_agents` is a static ceiling. `get_scaling_recommendation()` (controller.py:1562) uses a simple ratio (`pending > 2 * active` -> scale up), with no awareness of cost rate, token throughput, or diminishing returns. The `BudgetConfig.max_per_run_usd` exists but isn't enforced in the swarm controller's main loop -- agents keep spawning until the planner decides to stop.

3. **Failure cascades are not contained.** When a systemic issue hits (bad merge, broken dependency, corrupted `.venv`), every agent fails in sequence. `stagnation.py` detects the aftermath (high failure rate, repeated errors) but only after multiple cycles of wasted agent spawns. The existing `CircuitBreakerManager` in `circuit_breaker.py` is only wired to the legacy mission mode -- swarm mode has no equivalent.

4. **Stagnation detection is reactive, not predictive.** `analyze_stagnation()` looks at trailing metrics (flat tests, rising cost) but doesn't track leading indicators: token velocity declining, agents spending more time on tool retries, or the same file being modified by successive failing agents.

Cursor's blog describes converging on a planner/worker hierarchy (which autodev already has) but emphasizes patterns we lack: durable checkpoints, adaptive resource budgeting based on throughput, and circuit breakers that halt the swarm before cascading failures burn the budget.

---

## Changes Needed

### 1. Swarm Checkpoint/Resume (`controller.py`, `planner.py`)

**Goal:** Persist enough state that a crashed planner can resume without re-planning or losing track of still-running agents.

#### 1a. Checkpoint Writer

Add a `_write_checkpoint()` method to `SwarmController` that serializes recoverable state to `.autodev-swarm-checkpoint.json` at the project root. Call it:
- After every `execute_decisions()` batch
- After every `monitor_agents()` cycle
- On clean shutdown in `cleanup()`

**File:** `src/autodev/swarm/controller.py`

```python
def _write_checkpoint(self) -> None:
	"""Persist swarm state for crash recovery."""
	checkpoint = {
		"version": 1,
		"run_id": self._run_id,
		"timestamp": _now_iso(),
		"team_name": self._team_name,
		"start_commit": self._start_commit,
		"total_cost_usd": self._total_cost_usd,
		"agent_costs": dict(self._agent_costs),
		"agents": {
			aid: {
				"id": a.id,
				"name": a.name,
				"role": a.role.value,
				"status": a.status.value,
				"current_task_id": a.current_task_id,
				"spawned_at": a.spawned_at,
				"tasks_completed": a.tasks_completed,
				"tasks_failed": a.tasks_failed,
				"pid": self._processes[aid].pid if aid in self._processes and self._processes[aid].returncode is None else None,
			}
			for aid, a in self._agents.items()
		},
		"tasks": {
			tid: {
				"id": t.id,
				"title": t.title,
				"description": t.description,
				"priority": t.priority.value,
				"status": t.status.value,
				"claimed_by": t.claimed_by,
				"depends_on": t.depends_on,
				"files_hint": t.files_hint,
				"attempt_count": t.attempt_count,
				"max_attempts": t.max_attempts,
				"result_summary": t.result_summary,
			}
			for tid, t in self._tasks.items()
		},
		"planner": {
			"cycle_count": None,
			"test_history": [],
			"completion_history": [],
			"failure_history": [],
			"cost_history": [],
		},
	}
	checkpoint_path = Path(self._config.target.resolved_path) / ".autodev-swarm-checkpoint.json"
	tmp_path = checkpoint_path.with_suffix(".tmp")
	tmp_path.write_text(json.dumps(checkpoint, indent=2))
	tmp_path.rename(checkpoint_path)
```

The planner should pass its metrics via a new method `controller.update_checkpoint_planner_state(cycle_count, histories)` before or after checkpoint writes.

#### 1b. Resume from Checkpoint

Add a class method `SwarmController.from_checkpoint(config, swarm_config, db)` that:

1. Reads `.autodev-swarm-checkpoint.json`
2. Rebuilds `self._agents` and `self._tasks` from the persisted dicts
3. For agents with non-None `pid`: checks if the process is still alive via `os.kill(pid, 0)`. If alive, marks it as `WORKING` and monitors via the inbox. If dead, marks as `DEAD`.
4. Restores `self._total_cost_usd`, `self._agent_costs`, `self._start_commit`
5. Returns the controller in a ready state for the planner to resume

Add a companion `DrivingPlanner.resume_from_checkpoint(checkpoint_data)` that restores `_cycle_count`, `_test_history`, `_completion_history`, `_failure_history`, `_cost_history`.

**Limitation:** We cannot reattach to a running Claude Code subprocess's stdout pipe after a planner crash. If the agent's PID is alive, mark it as `WORKING` and rely on inbox messages for status. When it exits, `monitor_agents()` won't have its process handle, so we need a fallback: check the inbox for a completion/failure message from that agent, or after a configurable timeout (default: `task_claim_timeout`), treat it as failed.

#### 1c. CLI Integration

**File:** `src/autodev/cli.py`

- `autodev swarm --resume` reads the checkpoint and calls `SwarmController.from_checkpoint()` + `DrivingPlanner.resume_from_checkpoint()` instead of `initialize()` + `_initial_plan()`.
- If no checkpoint exists, print error and exit.
- On successful resume, delete the checkpoint file (the running loop creates fresh ones).

#### 1d. Orphan Detection

Add `_detect_orphan_agents()` to `SwarmController.initialize()`:

- On fresh start, read any existing `.autodev-swarm-checkpoint.json`
- If it exists and has agents with PIDs: check if those PIDs are alive
- If alive, warn the user and offer to kill them (via Telegram notification if configured)
- If dead, clean up the stale checkpoint

---

### 2. Adaptive Resource Budgeting (`controller.py`, `planner.py`)

**Goal:** Enforce cost ceilings and adjust agent count dynamically based on throughput efficiency, not just queue depth.

#### 2a. Cost Ceiling Enforcement

**File:** `src/autodev/swarm/controller.py`

Add `_check_budget()` method:

```python
def _check_budget(self) -> bool:
	"""Return True if budget allows spawning another agent."""
	max_run = self._config.budget.max_per_run_usd
	if max_run > 0 and self._total_cost_usd >= max_run:
		logger.warning("Budget ceiling reached: $%.2f >= $%.2f", self._total_cost_usd, max_run)
		return False
	dead_agents = [a for a in self._agents.values() if a.status == AgentStatus.DEAD]
	if dead_agents:
		avg_cost = self._total_cost_usd / len(dead_agents)
		active_count = sum(1 for a in self._agents.values() if a.status == AgentStatus.WORKING)
		projected = self._total_cost_usd + avg_cost * (active_count + 1)
		if max_run > 0 and projected > max_run * 0.9:
			logger.warning("Projected cost $%.2f would exceed 90%% of budget $%.2f", projected, max_run)
			return False
	return True
```

In `_handle_spawn()`, before the existing `max_agents` check (line 284), call `_check_budget()`. If `False`, return `{"error": "budget_exceeded", "spawned": False}`.

#### 2b. Throughput-Based Scaling

Replace `get_scaling_recommendation()` (controller.py:1562):

```python
def get_scaling_recommendation(self) -> dict[str, Any]:
	"""Throughput-aware scaling recommendation."""
	active = [a for a in self._agents.values() if a.status in (AgentStatus.WORKING, AgentStatus.SPAWNING)]
	pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
	dead = [a for a in self._agents.values() if a.status == AgentStatus.DEAD]

	rec: dict[str, Any] = {"scale_up": 0, "scale_down": 0, "reason": ""}

	if dead and self._total_cost_usd > 1.0:
		total_completed = sum(a.tasks_completed for a in dead)
		efficiency = total_completed / self._total_cost_usd
		rec["tasks_per_dollar"] = round(efficiency, 2)
		if efficiency < 0.5 and len(active) >= 3:
			rec["scale_down"] = 1
			rec["reason"] = f"Low efficiency ({efficiency:.2f} tasks/$)"
			return rec

	if len(pending) > 2 * max(len(active), 1):
		if self._check_budget():
			rec["scale_up"] = min(len(pending) - len(active), 3)
			rec["reason"] = f"{len(pending)} pending vs {len(active)} active"
		else:
			rec["reason"] = "Would scale up but budget constrained"

	idle = [a for a in self._agents.values() if a.status == AgentStatus.IDLE]
	if len(idle) >= 2:
		rec["scale_down"] = len(idle)
		rec["reason"] = f"{len(idle)} idle agents"
	return rec
```

Return type changes from `dict[str, int]` to `dict[str, Any]`. Update the caller in `planner.py:_plan_cycle()` (line 633):

```python
scaling = self._controller.get_scaling_recommendation()
if scaling.get("reason"):
	state_text += f"\n\n## Scaling Signal\n{scaling['reason']}"
	if scaling.get("tasks_per_dollar") is not None:
		state_text += f" (efficiency: {scaling['tasks_per_dollar']} tasks/$)"
```

#### 2c. Budget Warning in Planner Context

**File:** `src/autodev/swarm/planner.py` -- `_plan_cycle()`, after scaling injection:

```python
budget_max = self._controller._config.budget.max_per_run_usd
if budget_max > 0:
	budget_pct = state.total_cost_usd / budget_max * 100
	if budget_pct > 75:
		state_text += (
			f"\n\n## BUDGET WARNING\n"
			f"Spent ${state.total_cost_usd:.2f} of ${budget_max:.2f} ({budget_pct:.0f}%). "
			f"Prioritize high-value tasks. Reduce agent count if possible."
		)
```

---

### 3. Failure Cascade Circuit Breaker (new: `swarm/circuit_breaker.py`, `controller.py`)

**Goal:** Halt agent spawning when a systemic failure is detected, preventing the swarm from burning budget on doomed tasks.

#### 3a. Swarm-Level Circuit Breaker

Create **`src/autodev/swarm/circuit_breaker.py`** adapting the existing `CircuitBreakerManager` pattern from `src/autodev/circuit_breaker.py`:

```python
"""Swarm-level circuit breaker for failure cascade prevention."""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SwarmCircuitBreakerConfig:
	max_consecutive_failures: int = 3
	cooldown_seconds: float = 180.0
	max_failures_per_window: int = 5
	window_seconds: float = 300.0

class SwarmCircuitBreaker:
	def __init__(self, config: SwarmCircuitBreakerConfig | None = None) -> None:
		self._config = config or SwarmCircuitBreakerConfig()
		self._consecutive_failures = 0
		self._failure_timestamps: list[float] = []
		self._tripped = False
		self._tripped_at: float = 0.0
		self._trip_reason: str = ""

	def record_success(self) -> None:
		self._consecutive_failures = 0

	def record_failure(self) -> None:
		now = time.monotonic()
		self._consecutive_failures += 1
		self._failure_timestamps.append(now)
		cutoff = now - self._config.window_seconds
		self._failure_timestamps = [t for t in self._failure_timestamps if t > cutoff]
		if self._consecutive_failures >= self._config.max_consecutive_failures:
			self._trip(f"{self._consecutive_failures} consecutive agent failures")
		elif len(self._failure_timestamps) >= self._config.max_failures_per_window:
			self._trip(f"{len(self._failure_timestamps)} failures in {self._config.window_seconds:.0f}s window")

	def can_spawn(self) -> tuple[bool, str]:
		if not self._tripped:
			return True, ""
		elapsed = time.monotonic() - self._tripped_at
		if elapsed >= self._config.cooldown_seconds:
			self._tripped = False
			self._consecutive_failures = 0
			self._trip_reason = ""
			return True, ""
		remaining = self._config.cooldown_seconds - elapsed
		return False, f"Circuit breaker tripped ({self._trip_reason}), {remaining:.0f}s remaining"

	def _trip(self, reason: str) -> None:
		if not self._tripped:
			logger.warning("Swarm circuit breaker TRIPPED: %s", reason)
		self._tripped = True
		self._tripped_at = time.monotonic()
		self._trip_reason = reason

	@property
	def is_tripped(self) -> bool:
		return self._tripped

	@property
	def trip_reason(self) -> str:
		return self._trip_reason
```

#### 3b. Wire into Controller

**File:** `src/autodev/swarm/controller.py`

- `__init__()` (line ~87): `self._circuit_breaker = SwarmCircuitBreaker()`
- `_handle_spawn()` (line ~284): after max_agents check, add circuit breaker check
- `monitor_agents()` (line ~956): after agent status determination, call `record_success()` / `record_failure()`

#### 3c. Surface in Planner Context

**File:** `src/autodev/swarm/planner.py` -- `_plan_cycle()`:

```python
if self._controller._circuit_breaker.is_tripped:
	state_text += (
		f"\n\n## CIRCUIT BREAKER TRIPPED\n"
		f"Reason: {self._controller._circuit_breaker.trip_reason}\n"
		"Spawning is halted. Focus on diagnosing the root cause.\n"
		"Consider: create a diagnostic task, reduce scope, or escalate."
	)
```

---

### 4. Predictive Stagnation Signals (`stagnation.py`)

**Goal:** Detect leading indicators of stagnation before metrics go flat.

#### 4a. New Config Fields

**File:** `src/autodev/swarm/stagnation.py` -- `StagnationConfig`

```python
cost_per_completion_threshold: float = 5.0
file_hotspot_threshold: int = 3
```

#### 4b. New Detection Functions

**File:** `src/autodev/swarm/stagnation.py`

`_check_cost_efficiency(completion_history, cost_history, cfg, pivots)` -- compares first-half vs second-half efficiency. If second half drops below 50% of first half, emits `reduce_and_focus` pivot. Guards on `len >= 4` to avoid early false positives.

`_check_file_hotspots(file_changes, cfg, pivots)` -- inverts the `{agent: [files]}` map to `{file: {agents}}`, flags files touched by >= `file_hotspot_threshold` agents. Emits `serialize_hotspot` pivot.

#### 4c. Extended `analyze_stagnation()` Signature

Add `file_changes: dict[str, list[str]] | None = None` parameter. Call new detectors before `return pivots`.

#### 4d. Wire into Planner

**File:** `src/autodev/swarm/planner.py` -- `_plan_cycle()` (line ~618)

```python
pivots = analyze_stagnation(
	...,
	file_changes=dict(self._controller._recent_changes),
)
```

#### 4e. New Pivot Strategy: `serialize_hotspot`

**File:** `src/autodev/swarm/stagnation.py` -- `pivots_to_decisions()`

```python
elif p.strategy == "serialize_hotspot":
	decisions.append({
		"type": "adjust",
		"payload": {"max_agents": 1},
		"reasoning": f"Pivot: {p.trigger}. Serializing to avoid conflicts.",
		"priority": 9,
	})
```

---

## Testing Requirements

### Unit Tests

**`tests/test_swarm_checkpoint.py`** (new)
1. `test_write_checkpoint_creates_valid_json` -- schema validation
2. `test_resume_restores_tasks` -- roundtrip task state
3. `test_resume_marks_dead_pid_as_dead` -- non-existent PID -> DEAD
4. `test_resume_no_checkpoint_raises` -- clean error without file
5. `test_checkpoint_atomic_write` -- tmp+rename pattern

**`tests/test_swarm_circuit_breaker.py`** (new)
6. `test_consecutive_failures_trip` -- 3 failures -> blocked
7. `test_success_resets_consecutive` -- interleaved success resets counter
8. `test_cooldown_allows_retry` -- time advance past cooldown -> unblocked
9. `test_window_failures_trip` -- 5 failures in window -> blocked
10. `test_window_expiry` -- old failures outside window don't count

**`tests/test_stagnation.py`** (extend)
11. `test_cost_efficiency_decline_detected` -- declining efficiency triggers pivot
12. `test_file_hotspot_detected` -- 3+ agents on same file triggers pivot
13. `test_serialize_hotspot_decision` -- produces `adjust` with `max_agents: 1`
14. `test_cost_efficiency_insufficient_data` -- short histories -> no false positive

**`tests/test_swarm_controller.py`** (extend)
15. `test_budget_ceiling_blocks_spawn` -- cost > max -> spawn rejected
16. `test_circuit_breaker_blocks_spawn` -- tripped breaker -> spawn rejected
17. `test_scaling_recommendation_with_efficiency` -- efficiency in output

---

## Risk Assessment

### Low Risk
- **Checkpoint file corruption.** Atomic write (tmp + rename). Corrupted checkpoint = start fresh (current behavior).
- **Circuit breaker false positive.** 180s cooldown auto-resets. 3 consecutive failures is conservative. Trip is surfaced to planner for diagnosis.
- **Stagnation false positives.** `len >= 4` guard and half-split comparison prevent early triggers.

### Medium Risk
- **PID reuse on resume.** Stale PID might match a different process. Mitigation: verify process command contains "claude". If uncertain, treat as dead -- task requeues via `requeue_failed_tasks()`.
- **`get_scaling_recommendation()` return type change.** `dict[str, int]` -> `dict[str, Any]`. Only caller uses `.get()` so backward-compatible, but external consumers would need updating.
- **Budget enforcement creating a dead swarm.** If budget is too low, no agents spawn. Mitigation: planner sees "budget_exceeded" and can escalate. Budget of 0 (default) means unlimited.

### Not In Scope
- Reconnecting to running subprocess stdout pipes (architecturally impossible after crash)
- Distributed checkpointing across machines (local files only)
- TOML configuration for circuit breaker thresholds (in-code defaults; `[swarm.circuit_breaker]` deferred)