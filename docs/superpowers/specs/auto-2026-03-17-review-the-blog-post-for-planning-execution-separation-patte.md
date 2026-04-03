# Spec: Planning/Execution Separation Improvements

## Problem Statement

autodev's swarm planner (`swarm/planner.py`) and deliberative planner (`deliberative_planner.py`) both conflate research, planning, and execution into tightly coupled cycles. The driving planner's `_initial_plan()` jumps directly from state observation to task decomposition without a dedicated research phase. The deliberative planner's critic checks feasibility but not plan quality or completeness.

Boris Tane's workflow identifies three patterns autodev lacks:

1. **Research-before-planning**: A dedicated deep-read phase that produces persistent artifacts before any planning begins. autodev's planners receive state snapshots but never explicitly research the codebase or problem domain first.
2. **Plan-as-reviewable-artifact**: Plans written to persistent markdown files that can be reviewed, annotated, and refined 1-6 times before execution. autodev's plans are ephemeral LLM outputs parsed into `PlannerDecision` objects and immediately executed.
3. **Annotation/refinement cycle**: Iterative plan improvement with explicit quality gates between planning and execution. autodev's two-step analysis->decision pipeline is a single pass with no refinement loop.

Applying these patterns would reduce wasted agent spawns (workers executing poorly-researched tasks), improve task decomposition quality (plans informed by actual codebase research), and create an audit trail of planning artifacts for debugging and learning.

## Changes Needed

### 1. Research Phase in Swarm Planner

**File:** `src/autodev/swarm/planner.py`

Add a `_research_phase()` method called before `_initial_plan()` in the `run()` method. This phase:

- Dispatches 1-2 lightweight research agents (reusing the existing `spawn_agent` machinery in `swarm/controller.py`) with read-only directives targeting the objective's relevant codebase areas
- Agents write findings to `{target_path}/.autodev-swarm-research/{objective_slug}/research-{topic}.md`
- Planner waits for research agents to complete (with a configurable timeout, default 5 minutes)
- Research findings are injected into the initial planning prompt as structured context

```python
async def _research_phase(self, state: SwarmState) -> str:
	"""Run research agents before initial planning. Returns research summary."""
	# Generate research directives from objective
	# Spawn read-only research agents
	# Wait for completion (with timeout)
	# Collect and concatenate research artifacts
	# Return formatted research context for planning prompt
```

**Modify `run()`:**
```python
async def run(self, core_test_runner=None):
	self._running = True
	state = self._build_state(core_test_runner)

	# NEW: Research phase before planning
	research_context = await self._research_phase(state)

	decisions = await self._initial_plan(state, research_context=research_context)
	# ... rest unchanged
```

**Modify `_initial_plan()`** to accept and inject `research_context: str = ""` into the prompt.

### 2. Persistent Plan Artifacts

**File:** `src/autodev/swarm/planner.py`

Add plan persistence so every planning output is written to disk before execution.

**New method:**
```python
def _persist_plan(self, decisions: list[PlannerDecision], cycle: int, phase: str) -> Path:
	"""Write plan to .autodev-swarm-plans/{cycle:04d}-{phase}.md"""
```

**Directory structure:**
```
{target_path}/.autodev-swarm-plans/
  0000-initial.md          # Initial plan
  0001-cycle.md            # Cycle refinement
  research/                # Symlink to .autodev-swarm-research/
```

Each plan file contains:
- Timestamp, cycle number, state summary hash
- Decisions in human-readable format (task descriptions, assignments, rationale)
- Raw JSON block for machine parsing

**Modify `_initial_plan()` and `_plan_cycle()`** to call `_persist_plan()` before returning decisions.

### 3. Plan Refinement Loop

**File:** `src/autodev/swarm/planner.py`

Add an optional refinement pass between analysis and decision execution. This mirrors autodev's existing two-step pipeline (ANALYSIS_PROMPT_TEMPLATE -> DECISION_FROM_ANALYSIS_PROMPT) but adds an explicit quality gate.

**New method:**
```python
async def _refine_plan(
	self,
	decisions: list[PlannerDecision],
	state: SwarmState,
	research_context: str,
	max_rounds: int = 2,
) -> list[PlannerDecision]:
	"""Refine plan through self-critique. Returns improved decisions."""
```

The refinement prompt asks the LLM to review its own plan against:
- Research findings (do tasks target the right files/modules?)
- File overlap (will parallel workers conflict? uses existing `overlap.py`)
- Task granularity (are tasks too broad or too narrow?)
- Dependency ordering (are prerequisites correctly sequenced?)

**Config:** Add `plan_refinement_rounds: int = 1` to `SwarmConfig` (0 disables refinement, max 3).

### 4. Research Prompt Template

**File:** `src/autodev/swarm/prompts.py`

Add a new prompt template:

```python
RESEARCH_DIRECTIVE_PROMPT = """You are a research agent. Your job is to deeply understand the relevant parts of the codebase before any planning or implementation begins.

Objective: {objective}

Your research directive:
{directive}

Instructions:
1. Read the relevant files thoroughly. Understand structure, patterns, and edge cases.
2. Write your findings to {output_path} in markdown format.
3. Include: file inventory, key abstractions, data flow, gotchas, and dependencies.
4. Do NOT modify any code. Do NOT suggest changes. Only research and document.
5. Be specific -- cite file paths and line ranges, not vague descriptions.
"""
```

Add a plan refinement prompt:

```python
PLAN_REFINEMENT_PROMPT = """Review the following plan for quality before execution.

Research findings:
{research_context}

Current plan:
{plan_summary}

Evaluate against:
1. Do tasks target the correct files based on research? Flag any mismatches.
2. Will parallel tasks cause file conflicts? (High overlap = reduce parallelism)
3. Are tasks appropriately scoped? (Neither too broad nor too narrow)
4. Are dependencies correctly ordered?

Output a REVISED plan using the same JSON decision format, or output APPROVED if the plan is acceptable as-is.
"""
```

### 5. Planner Configuration

**File:** `src/autodev/config.py`

Add fields to `SwarmConfig`:

```python
# Planning phase isolation
research_phase_enabled: bool = True
research_timeout_seconds: int = 300
research_max_agents: int = 2
plan_refinement_rounds: int = 1  # 0 to disable, max 3
plan_persistence_enabled: bool = True
```

### 6. Deliberative Planner Alignment (Mission Mode)

**File:** `src/autodev/deliberative_planner.py`

The deliberative planner already has a critic loop, which is structurally similar to the annotation cycle. Two targeted improvements:

**a) Inject research context into planner proposals.** Modify the initial proposal prompt to accept an optional `research_context` parameter, populated by running a research subprocess before deliberation begins.

**b) Persist critic exchange.** Write the planner-critic dialogue to `{target_path}/.autodev-mission-plans/{mission_id}/deliberation-{round}.md` so the full planning rationale is auditable.

These are smaller changes since the deliberative planner already separates planning from execution more than the swarm planner does.

### 7. Worker Prompt Research Injection

**File:** `src/autodev/swarm/worker_prompt.py`

When research artifacts exist for the current objective, include a `## Research Context` section in worker prompts. This gives workers the same codebase understanding the planner used, reducing redundant exploration.

Add to the worker prompt builder:

```python
def _build_research_section(self, task_description: str) -> str:
	"""Load relevant research artifacts and format for worker context."""
	research_dir = self._target_path / ".autodev-swarm-research"
	if not research_dir.exists():
		return ""
	# Find research files, extract sections relevant to this task
	# Return formatted markdown section
```

## Testing Requirements

### Unit Tests

**File:** `tests/test_planner_research.py`

1. `test_research_phase_produces_artifacts` -- Mock controller's `spawn_agent`, verify research files are written to `.autodev-swarm-research/`
2. `test_research_timeout_does_not_block_planning` -- Verify planning proceeds after timeout even if research agents haven't finished
3. `test_research_context_injected_into_initial_plan` -- Verify the initial planning prompt includes research findings
4. `test_research_phase_disabled_by_config` -- Verify `research_phase_enabled=False` skips research entirely

**File:** `tests/test_plan_persistence.py`

5. `test_plan_persisted_to_disk` -- Verify `_persist_plan()` writes valid markdown with JSON block
6. `test_plan_files_incrementally_numbered` -- Verify sequential cycle plans get sequential filenames
7. `test_plan_roundtrip` -- Verify persisted plans can be re-parsed into `PlannerDecision` objects

**File:** `tests/test_plan_refinement.py`

8. `test_refinement_improves_plan` -- Mock LLM to return a revised plan, verify decisions are updated
9. `test_refinement_approved_returns_original` -- Mock LLM returning "APPROVED", verify original decisions pass through
10. `test_refinement_rounds_bounded` -- Verify refinement stops after `plan_refinement_rounds` even if LLM keeps suggesting changes
11. `test_refinement_detects_file_overlap` -- Verify high-overlap tasks trigger a warning or are serialized

**File:** `tests/test_config_planning.py`

12. `test_planning_config_defaults` -- Verify new config fields have correct defaults
13. `test_planning_config_from_toml` -- Verify TOML parsing of new fields

### Integration Test

**File:** `tests/test_swarm_planning_integration.py`

14. `test_full_research_plan_execute_cycle` -- End-to-end test with mocked LLM and subprocess: research phase -> plan persistence -> refinement -> decision execution. Verify artifacts exist, decisions are valid, and no code is written during research/planning phases.

### Verification Criteria

- All existing tests in `tests/` pass (no regressions)
- New tests cover research phase, plan persistence, and refinement
- `ruff check src/ tests/` passes
- `bandit -r src/ -lll -q` passes
- Plans are written to disk before any agent is spawned for execution
- Research agents are spawned with read-only constraints (no Edit/Write tools)
- Refinement loop terminates within bounded rounds
- Config defaults maintain backward compatibility (existing swarm behavior unchanged when new features are at defaults)

## Risk Assessment

### Low Risk

**Backward compatibility**: All new features are off-by-default or have safe defaults (`research_phase_enabled=True` but with a timeout that falls through gracefully, `plan_refinement_rounds=1` which adds one extra LLM call). Existing swarm behavior is preserved when config is unchanged.

**Plan persistence disk usage**: Plan files are small markdown documents. Even aggressive use (100 cycles) produces <1MB. The `.autodev-swarm-plans/` directory is gitignored by convention.

### Medium Risk

**Research phase latency**: Adding a research phase before initial planning adds 2-5 minutes of wall time. Mitigations:
- Configurable timeout (`research_timeout_seconds`, default 300)
- Research agents are lightweight read-only subprocesses, not full workers
- Can be disabled entirely via config
- Timeout fallback: planning proceeds without research context if agents don't finish

**LLM token cost of refinement**: Each refinement round is one additional LLM call. With `plan_refinement_rounds=1` (default), this adds ~1 call per initial plan and ~0 per cycle (refinement only runs on initial plans by default). At worst, 3 rounds = 3 extra calls.

**Refinement loop divergence**: The refinement LLM could keep suggesting changes indefinitely. Mitigation: hard cap at `max_rounds` (config-enforced max of 3), plus the "APPROVED" escape hatch in the prompt.

### Mitigations

- Feature flags for every new capability (research, refinement, persistence)
- Timeouts on all async operations (research agents, refinement rounds)
- Graceful degradation: if research fails, planning proceeds without it; if refinement fails, original plan executes
- No changes to the hot path (main cycle loop) when features are disabled
- Research agents use existing `spawn_agent` infrastructure -- no new subprocess machinery