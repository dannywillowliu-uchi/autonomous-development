# Autonomous Dev Scheduler -- Improvement Backlog

Derived from cross-system research (Feb 2026). Ordered by priority.

## P0: Priority Queue and Cross-Mission Scheduling System

**Problem**: Priorities are set once at discovery time and never revisited. BACKLOG.md is a markdown file with human-assigned P0-P9 labels that the scheduler doesn't parse or enforce. Each mission rediscovers from scratch with no memory of what was planned before. There's no way to reprioritize items between missions, no feedback loop for items that keep failing, and no human override mechanism.

**Solution**: Build a structured priority queue that persists across missions and supports reprioritization:

1. **Persistent backlog store**: SQLite table `backlog_items` with fields: id, title, description, priority_score, impact, effort, track, status (pending/in_progress/completed/deferred), source_mission_id, created_at, updated_at, attempt_count, last_failure_reason
2. **Priority recalculation**: After each mission, recalculate priorities based on:
   - Original impact/effort score
   - Failure history (items that failed get deprioritized unless the failure was infrastructure-related)
   - Dependency satisfaction (items whose blockers are now resolved get boosted)
   - Staleness penalty (items sitting untouched for N missions get flagged)
3. **Human override CLI**: `mc priority set <item-id> <score>` to manually pin an item's priority. `mc priority list` to show the current queue. `mc priority defer <item-id>` to push to bottom.
4. **Mission intake**: When a mission starts, instead of rediscovering everything, it reads the priority queue first and selects the top N items. Discovery adds NEW items to the queue rather than replacing it.
5. **Cross-mission continuity**: Post-mission discovery items get inserted into the persistent queue, not just logged to the DB. Next mission picks up where the last left off.

**Files**: `db.py` (new table), `models.py` (BacklogItem model), `cli.py` (priority subcommand), `auto_discovery.py` (integrate with queue), `continuous_controller.py` (read from queue at mission start)

**Why P0**: Without this, every other improvement is ad-hoc. The scheduler can't strategically sequence its own evolution. This is the foundation for the strategist agent and mission chaining.

---

## P1 (was P0): Replace LLM Evaluator with Objective Signals

**Problem**: The evaluator spawns a full Claude session per round to assign a subjective 0.0-1.0 score. This is expensive, noisy (Round 2 scored 0.0 despite real work), and contradicts our "objective signals only" design.

**Solution**: Replace `evaluator.py`'s LLM call with a deterministic scoring function that uses:
- Test count delta from verification snapshots (passing tests before vs after)
- Lint/type error delta
- Verification pass/fail (binary)
- Completion rate (units completed / planned)
- Files changed count

**Formula**: `score = 0.4 * test_improvement + 0.2 * lint_improvement + 0.2 * completion_rate + 0.2 * no_regression`

Where each component is 0.0-1.0 based on objective deltas from SnapshotDelta.

**Files**: `evaluator.py` (rewrite), `round_controller.py` (simplify evaluate step), `tests/test_evaluator.py`

**Source inspiration**: Our own feedback.py already computes objective rewards -- the evaluator should use the same pattern.

---

## P1: N-of-M Candidate Selection for Fixup

**Problem**: Fixup agent makes one attempt to fix verification failures. If it fails, it retries the same approach. No diversity in solutions.

**Solution**: Generate N candidate fix patches (N=3), filter by syntax validity and test regression, then select the best via verification score comparison. If multiple pass, pick the one with the best test delta.

**Implementation**:
- In `green_branch.py` `run_fixup()`: spawn N parallel fixup agents with slightly different prompts (vary temperature/approach hints)
- Each produces a patch on a temporary branch
- Run verification on each
- Select the best-scoring candidate
- Merge the winner

**Files**: `green_branch.py`, `config.py` (add `fixup_candidates` config)

**Source inspiration**: Agentless (UIUC) -- tournament-style patch selection accounts for 17% improvement on SWE-bench.

---

## P2: Architect/Editor Model Split for Workers

**Problem**: Workers use the same model for both reasoning about approach and writing code. This is suboptimal -- reasoning benefits from stronger models, editing is more mechanical.

**Solution**: Add per-component model configuration:
- `planner_model`: Opus (complex decomposition)
- `worker_model`: Opus (execution)
- `evaluator_model`: N/A after P0 (deterministic)
- `fixup_model`: Opus (needs reasoning about failures)

Within workers, optionally split into two passes:
1. Architect pass: "Analyze the codebase and describe what changes are needed" (reasoning model)
2. Editor pass: "Implement these specific changes" (editing model)

**Files**: `config.py` (per-component model fields), `worker.py`, `recursive_planner.py`, `green_branch.py`

**Source inspiration**: Aider's architect/editor pattern achieves SOTA results.

---

## P3: Structured Schedule Output for Planner

**Problem**: Planner output parsing fails sometimes ("Failed to parse planner output, falling back to single leaf"). The current approach tries to parse the entire LLM response as structured data.

**Solution**: Use embedded structured blocks in natural language output (like thebotcompany's `<!-- SCHEDULE -->` pattern). The planner reasons in prose, then emits a machine-readable plan block:

```
I analyzed the codebase and determined we need 3 work units...

<!-- PLAN -->
{"units": [{"title": "...", "scope": "...", "files_hint": "..."}]}
<!-- /PLAN -->
```

Parse only the embedded block, ignore the surrounding reasoning.

**Files**: `recursive_planner.py` (prompt + parser), `session.py` (MC_RESULT already uses a similar pattern)

**Source inspiration**: TheBotCompany (syifan) -- structured schedule blocks embedded in agent output.

---

## P4: EMA Budget Tracking

**Problem**: No cost tracking or budget enforcement. We set max_per_session_usd and max_per_run_usd in config but don't track actual spending.

**Solution**: Track per-cycle costs with Exponential Moving Average:
- Log cost per round to SQLite
- Compute EMA with alpha=0.3
- Outlier dampening: spikes >3x EMA clamped to 2x (after 3+ data points)
- Conservatism factor: `k = 1.0 + 0.5/sqrt(n)`
- Adaptive cooldown between rounds based on remaining budget

**Files**: `db.py` (cost tracking table), `round_controller.py` (budget checks), `config.py`

**Source inspiration**: TheBotCompany -- production-ready adaptive budget system.

---

## P5: Auto-Pause and Recovery

**Problem**: If all workers fail or infrastructure errors occur, the mission just fails or continues blindly.

**Solution**: Implement backoff + retry with condition checking:
- If all units in a round fail: pause for configurable interval, then retry
- If verification fails N times consecutively: pause and retry
- Condition function checks if underlying issue resolved before retry
- Max retries before escalation

**Files**: `round_controller.py`, `green_branch.py`, `config.py`

**Source inspiration**: TheBotCompany -- auto-pause on total failure.

---

## P6: Typed Context Store

**Problem**: Workers get flat text context via `memory.py`. No structure, no selective injection, no persistence within a round.

**Solution**: Replace flat context with typed items:
```python
@dataclass
class ContextItem:
    type: str  # "architecture", "gotcha", "pattern", "dependency"
    scope: str  # file/module/feature this relates to
    content: str
    source_unit_id: str
    round_id: str
    confidence: float  # 0.0-1.0
```

Store in SQLite. Workers produce context items as discoveries. Coordinator selectively injects relevant items into subsequent worker prompts based on scope overlap with the work unit.

**Files**: `models.py`, `db.py`, `memory.py` (rewrite), `round_controller.py`, `worker.py`

**Source inspiration**: Danau5tin/multi-agent-coding-system -- Context Store pattern.

---

## P7: Dynamic Agent Composition

**Problem**: All workers use the same prompt template. No specialization based on task type.

**Solution**: Define worker specializations as markdown templates:
- `test-writer.md` -- specialist for test coverage
- `refactorer.md` -- specialist for code cleanup
- `debugger.md` -- specialist for fixing failures

Let the planner select which specialist to assign to each work unit based on the task type.

**Files**: `worker.py`, `recursive_planner.py`, config directory for specialist templates

**Source inspiration**: TheBotCompany -- Ares creates/deletes worker skill files; GPT Pilot -- 14 specialized agents.

---

## P8: Runtime Tool Synthesis

**Problem**: Workers use a fixed set of tools. No adaptation to project-specific needs.

**Solution**: Add mid-task reflection checkpoint: "Would creating a custom tool accelerate this work?" Workers can create project-specific helpers (custom linters, test generators, analyzers) that persist for the duration of the round.

**Files**: `worker.py` (reflection prompt), tool persistence mechanism

**Source inspiration**: Live-SWE-agent -- self-evolving scaffold, 12% solve rate improvement from reflection-prompted tool creation.

---

## P9: Self-Play Training Loop (Long-term)

**Problem**: System doesn't improve its own prompts or strategies based on accumulated data.

**Solution**: Use the system to generate bug-injection / bug-fix pairs:
1. Agent A introduces a realistic bug into the codebase
2. Agent B attempts to find and fix it
3. Success/failure signal trains better prompts

Start with prompt-level self-play (no weight updates). The Self-Improving Coding Agent showed 17-53% gains from prompt/code self-editing alone.

**Files**: New module `self_play.py`, integration with feedback system

**Source inspiration**: Meta FAIR Self-Play SWE-RL, Self-Improving Coding Agent (ICLR 2025 Workshop).

## Next Mission: Traceability & Dashboard Features

- **MISSION_STATE.md generation**: Auto-generate and commit a MISSION_STATE.md file at mission start and update throughout execution. Contains: objective, discovered work items, status of each item, timestamps.
- **Dashboard mission statement view**: Add a section to the live web dashboard (live_ui.html) that displays the current mission objective and planned work items. Should be visible at the top of the dashboard alongside the status bar.

---

## Next Mission: Autonomous Engineering Lead Capabilities

The goal is to evolve mission control from a task executor into an autonomous engineering lead that can do what a human currently does manually.

### Strategic Decision-Making
**Problem**: The human currently decides _what_ to build and _how_ to architect it. Mission control only executes predefined objectives.

**Solution**: Add a strategy layer that can:
- Evaluate the codebase holistically (using analyze_codebase) and identify the highest-impact next step
- Choose between competing architectural approaches by reasoning about trade-offs (maintainability, performance, complexity)
- Make design decisions within a mission without human approval for non-destructive choices
- Maintain a rolling "strategic context" that accumulates across missions -- what worked, what didn't, what the long-term direction is

**Implementation**: A "strategist" agent that runs before discovery, reads BACKLOG.md + git history + past mission reports, and produces a focused objective autonomously. Checkpointed: human approves the objective, but the system proposes it.

### Test and Experiment
**Problem**: Workers implement a single approach and either pass or fail verification. No exploration of alternatives, no benchmarking, no "try it and see."

**Solution**: Add an experimentation mode where workers can:
- Prototype multiple approaches in isolated branches before committing to one
- Run benchmarks/profiling to compare approaches with data
- A/B test implementations: generate N candidates, run tests + benchmarks on each, pick the winner (extends P1 N-of-M beyond just fixup)
- Record experiment results for future reference ("approach X was 3x faster than Y for this pattern")

**Implementation**: New execution mode `experiment` alongside `implement`. Experiment units produce a comparison report rather than a merged commit. Winning approach gets promoted to an implement unit.

### Visual Verification
**Problem**: Mission control can run pytest/ruff/mypy but cannot look at UI changes, rendered output, dashboards, or visual artifacts to verify correctness. A human must visually inspect.

**Solution**: Integrate browser automation for visual verification:
- After UI-related units complete, launch the app and take screenshots
- Use vision-capable models to evaluate: "Does this look correct? Does the layout match the requirements?"
- Screenshot diffing: compare before/after for unintended visual regressions
- Interactive testing: navigate the UI, click buttons, verify behavior

**Implementation**: Integration with browser automation tools (Playwright or claude-in-chrome MCP). New verification step `visual_verify` that runs after standard verification for UI-touching units. Store screenshots in mission artifacts.

### Ambitious Scope
**Problem**: Current missions are constrained to small, safe, incremental units. The system avoids risk. A human would take on larger refactors, multi-system changes, or greenfield features.

**Solution**: Enable mission control to:
- Plan multi-epoch, multi-session work toward a large goal (not just single-session units)
- Take calculated risks with rollback capability -- try a large refactor, verify, roll back if it breaks
- Build entirely new modules/features from scratch, not just modify existing code
- Chain missions: output of one mission feeds the objective of the next, building toward a multi-day goal
- Self-evaluate ambition level: "Is this mission pushing boundaries or just doing busywork?"

**Implementation**: Mission chaining in config (next_mission field). Ambition scoring in the strategist. Rollback-safe mode that snapshots before risky changes and auto-reverts on failure.
