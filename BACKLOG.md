# Autonomous Dev Scheduler -- Improvement Backlog

Derived from cross-system research (Feb 2026). Updated Feb 2026.

> **Note**: This file is a static roadmap for human reference. The actual task
> queue lives in SQLite (`backlog_items` table) and is managed via `mc priority`.
> Discovery and post-mission feedback update the SQLite queue automatically.
> This file should be updated manually when features ship or new ideas are added.

---

## Completed

### P0: Priority Queue and Cross-Mission Scheduling System -- DONE

Persistent SQLite backlog with automatic priority recalculation, failure penalties, staleness detection, and human override CLI.

**What shipped**:
- `backlog_items` table in `db.py` with full lifecycle (pending/in_progress/completed/deferred)
- `BacklogItem` model in `models.py` with impact, effort, attempt_count, last_failure_reason
- `priority.py`: recalculate_priorities() with failure penalty (-20%/attempt, capped -60%), staleness penalty (-10% after 72h), infrastructure failure forgiveness, pinned score override
- `backlog_manager.py`: loads top N items at mission start, updates on unit/mission completion
- `mc priority` CLI: `list`, `set`, `defer`, `import`, `recalc` subcommands
- Discovery inserts new items into queue; post-mission discovery adds more
- Mission chaining reads remaining backlog for next objective

### P1 (was P0): Replace LLM Evaluator with Objective Signals -- DONE

The old `evaluator.py` LLM call replaced by deterministic grading in `grading.py`. `diff_reviewer.py` intentionally reintroduces LLM evaluation as a fire-and-forget post-merge quality signal (non-blocking, failures tolerated).

### P1: N-of-M Candidate Selection for Fixup -- DONE

3 parallel fixup candidates with different prompt strategies, tournament selection by (tests_passed DESC, lint_errors ASC, diff_lines ASC).

**What shipped**: `green_branch.py` `run_fixup()` spawns N candidates concurrently. `FIXUP_PROMPTS` list with 3 approaches. `FixupCandidate` dataclass. `fixup_candidates: int = 3` in config.

### P2: Architect/Editor Model Split for Workers -- DONE

Per-component model configuration with optional architect/editor two-pass mode.

**What shipped**: `ModelsConfig` with `planner_model`, `worker_model`, `fixup_model`, `architect_editor_mode`. Worker supports two-pass execution: architect pass (reasoning) then editor pass (implementation).

### P3: Structured Schedule Output for Planner -- DONE

Embedded `<!-- PLAN -->...<!-- /PLAN -->` blocks in planner output. Parser extracts structured JSON, ignores surrounding reasoning. Fallback to legacy `PLAN_RESULT` format.

### P4: EMA Budget Tracking -- DONE

`ema.py` with `ExponentialMovingAverage` class: alpha=0.3, outlier dampening (spikes >3x EMA clamped to 2x), conservatism factor `k = 1.0 + 0.5/sqrt(n)`, projected cost method.

### P6: Typed Context Store -- DONE

`ContextItem` dataclass with type, scope, content, source_unit_id, round_id, confidence. Stored in SQLite. Workers produce context items; coordinator selectively injects relevant items into subsequent worker prompts based on scope overlap.

### P7: Dynamic Agent Composition -- DONE

Specialist templates: test-writer, refactorer, debugger, simplifier. Planner assigns specialist per work unit. `load_specialist_template()` loads from project's `specialist_templates/` directory.

### P8: Runtime Tool Synthesis -- DONE

`tool_synthesis.py` module. Workers can create project-specific helper tools during execution. Tools persist across work units within a round.

### Strategist Agent -- DONE

`strategist.py`: reads BACKLOG.md, git history, past missions, pending backlog queue, strategic context. Proposes focused objectives autonomously. Human approves via `--strategist` flag.

### Mission Chaining -- DONE

`--chain` flag with `--max-chain-depth`. Post-mission: remaining backlog items compose next objective. Cleanup missions trigger every N missions (configurable).

### Experiment Mode -- DONE

`--experiment` flag. Experiment units produce comparison reports, not merged commits. `ExperimentResult` model tracks outcomes. Winning approach can be promoted to implement unit.

### MISSION_STATE.md Traceability -- DONE

`planner_context.py` `update_mission_state()` auto-generates and commits MISSION_STATE.md at mission start and throughout execution. Committed to mc/green branch after each unit merge.

---

## In Progress / Partial

### P5: Auto-Pause and Recovery -- PARTIAL

Config fields exist for backoff/retry. Continuous controller has pause/retry references. Needs verification that the full backoff-with-condition-checking loop works as designed (pause on total round failure, verify condition resolved before retry, max retries before escalation).

**Remaining work**: Audit the actual retry paths in `continuous_controller.py` and confirm they match the design. Add tests if missing.

---

## Remaining

### Visual Verification

**Problem**: Mission control can run pytest/ruff/mypy but cannot look at UI changes, rendered output, dashboards, or visual artifacts to verify correctness.

**Solution**: Integrate browser automation for visual verification:
- After UI-related units complete, launch the app and take screenshots
- Use vision-capable models to evaluate layout/correctness
- Screenshot diffing for unintended visual regressions
- Interactive testing: navigate UI, click buttons, verify behavior

**Implementation**: Integration with Playwright or claude-in-chrome MCP. New verification step `visual_verify` for UI-touching units. Store screenshots in mission artifacts.

**Priority**: Low -- only relevant when mission control targets projects with UI components.

---

### P9: Self-Play Training Loop (Long-term)

**Problem**: System doesn't improve its own prompts or strategies based on accumulated data.

**Solution**: Use the system to generate bug-injection / bug-fix pairs:
1. Agent A introduces a realistic bug into the codebase
2. Agent B attempts to find and fix it
3. Success/failure signal trains better prompts

Start with prompt-level self-play (no weight updates). The Self-Improving Coding Agent showed 17-53% gains from prompt/code self-editing alone.

**Priority**: Research/experimental. Requires significant design work.

**Source inspiration**: Meta FAIR Self-Play SWE-RL, Self-Improving Coding Agent (ICLR 2025 Workshop).

---

### Dashboard Mission Statement View

**Problem**: Live dashboard doesn't show the current mission objective or planned work items prominently.

**Solution**: Add a section to `live_ui.html` displaying the current mission objective and planned work items at the top of the dashboard alongside the status bar.

**Priority**: Nice-to-have UX improvement.

---

## Ideas (Not Yet Designed)

- **Cross-project missions**: Mission control targeting multiple repos in a single mission
- **Parallel workers on separate units**: Currently units are sequential within an epoch; could dispatch multiple workers concurrently on independent units
- **Prompt evolution**: Track which prompt variations produce better outcomes across missions and auto-select the best performing ones
- **Cost attribution**: Per-unit cost tracking to identify which types of work are expensive vs cheap
