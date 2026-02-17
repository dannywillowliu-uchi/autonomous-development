# Mission State
Objective: Codebase quality improvements based on architectural analysis. Each unit must include tests and pass all existing tests.

FIX 1 - DOCUMENT OBJECTIVE SIGNALS EXCEPTION:
BACKLOG.md states "objective signals only" as a design principle, but diff_reviewer.py intentionally reintroduces LLM evaluation (alignment/approach/tests scoring). Update CLAUDE.md and BACKLOG.md to document this as a deliberate exception: "LLM diff reviews provide richer quality signals for the planner feedback loop; they are fire-and-forget and do NOT gate merges (unlike the old evaluator which blocked progress)." Mark the P1 evaluator refactor as DONE with this caveat.

FIX 2 - EXTRACT FROM CONTINUOUS CONTROLLER:
continuous_controller.py is 2,183 LOC with 17 imports. Extract two responsibilities into dedicated modules:
(a) New src/mission_control/planner_context.py: Move _build_planner_context() and _update_mission_state() (~200 lines of prompt/state formatting). The new module should export build_planner_context(db, mission_id) and update_mission_state(db, mission, config). Controller calls these functions instead.
(b) New src/mission_control/backlog_manager.py: Move _load_backlog_objective(), _update_backlog_from_completion(), _update_backlog_on_completion(), _bridge_discovery_to_backlog() (~150 lines). Export a BacklogManager class initialized with db+config. Controller delegates to it.
Both modules: add unit tests. Controller should shrink by ~350 lines. Do NOT break existing tests.

FIX 3 - FIX SYNC SUBPROCESS IN STRATEGIST:
strategist.py line 99 uses synchronous subprocess.run() for git log, which blocks the async event loop. Replace with asyncio.create_subprocess_exec() to match the pattern used everywhere else (e.g. _invoke_llm in the same file). Make _get_git_log() async. Update all callers (propose_objective, _build_strategy_prompt). Update tests.

FIX 4 - ADD CONSTANTS MODULE:
Create src/mission_control/constants.py to centralize hardcoded scoring weights and limits scattered across the codebase:
- EVALUATOR_WEIGHTS from evaluator.py (0.4, 0.2, 0.2, 0.2)
- GRADING_WEIGHTS from grading.py (0.30, 0.25, 0.25, 0.20)
- DEFAULT_LIMITS: DB query limits (10, 20, 50), retry limits, timeout defaults
Import and use these constants from the original modules. Add a test that validates weights sum to 1.0.

FIX 5 - DEFENSIVE ASSERTION FOR RECURSIVE PLANNER CWD:
CLAUDE.md documents that recursive_planner.py MUST set cwd=target.resolved_path or the planner sees scheduler files. Add:
(a) An assertion in plan_round() that validates cwd equals config.target.resolved_path before spawning
(b) A clear comment explaining the gotcha
(c) A test that fails if cwd is wrong (mock subprocess, verify cwd argument)

Each unit: implement the fix, add tests, ensure ALL existing tests pass.

## Completed
- [x] fc1c005f (2026-02-17T02:11:59.328526+00:00) -- Documented diff_reviewer.py as deliberate LLM eval exception in CLAUDE.md Architecture section and m (files: CLAUDE.md, BACKLOG.md)
- [x] 9b8f44ba (2026-02-17T02:14:40.579053+00:00) -- Replaced synchronous subprocess.run() in strategist._get_git_log() with async asyncio.create_subproc (files: src/mission_control/strategist.py, tests/test_strategist.py)

## Files Modified
BACKLOG.md, CLAUDE.md, src/mission_control/strategist.py, tests/test_strategist.py

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
- 2026-02-17T02:11:59.328526+00:00 | fc1c005f merged (commit: e3f9731) -- Documented diff_reviewer.py as deliberate LLM eval exception in CLAUDE.md Archit
- 2026-02-17T02:14:40.579053+00:00 | 9b8f44ba merged (commit: 1d732cb) -- Replaced synchronous subprocess.run() in strategist._get_git_log() with async as
