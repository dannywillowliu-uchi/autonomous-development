# Mission State
Objective: Complete P2 architect/editor two-pass mode, implement P4 EMA budget tracking, and P5 auto-pause on total failure. Fixes pre-existing test failures first.

1. FIX PRE-EXISTING TEST FAILURES: In tests/test_recursive_planner.py, `test_uses_scheduler_model_when_no_models_config` expects fallback to `scheduler.model` ("sonnet") but `ModelsConfig.planner_model` defaults to "opus". Fix so that when no `[models]` section exists in config, per-component models fall back to `scheduler.model` instead of hardcoded "opus". Also fix `test_history_entry_fields` in test_live_dashboard.py if straightforward.

2. ARCHITECT/EDITOR TWO-PASS MODE: Implement the actual two-pass execution in worker.py. When `config.models.architect_editor_mode` is True, `_execute_unit` should run two Claude sessions: (a) Architect pass -- prompt instructs "Analyze the codebase and describe exactly what changes are needed, which files to modify, and why. Do NOT write code." Capture the architect's output. (b) Editor pass -- prompt includes the architect's analysis as context and instructs "Implement these specific changes: {architect_output}". When `architect_editor_mode` is False (default), behavior is unchanged -- single pass. Add an `ARCHITECT_PROMPT_TEMPLATE` alongside existing templates. Add tests for both paths with mocked subprocess calls.

3. EMA BUDGET TRACKING: Add per-cycle cost tracking with Exponential Moving Average. New module or section in an existing module. Log cost per completed unit. Compute EMA with alpha=0.3. Outlier dampening: spikes >3x EMA clamped to 2x (after 3+ data points). Conservatism factor: `k = 1.0 + 0.5/sqrt(n)` where n is number of data points. Wire into `_should_stop()` in continuous_controller.py: stop the mission if `mission.total_cost_usd >= config.budget.max_per_run_usd`. Add adaptive cooldown between rounds based on remaining budget (slow down as budget depletes). Add tests for EMA computation, outlier dampening, and budget enforcement.

4. AUTO-PAUSE ON TOTAL FAILURE: In continuous_controller.py, add auto-pause when all units in a dispatch round fail. Track consecutive all-fail rounds. After 1 all-fail round, pause for a configurable interval (default 60s) then retry. After 3 consecutive all-fail rounds, stop the mission with reason "repeated_total_failure". Add a `max_consecutive_failures` config field (default 3) and `failure_backoff_seconds` (default 60). Add tests for the pause/retry/stop logic.

Each unit: implement the feature, add tests, ensure all existing tests pass. Read BACKLOG.md for the full P2, P4, and P5 specs.

## Completed

## Files Modified

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
