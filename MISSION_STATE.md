# Mission State
Objective: Implement P2 (Architect/Editor Model Split) and P3 (Structured Schedule Output for Planner). Both improve worker quality and planner reliability.

1. PER-COMPONENT MODEL CONFIG: Add per-component model fields to config.py: planner_model, worker_model, fixup_model. Each defaults to "opus" but can be overridden in mission-control.toml under a [models] section. Update recursive_planner.py, worker.py, and green_branch.py to read their model from config instead of using the global scheduler.model. Add tests for config parsing with and without the [models] section.

2. ARCHITECT/EDITOR SPLIT IN WORKERS: Add an optional two-pass mode for workers. When enabled (architect_editor_mode = true in config), the worker runs two Claude sessions per unit: (a) Architect pass -- "Analyze the codebase and describe exactly what changes are needed, which files to modify, and why. Do NOT write code." Uses the worker_model. (b) Editor pass -- "Implement these specific changes: {architect_output}". Uses the worker_model. The architect output is passed as context to the editor. If architect_editor_mode is false (default), workers behave as today -- single pass. Add the mode flag to config.py. Modify worker.py to support both modes. Add tests for both paths.

3. STRUCTURED PLANNER OUTPUT: Change recursive_planner.py to use embedded structured blocks instead of parsing the entire LLM response. The planner prompt should instruct the LLM to reason in prose first, then emit a machine-readable block: <!-- PLAN -->{"units": [{"title": "...", "scope": "...", "files_hint": "..."}]}<!-- /PLAN -->. Update the parser to extract only the <!-- PLAN --> block using regex, ignoring surrounding prose. Fall back to the current parsing if no <!-- PLAN --> block is found (backwards compatibility). The MC_RESULT pattern in session.py already uses a similar approach -- follow that pattern.

4. PLANNER OUTPUT TESTS: Add comprehensive tests for the new structured parser: valid plan block extraction, missing plan block fallback, malformed JSON inside block, multiple plan blocks (use first), plan block with surrounding prose. Test that existing planner output formats still work via fallback.

Each unit: implement the feature, add tests, ensure all existing tests pass. Read BACKLOG.md for the full P2 and P3 specs.

Priority backlog items to address:
1. [security] Fix mission.mission_id AttributeError hidden by broad except at line 397 (backlog_item_id=a7fa92ca8e48, priority=9.0): In continuous_controller.py:397, `mission.mission_id` is used but the Mission dataclass (models.py:214) only has `id`. This raises AttributeError every time the strategic context append runs in the finally block after mission completion. The error is caught by the broad except at line 403, so strategic context is NEVER successfully written. Fix: change `mission.mission_id` to `mission.id`. Additionally, add a unit test that calls append_strategic_context after a mission completes to prevent regression.
2. [security] Fix mission.mission_id AttributeError hidden by broad except at line 397 (backlog_item_id=027ceccd65a1, priority=9.0): In continuous_controller.py:397, `mission.mission_id` is used but the Mission dataclass (models.py:214) only has `id`. This raises AttributeError every time the strategic context append runs in the finally block after mission completion. The error is caught by the broad except at line 403, so strategic context is NEVER successfully written. Fix: change `mission.mission_id` to `mission.id`. Additionally, add a unit test that calls append_strategic_context after a mission completes to prevent regression.
3. [security] Add exception logging to fire-and-forget asyncio tasks via done_callback (backlog_item_id=9f5600ae14e6, priority=7.2): In continuous_controller.py, fire-and-forget tasks created at lines 1024 (_review_merged_unit), 1298 (_retry_unit), and 1318 (_retry_unit re-dispatch) use only `task.add_done_callback(self._active_tasks.discard)` which silently swallows any unhandled exception from the coroutine. If _retry_unit raises (e.g., RuntimeError from missing semaphore at line 1316), the exception is lost -- asyncio prints 'Task exception was never retrieved' to stderr but doesn't log it. Add a shared done_callback that checks `task.exception()` and logs it, similar to the pattern already used for dispatch tasks at line 258 but missing for these secondary tasks.
4. [security] Add exception logging to fire-and-forget asyncio tasks via done_callback (backlog_item_id=e3f30bb498bf, priority=7.2): In continuous_controller.py, fire-and-forget tasks created at lines 1024 (_review_merged_unit), 1298 (_retry_unit), and 1318 (_retry_unit re-dispatch) use only `task.add_done_callback(self._active_tasks.discard)` which silently swallows any unhandled exception from the coroutine. If _retry_unit raises (e.g., RuntimeError from missing semaphore at line 1316), the exception is lost -- asyncio prints 'Task exception was never retrieved' to stderr but doesn't log it. Add a shared done_callback that checks `task.exception()` and logs it, similar to the pattern already used for dispatch tasks at line 258 but missing for these secondary tasks.
5. [quality] Extract unit event logging into a helper to eliminate 8 duplicate try/except blocks (backlog_item_id=bc2da83c3d5d, priority=6.3): In continuous_controller.py, there are 8 identical blocks that follow this pattern: `try: self.db.insert_unit_event(UnitEvent(...)) except Exception: pass`. These appear at lines 782-789, 1001-1011, 1038-1050, 1082-1091, 1112-1124, 1273-1282, 1342-1345, and the event_stream emit that follows each. Extract into a `_log_unit_event(self, mission_id, epoch_id, unit_id, event_type, *, details=None, input_tokens=0, output_tokens=0, cost_usd=0.0)` method that handles both the DB insert and event stream emit in one call with proper exception logging (`logger.debug` with exc binding). This removes ~80 lines of duplicated boilerplate.

## Completed
- [x] 7e07ed6b (2026-02-17T08:04:18.915936+00:00) -- Implemented architect/editor two-pass mode in worker.py. When config.models.architect_editor_mode is (files: src/mission_control/worker.py, tests/test_worker.py)
- [x] c737e58f (2026-02-17T08:05:28.066882+00:00) -- Added _task_done_callback for fire-and-forget exception logging (3 sites + dispatch loop) and extrac (files: src/mission_control/continuous_controller.py, tests/test_continuous_controller.py)
- [x] d80c9e94 (2026-02-17T08:11:27.327748+00:00) -- Added _on_secondary_task_done callback that logs exceptions from fire-and-forget tasks (diff review, (files: src/mission_control/continuous_controller.py, tests/test_continuous_controller.py)
- [x] 0470f118 (2026-02-17T08:09:41.839687+00:00) -- Added architect/editor two-pass mode to worker.py. When config.models.architect_editor_mode=True, wo (files: src/mission_control/worker.py, tests/test_worker.py)
- [x] b6bcee75 (2026-02-17T08:10:59.710471+00:00) -- The _log_unit_event helper already existed from prior unit c737e58f. Fixed the except handler to use (files: src/mission_control/continuous_controller.py)
- [x] d80c9e94 (2026-02-17T08:11:27.327748+00:00) -- Fire-and-forget exception logging already implemented by prior unit c737e58f (commit 35b0395). _task (files: src/mission_control/continuous_controller.py, tests/test_continuous_controller.py)

## Files Modified
src/mission_control/continuous_controller.py, src/mission_control/worker.py, tests/test_continuous_controller.py, tests/test_worker.py

## Quality Reviews
- b6bcee75 (Extract _log_unit_event helper in contin): alignment=3 approach=2 tests=1 avg=2.0
  "This diff only updates MISSION_STATE.md tracking entries — no actual implementat"

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
- 2026-02-17T08:04:18.915936+00:00 | 7e07ed6b merged (commit: 8d45d3c) -- Implemented architect/editor two-pass mode in worker.py. When config.models.arch
- 2026-02-17T08:05:28.066882+00:00 | c737e58f merged (commit: 35b0395) -- Added _task_done_callback for fire-and-forget exception logging (3 sites + dispa
- 2026-02-17T08:10:59.710471+00:00 | b6bcee75 merged (commit: 5fec880) -- The _log_unit_event helper already existed from prior unit c737e58f. Fixed the e
- 2026-02-17T08:11:27.327748+00:00 | d80c9e94 merged (commit: 35b0395) -- Fire-and-forget exception logging already implemented by prior unit c737e58f (co
