# Mission State
Objective: Implement P2 (Architect/Editor Model Split) and P3 (Structured Schedule Output for Planner). Both improve worker quality and planner reliability.

1. PER-COMPONENT MODEL CONFIG: Add per-component model fields to config.py: planner_model, worker_model, fixup_model. Each defaults to "opus" but can be overridden in mission-control.toml under a [models] section. Update recursive_planner.py, worker.py, and green_branch.py to read their model from config instead of using the global scheduler.model. Add tests for config parsing with and without the [models] section.

2. ARCHITECT/EDITOR SPLIT IN WORKERS: Add an optional two-pass mode for workers. When enabled (architect_editor_mode = true in config), the worker runs two Claude sessions per unit: (a) Architect pass -- "Analyze the codebase and describe exactly what changes are needed, which files to modify, and why. Do NOT write code." Uses the worker_model. (b) Editor pass -- "Implement these specific changes: {architect_output}". Uses the worker_model. The architect output is passed as context to the editor. If architect_editor_mode is false (default), workers behave as today -- single pass. Add the mode flag to config.py. Modify worker.py to support both modes. Add tests for both paths.

3. STRUCTURED PLANNER OUTPUT: Change recursive_planner.py to use embedded structured blocks instead of parsing the entire LLM response. The planner prompt should instruct the LLM to reason in prose first, then emit a machine-readable block: <!-- PLAN -->{"units": [{"title": "...", "scope": "...", "files_hint": "..."}]}<!-- /PLAN -->. Update the parser to extract only the <!-- PLAN --> block using regex, ignoring surrounding prose. Fall back to the current parsing if no <!-- PLAN --> block is found (backwards compatibility). The MC_RESULT pattern in session.py already uses a similar approach -- follow that pattern.

4. PLANNER OUTPUT TESTS: Add comprehensive tests for the new structured parser: valid plan block extraction, missing plan block fallback, malformed JSON inside block, multiple plan blocks (use first), plan block with surrounding prose. Test that existing planner output formats still work via fallback.

Each unit: implement the feature, add tests, ensure all existing tests pass. Read BACKLOG.md for the full P2 and P3 specs.

## Completed
- [x] 0a59694f (2026-02-17T05:05:28.903691+00:00) -- Added ModelsConfig dataclass with planner_model, worker_model, fixup_model, architect_editor_mode fi (files: src/mission_control/config.py, tests/test_config.py)
- [x] 30bb0756 (2026-02-17T05:07:44.818037+00:00) -- Wired fixup_model into green_branch.py: added _get_fixup_model() with config.models.fixup_model -> s (files: src/mission_control/green_branch.py, tests/test_green_branch.py)
- [x] 9b4289ba (2026-02-17T05:09:56.467201+00:00) -- Implemented architect/editor two-pass mode in worker.py. Added ModelsConfig to config.py with per-co (files: src/mission_control/config.py, src/mission_control/worker.py, tests/test_worker.py)
- [x] 4305d754 (2026-02-17T05:11:26.957398+00:00) -- Fixed test_fixup_falls_back_to_scheduler_model which was broken by the earlier ModelsConfig addition (files: tests/test_green_branch.py)
- [x] 9b34a3e1 (2026-02-17T05:11:35.991706+00:00) -- Wired worker_model from ModelsConfig into worker.py and continuous_controller.py with getattr fallba (files: src/mission_control/worker.py, src/mission_control/continuous_controller.py, tests/test_worker.py)
- [x] 5cfc26d (2026-02-17T05:28:43+00:00) -- Added PLAN block parsing to recursive_planner.py with <!-- PLAN -->...<!-- /PLAN --> structured output, fallback to PLAN_RESULT: marker and bare JSON extraction, planner_model config support (files: src/mission_control/recursive_planner.py, tests/test_recursive_planner.py)

## Files Modified
src/mission_control/config.py, src/mission_control/continuous_controller.py, src/mission_control/green_branch.py, src/mission_control/recursive_planner.py, src/mission_control/worker.py, tests/test_config.py, tests/test_green_branch.py, tests/test_recursive_planner.py, tests/test_worker.py

## Remaining
All 4 mission items are complete. Mission objective achieved.

## Changelog
- 2026-02-17T05:05:28.903691+00:00 | 0a59694f merged (commit: 485deb7) -- Added ModelsConfig dataclass with planner_model, worker_model, fixup_model, arch
- 2026-02-17T05:07:44.818037+00:00 | 30bb0756 merged (commit: 883345d) -- Wired fixup_model into green_branch.py: added _get_fixup_model() with config.mod
- 2026-02-17T05:11:26.957398+00:00 | 4305d754 merged (commit: 942d4ff) -- Fixed test_fixup_falls_back_to_scheduler_model which was broken by the earlier M
- 2026-02-17T05:11:35.991706+00:00 | 9b34a3e1 merged (commit: 8b7023f) -- Wired worker_model from ModelsConfig into worker.py and continuous_controller.py
- 2026-02-17T05:28:43+00:00 | 5cfc26d merged -- Added PLAN block parsing, structured planner output with fallback, planner tests (285 lines)
