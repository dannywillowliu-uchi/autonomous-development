# Mission State
Objective: Continue with 5 remaining backlog items. Top priorities: [quality] Add end-to-end integration test for the dispatch-merge-complete cycle (priority=7.2); [feature] Add backlog staleness detection with eviction and reprioritization (priority=7.2); [quality] Fix semaphore recreation bug that silently breaks worker count adjustment (priority=7.2)

Priority backlog items to address:
1. [quality] Add end-to-end integration test for the dispatch-merge-complete cycle (backlog_item_id=259988349210, priority=7.2): The critical path (controller dispatches unit -> worker executes -> MC_RESULT parsed -> green branch merge -> handoff ingested -> planner re-plans) is only tested at the unit level in separate test files. Create an integration test in tests/test_integration_cycle.py that: (1) sets up a real git repo in tmp_path, (2) initializes a controller with a mock worker backend that produces realistic MC_RESULT output, (3) runs 2-3 work units through the full cycle including merge to mc/green, (4) verifies the final git state, db state, and mission state updates. Use the existing real-git test pattern from test_green_branch.py (TestGreenBranchRealGit) as a foundation.
2. [feature] Add backlog staleness detection with eviction and reprioritization (backlog_item_id=fdf23154b8b9, priority=7.2): ContinuousPlanner._backlog is a plain `list[WorkUnit]` FIFO (lines 28, 100-107 of continuous_planner.py) with no age tracking. Units queued when the backlog exceeds `max_units` can sit indefinitely while the codebase diverges underneath them.

Add:
1. **Timestamp tracking**: Record `queued_at = time.monotonic()` when units enter the backlog (line 200 of continuous_planner.py where `self._backlog.extend()` is called).
2. **Staleness threshold**: Config field `continuous.backlog_max_age_seconds` (default 1800). On each `next_units()` call, evict units older than the threshold and log a warning.
3. **Codebase divergence check**: Before serving a backlog unit, compare its `files_hint` against `self._merged_files` accumulated since the unit was planned. If >50% of target files were modified by other merges, mark the unit stale and skip it.
4. **Priority reordering**: When evicting stale units, feed their descriptions into the next replan as `stale_context` so the planner can regenerate updated versions.

The existing `_check_already_merged()` only checks full file coverage, not partial divergence.
3. [quality] Fix semaphore recreation bug that silently breaks worker count adjustment (backlog_item_id=5a720e2ab223, priority=7.2): In `_dispatch_loop()` (line 1055), the semaphore is captured as a local variable: `semaphore = asyncio.Semaphore(num_workers)`. This local is passed to `_dispatch_deferred()` at line 1068 and used throughout the loop. When `_handle_adjust_signal()` (line 3068-3070) creates a new `asyncio.Semaphore(available)` and assigns it to `self._semaphore`, the dispatch loop's local `semaphore` variable still references the OLD object. Worker count adjustment via the dashboard 'adjust' signal silently does nothing.

Fix: Remove the local `semaphore` variable entirely. Always reference `self._semaphore` in the dispatch loop and in `_dispatch_deferred()`. In `_handle_adjust_signal()`, instead of replacing the Semaphore object, manipulate the existing one: call `semaphore.release()` to increase capacity or track a debt counter to temporarily block acquires when reducing. Alternatively, use a simple `asyncio.Event` + counter pattern that supports dynamic resizing.

Add a test that: (1) starts dispatch with num_workers=2, (2) sends an adjust signal to num_workers=4, (3) verifies that 4 concurrent dispatches are now possible.
4. [feature] Add backlog staleness detection with eviction and reprioritization (backlog_item_id=6162bf6b2feb, priority=7.2): ContinuousPlanner._backlog is a plain `list[WorkUnit]` FIFO (lines 28, 100-107 of continuous_planner.py) with no age tracking. Units queued when the backlog exceeds `max_units` can sit indefinitely while the codebase diverges underneath them.

Add:
1. **Timestamp tracking**: Record `queued_at = time.monotonic()` when units enter the backlog (line 200 of continuous_planner.py where `self._backlog.extend()` is called).
2. **Staleness threshold**: Config field `continuous.backlog_max_age_seconds` (default 1800). On each `next_units()` call, evict units older than the threshold and log a warning.
3. **Codebase divergence check**: Before serving a backlog unit, compare its `files_hint` against `self._merged_files` accumulated since the unit was planned. If >50% of target files were modified by other merges, mark the unit stale and skip it.
4. **Priority reordering**: When evicting stale units, feed their descriptions into the next replan as `stale_context` so the planner can regenerate updated versions.

The existing `_check_already_merged()` only checks full file coverage, not partial divergence.
5. [quality] Fix semaphore recreation bug that silently breaks worker count adjustment (backlog_item_id=15c57d870182, priority=7.2): In `_dispatch_loop()` (line 1055), the semaphore is captured as a local variable: `semaphore = asyncio.Semaphore(num_workers)`. This local is passed to `_dispatch_deferred()` at line 1068 and used throughout the loop. When `_handle_adjust_signal()` (line 3068-3070) creates a new `asyncio.Semaphore(available)` and assigns it to `self._semaphore`, the dispatch loop's local `semaphore` variable still references the OLD object. Worker count adjustment via the dashboard 'adjust' signal silently does nothing.

Fix: Remove the local `semaphore` variable entirely. Always reference `self._semaphore` in the dispatch loop and in `_dispatch_deferred()`. In `_handle_adjust_signal()`, instead of replacing the Semaphore object, manipulate the existing one: call `semaphore.release()` to increase capacity or track a debt counter to temporarily block acquires when reducing. Alternatively, use a simple `asyncio.Event` + counter pattern that supports dynamic resizing.

Add a test that: (1) starts dispatch with num_workers=2, (2) sends an adjust signal to num_workers=4, (3) verifies that 4 concurrent dispatches are now possible.

## Completed
- [x] eaa30c18 (2026-02-23T06:44:37.961939+00:00) -- Added backlog staleness detection with time-based and divergence-based eviction to ContinuousPlanner (files: src/mission_control/config.py, src/mission_control/continuous_planner.py, tests/test_continuous_planner.py)
- [x] 5b1ead03 (2026-02-23T06:50:43.164605+00:00) -- Added end-to-end integration test for the dispatch-merge-complete cycle in tests/test_integration_cy (files: tests/test_integration_cycle.py)

## In-Flight (DO NOT duplicate)
- [ ] 4ddb94e3 -- Fix semaphore recreation bug that silently breaks worker count adjustment (files: src/mission_control/continuous_controller.py,tests/test_continuous_controller.py)

## Files Modified
src/mission_control/config.py, src/mission_control/continuous_planner.py, tests/test_continuous_planner.py, tests/test_integration_cycle.py

## System Health
Degradation level: FULL_CAPACITY

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
- 2026-02-23T06:44:37.961939+00:00 | eaa30c18 merged (commit: 2468ca9) -- Added backlog staleness detection with time-based and divergence-based eviction 
- 2026-02-23T06:50:43.164605+00:00 | 5b1ead03 merged (commit: 22dfc82) -- Added end-to-end integration test for the dispatch-merge-complete cycle in tests
