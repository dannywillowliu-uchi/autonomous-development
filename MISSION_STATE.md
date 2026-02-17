# Mission State
Objective: Work through the priority backlog. Top items:

1. P4 - EMA BUDGET TRACKING: Implement exponential moving average cost tracking per work unit. Add EMA module with alpha=0.3, outlier dampening (>3x EMA clamped to 2x), conservatism factor (k=1.0+0.5/sqrt(n)). Wire into _should_stop() in continuous_controller.py. Add adaptive cooldown that increases when costs exceed budget. Add comprehensive tests.

2. P6 - TYPED CONTEXT STORE: Replace flat text memory.py with structured ContextItem dataclass backed by SQLite. Add scope-based filtering (mission, round, unit). Support selective injection into worker prompts based on relevance. Add tests.

3. QUALITY FIXES from backlog: Replace silent 'except Exception: pass' patterns with logged catches. Add cost accumulation resilience. Sanitize brace characters in worker prompt .format() inputs. Fix semaphore private attribute manipulation.

Each unit: implement the feature, add tests, ensure all existing tests pass. Read BACKLOG.md for full specs.

Priority backlog items to address:
1. [feature] Implement EMA budget tracking with adaptive cooldown (backlog_item_id=p4-ema-budget, priority=8.0): Add per-cycle cost tracking with Exponential Moving Average. Log cost per completed unit. Compute EMA with alpha=0.3. Outlier dampening: spikes >3x EMA clamped to 2x. Conservatism factor: k = 1.0 + 0.5/sqrt(n). Wire into _should_stop(). Add adaptive cooldown between rounds based on remaining budget. Add tests.
2. [feature] Add dry-run cost estimation to mission planning output (backlog_item_id=45fe865d0a4b, priority=5.4): Enhance the existing dry_run mode (continuous_controller.py:116-147) to include a cost estimate. After the planner returns units, estimate total cost by: counting units, multiplying by average tokens per unit (configurable in ContinuousConfig, default 50K input + 10K output based on typical Claude Code sessions), and applying the configured pricing rates from config.pricing. Print a summary table showing: unit count, estimated parallelism levels (already computed), estimated total tokens, estimated cost USD, and estimated wall time (units / num_workers * avg_duration). This gives operators a pre-flight check before committing to a potentially expensive mission.
3. [feature] Add per-unit correlation ID threading through dispatch-execute-merge pipeline (backlog_item_id=a03cc926bb6b, priority=4.9): Add a `correlation_id` field to WorkUnit (models.py) that combines mission_id[:8], epoch_id[:8], and unit_id[:8] into a compact trace ID (e.g., 'a1b2c3d4.e5f6g7h8.i9j0k1l2'). Thread this into: (1) the worker prompt via render_mission_worker_prompt so it appears in subprocess logs, (2) all logger calls in _execute_single_unit and _process_completions as an extra field, (3) EventStream emissions as a top-level key, and (4) the UnitEvent details JSON. This enables filtering all logs for a single unit's journey with one grep.
4. [quality] Replace dynamic attribute injection in recursive_planner with typed container (backlog_item_id=8f3f074d273f, priority=4.8): In recursive_planner.py, three dynamic attributes are injected onto PlanNode: `_subdivided_children` (line 153), `_child_leaves` (line 194), and `_forced_unit` (line 349), all with `# type: ignore[attr-defined]`. Replace with a typed container: add an `ExpandedNode` dataclass that wraps PlanNode with optional fields for `subdivided_children: list[PlanNode]`, `child_leaves: list[tuple[PlanNode, WorkUnit]]`, and `forced_unit: WorkUnit | None`. The expand_node method returns ExpandedNode, and _iter_leaves operates on ExpandedNode instead of raw PlanNode. This eliminates all three type: ignore comments and makes the tree structure inspectable.
5. [quality] Add integration test for dispatch-execute-complete pipeline with real SQLite and Queue (backlog_item_id=79d305cd24f6, priority=4.8): Create a thin integration test in tests/test_controller_integration.py that exercises the real dispatch->execute->merge pipeline without mocking internal components. Use: real SQLite via tmp_path, real asyncio.Queue, a FakeBackend that returns canned MC_RESULT output (mimicking a successful worker), and a FakeGreenBranch that records merge calls. Test the full flow: controller dispatches a unit, FakeBackend completes it with valid MC_RESULT, completion processor picks it up, FakeGreenBranch receives the merge call, and DB state transitions (pending->running->merged) are verified. This catches interaction bugs like the cwd assertion bug (commit 47479e9) that unit tests with mocks miss.

## Completed
- [x] 537b04cb (2026-02-17T16:32:42.258200+00:00) -- Replaced dynamic attribute injection in recursive_planner.py with typed ExpandedNode dataclass. Adde (files: src/mission_control/recursive_planner.py, tests/test_recursive_planner.py)
- [x] aa0ced62 (2026-02-17T16:35:03.142860+00:00) -- Implemented EMA budget tracking module with EMATracker class (alpha=0.3, outlier dampening, conserva (files: src/mission_control/ema.py, src/mission_control/config.py, src/mission_control/continuous_controller.py, tests/test_ema.py)
- [x] 12425b8a (2026-02-17T16:37:34.571004+00:00) -- Added integration test for dispatch-execute-complete pipeline with FakeBackend and FakeGreenBranch e (files: tests/test_controller_integration.py)
- [x] b1bee853 (2026-02-17T16:39:19.275036+00:00) -- Added dry-run cost estimation: avg_input/output_tokens_per_unit config fields, cost/wall-time summar (files: src/mission_control/config.py, src/mission_control/continuous_controller.py, tests/test_continuous_controller.py)

## Files Modified
src/mission_control/config.py, src/mission_control/continuous_controller.py, src/mission_control/ema.py, src/mission_control/recursive_planner.py, tests/test_continuous_controller.py, tests/test_controller_integration.py, tests/test_ema.py, tests/test_recursive_planner.py

## Quality Reviews
- aa0ced62 (Implement EMA budget tracking module wit): alignment=2 approach=3 tests=1 avg=2.0
  "The diff only updates MISSION_STATE.md with backlog bookkeeping; the entire EMA "

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
- 2026-02-17T16:32:42.258200+00:00 | 537b04cb merged (commit: 36ae414) -- Replaced dynamic attribute injection in recursive_planner.py with typed Expanded
- 2026-02-17T16:35:03.142860+00:00 | aa0ced62 merged (commit: 9bc59fd) -- Implemented EMA budget tracking module with EMATracker class (alpha=0.3, outlier
