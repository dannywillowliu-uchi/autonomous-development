# Mission State
Objective: [CLEANUP] Consolidate test suite: 68 files, 0 tests. Merge test files for the same module, consolidate small test classes, remove dead code. All tests must pass.

Priority backlog items to address:
1. [quality] Fix high-severity code bugs: duplicate field, dir() check, dead migrations (backlog_item_id=627393fa7bf8, priority=7.7): Three confirmed bugs that affect correctness: (1) models.py:202-204 declares `specialist: str = ''` twice on WorkUnit -- Python silently keeps the last, but the two have different docstrings indicating divergent intent. Remove the duplicate at line 202. (2) continuous_controller.py:594-595 uses `if 'failed_summaries' in dir()` which is semantically wrong (dir() behavior is implementation-dependent for locals). The variable `failed_summaries` IS always defined at line 569 in the same try block, so the guard is unnecessary -- but the real fix is to initialize `failed_summaries: list[str] = []` before the outer try block at line 567, making the dir() check and defensive list comprehension both unnecessary. (3) db.py:775-781 has two no-op migration methods (_migrate_backlog_table, _migrate_strategic_context) that log but do nothing -- these tables are already created by SCHEMA_SQL. Delete them and remove their calls from _create_tables(). (4) db.py has duplicate unit_type migration at lines 748 and 763 with inconsistent NOT NULL -- remove the one at line 763 since line 748 already handles it.
2. [feature] Wire intra-mission feedback loops into planner decisions (backlog_item_id=8392d5be5ac2, priority=7.5): Three feedback systems are built but disconnected from the planner:

1. **diff_reviewer.py** scores (alignment/approach/test_quality) are stored in `unit_reviews` table but the planner never reads them. Feed review scores into `build_planner_context()` so the planner can avoid repeating low-scoring patterns.

2. **feedback.py** is dead in continuous mode -- `get_worker_context()` calls `db.search_experiences()` but `insert_experience()` is only called from the old round-based scheduler. Add an `insert_experience()` call in the completion processor (after line 1934 of continuous_controller.py) to populate experiences from continuous mode outcomes, keyed by unit title/scope/files.

3. **causal.py** risk factors are injected into the planner via private attribute side-channel: `self._planner._inner._causal_risks = causal_risks` (line 1124). Add a public `set_causal_context(risks)` method to RecursivePlanner and ContinuousPlanner that stores risks in a typed field, then incorporates them into the planner prompt template.

The result is a closed loop: worker outcomes -> diff review scores + causal attribution + experiences -> planner context -> better unit generation -> fewer failures.
3. [feature] Wire intra-mission feedback loops into planner decisions (backlog_item_id=0ad84658ba6e, priority=7.5): Three feedback systems are built but disconnected from the planner:

1. **diff_reviewer.py** scores (alignment/approach/test_quality) are stored in `unit_reviews` table but the planner never reads them. Feed review scores into `build_planner_context()` so the planner can avoid repeating low-scoring patterns.

2. **feedback.py** is dead in continuous mode -- `get_worker_context()` calls `db.search_experiences()` but `insert_experience()` is only called from the old round-based scheduler. Add an `insert_experience()` call in the completion processor (after line 1934 of continuous_controller.py) to populate experiences from continuous mode outcomes, keyed by unit title/scope/files.

3. **causal.py** risk factors are injected into the planner via private attribute side-channel: `self._planner._inner._causal_risks = causal_risks` (line 1124). Add a public `set_causal_context(risks)` method to RecursivePlanner and ContinuousPlanner that stores risks in a typed field, then incorporates them into the planner prompt template.

The result is a closed loop: worker outcomes -> diff review scores + causal attribution + experiences -> planner context -> better unit generation -> fewer failures.
4. [security] Wire degradation and circuit breaker state transitions to Telegram alerts (backlog_item_id=dd0e4bf15a8e, priority=7.5): Both state machines are 'Telegram-dark':

1. **DegradationManager** (degradation.py:88-104): `_transition()` calls `self._on_transition(transition)` if set, but the callback is never wired to a notifier. The controller constructs `DegradationManager` without passing a notification callback.

2. **CircuitBreaker** (circuit_breaker.py:69-95): Has NO callback mechanism at all. State transitions (CLOSED->OPEN, OPEN->HALF_OPEN, HALF_OPEN->CLOSED) only emit `logger.warning`.

This means: system degrades to SAFE_STOP (all dispatch halted) or circuit breaker opens (workspace blocked) with ZERO operator notification. The operator discovers this only by checking logs or noticing the dashboard is stale.

Fix:
1. In continuous_controller.py `start()`, wire `DegradationManager(on_transition=self._on_degradation_transition)` and implement `_on_degradation_transition()` to call `self._notifier.send_alert()` for transitions to REDUCED_CAPACITY or higher.
2. Add `on_state_change: Callable[[str, str, str], None] | None` callback to CircuitBreaker. Call it from `record_failure()` when transitioning to OPEN, and from `_try_probe()` on HALF_OPEN transitions. Wire to a method that calls `self._notifier.send_alert()` with workspace name and state.
3. Filter noise: only notify on escalations (not on recovery to FULL_CAPACITY/CLOSED) unless configured otherwise.
5. [security] Wire degradation and circuit breaker state transitions to Telegram alerts (backlog_item_id=cf318fdd292c, priority=7.5): Both state machines are 'Telegram-dark':

1. **DegradationManager** (degradation.py:88-104): `_transition()` calls `self._on_transition(transition)` if set, but the callback is never wired to a notifier. The controller constructs `DegradationManager` without passing a notification callback.

2. **CircuitBreaker** (circuit_breaker.py:69-95): Has NO callback mechanism at all. State transitions (CLOSED->OPEN, OPEN->HALF_OPEN, HALF_OPEN->CLOSED) only emit `logger.warning`.

This means: system degrades to SAFE_STOP (all dispatch halted) or circuit breaker opens (workspace blocked) with ZERO operator notification. The operator discovers this only by checking logs or noticing the dashboard is stale.

Fix:
1. In continuous_controller.py `start()`, wire `DegradationManager(on_transition=self._on_degradation_transition)` and implement `_on_degradation_transition()` to call `self._notifier.send_alert()` for transitions to REDUCED_CAPACITY or higher.
2. Add `on_state_change: Callable[[str, str, str], None] | None` callback to CircuitBreaker. Call it from `record_failure()` when transitioning to OPEN, and from `_try_probe()` on HALF_OPEN transitions. Wire to a method that calls `self._notifier.send_alert()` with workspace name and state.
3. Filter noise: only notify on escalations (not on recovery to FULL_CAPACITY/CLOSED) unless configured otherwise.

## Completed
- [x] 3f8d13b1 (2026-02-23T05:46:25.957198+00:00) -- Merged TestZFCFixupPrompts from test_zfc.py into test_fixup.py, unified factory to use _manager() wi (files: tests/test_fixup.py, tests/test_zfc.py)
- [x] 02cdec96 (2026-02-23T05:46:30.443223+00:00) -- Added on_state_change callback parameter to CircuitBreakerManager.__init__. Callback fires with (wor (files: src/mission_control/circuit_breaker.py, tests/test_circuit_breaker.py)
- [x] 1bb6ad0f (2026-02-23T05:46:51.064821+00:00) -- Removed _migrate_unit_type_column (duplicate of migration in _migrate_token_columns), _migrate_backl (files: src/mission_control/db.py, tests/test_db.py)
- [x] ab961ee5 (2026-02-23T05:46:51.162355+00:00) -- Removed duplicate specialist field from WorkUnit dataclass in models.py. The first declaration (with (files: src/mission_control/models.py)
- [x] cd8a6db3 (2026-02-23T05:48:11.309034+00:00) -- Added set_causal_context() public API to RecursivePlanner and ContinuousPlanner, replacing private a (files: src/mission_control/recursive_planner.py, src/mission_control/continuous_planner.py, src/mission_control/continuous_controller.py, tests/test_recursive_planner.py, tests/test_continuous_planner.py)
- [x] 1406b949 (2026-02-23T05:49:08.060148+00:00) -- Consolidated test_ambition_scoring.py into strategist test files. Moved TestEvaluateAmbition, TestZF (files: tests/test_ambition_scoring.py, tests/test_strategist.py, tests/test_strategist_integration.py)
- [x] 4de7bee9 (2026-02-23T05:50:59.554887+00:00) -- Merged 13 tests (TestTopologicalLayers, TestPlanNodeMultiParent, TestOverlapWithLayers) from test_da (files: tests/test_overlap.py, tests/test_dag_planner.py)
- [x] 0fa9364a (2026-02-23T05:52:45.416911+00:00) -- Merged TestStrategicContextTable and TestSignalCRUD into test_db.py, absorbed TestMissionNewFields i (files: tests/test_db.py, tests/test_strategic_context_db.py, tests/test_signals.py)
- [x] 036ba2a1 (2026-02-23T05:58:54.375332+00:00) -- Merged TestWeightTuples and TestDefaultLimits from test_constants.py into test_models.py, and TestEx (files: tests/test_constants.py, tests/test_json_utils.py, tests/test_models.py, tests/test_session.py)
- [x] b4843e16 (2026-02-23T06:00:10.003769+00:00) -- Merged TestEventStream into test_event_sourcing.py, moved TestReviewSession/TestComputeDecomposition (files: tests/test_event_sourcing.py, tests/test_evaluator.py, tests/test_event_stream.py, tests/test_reviewer.py, tests/test_grading.py)
- [x] d9fdb434 (2026-02-23T06:01:55.974307+00:00) -- Wired diff_reviewer low scores into build_planner_context. Reviews with alignment, approach, or test (files: src/mission_control/planner_context.py, tests/test_planner_context.py)
- [x] f1dfeff6 (2026-02-23T06:08:52.877080+00:00) -- Fixed dir() check bug by initializing failed_summaries before try block, wired insert_experience() i (files: src/mission_control/continuous_controller.py, tests/test_continuous_controller.py)

## In-Flight (DO NOT duplicate)
- [ ] 5f121c38 -- Merge test_continuous_foundation.py into test_db.py and test_config.py (files: tests/test_continuous_foundation.py,tests/test_db.py,tests/test_config.py,tests/conftest.py)
- [ ] f052912b -- Add public APIs to circuit_breaker and recursive_planner (files: src/mission_control/circuit_breaker.py, src/mission_control/recursive_planner.py, src/mission_control/continuous_planner.py)

## Files Modified
src/mission_control/circuit_breaker.py, src/mission_control/continuous_controller.py, src/mission_control/continuous_planner.py, src/mission_control/db.py, src/mission_control/models.py, src/mission_control/planner_context.py, src/mission_control/recursive_planner.py, tests/test_ambition_scoring.py, tests/test_circuit_breaker.py, tests/test_constants.py, tests/test_continuous_controller.py, tests/test_continuous_planner.py, tests/test_dag_planner.py, tests/test_db.py, tests/test_evaluator.py, tests/test_event_sourcing.py, tests/test_event_stream.py, tests/test_fixup.py, tests/test_grading.py, tests/test_json_utils.py, tests/test_models.py, tests/test_overlap.py, tests/test_planner_context.py, tests/test_recursive_planner.py, tests/test_reviewer.py, tests/test_session.py, tests/test_signals.py, tests/test_strategic_context_db.py, tests/test_strategist.py, tests/test_strategist_integration.py, tests/test_zfc.py

## System Health
Degradation level: FULL_CAPACITY

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
- 2026-02-23T05:46:25.957198+00:00 | 3f8d13b1 merged (commit: 622f3ec) -- Merged TestZFCFixupPrompts from test_zfc.py into test_fixup.py, unified factory 
- 2026-02-23T05:46:30.443223+00:00 | 02cdec96 merged (commit: 31681db) -- Added on_state_change callback parameter to CircuitBreakerManager.__init__. Call
- 2026-02-23T05:46:51.064821+00:00 | 1bb6ad0f merged (commit: ee1b1c7) -- Removed _migrate_unit_type_column (duplicate of migration in _migrate_token_colu
- 2026-02-23T05:46:51.162355+00:00 | ab961ee5 merged (commit: 3b6c598) -- Removed duplicate specialist field from WorkUnit dataclass in models.py. The fir
- 2026-02-23T05:48:11.309034+00:00 | cd8a6db3 merged (commit: cca704e) -- Added set_causal_context() public API to RecursivePlanner and ContinuousPlanner,
- 2026-02-23T05:49:08.060148+00:00 | 1406b949 merged (commit: d065112) -- Consolidated test_ambition_scoring.py into strategist test files. Moved TestEval
- 2026-02-23T05:50:59.554887+00:00 | 4de7bee9 merged (commit: 48d2e2b) -- Merged 13 tests (TestTopologicalLayers, TestPlanNodeMultiParent, TestOverlapWith
- 2026-02-23T05:52:45.416911+00:00 | 0fa9364a merged (commit: 97d0897) -- Merged TestStrategicContextTable and TestSignalCRUD into test_db.py, absorbed Te
- 2026-02-23T05:58:54.375332+00:00 | 036ba2a1 merged (commit: e38f26d) -- Merged TestWeightTuples and TestDefaultLimits from test_constants.py into test_m
- 2026-02-23T06:00:10.003769+00:00 | b4843e16 merged (commit: e4ded89) -- Merged TestEventStream into test_event_sourcing.py, moved TestReviewSession/Test
- 2026-02-23T06:01:55.974307+00:00 | d9fdb434 merged (commit: deeaf35) -- Wired diff_reviewer low scores into build_planner_context. Reviews with alignmen
- 2026-02-23T06:08:52.877080+00:00 | f1dfeff6 merged (commit: b443f9e) -- Fixed dir() check bug by initializing failed_summaries before try block, wired i
