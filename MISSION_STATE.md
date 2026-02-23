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

## In-Flight (DO NOT duplicate)
- [ ] 1406b949 -- Consolidate test_ambition_scoring.py into strategist test files (files: tests/test_ambition_scoring.py,tests/test_strategist.py,tests/test_strategist_integration.py)
- [ ] cd8a6db3 -- Add set_causal_context() public API to RecursivePlanner and ContinuousPlanner (files: src/mission_control/recursive_planner.py,src/mission_control/continuous_planner.py,src/mission_control/continuous_controller.py)

## Files Modified
src/mission_control/circuit_breaker.py, src/mission_control/db.py, src/mission_control/models.py, tests/test_circuit_breaker.py, tests/test_db.py, tests/test_fixup.py, tests/test_zfc.py

## System Health
Degradation level: FULL_CAPACITY

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
- 2026-02-23T05:46:25.957198+00:00 | 3f8d13b1 merged (commit: 622f3ec) -- Merged TestZFCFixupPrompts from test_zfc.py into test_fixup.py, unified factory 
