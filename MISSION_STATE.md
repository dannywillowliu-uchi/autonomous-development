# Mission State
Objective: Work through the priority backlog -- P4 (EMA budget tracking), P6 (typed context store), and quality fixes.

## Priority Items
1. P4 - EMA BUDGET TRACKING (priority=8.0): Implement exponential moving average cost tracking per work unit. Add EMA module with alpha=0.3, outlier dampening (>3x EMA clamped to 2x), conservatism factor (k=1.0+0.5/sqrt(n)). Wire into _should_stop() in continuous_controller.py. Add adaptive cooldown. Add tests.

2. P6 - TYPED CONTEXT STORE (priority=7.0): Replace flat text memory.py with structured ContextItem dataclass backed by SQLite. Add scope-based filtering (mission, round, unit). Support selective injection into worker prompts. Add tests.

3. QUALITY FIXES: Replace silent 'except Exception: pass' with logged catches in continuous_controller.py. Add cost accumulation resilience. Sanitize brace characters in worker prompt .format() inputs. Fix semaphore private attribute manipulation.

## Completed (prior missions)
- [x] P0 - Priority Backlog Queue (all 5 sub-items)
- [x] P1 - N-of-M fixup retry
- [x] P2 - Architect/Editor two-pass mode
- [x] P3 - Structured planner output (<!-- PLAN --> blocks)
- [x] P5 - Auto-pause and recovery on total dispatch failure

## Files Modified (do NOT re-target these unless fixing a failure)
None yet this mission.

## Remaining
The planner should focus on P4, P6, and quality fixes from the backlog.
