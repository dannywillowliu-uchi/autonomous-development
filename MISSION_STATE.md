# Mission State
Objective: Work through the priority backlog -- P4 (EMA budget tracking) and quality fixes.

## Priority Items
1. P4 - EMA BUDGET TRACKING (priority=8.0): Implement exponential moving average cost tracking per work unit. Add EMA module with alpha=0.3, outlier dampening (>3x EMA clamped to 2x), conservatism factor (k=1.0+0.5/sqrt(n)). Wire into _should_stop() in continuous_controller.py. Add adaptive cooldown. Add tests.

2. QUALITY FIXES: Replace silent 'except Exception: pass' with logged catches in continuous_controller.py. Add cost accumulation resilience. Sanitize brace characters in worker prompt .format() inputs. Fix semaphore private attribute manipulation.

## Completed (prior missions)
- [x] P0 - Priority Backlog Queue (all 5 sub-items)
- [x] P1 - N-of-M fixup retry
- [x] P2 - Architect/Editor two-pass mode
- [x] P3 - Structured planner output (<!-- PLAN --> blocks)
- [x] P5 - Auto-pause and recovery on total dispatch failure
- [x] P6 - Typed context store (ContextItem dataclass, SQLite backing, scope filtering)

## Files Modified (do NOT re-target these unless fixing a failure)
src/mission_control/db.py, src/mission_control/memory.py, src/mission_control/models.py, tests/test_memory.py

## Remaining
The planner should focus on P4 and quality fixes from the backlog.
