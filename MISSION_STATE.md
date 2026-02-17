# Mission State
Objective: Work through the priority backlog. Top items:

1. P4 - EMA BUDGET TRACKING: Implement exponential moving average cost tracking per work unit. Add EMA module with alpha=0.3, outlier dampening (>3x EMA clamped to 2x), conservatism factor (k=1.0+0.5/sqrt(n)). Wire into _should_stop() in continuous_controller.py. Add adaptive cooldown that increases when costs exceed budget. Add comprehensive tests.

2. P6 - TYPED CONTEXT STORE: Replace flat text memory.py with structured ContextItem dataclass backed by SQLite. Add scope-based filtering (mission, round, unit). Support selective injection into worker prompts based on relevance. Add tests.

3. QUALITY FIXES from backlog: Replace silent 'except Exception: pass' patterns with logged catches. Add cost accumulation resilience. Sanitize brace characters in worker prompt .format() inputs. Fix semaphore private attribute manipulation.

Each unit: implement the feature, add tests, ensure all existing tests pass. Read BACKLOG.md for full specs.

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.
