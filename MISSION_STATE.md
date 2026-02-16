# Mission State
Objective: Evolve mission control from a task executor into an autonomous engineering lead. Currently, a human must decide what to build, architect solutions, and visually verify results. Build the foundation for the system to do this itself. Focus areas:

1. STRATEGIST AGENT: Add a strategy layer that runs before discovery. It reads BACKLOG.md, git history (last 20 commits), past mission reports from the DB, and the priority queue. It produces a focused mission objective autonomously. Implementation: new file src/mission_control/strategist.py with a Strategist class. Method propose_objective() calls Claude with the gathered context and returns a proposed objective string plus rationale. Add a --strategist flag to the CLI that runs the strategist before the mission starts. The proposed objective is shown to the user for approval (or auto-approved with --approve-all).

2. STRATEGIC CONTEXT ACCUMULATION: Add a rolling strategic context that persists across missions. After each mission, the strategist appends a brief summary to a strategic_context table in SQLite: what was attempted, what worked, what didn't, what the recommended next direction is. This context feeds into future propose_objective() calls. Add get_strategic_context() and append_strategic_context() to db.py.

3. EXPERIMENT MODE: Add an experiment execution mode alongside the existing implementation mode. Experiment units produce comparison reports rather than merged commits. Implementation: add experiment_mode field to WorkUnit. In the worker prompt, experiment units are told to try N approaches (default 2), benchmark each, and report which is better with data. The result is a JSON comparison report stored in the DB (new experiment_results table). No merge to green -- experiments are informational. Add a --experiment flag to the CLI for experiment-only missions.

4. MISSION CHAINING: Add ability to chain missions so the output of one feeds the next. Implementation: add next_objective field to ContinuousMissionResult. When a mission completes, if the strategist determines follow-up work is needed, it sets next_objective. Add a --chain flag to the CLI that automatically starts the next mission using the chained objective. Max chain depth configurable (default 3).

5. AMBITION SCORING: Add self-evaluation of mission ambition. After planning but before execution, score the planned work on a 1-10 ambition scale: 1-3 = busywork (lint fixes, minor refactors), 4-6 = moderate (new features, meaningful improvements), 7-10 = ambitious (architecture changes, new systems, multi-file refactors). Log the score. If ambition < 4 and there are higher-priority backlog items, the strategist should suggest replanning with a more ambitious objective. Add ambition_score to the mission DB record.

Each unit: implement the feature, add tests, ensure all existing tests pass. Read BACKLOG.md for the full spec (Autonomous Engineering Lead section).

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.
