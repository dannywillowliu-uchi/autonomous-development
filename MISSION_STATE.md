# Mission State
Objective: Evolve mission control from a task executor into an autonomous engineering lead. Currently, a human must decide what to build, architect solutions, and visually verify results. Build the foundation for the system to do this itself. Focus areas:

1. STRATEGIST AGENT: Add a strategy layer that runs before discovery. It reads BACKLOG.md, git history (last 20 commits), past mission reports from the DB, and the priority queue. It produces a focused mission objective autonomously. Implementation: new file src/mission_control/strategist.py with a Strategist class. Method propose_objective() calls Claude with the gathered context and returns a proposed objective string plus rationale. Add a --strategist flag to the CLI that runs the strategist before the mission starts. The proposed objective is shown to the user for approval (or auto-approved with --approve-all).

2. STRATEGIC CONTEXT ACCUMULATION: Add a rolling strategic context that persists across missions. After each mission, the strategist appends a brief summary to a strategic_context table in SQLite: what was attempted, what worked, what didn't, what the recommended next direction is. This context feeds into future propose_objective() calls. Add get_strategic_context() and append_strategic_context() to db.py.

3. EXPERIMENT MODE: Add an experiment execution mode alongside the existing implementation mode. Experiment units produce comparison reports rather than merged commits. Implementation: add experiment_mode field to WorkUnit. In the worker prompt, experiment units are told to try N approaches (default 2), benchmark each, and report which is better with data. The result is a JSON comparison report stored in the DB (new experiment_results table). No merge to green -- experiments are informational. Add a --experiment flag to the CLI for experiment-only missions.

4. MISSION CHAINING: Add ability to chain missions so the output of one feeds the next. Implementation: add next_objective field to ContinuousMissionResult. When a mission completes, if the strategist determines follow-up work is needed, it sets next_objective. Add a --chain flag to the CLI that automatically starts the next mission using the chained objective. Max chain depth configurable (default 3).

5. AMBITION SCORING: Add self-evaluation of mission ambition. After planning but before execution, score the planned work on a 1-10 ambition scale: 1-3 = busywork (lint fixes, minor refactors), 4-6 = moderate (new features, meaningful improvements), 7-10 = ambitious (architecture changes, new systems, multi-file refactors). Log the score. If ambition < 4 and there are higher-priority backlog items, the strategist should suggest replanning with a more ambitious objective. Add ambition_score to the mission DB record.

Each unit: implement the feature, add tests, ensure all existing tests pass. Read BACKLOG.md for the full spec (Autonomous Engineering Lead section).

## Completed
- [x] 27c55874 (2026-02-16T06:16:28.342807+00:00) -- Added StrategicContext dataclass to models.py, added ambition_score/next_objective/proposed_by_strat (files: src/mission_control/models.py, src/mission_control/db.py, src/mission_control/continuous_controller.py, tests/test_strategic_context_db.py)
- [x] 875c5f11 (2026-02-16T06:20:34.139144+00:00) -- Created src/mission_control/strategist.py with Strategist class that gathers context from BACKLOG.md (files: src/mission_control/strategist.py, tests/test_strategist.py)
- [x] 4dcf4bae (2026-02-16T06:27:13.205278+00:00) -- Added experiment_mode field to WorkUnit dataclass, new ExperimentResult dataclass in models.py. Adde (files: src/mission_control/models.py, src/mission_control/db.py)
- [x] 0e81c10f (2026-02-16T06:28:52.819137+00:00) -- Added --strategist CLI flag and post-mission strategic context integration. CLI proposes objective v (files: src/mission_control/cli.py, src/mission_control/continuous_controller.py, tests/test_strategist_integration.py)
- [x] 3d855d5b (2026-02-16T06:47:28.954859+00:00) -- Added --experiment CLI flag to mission subparser and 20 comprehensive tests in test_experiment_mode. (files: src/mission_control/cli.py, tests/test_experiment_mode.py)
- [x] 90c4d907 (2026-02-16T06:35:42.885341+00:00) -- Added EXPERIMENT_WORKER_PROMPT_TEMPLATE in worker.py with template selection for experiment units. A (files: src/mission_control/worker.py, src/mission_control/continuous_controller.py, src/mission_control/models.py, src/mission_control/db.py)
- [x] fee1ff2d (2026-02-16T06:39:48.945344+00:00) -- Added 11 tests in TestMissionAmbitionScore class verifying ambition_score/next_objective/proposed_by (files: tests/test_db.py)
- [x] 3d855d5b (2026-02-16T06:47:28.954859+00:00) -- Added --experiment CLI flag to mission subparser and 19 comprehensive tests in test_experiment_mode. (files: src/mission_control/cli.py, tests/test_experiment_mode.py)
- [x] 56b114f6 (2026-02-16T06:54:38.826428+00:00) -- Added --chain flag (store_true) and --max-chain-depth (int, default 3) to the mission subparser. In  (files: src/mission_control/cli.py, tests/test_cli.py)

## Files Modified
src/mission_control/cli.py, src/mission_control/continuous_controller.py, src/mission_control/db.py, src/mission_control/models.py, src/mission_control/strategist.py, src/mission_control/worker.py, tests/test_cli.py, tests/test_db.py, tests/test_experiment_mode.py, tests/test_strategic_context_db.py, tests/test_strategist.py, tests/test_strategist_integration.py

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
- 2026-02-16T06:16:28.342807+00:00 | 27c55874 merged (commit: eed6e1d) -- Added StrategicContext dataclass to models.py, added ambition_score/next_objecti
- 2026-02-16T06:20:34.139144+00:00 | 875c5f11 merged (commit: 00df582) -- Created src/mission_control/strategist.py with Strategist class that gathers con
- 2026-02-16T06:27:13.205278+00:00 | 4dcf4bae merged (commit: 707a535) -- Added experiment_mode field to WorkUnit dataclass, new ExperimentResult dataclas
- 2026-02-16T06:28:52.819137+00:00 | 0e81c10f merged (commit: 63cb7d4) -- Added --strategist CLI flag and post-mission strategic context integration. CLI 
- 2026-02-16T06:39:48.945344+00:00 | fee1ff2d merged (commit: 6b3420a) -- Added 11 tests in TestMissionAmbitionScore class verifying ambition_score/next_o
- 2026-02-16T06:47:28.954859+00:00 | 3d855d5b merged (commit: 8c052b4) -- Added --experiment CLI flag to mission subparser and 19 comprehensive tests in t
