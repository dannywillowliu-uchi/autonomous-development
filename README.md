# autonomous-dev-scheduler

Autonomous dev daemon that continuously improves a codebase toward a "north star" objective. Spawns parallel Claude Code workers, manages state in SQLite, and learns from its own outcomes via an RL-style feedback loop.

Point it at a repo with an objective and a verification command. It plans, executes, merges, verifies, and pushes -- in a loop -- until the objective is met or it stalls. Then it auto-discovers the next objective and chains into a new mission.

## How it works

```
                    +----------------------------------+
                    |     Continuous Controller         |
                    |  (epoch loop, wall-time budget,   |
                    |   ambition gate, review gate)     |
                    +-----------------+----------------+
                                      |
           +--------------------------+-------------------------+
           v                          v                         v
    +--------------+        +----------------+        +----------------+
    |   Planner    |        |   Workers      |        |  Green Branch  |
    |  (recursive, |------->|  (parallel,    |------->|  (merge queue  |
    |  adaptive,   |        |  architect/    |        |  + N-of-M      |
    |  replan on   |        |  editor split, |        |  fixup +       |
    |  stall)      |        |  specialist    |        |  verify +      |
    +--------------+        |  templates)    |        |  promote)      |
           ^                +----------------+        +-------+--------+
           |                                                  |
           |              +------------------+                |
           +--------------+   Feedback       |<---------------+
                          |  (grade, review, |
                          |  reflect, EMA    |
                          |  budget track)   |
                          +------------------+
```

Each epoch:
1. **Plan** -- Recursive planner decomposes the objective into a tree of work units with acceptance criteria, dependency ordering, and specialist assignments
2. **Ambition gate** -- Reject trivially scoped plans (configurable min score) and force replanning
3. **Execute** -- Parallel Claude workers run in isolated workspace clones, optionally using architect/editor two-pass mode
4. **Merge** -- Workers' branches queue into the merge queue and merge into `mc/working` via the green branch manager
5. **Fixup** -- If merge conflicts or verification failures occur, N candidate fixup agents run in parallel; the best-scoring candidate wins
6. **Verify + Promote** -- Verification runs on `mc/working`; passing code promotes to `mc/green`
7. **Review gate** -- LLM diff review scores each unit (alignment, approach, tests). Units below the threshold are retried
8. **Objective verification** -- LLM checks whether the overall objective is met before declaring the mission complete
9. **Feedback** -- Record reflections, compute rewards, track costs via EMA budget tracking
10. **Strategize** -- Strategist proposes follow-up objectives; `--chain` auto-starts the next mission

## Quick start

```bash
# Clone and install
git clone https://github.com/dannywillowliu-uchi/autonomous-dev-scheduler.git
cd autonomous-dev-scheduler
uv venv && uv pip install -e .

# Configure (edit to point at your repo)
cp mission-control.toml.example mission-control.toml
# Edit: target.path, target.objective, target.verification.command

# Run a mission
.venv/bin/python -m mission_control.cli mission --config mission-control.toml --workers 2

# Run with auto-chaining (continues after objective is met)
.venv/bin/python -m mission_control.cli mission --config mission-control.toml --workers 2 --chain

# Live web dashboard
.venv/bin/python -m mission_control.cli live --config mission-control.toml --port 8080
```

## Configuration

All config lives in `mission-control.toml`:

```toml
[target]
name = "my-project"
path = "/path/to/repo"
branch = "main"
objective = "Add comprehensive test coverage for the auth module"

[target.verification]
command = "pytest -q && ruff check src/"
timeout = 120

[scheduler]
model = "opus"           # Model for all Claude subprocesses
session_timeout = 900    # Max seconds per worker session

[scheduler.budget]
max_per_session_usd = 5.0
max_per_run_usd = 100.0

[scheduler.parallel]
num_workers = 2          # Parallel Claude workers
pool_dir = "/tmp/mc-pool"

[rounds]
max_rounds = 20          # Max rounds before stopping
stall_threshold = 5      # Rounds with no improvement before stopping

[planner]
max_depth = 2            # Recursive decomposition depth
max_children_per_node = 5

[continuous]
max_wall_time_seconds = 3600
min_ambition_score = 4          # Reject trivial plans (1-10 scale)
max_replan_attempts = 2         # Replan attempts on ambition rejection
verify_objective_completion = true  # LLM verifies objective before declaring done
max_objective_checks = 2

[review]
gate_completion = true       # Block low-quality units
min_review_score = 5.0       # Minimum average review score (1-10)

[green_branch]
auto_push = true         # Push mc/green to main after each round
push_branch = "main"
fixup_max_attempts = 3
# fixup_candidates = 3   # N-of-M parallel fixup agents

[discovery]
enabled = true
tracks = ["feature", "quality", "security"]
research_enabled = true
```

## Architecture

```
src/mission_control/
+-- cli.py                   # CLI (mission, live, init, discover, summary)
+-- config.py                # TOML config loader + validation
+-- models.py                # Dataclasses (Mission, WorkUnit, Epoch, ...)
+-- db.py                    # SQLite with WAL mode + migrations
+-- continuous_controller.py # Main epoch loop with ambition/review gates
+-- continuous_planner.py    # Adaptive planner with replan-on-stall
+-- recursive_planner.py     # Tree decomposition of objectives
+-- worker.py                # Worker prompt rendering + architect/editor mode
+-- grading.py               # Deterministic decomposition grading
+-- diff_reviewer.py         # LLM diff review (alignment/approach/tests scoring)
+-- reviewer.py              # Review gating logic
+-- feedback.py              # Reflections, rewards, experiences
+-- green_branch.py          # Green branch pattern + N-of-M fixup selection
+-- merge_queue.py           # Ordered merge queue for worker branches
+-- ema.py                   # Exponential moving average budget tracking
+-- memory.py                # Typed context store for workers
+-- session.py               # Claude subprocess spawning + output parsing
+-- strategist.py            # Follow-up objective proposal + mission chaining
+-- auto_discovery.py        # Gap analysis -> research -> backlog pipeline
+-- priority.py              # Backlog priority scoring + recalculation
+-- backlog_manager.py       # Persistent backlog across missions
+-- overlap.py               # Work unit file overlap detection + dependency injection
+-- planner_context.py       # Mission state formatting for planner prompts
+-- mission_report.py        # Post-mission JSON report generation
+-- heartbeat.py             # Liveness monitoring + stale worker recovery
+-- notifier.py              # Telegram notifications with batching + retry
+-- event_stream.py          # SSE event stream for live dashboard
+-- token_parser.py          # Token usage tracking + cost estimation
+-- json_utils.py            # Robust JSON extraction from LLM output
+-- state.py                 # Mission state formatting
+-- launcher.py              # Subprocess launcher utilities
+-- constants.py             # Shared constants
+-- registry.py              # Component registry
+-- dashboard/
|   +-- live.py              # FastAPI app for live web dashboard
|   +-- live_ui.html         # Dashboard frontend (SSE + real-time updates)
|   +-- tui.py               # Terminal UI
|   +-- provider.py          # Dashboard data provider
+-- backends/
|   +-- base.py              # WorkerBackend ABC
|   +-- local.py             # Local subprocess backend with workspace pool
|   +-- ssh.py               # Remote SSH backend
+-- workspace.py             # Git clone pool management
```

## Key concepts

**Green branch pattern**: Workers merge into `mc/working` via a merge queue. After merging, verification runs on `mc/working`. Passing code promotes (ff-merge) to `mc/green`. Only verified code reaches `mc/green`.

**N-of-M fixup selection**: When verification fails, N=3 parallel fixup agents run with different approach hints. Each produces a candidate patch. The system selects the candidate with the best test delta, inspired by tournament-style patch selection from Agentless (UIUC).

**Architect/editor mode**: Workers optionally operate in two passes. Pass 1 (architect) analyzes the codebase and describes needed changes without editing. Pass 2 (editor) implements the architect's plan. Inspired by Aider's architect/editor pattern.

**Ambition gate**: The controller scores planned work units on ambition (1-10). Plans below the threshold are rejected and the planner is forced to replan with feedback, preventing trivially scoped busywork.

**Review gate**: After each unit completes, an LLM diff review scores it on alignment, approach quality, and test meaningfulness. Units below the threshold are retried with review feedback injected.

**Objective verification**: Before declaring a mission complete, an LLM reads the codebase and objective to verify the goal was actually met, preventing premature completion.

**EMA budget tracking**: Per-cycle costs are tracked with exponential moving average (alpha=0.3) with outlier dampening and conservatism factor. Adaptive cooldown increases when costs exceed budget.

**Typed context store**: Workers produce structured context items (architecture notes, gotchas, patterns) stored in SQLite with scope-based filtering. Relevant items are selectively injected into subsequent worker prompts.

**Adaptive planning**: The recursive planner decomposes objectives into a tree of work units with acceptance criteria and dependencies. File overlap detection automatically adds dependency edges between units touching the same files.

**Auto-discovery**: Pipeline that analyzes the codebase for gaps, researches best practices, and populates a persistent backlog with prioritized improvement items. The backlog persists across missions.

**Mission chaining**: With `--chain`, after a mission completes, the strategist proposes the next objective and a new mission starts automatically. Combined with auto-discovery, the system can continuously improve a codebase without human intervention.

**Live dashboard**: Real-time web dashboard at `http://localhost:8080` showing mission state, worker status, merge activity, and cost tracking via Server-Sent Events.

**Workspace pool**: Parallel workers each get an isolated git clone from a pre-warmed pool. Clones are recycled between epochs.

## CLI commands

```bash
# Run a mission
mc mission --config mission-control.toml --workers 2 [--chain]

# Live web dashboard
mc live --config mission-control.toml --port 8080

# Auto-discover improvements
mc discover --config mission-control.toml

# Initialize green branches
mc init --config mission-control.toml

# View mission summary
mc summary --config mission-control.toml
```

## Tests

```bash
uv run pytest -q                           # 800+ tests
uv run ruff check src/ tests/              # Lint
uv run mypy src/mission_control --ignore-missing-imports  # Types
```

## Requirements

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` command available)
- Claude Max or API key with sufficient budget
- Git
