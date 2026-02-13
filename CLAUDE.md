# mission-control - Claude Code Project Instructions

Long-running autonomous development framework. Spawns Claude Code sessions as subprocesses, manages state in SQLite, discovers work by running code, reviews output algorithmically.

## Verification

Before ANY commit, run:
```
.venv/bin/python -m pytest -q && .venv/bin/ruff check src/ tests/ && .venv/bin/python -m mypy src/mission_control --ignore-missing-imports
```

## Architecture

### Core
- `config.py` -- TOML config loader (includes ContinuousConfig)
- `models.py` -- Dataclasses (Session, Snapshot, Plan, WorkUnit, Epoch, UnitEvent, etc.)
- `db.py` -- SQLite ops (WAL mode, all CRUD, epoch/unit_event tables)
- `state.py` -- Project health snapshots (run verification, parse output)
- `discovery.py` -- Code-based task discovery (priority 1-7)
- `session.py` -- Claude Code subprocess spawning + MC_RESULT parsing
- `reviewer.py` -- Algorithmic post-session review
- `memory.py` -- Context loading for sessions
- `scheduler.py` -- Main async loop (single-session mode)
- `cli.py` -- argparse CLI (`mc mission --mode rounds|continuous`)

### Mission Mode (Rounds)
- `round_controller.py` -- Batch rounds: plan ALL -> execute ALL -> fixup -> evaluate
- `recursive_planner.py` -- LLM-based recursive plan tree generation
- `green_branch.py` -- mc/working + mc/green branch lifecycle, fixup agent
- `feedback.py` -- Per-round reflection, reward computation, experience retrieval
- `evaluator.py` -- Deterministic scoring from snapshots and completion data

### Mission Mode (Continuous) -- Event-Driven Architecture
- `continuous_controller.py` -- Event-driven loop: dispatch + completion processor via asyncio.Queue
- `continuous_planner.py` -- Rolling backlog wrapper around RecursivePlanner
- `green_branch.py` -- verify_and_merge_unit() for per-unit verify-before-merge
- `feedback.py` -- record_unit_outcome() for per-unit feedback, get_continuous_planner_context()
- `evaluator.py` -- compute_running_score() for cumulative per-unit scoring

### Infrastructure
- `backends/` -- Worker execution backends (local pool, SSH)
- `worker.py` -- Worker agent, prompt rendering, handoff parsing
- `launcher.py` -- Mission subprocess launcher (supports --mode override)
- `registry.py` -- Multi-project registry
- `dashboard/` -- TUI (textual) and web (FastAPI+HTMX) dashboards
- `mcp_server.py` -- MCP server for external control

### Execution Modes
- `mc mission` (default: rounds) -- Batch round loop with fixup gate
- `mc mission --mode continuous` -- Event-driven, no round boundaries, verify-before-merge per unit

## Conventions

- Tabs for indentation
- 120 char line length
- Double quotes
- Python 3.11+
- Type hints on public functions
- Minimal comments
