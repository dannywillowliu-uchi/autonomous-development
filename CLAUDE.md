# mission-control - Claude Code Project Instructions

Long-running autonomous development framework. Spawns Claude Code sessions as subprocesses, manages state in SQLite, discovers work by running code, reviews output algorithmically.

## Verification

Before ANY commit, run:
```
.venv/bin/python -m pytest -q && .venv/bin/ruff check src/ tests/
```

## Architecture

### Core
- `config.py` -- TOML config loader (MissionConfig, HeartbeatConfig, NotificationConfig, etc.)
- `models.py` -- Dataclasses (Session, Snapshot, Plan, WorkUnit, Epoch, UnitEvent, Handoff, etc.)
- `db.py` -- SQLite ops (WAL mode, all CRUD, epoch/unit_event tables)
- `state.py` -- Project health snapshots (run verification, parse output)
- `discovery.py` -- Code-based task discovery (priority 1-7)
- `session.py` -- Claude Code subprocess spawning + MC_RESULT parsing
- `reviewer.py` -- Algorithmic post-session review
- `memory.py` -- Context loading for sessions
- `scheduler.py` -- Main async loop (single-session mode)
- `cli.py` -- argparse CLI (`mc mission`, `mc parallel`, `mc start`)

### Mission Mode (Continuous) -- the primary execution mode
- `continuous_controller.py` -- Event-driven loop: dispatch + completion processor via asyncio.Queue
- `continuous_planner.py` -- Rolling backlog wrapper around RecursivePlanner
- `recursive_planner.py` -- LLM-based recursive plan tree generation with PLAN_RESULT marker
- `green_branch.py` -- mc/green branch lifecycle, merge_unit() for direct merge without verification, ZFC fixup prompt generation
- `hitl.py` -- Human-in-the-loop approval gates (file-based + Telegram polling) for push and large merge actions
- `heartbeat.py` -- Time-based progress monitor (checks merge activity, sends Telegram alerts)
- `notifier.py` -- Telegram notifications (mission start/end, merge conflicts, heartbeat)
- `diff_reviewer.py` -- Fire-and-forget LLM diff review (alignment/approach/tests scoring); feeds quality signals to planner but does NOT gate merges
- `feedback.py` -- Worker context from past experiences
- `overlap.py` -- File overlap detection and dependency injection
- `strategist.py` -- Mission objective proposal, ambition scoring (heuristic + ZFC LLM-backed)

### Infrastructure
- `backends/` -- Worker execution backends (local pool, SSH)
- `worker.py` -- Worker agent, prompt rendering, handoff parsing
- `launcher.py` -- Mission subprocess launcher
- `registry.py` -- Multi-project registry
- `dashboard/` -- TUI (textual) and web (FastAPI+HTMX) dashboards
- `mcp_server.py` -- MCP server for external control

### Execution Flow
1. Controller creates mission, initializes backend + green branch + planner
2. Dispatch loop: planner generates work units, dispatches to workers via semaphore
3. Workers run as Claude Code subprocesses, emit MC_RESULT with handoff data
4. Completion processor: merge to mc/green, ingest handoff, update MISSION_STATE.md
5. Heartbeat monitors progress, sends Telegram alerts
6. Stopping: planner returns empty plan (objective met), heartbeat stall, wall time, or signal
7. Final verification runs on mc/green at mission end

## Gotchas

- recursive_planner.py subprocess MUST set `cwd=str(self.config.target.resolved_path)` -- without it the planner LLM sees the scheduler's own file tree and generates units targeting scheduler files
- Cross-cutting refactors (shared utilities used by multiple views) cause ~85% merge conflict rates with parallel workers -- when planned units have high file overlap, reduce to 1-2 workers or ensure strict dependency ordering
- mc/green must advance after each merge -- if promotion fails silently, subsequent units start from stale baseline
- Pool clones (`git clone --shared`) can corrupt editable installs -- reinstall with `uv pip install -e .` after pool cleanup
- Worker output-format MUST be `text` not `stream-json` -- MC_RESULT markers are invisible inside JSON
- Clean mission-control.db AND .db-shm/.db-wal when resetting (stale WAL causes sqlite3.OperationalError)

## Conventions

- Tabs for indentation
- 120 char line length
- Double quotes
- Python 3.11+
- Type hints on public functions
- Minimal comments
