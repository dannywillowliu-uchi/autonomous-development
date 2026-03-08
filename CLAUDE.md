# autodev -- Claude Code Project Instructions

Autonomous development framework. Spawns a driving planner that continuously dispatches parallel Claude Code agents, learns from outcomes, and loops until the objective is met. Default mode is swarm (real-time driving planner). Legacy mission mode (epoch-based) is still supported.

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
- `workspace.py` -- Workspace management and directory setup
- `path_security.py` -- Path validation and directory traversal prevention

### Swarm Mode (default)
- `swarm/controller.py` -- Agent lifecycle, task pool, team inbox messaging, subprocess spawning
- `swarm/planner.py` -- Driving planner: async LLM loop (observe -> reason -> decide -> execute)
- `swarm/models.py` -- Swarm data models (AgentRole, AgentStatus, TaskStatus, DecisionType, SwarmState)
- `swarm/prompts.py` -- Planner system prompt, cycle prompt, initial planning prompt
- `swarm/worker_prompt.py` -- Worker prompt builder with inbox reporting instructions
- `swarm/context.py` -- Context synthesizer: builds SwarmState, renders for planner, reads team inboxes
- `swarm/stagnation.py` -- Stagnation detection (flat tests, rising cost, high failure rate) and pivot suggestions
- `swarm/learnings.py` -- Persistent cross-run learnings in `.autodev-swarm-learnings.md`
- `swarm/tui.py` -- Rich-based TUI dashboard (agents, tasks, activity feed, sparklines)

### Mission Mode (legacy)
- `continuous_controller.py` -- Event-driven loop: dispatch + completion processor via asyncio.Queue
- `deliberative_planner.py` -- Ambitious planner + supplementary critic: planner proposes (with web search + project context), critic does feasibility review, iterates until approved
- `recursive_planner.py` -- Flat LLM-based planner: single-call decomposition into work units via <!-- PLAN --> block (fallback when deliberation disabled)
- `critic_agent.py` -- Supplementary critic: feasibility review of proposed plans, chaining objective proposal via CRITIC_RESULT marker
- `context_gathering.py` -- Shared context functions: backlog, git log, past missions, strategic context, episodic memory
- `planner_context.py` -- Minimal planner context (cross-mission semantic memories + recent failures) and fixed-size MISSION_STATE.md writer
- `batch_analyzer.py` -- Heuristic pattern detection from batch signals (file hotspots, failure clusters, stalled areas)
- `continuous_planner.py` -- Flat impact-focused planner wrapper around RecursivePlanner

### Green Branch & Merge
- `green_branch.py` -- autodev/green branch lifecycle, optimistic locking, batch merge, rollback via revert, ZFC fixup prompt generation
- `file_lock_registry.py` -- Advisory file locks for worker coordination
- `overlap.py` -- File overlap detection and dependency injection

### Workers
- `worker.py` -- Worker agent with 5+ prompt templates, handoff parsing, AD_RESULT emission
- `specialist_templates/` -- Domain-specific worker prompt templates
- `backends/local.py` -- Local execution backend with workspace pool

### Planning
- `deliberative_planner.py` + `critic_agent.py` -- Primary chain: planner proposes, critic reviews feasibility, iterates until approved
- `recursive_planner.py` -- Fallback flat planner when deliberation is disabled

### Observability
- `tracing.py` -- Distributed tracing with span context propagation
- `trace_log.py` -- Trace log persistence and querying
- `event_stream.py` -- Event stream for real-time mission state
- `mission_report.py` -- Post-mission summary reports
- `ema.py` -- Exponential moving average metrics

### Safety
- `circuit_breaker.py` -- Trips after N consecutive failures, prevents cascade
- `degradation.py` -- Graceful degradation under load or repeated failures
- `path_security.py` -- Path validation and directory traversal prevention

### Intelligence
- `prompt_evolution.py` -- Prompt mutation and fitness tracking
- `causal.py` -- Causal analysis of mission outcomes
- `grading.py` -- Automated output quality grading
- `intelligence/` -- Advanced intelligence subsystem

### Integration
- `a2a.py` -- Agent-to-agent protocol
- `mcp_server.py` -- MCP server for external control
- `tool_synthesis.py` -- Dynamic tool generation
- `mcp_registry.py` -- MCP server registry and discovery

### Human-in-the-loop
- `hitl.py` -- Approval gates (file-based + Telegram polling) for push and large merge actions
- `heartbeat.py` -- Time-based progress monitor (checks merge activity, sends Telegram alerts)
- `notifier.py` -- Telegram notifications (mission start/end, merge conflicts, heartbeat)

### Infrastructure
- `dashboard/` -- TUI (textual) and web (FastAPI+HTMX) dashboards
- `launcher.py` -- Mission subprocess launcher
- `registry.py` -- Multi-project registry
- `cli.py` -- argparse CLI (`autodev mission`, `autodev swarm`, `autodev swarm-tui`, etc.)
- `session.py` -- Claude Code subprocess spawning + AD_RESULT parsing
- `snapshot.py` -- State snapshots for recovery

## Execution Flow

### Swarm mode (default)
1. Controller initializes team directory (~/.claude/teams/{team_name}/), creates inboxes
2. DrivingPlanner runs initial planning: decompose objective, create tasks, spawn agents
3. Main loop: monitor_agents() -> record_learnings() -> requeue_failed() -> build_state() -> plan_cycle() -> execute_decisions()
4. Agents are Claude Code subprocesses with `--permission-mode auto` and `--max-turns 200`
5. Agents report progress via team inbox files (JSON messages to team-lead.json)
6. Planner reads inboxes each cycle for visibility into working agents
7. Stagnation detection: flat test count -> research pivot, rising cost -> reduce agents, high failure -> diagnostic agent
8. Learnings accumulate in `.autodev-swarm-learnings.md` (persists across runs)
9. State written to `.autodev-swarm-state.json` each cycle for TUI dashboard
10. Stopping: all tasks completed/failed and no active agents
11. Kill guard: agents must be 5+ minutes old before planner can kill them (unless force=True)

### Mission mode (legacy)
1. Controller creates mission, initializes backend + green branch + deliberative planner
2. Deliberation: planner proposes ambitious plan (with web search + project context) -> critic checks feasibility -> planner refines if needed (up to N rounds)
3. batch_analyzer signals (hotspots, clusters) from the previous batch feed into the NEXT planning pass
4. Orchestration loop: deliberate -> execute -> process, repeat
5. file_lock_registry coordinates advisory locks across parallel workers
6. Workers run as Claude Code subprocesses with MCP access, emit AD_RESULT with handoff data
7. Completion processor: merge to autodev/green, ingest handoff, update MISSION_STATE.md (fixed-size summary)
8. tracing/event_stream capture spans and events throughout execution
9. Heartbeat monitors progress, sends Telegram alerts
10. Stopping: planner returns empty plan (objective met), heartbeat stall, wall time, or signal
11. Final verification runs on autodev/green at mission end
12. Chaining (--chain): critic proposes next objective, approval prompt, new mission starts

## Gotchas

- recursive_planner.py subprocess MUST set `cwd=str(self.config.target.resolved_path)` -- without it the planner LLM sees the scheduler's own file tree and generates units targeting scheduler files
- Cross-cutting refactors (shared utilities used by multiple views) cause ~85% merge conflict rates with parallel workers -- when planned units have high file overlap, reduce to 1-2 workers or ensure strict dependency ordering
- autodev/green must advance after each merge -- if promotion fails silently, subsequent units start from stale baseline
- Pool clones (`git clone --shared`) can corrupt editable installs -- reinstall with `uv pip install -e .` after pool cleanup
- Worker output-format MUST be `text` not `stream-json` -- AD_RESULT markers are invisible inside JSON
- Clean autodev.db AND .db-shm/.db-wal when resetting (stale WAL causes sqlite3.OperationalError)
- All subprocess spawning uses `build_claude_cmd()` from config.py -- never build `claude` command lists manually
- MCP config (`[mcp]` in TOML) passes `--mcp-config` to all Claude subprocesses when enabled
- file_lock_registry.py uses advisory locks -- workers must release on crash via atexit or finally
- circuit_breaker.py trips after N consecutive failures -- reset requires explicit call or config change
- tracing.py span context must be propagated to worker subprocesses via environment
- Workers MUST NOT run pip install -- .venv is symlinked and editable install corruption is the #1 support issue
- batch_analyzer.py signals (hotspots, clusters) feed the NEXT planning pass -- they don't affect the current batch
- token_parser.py streaming output can exceed memory limits -- use max_output_mb in LocalBackend
- Swarm agents MUST have `--permission-mode auto` -- without it they hang on permission prompts (stdin is piped)
- Swarm planner cycles need minimum 60s interval -- without this it burns credits on empty responses when pending tasks exist
- Swarm kill guard: 5-minute minimum age prevents premature agent termination. Use `force: True` in tests
- Swarm agents must report progress via inbox -- planner has no visibility into working agents otherwise
- "report" message type must be in context.py filter list for planner to see agent progress reports
- Planner parse failures need exponential backoff + hard stop -- without this, truncated LLM output causes infinite credit-burning loop (MAX_CONSECUTIVE_PARSE_FAILURES=5)
- Use `autodev swarm-inject "message"` to send human directives to a running swarm -- planner sees them as highest-priority items via "directive" message type in team-lead inbox

## Conventions

- Tabs for indentation
- 120 char line length
- Double quotes
- Python 3.11+
- Type hints on public functions
- Minimal comments
