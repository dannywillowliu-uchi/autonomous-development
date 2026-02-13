# autonomous-dev-scheduler

Autonomous dev daemon that continuously improves a codebase toward a "north star" objective. Spawns parallel Claude Code workers, manages state in SQLite, and learns from its own outcomes via an RL-style feedback loop.

Point it at a repo with an objective and a verification command. It plans, executes, merges, verifies, and pushes -- in a loop -- until the objective is met or it stalls.

## How it works

```
                    ┌─────────────────────────────────┐
                    │         Round Controller         │
                    │   (outer loop, N rounds max)     │
                    └──────────────┬──────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
    ┌─────────────┐      ┌──────────────┐       ┌──────────────┐
    │   Planner   │      │   Workers    │       │   Fixup      │
    │  (Opus,     │─────▶│  (parallel,  │──────▶│  (verify +   │
    │  recursive  │      │  workspace   │       │  promote to  │
    │  tree)      │      │  pool)       │       │  mc/green)   │
    └─────────────┘      └──────────────┘       └──────┬───────┘
           ▲                                           │
           │              ┌──────────────┐             │
           └──────────────│   Feedback   │◀────────────┘
                          │  (reflect,   │
                          │  reward,     │
                          │  experience) │
                          └──────────────┘
```

Each round:
1. **Plan** -- Recursive planner decomposes the objective into a tree of work units, informed by feedback from prior rounds
2. **Execute** -- Parallel Claude workers run in isolated workspace clones, each on its own feature branch
3. **Merge** -- Workers' branches merge into `mc/working` via the green branch manager
4. **Fixup** -- Verification runs on `mc/working`. If it fails, a fixup agent patches until tests pass, then promotes to `mc/green`
5. **Evaluate** -- Score progress toward the objective
6. **Feedback** -- Record reflections, compute rewards, extract experiences for future rounds
7. **Push** -- Auto-push `mc/green` to `main` on origin (configurable)

## Quick start

```bash
# Clone and install
git clone https://github.com/dannywillowliu-uchi/autonomous-dev-scheduler.git
cd autonomous-dev-scheduler
uv venv && uv pip install -e .

# Configure (edit to point at your repo)
cp mission-control.toml.example mission-control.toml
# Edit: target.path, target.objective, target.verification.command

# Run
.venv/bin/python -m mission_control.cli mission --config mission-control.toml --workers 2
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

[green_branch]
auto_push = true         # Push mc/green to main after each round
push_branch = "main"
fixup_max_attempts = 3
```

## Architecture

```
src/mission_control/
├── cli.py                 # CLI entrypoint
├── config.py              # TOML config loader
├── models.py              # 14 dataclasses (Mission, Round, WorkUnit, Reflection, Reward, Experience, ...)
├── db.py                  # SQLite with WAL mode, 15 tables
├── round_controller.py    # Outer mission loop
├── recursive_planner.py   # Tree decomposition of objectives
├── worker.py              # Worker prompt template + subprocess management
├── evaluator.py           # Objective scoring
├── feedback.py            # RL-style feedback: reflections, rewards, experiences
├── green_branch.py        # Green branch pattern (mc/working -> mc/green)
├── memory.py              # Context loading for workers
├── session.py             # Claude subprocess spawning + output parsing
├── backends/
│   ├── base.py            # WorkerBackend ABC
│   └── local.py           # Local subprocess backend with workspace pool
└── workspace.py           # Git clone pool management
```

## Key concepts

**Green branch pattern**: Workers merge freely into `mc/working`. Nothing is verified at merge time. After all workers finish, the fixup agent runs verification on `mc/working`. If tests pass, it promotes (ff-merge) to `mc/green`. If not, the fixup agent patches until they do. Only verified code reaches `mc/green`.

**Recursive planning**: The planner decomposes an objective into a tree of work units. At each node, it decides whether to subdivide further or emit leaf work units. Feedback from prior rounds influences decomposition strategy.

**Feedback loop**: After each round, the system records:
- **Reflections** -- Objective metrics (test deltas, completion rate, score progression)
- **Rewards** -- Composite score from objective signals (no LLM self-evaluation)
- **Experiences** -- Per-unit outcomes indexed by keywords for retrieval in future rounds

**Workspace pool**: Parallel workers each get an isolated git clone from a pre-warmed pool. Clones are recycled between rounds.

## Tests

```bash
.venv/bin/python -m pytest -q              # 495+ tests
.venv/bin/ruff check src/ tests/           # Lint
.venv/bin/python -m mypy src/mission_control --ignore-missing-imports  # Types
```

## Requirements

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` command available)
- Claude Max or API key with sufficient budget
- Git
