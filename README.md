# autodev

Autonomous development framework. Point it at a repo with an objective and a verification command. It spawns a driving planner that continuously dispatches parallel Claude Code agents, learns from outcomes, and loops until the objective is met.

## Modes

**Swarm (default)**: Real-time driving planner. An async LLM loop cycles every 60s+, monitoring agents, recording learnings, detecting stagnation, and emitting structured decisions (spawn, kill, create_task, adjust, wait, escalate). Agents are Claude Code subprocesses that communicate via file-based team inboxes.

**Mission (legacy)**: Epoch-based orchestration with deliberative planning, critic review, green branch merging, and batch analysis. Still fully supported.

## Capabilities

### Swarm planner loop

```
    +-------------------+
    |  Driving Planner  |  (async LLM loop, cycles every 60s+)
    |                   |
    |  1. Monitor agents|
    |  2. Record learnings
    |  3. Detect stagnation
    |  4. Build state   |
    |  5. Call LLM      |  -> structured decisions
    |  6. Execute       |  -> spawn/kill/create_task/adjust/wait/escalate
    +--------+----------+
             |
    +--------+--------+--------+
    |        |        |        |
 [Agent]  [Agent]  [Agent]  [Agent]   (Claude Code subprocesses)
    |        |        |        |
    +--- team inbox messages ---+
    |                           |
    +---> planner reads <-------+
    |                           |
    +-> .autodev-swarm-learnings.md (persists across runs)
```

Each cycle: monitor agent processes for completion/failure, parse `AD_RESULT` handoffs, record successful/failed approaches to persistent learnings, re-queue failed tasks with retry budget, run stagnation heuristics (flat test count, rising cost with flat progress, high failure rate), feed full state snapshot to LLM for structured decisions, execute via controller.

### Intelligence system

Scans external sources for relevant AI/agent ecosystem developments:
- Hacker News (AI/agent discussions)
- GitHub trending repositories
- ArXiv cs.AI papers
- Claude Code changelogs
- Anthropic, OpenAI, DeepMind blogs

Findings are scored for relevance and converted into adaptation proposals.

### Self-recursion loop

The auto-update pipeline bridges intelligence proposals to swarm missions. It discovers improvements via the intelligence scanner, classifies risk, and either auto-launches low-risk proposals as swarm missions or gates high-risk ones via Telegram approval. Supports `--daemon` mode for recurring cycles.

Safety rails: rate limiting (max daily modifications), diff review gate, ratchet + oracle integration.

### Safety

- **Git ratchet**: Tags HEAD before each self-modification. Runs verification after. Passes keep the change; failures auto-rollback to the checkpoint tag. All outcomes logged to `.autodev-experiments.tsv`.
- **Immutable oracle**: Protects critical config files (pyproject.toml, ruff.toml, autodev.toml, conftest.py, etc.) from modification by auto-update agents.
- **Experiment log**: TSV recording every self-modification with commit, test counts before/after, outcome, cost, and duration.
- **Circuit breaker**: Trips after N consecutive failures to prevent cascading damage.

### Trace system

Full agent stdout/stderr captured to disk under `.autodev-traces/`. DB metadata records run-level stats. `trace-review` analyzes traces for error patterns, file hotspots, wasted agents, and generates recommendations. Supports LLM-assisted deep review with `--deep`.

### Contributor protocol

Git-native multi-user coordination. No central server. Contributors register, claim proposals, publish results, and sync learnings through `.autodev-claims.json` and `.autodev-contributor-registry.json` in the repo.

### Metrics

TSV recording of per-run metrics (test count, pass rate, cost, duration, success rate). Trend analysis across runs. Correlation with self-modifications from the experiment log.

### Green branch merge

Workers commit to isolated unit branches. Completed units merge to `autodev/green`. Pre-merge verification gates the merge; failures trigger fixup agents that attempt repair before rollback.

### Dashboards

- **TUI**: Rich-based terminal dashboard with agent status, task pool, activity feed, and sparklines (`autodev swarm-tui`)
- **Web**: FastAPI + HTMX live dashboard with SSE event stream (`autodev live`)

### Notifications and HITL

Telegram notifications for mission start/end, merge conflicts, heartbeat alerts, and HITL approval gates. File-based and Telegram polling approval for push and large merge actions.

## Quick start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` in PATH)
- Claude Max subscription or API key
- Git

### Install

```bash
git clone git@github.com:dannywillowliu-uchi/autonomous-development.git
cd autonomous-development
uv sync --extra dev
```

Or via pip:

```bash
pip install autonomous-dev
pip install autonomous-dev[mcp,dashboard,tracing]  # with extras
```

### Configure

```bash
cp autodev.toml.example autodev.toml
```

Set the three required fields:

```toml
[target]
name = "my-project"
path = "/absolute/path/to/your/repo"
objective = """What you want built or improved.
Be specific about the end state and success criteria."""

[target.verification]
command = "pytest -q && ruff check src/"   # must exit 0 when healthy
```

### Run

```bash
uv run autodev swarm --config autodev.toml

# With live TUI
uv run autodev swarm-tui --config autodev.toml
```

## Configuration

See [`autodev.toml.example`](autodev.toml.example) for the full annotated config.

| Section | Controls |
|---------|----------|
| `[target]` | Repo path, branch, objective, verification command |
| `[swarm]` | Max/min agents, planner model, cooldown, stagnation threshold |
| `[core_tests]` | Correctness test suite (opt-in, project-defined runner) |
| `[mcp]` | MCP server config passed to all Claude subprocesses |
| `[scheduler]` | Model choice, budget limits, session timeout |
| `[backend]` | Execution backend (local, SSH, or container) |
| `[heartbeat]` | Progress monitoring interval, idle alerts |

## CLI

```bash
# Swarm mode (default)
autodev swarm --config autodev.toml [--max-agents N]
autodev swarm-tui --config autodev.toml
autodev swarm-inject "directive message"

# Mission mode (legacy)
autodev mission --config autodev.toml [--workers N] [--chain] [--approve-all]

# Intelligence and self-update
autodev intel [--threshold 0.3] [--json]
autodev auto-update --config autodev.toml [--dry-run] [--daemon] [--interval 24]

# Trace review
autodev trace-review --config autodev.toml [--last] [--history] [--deep]
autodev trace --file trace.jsonl

# Metrics
autodev metrics --config autodev.toml [--trend] [--correlate] [--last-n 10]

# Contributor coordination
autodev contrib --config autodev.toml register
autodev contrib --config autodev.toml proposals
autodev contrib --config autodev.toml claim <proposal_id>
autodev contrib --config autodev.toml publish <proposal_id> --commit <hash>
autodev contrib --config autodev.toml sync

# Dashboards
autodev live --config autodev.toml [--port 8080]
autodev dashboard --config autodev.toml

# Status and history
autodev status --config autodev.toml
autodev summary --config autodev.toml
autodev history --config autodev.toml

# Setup and diagnostics
autodev init --config autodev.toml
autodev validate-config --config autodev.toml
autodev diagnose --config autodev.toml

# Multi-project registry
autodev register --config autodev.toml
autodev unregister --config autodev.toml
autodev projects

# External interfaces
autodev mcp --config autodev.toml     # MCP server (stdio)
autodev a2a --config autodev.toml     # Agent-to-Agent protocol
```

When running from source, prefix with `uv run`.

## Tests

```bash
uv run pytest -q
uv run ruff check src/ tests/
uv run mypy src/autodev --ignore-missing-imports
```

## Example: C compiler

Used autodev to build a [C compiler](https://github.com/dannywillowliu-uchi/C_compiler_orchestrated) from scratch. Zero human-written compiler code. The core test feedback loop ran against GCC torture tests, and the planner autonomously identified the highest-leverage compiler bugs each epoch.

| Metric | Value |
|--------|-------|
| Wall time | ~5 hours |
| API cost | ~$55 |
| Workers | 4 parallel |
| Units merged | ~35 |
| Source code | 8,106 lines (16 modules) |
| Tests passing | 1,788 |
| GCC torture tests | 221/221 (100%) |
| Human-written compiler code | 0 lines |

## License

MIT
