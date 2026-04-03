# GOAL.md Pattern Research

**Source**: https://github.com/jmilinovich/goal-md
**Author**: John Milinovich (@jmilinovich)
**Lineage**: Generalizes Karpathy's autoresearch (March 2026) to domains with constructed metrics
**License**: MIT

## Executive Summary

GOAL.md is a single-file specification that turns a coding agent into an autonomous improver. It defines:
1. A **fitness function** (computable numeric score)
2. An **improvement loop** (measure -> diagnose -> act -> verify -> keep/revert -> log)
3. An **action catalog** (ranked menu of concrete improvements)
4. An **operating mode** (converge, continuous, or supervised)
5. **Constraints** (hard guardrails the agent must never break)

The key insight: Karpathy's autoresearch works because it has a scalar metric (val_bpb) and an immutable eval. Most software doesn't have a natural scalar metric. GOAL.md is the pattern for *constructing* that metric so the same agent loop works on any codebase.

---

## 1. GOAL.md File Format

### Required Sections

Every GOAL.md MUST contain all of these sections (from the template):

```
# Goal: [Name]
## Fitness Function
### Metric Definition
### Metric Mutability
## Operating Mode
### Stopping Conditions
## Bootstrap
## Improvement Loop
## Action Catalog
## Constraints
## File Map
## When to Stop
```

### Optional Sections

```
## Iteration Log          -- format spec for iterations.jsonl
## Known Issues           -- gotchas the agent will encounter
## Architecture           -- file/module structure guidance
## Key Design Decisions   -- architectural rationale
```

### Section Details

#### `# Goal: [Name]`

One-line goal name. Often includes a brief narrative paragraph below it explaining context.

Examples from the repo:
- `# Goal: Make the GOAL.md pattern clear, credible, alive, and seen`
- `# Goal: Ship browser-grid as a working Playwright plugin`
- `# Goal: Bring Storefront API test coverage from 47% to 90%`
- `# Goal: Make the API fast -- then make it faster`

#### `## Fitness Function`

A **runnable shell command** that outputs a number or JSON score. NOT a description -- actual executable code.

```bash
# Run this to get the current score:
[command]
```

The command MUST:
- Be runnable
- Be deterministic
- Finish in under a minute
- Output a number or JSON with `score` and `max` keys

Two output formats are common:
- Human-readable: `./scripts/score.sh`
- Machine-readable: `./scripts/score.sh --json`

Example JSON output format:
```json
{"score": 47, "max": 100, "by_module": {"auth": 91, "products": 82, "orders": 12}}
```

Example human-readable output:
```
═══════════════════════════════════════════
  goal-md: 100 / 130 (77%)
═══════════════════════════════════════════

  CLARITY (is the pattern well-defined?)
    five-elements-defined        ✓ 10/10
    prior-art-section            ✓ 5/5

  RESONANCE (would someone feel something?)
    has-visuals                  ✓ 10/10
    anchor-story                 ✓ 10/10
```

#### `### Metric Definition`

Formula showing how score is computed from components:

```
score = [formula]
```

Plus a table of components:

| Component | What it measures |
|-----------|------------------|
| **[name]** | [description] |

Some examples include `Max` column for weighted components:

| Component | Max | What it measures |
|-----------|-----|------------------|
| **Clarity** | 25 | Are the five elements clearly defined? |
| **Resonance** | 30 | Does it have visuals, personality? |

Formulas can be simple or composite:
- Simple: `score = total_line_coverage_pct`
- Composite: `spec_quality = (clarity + resonance + examples + integrity + distribution) / 130`
- Weighted: `perf_score = (latency_score + throughput_score + cold_start_score + profile_score) / 100`

#### `### Metric Mutability`

Checkbox selection (pick exactly one):

```markdown
- [ ] **Locked** -- Agent cannot modify scoring code
- [ ] **Split** -- Agent can improve the instrument but not the outcome definition
- [ ] **Open** -- Agent can modify everything including how success is measured
```

**Locked**: Autoresearch-style. The eval is immutable. Agent optimizes against a fixed ruler.
**Split**: Agent can improve measurement tools (fix linters, add checks) but can't change what "success" means. Use when measurement infrastructure is imperfect.
**Open**: Everything is part of the work. Use for early-stage projects where the metric itself is being designed.

#### `## Operating Mode`

Checkbox selection (pick exactly one):

```markdown
- [ ] **Converge** -- Stop when criteria met
- [ ] **Continuous** -- Run until human interrupts
- [ ] **Supervised** -- Pause at gates for approval
```

#### `### Stopping Conditions`

Required for converge mode. Machine-checkable conditions, not vibes.

Format:
```markdown
Stop and report when ANY of:
- [condition 1]         e.g., "Score reaches 90%"
- [condition 2]         e.g., "All components score above 80% of their max"
- [max iterations]      e.g., "50 iterations completed"
- [stall detection]     e.g., "5 consecutive iterations with no improvement"
- [external]            e.g., "A test takes longer than 30 seconds"
```

For continuous mode, stopping conditions are replaced with a monitor/watchdog pattern:
```
After 5 consecutive no-improvement iterations, switch to "monitor" mode --
re-run benchmark every 30 minutes, alert if any metric regresses by >10%.
If regression detected, switch back to "optimize" and fix it.
```

#### `## Bootstrap`

Exact shell commands a human must run before the agent can work autonomously. Ends with establishing a baseline score.

```markdown
1. `npm install`
2. `cp .env.example .env.test`
3. Record the baseline: Starting score: [N]
```

#### `## Improvement Loop`

The core iteration protocol. Always follows this structure:

```
repeat:
  0. Read iterations.jsonl if it exists -- note what's been tried and what worked
  1. [measure command] > /tmp/before.json
  2. Read scores and component breakdowns
  3. Pick highest-impact action from Action Catalog
  4. Make the change
  5. If verifiable: run targeted test
  6. [measure command] > /tmp/after.json
  7. Compare: if improved without regression, commit
  8. If regressed or unchanged, revert
  9. Append to iterations.jsonl: before/after scores, action taken, result, one-sentence note
  10. Continue
```

**Commit message convention**: `[S:NN->NN] component: what you did`
- S = Score prefix (or C for Criteria count, P for Performance, COV for coverage)
- Before -> After values
- Component name and brief description

Examples:
- `[S:85->100] clarity: add when-to-use section`
- `[C:3/10->4/10] criterion 4: CDP-powered positioning`
- `[P:71->74] latency: replace JSON.parse with flatbuffers`
- `[COV:47->52] orders: add tests for POST /orders`

**Dual-score variant**: Insert a decision step between steps 2 and 3:
```
If [instrument metric] < [threshold]: fix the instrument first.
If [instrument metric] >= [threshold]: work on [outcome metric].
```

**Continuous-mode variant**: Uses git branches per iteration:
```
1. git checkout -b perf/iter-$(date +%s) main
... make changes and measure ...
10. If improved: commit, merge to main
11. If not: delete branch
```

#### `## Iteration Log`

Specifies `iterations.jsonl` format -- append-only, one JSON object per line:

```jsonl
{"iteration":1,"before":62,"after":65,"action":"Add tests for /api/users","result":"kept","note":"3 new integration tests, coverage +3%"}
{"iteration":2,"before":65,"after":65,"action":"Refactor auth middleware","result":"reverted","note":"broke session handling, no score change"}
```

Fields:
- `iteration`: Sequential number
- `before`: Score before action
- `after`: Score after action
- `action`: What was done
- `result`: `"kept"` or `"reverted"`
- `note`: One-sentence explanation

Purpose: Future agent sessions read this to avoid repeating failed experiments and to build on what worked.

#### `## Action Catalog`

Organized by score component. Each component has a subsection with a table:

```markdown
### [Component Name] (target: [value])

| Action | Impact | How |
|--------|--------|-----|
| [Concrete task] | +N pts | [Step-by-step instructions] |
```

Key properties:
- Actions are **concrete, single-session tasks** with estimated point impact
- Include step-by-step instructions (not just names)
- Include **gotchas and edge cases** inline (see the api-test-coverage example)
- Include **removal actions** where appropriate -- "the best version is often what remains after cutting"
- Status column (`Done`, `Partial`, etc.) can be added for tracking

The catalog is a **starting point**, not a complete map. The agent is expected to discover additional actions through measurement (e.g., flamegraph analysis revealing unexpected hotspots).

#### `## Constraints`

Hard rules the agent must never break. Numbered list with explanations.

```markdown
1. **[constraint]** -- [why]
2. **[constraint]** -- [why]
```

Common constraint patterns across examples:
- Don't modify scoring code/scripts
- Don't mock the database
- Mock all external services
- No new production dependencies
- Tests must pass before every commit
- Keep commits atomic for clean reverts
- No gaming the metric (no `# pragma: no cover`, no fake tests)
- Keep the test suite fast
- No database schema changes

#### `## File Map`

Explicit list of files the agent will read or write:

| File | Role | Editable? |
|------|------|-----------|
| [file] | [role] | Yes / No / Append only / Written by [tool] only |

Critical for preventing metric gaming -- scoring scripts and config are marked "No".

#### `## When to Stop`

Template for the final report the agent produces:

```
Starting score: NN.N
Ending score:   NN.N
Iterations:     N
Changes made:   (list)
Remaining gaps: (list)
Next actions:   (what a human or future agent should do next)
```

---

## 2. Fitness Function Implementation

### Scoring Script Conventions

All scoring scripts follow three rules:
1. Output is JSON with at least `score` and `max` keys
2. Score is an integer between 0 and `max` (no floats)
3. Exit 0 even on bad scores (non-zero exit = script itself broke, not that codebase is unhealthy)

### The `check()` Helper Pattern

The repo's own `score.sh` uses a reusable `check()` function:

```bash
check() {
  local points=$1 name=$2 result=$3 hint="${4:-}"
  max=$((max + points))
  if [[ "$result" == "pass" ]]; then
    score=$((score + points))
    details+=("{\"name\":\"$name\",\"points\":$points,\"max\":$points,\"status\":\"pass\"}")
  elif [[ "$result" == "partial" ]]; then
    local partial=$((points / 2))
    score=$((score + partial))
    details+=("{\"name\":\"$name\",\"points\":$partial,\"max\":$points,\"status\":\"partial\"}")
    [[ -n "$hint" ]] && feedback+=("$name (partial): $hint")
  else
    details+=("{\"name\":\"$name\",\"points\":0,\"max\":$points,\"status\":\"fail\"}")
    [[ -n "$hint" ]] && feedback+=("$name: $hint")
  fi
}
```

This gives three states: pass (full points), partial (half points), fail (zero points). The `hint` provides actionable feedback for the agent.

### Scoring Script Recipes

The repo provides copy-paste patterns for common scenarios:

| Domain | Command | Score Basis |
|--------|---------|-------------|
| Test coverage | `pytest --cov + coverage_score.py` | Line coverage % |
| Documentation | grep for docstrings on exports | % documented exports |
| Build health | `tsc --noEmit` + `eslint` error counts | 100 minus weighted penalty |
| API reliability | Parse access logs | 60% success rate + 40% latency score |
| Code quality | grep TODOs + cyclomatic complexity | 100 minus weighted penalty |
| Criteria checklist | Pass/fail each criterion | Passing count / total |
| Performance | wrk + hyperfine + k6 | Weighted composite of latency, throughput, cold start |

### Anti-Patterns for Fitness Functions

- **Binary metrics (pass/fail)**: Agent has no gradient for partial progress
- **Saturating metrics**: Coverage above ~95% becomes asymptotic, diminishing returns
- **Trivially gameable metrics**: Line count, file count
- **Best fitness functions**: Smooth gradient across the range you care about

### Dual-Score Pattern

For situations where measurement instruments themselves are imperfect:
- **Primary score**: What you're optimizing (e.g., documentation quality)
- **Secondary score**: Confidence in the measurement (e.g., linter precision/recall)

Prevents agents from "fooling themselves" by gaming broken metrics. The improvement loop prioritizes fixing the instrument when its score is below a threshold, then switches to optimizing the outcome.

---

## 3. Iteration Tracking

### File Format

`iterations.jsonl` -- append-only JSONL in the repo root.

```jsonl
{"iteration":1,"before":62,"after":65,"action":"Add tests for /api/users","result":"kept","note":"3 new integration tests, coverage +3%"}
{"iteration":2,"before":65,"after":65,"action":"Refactor auth middleware","result":"reverted","note":"broke session handling, no score change"}
```

### Purpose

1. **Cross-session continuity**: Future agent sessions read this to understand what's been tried
2. **Avoid repeating failures**: If an action was tried and reverted, don't try it again
3. **Build on successes**: Understand which approaches worked and why
4. **Audit trail**: Every iteration is logged, even failures

### How the Loop Uses It

Step 0 of every improvement loop: "Read iterations.jsonl if it exists -- note what's been tried and what worked"

The perf-optimization example also uses `benchmarks/history.jsonl` with richer data including machine info (`uname -a`, `node -v`).

---

## 4. Stopping Conditions

### Converge Mode

Machine-checkable conditions. Stop when ANY triggers:

| Condition Type | Examples |
|---------------|----------|
| Score threshold | `score >= 90`, `score reaches 125/130` |
| Component threshold | `All 5 components score above 80% of their max` |
| Criteria completion | `All 10 criteria pass` |
| Stall detection | `5 consecutive iterations with no improvement` |
| Iteration cap | `20 iterations completed`, `50 iterations completed` |
| External signal | `A test takes longer than 30 seconds` |
| Blocker detection | `3 consecutive criteria yield no progress` |

### Continuous Mode

No stopping conditions. The agent oscillates between two modes:
1. **Optimize**: Actively making improvements
2. **Monitor/Watchdog**: After N no-improvement iterations, switch to periodic benchmarking

Pattern: Optimize until stalled -> watchdog mode (re-benchmark every 30min) -> if regression detected, switch back to optimize -> fix it -> back to watchdog.

### Supervised Mode

Pause at human-defined gates for approval before proceeding. Not deeply explored in the examples.

---

## 5. Action Catalog Design

### Structure

Organized by score component, with target scores per component:

```markdown
### [Component] (target: [value])

| Action | Impact | How |
|--------|--------|-----|
| [task] | +N pts | [detailed step-by-step with gotchas] |
```

### Key Design Principles

1. **Concrete and actionable**: Each action is a single-session task, not a vague goal
2. **Estimated impact**: Point impact lets the agent prioritize
3. **Detailed instructions**: Include file paths, commands, edge cases, gotchas
4. **Ordered by impact**: Highest-impact actions first prevents wasting cycles on low-value changes
5. **Includes removal actions**: Sometimes the best improvement is removing something
6. **Living document**: Status columns can track completion (`Done`, `Partial`)
7. **Starting point, not exhaustive**: Agent discovers additional actions through measurement

### Rich Gotcha Documentation

The api-test-coverage example shows deeply documented gotchas inline with each action:

```markdown
| Test `POST /api/v1/orders` | +4-6% | Create order with valid cart. Assert 201.
**Gotcha**: the handler calls `inventory.reserve()` internally -- if you don't
seed inventory for the SKUs in your cart, you get a 409 that looks like a Stripe
error but is actually an inventory error. |
```

This is the difference between a useful action catalog and a to-do list. The gotchas encode tribal knowledge that prevents the agent from spending cycles debugging known issues.

---

## 6. Revert Mechanism

### Git-Based Reversal

The improvement loop has a built-in revert step:

```
7. Compare: if improved without regression, commit
8. If regressed or unchanged, revert
```

### Atomic Commits

Each iteration is one logical change = one commit. This ensures:
- Clean `git revert` if a change is later found harmful
- Easy `git diff` to understand what each iteration did
- Clean `git log` showing the progression

### Branch-Per-Iteration (Continuous Mode)

The perf-optimization example uses ephemeral branches:

```
1. git checkout -b perf/iter-$(date +%s) main
... make changes and measure ...
10. If improved: commit, merge to main
11. If not: git checkout main && git branch -D perf/iter-*
```

This is cleaner for continuous mode because failed experiments leave no trace in the main branch history.

### Regression Detection

- After every change, re-run the full fitness function
- Compare before/after scores
- If score decreased OR stayed the same, revert
- Some examples add a "retry once" step before reverting

The api-test-coverage example has a nuanced approach:
```
10. Compare: if coverage improved, commit
11. If coverage unchanged, check if tests are actually hitting the right code paths -- adjust and retry once
12. If still unchanged, move to the next endpoint
```

---

## 7. CLAUDE.md Integration

The repo's CLAUDE.md teaches Claude how to write GOAL.md files for other projects:

### Bootstrap Sequence (for writing a new GOAL.md)

1. Read `template/GOAL.md` (skeleton)
2. Read 2-3 examples in `examples/` (calibration)
3. Read the user's codebase to understand what "better" means
4. Write a GOAL.md with all five elements
5. Write or identify the scoring script (must be runnable, not a description)
6. Run the scoring script to establish baseline
7. Start the improvement loop if user wants

### Key Principle

"The GOAL.md you write should stand alone. A future Claude session with no context should be able to open that single file and start working autonomously."

### The One Rule (for working on the repo itself)

"After any change, run `./scripts/score.sh`. Score must not decrease."

---

## 8. Relationship to Autoresearch

| Aspect | Autoresearch | GOAL.md |
|--------|-------------|---------|
| Metric | Natural scalar (val_bpb) | Constructed composite |
| Domain | LLM training | Any software project |
| Eval mutability | Always locked | Locked, Split, or Open |
| Stopping | None (continuous) | Converge, Continuous, or Supervised |
| Action catalog | Implicit (agent decides) | Explicit ranked menu |
| Iteration log | Minimal | Structured JSONL |
| Dual scoring | No | Yes (instrument vs outcome) |
| File map | N/A | Explicit editability boundaries |
| Constraints | Minimal | Detailed guardrails |

GOAL.md's key addition: **constructed metrics**. When there's no natural scalar, you decompose the quality into measurable components, write scripts to measure each, combine into a score, and now you have a number to make go up.

---

## 9. Example Comparison

| Example | Domain | Mode | Metric | Mutability | Notable Feature |
|---------|--------|------|--------|-----------|-----------------|
| `browser-grid.md` | Playwright plugin | Converge | 10 criteria checklist | Locked | Criteria-based (not numeric score) |
| `api-test-coverage.md` | FastAPI test coverage | Converge | Line coverage % | Locked | Deep gotcha documentation, Known Issues section |
| `perf-optimization.md` | Node.js API perf | Continuous | Weighted composite | Locked | Branch-per-iteration, watchdog mode, "agent discovers" section |
| `docs-quality.md` | React docs quality | Converge | Dual-score (docs + instrument) | Split | Dual-score pattern, instrument improvement |
| Repo's own `GOAL.md` | Pattern development | Converge | 5-component composite /130 | Open | Self-referential dogfooding, video/distribution scoring |

---

## 10. Implementation Notes for autodev Integration

### What We Can Adopt

1. **GOAL.md as objective specification**: Replace or supplement our current objective string with a structured GOAL.md file
2. **Fitness function runner**: Execute a scoring command each cycle, use score delta to drive decisions
3. **iterations.jsonl**: Append-only log of what was tried and what worked -- feeds into planner context
4. **Action catalog for planner**: Give the driving planner a ranked menu of actions instead of free-form task decomposition
5. **Revert-on-regression**: Built into the worker completion flow -- if score decreased, revert
6. **Stopping conditions**: Machine-checkable criteria for when the swarm should stop
7. **Commit message format**: `[S:NN->NN] component: description` for score-tracking commits

### Differences from Our Architecture

| GOAL.md Pattern | autodev Swarm |
|----------------|---------------|
| Single agent loop | Multi-agent parallel execution |
| Sequential iterations | Concurrent workers |
| One commit per iteration | Multiple workers committing to green branch |
| Agent reads iterations.jsonl | Planner reads team inbox + learnings |
| Score.sh runs locally | Verification runs in controller |
| File map prevents conflicts | file_lock_registry prevents conflicts |

### Key Adaptation Questions

1. **Who runs the fitness function?** Controller after each merge? Planner each cycle? Workers before/after their changes?
2. **How does the action catalog interact with the planner?** Does it replace planner decomposition or inform it?
3. **Parallel workers + atomic commits**: Multiple workers may improve different components simultaneously. How do we handle score measurement when changes overlap?
4. **Dual scoring in swarm context**: One agent improving measurement while another improves the code being measured?

---

## Sources

- [jmilinovich/goal-md GitHub repository](https://github.com/jmilinovich/goal-md)
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
- [Fortune: "The Karpathy Loop"](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/)
- [VentureBeat: Karpathy's autoresearch](https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai)
- [Mager.co: Autoresearch pattern analysis](https://www.mager.co/blog/2026-03-14-autoresearch-pattern/)
