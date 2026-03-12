# Safety Rails, Shared Compute, and Trace Review

## Problem Statement

Autodev can now modify its own codebase via the auto-update pipeline, but has no guardrails against self-inflicted regressions. There's no mechanism for multiple contributors to pool their agents on the same project. Agent traces are captured but never reviewed -- patterns and failure modes go unnoticed.

## Goals

1. Self-modification safety: git ratchet, rate limiting, immutable oracle, diff review, experiment log
2. Multi-contributor shared compute: anyone with agents can contribute to the project, sharing learnings and pooling work
3. Agent trace review: automated analysis of traces to surface patterns, failures, and improvement opportunities

## Non-Goals

- Building a hosted platform or SaaS (this stays as a git-native protocol)
- Real-time agent coordination across contributors (async is fine)
- Replacing the existing swarm mode (this extends it)

---

## Part 1: Self-Modification Safety Rails

### 1.1 Git Ratchet (Karpathy pattern)

Before any self-modification mission, checkpoint the current state. After the mission, verify. Keep or rollback.

**New file: `src/autodev/ratchet.py`**

```python
class GitRatchet:
	"""Commit/reset ratchet for safe self-modification."""

	def __init__(self, repo_path: Path):
		self._repo = repo_path

	async def checkpoint(self, proposal_id: str) -> str:
		"""Tag current HEAD as pre-modification checkpoint. Returns tag name."""
		# git tag autodev/pre-{proposal_id} HEAD
		# Returns the tag name for rollback

	async def verify_and_decide(self, tag: str, verification_cmd: str) -> bool:
		"""Run verification. If passes, keep. If fails, rollback to tag."""
		# 1. Run verification_cmd (pytest + ruff)
		# 2. If exit code 0: return True (keep)
		# 3. If exit code != 0: git reset --hard {tag}, return False (rollback)

	async def rollback(self, tag: str) -> None:
		"""Hard reset to the checkpoint tag."""
		# git reset --hard {tag}
```

**Integration in `auto_update.py`**:
- Before `_auto_launch()`, call `ratchet.checkpoint(proposal.id)`
- After swarm completes, call `ratchet.verify_and_decide(tag, config.target.verification.command)`
- If rollback, mark proposal as "reverted" in DB

### 1.2 Rate Limiting

5 self-modifications per calendar day. Tracked via the existing `applied_proposals` table.

**In `auto_update.py`**:

```python
def _check_rate_limit(self) -> bool:
	"""Return True if under the daily self-modification limit."""
	today = datetime.now(timezone.utc).date().isoformat()
	count = self._db.count_proposals_applied_since(today)
	return count < self._max_daily_modifications  # default: 5
```

**In `db.py`**: Add `count_proposals_applied_since(since_date: str) -> int` method.

### 1.3 Immutable Oracle

The verification suite and its config are off-limits to self-modifying agents. The swarm cannot game the metric by weakening the tests.

**New file: `src/autodev/oracle.py`**

```python
PROTECTED_PATTERNS = [
	"pyproject.toml",
	"ruff.toml",
	".ruff.toml",
	"setup.cfg",
	"mypy.ini",
	".mypy.ini",
	"conftest.py",
	"autodev.toml",
]

def check_oracle_violation(changed_files: list[str]) -> list[str]:
	"""Return list of protected files that were modified."""
	# Check changed_files against PROTECTED_PATTERNS
	# Used by ratchet to auto-rollback if oracle files were touched
```

**Integration**: After swarm completes, before the keep/rollback decision, check if any protected files were modified. If so, rollback regardless of test results.

### 1.4 Diff Review Gate

After a self-modification swarm completes but before the keep/rollback decision, spawn a review agent that reads the full diff and flags concerns.

**In `auto_update.py`**:

```python
async def _review_modification(self, tag: str) -> dict:
	"""Spawn a review agent to assess the self-modification diff."""
	# 1. git diff {tag}..HEAD
	# 2. Send diff to an LLM with prompt:
	#    "Review this self-modification. Flag: regressions, unnecessary changes,
	#     security concerns, scope creep beyond the proposal. Return JSON with
	#     {approved: bool, concerns: list[str], summary: str}"
	# 3. Return the review result
```

This is a lightweight single-call LLM review, not a full swarm. Uses the configured planner model.

### 1.5 Experiment Log

Persistent TSV log of every self-modification attempt, inspired by Karpathy's `results.tsv`.

**File: `.autodev-experiments.tsv`** (in project root, gitignored)

```
commit	tests_before	tests_after	keep/discard/crash	proposal_title	duration_s	cost_usd	timestamp
abc1234	2742	2755	keep	"Add X feature scanner"	423.5	2.31	2026-03-12T10:00:00Z
def5678	2755	2740	discard	"Refactor planner loop"	312.1	1.87	2026-03-12T14:00:00Z
```

**In `ratchet.py`**: Append a row after every keep/rollback decision.

---

## Part 2: Multi-Contributor Shared Compute

A git-native protocol for multiple people to point their agents at the same project. No central server -- the git repo IS the coordination layer.

### 2.1 Architecture

```
contributor-A (their machine)        contributor-B (their machine)
    |                                     |
    v                                     v
autodev swarm                        autodev swarm
    |                                     |
    v                                     v
local work on branch                 local work on branch
autodev/contrib/{username}/{task}    autodev/contrib/{username}/{task}
    |                                     |
    +---------> shared git remote <-------+
                      |
              shared learnings file
              shared experiment log
              shared proposal registry
```

### 2.2 Contributor Protocol

**New file: `src/autodev/contrib.py`**

```python
class ContributorProtocol:
	"""Git-native multi-contributor coordination."""

	def __init__(self, config: MissionConfig, username: str):
		self._config = config
		self._username = username

	async def claim_proposal(self, proposal_id: str) -> bool:
		"""Claim a proposal for this contributor. Uses git-based locking."""
		# 1. Pull latest from remote
		# 2. Check .autodev-claims.json for existing claims
		# 3. If unclaimed, add claim, commit, push
		# 4. If push fails (someone else claimed), return False
		# Optimistic locking via git push -- first pusher wins

	async def publish_result(self, proposal_id: str, result: ExperimentResult) -> None:
		"""Publish experiment result to shared log."""
		# Append to .autodev-experiments.tsv, commit, push

	async def sync_learnings(self) -> None:
		"""Pull latest shared learnings into local swarm."""
		# git pull, merge .autodev-swarm-learnings.md
```

### 2.3 Shared State Files (in git)

| File | Purpose | Format |
|------|---------|--------|
| `.autodev-claims.json` | Which contributor claimed which proposal | `{proposal_id: {user, claimed_at, status}}` |
| `.autodev-experiments.tsv` | All experiment results from all contributors | TSV (append-only) |
| `.autodev-swarm-learnings.md` | Shared learnings across all contributors | Markdown (append-only, deduped) |
| `.autodev-contributor-registry.json` | Registered contributors and their capabilities | `{username: {joined_at, agent_count, proposals_completed}}` |

### 2.4 Contributor CLI

```bash
# Register as a contributor
autodev contrib register --username dannyliu

# List available proposals to work on
autodev contrib proposals

# Claim and work on a proposal
autodev contrib claim <proposal-id>

# Publish results
autodev contrib publish <proposal-id>

# Sync shared learnings
autodev contrib sync
```

### 2.5 Conflict Resolution

- Branches: each contributor works on `autodev/contrib/{username}/{task-slug}`
- Merge: PRs to main, standard review process
- Claims: optimistic locking via git push (first pusher wins)
- Learnings: append-only, deduped by content hash on sync

---

## Part 3: Agent Trace Review System

Automated analysis of agent traces to surface patterns, failure modes, and improvement opportunities.

### 3.1 Trace Analyzer

**New file: `src/autodev/trace_review.py`**

```python
class TraceAnalyzer:
	"""Review agent traces and surface patterns."""

	def __init__(self, db: Database, project_path: Path):
		self._db = db
		self._project_path = project_path

	async def analyze_run(self, run_id: str) -> RunAnalysis:
		"""Analyze all traces from a single swarm run."""
		traces = self._db.get_agent_traces(run_id=run_id)
		# For each trace:
		#   1. Read the trace file (full log)
		#   2. Extract: errors, retries, tool failures, file conflicts
		#   3. Measure: time-to-first-output, total duration, cost efficiency
		# Aggregate across all traces in the run:
		#   - Success rate, avg duration, cost distribution
		#   - File hotspots (files touched by multiple agents)
		#   - Error patterns (common failure modes)
		#   - Wasted work (agents that did nothing useful)

	async def analyze_history(self, last_n_runs: int = 10) -> HistoryAnalysis:
		"""Analyze patterns across multiple runs."""
		# Cross-run patterns:
		#   - Which task types consistently fail?
		#   - Which agents take longest?
		#   - Cost trend over time
		#   - Recurring error patterns
		#   - Improvement velocity (are modifications making things better?)

	async def generate_report(self, analysis: RunAnalysis | HistoryAnalysis) -> str:
		"""Generate a human-readable report from analysis."""
		# Structured markdown report with:
		#   - Executive summary (1-2 sentences)
		#   - Key metrics table
		#   - Failure patterns (ranked by frequency)
		#   - File conflict hotspots
		#   - Recommendations for next run
		#   - Cost efficiency analysis
```

### 3.2 Analysis Data Models

```python
@dataclass
class RunAnalysis:
	run_id: str
	total_agents: int
	success_rate: float
	total_cost_usd: float
	total_duration_s: float
	file_hotspots: list[tuple[str, int]]  # (file, touch_count)
	error_patterns: list[tuple[str, int]]  # (pattern, occurrence_count)
	wasted_agents: list[str]  # agent names that produced no useful output
	recommendations: list[str]

@dataclass
class HistoryAnalysis:
	runs_analyzed: int
	overall_success_rate: float
	cost_trend: list[float]  # cost per run over time
	recurring_failures: list[tuple[str, int]]
	improvement_velocity: float  # test count change per run
	top_recommendations: list[str]
```

### 3.3 LLM-Assisted Trace Review

For deeper analysis, send trace summaries to an LLM for pattern recognition. This catches things rule-based analysis misses.

**In `trace_review.py`**:

```python
async def llm_review_traces(self, run_id: str) -> str:
	"""Use an LLM to review traces and identify non-obvious patterns."""
	traces = self._db.get_agent_traces(run_id=run_id)
	# Build a summary of each trace: agent name, task, duration, exit code,
	# output_tail (last 2KB), files_changed
	# Send to LLM with prompt:
	#   "Review these agent execution traces from an autonomous dev swarm.
	#    Identify: coordination failures, wasted work, recurring errors,
	#    agents that struggled unnecessarily, patterns the planner should
	#    learn from. Be specific with file names and error messages."
	# Return the LLM's analysis
```

### 3.4 Trace Review CLI

```bash
# Review the most recent run
autodev trace-review --last

# Review a specific run
autodev trace-review --run-id 2026-03-12_03-16-27

# Review history across last N runs
autodev trace-review --history --runs 10

# Deep LLM-assisted review
autodev trace-review --last --deep
```

### 3.5 Auto-Review Hook

After every swarm completion, automatically run the trace analyzer and append findings to `.autodev-swarm-learnings.md`. This closes the feedback loop -- the planner sees trace insights in its next run.

**In `controller.py` cleanup**:

```python
# At swarm completion (after _generate_completion_report):
if self._trace_dir.exists():
	analyzer = TraceAnalyzer(self._db, Path(self._config.target.resolved_path))
	analysis = await analyzer.analyze_run(self._run_id)
	report = await analyzer.generate_report(analysis)
	# Append key findings to learnings file
	# Write full report to .autodev-traces/{run_id}/REVIEW.md
```

---

## File Changes Summary

| File | Change |
|------|--------|
| New: `src/autodev/ratchet.py` | Git ratchet (checkpoint, verify, rollback) |
| New: `src/autodev/oracle.py` | Immutable oracle file protection |
| New: `src/autodev/contrib.py` | Multi-contributor protocol |
| New: `src/autodev/trace_review.py` | Trace analyzer and LLM review |
| `src/autodev/auto_update.py` | Rate limiting, ratchet integration, diff review gate |
| `src/autodev/db.py` | `count_proposals_applied_since()`, trace query helpers |
| `src/autodev/cli.py` | `autodev contrib` and `autodev trace-review` subcommands |
| `src/autodev/swarm/controller.py` | Auto-review hook at swarm completion |
| `.gitignore` | Add `.autodev-experiments.tsv`, `.autodev-claims.json` |
| Tests | Unit tests for ratchet, oracle, rate limiting, trace analysis, contributor protocol |

## Testing

- Ratchet: mock git commands, verify checkpoint/rollback/keep flow
- Oracle: verify protected file detection, rollback on violation
- Rate limit: verify daily count enforcement
- Trace analyzer: mock trace data, verify pattern extraction and report generation
- Contributor protocol: mock git operations, verify claim/publish/sync flow
- Integration: end-to-end auto-update with ratchet and rate limiting (dry-run mode)
- All existing tests must continue to pass
