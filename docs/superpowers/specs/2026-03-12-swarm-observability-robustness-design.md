# Swarm Observability and Robustness Improvements

## Problem Statement

The swarm produces good code but has blind spots: cost tracking is broken ($0.00 reported for real runs), the planner can't see actual code changes (only inbox summaries), there's no crash recovery (all changes are unstaged until manual commit), learnings accumulate without quality filtering, and there's no completion report. These gaps make it hard to trust autonomous runs without manual post-run review.

## Goals

1. Accurate per-agent and per-cycle cost tracking
2. Planner sees diffs after agent completion (not just inbox messages)
3. Commit-per-task for crash recovery and audit trail
4. Workers self-verify before reporting success
5. Completion report auto-generated when swarm finishes
6. Learning quality filter (drop generic summaries, keep actionable gotchas)
7. Stale-task detection before spawning fix agents

## Non-Goals

- Per-agent verification gate (adds serial overhead, worker self-verify is sufficient)
- Review agent as standard phase (can be added later as planner decision)
- Changing the inbox-based coordination model

## Design

### 1. Fix Cost Tracking

**Problem**: `SwarmState.total_cost_usd` is always 0.0. Cost data exists in agent subprocess output but is never parsed.

**Fix in `controller.py`**:
- When an agent subprocess completes, parse its output for cost data. Claude Code prints `Total cost: $X.XX` or similar in its output.
- Add `_parse_agent_cost(output: str) -> float` method that extracts cost from agent output using regex.
- Accumulate in `self._total_cost_usd` and store per-agent cost in `AgentRecord`.
- Add `cost_usd: float = 0.0` field to agent tracking dict.

**Fix in `swarm/models.py`**:
- Ensure `SwarmState` properly carries cost data to state JSON.

**Fix in `context.py`**:
- Render cost breakdown in planner state: total, per-agent average, budget remaining.

### 2. Planner Reads Diffs

**After agent completion in `controller.py`**:
- Run `git diff --stat` and `git diff --name-only` on the working directory.
- Include a summary of changed files in the agent's completion record.
- Store as `files_changed: list[str]` in the agent's completion data.

**In `context.py render_for_planner()`**:
- Add a "## Recent Changes" section showing files modified by recently completed agents.
- This lets the planner detect when two agents modified the same file, or when a bug-fix target was already changed.

### 3. Commit-Per-Task

**After agent completion in `controller.py`**:
- When a task completes successfully, auto-commit its changes with message: `autodev: {task_title} (agent: {agent_name})`
- Use `git add -A && git commit` scoped to the working directory.
- Store the commit hash in the task's completion record.
- If no files changed, skip the commit.
- On agent failure, do NOT commit (leave changes for retry agent to work with or clean up).

**Guard**: Only commit if there are actual staged changes (avoid empty commits).

### 4. Worker Self-Verification in Prompt

**In `worker_prompt.py build_worker_prompt()`**:
- Add instruction: "Before reporting completion, run the project's verification command to confirm your changes don't break existing tests. If verification fails, fix the issues before reporting."
- Include the verification command from config (`config.target.verification.command`).

### 5. Completion Report

**New method in `controller.py`**: `_generate_completion_report() -> str`

When the swarm stops (all tasks done or hard stop):
- Generate a report with:
  - Tasks completed/failed/skipped
  - Total cost and per-agent breakdown
  - Files changed (from git diff --stat against the starting commit)
  - New modules created
  - Test count delta (if available from agent output)
  - Duration and cycle count
  - Any learnings recorded this run
- Write to `.autodev-swarm-report.md`
- Log the report summary

### 6. Learning Quality Filter

**In `learnings.py`**:
- Add `_score_learning(text: str) -> float` that scores by:
  - Contains a file path or function name: +1
  - Contains "bug", "fix", "race", "gotcha", "must", "never": +1
  - Is under 50 chars (likely generic): -1
  - Contains "all pass", "completed", "successfully": -0.5
- Only persist learnings with score >= 0.5
- Add dedup: hash the learning text, skip if already in the file

### 7. Stale-Task Detection

**In `controller.py` before spawning an agent for a task**:
- If the task description mentions fixing a specific file or function, run a quick `git diff` check on that file.
- If the file was already modified by a recently completed agent, add a note to the spawned agent's prompt: "Note: {file} was recently modified by {agent_name}. Check if the issue is already resolved before making changes."
- This doesn't block spawning (the planner already decided to spawn), but gives the agent context to avoid wasted work.

## File Changes Summary

| File | Change |
|------|--------|
| `src/autodev/swarm/controller.py` | Cost parsing, commit-per-task, diff tracking, completion report, stale-task hints |
| `src/autodev/swarm/context.py` | Recent changes section, cost breakdown in planner state |
| `src/autodev/swarm/models.py` | Cost field on agent data if needed |
| `src/autodev/swarm/worker_prompt.py` | Self-verification instruction |
| `src/autodev/swarm/learnings.py` | Quality filter, dedup |
| `tests/` | Tests for cost parsing, learning filter, completion report |

## Testing

- Unit test for `_parse_agent_cost` with sample Claude Code output formats
- Unit test for learning quality scorer and dedup
- Unit test for completion report generation
- Integration test: mock agent completion -> verify commit created
- All existing tests must continue to pass
