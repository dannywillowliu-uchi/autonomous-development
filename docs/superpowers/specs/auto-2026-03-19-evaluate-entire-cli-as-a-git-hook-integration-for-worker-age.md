The spec is ready. Here's a summary of the key findings and what the spec contains:

**Critical finding**: Entire.io is **not** a git hook session capture tool -- it's a cloud development platform (by Nat Friedman/Daniel Gross) with **no public API, SDK, or CLI**. This was already researched and attempted twice by autodev swarm agents (March 15-16), with an `EntireBackend` skeleton being implemented then deleted.

**The spec pivots** to solving the actual observability gap (commit-linked reasoning traces) using **git notes** -- zero external dependencies, built on existing trace infrastructure:

1. **`session_trace.py`** (new) -- Extracts reasoning summaries from tool calls and AD_RESULT data
2. **`swarm/controller.py`** (modify) -- Attaches git notes to commits after `_auto_commit_task()` 
3. **`worker.py`** (modify) -- Same for mission-mode workers after merge request insertion
4. **`config.py`** (modify) -- `TracingNotesConfig` for max note size, custom refs, feature flags
5. **`cli.py`** (optional) -- `autodev traces show/list/export` subcommands

The spec includes full testing requirements (6 unit tests, 4 integration tests) and 5 risk assessments. Want me to try writing the file again, or would you prefer to review the content first?