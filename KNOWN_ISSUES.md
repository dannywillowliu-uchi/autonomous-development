# Known Issues

Tracked from dogfooding session 2026-02-22/23. Missions 1-4 ran autonomously.

## Critical (silently breaks functionality)

### 1. Editable install hijacked by pool clones
**Status**: Unfixed (workaround: manual reinstall)
**Files**: `green_branch.py`, pool clone setup
**Symptom**: After mc/green reset, `pip show mission-control` points to `.mc-pool-ads/worker-XXXXX` instead of source tree. `Path(__file__)` resolves to the clone, breaking dashboard UI file lookup and any import that depends on package location.
**Root cause**: `git clone --shared` pool clones contain their own `pyproject.toml`. When workers run `uv pip install -e .` in their clone, it overwrites the editable install path globally in the venv.
**Fix needed**: Either (a) workers should not reinstall the package, (b) use `--no-deps` or a separate venv per worker, or (c) dashboard should resolve UI path relative to source config, not `__file__`.

### 2. Diff reviewer never parses REVIEW_RESULT
**Status**: Unfixed
**Files**: `diff_reviewer.py`
**Symptom**: Every single unit review across missions 1-4 hit "Could not parse REVIEW_RESULT for unit X". Quality signals never reach the planner.
**Root cause**: Likely the LLM output format doesn't match the expected marker pattern, or the subprocess output is being truncated/corrupted.
**Fix needed**: Investigate what the reviewer subprocess actually outputs vs what the parser expects. May need to update the marker format or add fallback parsing.

### 3. Auto-push fetch refspec missing force flag
**Status**: FIXED (commit 37e5301)
**Files**: `green_branch.py:648`
**Symptom**: Every auto-push silently failed. `push_green_to_main()` fetch returned non-fast-forward error because mc/green was reset at mission start but `refs/mc/green-push` still pointed to previous mission's history.
**Fix**: Added `+` prefix to refspec: `+mc/green:refs/mc/green-push`.
**Note**: Running missions still use old code. Takes effect on next mission start.

## Medium (causes friction but has workarounds)

### 4. Duplicate backlog items from discovery
**Status**: Unfixed
**Files**: `auto_discovery.py`, `discovery.py`
**Symptom**: Discovery engine inserts the same item multiple times with different IDs. E.g., "DB lock bypass" appeared twice, "parallel planner" appeared twice. Planner wastes time deduplicating and duplicate items consume 2 of 5 mission objective slots.
**Fix needed**: Deduplicate by title similarity or content hash before inserting into backlog.

### 5. Dashboard requires auth token in URL but `mc live` output gets swallowed
**Status**: Unfixed
**Files**: `cli.py:cmd_live`
**Symptom**: When `mc live` runs as a background process, the token URL printed to stdout is lost. User can't access the dashboard without the token.
**Fix needed**: Write token URL to a well-known file (e.g., `.mc-dashboard-url`) or a temp file, in addition to printing it.

### 6. Dashboard `mc live` uses relative config path
**Status**: Unfixed
**Files**: `cli.py:_get_db_path`
**Symptom**: If CWD differs from project root, dashboard creates/opens wrong DB path and shows "no active mission".
**Fix needed**: Resolve config path to absolute before deriving DB path, or search upward for `mission-control.toml`.

## Low (cosmetic / noisy logs)

### 7. Non-fast-forward fetch errors in logs
**Status**: Will be fixed by #3's fix propagating
**Symptom**: Every merge logs `ERROR: Failed to fetch mc/green: [rejected] non-fast-forward`. Noisy but non-blocking since the merge itself succeeds.

### 8. Layer completion drain timeout warnings
**Status**: Unfixed
**Files**: `continuous_controller.py`
**Symptom**: "Layer 0: completion drain timeout (2/4 processed)" appears when some units complete faster than others. The completion queue drain has a timeout that fires before all completions are processed, even though they complete fine.
**Fix needed**: Increase drain timeout or make it proportional to unit count.
