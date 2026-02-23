# Dogfooding Issues - 2026-02-23

## P0: Fix diff reviewer REVIEW_RESULT parsing (100% failure rate)
diff_reviewer.py's `_build_review_prompt()` generates a prompt that asks the LLM to output a REVIEW_RESULT marker, but the subprocess output never matches the expected pattern. Every unit review across 4 missions logged "Could not parse REVIEW_RESULT for unit X". This means quality signals (alignment, approach, tests scores) never reach the planner, defeating the purpose of diff review entirely. Debug by: (1) logging the raw subprocess stdout before parsing, (2) checking if the marker regex in the parser matches what the prompt asks for, (3) checking if output is truncated by subprocess capture. The review prompt is in `_build_review_prompt()` and parsing is in `review_unit()`.
**Files**: src/mission_control/diff_reviewer.py

## P0: Prevent pool clones from hijacking editable install
When workers run in pool clones (`.mc-pool-ads/worker-XXXXX`), they execute `uv pip install -e .` which overwrites the venv's editable install path to point at the clone instead of the source tree. After mc/green reset, `Path(__file__)` in any mission_control module resolves to the stale clone path. This breaks the dashboard (UI file not found), imports (ModuleNotFoundError), and any code that uses package-relative paths. Fix options: (a) workers should skip `pip install -e .` if already installed, (b) use `--no-deps` flag, (c) give each worker its own venv, (d) resolve critical paths (like UI file) relative to config target path instead of `__file__`.
**Files**: src/mission_control/dashboard/live.py, src/mission_control/worker.py, src/mission_control/backends/local.py

## P1: Deduplicate backlog items from discovery engine
The discovery engine (`auto_discovery.py`) inserts duplicate items with different IDs. In missions 2-3, "DB lock bypass" and "parallel planner" each appeared twice, consuming 4 of 5 objective slots for only 2 unique tasks. The planner had to manually identify and skip duplicates. Fix: before inserting a new discovery item, check for existing items with similar titles (fuzzy match or normalized title hash). Deduplicate at insert time, not at planning time.
**Files**: src/mission_control/auto_discovery.py, src/mission_control/discovery.py

## P1: Write dashboard token URL to file for background process access
When `mc live` or `mc mission --dashboard-port` runs as a background process, the auth token URL printed to stdout is lost. Users cannot access the dashboard without the token. Write the full URL (with token) to a well-known file like `.mc-dashboard-url` in the project directory, or to a predictable temp path. The `cmd_live()` function in cli.py already generates the token -- just add a file write before starting uvicorn.
**Files**: src/mission_control/cli.py

## P2: Resolve dashboard config/DB path absolutely
`mc live` derives the DB path from the relative config path (`mission-control.toml`). If CWD differs from the project root, it silently creates or opens the wrong database and shows "no active mission". Fix `_get_db_path()` to resolve the config path to absolute before deriving the DB path. Alternatively, search upward for `mission-control.toml` like many config-based tools do.
**Files**: src/mission_control/cli.py

## P2: Increase layer completion drain timeout
The completion queue drain in the dispatch loop has a fixed timeout that fires before all unit completions are processed, logging "Layer 0: completion drain timeout (2/4 processed)". This is cosmetic (units still complete) but misleading. Make the timeout proportional to the number of units in the layer, or use a longer default.
**Files**: src/mission_control/continuous_controller.py
