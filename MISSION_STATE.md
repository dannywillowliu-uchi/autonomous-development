# Mission State
Objective: Objective met. Continue with 5 remaining backlog items. Top priorities: [security] Add path traversal protection to MCP server and CLI config loading (priority=8.4); [security] Add path traversal protection to MCP server and CLI config loading (priority=8.4); [] Deduplicate backlog items from discovery engine (priority=7.8)

Priority backlog items to address:
1. [security] Add path traversal protection to MCP server and CLI config loading (backlog_item_id=623ec2eea762, priority=8.4): Three entry points accept user-supplied paths with no canonicalization or containment check: (1) mcp_server.py _tool_register_project() at line 288 takes config_path directly from args and passes it to load_config() -- a caller can pass '../../../../etc/passwd' or a symlink. (2) cli.py at ~line 45 accepts --config from argv with no validation. (3) registry.py at ~line 79 stores and retrieves config_path strings without resolving them. Fix: create a validate_config_path(path: str, allowed_bases: list[Path]) utility that (a) resolves the path with pathlib.Path(path).resolve() to eliminate symlinks and '..' components, (b) checks .is_relative_to(base) against each allowed base directory, and (c) raises ValueError with a safe message (no path details) on violation. Apply this at all three entry points. For the MCP server specifically, the allowed_bases should be configurable (default to the user's home directory). For registry.py, validate stored paths on both write and read to prevent a corrupted registry file from being an exploitation vector.
2. [security] Add path traversal protection to MCP server and CLI config loading (backlog_item_id=4a00da0504ad, priority=8.4): Three entry points accept user-supplied paths with no canonicalization or containment check: (1) mcp_server.py _tool_register_project() at line 288 takes config_path directly from args and passes it to load_config() -- a caller can pass '../../../../etc/passwd' or a symlink. (2) cli.py at ~line 45 accepts --config from argv with no validation. (3) registry.py at ~line 79 stores and retrieves config_path strings without resolving them. Fix: create a validate_config_path(path: str, allowed_bases: list[Path]) utility that (a) resolves the path with pathlib.Path(path).resolve() to eliminate symlinks and '..' components, (b) checks .is_relative_to(base) against each allowed base directory, and (c) raises ValueError with a safe message (no path details) on violation. Apply this at all three entry points. For the MCP server specifically, the allowed_bases should be configurable (default to the user's home directory). For registry.py, validate stored paths on both write and read to prevent a corrupted registry file from being an exploitation vector.
3. [] Deduplicate backlog items from discovery engine (backlog_item_id=6bd11f5b1b31, priority=7.8): The discovery engine (`auto_discovery.py`) inserts duplicate items with different IDs. In missions 2-3, "DB lock bypass" and "parallel planner" each appeared twice, consuming 4 of 5 objective slots for only 2 unique tasks. The planner had to manually identify and skip duplicates. Fix: before inserting a new discovery item, check for existing items with similar titles (fuzzy match or normalized title hash). Deduplicate at insert time, not at planning time.
**Files**: src/mission_control/auto_discovery.py, src/mission_control/discovery.py
4. [] Write dashboard token URL to file for background process access (backlog_item_id=f88166926efb, priority=7.8): When `mc live` or `mc mission --dashboard-port` runs as a background process, the auth token URL printed to stdout is lost. Users cannot access the dashboard without the token. Write the full URL (with token) to a well-known file like `.mc-dashboard-url` in the project directory, or to a predictable temp path. The `cmd_live()` function in cli.py already generates the token -- just add a file write before starting uvicorn.
**Files**: src/mission_control/cli.py
5. [] Add external data ingestion for informed decision-making (backlog_item_id=741d79d155b0, priority=7.8): The strategist and planner make decisions in a vacuum. They don't know about: new model releases (Claude 4.5 capabilities), API changes, dependency vulnerabilities, production metrics, user feedback, or market signals. Build an external data ingestion pipeline that: (1) periodically fetches relevant signals (GitHub release feeds, npm/pypi advisories, configured RSS/webhook sources), (2) stores them in a context table, (3) injects relevant signals into strategist and planner prompts. Example: if a new Claude model drops with better tool use, the strategist should know and propose objectives that leverage it. If a dependency has a CVE, it should prioritize upgrading.
**Files**: src/mission_control/strategist.py, src/mission_control/db.py, src/mission_control/config.py

## Completed
- [x] ca67d459 (2026-02-23T07:45:58.409485+00:00) -- Added validate_config_path() utility in path_security.py that resolves paths via Path.resolve() and  (files: src/mission_control/path_security.py, tests/test_path_security.py)

## In-Flight (DO NOT duplicate)
- [ ] daba5f1d -- Deduplicate backlog items in discovery engine (files: src/mission_control/auto_discovery.py, src/mission_control/discovery.py, tests/test_auto_discovery.py)
- [ ] e5a244f9 -- Apply path validation to MCP server and CLI (files: src/mission_control/mcp_server.py, src/mission_control/cli.py, src/mission_control/registry.py, src/mission_control/config.py)

## Files Modified
src/mission_control/path_security.py, tests/test_path_security.py

## System Health
Degradation level: FULL_CAPACITY

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
- 2026-02-23T07:45:58.409485+00:00 | ca67d459 merged (commit: f11b389) -- Added validate_config_path() utility in path_security.py that resolves paths via
