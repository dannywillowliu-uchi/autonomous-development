# Mission State
Objective: Fix diff_reviewer.py REVIEW_RESULT parsing bug (100% failure rate) and verify quality review signals flow end-to-end from unit merge through build_planner_context() into subsequent planning rounds

Priority backlog items to address:
1. [] Fix diff reviewer REVIEW_RESULT parsing (100% failure rate) (backlog_item_id=3c979afd1c37, priority=8.7): diff_reviewer.py's `_build_review_prompt()` generates a prompt that asks the LLM to output a REVIEW_RESULT marker, but the subprocess output never matches the expected pattern. Every unit review across 4 missions logged "Could not parse REVIEW_RESULT for unit X". This means quality signals (alignment, approach, tests scores) never reach the planner, defeating the purpose of diff review entirely. Debug by: (1) logging the raw subprocess stdout before parsing, (2) checking if the marker regex in the parser matches what the prompt asks for, (3) checking if output is truncated by subprocess capture. The review prompt is in `_build_review_prompt()` and parsing is in `review_unit()`.
**Files**: src/mission_control/diff_reviewer.py
2. [] Prevent pool clones from hijacking editable install (backlog_item_id=49ad5aace36e, priority=8.7): When workers run in pool clones (`.mc-pool-ads/worker-XXXXX`), they execute `uv pip install -e .` which overwrites the venv's editable install path to point at the clone instead of the source tree. After mc/green reset, `Path(__file__)` in any mission_control module resolves to the stale clone path. This breaks the dashboard (UI file not found), imports (ModuleNotFoundError), and any code that uses package-relative paths. Fix options: (a) workers should skip `pip install -e .` if already installed, (b) use `--no-deps` flag, (c) give each worker its own venv, (d) resolve critical paths (like UI file) relative to config target path instead of `__file__`.
**Files**: src/mission_control/dashboard/live.py, src/mission_control/worker.py, src/mission_control/backends/local.py
3. [] Add web research capability to workers and strategist (backlog_item_id=ee4d8e0c4b7b, priority=8.7): Workers currently operate in a filesystem-only bubble. They cannot look up API docs, check library versions, read GitHub issues, or research best practices. The strategist proposes objectives based solely on static code analysis, missing external context like new Claude API features, community patterns, or competitor approaches. Add a web research tool (via MCP or direct subprocess) that workers and the strategist can invoke. The strategist should be able to research the state of the art before proposing objectives. Workers should be able to look up documentation while implementing. This is the difference between a code monkey and an engineer. Start with: (1) a `WebResearchTool` wrapper that workers can call via MCP, (2) inject web search results into strategist context when proposing objectives, (3) allow the planner to flag units as "needs-research" which get web access enabled.
**Files**: src/mission_control/strategist.py, src/mission_control/worker.py, src/mission_control/mcp_server.py
4. [] Implement strategist ambition ladder with capability-expanding objectives (backlog_item_id=c9182619cc10, priority=8.7): The strategist currently proposes objectives by looking at backlog items, which are all internal fixes and quality improvements discovered by static analysis. It never proposes objectives that expand the system's capabilities because the discovery engine only finds bugs and code smells. Add an ambition ladder to the strategist: (1) Level 1: fix bugs and quality issues (current behavior), (2) Level 2: improve existing features (adaptive concurrency, better planning), (3) Level 3: add new capabilities that compound (web research, browser testing, multi-repo support, external integrations), (4) Level 4: meta-improvements that make the system better at improving itself. The strategist should aim for the highest level that's feasible given current state. When all Level 1-2 items are done, it MUST escalate to Level 3-4 instead of finding more lint to fix. Include a "capability gap analysis" step where the strategist compares what the system can do vs what it could do.
**Files**: src/mission_control/strategist.py, src/mission_control/auto_discovery.py
5. [] Add browser automation for live testing of built software (backlog_item_id=c3b4c7465b04, priority=8.7): Mission control builds software but never tests it the way a user would. It runs pytest but never opens a browser, clicks through flows, or validates that a web app actually works. Integrate browser automation (via Playwright MCP or Claude's computer use) so that: (1) after deploying/building a web project, workers can launch a browser and validate the UI, (2) verification includes visual/functional checks not just unit tests, (3) the system can catch issues that only manifest in a real browser (CSS broken, API calls failing, auth flows broken). This is critical for mission control to autonomously ship production software, not just pass tests.
**Files**: src/mission_control/worker.py, src/mission_control/state.py, src/mission_control/config.py

## Completed
- [x] 811d3867 (2026-02-23T07:18:25.318661+00:00) -- Added 4 end-to-end tests covering quality review aggregation in build_planner_context: scores sectio (files: tests/test_planner_context.py)

## In-Flight (DO NOT duplicate)
- [ ] 8934c84e -- Add WebResearchTool to MCP server for strategist and worker web access (files: src/mission_control/mcp_server.py)
- [ ] cd5f000f -- Fix diff_reviewer parsing, prompt structure, and logging (files: src/mission_control/diff_reviewer.py)
- [ ] d2a672c1 -- Add needs_research flag to WorkUnit and planner awareness (files: src/mission_control/models.py, src/mission_control/recursive_planner.py)
- [ ] 01a06c0b -- Resolve dashboard UI path via importlib.resources instead of __file__ (files: src/mission_control/dashboard/live.py, tests/test_live_dashboard.py)

## Files Modified
tests/test_planner_context.py

## System Health
Degradation level: FULL_CAPACITY

## Remaining
The planner should focus on what hasn't been done yet.
Do NOT re-target files in the 'Files Modified' list unless fixing a failure.

## Changelog
- 2026-02-23T07:17:34.133337+00:00 | 47d035ed merged (commit: no-commit) -- n    13→from mission_control.db import Database\n    14→from mission_control.jso
- 2026-02-23T07:17:38.753680+00:00 | f7ebfd6f merged (commit: no-commit) -- ror(\n\t\t\t\t\t\"Worker %s output exceeded limit (%dMB). Killing worker.\",\n\t
