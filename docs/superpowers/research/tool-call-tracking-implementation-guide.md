# Tool Call Tracking -- Implementation Guide

Research report for implementation agents. Covers the stream-json output format,
current codebase state, what's already done, what's missing, and exact change
specifications for each component.

---

## 0. Current State Summary

**Most of the spec is already implemented.** A prior swarm run landed the following:

- DB tables `tool_calls` and `mcp_status` -- fully implemented in `db.py:595-618`
- DB methods `record_tool_call`, `get_tool_usage`, `get_tool_failure_summary`, `record_mcp_status`, `get_mcp_status` -- fully implemented in `db.py:3374-3484`
- `_stream_agent_output` -- already rewritten for stream-json parsing (`controller.py:768-814`)
- `_parse_stream_event` -- already implemented (`controller.py:816-896`)
- `_spawn_agent` already passes `output_format="stream-json"` (`controller.py:740`)
- `monitor_agents` already extracts from `_agent_final_results` first (`controller.py:919-921`)
- `__init__` already has `_agent_tool_calls`, `_agent_final_results` dicts (`controller.py:87-88`)
- `context.py` already has `_render_tool_failures` and `_render_mcp_status` (`context.py:494-515`)
- `render_for_planner` already calls them when `run_id` is provided (`context.py:471-477`)

**Critical gap found:** `render_state()` in `controller.py:179-181` does NOT pass `run_id` to `render_for_planner()`. This means the tool failure and MCP status sections never render in the planner state. This is the key missing piece for the "planner visibility" spec item.

---

## 1. Claude Code stream-json Output Format

When `claude -p --output-format stream-json` is used, stdout emits one JSON object per line (NDJSON). The event types are:

### 1a. System/Init Event

Emitted once at startup. Contains MCP server connection status:

```json
{
  "type": "system",
  "subtype": "init",
  "mcp_servers": [
    {"name": "obsidian", "status": "connected"},
    {"name": "nanobanana", "status": "connected"},
    {"name": "browser-use", "status": "error", "error": "Connection refused"}
  ],
  "tools": [...],
  "model": "claude-sonnet-4-20250514"
}
```

Key fields:
- `type`: always `"system"`
- `subtype`: `"init"` for startup event
- `mcp_servers`: array of `{name, status}` objects. Status is `"connected"` or `"error"`

### 1b. Assistant Event (contains tool_use blocks)

Emitted when the model produces a response. Contains text and/or tool_use content:

```json
{
  "type": "assistant",
  "message": {
    "id": "msg_xxx",
    "role": "assistant",
    "content": [
      {"type": "text", "text": "Let me read that file."},
      {
        "type": "tool_use",
        "id": "toolu_01ABC",
        "name": "Read",
        "input": {"file_path": "/path/to/file"}
      }
    ],
    "usage": {
      "input_tokens": 1234,
      "output_tokens": 567,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 890
    }
  }
}
```

Key fields for tool tracking:
- `message.content[]` -- iterate for `type == "tool_use"` blocks
- Each tool_use has `id` (correlates to tool_result), `name` (tool name)
- MCP tools have name format `mcp__{server}__{tool}` -- e.g. `mcp__obsidian__read_note`

### 1c. User Event (contains tool_result blocks)

Emitted after tool execution with results:

```json
{
  "type": "user",
  "message": {
    "role": "user",
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_01ABC",
        "content": "File contents here...",
        "is_error": false
      }
    ]
  }
}
```

Key fields:
- `tool_use_id` -- correlates back to the tool_use that initiated it
- `content` -- either a string or array of content blocks
- `is_error` -- boolean, true if the tool call failed

When `is_error` is true, `content` contains the error message:
```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_01DEF",
  "content": "Error: File not found: /bad/path",
  "is_error": true
}
```

### 1d. Result Event (final output)

Emitted once when the session completes:

```json
{
  "type": "result",
  "result": "AD_RESULT:{\"status\":\"completed\",\"summary\":\"...\",\"commits\":[],\"files_changed\":[],\"discoveries\":[],\"concerns\":[]}",
  "usage": {
    "input_tokens": 50000,
    "output_tokens": 3000
  },
  "cost_usd": 0.42,
  "session_id": "xxx"
}
```

Key fields:
- `result` -- the final text output (contains AD_RESULT marker)
- `usage` -- cumulative token usage
- `cost_usd` -- total cost

**Important:** AD_RESULT is embedded inside `result` as text, not as structured JSON at the top level. The existing `_parse_ad_result()` method extracts it via string search for the `AD_RESULT:` marker.

### 1e. Content Block Delta (streaming)

These are intermediate events during streaming:

```json
{
  "type": "content_block_delta",
  "delta": {"type": "text_delta", "text": "partial text..."}
}
```

The current `_parse_stream_event` does not handle these (and doesn't need to for tool tracking).

---

## 2. How `_stream_agent_output` Currently Works

**Location:** `controller.py:768-814`

**Already rewritten for stream-json.** The current implementation:

1. Creates a trace file at `.autodev-traces/{run_id}/{agent_name}-{agent_id[:8]}.log`
2. Initializes `tool_calls: list[dict]` and `pending_tool_uses: dict[str, dict]` for correlation
3. Reads stdout and stderr line-by-line via `asyncio.StreamReader.readline()`
4. For each stdout line, calls `_parse_stream_event()` which handles:
   - `system/init` -- records MCP status to DB via `self._db.record_mcp_status()`
   - `assistant` -- extracts `tool_use` blocks, stores in `pending_tool_uses` with `time.monotonic()` start time
   - `user` -- extracts `tool_result` blocks, correlates via `tool_use_id`, computes duration, records to DB and `tool_calls` list
   - `result` -- extracts `result` text, stores in `self._agent_final_results[agent_id]`
5. All raw output lines are written to the trace file with `[OUT]`/`[ERR]` prefixes
6. Accumulated raw output goes to `self._agent_outputs[agent_id]` (fallback for AD_RESULT extraction)
7. After streaming completes, stores `tool_calls` list in `self._agent_tool_calls[agent_id]`

**No changes needed** to this method or `_parse_stream_event`.

---

## 3. How `build_claude_cmd` Works

**Location:** `config.py:1192-1245`

The function already accepts `output_format: str = "text"` and uses it on line 1215/1217:

```python
cmd: list[str] = [claude_bin, "-p", "--output-format", output_format]
```

**No changes needed.** The swarm controller already passes `output_format="stream-json"` at `controller.py:740`.

**Note for CLAUDE.md:** The gotcha "Worker output-format MUST be `text` not `stream-json`" at CLAUDE.md:135 refers to **legacy mission mode workers** only. Swarm agents use stream-json and parse AD_RESULT correctly via the `result` event and fallback to raw output. The CLAUDE.md entry should be clarified to specify it applies to mission mode (`continuous_controller.py`), not swarm mode.

---

## 4. How `monitor_agents` Extracts AD_RESULT

**Location:** `controller.py:899-1012`

Current flow for a completed agent (process exited):

1. **Wait for trace task** -- `controller.py:910-915`: Waits up to 5s for `_stream_agent_output` to finish, ensuring all output is parsed and `_agent_final_results` is populated.

2. **Extract output** -- `controller.py:917-927`:
   ```python
   output = self._agent_final_results.pop(agent_id, "")  # stream-json result text
   if not output:
       output = self._agent_outputs.pop(agent_id, "")    # raw accumulated output
   if not output:
       # fallback: read remaining stdout bytes directly
       stdout_bytes = await proc.stdout.read() if proc.stdout else b""
       output = stdout_bytes.decode(errors="replace")
   ```

3. **Parse AD_RESULT** -- `controller.py:929`: Calls `self._parse_ad_result(output)` which searches for the last `AD_RESULT:` marker and extracts the JSON.

4. **Cost parsing** -- `controller.py:944-948`: Calls `_parse_agent_cost(output)` on the raw output.

**No changes needed.** The three-tier extraction (final_results -> accumulated output -> pipe read) is already in place and handles both stream-json result events and fallback scenarios.

---

## 5. What `build_state` Includes and Where Tool Failure/MCP Status Sections Go

**Location:** `context.py:178-220` (build_state), `context.py:256-492` (render_for_planner)

### 5a. What `build_state` returns

`SwarmState` dataclass with these fields (populated from various sources):

| Field | Source |
|-------|--------|
| `mission_objective` | Config |
| `agents` | Controller active agents list |
| `tasks` | Controller task pool |
| `recent_completions` | Completed tasks from task list |
| `recent_failures` | Failed tasks from task list |
| `recent_discoveries` | Inbox messages + task results + DB knowledge |
| `available_skills` | `.claude/skills/` directory scan |
| `available_tools` | MCP tool registry |
| `stagnation_signals` | Metric history analysis (flat tests, rising cost, high failure rate) |
| `core_test_results` | Verification runner output |
| `cycle_number` | Incremented each cycle |
| `total_cost_usd` | Controller cost accumulator |
| `wall_time_seconds` | Elapsed time |
| `files_in_flight` | Files being modified by active agents |
| `capabilities` | Skills, agents, hooks, MCP servers manifest |
| `dead_agent_history` | Recently cleaned up agents |
| `recent_file_changes` | Files changed by recently completed agents |
| `agent_costs` | Per-agent cost breakdown |

### 5b. What `render_for_planner` renders

Renders SwarmState into a structured text block with these sections (in order):

1. `## Mission` -- objective text
2. `## HUMAN DIRECTIVES (PRIORITY)` -- injected directives (if any)
3. `## Task Progress` -- completed/in_progress/pending/blocked/failed counts
4. `## Active Agents` -- agent list with role, task, elapsed time, stats
5. `## Agent Progress Reports` -- structured inbox reports from agents
6. `## Task Pool` -- all non-completed tasks with dependency status
7. `## Completed Tasks` -- brief summaries (last 10)
8. `## Failed Tasks` -- with retry info (last 10)
9. `## Recent Discoveries` -- grouped by source
10. `## Core Test Results` -- pass/fail/skip counts
11. `## STAGNATION WARNINGS` -- if stagnation detected
12. `## Available Skills` / `## Available Tools`
13. `## Available Capabilities` -- skills, agent defs, hooks, MCP servers
14. `## Recent Changes` -- files modified by recently completed agents
15. `## Files Currently Being Modified`
16. `## Recently Cleaned Up Agents`
17. `## Completed Work Summary`
18. **`## Tool Failures This Run`** -- **already implemented** at `context.py:494-502`
19. **`## MCP Server Issues`** -- **already implemented** at `context.py:504-515`
20. `## Meta` -- cycle, cost, wall time, avg/top spender

### 5c. The Missing Link: `run_id` Not Passed Through

**This is the critical bug.** The tool failures and MCP status sections (items 18-19 above) are gated on `run_id`:

```python
# context.py:470-477
if run_id:
    tool_failures = self._render_tool_failures(run_id)
    if tool_failures:
        sections.append(tool_failures)
    mcp_issues = self._render_mcp_status(run_id)
    if mcp_issues:
        sections.append(mcp_issues)
```

But the call chain is:

```
planner.py:509  ->  controller.render_state(state)
controller.py:181  ->  self._context.render_for_planner(state)  # NO run_id!
```

`render_state()` at `controller.py:179-181` does not pass `run_id`:

```python
def render_state(self, state: SwarmState) -> str:
    return self._context.render_for_planner(state)
```

**Fix required:** `render_state` must pass `self._run_id` to `render_for_planner`:

```python
def render_state(self, state: SwarmState) -> str:
    return self._context.render_for_planner(state, run_id=self._run_id)
```

---

## 6. Remaining Work Items

### 6a. HIGH PRIORITY: Planner Visibility (the `run_id` gap)

**File:** `src/autodev/swarm/controller.py` (line 179-181)
**Change:** Pass `self._run_id` to `render_for_planner`

```python
# Before:
def render_state(self, state: SwarmState) -> str:
    return self._context.render_for_planner(state)

# After:
def render_state(self, state: SwarmState) -> str:
    return self._context.render_for_planner(state, run_id=self._run_id)
```

This is a one-line change but it's the linchpin -- without it, all the DB recording and rendering code is dead code from the planner's perspective.

### 6b. NORMAL: CLI `autodev tool-usage` Command

**File:** `src/autodev/cli.py`
**Spec:** Add `autodev tool-usage [--run-id RUN_ID] [--failures-only] [--top N]`

This requires:
- Adding a `tool_usage` subcommand to the argparse setup
- Loading the DB
- Calling `get_tool_usage()` or `get_tool_failure_summary()`
- Formatting output as a table

### 6c. CLAUDE.md Gotcha Update

**File:** `CLAUDE.md` (line 135)
**Change:** Clarify that the "Worker output-format MUST be `text`" gotcha applies only to **mission mode workers** (`continuous_controller.py`), not swarm agents. Swarm agents use `stream-json` and handle AD_RESULT extraction correctly via the `result` event type and fallback to raw output.

---

## 7. Files Already Modified (Do Not Re-Implement)

| File | What's Done |
|------|-------------|
| `src/autodev/db.py` | `tool_calls` and `mcp_status` tables, all 5 query methods |
| `src/autodev/swarm/controller.py` | `_stream_agent_output`, `_parse_stream_event`, `__init__` dicts, `output_format="stream-json"`, `monitor_agents` extraction chain |
| `src/autodev/swarm/context.py` | `_render_tool_failures`, `_render_mcp_status`, `render_for_planner` `run_id` parameter |

---

## 8. Testing Notes

- `tests/test_tool_tracking.py` should exist from the prior run -- verify it covers:
  - `_parse_stream_event` with init, tool_use, tool_result, result, and invalid JSON
  - DB methods: `record_tool_call`, `get_tool_usage`, `get_tool_failure_summary`
  - MCP status: `record_mcp_status`, `get_mcp_status`
  - MCP tool name parsing (`mcp__obsidian__read_note` -> server=`obsidian`)
  - AD_RESULT extraction from stream-json result event
  - Tool failure summary grouping
- Existing `monitor_agents` tests may need updating if they mock output format assumptions
- Run full suite: `.venv/bin/python -m pytest -q && .venv/bin/ruff check src/ tests/ && .venv/bin/bandit -r src/ -lll -q`
