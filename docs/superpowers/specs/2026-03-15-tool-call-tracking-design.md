# Tool Call Tracking for Swarm Agents

## Problem

We have no visibility into what tools agents actually use during execution. When an MCP tool fails (auth error, connection refused, missing env var), the failure is buried in the agent's context and we only see it if the entire task fails. We can't answer basic questions like "did any agent try to use nanobanana?" or "which tools fail most often?"

## Solution

Switch swarm agents from `--output-format text` to `--output-format stream-json`. Parse the structured JSON stream in real-time to extract:
1. Every tool call (name, success/fail, duration)
2. MCP server connection status at agent startup
3. AD_RESULT from the final result message

Store tool usage in a new DB table and surface failures in the planner state.

## Implementation

### 1. New DB table: `tool_calls`

In `db.py`, add migration:

```sql
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    mcp_server TEXT NOT NULL DEFAULT '',
    success INTEGER NOT NULL DEFAULT 1,
    error_message TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    duration_ms REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_agent ON tool_calls(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name);
```

Add methods:
- `record_tool_call(run_id, agent_id, agent_name, tool_name, mcp_server, success, error_message, timestamp, duration_ms)`
- `get_tool_usage(run_id=None, agent_id=None, limit=100) -> list[dict]`
- `get_tool_failure_summary(run_id=None) -> list[dict]` -- grouped by tool_name, count of failures, most recent error
- `get_mcp_status(run_id) -> list[dict]` -- MCP server connection status from init events

### 2. New DB table: `mcp_status`

```sql
CREATE TABLE IF NOT EXISTS mcp_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    status TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
```

### 3. Switch agent output format to stream-json

In `swarm/controller.py`, change `build_claude_cmd` call:

```python
cmd = build_claude_cmd(
    self._config,
    model=self._config.scheduler.model,
    prompt=prompt,
    setting_sources=setting_sources,
    permission_mode="auto",
    max_turns=200,
    output_format="stream-json",  # Changed from default "text"
)
```

### 4. Rewrite `_stream_agent_output` to parse JSON stream

Replace the current line-by-line text streaming with JSON line parsing:

```python
async def _stream_agent_output(
    self, agent_id: str, agent_name: str, proc: asyncio.subprocess.Process
) -> None:
    self._trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = self._trace_dir / f"{agent_name}-{agent_id[:8]}.log"
    tool_calls: list[dict] = []
    pending_tool_uses: dict[str, dict] = {}  # tool_use_id -> {name, start_time}

    try:
        with open(trace_path, "w") as f:
            f.write(f"# Agent: {agent_name} ({agent_id})\n")
            f.write(f"# Started: {self._agent_spawn_times.get(agent_id, '')}\n")
            f.write(f"# Format: stream-json\n\n")

            async def read_stream(stream, prefix):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode(errors="replace").strip()
                    f.write(f"{prefix}{decoded}\n")
                    f.flush()

                    if prefix == "[OUT] " and decoded:
                        self._parse_stream_event(
                            decoded, agent_id, agent_name,
                            pending_tool_uses, tool_calls,
                        )

                    if prefix == "[OUT] ":
                        self._agent_outputs[agent_id] = (
                            self._agent_outputs.get(agent_id, "") + decoded + "\n"
                        )

            tasks = []
            if proc.stdout:
                tasks.append(asyncio.create_task(read_stream(proc.stdout, "[OUT] ")))
            if proc.stderr:
                tasks.append(asyncio.create_task(read_stream(proc.stderr, "[ERR] ")))
            if tasks:
                await asyncio.gather(*tasks)

    except Exception as e:
        logger.warning("Trace streaming error for %s: %s", agent_name, e)

    # Store accumulated tool calls to DB
    self._agent_tool_calls[agent_id] = tool_calls
```

### 5. New method `_parse_stream_event` on SwarmController

```python
def _parse_stream_event(
    self,
    line: str,
    agent_id: str,
    agent_name: str,
    pending_tool_uses: dict[str, dict],
    tool_calls: list[dict],
) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return

    event_type = event.get("type")

    # Init event: capture MCP server status
    if event_type == "system" and event.get("subtype") == "init":
        for server in event.get("mcp_servers", []):
            self._db.record_mcp_status(
                run_id=self._run_id,
                agent_id=agent_id,
                server_name=server["name"],
                status=server["status"],
                timestamp=_now_iso(),
            )

    # Tool use: record start
    if event_type == "assistant":
        for content in event.get("message", {}).get("content", []):
            if content.get("type") == "tool_use":
                tool_id = content["id"]
                tool_name = content["name"]
                pending_tool_uses[tool_id] = {
                    "name": tool_name,
                    "start_time": time.monotonic(),
                    "mcp_server": tool_name.split("__")[1] if "__" in tool_name else "",
                }

    # Tool result: record completion
    if event_type == "user":
        for content in event.get("message", {}).get("content", []):
            if content.get("type") == "tool_result":
                tool_id = content.get("tool_use_id", "")
                if tool_id in pending_tool_uses:
                    info = pending_tool_uses.pop(tool_id)
                    duration = (time.monotonic() - info["start_time"]) * 1000
                    result_text = ""
                    if isinstance(content.get("content"), str):
                        result_text = content["content"]
                    elif isinstance(content.get("content"), list):
                        result_text = " ".join(
                            c.get("text", "") for c in content["content"]
                            if isinstance(c, dict)
                        )
                    is_error = content.get("is_error", False)
                    error_msg = result_text[:500] if is_error else ""

                    call = {
                        "tool_name": info["name"],
                        "mcp_server": info["mcp_server"],
                        "success": not is_error,
                        "error_message": error_msg,
                        "timestamp": _now_iso(),
                        "duration_ms": duration,
                    }
                    tool_calls.append(call)

                    self._db.record_tool_call(
                        run_id=self._run_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        **call,
                    )

    # Final result: extract text output and AD_RESULT
    if event_type == "result":
        result_text = event.get("result", "")
        if result_text:
            self._agent_final_results[agent_id] = result_text
```

### 6. Extract AD_RESULT from stream-json result

In `monitor_agents()` where AD_RESULT is currently parsed from text output, change to:

```python
# Try final result text first (from stream-json "result" event)
output = self._agent_final_results.pop(agent_id, "")
if not output:
    output = self._agent_outputs.pop(agent_id, "")
# ... existing AD_RESULT parsing via session.parse_mc_result(output)
```

Add `self._agent_final_results: dict[str, str] = {}` to `__init__`.

### 7. Surface tool failures in planner state

In `swarm/context.py`, add tool failure summary to the state text:

```python
def _render_tool_failures(self, run_id: str) -> str:
    failures = self._db.get_tool_failure_summary(run_id=run_id)
    if not failures:
        return ""
    lines = ["## Tool Failures This Run"]
    for f in failures:
        lines.append(f"- {f['tool_name']}: {f['failure_count']} failures. Last error: {f['last_error'][:100]}")
    return "\n".join(lines)
```

Add MCP connection status section:

```python
def _render_mcp_status(self, run_id: str) -> str:
    statuses = self._db.get_mcp_status(run_id=run_id)
    if not statuses:
        return ""
    failed = [s for s in statuses if s["status"] != "connected"]
    if not failed:
        return ""
    lines = ["## MCP Server Issues"]
    for s in failed:
        lines.append(f"- {s['server_name']}: {s['status']}")
    return "\n".join(lines)
```

### 8. CLI command: `autodev tool-usage`

In `cli.py`, add:

```
autodev tool-usage [--run-id RUN_ID] [--failures-only] [--top N]
```

- Default: show tool usage summary for latest run
- `--failures-only`: only show failed tool calls
- `--top N`: show top N most-used tools

### 9. Update `build_claude_cmd` default output_format

The `output_format` parameter already exists and defaults to `"text"`. No change needed to the function signature. The swarm controller just passes `"stream-json"` when calling it.

Note: the legacy mission mode (`continuous_controller.py`) continues to use `"text"` format -- this change only affects swarm agents.

## Testing

### New tests in `tests/test_tool_tracking.py`

- `test_parse_stream_init_captures_mcp_status`: Feed init JSON, verify mcp_status recorded in DB
- `test_parse_stream_tool_use_and_result`: Feed tool_use then tool_result, verify tool_calls recorded
- `test_parse_stream_tool_error`: Feed tool_result with is_error=True, verify error captured
- `test_parse_stream_result_extracts_text`: Feed result event, verify final text extracted
- `test_parse_stream_invalid_json_skipped`: Feed non-JSON line, verify no crash
- `test_mcp_tool_name_parsing`: Verify "mcp__obsidian__read_note" extracts server="obsidian"
- `test_ad_result_extracted_from_stream`: Full flow with AD_RESULT in result text
- `test_tool_failure_summary_groups_correctly`: Multiple failures for same tool grouped
- `test_mcp_status_deduplication`: Same server reported by multiple agents

### Existing tests

- All existing tests must pass
- `monitor_agents` tests need update to handle stream-json output extraction
- `_stream_agent_output` tests need update for JSON parsing

## Verification

```bash
.venv/bin/python -m pytest -q && .venv/bin/ruff check src/ tests/ && .venv/bin/bandit -r src/ -lll -q
```
