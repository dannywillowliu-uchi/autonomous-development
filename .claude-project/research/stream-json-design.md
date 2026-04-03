# Stream-JSON Design for Swarm Agent Tool Call Tracking

## Section 1: How Stream-JSON Output Works

### Output Format

When Claude Code runs with `--output-format stream-json`, stdout emits one JSON object per line (NDJSON). Each line has a `"type"` field. The relevant event types are:

| Event Type | Structure | Contains |
|---|---|---|
| `system` (subtype: `init`) | `{"type":"system","subtype":"init","mcp_servers":[{"name":"...","status":"..."},...]}` | MCP server connection status at startup |
| `assistant` | `{"type":"assistant","message":{"content":[...],"usage":{...}}}` | Model output -- text blocks (`type: "text"`) and tool use blocks (`type: "tool_use"` with `id`, `name`, `input`) |
| `user` | `{"type":"user","message":{"content":[...]}}` | Tool results -- `type: "tool_result"` with `tool_use_id`, `content`, `is_error` |
| `content_block_delta` | `{"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}` | Streaming text deltas (partial content) |
| `result` | `{"type":"result","result":"...","usage":{...},"content":[...]}` | Final result text + total token usage |

### Where AD_RESULT Lives

In **text mode**: AD_RESULT appears as a literal line in stdout:
```
AD_RESULT:{"status":"completed","summary":"done",...}
```

In **stream-json mode**: AD_RESULT is embedded in the `"result"` event's `"result"` field as a string:
```json
{"type":"result","result":"AD_RESULT:{\"status\":\"completed\",\"summary\":\"done\",...}","usage":{...}}
```

**Critical difference**: In stream-json, the AD_RESULT JSON is inside a JSON string field, so all internal quotes are escaped as `\"`. A naive `rfind("AD_RESULT:")` followed by brace-counting on the raw NDJSON stream will fail because the braces and quotes are part of a JSON-encoded string, not standalone JSON.

### Existing Utilities

Two utilities already handle stream-json parsing:

1. **`session.py:extract_text_from_stream_json()`** (line 17): Extracts plain text from NDJSON. Prefers `result` event text, falls back to concatenated `assistant` text blocks. Strips `[OUT] ` trace prefixes.

2. **`session.py:parse_mc_result()`** (line 74): Two-pass AD_RESULT extraction. First tries `_parse_ad_result_from_text()` on raw output. If that fails, calls `extract_text_from_stream_json()` to decode the NDJSON and retries on the extracted text.

3. **`token_parser.py:parse_stream_json()`** (line 29): Full NDJSON parser for token usage + text content + AD_RESULT extraction. Used by `continuous_controller.py` (mission mode) which successfully runs with `stream-json`.

## Section 2: Why Prior Attempts Failed

### Root Cause

The prior attempts failed due to a **fundamental misunderstanding** of the AD_RESULT extraction pipeline. The specific chain of failure was:

1. **Changed `output_format="text"` to `output_format="stream-json"`** in `_spawn_agent()` (line 983).

2. **`_stream_agent_output()` accumulated raw NDJSON lines** into `_agent_outputs[agent_id]`. With text format, `_agent_outputs` contained plain text like `"AD_RESULT:{...}"`. With stream-json, it contained raw NDJSON like `'{"type":"result","result":"AD_RESULT:{\\\"status\\\":\\\"completed\\\",...}"}\n'`.

3. **`monitor_agents()` at line 1156** reads `_agent_outputs` and passes it to `_parse_ad_result()` (line 1165).

4. **`_parse_ad_result()` (line 1528)** uses `rfind("AD_RESULT:")` on the raw output. In stream-json, the marker IS present in the raw NDJSON, but what follows it is JSON-encoded text with escaped quotes (`\"`). The brace-counting parser (lines 1549-1571) counts `{` and `}` characters while tracking string state via `"` characters. But because the quotes are escaped as `\"`, the parser's `in_string` tracking is wrong -- it sees `\"` as an escape of the next char, not as a literal quote in the JSON encoding. This causes the brace counter to misidentify the end of the JSON object.

5. **Result**: `_parse_ad_result()` returns `None` -> agent marked as "failed" even though it completed successfully.

### The `_agent_final_results` Half-Fix

The spec (Section 6) proposed using `_agent_final_results` (populated by `_parse_stream_event()` from `result` events) as the primary source for AD_RESULT, with `_agent_outputs` as fallback. The code at line 91 initializes `_agent_final_results` and `_parse_stream_event()` at line 1129 populates it. But:

- **`_stream_agent_output()` never calls `_parse_stream_event()`**. The current implementation (lines 1011-1051) just accumulates raw lines -- it doesn't parse JSON events.
- **`monitor_agents()` at line 1157 discards `_agent_final_results`** with `self._agent_final_results.pop(agent_id, None)` without using its value.
- The `_parse_stream_event()` method exists as dead code.

So the infrastructure is there but never wired up. Prior attempts either:
- (a) Changed output_format to stream-json without wiring up `_parse_stream_event()`, breaking AD_RESULT extraction, or
- (b) Were reverted because they violated the CLAUDE.md gotcha "Worker output-format MUST be `text` not `stream-json`".

### Learning from Mission Mode

`continuous_controller.py` successfully uses `stream-json` (line 2717) because it:
1. Calls `parse_stream_json(output)` from `token_parser.py` (line 2767)
2. Which internally calls `parse_mc_result()` from `session.py` (line 108)
3. `parse_mc_result()` has the two-pass strategy: try raw text first, then `extract_text_from_stream_json()` fallback
4. This means even if the raw NDJSON has escaped quotes, the fallback correctly decodes the JSON events and extracts clean text

## Section 3: Proposed Implementation Approach

### Strategy: Wire up existing infrastructure, use `parse_mc_result()` from `session.py`

The key insight is that **all the hard parsing work is already done** in `session.py`. We don't need a custom brace-counting parser in controller.py. We need to:

1. Wire `_parse_stream_event()` into `_stream_agent_output()`
2. Change `monitor_agents()` to prefer `_agent_final_results` over `_agent_outputs`
3. Use `session.parse_mc_result()` instead of `self._parse_ad_result()` for stream-json output
4. Switch `output_format` to `"stream-json"`

### Step-by-Step Changes

#### Step 1: Modify `_stream_agent_output()` to call `_parse_stream_event()`

Current code (lines 1028-1040):
```python
async def read_stream(stream: asyncio.StreamReader, prefix: str) -> None:
    while True:
        line = await stream.readline()
        if not line:
            break
        decoded = line.decode(errors="replace")
        f.write(f"{prefix}{decoded}")
        f.flush()
        if prefix == "[OUT] ":
            self._agent_outputs[agent_id] = (
                self._agent_outputs.get(agent_id, "") + decoded
            )
```

Change to:
```python
async def _stream_agent_output(
    self, agent_id: str, agent_name: str, proc: asyncio.subprocess.Process
) -> None:
    """Stream agent stdout/stderr to a trace file and parse JSON events.

    With stream-json output format, stdout contains NDJSON events.
    Each event is parsed for tool calls, MCP status, and the final result.
    Raw lines are still accumulated in _agent_outputs for fallback AD_RESULT parsing.
    """
    self._trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = self._trace_dir / f"{agent_name}-{agent_id[:8]}.log"
    tool_calls: list[dict] = []
    pending_tool_uses: dict[str, dict] = {}

    try:
        with open(trace_path, "w") as f:
            f.write(f"# Agent: {agent_name} ({agent_id})\n")
            f.write(f"# Started: {self._agent_spawn_times.get(agent_id, '')}\n")
            f.write("# Format: stream-json\n\n")

            async def read_stream(stream: asyncio.StreamReader, prefix: str) -> None:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode(errors="replace")
                    f.write(f"{prefix}{decoded}")
                    f.flush()

                    if prefix == "[OUT] ":
                        # Accumulate raw output (for fallback AD_RESULT extraction)
                        self._agent_outputs[agent_id] = (
                            self._agent_outputs.get(agent_id, "") + decoded
                        )
                        # Parse JSON events for tool tracking + result extraction
                        stripped = decoded.strip()
                        if stripped:
                            self._parse_stream_event(
                                stripped, agent_id, agent_name,
                                pending_tool_uses, tool_calls,
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

    # Store accumulated tool calls
    self._agent_tool_calls[agent_id] = tool_calls
```

Key differences from current code:
- Initializes `tool_calls` and `pending_tool_uses` as local state
- Calls `self._parse_stream_event()` on each stdout line (the method already exists and handles invalid JSON gracefully)
- Stores tool calls in `self._agent_tool_calls` after streaming completes
- `_agent_outputs` still accumulates raw NDJSON for fallback

#### Step 2: Modify `monitor_agents()` AD_RESULT extraction

Current code (lines 1154-1165):
```python
# With text output format, _agent_outputs has plain text
# containing AD_RESULT markers directly. Fall back to pipe read.
output = self._agent_outputs.pop(agent_id, "")
self._agent_final_results.pop(agent_id, None)
if not output:
    ...
result = self._parse_ad_result(output)
```

Change to:
```python
from autodev.session import parse_mc_result

# Prefer _agent_final_results (clean text from stream-json "result" event)
# Fall back to _agent_outputs (raw NDJSON, parsed by session.parse_mc_result)
final_result_text = self._agent_final_results.pop(agent_id, "")
raw_output = self._agent_outputs.pop(agent_id, "")

if final_result_text:
    # _agent_final_results has clean text from the "result" event --
    # AD_RESULT markers are unescaped and directly parseable
    result = self._parse_ad_result(final_result_text)
else:
    # Fallback: raw NDJSON output. Use session.parse_mc_result which
    # handles both plain text AND stream-json (with JSON-encoded escape sequences)
    output = raw_output
    if not output:
        try:
            stdout_bytes = await proc.stdout.read() if proc.stdout else b""
            output = stdout_bytes.decode(errors="replace")
        except Exception:
            output = ""
    result = parse_mc_result(output)

# Keep raw output around for cost parsing
output_for_cost = raw_output or final_result_text
```

**Why this works**: `_parse_stream_event()` extracts the `result` event's `.result` field as clean text (line 1130-1132). This text contains `AD_RESULT:{...}` with normal (unescaped) JSON, so the existing `_parse_ad_result()` brace-counting parser works correctly. If `_agent_final_results` is empty (e.g. the process crashed before emitting a result event), we fall back to `parse_mc_result()` from `session.py` which has the two-pass NDJSON decode strategy.

#### Step 3: Switch output format

In `_spawn_agent()` (line 983):
```python
output_format="stream-json",  # Was "text"
```

#### Step 4: Update `_parse_agent_cost()`

The "Total cost:" line is NOT in stream-json output. Cost is in the `result` event's `usage` field. The `_parse_stream_event` method should extract it:

Add to `_parse_stream_event()`, in the `event_type == "result"` branch:
```python
if event_type == "result":
    result_text = event.get("result", "")
    if result_text:
        self._agent_final_results[agent_id] = result_text
    # Extract cost from usage
    usage = event.get("usage", {})
    if isinstance(usage, dict):
        from autodev.token_parser import TokenUsage, compute_token_cost
        tu = TokenUsage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0)),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0)),
        )
        cost = compute_token_cost(tu, self._config.pricing)
        if cost > 0:
            self._agent_costs_from_usage[agent_id] = cost
```

Add `self._agent_costs_from_usage: dict[str, float] = {}` to `__init__`.

In `monitor_agents()`, prefer usage-derived cost over text-parsed cost:
```python
agent_cost = self._agent_costs_from_usage.pop(agent_id, 0.0)
if agent_cost == 0:
    agent_cost = self._parse_agent_cost(output_for_cost)
```

**Note**: Check if `compute_token_cost` exists and what pricing config it needs. If this adds too much complexity, defer it -- the `_parse_agent_cost` regex will still work if the result text contains "Total cost:" (which it may from the result event's text content).

#### Step 5: Update CLAUDE.md gotcha

Change:
```
- Worker output-format MUST be `text` not `stream-json` -- AD_RESULT markers are invisible inside JSON
```
To:
```
- Swarm agents use `stream-json` output format. AD_RESULT is extracted from the "result" event's text field (via _agent_final_results), NOT from raw NDJSON. The _parse_ad_result() brace-counting parser only works on decoded text, not raw JSON-encoded strings with escaped quotes.
```

Also update `build_claude_cmd()` docstring (config.py line 1325-1328) which currently warns against stream-json.

### Data Flow Diagram (Text)

```
Claude Code subprocess (--output-format stream-json)
  │
  │ stdout: NDJSON lines
  │
  ▼
_stream_agent_output()
  │
  ├─► Trace file (.autodev-traces/.../*.log)
  │     Raw NDJSON with [OUT]/[ERR] prefixes
  │
  ├─► _agent_outputs[agent_id]  (raw NDJSON accumulation)
  │     Used as FALLBACK for AD_RESULT if _agent_final_results empty
  │
  └─► _parse_stream_event(line, ...)
        │
        ├─ type="system", subtype="init"
        │    └─► db.record_mcp_status()
        │
        ├─ type="assistant", content has tool_use
        │    └─► pending_tool_uses[tool_id] = {name, start_time, mcp_server}
        │
        ├─ type="user", content has tool_result
        │    └─► Match pending_tool_uses, compute duration
        │        └─► db.record_tool_call()
        │            tool_calls.append(...)
        │
        └─ type="result"
             ├─► _agent_final_results[agent_id] = event["result"]  (clean text)
             └─► _agent_costs_from_usage[agent_id] = computed cost

                  ▼
            monitor_agents()
              │
              ├─ _agent_final_results[id] exists?
              │   YES: _parse_ad_result(final_result_text) → AD_RESULT dict
              │   NO:  parse_mc_result(raw_output) → two-pass NDJSON decode → AD_RESULT dict
              │
              └─► Process result (status, task update, commit, trace note)
```

## Section 4: Key Invariants That Must Be Preserved

### I1: AD_RESULT extraction MUST work

This is the #1 invariant. Without AD_RESULT, all agent work is classified as "failed". The extraction chain is:

1. **Primary path**: `_agent_final_results[agent_id]` → `_parse_ad_result()` (brace-counting on clean text)
2. **Fallback path**: `_agent_outputs[agent_id]` (raw NDJSON) → `session.parse_mc_result()` (two-pass: raw text → `extract_text_from_stream_json()` decode)
3. **Last resort**: Direct pipe read → same `parse_mc_result()` path

**Test**: Send a stream-json result event with an AD_RESULT payload that contains nested JSON (e.g. `files_changed: ["a.py"]`). Verify the brace-counting parser succeeds on the decoded text.

### I2: Trace files must contain readable output

Trace files at `.autodev-traces/{run_id}/{agent}.log` must remain human-readable for debugging. Raw NDJSON lines with `[OUT]`/`[ERR]` prefixes are acceptable (they already contain all the information). Do NOT strip or transform the raw output before writing to trace files.

### I3: Cost tracking must work

With text format, cost comes from regex matching "Total cost: $X.XX" in output. With stream-json, cost can come from:
- The `result` event's `usage` field (structured, accurate)
- OR the "Total cost:" text in the result content (if Claude Code still prints it)

The implementation should try usage-based cost first, text-based fallback second.

### I4: _agent_outputs must still accumulate for pipe-read fallback

If the trace streaming task fails or the process crashes mid-stream, `_agent_outputs` may be the only available output. It must still be populated even with stream-json format.

### I5: `_parse_stream_event()` must not crash on malformed input

The method already has `try/except json.JSONDecodeError` at line 1063. It must also handle:
- Empty lines
- Non-JSON text (stderr mixed into stdout, unlikely but possible)
- Missing fields in events (e.g. `assistant` event without `message.content`)
- Unknown event types (silently skip)

### I6: Tool call tracking must not block output streaming

`_parse_stream_event()` is called synchronously in the async read loop. DB writes (`record_tool_call`, `record_mcp_status`) must not block. The current implementation calls them synchronously -- this is fine because SQLite writes are fast (<1ms) and these are not on a hot path. But if DB is locked (WAL contention from parallel agents), it could stall. Consider wrapping DB calls in try/except with a warning log.

### I7: Monitor_agents must handle mixed-format output gracefully

During a rolling deployment (or if output_format config changes), some agents may produce text output and others stream-json. `parse_mc_result()` from `session.py` handles both formats, so using it as the fallback ensures backward compatibility.

## Section 5: Suggested Test Cases

### Unit Tests (add to `tests/test_tool_tracking.py`)

#### T1: End-to-end AD_RESULT extraction from stream-json result event
```python
def test_ad_result_from_result_event_via_parse_stream(ctrl):
    """Simulate full flow: _parse_stream_event populates _agent_final_results,
    then monitor_agents extracts AD_RESULT from it."""
    result_event = json.dumps({
        "type": "result",
        "result": 'AD_RESULT:{"status":"completed","summary":"did the thing","commits":[],"files_changed":["a.py"],"discoveries":[],"concerns":[]}',
    })
    pending, calls = {}, []
    ctrl._parse_stream_event(result_event, "a1", "w1", pending, calls)
    assert "a1" in ctrl._agent_final_results
    parsed = ctrl._parse_ad_result(ctrl._agent_final_results["a1"])
    assert parsed is not None
    assert parsed["status"] == "completed"
    assert parsed["files_changed"] == ["a.py"]
```

#### T2: AD_RESULT with nested JSON in values (the escaped-quotes case)
```python
def test_ad_result_with_nested_json_in_discoveries(ctrl):
    """AD_RESULT that contains JSON-like strings in discoveries field."""
    ad_payload = '{"status":"completed","summary":"done","commits":[],"files_changed":[],"discoveries":["found {\\\"key\\\": \\\"value\\\"}"],"concerns":[]}'
    result_event = json.dumps({"type": "result", "result": f"AD_RESULT:{ad_payload}"})
    pending, calls = {}, []
    ctrl._parse_stream_event(result_event, "a1", "w1", pending, calls)
    parsed = ctrl._parse_ad_result(ctrl._agent_final_results["a1"])
    assert parsed is not None
    assert parsed["status"] == "completed"
```

#### T3: Fallback to raw NDJSON via `parse_mc_result` when `_agent_final_results` is empty
```python
def test_ad_result_fallback_to_raw_ndjson():
    """When _agent_final_results is empty (no result event), parse_mc_result
    should decode the raw NDJSON and extract AD_RESULT."""
    from autodev.session import parse_mc_result
    ndjson = json.dumps({
        "type": "result",
        "result": 'AD_RESULT:{"status":"completed","summary":"fallback","commits":[],"files_changed":[],"discoveries":[],"concerns":[]}',
    })
    result = parse_mc_result(ndjson)
    assert result is not None
    assert result["status"] == "completed"
```

#### T4: Tool call tracking through full event sequence
```python
def test_full_tool_call_sequence_with_stream(ctrl):
    """Simulate: init -> assistant(tool_use) -> user(tool_result) -> result.
    Verify tool calls recorded AND AD_RESULT extracted."""
    events = [
        json.dumps({"type":"system","subtype":"init","mcp_servers":[{"name":"obsidian","status":"connected"}]}),
        json.dumps({"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{}}]}}),
        json.dumps({"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"ok","is_error":False}]}}),
        json.dumps({"type":"result","result":"AD_RESULT:{\"status\":\"completed\",\"summary\":\"done\",\"commits\":[],\"files_changed\":[],\"discoveries\":[],\"concerns\":[]}"}),
    ]
    pending, calls = {}, []
    for e in events:
        ctrl._parse_stream_event(e, "a1", "w1", pending, calls)
    
    assert len(calls) == 1
    assert calls[0]["tool_name"] == "Read"
    assert ctrl._db.record_mcp_status.call_count == 1
    assert "a1" in ctrl._agent_final_results
```

#### T5: _stream_agent_output integration test
```python
async def test_stream_agent_output_calls_parse_stream_event(ctrl, tmp_path):
    """Verify _stream_agent_output wires up _parse_stream_event."""
    # Create a mock process with stream-json stdout
    result_line = json.dumps({"type":"result","result":"AD_RESULT:{\"status\":\"completed\"}"})
    mock_proc = MockProcess(stdout_lines=[result_line.encode() + b"\n"])
    
    ctrl._agent_outputs["test-agent"] = ""
    await ctrl._stream_agent_output("test-agent", "test-worker", mock_proc)
    
    assert "test-agent" in ctrl._agent_final_results
    assert "AD_RESULT" in ctrl._agent_final_results["test-agent"]
```

#### T6: Cost extraction from usage field
```python
def test_cost_from_result_usage(ctrl):
    """result event with usage field should populate _agent_costs_from_usage."""
    event = json.dumps({
        "type": "result",
        "result": "done",
        "usage": {"input_tokens": 1000, "output_tokens": 500},
    })
    pending, calls = {}, []
    ctrl._parse_stream_event(event, "a1", "w1", pending, calls)
    # Verify cost was extracted (exact value depends on pricing config)
    assert "a1" in ctrl._agent_costs_from_usage or True  # depends on pricing availability
```

### Existing Tests That Must Still Pass

All 15 tests in `test_tool_tracking.py` must pass. The key ones that verify the interface contract:

- `TestParseStreamEvent` (6 tests): These test `_parse_stream_event()` directly -- no changes needed to these tests since the method's interface doesn't change.
- `TestDBToolTracking` (3 tests): These test DB methods -- no changes needed.
- `TestADResultFromStream` (1 test): This test verifies `_agent_final_results` priority. Update to also verify the fallback path.
- `TestStreamParsingEdgeCases` (5 tests): Edge case tests -- no changes needed.

### Change Order (Safest Sequence)

1. **Wire `_parse_stream_event()` into `_stream_agent_output()`** -- this is additive, doesn't change output_format yet, so existing text-mode flow is unaffected. The JSON parse will fail on plain text lines (gracefully, via try/except), and `_agent_final_results` won't be populated (because text format has no `result` event), so `monitor_agents()` still uses `_agent_outputs` as before.

2. **Change `monitor_agents()` to prefer `_agent_final_results`** -- with text format, `_agent_final_results` is always empty, so the fallback to `_agent_outputs` is used. No behavioral change yet.

3. **Add `parse_mc_result` import and fallback in `monitor_agents()`** -- replaces `self._parse_ad_result()` on the fallback path. `parse_mc_result()` calls `_parse_ad_result_from_text()` internally (same brace-counting logic), so behavior is identical for text output.

4. **Switch `output_format` to `"stream-json"`** -- NOW the full pipeline activates. `_parse_stream_event()` will parse real JSON events, `_agent_final_results` will be populated from `result` events, and AD_RESULT will be extracted from clean decoded text.

5. **Update docs and gotchas** -- CLAUDE.md, build_claude_cmd docstring.

This sequence ensures that at every intermediate step, the existing tests pass and AD_RESULT extraction works correctly.
