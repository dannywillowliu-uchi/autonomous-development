Here's the complete implementation spec:

# Spec: context-mode MCP Server Integration

**Type:** integration
**Priority:** 2
**Effort:** medium
**Risk:** low
**Target modules:** `config.py`, `mcp_registry.py`, `swarm/controller.py`, `swarm/worker_prompt.py`

---

## 1. Problem Statement

Autodev spawns many Claude Code subprocesses (swarm agents and mission workers). Each subprocess has a 200K context window that gets consumed by:

- MCP tool definitions loaded at session start (81+ tools = ~143K tokens before the first message)
- Raw tool outputs: Playwright snapshots (56 KB), `gh issue list` (59 KB), access logs (45 KB)
- After ~30 minutes, 40%+ of context is gone; compaction causes agents to lose track of in-progress work

This directly increases cost (more tokens = more spend) and degrades agent quality (compacted context drops file-edit history, task state, and user decisions).

**context-mode** (`npm: context-mode@1.0.24`, repo: `github.com/mksglu/context-mode`) is an MCP server that addresses both halves:

1. **Context Saving** -- Sandboxed code execution keeps raw data out of the context window. 315 KB becomes 5.4 KB (98% reduction). Ten language runtimes (JS, TS, Python, Shell, Ruby, Go, Rust, PHP, Perl, R) run in isolated subprocesses; only stdout enters context.
2. **Session Continuity** -- Every file edit, git op, task, error, and decision is tracked in a local SQLite FTS5 index. On compaction, only BM25-relevant events are retrieved instead of dumping full history.

### Tools Exposed

| Tool | Purpose |
|------|---------|
| `ctx_execute` | Run code in a sandboxed subprocess (any of 10 runtimes). Only stdout enters context. |
| `ctx_batch_execute` | Run multiple scripts in sequence within one tool call. |
| `ctx_execute_file` | Execute a file from disk in the sandbox. |
| `ctx_index` | Index content into the FTS5 knowledge base for later retrieval. |
| `ctx_search` | BM25 search over indexed knowledge base entries. |
| `ctx_fetch_and_index` | Fetch a URL, convert to markdown, index into knowledge base. |

### Why This Matters for Autodev

- **Swarm agents** run for extended periods (max 200 turns). Context exhaustion is the #1 reason agents degrade mid-task.
- **Parallel workers** (4-8 concurrent) multiply the cost impact. 98% context reduction on tool outputs translates to significant cost savings.
- **Planner visibility** improves when agents maintain coherent context longer (fewer compaction-induced amnesia events).
- **No cloud dependency** -- all data stays local (SQLite in `~/.context-mode/`), no telemetry, no account required. Aligns with autodev's local-first architecture.

---

## 2. Changes Needed

### 2.1 Install context-mode as a project dependency

**Not a code change** -- prerequisite setup:

```bash
npm install -g context-mode
```

Verify the binary is available: `which context-mode` should resolve. The MCP server is a stdio server invoked as `npx -y context-mode` (or just `context-mode` if installed globally).

### 2.2 Add context-mode to `.mcp.json` (target project)

**File:** `{target_project}/.mcp.json` (created or updated)

The simplest integration path: register context-mode as an MCP server in the target project's `.mcp.json`. Since swarm agents inherit MCP servers from the project directory (via `cwd=str(self._config.target.resolved_path)` in `_spawn_claude_session` at `controller.py:756`), all agents automatically get access.

```json
{
  "mcpServers": {
    "context-mode": {
      "command": "npx",
      "args": ["-y", "context-mode"]
    }
  }
}
```

**Alternative (global):** Add to `~/.claude.json` under `mcpServers` for all projects. However, project-scoped is safer for evaluation.

### 2.3 Config: Add `ContextModeConfig` dataclass

**File:** `src/autodev/config.py` (near line 472, after `MCPConfig`)

Add a new config dataclass and wire it into `MissionConfig`:

```python
@dataclass
class ContextModeConfig:
	"""context-mode MCP server settings for context window optimization."""

	enabled: bool = False  # opt-in during evaluation
	scope: str = "project"  # "project" (.mcp.json) or "global" (~/.claude.json)
	install_command: str = "npx -y context-mode"  # command to start the server
	auto_register: bool = True  # auto-add to .mcp.json on swarm start
```

Add to `MissionConfig` (near line 534):

```python
context_mode: ContextModeConfig = field(default_factory=ContextModeConfig)
```

Add TOML parser function `_build_context_mode()` (near line 1254, after `_build_mcp()`):

```python
def _build_context_mode(data: dict[str, Any]) -> ContextModeConfig:
	cm = ContextModeConfig()
	if "enabled" in data:
		cm.enabled = bool(data["enabled"])
	if "scope" in data:
		cm.scope = str(data["scope"])
	if "install_command" in data:
		cm.install_command = str(data["install_command"])
	if "auto_register" in data:
		cm.auto_register = bool(data["auto_register"])
	return cm
```

Wire into `_build_mission_config()` where other `_build_*` calls happen:

```python
if "context_mode" in data:
	config.context_mode = _build_context_mode(data["context_mode"])
```

**TOML usage:**

```toml
[context_mode]
enabled = true
scope = "project"
```

### 2.4 Auto-registration in SwarmController

**File:** `src/autodev/swarm/controller.py` (new method, call from `run()`)

Add a method to register context-mode into `.mcp.json` at swarm startup, reusing the pattern from `_handle_register_mcp` (line 499):

```python
async def _ensure_context_mode(self) -> None:
	"""Register context-mode MCP server if enabled and not already present."""
	cm = self._config.context_mode
	if not cm.enabled or not cm.auto_register:
		return

	if cm.scope == "global":
		mcp_path = Path.home() / ".claude.json"
	else:
		mcp_path = Path(self._config.target.resolved_path) / ".mcp.json"

	config = {}
	if mcp_path.exists():
		try:
			config = json.loads(mcp_path.read_text())
		except (json.JSONDecodeError, OSError):
			pass

	servers = config.get("mcpServers", {})
	if "context-mode" in servers:
		logger.debug("context-mode already registered in %s", mcp_path)
		return

	# Parse install_command into command + args
	parts = cm.install_command.split()
	command = parts[0]
	args = parts[1:] if len(parts) > 1 else []

	servers["context-mode"] = {
		"type": "stdio",
		"command": command,
		"args": args,
	}
	config["mcpServers"] = servers
	mcp_path.write_text(json.dumps(config, indent=2))
	logger.info("Registered context-mode MCP server in %s", mcp_path)
```

Call `_ensure_context_mode()` in the swarm's `run()` method, after team directory init but before the first planning cycle.

### 2.5 Worker Prompt Guidance

**File:** `src/autodev/swarm/worker_prompt.py`

When context-mode is enabled, append routing guidance to worker prompts so agents prefer sandbox tools for data-heavy operations:

```python
_CONTEXT_MODE_INSTRUCTIONS = """
## Context Efficiency (context-mode)

You have access to context-mode sandbox tools. Use them to keep raw data OUT of your context window:

- **ctx_execute**: Run shell commands, scripts, or data processing in a sandbox. Only stdout enters your context. Use this instead of raw Bash for commands that produce large output (git log, test suites, log files, API responses).
- **ctx_batch_execute**: Chain multiple commands in one tool call.
- **ctx_fetch_and_index**: Fetch URLs and index into knowledge base (instead of raw WebFetch for large pages).
- **ctx_search**: Search previously indexed content by keywords.

Rules:
1. If a command might produce >5 KB of output, use ctx_execute instead of Bash.
2. For fetching documentation or web content, prefer ctx_fetch_and_index + ctx_search.
3. For reading large files, use ctx_execute with a script that extracts only what you need.
4. Short, targeted commands (cd, mkdir, git status, echo) can still use Bash directly.
"""
```

In the worker prompt builder function, conditionally include this block when `config.context_mode.enabled` is True.

### 2.6 Planner Prompt Awareness

**File:** `src/autodev/swarm/prompts.py`

Add a brief note to the planner system prompt when context-mode is enabled, so the planner understands agents have extended effective context:

```python
# In the system prompt builder, conditionally:
if config.context_mode.enabled:
	# Append after the existing agent capability description
	prompt += "\n\nAgents have context-mode enabled (sandbox tools for 98% context reduction). They can handle longer tasks and more tool calls before context degradation."
```

### 2.7 MCP Registry -- No Changes

**File:** `src/autodev/mcp_registry.py`

No changes needed. The `MCPToolRegistry` class is for *synthesized* tools (dynamically generated scripts with quality gating). context-mode is a static, external MCP server -- it doesn't go through the quality-gating pipeline.

The `scan_capabilities()` function in `swarm/capabilities.py` (line 214) already reads from `project_path / ".mcp.json"`, so it will automatically discover context-mode once registered. No code changes needed.

### 2.8 Environment Passthrough -- No Changes

**File:** `src/autodev/config.py`

context-mode's sandbox inherits environment variables from the Claude Code subprocess, which already gets a filtered env via `claude_subprocess_env()` (line 1148). The `_ENV_DENYLIST` strips sensitive tokens (`GH_TOKEN`, `AWS_SECRET_ACCESS_KEY`, etc.) -- this is correct behavior. context-mode's credential passthrough is limited to what's in the subprocess env.

**Limitation (acceptable):** Authenticated CLI tools (`gh`, `aws`) won't work inside context-mode's sandbox because their tokens are stripped. This is fine for initial rollout -- agents have native Bash access for those commands. If needed later, add a `context_mode_extra_env` list to `ContextModeConfig`.

### 2.9 CLI: Add context-mode management command

**File:** `src/autodev/cli.py`

Add a subcommand for context-mode lifecycle:

- `autodev context-mode install` -- Run `npm install -g context-mode`, check exit code
- `autodev context-mode register` -- Write/update `.mcp.json` with the context-mode entry
- `autodev context-mode status` -- Check if installed (`which context-mode`) and registered (`.mcp.json` contains entry)
- `autodev context-mode unregister` -- Remove `context-mode` key from `.mcp.json`

---

## 3. Testing Requirements

### 3.1 Unit Tests

**File:** `tests/test_context_mode.py`

| Test | Description |
|------|-------------|
| `test_context_mode_config_defaults` | `ContextModeConfig()` has `enabled=False`, `scope="project"`, correct `install_command` |
| `test_context_mode_config_from_toml` | Parse `[context_mode]` TOML section into `ContextModeConfig` with all fields |
| `test_context_mode_config_missing_section` | Missing `[context_mode]` in TOML uses defaults (`enabled=False`) |
| `test_ensure_context_mode_creates_mcp_json` | When enabled + auto_register, creates `.mcp.json` with context-mode entry in a tmp dir |
| `test_ensure_context_mode_updates_existing` | When `.mcp.json` exists with other servers, adds context-mode without clobbering |
| `test_ensure_context_mode_skips_if_present` | When context-mode already in `.mcp.json`, no-op (idempotent) |
| `test_ensure_context_mode_disabled` | When `enabled=False`, does not touch `.mcp.json` |
| `test_ensure_context_mode_global_scope` | When `scope="global"`, writes to `~/.claude.json` (use `tmp_path` + monkeypatch `Path.home()`) |
| `test_worker_prompt_includes_context_mode` | When enabled, worker prompt contains `ctx_execute` guidance |
| `test_worker_prompt_excludes_context_mode` | When disabled, worker prompt has no context-mode references |
| `test_planner_prompt_includes_context_mode` | When enabled, planner system prompt mentions context-mode |

### 3.2 Integration Test

**File:** `tests/test_context_mode_integration.py` (marked with `@pytest.mark.skipif` if context-mode not installed)

| Test | Description |
|------|-------------|
| `test_context_mode_binary_available` | Verify `npx -y context-mode --help` exits 0 |
| `test_mcp_json_roundtrip` | Write `.mcp.json`, read back, verify context-mode entry structure |
| `test_scan_capabilities_discovers_context_mode` | Register in `.mcp.json`, run `scan_capabilities()`, verify `MCPInfo(name="context-mode")` in result |

### 3.3 Regression Verification

```bash
.venv/bin/python -m pytest -q && .venv/bin/ruff check src/ tests/ && .venv/bin/bandit -r src/ -lll -q
```

Key areas to verify don't break:
- `test_config.py` -- all existing TOML parsing tests (new section must not interfere)
- `test_build_claude_cmd*` -- MCP flag handling unchanged
- Swarm controller tests -- agent spawning unchanged

---

## 4. Risk Assessment

### Low Risk

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| context-mode server crashes | Low | Low | Claude Code handles MCP server failures gracefully (tools become unavailable, agent continues with native tools). `--strict-mcp-config` surfaces errors early. |
| Agents ignore routing guidance | Medium | Low | Expected -- prompt-based routing achieves ~60% compliance without hooks (per context-mode docs). Even partial adoption reduces context consumption. Hooks are a future enhancement. |
| npm dependency adds attack surface | Low | Medium | Open source (Elastic-2.0), 82 published versions, active maintenance. Dependencies are minimal (MCP SDK, better-sqlite3, zod). Pin to known-good version. |
| FTS5 knowledge base conflicts | Very Low | Low | context-mode uses its own SQLite in `~/.context-mode/` -- completely isolated from `autodev.db`. |

### Medium Risk

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Hook conflict with stream-json parsing | Medium | Medium | **Do NOT install as Claude Code plugin** (which adds PreToolUse/PostToolUse hooks). Use MCP-only install (`npx -y context-mode`) to get the 6 sandbox tools without hooks. Plugin hooks could interfere with `_stream_agent_output()` in controller.py. |
| `npx -y` cold-start latency | Medium | Low | First invocation downloads ~1.7 MB. For production, recommend global install to eliminate cold-start. |
| Credential passthrough limitations | Medium | Low | `_ENV_DENYLIST` strips `GH_TOKEN`/`GITHUB_TOKEN`. context-mode sandbox can't use authenticated `gh`. Acceptable for initial rollout. |

### Not Risks (Addressed by Design)

- **Data leaving the machine**: Fully local. No telemetry, no cloud.
- **Breaking existing agents**: `enabled=False` by default. Opt-in via TOML.
- **License**: Elastic-2.0 allows use as a tool; prohibits offering as managed service (irrelevant to autodev).

---

## 5. Rollout Plan

| Phase | Description | Code Changes |
|-------|-------------|-------------|
| **1. Manual Evaluation** | Install globally, add to `.mcp.json` manually, run a swarm session, observe agent behavior and context savings | None |
| **2. Config Integration** | Implement this spec: `ContextModeConfig`, auto-registration, prompt guidance, CLI, tests | This spec |
| **3. Future: Hook-based routing** | Install as plugin for ~98% compliance (vs ~60% prompt-only). Requires testing hook + stream-json compatibility | Out of scope |
| **4. Future: Metrics** | Track context savings per agent in `tool_calls` DB table | Out of scope |
| **5. Future: KB seeding** | Pre-index project docs into FTS5 before agent start | Out of scope |

---

## 6. File Change Summary

| File | Change Type | Description |
|------|------------|-------------|
| `src/autodev/config.py` | Modify | Add `ContextModeConfig`, `_build_context_mode()`, wire into `MissionConfig` + TOML parser |
| `src/autodev/swarm/controller.py` | Modify | Add `_ensure_context_mode()`, call from `run()` |
| `src/autodev/swarm/worker_prompt.py` | Modify | Add context-mode routing instructions, include conditionally |
| `src/autodev/swarm/prompts.py` | Modify | Add context-mode awareness to planner system prompt |
| `src/autodev/cli.py` | Modify | Add `context-mode` subcommand |
| `tests/test_context_mode.py` | Create | Unit tests for config, auto-registration, prompt injection |
| `tests/test_context_mode_integration.py` | Create | Integration tests (binary, capabilities scan) |
| `autodev.toml` | Modify | Add `[context_mode]` section (disabled by default) |
| `src/autodev/mcp_registry.py` | No change | Not applicable -- context-mode is external, not synthesized |