# Claude Code CLI Audit for autodev Integration

**Date**: 2026-03-17
**CLI Version**: 2.1.77
**Agent**: cli-auditor
**Purpose**: Identify new/unused CLI flags that could benefit autodev subprocess spawning

---

## 1. Full CLI Help Output

```
Usage: claude [options] [command] [prompt]

Claude Code - starts an interactive session by default, use -p/--print for
non-interactive output

Arguments:
  prompt                                            Your prompt

Options:
  --add-dir <directories...>                        Additional directories to allow tool access to
  --agent <agent>                                   Agent for the current session. Overrides the 'agent' setting.
  --agents <json>                                   JSON object defining custom agents (e.g. '{"reviewer": {"description": "Reviews code", "prompt": "You are a code reviewer"}}')
  --allow-dangerously-skip-permissions              Enable bypassing all permission checks as an option, without it being enabled by default. Recommended only for sandboxes with no internet access.
  --allowedTools, --allowed-tools <tools...>        Comma or space-separated list of tool names to allow (e.g. "Bash(git:*) Edit")
  --append-system-prompt <prompt>                   Append a system prompt to the default system prompt
  --betas <betas...>                                Beta headers to include in API requests (API key users only)
  --brief                                           Enable SendUserMessage tool for agent-to-user communication
  --chrome                                          Enable Claude in Chrome integration
  -c, --continue                                    Continue the most recent conversation in the current directory
  --dangerously-skip-permissions                    Bypass all permission checks. Recommended only for sandboxes with no internet access.
  -d, --debug [filter]                              Enable debug mode with optional category filtering (e.g., "api,hooks" or "!1p,!file")
  --debug-file <path>                               Write debug logs to a specific file path (implicitly enables debug mode)
  --disable-slash-commands                          Disable all skills
  --disallowedTools, --disallowed-tools <tools...>  Comma or space-separated list of tool names to deny (e.g. "Bash(git:*) Edit")
  --effort <level>                                  Effort level for the current session (low, medium, high, max)
  --fallback-model <model>                          Enable automatic fallback to specified model when default model is overloaded (only works with --print)
  --file <specs...>                                 File resources to download at startup. Format: file_id:relative_path (e.g., --file file_abc:doc.txt file_def:img.png)
  --fork-session                                    When resuming, create a new session ID instead of reusing the original (use with --resume or --continue)
  --from-pr [value]                                 Resume a session linked to a PR by PR number/URL, or open interactive picker with optional search term
  -h, --help                                        Display help for command
  --ide                                             Automatically connect to IDE on startup if exactly one valid IDE is available
  --include-partial-messages                        Include partial message chunks as they arrive (only works with --print and --output-format=stream-json)
  --input-format <format>                           Input format (only works with --print): "text" (default), or "stream-json" (realtime streaming input) (choices: "text", "stream-json")
  --json-schema <schema>                            JSON Schema for structured output validation. Example: {"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
  --max-budget-usd <amount>                         Maximum dollar amount to spend on API calls (only works with --print)
  --mcp-config <configs...>                         Load MCP servers from JSON files or strings (space-separated)
  --mcp-debug                                       [DEPRECATED. Use --debug instead] Enable MCP debug mode (shows MCP server errors)
  --model <model>                                   Model for the current session. Provide an alias for the latest model (e.g. 'sonnet' or 'opus') or a model's full name (e.g. 'claude-sonnet-4-6').
  -n, --name <name>                                 Set a display name for this session (shown in /resume and terminal title)
  --no-chrome                                       Disable Claude in Chrome integration
  --no-session-persistence                          Disable session persistence - sessions will not be saved to disk and cannot be resumed (only works with --print)
  --output-format <format>                          Output format (only works with --print): "text" (default), "json" (single result), or "stream-json" (realtime streaming) (choices: "text", "json", "stream-json")
  --permission-mode <mode>                          Permission mode to use for the session (choices: "acceptEdits", "bypassPermissions", "default", "dontAsk", "plan", "auto")
  --plugin-dir <path>                               Load plugins from a directory for this session only (repeatable: --plugin-dir A --plugin-dir B) (default: [])
  -p, --print                                       Print response and exit (useful for pipes). Note: The workspace trust dialog is skipped when Claude is run with the -p mode. Only use this flag in directories you trust.
  --replay-user-messages                            Re-emit user messages from stdin back on stdout for acknowledgment (only works with --input-format=stream-json and --output-format=stream-json)
  -r, --resume [value]                              Resume a conversation by session ID, or open interactive picker with optional search term
  --session-id <uuid>                               Use a specific session ID for the conversation (must be a valid UUID)
  --setting-sources <sources>                       Comma-separated list of setting sources to load (user, project, local).
  --settings <file-or-json>                         Path to a settings JSON file or a JSON string to load additional settings from
  --strict-mcp-config                               Only use MCP servers from --mcp-config, ignoring all other MCP configurations
  --system-prompt <prompt>                          System prompt to use for the session
  --tmux                                            Create a tmux session for the worktree (requires --worktree). Uses iTerm2 native panes when available; use --tmux=classic for traditional tmux.
  --tools <tools...>                                Specify the list of available tools from the built-in set. Use "" to disable all tools, "default" to use all tools, or specify tool names (e.g. "Bash,Edit,Read").
  --verbose                                         Override verbose mode setting from config
  -v, --version                                     Output the version number
  -w, --worktree [name]                             Create a new git worktree for this session (optionally specify a name)

Hidden/undocumented flags (confirmed working):
  --max-turns <turns>                               Maximum number of agentic turns (used in autodev, not in --help output)

Commands:
  agents [options]                                  List configured agents
  auth                                              Manage authentication
  doctor                                            Check the health of your Claude Code auto-updater
  install [options] [target]                        Install Claude Code native build. Use [target] to specify version (stable, latest, or specific version)
  mcp                                               Configure and manage MCP servers
  plugin|plugins                                    Manage Claude Code plugins
  setup-token                                       Set up a long-lived authentication token (requires Claude subscription)
  update|upgrade                                    Check for updates and install if available
```

**Subcommand: `cowork`** -- Does NOT exist (no such subcommand in v2.1.77)

---

## 2. Flags Currently Used by autodev

### build_claude_cmd() (config.py:1192-1245)

The central command builder accepts these parameters and maps them to CLI flags:

| Parameter | CLI Flag | Used By |
|-----------|----------|---------|
| `-p` | `--print` | Always (hardcoded) |
| `output_format` | `--output-format` | Always (default "text") |
| `model` | `--model` | Always (required param) |
| `budget` | `--max-budget-usd` | When budget is not None |
| `max_turns` | `--max-turns` | When max_turns is not None |
| `permission_mode="auto"` or `"bypassPermissions"` | `--dangerously-skip-permissions` | Swarm agents, workers |
| `permission_mode` (other) | `--permission-mode` | When set to other modes |
| `session_id` | `--session-id` | When session_id is provided |
| `mcp.config_path` | `--mcp-config` + `--strict-mcp-config` | When MCP is enabled in config |
| `allowed_tools` | `--allowedTools` | When allowed_tools list is provided |
| `setting_sources` | `--setting-sources` | When set and not inheriting globals |
| `json_schema` | `--json-schema` | When json_schema is provided |
| `resume_session` | `--resume` | When resuming a session |
| (prompt) | (positional arg) | When prompt is provided |

### Per-spawn-site flags summary:

| File:Function | model | budget | max_turns | permission_mode | output_format | Other |
|---|---|---|---|---|---|---|
| `swarm/controller.py:_spawn_agent()` | scheduler.model | - | 200 | auto | stream-json | setting_sources conditional |
| `swarm/planner.py:_call_llm()` | planner_model | - | - | - | text | - |
| `recursive_planner.py:_invoke_planner()` | planner_model | planner budget | - | - | text | allowed_tools |
| `worker.py:_spawn_session()` | worker_model | per-session budget | - | bypassPermissions | text | setting_sources="project" |
| `continuous_controller.py:_run_evaluator()` | ev.model | ev.budget | ev.max_turns | - | text | - |
| `continuous_controller.py:_run_fixup()` | model | - | 5 | bypassPermissions | text | resume_session possible |
| `green_branch.py` fixup calls | model | fixup budget | 1 or 5 | bypassPermissions | text | resume_session |
| `critic_agent.py` | critic_model | budget | - | - | text | - |
| `diff_reviewer.py` | review_model | - | 1 | - | text | - |

---

## 3. Flag Gap Analysis -- Unused Flags

### Directly Applicable (can use today in autodev subprocess spawning)

| Flag | Description | Default | Recommendation |
|---|---|---|---|
| `--add-dir <directories...>` | Additional directories to allow tool access to | none | **HIGH PRIORITY**: Workers often need access to sibling repos or shared libs. Currently they can only access the cwd. Adding `--add-dir` to `build_claude_cmd()` would let workers access test fixtures, shared packages, or reference code in other directories. |
| `--agent <agent>` | Use a named agent definition for the session | none | **HIGH PRIORITY**: Instead of massive system prompts in the positional arg, define worker/planner/reviewer agent definitions in settings and reference them by name. Cleaner separation of concerns. |
| `--agents <json>` | JSON object defining custom agents inline | none | **HIGH PRIORITY**: Define agent personas dynamically at spawn time (e.g., `{"worker": {"description": "...", "prompt": "..."}}` + `--agent worker`). Would replace current prompt template system with native Claude Code agent definitions. |
| `--append-system-prompt <prompt>` | Append to (not replace) the default system prompt | none | **HIGH PRIORITY**: Currently prompts are passed as positional args which REPLACE the system prompt. Using `--append-system-prompt` preserves Claude's built-in instructions while adding autodev-specific directives. This could dramatically improve worker quality. |
| `--effort <level>` | Effort level: low, medium, high, max | (not set) | **HIGH PRIORITY**: Use `--effort low` for planners (structured output, no need for deep reasoning), `--effort high` for workers (need thorough implementation), `--effort max` for complex refactoring tasks. Cost/quality tradeoff control. |
| `--fallback-model <model>` | Auto-fallback when primary model is overloaded | none | **MEDIUM PRIORITY**: When running many parallel agents, rate limits and overload are common. Setting `--fallback-model sonnet` when primary is opus would prevent agent failures from API overload. Only works with `--print`. |
| `--no-session-persistence` | Don't save sessions to disk | false | **HIGH PRIORITY**: Swarm agents spawn hundreds of sessions that clutter `~/.claude/projects/`. Using `--no-session-persistence` for workers and planners would prevent disk bloat and improve performance. |
| `--name <name>` | Display name for the session | none | **MEDIUM PRIORITY**: Set session names like "autodev-worker-{task_id}" or "autodev-planner-{run_id}" for easier identification when debugging with `claude --resume`. |
| `--system-prompt <prompt>` | Override the entire system prompt | none | **MEDIUM PRIORITY**: Alternative to positional arg for system prompt. More explicit and avoids ambiguity between "system prompt" and "user message". Currently the positional arg is used which works but `--system-prompt` is more semantic. |
| `--disallowedTools <tools...>` | Deny specific tools | none | **MEDIUM PRIORITY**: Complement to `--allowedTools`. Could deny dangerous tools (e.g., deny `WebFetch` for implementation workers, deny `Edit` for review-only agents). More surgical than allowlists. |
| `--tools <tools...>` | Specify exact built-in tool set | "default" (all) | **MEDIUM PRIORITY**: Strip down tools for specialized agents. Planners only need `Bash,Read,Grep,Glob`. Reviewers only need `Read,Grep,Glob`. Reduces prompt noise and prevents off-task behavior. |
| `--disable-slash-commands` | Disable all skills | false | **MEDIUM PRIORITY**: Workers shouldn't be invoking skills (they have their own prompt). Disabling skills reduces distraction and potential interference from injected skill content. |
| `--debug-file <path>` | Write debug logs to file | none | **LOW PRIORITY**: Could write per-agent debug logs to `.autodev-traces/` for post-mortem analysis. Currently only stdout/stderr is captured. |
| `--input-format stream-json` | Streaming JSON input | "text" | **LOW PRIORITY**: Enable bidirectional streaming communication between autodev and agents. Currently autodev sends prompt via positional arg or stdin text. Stream-json input would allow sending additional context mid-session. |
| `--include-partial-messages` | Include partial message chunks | false | **LOW PRIORITY**: With `--output-format stream-json`, get partial messages as they arrive. Could improve TUI responsiveness by showing agent thoughts in real-time. |
| `--replay-user-messages` | Re-emit user messages on stdout | false | **LOW PRIORITY**: With stream-json I/O, acknowledge received messages. Useful for bidirectional communication protocol. |
| `--fork-session` | Create new session ID when resuming | false | **LOW PRIORITY**: When resuming sessions for fixup (continuous_controller), fork to preserve original session history. Currently reuses the session ID which overwrites history. |

### Potentially Applicable (needs wrapper or config support)

| Flag | Description | Applicability |
|---|---|---|
| `--worktree [name]` | Create git worktree for session | Currently autodev has its own workspace pool (backends/local.py). Could delegate worktree management to Claude Code native worktrees instead of custom pool clones. Would need rearchitecting `LocalBackend`. |
| `--tmux` | Create tmux session for worktree | Interactive debugging -- could spawn visible tmux panes for each agent during development/debugging mode. Not for production swarm runs. |
| `--settings <file-or-json>` | Load additional settings from file/JSON | Could inject per-agent settings (model preferences, tool permissions) via JSON string. Alternative to multiple CLI flags. |
| `--plugin-dir <path>` | Load plugins from directory | Could load autodev-specific plugins for workers (e.g., custom tools, MCP bridges). Requires building Claude Code plugins. |
| `--from-pr [value]` | Resume session from PR | Could be used when workers are fixing PR review feedback. Need to integrate with GitHub PR workflow. |
| `--brief` | Enable SendUserMessage for agent-to-user comms | Enables structured agent-to-user communication. Could replace the inbox-based communication with native Claude Code messaging. Requires understanding how SendUserMessage works with `--print` mode. |
| `--chrome` / `--no-chrome` | Chrome integration toggle | If browser testing is needed for workers. Currently handled via MCP browser-use server. |
| `--betas <betas...>` | Beta API headers | Only relevant for API key users with experimental features. |

### Not Applicable

| Flag | Reason |
|---|---|
| `-c, --continue` | Interactive only; autodev uses `--resume` with session IDs |
| `-r, --resume` (interactive picker) | Already used via `resume_session` param in build_claude_cmd |
| `--ide` | IDE integration, not relevant for subprocess agents |
| `--file <specs...>` | File resource downloading -- not relevant for local agents |
| `--mcp-debug` | Deprecated, use `--debug` instead |
| `--verbose` | Would clutter subprocess output; debug-file is better |
| `-v, --version` | Informational only |
| `-h, --help` | Informational only |
| `--allow-dangerously-skip-permissions` | Enables the option but doesn't activate it; autodev already uses `--dangerously-skip-permissions` directly |

---

## 4. Capability Mapping Table

| File | Function | Spawn Purpose | Current Flags | Missing High-Value Flags |
|---|---|---|---|---|
| `config.py` | `build_claude_cmd()` | Central builder | -p, --output-format, --model, --max-budget-usd, --max-turns, --dangerously-skip-permissions / --permission-mode, --session-id, --mcp-config, --strict-mcp-config, --allowedTools, --setting-sources, --json-schema, --resume | --append-system-prompt, --effort, --no-session-persistence, --agent/--agents, --fallback-model, --name, --add-dir, --disallowedTools, --tools, --disable-slash-commands |
| `swarm/controller.py` | `_spawn_agent()` | Worker agents | model, permission=auto, max_turns=200, output_format=stream-json, setting_sources conditional | --effort, --no-session-persistence, --name, --fallback-model, --add-dir, --append-system-prompt, --disable-slash-commands |
| `swarm/planner.py` | `_call_llm()` | Planner reasoning | model, output_format=text | --effort, --no-session-persistence, --tools (restrict to read-only), --disable-slash-commands |
| `recursive_planner.py` | `_invoke_planner()` | Plan decomposition | model, budget, allowed_tools | --effort low, --no-session-persistence, --tools, --disable-slash-commands |
| `worker.py` | `_spawn_session()` | Task execution | model, budget, permission=bypassPermissions, setting_sources=project | --effort, --no-session-persistence, --fallback-model, --add-dir, --name |
| `continuous_controller.py` | `_run_evaluator()` | Evaluation | model, budget, max_turns | --effort low, --no-session-persistence, --tools (read-only) |
| `continuous_controller.py` | `_run_fixup()` | Post-merge fixes | model, max_turns=5, permission=bypass, resume_session | --fork-session (with resume), --effort high |
| `green_branch.py` | fixup calls | Merge conflict resolution | model, budget, max_turns=1/5, permission=bypass, resume_session | --fork-session, --effort high |
| `critic_agent.py` | `_call_llm()` | Feasibility review | model, budget | --effort medium, --no-session-persistence, --tools (read-only), --disable-slash-commands |
| `diff_reviewer.py` | `review()` | Code review | model, max_turns=1 | --effort medium, --no-session-persistence, --tools (read-only) |

---

## 5. Key Recommendations

### Tier 1: Immediate Integration (High Impact, Low Effort)

1. **`--no-session-persistence`**: Add to `build_claude_cmd()` as a boolean parameter, default True for all autodev spawns. Prevents hundreds of stale sessions from accumulating in `~/.claude/projects/`. Single-line change.

2. **`--effort <level>`**: Add to `build_claude_cmd()` as an optional parameter. Map:
   - Planners/critics: `--effort medium` (structured output, moderate reasoning)
   - Workers: `--effort high` (thorough implementation)
   - Evaluators/reviewers: `--effort medium`
   - Quick fixups: `--effort low`
   - Complex refactors: `--effort max`

3. **`--append-system-prompt`**: Use instead of positional arg for autodev directives. The positional arg currently works but `--append-system-prompt` ADDS to Claude's built-in system prompt rather than acting as user input. This preserves Claude's native capabilities (tool use patterns, safety) while layering autodev instructions on top.

4. **`--fallback-model`**: Add as optional parameter. When spawning many parallel workers with opus, set `--fallback-model sonnet` to handle rate limit / overload gracefully instead of hard-failing.

5. **`--name <name>`**: Set meaningful session names for debugging. Format: `autodev-{role}-{task_id[:8]}`.

### Tier 2: Medium-Term Integration (Higher Impact, Moderate Effort)

6. **`--disable-slash-commands`**: Add for all non-interactive spawns. Skills/slash-commands are designed for human interaction and can interfere with agent operations (injecting unexpected context, consuming tokens on irrelevant skill matches).

7. **`--tools`**: Restrict tool sets per agent role:
   - Planners: `"Bash,Read,Grep,Glob,WebSearch,WebFetch"` (research + read, no edit)
   - Workers: `"default"` (all tools)
   - Reviewers: `"Read,Grep,Glob,Bash"` (read-only + verification commands)
   - Critics: `"Read,Grep,Glob,WebSearch"` (analysis only)

8. **`--add-dir`**: Allow workers to access additional directories (shared packages, test fixtures). Add as optional list parameter to `build_claude_cmd()`.

9. **`--disallowedTools`**: Surgically deny dangerous tools per role rather than maintaining allowlists. E.g., deny `WebFetch` for implementation workers to prevent distraction.

10. **`--agent` / `--agents`**: Define agent personas as named agents instead of massive prompt strings. Would require creating agent definition JSON and passing via `--agents` or configuring in settings and referencing via `--agent`.

### Tier 3: Architectural Opportunities

11. **Native worktrees (`--worktree`)**: Replace custom `LocalBackend` workspace pool with Claude Code native worktree management. Each agent gets isolated worktree. Major refactor but eliminates pool corruption issues.

12. **Bidirectional streaming** (`--input-format stream-json` + `--output-format stream-json`): Replace current "fire and forget" spawning with interactive sessions where autodev can inject additional context mid-execution. Enables dynamic task adjustment.

13. **`--settings <json>`**: Pass per-agent settings as JSON string, consolidating multiple flags into one structured config object.

---

## 6. Notable Observations

### No `--mode`, `--research`, or `--cowork` flags exist

- There is **no `--cowork` subcommand** or multi-agent coordination mode built into the CLI.
- There is **no `--research` flag** for a research-only mode.
- There is **no `--mode` flag** -- the closest equivalent is `--permission-mode` which controls tool permissions (not agent behavior mode).
- The `--permission-mode plan` mode exists but it's about permission approval workflow, not a "planning mode" for the agent.
- The `--agent` flag combined with `--agents` JSON is the closest thing to role-based agent modes.

### --max-turns is undocumented

The `--max-turns` flag works but does not appear in `--help` output. It's a hidden/undocumented flag. autodev relies on it heavily (200 turns for swarm workers, 1-10 for quick tasks). This is a stability risk -- undocumented flags can be removed without notice.

### Permission mode "auto" maps to --dangerously-skip-permissions

In `build_claude_cmd()`, both `permission_mode="auto"` and `permission_mode="bypassPermissions"` are mapped to `--dangerously-skip-permissions`. The native `--permission-mode auto` flag exists and may have different behavior (learning/approving permissions dynamically vs. bypassing all). This should be investigated -- `auto` mode might be safer than full bypass while still being non-interactive.

### Stream-json output is only used for swarm agents

Only `swarm/controller.py:_spawn_agent()` uses `output_format="stream-json"`. All other spawns use `"text"`. The stream-json format enables real-time output monitoring but the text format is simpler for batch processing.

### --json-schema is supported but rarely used

The `json_schema` parameter exists in `build_claude_cmd()` but grep shows no callers actually pass it. This could be used for planners to enforce structured plan output (ensuring the planner returns valid JSON matching a schema rather than relying on regex parsing of `<!-- PLAN -->` blocks).

---

## 7. Version Note

CLI version 2.1.77 is outdated (latest is 50.32.5 per the system notification). Many of these flags may have been recently added. The `--effort`, `--agent`, `--agents`, and `--worktree` flags in particular appear to be newer additions. Upgrading may reveal additional capabilities not present in older versions.
