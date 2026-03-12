# Capability Awareness and Auto-Update Pipeline

## Problem Statement

Autodev workers are spawned as bare Claude Code subprocesses with no awareness of the rich skill/hook/plugin ecosystem available on the host system. The planner can only create skills -- it can't create hooks, register MCP servers, or define new agent types. There's no mechanism to discover and integrate new Claude Code features automatically, so the project falls behind as Claude Code evolves.

## Goals

1. Workers inherit and leverage all installed Claude Code capabilities (skills, hooks, plugins, MCP servers)
2. Planner is aware of available capabilities and can instruct workers to use specific skills
3. Planner can dynamically create hooks, MCP registrations, and agent definitions (not just skills)
4. Intel module monitors Claude Code releases and generates improvement proposals
5. Auto-update pipeline feeds low-risk proposals as swarm missions, gates high-risk ones via Telegram
6. Telegram notifications work end-to-end

## Non-Goals

- Worktree isolation for agents (soft coordination via inboxes is sufficient)
- Shipping capabilities as a separate plugin/package
- Creating a new abstraction layer/module for capabilities

## Design

### 1. Capability Manifest

At swarm initialization (`controller.initialize()`), scan for all available capabilities and build a manifest.

**Scanner function** in new file `src/autodev/swarm/capabilities.py`:

```python
@dataclass
class CapabilityManifest:
	skills: list[SkillInfo]       # name, description, path, invocation
	agents: list[AgentDefInfo]    # name, description, model, tools
	hooks: list[HookInfo]        # event, matcher, type
	mcp_servers: list[MCPInfo]    # name, type, tools exposed

def scan_capabilities(project_path: Path) -> CapabilityManifest:
	"""Scan all capability sources and build manifest."""
	# Scan: ~/.claude/skills/, .claude/skills/, plugin skills
	# Scan: ~/.claude/agents/, .claude/agents/
	# Scan: hooks from ~/.claude/settings.json, .claude/settings.json
	# Scan: MCP servers from ~/.claude.json, .mcp.json
```

**Integration points:**
- Called in `SwarmController.initialize()`
- Stored as `self._capabilities: CapabilityManifest`
- Rendered as `## Available Capabilities` section in `context.py render_for_planner()`
- Passed to `build_worker_prompt()` so workers know what skills they can invoke

### 2. Extended Decision Types

Add to `DecisionType` enum in `models.py`:

```python
class DecisionType(str, Enum):
	# ... existing ...
	CREATE_HOOK = "create_hook"
	REGISTER_MCP = "register_mcp"
	CREATE_AGENT_DEF = "create_agent_def"
	USE_SKILL = "use_skill"
```

**Handler implementations in `controller.py`:**

#### `create_hook`
```python
async def _handle_create_hook(self, payload: dict) -> dict:
	"""Install a hook into the project's .claude/settings.json."""
	# payload: {event, matcher, type, command/prompt, background}
	# Read .claude/settings.json, add hook entry, write back
	# Example: PreToolUse hook running bandit on edited .py files
```

#### `register_mcp`
```python
async def _handle_register_mcp(self, payload: dict) -> dict:
	"""Add an MCP server to .mcp.json."""
	# payload: {name, type, command/args/url, env, scope}
	# Read .mcp.json, add server entry, write back
```

#### `create_agent_def`
```python
async def _handle_create_agent_def(self, payload: dict) -> dict:
	"""Write a .claude/agents/<name>.md file."""
	# payload: {name, description, tools, disallowed_tools, model, system_prompt}
	# Write frontmatter + system prompt to .claude/agents/<name>.md
```

#### `use_skill`
```python
async def _handle_use_skill(self, payload: dict) -> dict:
	"""Instruct an active agent to invoke a skill via inbox message."""
	# payload: {agent_name, skill_name, args}
	# Write directive to agent's inbox: "invoke /<skill_name> <args>"
```

### 3. Planner Prompt Updates

Update `SYSTEM_PROMPT` in `prompts.py` to include:

- List of available decision types with payload schemas
- Capability manifest section showing what skills/hooks/agents/MCPs exist
- Guidance: "Use existing capabilities before creating new ones. Instruct workers to invoke specific skills when appropriate (e.g., /code-review after implementation, /verify-by-consensus for high-stakes changes)."

### 4. Intel Module: Claude Code Scanner

New file `src/autodev/intelligence/claude_code.py`:

```python
class ClaudeCodeScanner:
	"""Monitor Claude Code releases for new features."""

	REPO = "anthropics/claude-code"
	CHANGELOG_PATH = "CHANGELOG.md"

	async def scan(self) -> list[Finding]:
		"""Fetch releases since last check, parse for automation-relevant changes."""
		# 1. Get current installed version: subprocess claude --version
		# 2. Fetch GitHub releases via API: GET /repos/{REPO}/releases
		# 3. Filter releases newer than installed version
		# 4. Parse release notes for: new tools, hooks, skills, CLI flags, breaking changes
		# 5. Score relevance based on automation keywords
		# 6. Return list[Finding]
```

**Evaluator extension** in `evaluator.py`:
- New category: `claude_code_release`
- Risk classification: `low` (new skill, docs, test coverage), `high` (architecture, spawn command, config schema)
- Target module mapping for autodev-specific integration points

### 5. Auto-Update Pipeline

New file `src/autodev/auto_update.py`:

```python
class AutoUpdatePipeline:
	"""Bridge intel proposals to swarm missions."""

	def __init__(self, config, db):
		self._config = config
		self._db = db

	async def run(self):
		"""Full pipeline: scan -> evaluate -> propose -> feed."""
		# 1. Run intel scan (includes claude_code scanner)
		# 2. Filter proposals not already applied (check applied_proposals table)
		# 3. Classify risk level
		# 4. Low-risk: auto-generate objective, launch swarm
		# 5. High-risk: send to Telegram, wait for approval
		# 6. Track applied proposals in DB

	def _generate_objective(self, proposal: AdaptationProposal) -> str:
		"""Convert a proposal into a swarm mission objective."""

	def _is_already_applied(self, proposal_id: str) -> bool:
		"""Check if proposal was already run."""
```

**New DB table:** `applied_proposals` (id, proposal_id, finding_title, applied_at, mission_id, status)

**CLI command in `cli.py`:**
```python
# autodev auto-update
auto_update = sub.add_parser("auto-update", help="Scan for improvements and auto-apply")
auto_update.add_argument("--config", default=DEFAULT_CONFIG)
auto_update.add_argument("--dry-run", action="store_true", help="Show proposals without launching")
auto_update.add_argument("--approve-all", action="store_true", help="Skip approval for high-risk")
```

### 6. Telegram Fix

Debug and fix `notifier.py`:
- Verify env vars (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) are read correctly
- Test async httpx client against Bot API
- Add `request_approval()` method that sends a message and polls for reply
- End-to-end test: notification delivery + approval flow

### 7. Spawn Command Updates

In `config.py` `build_claude_cmd()`:
- Do not pass `--setting-sources` flags that would block global skill/plugin access
- When `inherit_global_mcps` is true (default), workers see everything

In `controller.py` `_build_worker_prompt()`:
- Include relevant capability manifest entries in worker context
- When planner uses `use_skill`, the skill invocation gets injected into the worker's inbox as a directive

## File Changes Summary

| File | Change |
|------|--------|
| New: `src/autodev/swarm/capabilities.py` | Capability scanner and manifest dataclass |
| New: `src/autodev/intelligence/claude_code.py` | Claude Code release scanner |
| New: `src/autodev/auto_update.py` | Proposal-to-mission pipeline |
| `src/autodev/swarm/models.py` | Add 4 new DecisionType values |
| `src/autodev/swarm/controller.py` | Add 4 handler methods, call capability scan in initialize() |
| `src/autodev/swarm/context.py` | Render capability manifest in planner state |
| `src/autodev/swarm/prompts.py` | Update system prompt with new decision types + capability guidance |
| `src/autodev/swarm/worker_prompt.py` | Include capability info for workers |
| `src/autodev/intelligence/evaluator.py` | Add claude_code_release category + risk classification |
| `src/autodev/intelligence/scanner.py` | Register claude_code scanner |
| `src/autodev/notifier.py` | Debug/fix Telegram, add approval gate |
| `src/autodev/cli.py` | Add `autodev auto-update` command |
| `src/autodev/db.py` | Add applied_proposals table |

## Testing

- Unit tests for capability scanner (mock filesystem with skills/agents/hooks)
- Unit tests for new decision handlers
- Unit tests for claude_code scanner (mock GitHub API responses)
- Unit tests for auto-update pipeline (mock intel + DB)
- Integration test for Telegram notification delivery
- End-to-end: run auto-update in dry-run mode against real Claude Code releases
