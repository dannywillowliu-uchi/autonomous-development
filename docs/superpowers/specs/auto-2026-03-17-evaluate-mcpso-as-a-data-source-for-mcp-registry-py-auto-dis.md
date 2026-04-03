Here's the spec. I'll output it directly since the file write needs permission:

# Spec: mcpso Directory Integration for MCP Registry Auto-Discovery

**Type:** integration  
**Priority:** 2  
**Effort:** medium  
**Risk:** low  
**Target modules:** `mcp_registry.py`, `config.py`, new `mcpso_client.py`

---

## 1. Problem Statement

The existing `MCPToolRegistry` in `mcp_registry.py` only contains tools that were **synthesized by worker agents during missions** and promoted via `promote_to_mcp_registry()`. There is no mechanism for agents to discover **externally available MCP servers** that already exist in the ecosystem (filesystem tools, database connectors, web search, etc.).

The [chatmcp/mcpso](https://github.com/chatmcp/mcpso) project maintains a community-curated directory of MCP servers. Integrating this as a data source would let the planner and worker agents:

1. **Discover relevant MCP servers** for a given task (e.g., "I need a database tool" -> find the Postgres MCP server)
2. **Recommend MCP server installation** to workers via prompt injection, so they can use external tools they wouldn't otherwise know about
3. **Reduce redundant tool synthesis** -- if a high-quality MCP server already exists for a capability, agents should use it instead of re-inventing

This is distinct from the existing `tool_synthesis.py` flow (which creates ephemeral Python scripts) and the existing `MCPToolRegistry` (which stores promoted scripts in SQLite). The mcpso integration adds a **read-only external catalog** alongside the internal registry.

---

## 2. Architecture Overview

```
                          +---------------------+
                          |   mcpso (GitHub)     |
                          |  servers.json / API  |
                          +---------+-----------+
                                    | fetch + cache
                          +---------v-----------+
                          |   McpsoClient        |
                          | (mcpso_client.py)    |
                          |  - fetch_servers()   |
                          |  - search()          |
                          |  - local JSON cache  |
                          +---------+-----------+
                                    |
                          +---------v-----------+
                          |  MCPToolRegistry     |
                          | (mcp_registry.py)    |
                          |  + discover_external |
                          |  + suggest_servers() |
                          +---------+-----------+
                                    |
                    +---------------+---------------+
                    v               v               v
             Swarm Planner    Worker Prompts   Context Synthesizer
             (planner.py)    (worker_prompt.py) (context.py)
```

The integration is **layered**: a new `McpsoClient` handles all HTTP fetching and caching, the existing `MCPToolRegistry` gains a `discover_external()` method that queries the client, and consumers (planner, worker prompts, context synthesizer) call the registry's unified discover interface.

---

## 3. Changes Needed

### 3.1 New File: `src/autodev/mcpso_client.py`

A standalone HTTP client for fetching and caching the mcpso server directory.

**Data source strategy (ordered by preference):**

1. **GitHub Contents API** -- `GET https://api.github.com/repos/chatmcp/mcpso/contents/` to discover if the repo has a structured data file (e.g., `servers.json`, `data/servers.json`). If found, fetch and parse it.
2. **README.md parsing** -- If no structured file exists, fetch the raw README via `https://raw.githubusercontent.com/chatmcp/mcpso/main/README.md` and parse the markdown table/list structure that directories typically use (server name, description, GitHub URL, install command).
3. **Web API fallback** -- If mcpso exposes a REST API (e.g., at `https://mcp.so/api/servers`), prefer that over scraping.

The client should try strategy 1 first, fall back to 2, and support 3 via config.

```python
@dataclass
class McpsoServer:
	"""A single MCP server entry from the mcpso directory."""
	name: str
	description: str
	github_url: str = ""
	package_name: str = ""          # npm/pip package name
	install_command: str = ""       # e.g. "npx -y @modelcontextprotocol/server-filesystem"
	categories: list[str] = field(default_factory=list)
	capabilities: list[str] = field(default_factory=list)  # tools/resources/prompts
	stars: int = 0
	last_updated: str = ""

class McpsoClient:
	"""Fetches and caches the mcpso MCP server directory."""

	def __init__(self, config: McpsoConfig) -> None: ...

	async def fetch_servers(self, force_refresh: bool = False) -> list[McpsoServer]: ...
	def search(self, query: str, limit: int = 10) -> list[McpsoServer]: ...
	def get_by_name(self, name: str) -> McpsoServer | None: ...
	def cache_age_seconds(self) -> float: ...
```

**Caching:**
- Cache fetched data as a JSON file at `{cache_dir}/mcpso_servers.json` (default cache_dir: `~/.autodev/cache/`)
- Cache TTL configurable (default: 24 hours)
- `fetch_servers()` returns cached data if fresh, fetches if stale or `force_refresh=True`
- Cache file includes a `fetched_at` ISO timestamp for TTL checks

**Search implementation:**
- In-memory keyword search over `name`, `description`, `categories`, and `capabilities` fields
- Case-insensitive substring matching (same pattern as `db.search_tool_registry()`)
- Results sorted by star count descending as a quality proxy

**Error handling:**
- GitHub API rate limiting: respect `X-RateLimit-Remaining` header, log warning when low
- Network failures: return cached data if available, empty list if not, never raise
- Malformed data: skip entries that fail to parse, log warnings

### 3.2 Modified: `src/autodev/config.py`

Add `McpsoConfig` dataclass and wire it into `MCPRegistryConfig`.

```python
@dataclass
class McpsoConfig:
	"""Configuration for mcpso directory integration."""
	enabled: bool = False
	source_url: str = "https://api.github.com/repos/chatmcp/mcpso/contents/"
	readme_url: str = "https://raw.githubusercontent.com/chatmcp/mcpso/main/README.md"
	api_url: str = ""                        # optional REST API endpoint
	cache_dir: str = "~/.autodev/cache"
	cache_ttl_hours: int = 24
	github_token: str = ""                   # optional, for higher rate limits
	max_results: int = 10                    # default limit for search queries
```

**Changes to existing `MCPRegistryConfig` (config.py:433):**

```python
@dataclass
class MCPRegistryConfig:
	"""MCP tool registry settings."""
	enabled: bool = False
	promotion_threshold: float = 0.7
	ema_alpha: float = 0.3
	mcpso: McpsoConfig = field(default_factory=McpsoConfig)  # NEW
```

**Changes to `_build_mcp_registry()` (config.py:998):** Parse the nested `[mcp_registry.mcpso]` TOML table into `McpsoConfig`.

**TOML example:**

```toml
[mcp_registry]
enabled = true

[mcp_registry.mcpso]
enabled = true
cache_ttl_hours = 12
github_token = ""  # or set via GITHUB_TOKEN env var
```

### 3.3 Modified: `src/autodev/mcp_registry.py`

Add external discovery methods alongside the existing internal registry.

**New methods on `MCPToolRegistry`:**

```python
class MCPToolRegistry:
	def __init__(self, db: Database, config: MCPRegistryConfig) -> None:
		self._db = db
		self._config = config
		self._mcpso: McpsoClient | None = None
		if config.mcpso.enabled:
			from autodev.mcpso_client import McpsoClient
			self._mcpso = McpsoClient(config.mcpso)

	# Existing methods unchanged...

	def discover_external(self, intent: str, limit: int = 5) -> list[McpsoServer]:
		"""Search the mcpso directory for MCP servers matching intent."""
		if self._mcpso is None:
			return []
		return self._mcpso.search(intent, limit=limit)

	def discover_all(self, intent: str, limit: int = 5) -> tuple[list[MCPToolEntry], list[McpsoServer]]:
		"""Unified discovery: internal tools + external MCP servers."""
		internal = self.discover(intent, limit=limit)
		external = self.discover_external(intent, limit=limit)
		return internal, external

	def suggest_servers_for_task(self, task_description: str) -> str:
		"""Render a prompt section suggesting relevant MCP servers for a task.

		Returns empty string if no relevant servers found or mcpso disabled.
		Used by worker_prompt.py and swarm/context.py to inject suggestions.
		"""
		external = self.discover_external(task_description, limit=3)
		if not external:
			return ""
		lines = ["## Available MCP Servers", ""]
		lines.append("These community MCP servers may help with this task:")
		lines.append("")
		for srv in external:
			lines.append(f"### {srv.name}")
			lines.append(f"{srv.description}")
			if srv.install_command:
				lines.append(f"Install: `{srv.install_command}`")
			if srv.github_url:
				lines.append(f"Source: {srv.github_url}")
			lines.append("")
		return "\n".join(lines)
```

**Important:** `discover()` (internal) and `discover_external()` (mcpso) remain separate and return different types (`MCPToolEntry` vs `McpsoServer`). `discover_all()` is a convenience that returns both. This avoids mixing internal synthesized tools (which have handler_paths, quality scores, usage counts) with external catalog entries (which are informational pointers).

### 3.4 Modified: `src/autodev/swarm/context.py`

In `_discover_tools()` (line 744-754), extend to include external server suggestions when mcpso is enabled.

```python
def _discover_tools(self) -> list[str]:
	"""Find available synthesized tools and external MCP servers."""
	tools: list[str] = []
	try:
		from autodev.mcp_registry import MCPToolRegistry
		registry = MCPToolRegistry(self._db, self._config.mcp_registry)
		internal = registry.discover("", limit=20)
		tools = [t.name for t in internal]
		external = registry.discover_external("", limit=10)
		for srv in external:
			tools.append(f"[mcpso] {srv.name}")
	except Exception:
		pass
	return tools
```

### 3.5 Modified: `src/autodev/swarm/worker_prompt.py`

When building worker prompts, inject `suggest_servers_for_task()` output alongside the existing `render_tool_reflection_section()` output. Only injected when mcpso is enabled and relevant servers exist.

---

## 4. Data Flow

### Fetch Flow (background, async)

```
1. Controller/planner starts -> MCPToolRegistry.__init__() creates McpsoClient
2. McpsoClient checks cache file age
3. If stale (> cache_ttl_hours):
   a. GET GitHub Contents API -> find data file
   b. If data file found: parse JSON -> list[McpsoServer]
   c. If no data file: GET raw README -> parse markdown -> list[McpsoServer]
   d. Write cache file with fetched_at timestamp
4. If fresh: load from cache file
```

### Discovery Flow (sync, on-demand)

```
1. Planner/worker needs tools for task "analyze database schema"
2. Calls registry.discover_all("database schema")
3. Internal: SQLite LIKE query -> [MCPToolEntry(name="schema-analyzer", ...)]
4. External: McpsoClient.search("database schema") -> [McpsoServer(name="postgres-mcp", ...)]
5. Both returned to caller for prompt injection or decision-making
```

---

## 5. README Parsing Strategy

Since the mcpso repo likely stores its server directory as a curated markdown list (common for "awesome-*" repos), the README parser needs to handle this format:

**Expected structure (typical awesome-list pattern):**

```markdown
## Category Name

- [Server Name](https://github.com/org/repo) - Description of what it does
- [Another Server](https://github.com/org/repo2) - Another description
```

**Parser approach:**

1. Split README by `## ` headers to identify categories
2. Within each category, match lines against regex: `^\s*-\s*\[(.+?)\]\((.+?)\)\s*[-\u2013\u2014]\s*(.+)$`
3. Extract: name (group 1), url (group 2), description (group 3)
4. Current category becomes the `categories` field
5. Skip lines that don't match (prose, sub-headers, etc.)

**Fallback:** If the regex captures fewer than 5 entries, log a warning and return what we got. The cache will hold partial results until the next refresh.

---

## 6. Testing Requirements

### 6.1 Unit Tests: `tests/test_mcpso_client.py`

| Test | What it verifies |
|------|-----------------|
| `test_parse_readme_typical_format` | Parse an awesome-list style README into `McpsoServer` entries |
| `test_parse_readme_edge_cases` | Handle: empty README, no links, malformed lines, unicode |
| `test_parse_json_data_file` | Parse a JSON array of server objects |
| `test_search_keyword_matching` | Case-insensitive search across name, description, categories |
| `test_search_no_results` | Returns empty list for non-matching query |
| `test_search_respects_limit` | Returns at most `limit` results |
| `test_search_ordered_by_stars` | Higher-star servers ranked first |
| `test_cache_write_and_read` | Cache file round-trips correctly |
| `test_cache_ttl_fresh` | Fresh cache is not re-fetched |
| `test_cache_ttl_stale` | Stale cache triggers re-fetch |
| `test_cache_missing_returns_empty` | No cache + network failure -> empty list, no exception |
| `test_network_failure_uses_cache` | Network error with fresh cache -> cached data returned |
| `test_network_failure_no_cache` | Network error with no cache -> empty list, no exception |
| `test_rate_limit_handling` | Respects GitHub rate limit headers |
| `test_malformed_entries_skipped` | Bad entries are skipped, valid ones still returned |
| `test_get_by_name` | Exact name lookup returns correct server |
| `test_github_token_header` | Token is sent as Authorization header when configured |

### 6.2 Unit Tests: `tests/test_mcp_registry.py` (additions)

| Test | What it verifies |
|------|-----------------|
| `test_discover_external_disabled` | Returns empty when mcpso.enabled=False |
| `test_discover_external_enabled` | Returns McpsoServer entries when enabled (mock McpsoClient) |
| `test_discover_all_combines_results` | Returns both internal MCPToolEntry and external McpsoServer |
| `test_suggest_servers_for_task` | Renders markdown section with server details |
| `test_suggest_servers_empty` | Returns empty string when no matches or disabled |

### 6.3 Config Tests: `tests/test_mcp_registry.py` (additions)

| Test | What it verifies |
|------|-----------------|
| `test_mcpso_config_defaults` | Default values: enabled=False, cache_ttl_hours=24, etc. |
| `test_mcpso_config_toml_parsing` | TOML `[mcp_registry.mcpso]` section parsed correctly |
| `test_mcpso_config_env_token` | `github_token` falls back to `GITHUB_TOKEN` env var |

### 6.4 Integration Test (optional, requires network)

| Test | What it verifies |
|------|-----------------|
| `test_fetch_real_mcpso_data` | Actually fetches from GitHub, parses >0 entries. Marked `@pytest.mark.network`. |

### 6.5 Mocking Strategy

- Use `unittest.mock.patch` on `httpx.AsyncClient.get` for all network tests
- Provide fixture README content (`SAMPLE_README`) and JSON content (`SAMPLE_JSON`) as test constants
- Mock `McpsoClient` in registry tests to avoid network dependency

---

## 7. Risk Assessment

### Low Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| mcpso repo format changes | Medium | Low | Cache still serves stale data; parser gracefully degrades to empty list |
| GitHub API rate limiting | Low | Low | Unauthenticated limit is 60 req/hour, we fetch ~1/day; optional token |
| Network unavailable | Medium | Low | Cache-first strategy; no network = empty suggestions, agents still work |
| Large README parsing cost | Low | Low | README is typically <100KB; parsed once per cache refresh |

### Medium Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Stale/inaccurate server info | Medium | Medium | Cache TTL keeps data fresh; suggestions are informational, not auto-installed |
| Prompt bloat from suggestions | Low | Medium | Limit to top 3 suggestions; only inject when mcpso.enabled=True |

### Non-Risks

| Concern | Why it's not a risk |
|---------|-------------------|
| Security of external MCP servers | We never auto-install or auto-run servers; suggestions are informational text in prompts |
| Breaking existing registry flow | All new code is additive; `discover()` unchanged; new methods are opt-in via `mcpso.enabled` |
| Database schema changes | Using JSON file cache, not modifying SQLite schema |

---

## 8. Implementation Phases

### Phase 1: McpsoClient + Config (core)

1. Create `McpsoConfig` dataclass in `config.py`
2. Add `mcpso` field to `MCPRegistryConfig`
3. Update `_build_mcp_registry()` to parse `[mcp_registry.mcpso]`
4. Create `src/autodev/mcpso_client.py` with `McpsoClient` and `McpsoServer`
5. Implement README parser and JSON parser
6. Implement file-based caching
7. Write `tests/test_mcpso_client.py`

### Phase 2: Registry Integration

1. Add `discover_external()`, `discover_all()`, `suggest_servers_for_task()` to `MCPToolRegistry`
2. Update `MCPToolRegistry.__init__()` to create `McpsoClient` when enabled
3. Add registry tests in `tests/test_mcp_registry.py`

### Phase 3: Consumer Wiring

1. Update `swarm/context.py` `_discover_tools()` to include external servers
2. Update worker prompt builder to inject `suggest_servers_for_task()` output
3. End-to-end test: planner sees external servers in context

---

## 9. Configuration Reference

```toml
[mcp_registry]
enabled = true
promotion_threshold = 0.7
ema_alpha = 0.3

[mcp_registry.mcpso]
enabled = true
source_url = "https://api.github.com/repos/chatmcp/mcpso/contents/"
readme_url = "https://raw.githubusercontent.com/chatmcp/mcpso/main/README.md"
api_url = ""
cache_dir = "~/.autodev/cache"
cache_ttl_hours = 24
github_token = ""
max_results = 10
```

---

## 10. Future Extensions (Out of Scope)

- **Auto-install MCP servers**: Generate `--mcp-config` entries from mcpso data for worker subprocesses. Requires HITL approval gate.
- **Quality scoring of external servers**: Track which mcpso servers agents actually use and EMA-score them like internal tools.
- **Multiple directories**: Support additional MCP directories beyond mcpso (e.g., official Anthropic registry if one launches).
- **Bidirectional sync**: Publish high-quality internal tools back to mcpso or a private registry.