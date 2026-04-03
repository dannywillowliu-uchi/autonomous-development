The file write keeps getting blocked. Here's the complete spec -- I'll output it directly so you can review it and save it yourself if you'd like.

# Spec: GitMCP Integration for Worker Agents

## Problem Statement

When autodev worker agents (research, planning, or implementation roles) encounter unfamiliar dependencies, they currently have no structured way to access that dependency's documentation. They rely on whatever the LLM already knows, which may be stale or hallucinated.

GitMCP (gitmcp.io) auto-generates a remote MCP server for any public GitHub repo by converting `github.com/owner/repo` to `gitmcp.io/owner/repo`. Agents get structured access to `llms.txt`, `llms-full.txt`, `README.md`, and source code -- served over MCP with zero setup.

Integrating GitMCP means:
1. The **planner** can detect when a task involves an unfamiliar dependency and auto-provision a GitMCP endpoint
2. **Worker agents** spawned for research/planning get that endpoint injected as an MCP server
3. The **registry** caches resolved endpoints so the same dependency isn't re-provisioned across agents or runs

---

## Changes Needed

### 1. New module: `src/autodev/gitmcp.py`

URL generation and validation utility.

**Functions:**
- `github_url_to_gitmcp(github_url: str) -> str | None` -- converts `github.com/owner/repo` to `gitmcp.io/owner/repo`, handles GitHub Pages (`owner.github.io/repo` -> `owner.gitmcp.io/repo`), strips subpaths, returns `None` for non-GitHub URLs
- `extract_github_repos_from_dependencies(project_path: Path) -> list[dict[str, str]]` -- scans `pyproject.toml`, `requirements.txt`, `package.json`, `go.mod`, `Cargo.toml` and returns `[{"name": ..., "github_url": ..., "gitmcp_url": ...}]`. No network calls -- local file parsing only, uses `resolve_package_to_github` for mapping
- `resolve_package_to_github(package_name: str, ecosystem: str) -> str | None` -- best-effort local heuristics: reads npm `node_modules/*/package.json` repo field, Python `.venv` METADATA `Home-page`/`Project-URL`, no network calls
- `async validate_gitmcp_endpoint(url: str, timeout: float = 5.0) -> bool` -- HEAD request, returns True on 2xx

### 2. Modify: `src/autodev/mcp_registry.py`

New dataclass `GitMCPEndpoint` with fields: `id`, `package_name`, `github_url`, `gitmcp_url`, `validated`, `usage_count`, `created_at`, `last_used_at`.

New methods on `MCPToolRegistry`:
- `register_gitmcp(package_name, github_url, gitmcp_url, validated) -> GitMCPEndpoint` -- idempotent on `package_name`
- `get_gitmcp(package_name) -> GitMCPEndpoint | None`
- `get_gitmcp_for_packages(package_names) -> list[GitMCPEndpoint]`
- `record_gitmcp_usage(package_name) -> None`

### 3. DB changes: `src/autodev/db.py`

New table `gitmcp_endpoints` (`id TEXT PRIMARY KEY, package_name TEXT UNIQUE NOT NULL, github_url, gitmcp_url, validated INTEGER, usage_count INTEGER, created_at, last_used_at`). Index on `package_name`. CRUD methods: `insert_gitmcp_endpoint`, `get_gitmcp_endpoint`, `get_gitmcp_endpoints`, `update_gitmcp_endpoint`.

### 4. Modify: `src/autodev/swarm/worker_prompt.py`

New `_gitmcp_section(gitmcp_endpoints: list[GitMCPEndpoint]) -> str` that renders available dependency documentation servers. Add `gitmcp_endpoints: list[GitMCPEndpoint] | None = None` parameter to `build_worker_prompt()`, wire into sections after `_mcp_section`.

### 5. Modify: `src/autodev/swarm/controller.py`

Three changes:

**a) `initialize()`**: After `scan_capabilities()` (~line 143), if `config.mcp_registry.gitmcp_enabled`, call `extract_github_repos_from_dependencies`, validate each endpoint, register in `MCPToolRegistry`. Cap at `gitmcp_max_endpoints`. Requires adding `self._mcp_registry = MCPToolRegistry(db, config.mcp_registry)` in `__init__`.

**b) `_handle_spawn()`**: New helper `_get_relevant_gitmcp_endpoints(agent, task)` returns validated endpoints -- all for researcher/planner roles, import-matched subset for implementers, capped at 5. Pass to `build_worker_prompt`.

**c) Per-agent MCP config**: New `_write_agent_mcp_config(agent, endpoints) -> Path | None` merges project `.mcp.json` with GitMCP SSE entries, writes to `.autodev-traces/{run_id}/{agent_name}.mcp.json`. Pass via `--mcp-config` in `_spawn_claude_session`.

### 6. Modify: `src/autodev/context_gathering.py`

New `get_dependency_docs_context(registry: MCPToolRegistry) -> str` for planner context -- lists available GitMCP endpoints with validation status and usage counts.

### 7. Modify: `src/autodev/swarm/prompts.py`

Add GitMCP shortcut guidance to `register_mcp` decision type docs in `SYSTEM_PROMPT`: `{"name": "gitmcp-<repo>", "type": "sse", "url": "https://gitmcp.io/owner/repo"}`. Note that common deps are auto-provisioned; `register_mcp` is for ad-hoc repos.

### 8. Modify: `src/autodev/config.py`

Add to `MCPRegistryConfig`: `gitmcp_enabled: bool = True`, `gitmcp_max_endpoints: int = 10`, `gitmcp_validation_timeout: float = 5.0`. Update `_build_mcp_registry()` parser.

Add `mcp_config_override: str | None = None` to `build_claude_cmd()` -- when set, takes precedence over `config.mcp.config_path`.

---

## Testing Requirements

### `tests/test_gitmcp.py`
- URL conversion: standard repo, GitHub Pages, trailing slash, subpath stripping, non-GitHub (None), passthrough, invalid (None)
- Dependency extraction: pyproject.toml, package.json, requirements.txt, empty project
- Package resolution: node_modules, venv METADATA, unknown (None)

### `tests/test_mcp_registry_gitmcp.py`
- Register new, register idempotent, get existing, get missing, batch get, usage increment

### `tests/test_worker_prompt_gitmcp.py`
- Section renders with endpoints, empty returns "", full prompt includes section

### `tests/test_gitmcp_integration.py`
- Real endpoint validation (`@pytest.mark.network`)
- Controller init provisions endpoints (mock validate)
- Spawn includes GitMCP in prompt (mock spawn)
- Per-agent `.mcp.json` written to trace dir

---

## Risk Assessment

| Level | Risk | Mitigation |
|-------|------|-----------|
| Low | GitMCP down/slow | `validate_gitmcp_endpoint` gates provisioning with 5s timeout. Additive -- agents work without it. |
| Low | Prompt bloat from too many endpoints | `gitmcp_max_endpoints` cap (10). Workers get task-relevant subset (5). |
| Low | Private repos 404 | Validation catches; only validated endpoints cached. |
| Low | GitMCP URL scheme changes | All generation in `gitmcp.py` -- single point of change. |
| Medium | Per-agent MCP configs accumulate | Under `.autodev-traces/{run_id}/` (ephemeral). Cleanup in controller shutdown. |
| Medium | Wrong package->repo mapping | Validation catches bad URLs. Manual override via `package_name` key. |
| None | Security | Read-only public repos, no auth tokens. |
| None | Cost | GitMCP is free. Only HEAD requests for validation. |
| None | Breaking changes | All additive. `gitmcp_endpoints` defaults to `None`. New DB table uses `CREATE TABLE IF NOT EXISTS`. |

---

## Implementation Order

1. `src/autodev/gitmcp.py` + `tests/test_gitmcp.py`
2. DB schema + CRUD in `db.py` + `tests/test_mcp_registry_gitmcp.py`
3. `GitMCPEndpoint` + registry methods in `mcp_registry.py`
4. Config additions in `config.py` (`MCPRegistryConfig` fields + `mcp_config_override` param)
5. Worker prompt integration in `worker_prompt.py` + `tests/test_worker_prompt_gitmcp.py`
6. Planner context in `context_gathering.py`
7. Controller integration in `controller.py` (provisioning + per-agent MCP config + spawn wiring)
8. Planner prompt in `prompts.py`
9. Integration tests in `tests/test_gitmcp_integration.py`