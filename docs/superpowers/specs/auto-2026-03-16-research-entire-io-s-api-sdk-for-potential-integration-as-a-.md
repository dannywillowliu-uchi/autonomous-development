Now I have everything I need. Here's the spec:

---

# Implementation Spec: Entire.io Backend Integration for Autodev

## Problem Statement

Autodev currently supports two worker backends (`local` and `ssh`/`container` via config) and runs all agents as local Claude Code subprocesses. Entire.io is a cloud development platform by Nat Friedman (ex-GitHub CEO) and Daniel Gross, designed to provide persistent cloud environments purpose-built for AI agents. Integrating it would let autodev spawn workers in cloud-hosted environments without users managing SSH hosts or containers.

**Critical context: A previous autodev swarm already implemented and then deleted an `EntireBackend` + `EntireConfig` integration (commit `c07d983`, March 16 2026) because no public API exists yet.** This spec accounts for that history and takes a gated approach -- Phase 1 is config plumbing + interface skeleton only, Phases 2-3 activate when the API becomes available.

### Why this matters

- **Scalability**: Local backends are bounded by the host machine's resources (CPU, RAM, disk). Cloud environments remove this ceiling.
- **Isolation**: Each worker gets a fully isolated cloud environment, eliminating the `WorkspacePool` git corruption issues documented in the project gotchas.
- **Cost model shift**: Instead of provisioning beefy dev machines, pay per-environment-hour. Aligns with autodev's budget-aware architecture (`BudgetConfig`).

## Changes Needed

### Phase 1: Config + Backend Skeleton (implementable now, no external deps)

#### 1.1 `src/autodev/config.py` -- Add `EntireConfig` dataclass

Add after `ContainerConfig` (after line ~206):

```python
@dataclass
class EntireConfig:
	"""Entire.io cloud backend settings (gated on API availability)."""

	api_key: str = ""           # ENTIRE_API_KEY env var fallback
	api_base_url: str = "https://api.entire.io"
	org_id: str = ""
	environment_template: str = ""  # pre-configured env template ID
	startup_timeout: int = 120
	machine_type: str = ""      # e.g. "small", "medium", "large"
	region: str = ""            # e.g. "us-east-1"
	persist_environments: bool = False  # keep envs alive between tasks
```

Update `BackendConfig` (line 209-216) to add `entire` field:

```python
@dataclass
class BackendConfig:
	"""Worker backend settings."""

	type: str = "local"  # local/ssh/container/entire
	max_output_mb: int = 50
	ssh_hosts: list[SSHHostConfig] = field(default_factory=list)
	container: ContainerConfig = field(default_factory=ContainerConfig)
	entire: EntireConfig = field(default_factory=EntireConfig)
```

#### 1.2 `src/autodev/config.py` -- Add `_build_entire()` builder

Add after `_build_container()`, following the same pattern:

```python
def _build_entire(data: dict[str, Any]) -> EntireConfig:
	ec = EntireConfig()
	for key in ("api_key", "api_base_url", "org_id", "environment_template",
	            "machine_type", "region"):
		if key in data:
			setattr(ec, key, str(data[key]))
	if "startup_timeout" in data:
		ec.startup_timeout = int(data["startup_timeout"])
	if "persist_environments" in data:
		ec.persist_environments = bool(data["persist_environments"])
	# Env var fallback for API key
	if not ec.api_key:
		ec.api_key = os.environ.get("ENTIRE_API_KEY", "")
	return ec
```

#### 1.3 `src/autodev/config.py` -- Wire into `_build_backend()` (line 774)

Add after the `container` block at line 793:

```python
if "entire" in data:
	bc.entire = _build_entire(data["entire"])
```

#### 1.4 `src/autodev/config.py` -- Add validation in `validate_config()` (line 1392)

Add after existing backend validation:

```python
if config.backend.type == "entire":
	if not config.backend.entire.api_key:
		issues.append(("error", "backend.entire.api_key is required (or set ENTIRE_API_KEY env var)"))
	if not config.backend.entire.api_base_url:
		issues.append(("error", "backend.entire.api_base_url cannot be empty"))
```

#### 1.5 `src/autodev/backends/entire.py` -- New file (skeleton)

Implements `WorkerBackend` from `backends/base.py`. All methods raise `NotImplementedError` until Phase 3 but the class structure, type signatures, and internal state management are complete.

```python
"""Entire.io cloud backend -- runs workers in cloud-hosted environments."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from autodev.backends.base import WorkerBackend, WorkerHandle
from autodev.config import EntireConfig, MissionConfig

logger = logging.getLogger(__name__)


@dataclass
class _TrackedEnvironment:
	"""Internal tracking of a provisioned Entire.io environment."""
	env_id: str
	worker_id: str
	workspace_identifier: str  # opaque ID, not a filesystem path


class EntireBackend(WorkerBackend):
	"""Execute workers in Entire.io cloud environments.

	Phase 1: Skeleton only. All methods raise NotImplementedError.
	Gate on public API availability before implementing.
	"""

	def __init__(
		self,
		config: EntireConfig,
		mission_config: MissionConfig | None = None,
		max_output_mb: int = 50,
	) -> None:
		self._config = config
		self._mission_config = mission_config
		self._max_output_bytes = max_output_mb * 1024 * 1024
		self._environments: dict[str, _TrackedEnvironment] = {}
		self._api_key = self._resolve_api_key()

	def _resolve_api_key(self) -> str:
		"""Resolve API key: explicit config > env var."""
		if self._config.api_key:
			return self._config.api_key
		env_key = os.environ.get("ENTIRE_API_KEY", "")
		if env_key:
			return env_key
		return ""

	async def initialize(self, warm_count: int = 0) -> None:
		raise NotImplementedError(
			"Entire.io API not yet available. "
			"Monitor https://entire.io/docs for public API documentation."
		)

	async def provision_workspace(
		self, worker_id: str, source_repo: str, base_branch: str
	) -> str:
		raise NotImplementedError("Entire.io API not yet available")

	async def spawn(
		self, worker_id: str, workspace_path: str, command: list[str], timeout: int
	) -> WorkerHandle:
		raise NotImplementedError("Entire.io API not yet available")

	async def check_status(self, handle: WorkerHandle) -> str:
		raise NotImplementedError("Entire.io API not yet available")

	async def get_output(self, handle: WorkerHandle) -> str:
		raise NotImplementedError("Entire.io API not yet available")

	async def kill(self, handle: WorkerHandle) -> None:
		raise NotImplementedError("Entire.io API not yet available")

	async def release_workspace(self, workspace_path: str) -> None:
		raise NotImplementedError("Entire.io API not yet available")

	async def cleanup(self) -> None:
		raise NotImplementedError("Entire.io API not yet available")
```

#### 1.6 `src/autodev/continuous_controller.py` -- Wire backend selection

Find the backend selection logic in `_init_components()`. Add an `elif` branch:

```python
elif self.config.backend.type == "entire":
	from autodev.backends.entire import EntireBackend
	backend = EntireBackend(
		config=self.config.backend.entire,
		mission_config=self.config,
		max_output_mb=self.config.backend.max_output_mb,
	)
	await backend.initialize()
	self._backend = backend
```

Note: The import is lazy (inside the branch) to avoid import errors when Entire.io isn't configured.

### Phase 2: API Client (gated on public API availability)

**Trigger**: Any of these become true:
- Public API docs at `https://entire.io/docs` or `https://docs.entire.io`
- PyPI package `entire-sdk` published
- GitHub org `https://github.com/entireio` with SDK code

#### 2.1 `src/autodev/backends/entire_client.py` -- HTTP client

Thin `httpx.AsyncClient`-based wrapper. Assumed REST endpoints (to be confirmed against actual docs):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Connectivity check |
| `POST` | `/environments` | Create cloud environment |
| `POST` | `/environments/{id}/exec` | Execute command in environment |
| `GET` | `/executions/{id}/status` | Poll execution status |
| `GET` | `/executions/{id}/output` | Fetch stdout/stderr |
| `DELETE` | `/environments/{id}` | Destroy environment |

```python
class EntireClient:
	def __init__(self, base_url: str, api_key: str, org_id: str = "") -> None: ...
	async def health_check(self) -> bool: ...
	async def create_environment(self, template_id: str, repo_url: str, branch: str, machine_type: str = "", region: str = "") -> str: ...
	async def exec_command(self, env_id: str, command: list[str], timeout: int = 300) -> str: ...
	async def get_execution_status(self, exec_id: str) -> dict: ...
	async def get_execution_output(self, exec_id: str) -> str: ...
	async def stop_execution(self, exec_id: str) -> None: ...
	async def destroy_environment(self, env_id: str) -> None: ...
```

#### 2.2 Fill in `EntireBackend` method bodies

Replace `NotImplementedError` stubs with real implementations using `EntireClient`. Key design decisions:

- `provision_workspace()` returns an opaque environment ID string (not a filesystem path) -- this is stored in `WorkerHandle.backend_metadata`
- `spawn()` calls `exec_command()` and starts an async polling loop for status
- `get_output()` fetches from the API with the same `max_output_bytes` truncation pattern as `LocalBackend` (lines 333-366 in `backends/local.py`)
- `release_workspace()` checks `persist_environments` flag before destroying
- `cleanup()` destroys all tracked environments in `self._environments`

### Phase 3: Swarm Mode + MCP Tools (optional, depends on Phase 2 success)

#### 3.1 `src/autodev/swarm/controller.py` -- Swarm backend wiring

The swarm controller spawns agents via `build_claude_cmd()` + `asyncio.create_subprocess_exec()`. For Entire.io, the controller would:
1. Provision a cloud environment via `EntireClient`
2. Execute the Claude command inside the cloud environment
3. Monitor via API polling instead of local process handles

This requires a different spawning path in `SwarmController._spawn_agent()`.

#### 3.2 `src/autodev/mcp_servers/entire_mcp.py` -- MCP server (optional)

Expose Entire.io environment management as MCP tools so planner agents can directly manage cloud environments:

| Tool | Description |
|------|-------------|
| `entire_create_env` | Create a new cloud environment |
| `entire_exec` | Run a command in an environment |
| `entire_list_envs` | List active environments |
| `entire_destroy_env` | Tear down an environment |

Follow the pattern in `src/autodev/mcp_server.py`.

### TOML Config Example

```toml
[backend]
type = "entire"
max_output_mb = 50

[backend.entire]
# api_key = ""  # prefer ENTIRE_API_KEY env var
api_base_url = "https://api.entire.io"
org_id = "my-org"
environment_template = "python-3.11"
startup_timeout = 120
machine_type = "medium"
region = "us-east-1"
persist_environments = false
```

### Files Summary

| File | Action | Phase | Description |
|------|--------|-------|-------------|
| `src/autodev/config.py` | Modify | 1 | Add `EntireConfig`, update `BackendConfig`, `_build_backend()`, `_build_entire()`, `validate_config()` |
| `src/autodev/backends/entire.py` | Create | 1 | `EntireBackend(WorkerBackend)` skeleton with `NotImplementedError` stubs |
| `tests/test_entire_backend.py` | Create | 1 | Config parsing + skeleton tests |
| `tests/test_entire_config.py` | Create | 1 | TOML round-trip + validation tests |
| `src/autodev/backends/entire_client.py` | Create | 2 | `httpx`-based HTTP client for Entire.io API |
| `src/autodev/continuous_controller.py` | Modify | 2 | Add `entire` branch in `_init_components()` |
| `src/autodev/swarm/controller.py` | Modify | 3 | Swarm agent spawn path for cloud environments |
| `src/autodev/mcp_servers/entire_mcp.py` | Create | 3 | MCP server exposing Entire.io tools |

## Testing Requirements

### Phase 1 Unit Tests

#### `tests/test_entire_config.py`

1. **TOML parsing**: `[backend.entire]` section correctly produces `EntireConfig` with all fields populated
2. **Default values**: `EntireConfig()` matches expected defaults (`api_base_url="https://api.entire.io"`, `startup_timeout=120`, `persist_environments=False`)
3. **Env var fallback**: When `api_key` is empty in TOML, `_build_entire()` reads `ENTIRE_API_KEY` from `os.environ` (use `monkeypatch`)
4. **Env var precedence**: Explicit `api_key` in TOML takes precedence over env var
5. **BackendConfig type**: `type = "entire"` is accepted without error when API key is set
6. **Validation errors**: `validate_config()` produces `("error", ...)` when `backend.type == "entire"` but `api_key` is empty
7. **No side effects**: `_build_backend()` ignores `[backend.entire]` section when `type != "entire"` (i.e., `EntireConfig` is still default-constructed but not validated)

#### `tests/test_entire_backend.py`

1. **Instantiation**: `EntireBackend(config=EntireConfig(api_key="test"))` succeeds
2. **API key resolution**: `_resolve_api_key()` returns config key > env var > empty string
3. **NotImplementedError**: All `WorkerBackend` methods raise `NotImplementedError` with descriptive message
4. **Interface conformance**: `EntireBackend` satisfies `isinstance(backend, WorkerBackend)`

### Phase 2 Tests (gated on API availability)

1. **`EntireClient` unit tests** with `httpx` mock responses:
   - `health_check()` returns `True` on 200, `False` on 5xx
   - `create_environment()` returns env ID from response JSON
   - `exec_command()` returns execution ID
   - `get_execution_output()` returns stdout text
   - HTTP errors raise appropriate exceptions
   - Auth header is set correctly (`Authorization: Bearer <api_key>`)

2. **`EntireBackend` integration tests** with mocked `EntireClient`:
   - `provision_workspace()` calls `create_environment()` and returns env ID
   - `spawn()` calls `exec_command()` and returns `WorkerHandle` with `backend_metadata` set
   - `release_workspace()` calls `destroy_environment()` when `persist_environments=False`
   - `release_workspace()` skips destruction when `persist_environments=True`
   - `cleanup()` destroys all environments in `self._environments`
   - `get_output()` truncates at `max_output_bytes`

### What to Verify Before Any Commit

- `pytest -q` passes (all existing tests + new tests)
- `ruff check src/ tests/` passes
- `bandit -r src/ -lll -q` passes (no new security issues)
- Existing `tests/test_config.py` still passes (no regressions from `BackendConfig` changes)
- Lazy import in `continuous_controller.py` doesn't break when `entire.py` has syntax errors (test by temporarily breaking the file)

## Risk Assessment

### Risk 1: No Public API (HIGH)

**What**: As of March 2026, Entire.io appears to be invite-only with no documented public API, no published SDK on PyPI/npm, and no public GitHub org.
**Evidence**: A prior autodev swarm implemented the full integration on March 15, then another swarm deleted it on March 16 (commit `c07d983`) after concluding there was nothing to target.
**Mitigation**: Phase 1 is zero-dependency config plumbing only. All methods raise `NotImplementedError`. No `httpx` import, no network calls. The backend only activates when explicitly configured with `type = "entire"`.
**Gate**: Do NOT proceed to Phase 2 until at least one of: (a) public API docs at `https://entire.io/docs`, (b) PyPI package `entire-sdk`, (c) GitHub SDK repo at `https://github.com/entireio`.

### Risk 2: Repeated Implement-Delete Cycle (MEDIUM)

**What**: This is the second attempt at this integration. If Phase 1 is implemented and the API never materializes, it becomes dead code that a future cleanup agent may delete again.
**Mitigation**: Keep Phase 1 minimal (~150 lines config, ~80 lines skeleton). Add a comment in `EntireConfig` docstring noting the API gate condition. If the config plumbing isn't worth maintaining, it can be cleanly removed by deleting `EntireConfig`, the `entire` field from `BackendConfig`, `_build_entire()`, and `backends/entire.py`.

### Risk 3: API Shape Mismatch (MEDIUM)

**What**: The assumed REST endpoints (create env, exec, poll, destroy) may not match Entire.io's actual API when it ships. They might use WebSockets, gRPC, or a different paradigm entirely.
**Mitigation**: The `WorkerBackend` interface is transport-agnostic -- it only cares about `provision_workspace`, `spawn`, `check_status`, `get_output`, `kill`, `release_workspace`, `cleanup`. The `EntireClient` in Phase 2 is a thin HTTP wrapper that can be rewritten for any protocol without changing the backend's public interface.

### Risk 4: Green Branch Incompatibility (MEDIUM)

**What**: `GreenBranchManager` assumes workspaces are local filesystem paths and adds them as git remotes. Cloud environments break this assumption.
**Mitigation**: For Phase 1/2, document that `green_branch` mode is incompatible with `entire` backend. Workers in cloud environments would need to push to a remote branch that the controller merges. This requires a `RemoteMergeStrategy` that doesn't exist yet -- defer to Phase 3 based on how Entire.io handles git.

### Risk 5: Credential Leakage (LOW)

**What**: API keys in TOML config could be committed to git.
**Mitigation**: `_build_entire()` supports env var fallback (`ENTIRE_API_KEY`) as the recommended auth path. `validate_config()` produces a warning if the key appears to be hardcoded in the config file (check if `api_key` is non-empty and not an env var reference). The project `.gitignore` already excludes `.env` files.

### Risk 6: Orphaned Cloud Environments (MEDIUM, Phase 2+)

**What**: If an agent crashes before `release_workspace()` or `cleanup()`, cloud environments may persist and accumulate costs.
**Mitigation**:
- `cleanup()` in `finally` blocks (same pattern as `LocalBackend`)
- `persist_environments=False` by default
- Set environment TTL/auto-destroy in the API call if supported
- Log all environment IDs at creation for manual cleanup
- Add a `autodev entire-cleanup` CLI subcommand that lists and destroys orphaned environments