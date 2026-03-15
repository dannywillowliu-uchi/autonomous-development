The spec is ready but awaiting write permission. Here's a summary of what the spec covers:

**10 new MCP tools** added to the existing `mcp_server.py`:

**Swarm tools (6):**
- `launch_swarm` -- spawn swarm subprocess for a registered project
- `get_swarm_status` -- read `.autodev-swarm-state.json` for live state
- `inject_swarm_directive` -- write directives to team-lead inbox
- `stop_swarm` -- SIGTERM with fallback to shutdown directive
- `list_swarm_agents` -- extract agent details from state
- `get_swarm_learnings` -- read `.autodev-swarm-learnings.md`

**Adaptation tools (4):**
- `adapt_scan` -- run intelligence scanner via existing `IncrementalScanner`
- `adapt_apply` -- generate spec + launch swarm via `AutoUpdatePipeline` (dry_run defaults true)
- `adapt_history` -- query applied proposals from DB
- `adapt_generate_spec` -- generate implementation spec via `SpecGenerator`

**Supporting changes:**
- New `src/autodev/swarm/state_reader.py` for reusable state file reading
- `team_name` field added to `SwarmConfig`
- `_dispatch` made async to support adapt tool handlers
- Single server approach (no separate `mcp_adapt.py`)

**Key safety:** `adapt_apply` defaults to `dry_run: true` and reuses all existing safety rails (ratchet checkpoints, oracle checks, diff review gate, rate limiting, rollback on failure).