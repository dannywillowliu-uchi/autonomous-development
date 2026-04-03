# Implementation Spec: Cowork Architecture Review for autodev Integration

## Problem Statement

autodev's research and planning phases currently treat Claude Code as a coding-only tool -- all subprocess spawning in `session.py` uses coding-oriented flags (`--permission-mode auto`, `--output-format text`, `--max-turns`). Cowork ("Claude Code for the rest of your work") signals that Anthropic is expanding Claude Code's CLI capabilities beyond code editing into research, planning, and documentation workflows. If new CLI flags, modes, or interaction patterns are exposed, autodev can leverage them to improve the quality and efficiency of its non-coding phases (research subagents, planner deliberation, documentation generation) without reinventing what the platform now provides natively.

**Specific improvements targeted:**
- Research subagents (spawned in `swarm/planner.py` and `deliberative_planner.py`) could use a dedicated research mode if one exists, getting better tool access and output formatting
- Planning phases could leverage structured output modes for plans rather than parsing freeform markdown
- `config.py`'s `build_claude_cmd()` needs to support any new flags so the entire framework benefits
- `session.py`'s result parsing may need updates if Cowork introduces new output formats or structured result protocols

## Changes Needed

### Phase 1: Research & Audit (no code changes)

**Goal:** Enumerate all new CLI flags, modes, and capabilities exposed by Cowork-era Claude Code.

#### 1.1 CLI Flag Discovery

Run the following commands and document output:
```bash
claude --help 2>&1
claude --help-all 2>&1  # if exists
claude cowork --help 2>&1  # if subcommand exists
```

**File to create:** `docs/superpowers/specs/cowork-cli-audit.md`

Document every flag not currently used in autodev, categorized as:
- **Directly applicable** -- can use today in autodev subprocess spawning
- **Potentially applicable** -- needs wrapper or config support
- **Not applicable** -- coding-specific or irrelevant

#### 1.2 Capability Mapping

Cross-reference discovered capabilities against autodev's current subprocess spawning in these locations:

| File | Function | Current flags used | Gap |
|------|----------|--------------------|-----|
| `src/autodev/config.py` | `build_claude_cmd()` | `--print`, `--output-format`, `--max-turns`, `--permission-mode`, `--mcp-config`, `--model`, `--permission-prompt-tool` | New flags from audit |
| `src/autodev/swarm/controller.py` | `_spawn_agent()` | Uses `build_claude_cmd()` + `--permission-mode auto` | Research/planning mode flags |
| `src/autodev/session.py` | subprocess invocation patterns | `--output-format text` | Structured output modes |
| `src/autodev/deliberative_planner.py` | planner subprocess | Uses `build_claude_cmd()` | Planning-specific modes |

### Phase 2: Config Layer Updates

**File:** `src/autodev/config.py`

#### 2.1 Add `AgentMode` enum to config

```python
class AgentMode(str, Enum):
	"""Claude Code agent mode selection."""
	CODE = "code"        # Default coding mode
	RESEARCH = "research"  # Research-oriented (if Cowork exposes this)
	PLAN = "plan"        # Planning-oriented (if Cowork exposes this)
	AUTO = "auto"        # Let Claude Code decide based on prompt
```

**Where:** After existing enums in `config.py`, before `MissionConfig`.

#### 2.2 Extend `build_claude_cmd()` to accept agent mode

**Function:** `build_claude_cmd()` in `config.py`

**Current signature** (inferred from usage):
```python
def build_claude_cmd(config, prompt, *, output_format="text", max_turns=None, mcp_config=None, model=None, ...):
```

**Change:** Add `agent_mode: AgentMode | None = None` parameter. If the CLI audit reveals a `--mode` or `--cowork` flag, append it. If no such flag exists, this becomes a no-op that documents the intent for future use.

```python
def build_claude_cmd(config, prompt, *, agent_mode: AgentMode | None = None, ...):
	cmd = [...]
	# Phase 2: append mode flag if supported and specified
	if agent_mode and agent_mode != AgentMode.CODE:
		if _CLI_SUPPORTS_MODE:  # constant set during Phase 1 audit
			cmd.extend(["--mode", agent_mode.value])
		else:
			logger.debug("Agent mode %s requested but CLI does not support --mode flag", agent_mode)
	return cmd
```

#### 2.3 TOML config support

**File:** `src/autodev/config.py` (config loading section)

Add optional `agent_mode` field to `[swarm]` and `[mission]` TOML sections:

```toml
[swarm]
research_agent_mode = "research"  # mode for research-phase agents
planning_agent_mode = "plan"      # mode for planner subprocess
worker_agent_mode = "code"        # mode for coding workers (default)
```

**Parsing:** In the TOML loader, read these as `AgentMode` enums with fallback to `AgentMode.CODE`.

### Phase 3: Swarm Planner Integration

**File:** `src/autodev/swarm/planner.py`

#### 3.1 Mode-aware agent spawning decisions

In `DrivingPlanner`, when the planner creates `spawn_agent` decisions, include `agent_mode` in the decision payload:

**Current pattern** (in planner decision handling):
```python
# Planner emits: {"type": "spawn_agent", "task_id": "...", "prompt": "..."}
```

**New pattern:**
```python
# Planner emits: {"type": "spawn_agent", "task_id": "...", "prompt": "...", "agent_mode": "research"}
```

**Where this propagates:** `src/autodev/swarm/controller.py` in `_spawn_agent()` reads the decision and passes `agent_mode` to `build_claude_cmd()`.

#### 3.2 Task type to mode mapping

**File:** `src/autodev/swarm/models.py`

Add a mapping from task categories to agent modes:

```python
TASK_TYPE_MODE_MAP: dict[str, AgentMode] = {
	"research": AgentMode.RESEARCH,
	"planning": AgentMode.PLAN,
	"documentation": AgentMode.RESEARCH,
	"implementation": AgentMode.CODE,
	"bugfix": AgentMode.CODE,
	"test": AgentMode.CODE,
	"refactor": AgentMode.CODE,
}
```

**Used by:** `swarm/planner.py` when it classifies tasks and decides which mode to use. Falls back to `AgentMode.CODE` for unknown task types.

#### 3.3 Planner system prompt update

**File:** `src/autodev/swarm/prompts.py`

Update the planner system prompt to include mode awareness in its decision schema:

```
When spawning agents, specify the agent_mode field:
- "research" for information gathering, API exploration, documentation reading
- "plan" for architecture design, task decomposition, feasibility analysis
- "code" for implementation, bugfixes, tests, refactoring (default)
```

### Phase 4: Session Layer Updates

**File:** `src/autodev/session.py`

#### 4.1 Extended result parsing

If Cowork introduces a new structured output format (e.g., JSON-LD, structured research results), add a parser alongside the existing `parse_mc_result()`:

```python
def parse_research_result(output: str) -> dict[str, object] | None:
	"""Extract structured research findings from research-mode output.

	Falls back to parse_mc_result() if no research-specific format is detected.
	"""
	# Check for research-specific markers (TBD from Phase 1 audit)
	# e.g., RESEARCH_RESULT: or structured JSON with "findings" key
	...
	return parse_mc_result(output)  # fallback
```

#### 4.2 Result schema extension

**File:** `src/autodev/models.py`

If the audit reveals research-mode agents produce richer output, extend `MCResultSchema`:

```python
class MCResultSchema(BaseModel):
	status: str
	commits: list[str] = []
	summary: str = ""
	files_changed: list[str] = []
	discoveries: list[str] = []
	concerns: list[str] = []
	# New fields for research/planning mode output
	research_findings: list[dict[str, str]] = []  # [{topic, finding, confidence}]
	references: list[str] = []  # URLs or file paths consulted
	recommendations: list[str] = []  # Actionable next steps
```

**Backward compatible:** All new fields have defaults, so existing workers are unaffected.

### Phase 5: Conditional Feature Gate

**File:** `src/autodev/config.py`

Since Cowork capabilities depend on the installed Claude Code version, add a runtime capability check:

```python
import subprocess

_COWORK_CAPABILITIES: dict[str, bool] = {}

def detect_claude_capabilities() -> dict[str, bool]:
	"""Probe the installed Claude Code binary for Cowork-era capabilities."""
	global _COWORK_CAPABILITIES
	if _COWORK_CAPABILITIES:
		return _COWORK_CAPABILITIES

	try:
		result = subprocess.run(
			["claude", "--help"],
			capture_output=True, text=True, timeout=10,
		)
		help_text = result.stdout + result.stderr
		_COWORK_CAPABILITIES = {
			"mode_flag": "--mode" in help_text,
			"cowork_subcommand": "cowork" in help_text,
			"structured_research": "--research" in help_text,
			# Add more as discovered in Phase 1
		}
	except (FileNotFoundError, subprocess.TimeoutExpired):
		_COWORK_CAPABILITIES = {}

	return _COWORK_CAPABILITIES
```

**Called by:** `build_claude_cmd()` on first invocation (cached thereafter). Gracefully degrades -- if no new flags exist, autodev continues operating exactly as it does today.

## Testing Requirements

### Unit Tests

**File:** `tests/test_cowork_integration.py`

```
test_agent_mode_enum_values()
	- Verify AgentMode enum has expected members
	- Verify string serialization matches CLI flag values

test_build_claude_cmd_with_mode()
	- When CLI supports --mode: verify flag appears in command list
	- When CLI does not support --mode: verify flag is absent, no error
	- When agent_mode is None: verify no mode flag (backward compat)

test_task_type_mode_mapping()
	- "research" -> AgentMode.RESEARCH
	- "implementation" -> AgentMode.CODE
	- Unknown type -> AgentMode.CODE (fallback)

test_detect_claude_capabilities()
	- Mock subprocess.run with help text containing "--mode": capabilities["mode_flag"] is True
	- Mock subprocess.run with old help text: capabilities["mode_flag"] is False
	- Mock FileNotFoundError: returns empty dict

test_parse_research_result_fallback()
	- Standard AD_RESULT output: falls back to parse_mc_result successfully
	- Research-specific output (if format discovered): parses correctly

test_mc_result_schema_backward_compat()
	- Old-format AD_RESULT (without new fields): validates successfully
	- New-format AD_RESULT (with research_findings): validates successfully
```

### Integration Tests

**File:** `tests/test_cowork_e2e.py`

```
test_swarm_research_task_uses_research_mode()
	- Create a swarm task with type "research"
	- Verify the spawned subprocess command includes agent_mode appropriately
	- Mock subprocess to avoid actual Claude Code invocation

test_config_toml_agent_mode_parsing()
	- Load a TOML config with research_agent_mode = "research"
	- Verify config object exposes AgentMode.RESEARCH
	- Load TOML without agent_mode fields: verify defaults to CODE
```

### Manual Verification

After Phase 1 audit:
1. Run `autodev swarm` with a research-heavy objective and `research_agent_mode = "research"` in config
2. Verify agent spawning logs show correct mode flag (or graceful degradation)
3. Compare research quality/speed with and without mode flag (if applicable)

## Risk Assessment

### Low Risk: No Cowork-specific CLI flags exist yet

**Likelihood:** Medium-high (Cowork may be a product feature, not a CLI mode)

**Impact:** Low -- all code changes are behind capability detection. `detect_claude_capabilities()` returns empty dict, `build_claude_cmd()` skips the flag, everything works exactly as before.

**Mitigation:** Phase 5's feature gate ensures zero regression. The `AgentMode` enum and task-type mapping are still valuable as documentation of intent and future-proofing.

### Low Risk: Output format changes break AD_RESULT parsing

**Likelihood:** Low (Anthropic maintains backward compatibility)

**Impact:** Medium -- workers could silently fail result extraction

**Mitigation:** `parse_research_result()` falls back to `parse_mc_result()`. The existing `validate_mc_result()` degraded parsing already handles partial/unexpected fields. New schema fields all have defaults.

### Low Risk: Version skew across environments

**Likelihood:** Medium (different machines may have different Claude Code versions)

**Impact:** Low -- capability detection runs per-process and caches

**Mitigation:** `detect_claude_capabilities()` probes the actual binary at runtime. No hardcoded version assumptions.

### Medium Risk: Planner generates invalid agent_mode values

**Likelihood:** Medium (LLM output is inherently unpredictable)

**Impact:** Low -- invalid mode falls back to `AgentMode.CODE`

**Mitigation:** Validate `agent_mode` in decision parsing within `swarm/controller.py`. Log a warning for unrecognized values and default to CODE. Add the valid values to the planner prompt (Phase 3.3).

### Not a risk: Scope creep

This spec is deliberately structured as a research-first implementation. Phase 1 produces a document, not code. Phases 2-5 are conditional on Phase 1 findings. If the audit reveals no actionable CLI changes, the deliverable is the audit document plus the future-proofing infrastructure (enum, mapping, capability detection) which has near-zero maintenance cost.