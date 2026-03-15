"""Enhanced worker prompt builder for swarm mode.

Constructs worker prompts with full A2A communication instructions,
peer discovery, task claiming, skill creation, and inbox usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from autodev.swarm.models import AgentStatus, SwarmAgent, SwarmTask, TaskStatus

if TYPE_CHECKING:
	from autodev.config import MissionConfig, SwarmConfig
	from autodev.swarm.capabilities import CapabilityManifest


def build_worker_prompt(
	agent: SwarmAgent,
	task_prompt: str,
	team_name: str,
	agents: list[SwarmAgent],
	tasks: list[SwarmTask],
	config: MissionConfig,
	swarm_config: SwarmConfig,
	capabilities: CapabilityManifest | None = None,
) -> str:
	"""Build a full worker prompt with swarm communication context."""
	sections = [task_prompt]
	sections.append(_identity_section(agent, team_name))
	sections.append(_peer_section(agent, agents))
	sections.append(_task_pool_section(tasks))
	sections.append(_inbox_section(agent, team_name))
	sections.append(_file_conflict_section(agent, agents, tasks))
	sections.append(_skills_section(config))
	sections.append(_verification_section(config))
	sections.append(_capabilities_section(capabilities))
	sections.append(_auth_request_section(agent, team_name))
	sections.append(_skill_creation_section())
	sections.append(_mcp_section(swarm_config))
	sections.append(_result_protocol_section())
	return "\n\n".join(s for s in sections if s)


def _identity_section(agent: SwarmAgent, team_name: str) -> str:
	return f"""## Your Identity

You are agent **{agent.name}** (role: {agent.role.value}) in the autodev swarm.
Team: {team_name} | Agent ID: {agent.id}"""


def _peer_section(agent: SwarmAgent, agents: list[SwarmAgent]) -> str:
	peers = [
		a for a in agents
		if a.id != agent.id and a.status in (AgentStatus.WORKING, AgentStatus.IDLE)
	]
	if not peers:
		return "## Peers\nNo other agents active right now."

	lines = ["## Peers\n"]
	lines.append("These agents are currently active. You can message them via their inbox:")
	for p in peers:
		task_info = f", working on task {p.current_task_id}" if p.current_task_id else ""
		lines.append(f"- **{p.name}** [{p.role.value}]{task_info}")
	lines.append("")
	lines.append("To message a peer, write JSON to their inbox file:")
	lines.append(f'`~/.claude/teams/{agent.id.split("-")[0]}/.../inboxes/<peer-name>.json`')
	lines.append("Or include the message in your AD_RESULT discoveries for the planner to relay.")
	return "\n".join(lines)


def _task_pool_section(tasks: list[SwarmTask]) -> str:
	pending = [t for t in tasks if t.status == TaskStatus.PENDING]
	if not pending:
		return ""

	lines = ["## Unclaimed Tasks\n"]
	lines.append("These tasks are available if you finish early or see a dependency:")
	for t in pending[:5]:
		deps = f" (blocked by: {', '.join(t.depends_on)})" if t.depends_on else ""
		lines.append(f"- [{t.priority.name}] {t.title}{deps}")
	if len(pending) > 5:
		lines.append(f"- ... and {len(pending) - 5} more")
	return "\n".join(lines)


def _inbox_section(agent: SwarmAgent, team_name: str) -> str:
	return f"""## Inbox Communication

You can communicate with the planner and peers via the team inbox system.

**Your inbox**: `~/.claude/teams/{team_name}/inboxes/{agent.name}.json`
**Planner inbox**: `~/.claude/teams/{team_name}/inboxes/team-lead.json`

To send a message to the planner:
1. Read the current contents of `team-lead.json`
2. Append your message object (see format below)
3. Write the updated array back

### Structured Progress Reports

When reporting progress, use this structured format so the planner can parse your status:

```json
{{
  "from": "{agent.name}",
  "type": "report",
  "status": "working|blocked|completed",
  "progress": "Brief description of current work",
  "files_changed": ["src/foo.py", "tests/test_foo.py"],
  "tests_passing": 42,
  "error": "Only set when status is blocked -- describe the blocker",
  "text": "Human-readable summary (always include this)"
}}
```

**Required fields**: `from`, `type`, `text`
**Structured fields** (include when applicable):
- `status`: One of `working`, `blocked`, or `completed`
- `progress`: Brief description of what you're doing right now
- `files_changed`: List of files you've modified so far
- `tests_passing`: Number of tests passing (if you've run tests)
- `error`: Description of what's blocking you (when status is `blocked`)

### Other Message Types

- `"type": "question"` -- Ask the planner for guidance
- `"type": "discovery"` -- Share findings other agents should know
- `"type": "blocked"` -- Signal you're stuck and need help (include `error` field)

**IMPORTANT: You MUST report progress to the planner.** The planner cannot see your \
work until you finish. Send a "report" message to `team-lead.json` at these points:
1. When you've diagnosed the root cause of the problem
2. When you start implementing a fix
3. If you hit a blocker or unexpected complexity
4. When you've completed your task (before emitting AD_RESULT)

Without these reports, the planner has no visibility into your progress and may \
incorrectly assume you're stuck."""


def _file_conflict_section(
	agent: SwarmAgent,
	agents: list[SwarmAgent],
	tasks: list[SwarmTask],
) -> str:
	in_flight: list[str] = []
	task_map = {t.id: t for t in tasks}
	for a in agents:
		if a.id == agent.id or a.status != AgentStatus.WORKING:
			continue
		if a.current_task_id and a.current_task_id in task_map:
			in_flight.extend(task_map[a.current_task_id].files_hint)
	in_flight = sorted(set(in_flight))

	if not in_flight:
		return ""

	lines = ["## Files to Avoid\n"]
	lines.append("Other agents are working on these files. Do NOT modify them:")
	for f in in_flight:
		lines.append(f"- {f}")
	return "\n".join(lines)


def _skills_section(config: MissionConfig) -> str:
	skills_dir = Path(config.target.resolved_path) / ".claude" / "skills"
	if not skills_dir.exists():
		return ""

	skills = [
		d.name for d in skills_dir.iterdir()
		if d.is_dir() and (d / "SKILL.md").exists()
	]
	if not skills:
		return ""

	lines = ["## Available Skills\n"]
	lines.append("These skills have been created and are available via `/<skill-name>`:")
	for s in skills:
		lines.append(f"- /{s}")
	return "\n".join(lines)


def _verification_section(config: MissionConfig) -> str:
	"""Return a self-verification instruction section if a verification command is configured."""
	verification = getattr(config.target, "verification", None)
	if verification is None:
		return ""
	command = getattr(verification, "command", None)
	if not command:
		return ""
	return f"""## Self-Verification

Before reporting task completion, run the project verification command \
to confirm your changes don't break existing tests or linting:

```
{command}
```

If any tests fail or linting errors appear, fix them before emitting \
AD_RESULT. Do not report completion with failing tests."""


def _capabilities_section(capabilities: CapabilityManifest | None) -> str:
	if not capabilities:
		return ""

	lines: list[str] = []
	if capabilities.skills:
		if not lines:
			lines.append("## Available Capabilities\n")
		lines.append("**Skills:**")
		for s in capabilities.skills:
			desc = f" -- {s.description}" if s.description else ""
			lines.append(f"- `{s.invocation}`{desc}")
	if capabilities.agents:
		if not lines:
			lines.append("## Available Capabilities\n")
		lines.append("**Agent Definitions:**")
		for a in capabilities.agents:
			desc = f" -- {a.description}" if a.description else ""
			lines.append(f"- {a.name}{desc}")
	if capabilities.mcp_servers:
		if not lines:
			lines.append("## Available Capabilities\n")
		lines.append("**MCP Servers:**")
		for m in capabilities.mcp_servers:
			lines.append(f"- {m.name} [{m.server_type}]")

	return "\n".join(lines) if lines else ""


def _auth_request_section(agent: SwarmAgent, team_name: str) -> str:
	return f"""## Authentication

If you encounter an auth wall (OAuth screen, API key required, CLI login needed), \
send an auth request to the planner via the team inbox:

1. Write to `~/.claude/teams/{team_name}/inboxes/team-lead.json`:
   ```json
   {{"from": "{agent.name}", "type": "auth_request", \
"service": "service-name", "url": "https://auth-url", \
"purpose": "why you need access"}}
   ```
2. Poll your inbox (`~/.claude/teams/{team_name}/inboxes/{agent.name}.json`) \
for a response with `"type": "auth_response"`
3. The gateway will handle OAuth flows, credential storage, and ask Danny for help if stuck
4. Credentials are stored in macOS Keychain for future use"""


def _skill_creation_section() -> str:
	return """## Creating Skills for Peers

If you build a reusable workflow, create a skill so other agents can use it:

1. Create `.claude/skills/<name>/SKILL.md` with frontmatter:
   ```
   ---
   name: <skill-name>
   description: <what it does>
   ---
   <instructions>
   ```
2. Add supporting files in the same directory if needed
3. Include the skill name in your AD_RESULT discoveries"""


def _mcp_section(swarm_config: SwarmConfig) -> str:
	if not swarm_config.inherit_global_mcps:
		return ""

	lines = ["## MCP Tools\n"]
	lines.append("You have access to global MCP tools. Use them when helpful:")
	lines.append("- **browser-use / claude-in-chrome**: Browser automation")
	lines.append("- **nanobanana**: Image generation via Gemini")
	lines.append("- **obsidian**: Knowledge vault read/write")
	lines.append("- **documentation-sync**: Library documentation search")
	lines.append("- **stitch**: UI design comps")

	if swarm_config.allowed_mcps:
		lines.append(f"\nAllowed MCP servers: {', '.join(swarm_config.allowed_mcps)}")

	return "\n".join(lines)


def _result_protocol_section() -> str:
	return """## Reporting Results

When done, emit your result as the LAST line of output:
```
AD_RESULT:{"status":"completed|failed|blocked","commits":[],"summary":"...","files_changed":[],"discoveries":[],"concerns":[]}
```

- **discoveries**: Findings other agents should know (root causes, patterns, insights)
- **concerns**: Issues the planner should address (blockers, risks, design questions)
- **files_changed**: All files you modified (for conflict tracking)"""
