# autodev Program

## Identity
Autonomous development framework. Spawns parallel Claude Code agents in swarm mode.
Substrate: Claude Code CLI with --permission-mode auto.

## Danny's Goals
Building agent systems to tackle problems like agentic GPU kernel optimization
and building tools for frontier labs. Expanding what agents can autonomously do
is the priority.

## What To Integrate
- New CLIs, SDKs, APIs that agents can use as tools
- MCP servers that add capabilities (browser, database, cloud services)
- Scheduling, orchestration, coordination patterns
- Auth and credential management improvements
- Anything that expands the surface area of what agents can do autonomously
- Developer tooling that improves agent output quality
- Monitoring, observability, debugging tools for agent systems

## What To Skip
- Game engines, mobile-only frameworks, frontend-only libraries
- Academic papers with no practical code or implementation
- Things autodev already has (check architecture section)
- Marginal improvements to existing capabilities
- Tools that only work with non-Claude LLMs

## Resource Context
Danny is on the Claude Max plan. Swarm runs cost compute but not API dollars.
Be aggressive on high-value integrations. The cost of missing something useful
is higher than the cost of trying something that doesn't work out (ratchet
handles rollback).
