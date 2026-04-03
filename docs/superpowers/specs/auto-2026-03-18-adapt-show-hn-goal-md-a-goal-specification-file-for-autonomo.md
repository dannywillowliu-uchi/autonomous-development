The spec is ready. Here's a summary of what it covers:

**GOAL.md Integration Spec** -- Adapts the [goal-md](https://github.com/jmilinovich/goal-md) pattern (constructed fitness functions for autonomous coding agents) into autodev's planner and controller architecture.

**Core idea:** Instead of free-form string objectives, agents work against a measurable composite score (e.g., "routing confidence = health + accuracy + coverage + consistency"). They measure before/after each change, revert on regression, prioritize by impact, and auto-stop when the score target is hit.

**7 change areas:**

1. **`src/autodev/goal.py`** -- New module: GOAL.md parser, fitness function runner, iteration log, stopping condition checker, action ranker
2. **`src/autodev/config.py`** -- `GoalConfig` dataclass with auto-detection (drop a GOAL.md in the target project = auto-enable)
3. **Swarm integration** -- Goal context in SwarmState, score rendering for planner, stopping conditions in planner loop, system prompt additions
4. **Mission integration** -- Per-epoch fitness measurement, auto-revert on regression, planner context injection, goal-based stopping
5. **Worker integration** -- Fitness command + constraints + catalog action injected into worker prompts
6. **CLI** -- `--goal` flag and `autodev goal-status` subcommand
7. **TUI** -- Fitness score panel with component bars and trend sparkline

**17 tests** covering parser, fitness function execution, stopping conditions, iteration log, auto-detection, revert, and end-to-end swarm stopping.

**7 risks assessed** with mitigations: execution safety, parser fragility, flaky scores, context bloat, dual-score complexity, verification interaction, revert reliability.

Shall I save the spec file?