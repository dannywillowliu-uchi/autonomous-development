---
name: create-tool
description: Create a reusable tool (Python script or Claude Code skill) that persists for the swarm
---

# Create Tool

Use this when you've built something reusable that other agents should have access to.

## Choose the right format

**Python script** (for data processing, analysis, linting, test helpers):
1. Write to `.autodev-tools/<name>.py`
2. Must be runnable standalone: `python .autodev-tools/<name>.py [args]`
3. Use only stdlib imports (no blocked modules: os, subprocess, shutil, socket, http, requests)
4. Include a `# <description>` comment on line 1

**Claude Code skill** (for workflows, multi-step processes, prompt templates):
1. Create `.claude/skills/<name>/SKILL.md` with frontmatter
2. Add supporting files in the same directory if needed
3. Skills are invoked by other agents via `/<name>`

## After creating

Include in your AD_RESULT:
```json
{
  "discoveries": ["Created tool: <name> - <description>"],
  "tools_created": [{"name": "<name>", "type": "script|skill", "description": "<what it does>"}]
}
```

The planner will notify other agents about the new tool.
