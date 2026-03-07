---
name: discover-peers
description: Find and communicate with other agents in the swarm
---

# Discover Peers

Use this to find other agents, check their status, or send them messages.

## Find active peers

Check the team directory for active agents:
```bash
ls ~/.claude/teams/$AUTODEV_TEAM_NAME/inboxes/
```

Each `.json` file corresponds to an agent's inbox. The file name is the agent name.

## Read a peer's recent messages

```bash
cat ~/.claude/teams/$AUTODEV_TEAM_NAME/inboxes/<peer-name>.json
```

## Send a message to a peer

1. Read their inbox file
2. Append your message:
   ```json
   {"from": "<your-name>", "type": "question|info|request", "text": "...", "timestamp": "<ISO-8601>"}
   ```
3. Write back the full array

## Send a message to the planner

Write to `team-lead.json` inbox with type "discovery", "blocked", or "question".

## Environment variables

These are set for every swarm agent:
- `AUTODEV_TEAM_NAME`: Your team name
- `AUTODEV_AGENT_ID`: Your unique ID
- `AUTODEV_AGENT_NAME`: Your display name
- `AUTODEV_AGENT_ROLE`: Your role (implementer, researcher, tester, etc.)
