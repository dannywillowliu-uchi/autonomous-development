"""Swarm TUI dashboard -- rich.live-based terminal dashboard for monitoring swarm state.

Usage: python -m autodev.swarm.tui [path-to-project]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

STATUS_COLORS = {
	"working": "green",
	"spawning": "yellow",
	"idle": "cyan",
	"blocked": "red",
	"dead": "dim",
	"shutting_down": "dim",
}

TASK_COLORS = {
	"pending": "yellow",
	"claimed": "cyan",
	"in_progress": "green",
	"completed": "dim green",
	"failed": "red",
	"blocked": "red",
	"cancelled": "dim",
}

PRIORITY_ICONS = {
	"LOW": " ",
	"NORMAL": " ",
	"HIGH": "!",
	"CRITICAL": "!!",
}

REPORT_TYPE_STYLES = {
	"report": "cyan",
	"discovery": "green",
	"blocked": "red",
	"question": "yellow",
}


def _load_state(state_path: Path) -> dict | None:
	try:
		if state_path.exists():
			return json.loads(state_path.read_text())
	except (json.JSONDecodeError, OSError):
		pass
	return None


def _load_inbox_messages(team_name: str | None) -> list[dict]:
	"""Read all messages from team inboxes, sorted by timestamp."""
	if not team_name:
		return []
	inbox_dir = Path.home() / ".claude" / "teams" / team_name / "inboxes"
	if not inbox_dir.exists():
		return []

	messages = []
	for inbox_file in inbox_dir.glob("*.json"):
		try:
			items = json.loads(inbox_file.read_text())
			for msg in items:
				if not msg.get("from"):
					msg["from"] = inbox_file.stem
				messages.append(msg)
		except (json.JSONDecodeError, OSError):
			pass

	messages.sort(key=lambda m: str(m.get("timestamp", "")), reverse=True)
	return messages


def _build_header(data: dict) -> Panel:
	cycle = data.get("cycle", 0)
	mission = data.get("mission", "?")
	cost = data.get("cost_usd", 0)
	wall = data.get("wall_minutes", 0)
	ct = data.get("core_tests", {})
	agents = data.get("agents", [])
	tasks = data.get("tasks", [])

	alive = sum(1 for a in agents if a["status"] not in ("dead",))
	active_tasks = sum(1 for t in tasks if t["status"] not in ("completed", "cancelled"))
	done_tasks = sum(1 for t in tasks if t["status"] == "completed")

	header_text = Text()
	header_text.append("AUTODEV SWARM", style="bold white")
	header_text.append(f"  |  Cycle {cycle}", style="cyan")
	header_text.append(f"  |  ${cost:.2f}", style="yellow")
	header_text.append(f"  |  {wall:.1f}min", style="dim")
	header_text.append(f"  |  {alive} agents", style="green")
	header_text.append(f"  |  {done_tasks}/{active_tasks + done_tasks} tasks", style="cyan")

	if ct:
		p = ct.get("pass", "?")
		f = ct.get("fail", "?")
		t = ct.get("total", "?")
		color = "green" if f == 0 else "red"
		header_text.append("  |  Tests: ", style="dim")
		header_text.append(f"{p}/{t} pass", style=color)

	header_text.append(f"\n{mission}", style="dim")

	return Panel(header_text, border_style="blue", padding=(0, 1))


def _build_agents_table(agents: list[dict]) -> Panel:
	table = Table(expand=True, show_header=True, header_style="bold", box=None, padding=(0, 1))
	table.add_column("Agent", style="bold", ratio=3)
	table.add_column("Role", ratio=1)
	table.add_column("Status", ratio=1)
	table.add_column("Task", ratio=1)
	table.add_column("Done", justify="right", ratio=1)
	table.add_column("Fail", justify="right", ratio=1)

	alive = [a for a in agents if a["status"] not in ("dead",)]
	dead_count = len(agents) - len(alive)

	for a in alive:
		status_color = STATUS_COLORS.get(a["status"], "white")
		table.add_row(
			a["name"],
			a["role"],
			Text(a["status"], style=status_color),
			a.get("task_id") or "-",
			str(a.get("completed", 0)),
			str(a.get("failed", 0)),
		)

	if dead_count > 0:
		table.add_row(
			Text(f"({dead_count} dead)", style="dim"),
			"", "", "", "", "",
		)

	title = f"Agents ({len(alive)} active)"
	return Panel(table, title=title, border_style="green", padding=(0, 0))


def _build_tasks_table(tasks: list[dict]) -> Panel:
	table = Table(expand=True, show_header=True, header_style="bold", box=None, padding=(0, 1))
	table.add_column("Pri", width=2)
	table.add_column("Task", ratio=4)
	table.add_column("Status", ratio=1)
	table.add_column("Claimed", ratio=1)
	table.add_column("Att", justify="right", width=5)

	active_tasks = [t for t in tasks if t["status"] not in ("completed", "cancelled")]
	done_tasks = [t for t in tasks if t["status"] in ("completed", "cancelled")]

	for t in active_tasks:
		color = TASK_COLORS.get(t["status"], "white")
		pri = PRIORITY_ICONS.get(t.get("priority", "NORMAL"), " ")
		table.add_row(
			Text(pri, style="red bold" if pri == "!!" else "yellow" if pri == "!" else ""),
			t["title"],
			Text(t["status"], style=color),
			t.get("claimed_by") or "-",
			f"{t.get('attempts', 0)}/{t.get('max_attempts', 3)}",
		)

	if done_tasks:
		completed = sum(1 for t in done_tasks if t["status"] == "completed")
		table.add_row(
			"", Text(f"({completed} completed, {len(done_tasks) - completed} cancelled)", style="dim"),
			"", "", "",
		)

	title = f"Tasks ({len(active_tasks)} active, {len(done_tasks)} done)"
	return Panel(table, title=title, border_style="yellow", padding=(0, 0))


def _build_activity_feed(inbox_messages: list[dict], log_events: list[dict]) -> Panel:
	"""Combined activity feed: agent reports + planner decisions in reverse chronological order."""
	lines = Text()

	# Interleave planner log entries and inbox messages
	# Show most recent first, limit to ~20 entries
	entries: list[tuple[str, Text]] = []

	# Add planner decisions
	for ev in log_events:
		cycle = ev.get("cycle", "?")
		decisions = ev.get("decisions", [])
		ok = ev.get("ok", 0)
		failed = ev.get("failed", 0)
		reasonings = ev.get("reasonings", [])

		entry = Text()
		entry.append("[planner] ", style="magenta bold")
		entry.append(f"C{cycle} ", style="cyan")
		entry.append(", ".join(decisions[:6]), style="dim")
		if len(decisions) > 6:
			entry.append(f" +{len(decisions) - 6}", style="dim")
		entry.append(f"  {ok}ok", style="green" if ok else "dim")
		if failed:
			entry.append(f" {failed}fail", style="red")
		if reasonings:
			entry.append(f"\n         {reasonings[0][:100]}", style="dim italic")

		sort_key = f"planner-{cycle:04d}"
		entries.append((sort_key, entry))

	# Add agent inbox messages
	for msg in inbox_messages[:30]:
		sender = msg.get("from", "?")
		msg_type = msg.get("type", "?")
		text = msg.get("text", "")
		ts = msg.get("timestamp", "")

		if msg_type == "shutdown_request":
			continue

		style = REPORT_TYPE_STYLES.get(msg_type, "white")
		entry = Text()
		entry.append(f"[{sender}] ", style=f"{style} bold")
		entry.append(f"({msg_type}) ", style=style)
		entry.append(text[:120], style="white")

		sort_key = str(ts) if ts else f"msg-{sender}"
		entries.append((sort_key, entry))

	# Sort reverse chronological, show last 15
	entries.sort(key=lambda x: str(x[0]), reverse=True)
	for _, entry in entries[:15]:
		lines.append_text(entry)
		lines.append("\n")

	if not entries:
		lines.append("No activity yet...", style="dim")

	return Panel(lines, title="Activity Feed", border_style="magenta", padding=(0, 1))


def _score_color(ratio: float, goal_met: bool = False) -> str:
	"""Return color based on score/target ratio."""
	if goal_met or ratio >= 1.0:
		return "green"
	if ratio >= 0.7:
		return "yellow"
	return "red"


def _build_goal_panel(data: dict) -> Panel | None:
	"""Build goal fitness score panel. Returns None if no goal data."""
	goal = data.get("goal")
	if not goal:
		return None

	text = Text()
	composite = goal.get("composite", 0.0)
	target = goal.get("target", 1.0)
	goal_met = goal.get("goal_met", False)

	# Large composite score
	ratio = composite / target if target > 0 else 0
	score_style = f"{_score_color(ratio, goal_met)} bold"
	text.append(f"{composite:.1f}", style=score_style)
	text.append(f" / {target:.1f}", style="dim")
	if goal_met:
		text.append("  MET", style="green bold")
	text.append("\n")

	# Per-component breakdown as bars
	components = goal.get("components", {})
	if components:
		text.append("\n")
		for name, score in components.items():
			comp_ratio = score / target if target > 0 else 0
			bar_color = _score_color(comp_ratio)

			bar_width = 15
			filled = max(0, min(int(comp_ratio * bar_width), bar_width))
			bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

			text.append(f"  {name}", style="bold")
			text.append("\n  ")
			text.append(bar, style=bar_color)
			text.append(f" {score:.1f}\n", style=bar_color)

	# Score trend sparkline
	score_history = goal.get("score_history", [])
	if score_history and len(score_history) > 1:
		text.append("\nTrend\n", style="bold")
		sparkchars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
		mn, mx = min(score_history), max(score_history)
		for v in score_history[-15:]:
			idx = 0 if mx == mn else int((v - mn) / (mx - mn) * (len(sparkchars) - 1))
			text.append(sparkchars[idx], style="cyan")
		text.append(f" ({score_history[-1]:.1f})\n", style="cyan")

	border_color = "green" if goal_met else "yellow"
	title = f"Goal: {goal.get('name', '?')}"
	return Panel(text, title=title, border_style=border_color, padding=(0, 1))


def _build_sidebar(data: dict) -> Panel:
	text = Text()

	# Stagnation
	stags = data.get("stagnation", [])
	if stags:
		text.append("STAGNATION\n", style="red bold")
		for s in stags:
			text.append(f"  {s['metric']}: {s['cycles']} cycles\n", style="red")
			text.append(f"  {s['pivot']}\n", style="dim")
	else:
		text.append("No stagnation\n", style="green")

	# Test history sparkline
	test_hist = data.get("test_history", [])
	if test_hist:
		text.append("\nTest History\n", style="bold")
		mn, mx = min(test_hist), max(test_hist)
		sparkchars = " _.-~*"
		for v in test_hist[-15:]:
			idx = 0 if mx == mn else int((v - mn) / (mx - mn) * (len(sparkchars) - 1))
			text.append(sparkchars[idx], style="green")
		text.append(f" ({test_hist[-1]})\n", style="green")

	# Completion history sparkline
	comp_hist = data.get("completion_history", [])
	if comp_hist:
		text.append("\nCompletions\n", style="bold")
		mn, mx = min(comp_hist), max(comp_hist)
		sparkchars = " _.-~*"
		for v in comp_hist[-15:]:
			idx = 0 if mx == mn else int((v - mn) / (mx - mn) * (len(sparkchars) - 1))
			text.append(sparkchars[idx], style="cyan")
		text.append(f" ({comp_hist[-1]})\n", style="cyan")

	# Discoveries
	discoveries = data.get("discoveries", [])
	if discoveries:
		text.append("\nDiscoveries\n", style="bold")
		for d in discoveries[-5:]:
			text.append(f"  {d[:60]}\n", style="dim")

	return Panel(text, title="Signals", border_style="cyan", padding=(0, 1))


def _build_layout(data: dict, inbox_messages: list[dict]) -> Layout:
	layout = Layout()

	layout.split_column(
		Layout(name="header", size=4),
		Layout(name="body"),
		Layout(name="feed", size=20),
	)

	layout["body"].split_row(
		Layout(name="main", ratio=3),
		Layout(name="sidebar", ratio=1),
	)

	layout["main"].split_column(
		Layout(name="agents", ratio=1),
		Layout(name="tasks", ratio=1),
	)

	# Conditionally add goal panel to sidebar
	goal_panel = _build_goal_panel(data)
	if goal_panel:
		layout["sidebar"].split_column(
			Layout(name="goal", ratio=1),
			Layout(name="signals", ratio=1),
		)
		layout["goal"].update(goal_panel)
		layout["signals"].update(_build_sidebar(data))
	else:
		layout["sidebar"].update(_build_sidebar(data))

	layout["header"].update(_build_header(data))
	layout["agents"].update(_build_agents_table(data.get("agents", [])))
	layout["tasks"].update(_build_tasks_table(data.get("tasks", [])))
	layout["feed"].update(_build_activity_feed(inbox_messages, data.get("log_events", [])))

	return layout


def _waiting_layout(state_path: Path) -> Layout:
	layout = Layout()
	layout.update(Panel(
		Text(f"Waiting for swarm state...\n\n{state_path}\n\nSwarm writes this file each planner cycle.", style="dim"),
		title="AUTODEV SWARM",
		border_style="blue",
	))
	return layout


def main(project_path: str | None = None) -> None:
	if project_path:
		state_path = Path(project_path) / ".autodev-swarm-state.json"
	else:
		state_path = Path.cwd() / ".autodev-swarm-state.json"

	console = Console()

	with Live(console=console, refresh_per_second=2, screen=True) as live:
		try:
			while True:
				data = _load_state(state_path)
				if data:
					team_name = data.get("team_name")
					inbox_messages = _load_inbox_messages(team_name)
					live.update(_build_layout(data, inbox_messages))
				else:
					live.update(_waiting_layout(state_path))
				time.sleep(0.5)
		except KeyboardInterrupt:
			pass


if __name__ == "__main__":
	path = sys.argv[1] if len(sys.argv) > 1 else None
	main(path)
