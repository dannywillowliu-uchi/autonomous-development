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


def _load_state(state_path: Path) -> dict | None:
	try:
		if state_path.exists():
			return json.loads(state_path.read_text())
	except (json.JSONDecodeError, OSError):
		pass
	return None


def _build_header(data: dict) -> Panel:
	cycle = data.get("cycle", 0)
	mission = data.get("mission", "?")
	cost = data.get("cost_usd", 0)
	wall = data.get("wall_minutes", 0)
	ct = data.get("core_tests", {})

	header_text = Text()
	header_text.append("AUTODEV SWARM", style="bold white")
	header_text.append(f"  |  Cycle {cycle}", style="cyan")
	header_text.append(f"  |  ${cost:.2f}", style="yellow")
	header_text.append(f"  |  {wall:.1f}min", style="dim")

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


def _build_log_panel(log_events: list[dict]) -> Panel:
	lines = Text()
	for ev in log_events[-10:]:
		cycle = ev.get("cycle", "?")
		decisions = ev.get("decisions", [])
		ok = ev.get("ok", 0)
		failed = ev.get("failed", 0)

		lines.append(f"C{cycle} ", style="cyan bold")
		lines.append(", ".join(decisions[:6]), style="dim")
		if len(decisions) > 6:
			lines.append(f" +{len(decisions)-6}", style="dim")
		lines.append(f"  {ok}ok", style="green" if ok else "dim")
		if failed:
			lines.append(f" {failed}fail", style="red")
		lines.append("\n")

		# Show first reasoning
		reasonings = ev.get("reasonings", [])
		if reasonings:
			lines.append(f"  {reasonings[0]}\n", style="dim italic")

	return Panel(lines, title="Planner Log", border_style="magenta", padding=(0, 1))


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

	# Discoveries
	discoveries = data.get("discoveries", [])
	if discoveries:
		text.append("\nDiscoveries\n", style="bold")
		for d in discoveries[-5:]:
			text.append(f"  {d[:60]}\n", style="dim")

	return Panel(text, title="Signals", border_style="cyan", padding=(0, 1))


def _build_layout(data: dict) -> Layout:
	layout = Layout()

	layout.split_column(
		Layout(name="header", size=4),
		Layout(name="body"),
		Layout(name="log", size=14),
	)

	layout["body"].split_row(
		Layout(name="main", ratio=3),
		Layout(name="sidebar", ratio=1),
	)

	layout["main"].split_column(
		Layout(name="agents", ratio=1),
		Layout(name="tasks", ratio=1),
	)

	layout["header"].update(_build_header(data))
	layout["agents"].update(_build_agents_table(data.get("agents", [])))
	layout["tasks"].update(_build_tasks_table(data.get("tasks", [])))
	layout["log"].update(_build_log_panel(data.get("log_events", [])))
	layout["sidebar"].update(_build_sidebar(data))

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
					live.update(_build_layout(data))
				else:
					live.update(_waiting_layout(state_path))
				time.sleep(0.5)
		except KeyboardInterrupt:
			pass


if __name__ == "__main__":
	path = sys.argv[1] if len(sys.argv) > 1 else None
	main(path)
