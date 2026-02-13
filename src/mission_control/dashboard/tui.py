"""Textual TUI dashboard for mission-control monitoring.

Displays live mission status, worker activity, work unit progress,
and recent events from the SQLite database via DashboardProvider.
"""

from __future__ import annotations

import sys
from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from mission_control.dashboard.provider import (
	DashboardProvider,
	DashboardSnapshot,
)

# Worker status indicators
_STATUS_ICON = {
	"working": "[green]\u25cf[/green]",
	"idle": "[dim]\u25cb[/dim]",
	"dead": "[red]\u2717[/red]",
}


def _trunc_id(worker_id: str, length: int = 8) -> str:
	"""Truncate a 12-char hex worker ID to `length` chars."""
	return worker_id[:length]


def _fmt_cost(value: float) -> str:
	"""Format a USD cost value."""
	return f"${value:.2f}"


def _fmt_time(iso_timestamp: str) -> str:
	"""Extract HH:MM:SS from an ISO timestamp."""
	try:
		dt = datetime.fromisoformat(iso_timestamp)
		return dt.strftime("%H:%M:%S")
	except (ValueError, TypeError):
		return "??:??:??"


def _score_text(snap: DashboardSnapshot) -> str:
	"""Build the score display string with delta."""
	if not snap.score_history:
		return "Score: --"
	current = snap.score_history[-1][1]
	if snap.previous_score:
		delta = snap.score_delta
		sign = "+" if delta >= 0 else ""
		return f"Score: {snap.previous_score:.2f} \u2192 {current:.2f} ({sign}{delta:.2f})"
	return f"Score: {current:.2f}"


def _progress_bar_text(pct: float, width: int = 20) -> str:
	"""Render a Unicode block progress bar."""
	filled = int(width * pct / 100)
	empty = width - filled
	return "\u2588" * filled + "\u2591" * empty


class MissionHeader(Static):
	"""Top section: objective, status, phase, cost."""

	def render(self) -> str:
		snap: DashboardSnapshot = self.app.snapshot  # type: ignore[attr-defined]
		if snap.mission is None:
			return (
				"[bold]Objective:[/bold] [dim]No active mission[/dim]\n"
				"[bold]Status:[/bold] [dim]Idle[/dim]"
			)

		mission = snap.mission
		rnd = snap.current_round

		round_info = ""
		if rnd:
			if mission.total_rounds:
				round_info = f" | Round {rnd.number}/{mission.total_rounds}"
			else:
				round_info = f" | Round {rnd.number}"

		score = _score_text(snap)

		status_color = {
			"running": "green",
			"completed": "blue",
			"failed": "red",
			"stalled": "yellow",
			"stopped": "yellow",
			"pending": "dim",
		}.get(mission.status, "white")

		phase_color = {
			"executing": "green",
			"planning": "cyan",
			"evaluating": "yellow",
			"idle": "dim",
		}.get(snap.phase, "white")

		status_line = (
			f"[bold]Status:[/bold] [{status_color}]{mission.status.title()}"
			f"[/{status_color}]{round_info} | {score}"
		)
		cost_line = (
			f"[bold]Phase:[/bold] [{phase_color}]{snap.phase.title()}[/{phase_color}]"
			f" | [bold]Cost:[/bold] {_fmt_cost(snap.round_cost)} (round)"
			f" / {_fmt_cost(snap.total_cost)} (total)"
		)
		lines = [
			f"[bold]Objective:[/bold] {mission.objective}",
			status_line,
			cost_line,
		]
		return "\n".join(lines)


class WorkersPanel(Static):
	"""Left panel: worker list with status indicators."""

	def render(self) -> str:
		snap: DashboardSnapshot = self.app.snapshot  # type: ignore[attr-defined]
		total = len(snap.workers)
		active = snap.workers_active

		header = f"[bold]Workers ({active}/{total} active)[/bold]"
		if not snap.workers:
			return f"{header}\n[dim]No workers[/dim]"

		lines = [header]
		for w in snap.workers:
			icon = _STATUS_ICON.get(w.status, "\u25cb")
			wid = _trunc_id(w.id)
			if w.status == "working" and w.current_unit_title:
				title = w.current_unit_title
				if len(title) > 24:
					title = title[:21] + "..."
				lines.append(f" {icon} {wid} {title}")
			elif w.status == "dead":
				lines.append(f" {icon} {wid} [red]dead[/red]")
			else:
				lines.append(f" {icon} {wid} [dim]idle[/dim]")

		return "\n".join(lines)


class UnitsPanel(Static):
	"""Right panel: work unit summary and progress bar."""

	def render(self) -> str:
		snap: DashboardSnapshot = self.app.snapshot  # type: ignore[attr-defined]

		header = "[bold]Work Units[/bold]"
		if snap.units_total == 0:
			return f"{header}\n[dim]No work units[/dim]"

		pct = snap.completion_pct
		bar = _progress_bar_text(pct)

		lines = [
			header,
			f" Total: {snap.units_total}",
			f" {bar} {pct:.0f}% complete",
			f" Pending: {snap.units_pending} | Running: {snap.units_running}",
			f" Completed: [green]{snap.units_completed}[/green] | Failed: [red]{snap.units_failed}[/red]",
			f" Merge queue: {snap.merge_queue_depth} pending",
		]
		return "\n".join(lines)


class ActivityLog(Static):
	"""Bottom section: recent events."""

	MAX_VISIBLE = 8

	def render(self) -> str:
		snap: DashboardSnapshot = self.app.snapshot  # type: ignore[attr-defined]

		header = "[bold]Recent Activity[/bold]"
		if not snap.recent_events:
			return f"{header}\n [dim]No recent activity[/dim]"

		lines = [header]
		for event in snap.recent_events[: self.MAX_VISIBLE]:
			ts = _fmt_time(event.timestamp)
			color = {
				"completed": "green",
				"merged": "cyan",
				"claimed": "yellow",
				"failed": "red",
			}.get(event.event_type, "white")
			lines.append(f" [{color}]{ts}[/{color}] {event.message}")

		return "\n".join(lines)


class DashboardApp(App):
	"""Mission Control TUI dashboard.

	Monitors missions from the SQLite database via DashboardProvider.
	Auto-refreshes every 2 seconds.
	"""

	TITLE = "Mission Control"

	CSS = """
	Screen {
		layout: vertical;
	}

	#mission-header {
		dock: top;
		height: auto;
		min-height: 3;
		padding: 0 1;
		border-bottom: solid $accent;
	}

	#middle {
		height: 1fr;
		min-height: 6;
	}

	#workers-panel {
		width: 1fr;
		padding: 0 1;
		border-right: solid $accent;
	}

	#units-panel {
		width: 1fr;
		padding: 0 1;
	}

	#activity-log {
		dock: bottom;
		height: auto;
		min-height: 4;
		max-height: 12;
		padding: 0 1;
		border-top: solid $accent;
	}
	"""

	BINDINGS = [
		Binding("q", "quit", "Quit"),
		Binding("r", "refresh", "Refresh"),
	]

	snapshot: reactive[DashboardSnapshot] = reactive(DashboardSnapshot, recompose=False)

	def __init__(self, db_path: str, **kwargs) -> None:
		super().__init__(**kwargs)
		self._provider = DashboardProvider(db_path)

	def compose(self) -> ComposeResult:
		yield Header()
		yield MissionHeader(id="mission-header")
		with Horizontal(id="middle"):
			yield WorkersPanel(id="workers-panel")
			yield UnitsPanel(id="units-panel")
		yield ActivityLog(id="activity-log")
		yield Footer()

	def on_mount(self) -> None:
		"""Start the provider polling and subscribe for updates."""
		self._provider.subscribe(self._on_snapshot)
		self._provider.start_polling(interval=2.0)
		# Immediate first refresh
		self._provider.refresh()

	def on_unmount(self) -> None:
		"""Stop the provider when the app exits."""
		self._provider.stop()

	def _on_snapshot(self, snap: DashboardSnapshot) -> None:
		"""Called from the provider's poll thread -- post to main thread."""
		self.call_from_thread(self._apply_snapshot, snap)

	def _apply_snapshot(self, snap: DashboardSnapshot) -> None:
		"""Apply snapshot on the main thread and refresh widgets."""
		self.snapshot = snap
		self.query_one("#mission-header", MissionHeader).refresh()
		self.query_one("#workers-panel", WorkersPanel).refresh()
		self.query_one("#units-panel", UnitsPanel).refresh()
		self.query_one("#activity-log", ActivityLog).refresh()

	def action_refresh(self) -> None:
		"""Manual refresh via keybinding."""
		self._provider.refresh()


def main(db_path: str | None = None) -> None:
	"""Entry point for the TUI dashboard."""
	path = db_path or (sys.argv[1] if len(sys.argv) > 1 else "mission-control.db")
	app = DashboardApp(db_path=path)
	app.run()


if __name__ == "__main__":
	main()
