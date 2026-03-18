"""Tests for swarm TUI dashboard goal fitness panel."""

from rich.panel import Panel

from autodev.swarm.tui import (
	_build_goal_panel,
	_build_layout,
	_score_color,
)

# -- Fixtures --

def _make_goal_data(
	composite: float = 72.0,
	target: float = 90.0,
	goal_met: bool = False,
	components: dict | None = None,
	spec_components: list | None = None,
	score_history: list | None = None,
	timestamp: str = "2026-03-18T12:30:00Z",
) -> dict:
	"""Build a state dict with goal data for testing."""
	if components is None:
		components = {"tests": 80.0, "lint": 60.0}
	if spec_components is None:
		spec_components = [
			{"name": "tests", "weight": 0.6, "command": "echo 80"},
			{"name": "lint", "weight": 0.4, "command": "echo 60"},
		]
	if score_history is None:
		score_history = [50.0, 60.0, 65.0, 70.0, 72.0]

	return {
		"goal_spec": {
			"name": "Test Goal",
			"target_score": target,
			"components": spec_components,
		},
		"current_score": {
			"composite": composite,
			"components": components,
			"timestamp": timestamp,
			"success": True,
		},
		"score_history": score_history,
		"goal_met": goal_met,
		"cycle": 5,
		"mission": "Test mission",
		"cost_usd": 1.50,
		"wall_minutes": 10.0,
		"agents": [],
		"tasks": [],
	}


def _make_base_data() -> dict:
	"""Build a minimal state dict without goal data."""
	return {
		"cycle": 1,
		"mission": "No goal mission",
		"cost_usd": 0.0,
		"wall_minutes": 1.0,
		"agents": [],
		"tasks": [],
	}


# -- _score_color tests --

class TestScoreColor:
	def test_green_when_goal_met(self) -> None:
		assert _score_color(0.5, goal_met=True) == "green"

	def test_green_when_ratio_at_target(self) -> None:
		assert _score_color(1.0) == "green"

	def test_green_when_ratio_above_target(self) -> None:
		assert _score_color(1.5) == "green"

	def test_yellow_when_ratio_close(self) -> None:
		assert _score_color(0.8) == "yellow"

	def test_yellow_at_boundary(self) -> None:
		assert _score_color(0.7) == "yellow"

	def test_red_when_far(self) -> None:
		assert _score_color(0.3) == "red"

	def test_red_at_boundary(self) -> None:
		assert _score_color(0.69) == "red"


# -- _build_goal_panel tests --

class TestBuildGoalPanel:
	def test_returns_none_without_goal_data(self) -> None:
		data = _make_base_data()
		assert _build_goal_panel(data) is None

	def test_returns_none_without_current_score(self) -> None:
		data = _make_base_data()
		data["goal_spec"] = {"name": "Test", "target_score": 90}
		assert _build_goal_panel(data) is None

	def test_returns_none_without_goal_spec(self) -> None:
		data = _make_base_data()
		data["current_score"] = {"composite": 72.0}
		assert _build_goal_panel(data) is None

	def test_returns_panel_with_goal_data(self) -> None:
		data = _make_goal_data()
		panel = _build_goal_panel(data)
		assert isinstance(panel, Panel)

	def test_panel_title_contains_goal_name(self) -> None:
		data = _make_goal_data()
		panel = _build_goal_panel(data)
		assert panel is not None
		assert "Test Goal" in str(panel.title)

	def test_panel_shows_composite_score(self) -> None:
		data = _make_goal_data(composite=72.0, target=90.0)
		panel = _build_goal_panel(data)
		assert panel is not None
		rendered = str(panel.renderable)
		assert "72.0" in rendered
		assert "90.0" in rendered

	def test_panel_shows_met_indicator(self) -> None:
		data = _make_goal_data(composite=95.0, target=90.0, goal_met=True)
		panel = _build_goal_panel(data)
		assert panel is not None
		rendered = str(panel.renderable)
		assert "MET" in rendered

	def test_panel_shows_component_names(self) -> None:
		data = _make_goal_data()
		panel = _build_goal_panel(data)
		assert panel is not None
		rendered = str(panel.renderable)
		assert "tests" in rendered
		assert "lint" in rendered

	def test_panel_shows_component_scores(self) -> None:
		data = _make_goal_data()
		panel = _build_goal_panel(data)
		assert panel is not None
		rendered = str(panel.renderable)
		assert "80.0" in rendered
		assert "60.0" in rendered

	def test_panel_shows_timestamp(self) -> None:
		data = _make_goal_data(timestamp="2026-03-18T14:30:00Z")
		panel = _build_goal_panel(data)
		assert panel is not None
		rendered = str(panel.renderable)
		assert "14:30:00" in rendered

	def test_panel_shows_sparkline_trend(self) -> None:
		data = _make_goal_data(score_history=[10.0, 20.0, 30.0, 40.0, 50.0])
		panel = _build_goal_panel(data)
		assert panel is not None
		rendered = str(panel.renderable)
		assert "Trend" in rendered
		assert "50.0" in rendered

	def test_no_sparkline_with_single_history_point(self) -> None:
		data = _make_goal_data(score_history=[72.0])
		panel = _build_goal_panel(data)
		assert panel is not None
		rendered = str(panel.renderable)
		assert "Trend" not in rendered

	def test_handles_zero_target(self) -> None:
		data = _make_goal_data(target=0.0)
		panel = _build_goal_panel(data)
		assert panel is not None  # no crash

	def test_handles_empty_components(self) -> None:
		data = _make_goal_data(components={}, spec_components=[])
		panel = _build_goal_panel(data)
		assert panel is not None

	def test_green_border_when_goal_met(self) -> None:
		data = _make_goal_data(goal_met=True)
		panel = _build_goal_panel(data)
		assert panel is not None
		assert panel.border_style == "green"

	def test_yellow_border_when_goal_not_met(self) -> None:
		data = _make_goal_data(goal_met=False)
		panel = _build_goal_panel(data)
		assert panel is not None
		assert panel.border_style == "yellow"


# -- Layout integration tests --

class TestLayoutGoalIntegration:
	def test_layout_renders_without_goal_data(self) -> None:
		"""Layout should render without crashing when no goal data present."""
		data = _make_base_data()
		layout = _build_layout(data, [])
		# Should have sidebar but no goal sub-layout
		assert layout["sidebar"] is not None

	def test_layout_renders_with_goal_data(self) -> None:
		"""Layout should include goal panel when goal data is present."""
		data = _make_goal_data()
		layout = _build_layout(data, [])
		# Should have goal and signals sub-layouts in sidebar
		assert layout["goal"] is not None
		assert layout["signals"] is not None

	def test_layout_no_goal_sublayout_without_data(self) -> None:
		"""Sidebar should not be split when no goal data."""
		data = _make_base_data()
		layout = _build_layout(data, [])
		# Accessing "goal" should raise KeyError since it doesn't exist
		try:
			_ = layout["goal"]
			has_goal = True
		except KeyError:
			has_goal = False
		assert not has_goal
