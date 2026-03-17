"""Tests for planning/execution separation config fields on SwarmConfig."""

from __future__ import annotations

from pathlib import Path

from autodev.config import SwarmConfig, load_config


def test_planning_config_defaults() -> None:
	"""SwarmConfig planning fields have correct defaults."""
	sc = SwarmConfig()
	assert sc.research_phase_enabled is True
	assert sc.research_timeout_seconds == 300
	assert sc.research_max_agents == 2
	assert sc.plan_refinement_rounds == 1
	assert sc.plan_persistence_enabled is True


def test_planning_config_from_toml(tmp_path: Path) -> None:
	"""[swarm] planning fields are parsed from TOML with non-default values."""
	toml = tmp_path / "autodev.toml"
	toml.write_text("""\
[target]
name = "test"
path = "/tmp/test"

[swarm]
research_phase_enabled = false
research_timeout_seconds = 600
research_max_agents = 4
plan_refinement_rounds = 3
plan_persistence_enabled = false
""")
	cfg = load_config(toml)
	assert cfg.swarm.research_phase_enabled is False
	assert cfg.swarm.research_timeout_seconds == 600
	assert cfg.swarm.research_max_agents == 4
	assert cfg.swarm.plan_refinement_rounds == 3
	assert cfg.swarm.plan_persistence_enabled is False
