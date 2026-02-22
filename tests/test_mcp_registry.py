"""Tests for MCP tool registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_control.config import MCPRegistryConfig, load_config
from mission_control.db import Database
from mission_control.mcp_registry import MCPToolEntry, MCPToolRegistry
from mission_control.tool_synthesis import ToolEntry, promote_to_mcp_registry


@pytest.fixture()
def db() -> Database:
	return Database(":memory:")


@pytest.fixture()
def registry_config() -> MCPRegistryConfig:
	return MCPRegistryConfig(enabled=True, promotion_threshold=0.7, ema_alpha=0.3)


@pytest.fixture()
def registry(db: Database, registry_config: MCPRegistryConfig) -> MCPToolRegistry:
	return MCPToolRegistry(db, registry_config)


SAFE_SCRIPT = """\
# A safe helper script
import sys
import json

data = json.loads(sys.argv[1])
print(json.dumps(data))
"""

UNSAFE_SCRIPT = """\
import os
os.system("rm -rf /")
"""


# -- MCPToolEntry tests --

def test_mcp_tool_entry_defaults() -> None:
	entry = MCPToolEntry()
	assert entry.name == ""
	assert entry.quality_score == 0.5
	assert entry.usage_count == 0
	assert entry.input_schema == "{}"


def test_mcp_tool_entry_fields() -> None:
	entry = MCPToolEntry(
		id="t1", name="my-tool", description="Does things",
		quality_score=0.9, usage_count=5, created_by_mission="m1",
	)
	assert entry.id == "t1"
	assert entry.name == "my-tool"
	assert entry.created_by_mission == "m1"


# -- Register tests --

def test_register_above_threshold(registry: MCPToolRegistry) -> None:
	entry = registry.register("good-tool", "A good tool", SAFE_SCRIPT, 0.8, "mission-1")
	assert entry is not None
	assert entry.name == "good-tool"
	assert entry.quality_score == 0.8


def test_register_below_threshold(registry: MCPToolRegistry, db: Database) -> None:
	result = registry.register("bad-tool", "A bad tool", SAFE_SCRIPT, 0.5, "mission-1")
	assert result is None
	assert db.get_tool_registry_entry("bad-tool") is None


def test_register_at_threshold(registry: MCPToolRegistry) -> None:
	entry = registry.register("edge-tool", "Edge case", SAFE_SCRIPT, 0.7, "mission-1")
	assert entry is not None


def test_register_blocked_script(registry: MCPToolRegistry) -> None:
	with pytest.raises(ValueError, match="Blocked import"):
		registry.register("unsafe-tool", "Unsafe", UNSAFE_SCRIPT, 0.9, "mission-1")


# -- Discover tests --

def test_discover_matching(registry: MCPToolRegistry) -> None:
	registry.register("format-checker", "Checks formatting", SAFE_SCRIPT, 0.8, "m1")
	registry.register("lint-runner", "Runs linting", SAFE_SCRIPT, 0.9, "m1")
	results = registry.discover("format")
	assert len(results) == 1
	assert results[0].name == "format-checker"


def test_discover_no_match(registry: MCPToolRegistry) -> None:
	registry.register("format-checker", "Checks formatting", SAFE_SCRIPT, 0.8, "m1")
	results = registry.discover("nonexistent")
	assert len(results) == 0


def test_discover_limit(registry: MCPToolRegistry) -> None:
	for i in range(5):
		registry.register(f"tool-{i}", f"Tool number {i}", SAFE_SCRIPT, 0.8 + i * 0.01, f"m{i}")
	results = registry.discover("tool", limit=3)
	assert len(results) == 3


def test_discover_ordered_by_quality(registry: MCPToolRegistry) -> None:
	registry.register("low-tool", "Low quality", SAFE_SCRIPT, 0.7, "m1")
	registry.register("high-tool", "High quality", SAFE_SCRIPT, 0.95, "m1")
	results = registry.discover("tool")
	assert results[0].name == "high-tool"
	assert results[1].name == "low-tool"


# -- Record usage tests --

def test_record_usage_increments_count(registry: MCPToolRegistry, db: Database) -> None:
	registry.register("my-tool", "A tool", SAFE_SCRIPT, 0.8, "m1")
	registry.record_usage("my-tool")
	entry = db.get_tool_registry_entry("my-tool")
	assert entry is not None
	assert entry.usage_count == 1


def test_record_usage_updates_quality(registry: MCPToolRegistry, db: Database) -> None:
	registry.register("my-tool", "A tool", SAFE_SCRIPT, 0.8, "m1")
	registry.record_usage("my-tool")
	entry = db.get_tool_registry_entry("my-tool")
	assert entry is not None
	# EMA: 0.3 * 1.0 + 0.7 * 0.8 = 0.86
	assert abs(entry.quality_score - 0.86) < 0.01


def test_record_usage_nonexistent(registry: MCPToolRegistry) -> None:
	# Should not raise
	registry.record_usage("nonexistent-tool")


# -- Update quality tests --

def test_update_quality_ema(registry: MCPToolRegistry, db: Database) -> None:
	registry.register("my-tool", "A tool", SAFE_SCRIPT, 0.8, "m1")
	registry.update_quality("my-tool", 0.6)
	entry = db.get_tool_registry_entry("my-tool")
	assert entry is not None
	# EMA: 0.3 * 0.6 + 0.7 * 0.8 = 0.74
	assert abs(entry.quality_score - 0.74) < 0.01


def test_update_quality_nonexistent(registry: MCPToolRegistry) -> None:
	# Should not raise
	registry.update_quality("nonexistent-tool", 0.5)


# -- Prune tests --

def test_prune_removes_below_threshold(registry: MCPToolRegistry, db: Database) -> None:
	registry.register("good-tool", "Good", SAFE_SCRIPT, 0.9, "m1")
	registry.register("ok-tool", "OK", SAFE_SCRIPT, 0.75, "m1")
	removed = registry.prune(min_quality=0.8)
	assert removed == 1
	assert db.get_tool_registry_entry("ok-tool") is None
	assert db.get_tool_registry_entry("good-tool") is not None


def test_prune_returns_zero_when_nothing_to_remove(registry: MCPToolRegistry) -> None:
	registry.register("good-tool", "Good", SAFE_SCRIPT, 0.9, "m1")
	removed = registry.prune(min_quality=0.5)
	assert removed == 0


# -- Handler path tests --

def test_get_handler_existing(registry: MCPToolRegistry) -> None:
	registry.register("my-tool", "A tool", SAFE_SCRIPT, 0.8, "m1")
	# handler_path is empty by default from register()
	handler = registry.get_handler("my-tool")
	assert handler is None  # empty handler_path returns None


def test_get_handler_nonexistent(registry: MCPToolRegistry) -> None:
	handler = registry.get_handler("nope")
	assert handler is None


# -- Cross-mission persistence --

def test_cross_mission_discovery(db: Database, registry_config: MCPRegistryConfig) -> None:
	registry1 = MCPToolRegistry(db, registry_config)
	registry1.register("shared-tool", "Shared", SAFE_SCRIPT, 0.8, "mission-A")

	registry2 = MCPToolRegistry(db, registry_config)
	results = registry2.discover("shared")
	assert len(results) == 1
	assert results[0].created_by_mission == "mission-A"


# -- Config tests --

def test_mcp_registry_config_defaults() -> None:
	cfg = MCPRegistryConfig()
	assert cfg.enabled is False
	assert cfg.promotion_threshold == 0.7
	assert cfg.ema_alpha == 0.3


def test_mcp_registry_config_toml_parsing(tmp_path: Path) -> None:
	p = tmp_path / "mission-control.toml"
	p.write_text("""\
[target]
name = "test"
path = "."

[mcp_registry]
enabled = true
promotion_threshold = 0.6
ema_alpha = 0.2
""")
	config = load_config(p)
	assert config.mcp_registry.enabled is True
	assert config.mcp_registry.promotion_threshold == 0.6
	assert config.mcp_registry.ema_alpha == 0.2


# -- promote_to_mcp_registry bridge --

def test_promote_to_mcp_registry_success(
	registry: MCPToolRegistry, tmp_path: Path,
) -> None:
	script_path = tmp_path / "my_tool.py"
	script_path.write_text(SAFE_SCRIPT)
	tool = ToolEntry(name="my-tool", description="A tool", script_path=script_path)

	result = promote_to_mcp_registry(tool, 0.8, "mission-1", registry)
	assert result is not None
	assert result.name == "my-tool"


def test_promote_to_mcp_registry_below_threshold(
	registry: MCPToolRegistry, tmp_path: Path,
) -> None:
	script_path = tmp_path / "my_tool.py"
	script_path.write_text(SAFE_SCRIPT)
	tool = ToolEntry(name="my-tool", description="A tool", script_path=script_path)

	result = promote_to_mcp_registry(tool, 0.3, "mission-1", registry)
	assert result is None


def test_promote_to_mcp_registry_blocked_script(
	registry: MCPToolRegistry, tmp_path: Path,
) -> None:
	script_path = tmp_path / "unsafe_tool.py"
	script_path.write_text(UNSAFE_SCRIPT)
	tool = ToolEntry(name="unsafe-tool", description="Unsafe", script_path=script_path)

	with pytest.raises(ValueError, match="Blocked import"):
		promote_to_mcp_registry(tool, 0.9, "mission-1", registry)


def test_promote_to_mcp_registry_missing_file(
	registry: MCPToolRegistry, tmp_path: Path,
) -> None:
	tool = ToolEntry(name="ghost", description="Missing", script_path=tmp_path / "nope.py")
	result = promote_to_mcp_registry(tool, 0.9, "mission-1", registry)
	assert result is None
