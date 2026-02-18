"""Tests for the runtime tool synthesis module."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_control.config import MissionConfig, TargetConfig, ToolSynthesisConfig, VerificationConfig
from mission_control.models import WorkUnit
from mission_control.tool_synthesis import (
	TOOL_REFLECTION_PROMPT,
	ToolEntry,
	ToolRegistry,
	render_available_tools_section,
	render_tool_reflection_section,
)
from mission_control.worker import render_mission_worker_prompt


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
	return tmp_path / "workspace"


@pytest.fixture()
def ts_config() -> ToolSynthesisConfig:
	return ToolSynthesisConfig(enabled=True, tools_dir=".mc-tools")


@pytest.fixture()
def config_with_tools(tmp_path: Path, ts_config: ToolSynthesisConfig) -> MissionConfig:
	cfg = MissionConfig()
	cfg.target = TargetConfig(
		name="test-proj",
		path=str(tmp_path),
		branch="main",
		verification=VerificationConfig(command="pytest -q"),
	)
	cfg.tool_synthesis = ts_config
	return cfg


@pytest.fixture()
def registry(workspace: Path, config_with_tools: MissionConfig) -> ToolRegistry:
	return ToolRegistry(workspace, config_with_tools)


# ── ToolRegistry register/get/list ──────────────────────────────────────


class TestToolRegistryRegister:
	def test_register_creates_file(self, registry: ToolRegistry, workspace: Path) -> None:
		entry = registry.register_tool("my_linter", "print('lint')", description="Custom linter")
		assert entry.name == "my_linter"
		assert entry.description == "Custom linter"
		assert entry.script_path.exists()
		assert entry.script_path.name == "my_linter.py"

	def test_register_prepends_description_comment(self, registry: ToolRegistry) -> None:
		registry.register_tool("checker", "import sys\nprint(sys.argv)", description="Arg checker")
		entry = registry.get_tool("checker")
		assert entry is not None
		content = entry.script_path.read_text()
		assert content.startswith("# Arg checker\n")
		assert "import sys" in content

	def test_register_no_double_description(self, registry: ToolRegistry) -> None:
		script = "# Already has desc\nprint('ok')"
		registry.register_tool("tool", script, description="Already has desc")
		entry = registry.get_tool("tool")
		assert entry is not None
		content = entry.script_path.read_text()
		assert content.count("# Already has desc") == 1

	def test_register_no_description(self, registry: ToolRegistry) -> None:
		registry.register_tool("bare", "print('bare')")
		entry = registry.get_tool("bare")
		assert entry is not None
		assert entry.description == ""
		content = entry.script_path.read_text()
		assert content == "print('bare')"

	def test_register_invalid_name_empty(self, registry: ToolRegistry) -> None:
		with pytest.raises(ValueError, match="Invalid tool name"):
			registry.register_tool("", "print('x')")

	def test_register_invalid_name_special_chars(self, registry: ToolRegistry) -> None:
		with pytest.raises(ValueError, match="Invalid tool name"):
			registry.register_tool("foo bar", "print('x')")

	def test_register_allows_hyphens(self, registry: ToolRegistry) -> None:
		entry = registry.register_tool("my-tool", "print('ok')")
		assert entry.name == "my-tool"

	def test_register_allows_underscores(self, registry: ToolRegistry) -> None:
		entry = registry.register_tool("my_tool", "print('ok')")
		assert entry.name == "my_tool"


class TestToolRegistryGet:
	def test_get_existing(self, registry: ToolRegistry) -> None:
		registry.register_tool("finder", "print('find')")
		entry = registry.get_tool("finder")
		assert entry is not None
		assert entry.name == "finder"

	def test_get_nonexistent(self, registry: ToolRegistry) -> None:
		assert registry.get_tool("nonexistent") is None


class TestToolRegistryList:
	def test_list_empty(self, registry: ToolRegistry) -> None:
		assert registry.list_tools() == []

	def test_list_sorted_by_name(self, registry: ToolRegistry) -> None:
		registry.register_tool("zeta", "print('z')")
		registry.register_tool("alpha", "print('a')")
		registry.register_tool("mid", "print('m')")
		tools = registry.list_tools()
		assert [t.name for t in tools] == ["alpha", "mid", "zeta"]

	def test_list_returns_tool_entries(self, registry: ToolRegistry) -> None:
		registry.register_tool("tool1", "print(1)", description="First tool")
		tools = registry.list_tools()
		assert len(tools) == 1
		assert isinstance(tools[0], ToolEntry)
		assert tools[0].description == "First tool"


# ── Duplicate tool names (overwrite) ────────────────────────────────────


class TestToolRegistryOverwrite:
	def test_duplicate_name_overwrites(self, registry: ToolRegistry) -> None:
		registry.register_tool("dup", "print('v1')", description="Version 1")
		registry.register_tool("dup", "print('v2')", description="Version 2")
		entry = registry.get_tool("dup")
		assert entry is not None
		assert entry.description == "Version 2"
		content = entry.script_path.read_text()
		assert "v2" in content

	def test_duplicate_name_keeps_single_entry(self, registry: ToolRegistry) -> None:
		registry.register_tool("dup", "print('v1')")
		registry.register_tool("dup", "print('v2')")
		assert len(registry.list_tools()) == 1


# ── Filesystem persistence ──────────────────────────────────────────────


class TestToolPersistence:
	def test_tools_dir_auto_created(self, workspace: Path, config_with_tools: MissionConfig) -> None:
		tools_dir = workspace / config_with_tools.tool_synthesis.tools_dir
		assert not tools_dir.exists()
		reg = ToolRegistry(workspace, config_with_tools)
		reg.register_tool("auto", "print('auto')")
		assert tools_dir.is_dir()

	def test_tools_dir_does_not_exist_on_init(self, workspace: Path, config_with_tools: MissionConfig) -> None:
		"""Loading from a non-existent tools_dir should not error."""
		reg = ToolRegistry(workspace, config_with_tools)
		assert reg.list_tools() == []

	def test_reload_existing_tools(self, workspace: Path, config_with_tools: MissionConfig) -> None:
		reg1 = ToolRegistry(workspace, config_with_tools)
		reg1.register_tool("persist_me", "# A persistent tool\nprint('hello')", description="A persistent tool")

		reg2 = ToolRegistry(workspace, config_with_tools)
		tools = reg2.list_tools()
		assert len(tools) == 1
		assert tools[0].name == "persist_me"
		assert tools[0].description == "A persistent tool"

	def test_reload_tool_without_description_comment(self, workspace: Path, config_with_tools: MissionConfig) -> None:
		reg1 = ToolRegistry(workspace, config_with_tools)
		reg1.register_tool("nodesc", "print('no desc')")

		reg2 = ToolRegistry(workspace, config_with_tools)
		entry = reg2.get_tool("nodesc")
		assert entry is not None
		assert entry.description == ""

	def test_reload_multiple_tools_sorted(self, workspace: Path, config_with_tools: MissionConfig) -> None:
		reg1 = ToolRegistry(workspace, config_with_tools)
		reg1.register_tool("beta", "# Beta\nprint('b')", description="Beta")
		reg1.register_tool("alpha", "# Alpha\nprint('a')", description="Alpha")

		reg2 = ToolRegistry(workspace, config_with_tools)
		names = [t.name for t in reg2.list_tools()]
		assert names == ["alpha", "beta"]


# ── Cleanup ─────────────────────────────────────────────────────────────


class TestToolRegistryCleanup:
	def test_cleanup_removes_tools_dir(
		self, registry: ToolRegistry, workspace: Path, config_with_tools: MissionConfig,
	) -> None:
		registry.register_tool("to_clean", "print('clean me')")
		tools_dir = workspace / config_with_tools.tool_synthesis.tools_dir
		assert tools_dir.is_dir()

		registry.cleanup_all()
		assert not tools_dir.exists()
		assert registry.list_tools() == []

	def test_cleanup_clears_internal_dict(self, registry: ToolRegistry) -> None:
		registry.register_tool("a", "print('a')")
		registry.register_tool("b", "print('b')")
		assert len(registry.list_tools()) == 2

		registry.cleanup_all()
		assert registry.list_tools() == []
		assert registry.get_tool("a") is None

	def test_cleanup_nonexistent_directory(self, registry: ToolRegistry) -> None:
		"""Cleanup when tools_dir doesn't exist should not raise."""
		registry.cleanup_all()
		assert registry.list_tools() == []

	def test_cleanup_then_register(self, registry: ToolRegistry) -> None:
		registry.register_tool("old", "print('old')")
		registry.cleanup_all()
		registry.register_tool("new", "print('new')")
		assert len(registry.list_tools()) == 1
		assert registry.get_tool("new") is not None
		assert registry.get_tool("old") is None


# ── render_tool_reflection_section ──────────────────────────────────────


class TestRenderToolReflectionSection:
	def test_no_tools_returns_base_prompt(self, registry: ToolRegistry) -> None:
		section = render_tool_reflection_section(registry)
		assert section == TOOL_REFLECTION_PROMPT
		assert "Existing Project Tools" not in section

	def test_with_tools_includes_listing(self, registry: ToolRegistry) -> None:
		registry.register_tool("my_lint", "print('lint')", description="Custom linter")
		section = render_tool_reflection_section(registry)
		assert TOOL_REFLECTION_PROMPT in section
		assert "### Existing Project Tools" in section
		assert "`my_lint`" in section
		assert "Custom linter" in section
		assert "Consider using these" in section

	def test_with_multiple_tools(self, registry: ToolRegistry) -> None:
		registry.register_tool("alpha", "print('a')", description="Alpha tool")
		registry.register_tool("beta", "print('b')", description="Beta tool")
		section = render_tool_reflection_section(registry)
		assert "`alpha`" in section
		assert "`beta`" in section
		assert "Alpha tool" in section
		assert "Beta tool" in section

	def test_tool_without_description(self, registry: ToolRegistry) -> None:
		registry.register_tool("bare_tool", "print('x')")
		section = render_tool_reflection_section(registry)
		assert "`bare_tool`" in section
		assert " -- " not in section.split("bare_tool")[1].split("\n")[0]


# ── render_available_tools_section ──────────────────────────────────────


class TestRenderAvailableToolsSection:
	def test_no_tools_returns_empty(self, registry: ToolRegistry) -> None:
		assert render_available_tools_section(registry) == ""

	def test_with_tools_returns_section(self, registry: ToolRegistry) -> None:
		registry.register_tool("analyzer", "print('analyze')", description="Code analyzer")
		section = render_available_tools_section(registry)
		assert "## Available Project Tools" in section
		assert "### analyzer" in section
		assert "Code analyzer" in section
		assert "python" in section

	def test_with_multiple_tools(self, registry: ToolRegistry) -> None:
		registry.register_tool("tool_a", "print('a')", description="Tool A")
		registry.register_tool("tool_b", "print('b')", description="Tool B")
		section = render_available_tools_section(registry)
		assert "### tool_a" in section
		assert "### tool_b" in section

	def test_tool_without_description_omits_desc_line(self, registry: ToolRegistry) -> None:
		registry.register_tool("nodesc", "print('x')")
		section = render_available_tools_section(registry)
		assert "### nodesc" in section
		lines = section.split("\n")
		header_idx = next(i for i, line in enumerate(lines) if "### nodesc" in line)
		assert "Run:" in lines[header_idx + 1]

	def test_includes_script_path(self, registry: ToolRegistry) -> None:
		registry.register_tool("pathtool", "print('p')", description="Path test")
		section = render_available_tools_section(registry)
		entry = registry.get_tool("pathtool")
		assert entry is not None
		assert str(entry.script_path) in section


# ── Worker prompt rendering with tool sections ──────────────────────────


class TestWorkerPromptWithToolSections:
	def test_prompt_renders_without_tool_sections(self, config: MissionConfig) -> None:
		"""When tool_synthesis is disabled (default), prompt has no tool sections."""
		unit = WorkUnit(title="Fix bug", description="Fix the bug")
		prompt = render_mission_worker_prompt(
			unit=unit, config=config, workspace_path="/tmp/ws", branch_name="mc/unit-1",
		)
		assert "Fix bug" in prompt
		assert "Available Project Tools" not in prompt
		assert "Tool Reflection Checkpoint" not in prompt

	def test_prompt_with_tools_includes_available_section(
		self, workspace: Path, config_with_tools: MissionConfig,
	) -> None:
		"""When tools exist, render_available_tools_section output can be injected as context."""
		reg = ToolRegistry(workspace, config_with_tools)
		reg.register_tool("helper", "print('help')", description="A helper")
		tools_section = render_available_tools_section(reg)
		unit = WorkUnit(title="Use tools", description="Use existing tools")
		prompt = render_mission_worker_prompt(
			unit=unit, config=config_with_tools,
			workspace_path=str(workspace), branch_name="mc/unit-2",
			context=tools_section,
		)
		assert "Use tools" in prompt
		assert "Available Project Tools" in prompt
		assert "### helper" in prompt

	def test_prompt_with_reflection_section(
		self, workspace: Path, config_with_tools: MissionConfig,
	) -> None:
		"""Tool reflection prompt can be injected as context."""
		reg = ToolRegistry(workspace, config_with_tools)
		reflection = render_tool_reflection_section(reg)
		unit = WorkUnit(title="Reflect", description="Check for tools")
		prompt = render_mission_worker_prompt(
			unit=unit, config=config_with_tools,
			workspace_path=str(workspace), branch_name="mc/unit-3",
			context=reflection,
		)
		assert "Tool Reflection Checkpoint" in prompt


# ── Disabled tool synthesis ─────────────────────────────────────────────


class TestDisabledToolSynthesis:
	def test_disabled_config_defaults(self) -> None:
		cfg = ToolSynthesisConfig()
		assert cfg.enabled is False
		assert cfg.tools_dir == ".mc-tools"

	def test_registry_works_when_disabled(self, workspace: Path) -> None:
		"""Registry is functional even when config says disabled -- it's the caller's choice."""
		cfg = MissionConfig()
		cfg.tool_synthesis = ToolSynthesisConfig(enabled=False)
		reg = ToolRegistry(workspace, cfg)
		assert reg.list_tools() == []

	def test_no_sections_in_prompt_when_disabled(self, config: MissionConfig) -> None:
		"""Default config (disabled) produces no tool-related prompt sections."""
		unit = WorkUnit(title="Normal task", description="Just a normal task")
		prompt = render_mission_worker_prompt(
			unit=unit, config=config, workspace_path="/tmp/ws", branch_name="mc/unit-4",
		)
		assert "Tool Reflection" not in prompt
		assert "Available Project Tools" not in prompt


# ── ToolSynthesisConfig ─────────────────────────────────────────────────


class TestToolSynthesisConfig:
	def test_defaults(self) -> None:
		cfg = ToolSynthesisConfig()
		assert cfg.enabled is False
		assert cfg.tools_dir == ".mc-tools"

	def test_custom_values(self) -> None:
		cfg = ToolSynthesisConfig(enabled=True, tools_dir=".custom-tools")
		assert cfg.enabled is True
		assert cfg.tools_dir == ".custom-tools"

	def test_mission_config_has_tool_synthesis(self) -> None:
		mc = MissionConfig()
		assert hasattr(mc, "tool_synthesis")
		assert isinstance(mc.tool_synthesis, ToolSynthesisConfig)

	def test_custom_tools_dir_used(self, workspace: Path) -> None:
		cfg = MissionConfig()
		cfg.tool_synthesis = ToolSynthesisConfig(enabled=True, tools_dir=".my-tools")
		reg = ToolRegistry(workspace, cfg)
		reg.register_tool("custom_dir", "print('ok')")
		expected_dir = workspace / ".my-tools"
		assert expected_dir.is_dir()
		assert (expected_dir / "custom_dir.py").exists()


# ── ToolEntry dataclass ─────────────────────────────────────────────────


class TestToolEntry:
	def test_fields(self, tmp_path: Path) -> None:
		entry = ToolEntry(name="test", description="A test tool", script_path=tmp_path / "test.py")
		assert entry.name == "test"
		assert entry.description == "A test tool"
		assert entry.script_path == tmp_path / "test.py"
