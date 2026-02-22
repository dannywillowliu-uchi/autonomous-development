"""Runtime tool synthesis module for mission-control.

Workers can create project-specific helper tools (custom linters, test generators,
analyzers) that persist for the duration of the mission. Tools are simple Python
scripts stored in a configurable directory.
"""

from __future__ import annotations

import ast
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mission_control.config import MissionConfig


TOOL_REFLECTION_PROMPT = """\
## Tool Reflection Checkpoint

Before continuing, reflect on your current task and consider:

1. Are there repetitive operations you're performing that could be automated?
2. Would a custom helper script (linter, test generator, analyzer, formatter) accelerate your work?
3. Could a reusable tool benefit other workers tackling similar tasks in this project?

If YES to any of the above, create a Python script that:
- Has a clear single purpose
- Accepts arguments via sys.argv or stdin
- Prints output to stdout
- Can be run standalone with `python <script>`

Register the tool by including in your MC_RESULT handoff:
  "tools_created": [{"name": "<tool-name>", "description": "<what it does>", "script": "<full script content>"}]

If NO, proceed with the task directly. Do not force tool creation when simple inline code suffices.
"""


@dataclass
class ToolEntry:
	"""Metadata for a registered tool."""

	name: str
	description: str
	script_path: Path


class ToolRegistry:
	"""Manages tool persistence -- tools are Python scripts stored in a configurable directory."""

	BLOCKED_MODULES: set[str] = {
		"os", "subprocess", "shutil", "pathlib",
		"socket", "http", "urllib", "requests",
		"ctypes", "importlib",
	}

	BLOCKED_CALLS: set[str] = {"exec", "eval", "compile", "__import__"}

	def __init__(self, workspace: Path, config: MissionConfig) -> None:
		self._workspace = workspace
		self._config = config
		self._tools_dir = workspace / config.tool_synthesis.tools_dir
		self._tools: dict[str, ToolEntry] = {}
		self._load_existing()

	def _load_existing(self) -> None:
		"""Load any tools already on disk from a previous session."""
		if not self._tools_dir.is_dir():
			return
		for script_path in sorted(self._tools_dir.glob("*.py")):
			name = script_path.stem
			description = ""
			first_line = script_path.read_text().split("\n", 1)[0]
			if first_line.startswith("#"):
				description = first_line.lstrip("# ").strip()
			self._tools[name] = ToolEntry(name=name, description=description, script_path=script_path)

	@staticmethod
	def _validate_script_content_static(content: str) -> None:
		"""Validate script content against blocked imports and calls (static).

		Raises:
			ValueError: If the script contains blocked imports or function calls.
		"""
		try:
			tree = ast.parse(content)
		except SyntaxError:
			return  # Let Python's own error handling deal with syntax errors

		for node in ast.walk(tree):
			if isinstance(node, ast.Import):
				for alias in node.names:
					top_module = alias.name.split(".")[0]
					if top_module in ToolRegistry.BLOCKED_MODULES:
						raise ValueError(
							f"Blocked import: '{alias.name}' (module '{top_module}' is not allowed)"
						)
			elif isinstance(node, ast.ImportFrom):
				if node.module:
					top_module = node.module.split(".")[0]
					if top_module in ToolRegistry.BLOCKED_MODULES:
						raise ValueError(
							f"Blocked import: 'from {node.module} import ...' "
							f"(module '{top_module}' is not allowed)"
						)
			elif isinstance(node, ast.Call):
				func_name = None
				if isinstance(node.func, ast.Name):
					func_name = node.func.id
				elif isinstance(node.func, ast.Attribute):
					func_name = node.func.attr
				if func_name and func_name in ToolRegistry.BLOCKED_CALLS:
					raise ValueError(
						f"Blocked function call: '{func_name}()' is not allowed"
					)

	def _validate_script_content(self, content: str) -> None:
		"""Validate script content against blocked imports and calls."""
		self._validate_script_content_static(content)

	def register_tool(self, name: str, script_content: str, description: str = "") -> ToolEntry:
		"""Register a new tool by writing a Python script to the tools directory.

		Args:
			name: Tool name (used as filename stem, must be a valid identifier).
			script_content: Full Python script content.
			description: Human-readable description of what the tool does.

		Returns:
			The created ToolEntry.

		Raises:
			ValueError: If name is empty or not a valid Python identifier.
			ValueError: If script contains blocked imports or function calls.
		"""
		if not name or not name.replace("-", "_").replace("_", "a").isidentifier():
			raise ValueError(f"Invalid tool name: {name!r}")

		self._validate_script_content(script_content)

		self._tools_dir.mkdir(parents=True, exist_ok=True)
		script_path = self._tools_dir / f"{name}.py"

		if description and not script_content.startswith(f"# {description}"):
			content = f"# {description}\n{script_content}"
		else:
			content = script_content
		script_path.write_text(content)

		entry = ToolEntry(name=name, description=description, script_path=script_path)
		self._tools[name] = entry
		return entry

	def get_tool(self, name: str) -> ToolEntry | None:
		"""Retrieve a registered tool by name.

		Args:
			name: Tool name.

		Returns:
			The ToolEntry if found, None otherwise.
		"""
		return self._tools.get(name)

	def list_tools(self) -> list[ToolEntry]:
		"""List all registered tools.

		Returns:
			List of ToolEntry objects sorted by name.
		"""
		return sorted(self._tools.values(), key=lambda t: t.name)

	def cleanup_all(self) -> None:
		"""Remove all registered tools and delete the tools directory."""
		if self._tools_dir.is_dir():
			shutil.rmtree(self._tools_dir)
		self._tools.clear()


def render_tool_reflection_section(registry: ToolRegistry) -> str:
	"""Build the reflection prompt section including listing any existing tools.

	Args:
		registry: The ToolRegistry to query for existing tools.

	Returns:
		Formatted prompt section string.
	"""
	section = TOOL_REFLECTION_PROMPT
	tools = registry.list_tools()
	if tools:
		section += "\n### Existing Project Tools\n\n"
		section += "The following tools have been created by previous workers and are available for reuse:\n\n"
		for tool in tools:
			desc = f" -- {tool.description}" if tool.description else ""
			section += f"- `{tool.name}`{desc} (path: {tool.script_path})\n"
		section += "\nConsider using these before creating new ones.\n"
	return section


def render_available_tools_section(registry: ToolRegistry) -> str:
	"""Format registered tools for injection into worker prompts.

	Args:
		registry: The ToolRegistry to query.

	Returns:
		Formatted tools section string, or empty string if no tools exist.
	"""
	tools = registry.list_tools()
	if not tools:
		return ""

	lines = ["## Available Project Tools\n"]
	lines.append("These helper tools were created during this mission and are available for use:\n")
	for tool in tools:
		lines.append(f"### {tool.name}")
		if tool.description:
			lines.append(f"{tool.description}")
		lines.append(f"Run: `python {tool.script_path}`")
		lines.append("")
	return "\n".join(lines)


def promote_to_mcp_registry(
	tool: ToolEntry,
	quality_score: float,
	mission_id: str,
	registry: object,
) -> object | None:
	"""Bridge function: promote a ToolEntry to the MCP registry.

	Reads script content from tool.script_path and calls registry.register().
	Returns the MCPToolEntry if promoted, None if below threshold.
	The registry param is typed as object to avoid circular imports.
	"""
	from mission_control.mcp_registry import MCPToolRegistry

	assert isinstance(registry, MCPToolRegistry)
	try:
		script_content = tool.script_path.read_text()
	except OSError:
		return None
	return registry.register(
		name=tool.name,
		description=tool.description,
		script_content=script_content,
		quality_score=quality_score,
		mission_id=mission_id,
	)
