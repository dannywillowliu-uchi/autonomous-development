"""AST-based codebase snapshot builder.

Produces a compact text representation of the project's file tree and public
API signatures, suitable for injecting into LLM prompts to ground them in
the actual codebase structure.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SKIP_DIRS = {"__pycache__", ".venv", ".git", ".tox", ".mypy_cache", ".ruff_cache", "node_modules"}
_SKIP_FILES = {"__init__.py", "conftest.py"}
_SKIP_PREFIXES = ("test_",)

_snapshot_cache: dict[Path, str] = {}


def get_project_snapshot(root: Path) -> str:
	"""Return a cached codebase snapshot for *root*, rebuilding if needed."""
	if root in _snapshot_cache:
		return _snapshot_cache[root]
	result = build_project_snapshot(root)
	_snapshot_cache[root] = result
	return result


def clear_snapshot_cache() -> None:
	"""Flush the snapshot cache (call at epoch boundaries)."""
	_snapshot_cache.clear()


def build_project_snapshot(root: Path) -> str:
	"""Walk *.py* files under *root*, extract signatures, format as full tree."""
	root = root.resolve()
	entries: list[tuple[str, list[str]]] = []

	py_files = sorted(_collect_py_files(root))
	for fpath in py_files:
		rel = str(fpath.relative_to(root))
		sigs = _extract_signatures(fpath)
		entries.append((rel, sigs))

	if not entries:
		return ""

	return _format_full(entries)


def _collect_py_files(root: Path) -> list[Path]:
	"""Recursively collect .py files, skipping excluded dirs/files."""
	results: list[Path] = []
	try:
		children = sorted(root.iterdir())
	except OSError:
		return results
	for child in children:
		if child.is_dir():
			if child.name in _SKIP_DIRS:
				continue
			results.extend(_collect_py_files(child))
		elif child.suffix == ".py":
			if child.name in _SKIP_FILES:
				continue
			if any(child.name.startswith(p) for p in _SKIP_PREFIXES):
				continue
			results.append(child)
	return results


def _extract_signatures(file_path: Path) -> list[str]:
	"""Extract public class and function signatures from a Python file."""
	try:
		source = file_path.read_text(encoding="utf-8", errors="replace")
		tree = ast.parse(source, filename=str(file_path))
	except (SyntaxError, OSError) as exc:
		logger.debug("Could not parse %s: %s", file_path, exc)
		return []

	sigs: list[str] = []
	for node in ast.iter_child_nodes(tree):
		if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
			methods: list[str] = []
			for item in ast.iter_child_nodes(node):
				if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
					if not item.name.startswith("_"):
						methods.append(f"    {_func_sig(item)}")
			sigs.append(f"  class {node.name}:")
			sigs.extend(methods)
		elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
			if not node.name.startswith("_"):
				sigs.append(f"  {_func_sig(node)}")
	return sigs


def _func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
	"""Build a compact one-line function signature."""
	prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
	params = _format_params(node.args)
	ret = ""
	if node.returns:
		try:
			ret = f" -> {ast.unparse(node.returns)}"
		except Exception:
			pass
	return f"{prefix} {node.name}({params}){ret}"


def _format_params(args: ast.arguments) -> str:
	"""Compact param string; abbreviates if >4 params."""
	parts: list[str] = []
	# Positional + regular args (skip 'self'/'cls')
	all_args = args.posonlyargs + args.args
	defaults_offset = len(all_args) - len(args.defaults)

	for i, arg in enumerate(all_args):
		if arg.arg in ("self", "cls"):
			continue
		ann = ""
		if arg.annotation:
			try:
				ann = f": {ast.unparse(arg.annotation)}"
			except Exception:
				pass
		default = ""
		di = i - defaults_offset
		if di >= 0 and di < len(args.defaults):
			try:
				default = f"={ast.unparse(args.defaults[di])}"
			except Exception:
				default = "=..."
		parts.append(f"{arg.arg}{ann}{default}")

	# *args
	if args.vararg:
		parts.append(f"*{args.vararg.arg}")

	# **kwargs
	if args.kwarg:
		parts.append(f"**{args.kwarg.arg}")

	if len(parts) > 4:
		return ", ".join(parts[:2]) + ", ..."
	return ", ".join(parts)


def _format_full(entries: list[tuple[str, list[str]]]) -> str:
	"""Format file tree with all signatures."""
	lines: list[str] = []
	for rel, sigs in entries:
		lines.append(rel)
		lines.extend(sigs)
	return "\n".join(lines)
