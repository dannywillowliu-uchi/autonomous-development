"""Path traversal protection utilities."""

from __future__ import annotations

from pathlib import Path


def validate_config_path(path: str, allowed_bases: list[Path]) -> Path:
	"""Validate that a path resolves within one of the allowed base directories.

	Resolves symlinks and '..' components, then checks containment.

	Args:
		path: The path string to validate.
		allowed_bases: List of allowed base directories.

	Returns:
		The resolved Path if valid.

	Raises:
		ValueError: If the path is empty, contains null bytes, or resolves
			outside all allowed base directories.
	"""
	if not path:
		raise ValueError("Path validation failed: empty path")

	if "\x00" in path:
		raise ValueError("Path validation failed: invalid path")

	resolved = Path(path).resolve()

	for base in allowed_bases:
		if resolved.is_relative_to(base.resolve()):
			return resolved

	raise ValueError("Path validation failed: path outside allowed directories")
