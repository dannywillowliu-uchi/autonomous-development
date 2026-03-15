"""Shared utilities for the intelligence subsystem."""

from __future__ import annotations

import shutil
from pathlib import Path


def find_claude_binary() -> str:
	"""Resolve the full path to the claude binary."""
	found = shutil.which("claude")
	if found:
		return found
	for candidate in [
		Path.home() / ".local" / "bin" / "claude",
		Path("/usr/local/bin/claude"),
		Path("/opt/homebrew/bin/claude"),
	]:
		if candidate.exists():
			return str(candidate)
	return "claude"
