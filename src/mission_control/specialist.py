"""Specialist template loader for mission-control."""

from __future__ import annotations

from pathlib import Path

from mission_control.config import MissionConfig


def load_specialist_template(name: str, config: MissionConfig) -> str:
	"""Load a specialist template by name from the configured templates directory.

	Args:
		name: Template name without extension (e.g. "test-writer").
		config: Mission config containing specialist.templates_dir.

	Returns:
		The template content as a string.

	Raises:
		FileNotFoundError: If the template file does not exist.
	"""
	templates_dir = Path(config.specialist.templates_dir)
	template_path = templates_dir / f"{name}.md"
	if not template_path.exists():
		raise FileNotFoundError(f"Specialist template not found: {template_path}")
	return template_path.read_text()
