"""Immutable oracle -- protects critical config files from modification."""

from pathlib import PurePosixPath

PROTECTED_PATTERNS = [
	"pyproject.toml",
	"ruff.toml",
	".ruff.toml",
	"setup.cfg",
	"mypy.ini",
	".mypy.ini",
	"conftest.py",
	"autodev.toml",
]


def check_oracle_violation(changed_files: list[str]) -> list[str]:
	"""Return list of protected files that were modified."""
	violations = []
	for filepath in changed_files:
		basename = PurePosixPath(filepath).name
		if basename in PROTECTED_PATTERNS or filepath in PROTECTED_PATTERNS:
			violations.append(filepath)
	return violations
