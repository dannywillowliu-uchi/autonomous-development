"""Pre-dispatch acceptance criteria validation.

Validates criteria strings (shell commands) before dispatch or execution,
checking for file existence, shell syntax, and path security issues.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from mission_control.path_security import validate_config_path


class Severity(str, Enum):
	"""Severity level for validation issues."""

	error = "error"
	warning = "warning"


@dataclass(frozen=True)
class ValidationIssue:
	"""A single validation issue found in criteria."""

	severity: Severity
	message: str
	criteria_fragment: str


def _check_shell_syntax(criteria_str: str) -> list[ValidationIssue]:
	"""Check for basic shell syntax problems."""
	issues: list[ValidationIssue] = []

	try:
		shlex.split(criteria_str)
	except ValueError as exc:
		issues.append(ValidationIssue(
			severity=Severity.error,
			message=f"Shell syntax error: {exc}",
			criteria_fragment=criteria_str,
		))

	# Check for empty segments around &&
	segments = criteria_str.split("&&")
	for segment in segments:
		stripped = segment.strip()
		if not stripped and len(segments) > 1:
			issues.append(ValidationIssue(
				severity=Severity.error,
				message="Empty command in && chain",
				criteria_fragment=criteria_str.strip(),
			))
			break

	return issues


_PATH_LIKE = re.compile(r"[\w./][\w./-]*")


def _extract_paths_from_command(command: str) -> list[str]:
	"""Extract file/directory path arguments from a shell command."""
	try:
		tokens = shlex.split(command.strip())
	except ValueError:
		return []

	if not tokens:
		return []

	paths: list[str] = []
	skip_next = False
	for i, token in enumerate(tokens):
		if skip_next:
			skip_next = False
			continue
		# Skip the command itself and common flags
		if i == 0:
			continue
		if token.startswith("-"):
			# Flags like -v, -q, --tb=short don't have path args
			# But flags like -m might take a next arg that's not a path
			if token in ("-m", "-k", "-p", "--tb", "--timeout"):
				skip_next = True
			continue
		# Looks like a path if it contains / or starts with a known dir pattern
		if _PATH_LIKE.fullmatch(token) and ("/" in token or "." in token):
			paths.append(token)

	return paths


def _check_path_security(path_str: str, project_root: Path) -> list[ValidationIssue]:
	"""Check a single path for security issues."""
	issues: list[ValidationIssue] = []

	if ".." in path_str.split("/"):
		issues.append(ValidationIssue(
			severity=Severity.error,
			message=f"Directory traversal detected: {path_str}",
			criteria_fragment=path_str,
		))
		return issues

	if Path(path_str).is_absolute():
		try:
			validate_config_path(path_str, [project_root])
		except ValueError:
			issues.append(ValidationIssue(
				severity=Severity.error,
				message=f"Absolute path outside project: {path_str}",
				criteria_fragment=path_str,
			))
		return issues

	return issues


def _check_file_existence(path_str: str, project_root: Path) -> list[ValidationIssue]:
	"""Check if a referenced file/directory exists relative to project_root."""
	issues: list[ValidationIssue] = []

	if Path(path_str).is_absolute():
		return issues

	full_path = project_root / path_str
	if not full_path.exists():
		issues.append(ValidationIssue(
			severity=Severity.error,
			message=f"Referenced path does not exist: {path_str}",
			criteria_fragment=path_str,
		))

	return issues


def validate_criteria(criteria_str: str, project_root: Path) -> list[ValidationIssue]:
	"""Validate acceptance criteria before dispatch.

	Checks shell syntax, file existence, and path security.

	Args:
		criteria_str: Shell command string (may contain && chains).
		project_root: Root directory of the target project.

	Returns:
		List of validation issues found. Empty list means criteria are valid.
	"""
	if not criteria_str or not criteria_str.strip():
		return []

	issues: list[ValidationIssue] = []

	# Shell syntax check on the full string
	issues.extend(_check_shell_syntax(criteria_str))

	# If syntax is broken, skip path extraction (shlex.split would fail)
	if any(i.severity == Severity.error and "Shell syntax error" in i.message for i in issues):
		return issues

	# Validate each command segment independently
	segments = criteria_str.split("&&")
	for segment in segments:
		segment = segment.strip()
		if not segment:
			continue

		paths = _extract_paths_from_command(segment)
		for path_str in paths:
			issues.extend(_check_path_security(path_str, project_root))
			# Only check existence if security passed
			security_errors = [
				i for i in issues
				if i.criteria_fragment == path_str and i.severity == Severity.error
			]
			if not security_errors:
				issues.extend(_check_file_existence(path_str, project_root))

	return issues


def is_criteria_valid(criteria_str: str, project_root: Path) -> bool:
	"""Check if criteria are valid (no errors, warnings OK).

	Args:
		criteria_str: Shell command string (may contain && chains).
		project_root: Root directory of the target project.

	Returns:
		True if no error-level issues found.
	"""
	issues = validate_criteria(criteria_str, project_root)
	return not any(i.severity == Severity.error for i in issues)
