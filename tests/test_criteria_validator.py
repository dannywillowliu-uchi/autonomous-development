"""Tests for pre-dispatch acceptance criteria validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_control.criteria_validator import (
	Severity,
	ValidationIssue,
	is_criteria_valid,
	validate_criteria,
)


class TestValidateCriteriaExistingFiles:
	"""Criteria referencing existing files should pass."""

	def test_pytest_with_existing_test_file(self, tmp_path: Path) -> None:
		test_file = tmp_path / "tests" / "test_foo.py"
		test_file.parent.mkdir()
		test_file.touch()

		issues = validate_criteria("pytest tests/test_foo.py -v", tmp_path)
		assert issues == []

	def test_ruff_with_existing_directory(self, tmp_path: Path) -> None:
		src_dir = tmp_path / "src/"
		src_dir.mkdir()

		issues = validate_criteria("ruff check src/", tmp_path)
		assert issues == []

	def test_chained_commands_both_exist(self, tmp_path: Path) -> None:
		(tmp_path / "tests").mkdir()
		(tmp_path / "tests" / "test_bar.py").touch()
		(tmp_path / "src").mkdir()

		criteria = "pytest tests/test_bar.py -v && ruff check src/"
		issues = validate_criteria(criteria, tmp_path)
		assert issues == []

	def test_command_with_no_path_args(self, tmp_path: Path) -> None:
		issues = validate_criteria("pytest -q", tmp_path)
		assert issues == []


class TestValidateCriteriaMissingFiles:
	"""Criteria referencing missing files should produce warnings."""

	def test_missing_test_file(self, tmp_path: Path) -> None:
		issues = validate_criteria("pytest tests/test_missing.py", tmp_path)
		assert len(issues) == 1
		assert issues[0].severity == Severity.warning
		assert "does not exist" in issues[0].message
		assert issues[0].criteria_fragment == "tests/test_missing.py"

	def test_missing_directory(self, tmp_path: Path) -> None:
		issues = validate_criteria("ruff check src/", tmp_path)
		warnings = [i for i in issues if i.severity == Severity.warning]
		assert len(warnings) == 1
		assert "does not exist" in warnings[0].message

	def test_multiple_missing_paths(self, tmp_path: Path) -> None:
		criteria = "pytest tests/test_a.py && ruff check src/"
		issues = validate_criteria(criteria, tmp_path)
		warnings = [i for i in issues if i.severity == Severity.warning]
		assert len(warnings) == 2

	def test_one_exists_one_missing(self, tmp_path: Path) -> None:
		(tmp_path / "tests").mkdir()
		(tmp_path / "tests" / "test_a.py").touch()

		criteria = "pytest tests/test_a.py && ruff check src/"
		issues = validate_criteria(criteria, tmp_path)
		warnings = [i for i in issues if i.severity == Severity.warning]
		assert len(warnings) == 1
		assert "src/" in warnings[0].criteria_fragment


class TestDirectoryTraversal:
	"""Paths with ../ should be rejected."""

	def test_dotdot_in_path_rejected(self, tmp_path: Path) -> None:
		issues = validate_criteria("pytest ../outside/test_foo.py", tmp_path)
		errors = [i for i in issues if i.severity == Severity.error]
		assert len(errors) >= 1
		assert any("traversal" in e.message.lower() for e in errors)

	def test_nested_traversal_rejected(self, tmp_path: Path) -> None:
		issues = validate_criteria("ruff check tests/../../etc/passwd", tmp_path)
		errors = [i for i in issues if i.severity == Severity.error]
		assert len(errors) >= 1
		assert any("traversal" in e.message.lower() for e in errors)

	def test_absolute_path_outside_project(self, tmp_path: Path) -> None:
		issues = validate_criteria("pytest /etc/passwd", tmp_path)
		errors = [i for i in issues if i.severity == Severity.error]
		assert len(errors) >= 1
		assert any("absolute path outside project" in e.message.lower() for e in errors)

	def test_absolute_path_inside_project_ok(self, tmp_path: Path) -> None:
		test_file = tmp_path / "tests" / "test_ok.py"
		test_file.parent.mkdir()
		test_file.touch()

		issues = validate_criteria(f"pytest {test_file}", tmp_path)
		assert not any(
			i.severity == Severity.error and "outside project" in i.message.lower()
			for i in issues
		)


class TestShellSyntax:
	"""Malformed shell syntax should be caught."""

	def test_unmatched_single_quote(self, tmp_path: Path) -> None:
		issues = validate_criteria("pytest 'tests/test_foo.py", tmp_path)
		errors = [i for i in issues if i.severity == Severity.error]
		assert len(errors) >= 1
		assert any("syntax" in e.message.lower() for e in errors)

	def test_unmatched_double_quote(self, tmp_path: Path) -> None:
		issues = validate_criteria('pytest "tests/test_foo.py', tmp_path)
		errors = [i for i in issues if i.severity == Severity.error]
		assert len(errors) >= 1
		assert any("syntax" in e.message.lower() for e in errors)

	def test_empty_segment_in_chain(self, tmp_path: Path) -> None:
		issues = validate_criteria("pytest -q && && ruff check", tmp_path)
		errors = [i for i in issues if i.severity == Severity.error]
		assert len(errors) >= 1
		assert any("empty command" in e.message.lower() for e in errors)

	def test_trailing_ampersand_chain(self, tmp_path: Path) -> None:
		issues = validate_criteria("pytest -q &&", tmp_path)
		errors = [i for i in issues if i.severity == Severity.error]
		assert len(errors) >= 1
		assert any("empty command" in e.message.lower() for e in errors)

	def test_leading_ampersand_chain(self, tmp_path: Path) -> None:
		issues = validate_criteria("&& pytest -q", tmp_path)
		errors = [i for i in issues if i.severity == Severity.error]
		assert len(errors) >= 1
		assert any("empty command" in e.message.lower() for e in errors)

	def test_valid_quoted_path(self, tmp_path: Path) -> None:
		(tmp_path / "tests").mkdir()
		(tmp_path / "tests" / "test_foo.py").touch()
		issues = validate_criteria("pytest 'tests/test_foo.py'", tmp_path)
		assert not any(
			i.severity == Severity.error and "syntax" in i.message.lower()
			for i in issues
		)


class TestEmptyCriteria:
	"""Empty or whitespace-only criteria should pass."""

	def test_empty_string(self, tmp_path: Path) -> None:
		issues = validate_criteria("", tmp_path)
		assert issues == []

	def test_whitespace_only(self, tmp_path: Path) -> None:
		issues = validate_criteria("   ", tmp_path)
		assert issues == []

	def test_none_like_empty(self, tmp_path: Path) -> None:
		"""Empty criteria should not produce issues."""
		assert is_criteria_valid("", tmp_path) is True


class TestIsCriteriaValid:
	"""Convenience wrapper returns bool based on error presence."""

	def test_valid_returns_true(self, tmp_path: Path) -> None:
		(tmp_path / "tests").mkdir()
		(tmp_path / "tests" / "test_ok.py").touch()
		assert is_criteria_valid("pytest tests/test_ok.py", tmp_path) is True

	def test_missing_file_is_valid(self, tmp_path: Path) -> None:
		# Missing files are warnings, not errors -- workers can create them
		assert is_criteria_valid("pytest tests/test_nope.py", tmp_path) is True

	def test_traversal_returns_false(self, tmp_path: Path) -> None:
		assert is_criteria_valid("pytest ../bad.py", tmp_path) is False

	def test_syntax_error_returns_false(self, tmp_path: Path) -> None:
		assert is_criteria_valid("pytest 'unclosed", tmp_path) is False

	def test_empty_is_valid(self, tmp_path: Path) -> None:
		assert is_criteria_valid("", tmp_path) is True


class TestValidationIssueDataclass:
	"""ValidationIssue is a proper frozen dataclass."""

	def test_fields(self) -> None:
		issue = ValidationIssue(
			severity=Severity.error,
			message="test message",
			criteria_fragment="frag",
		)
		assert issue.severity == Severity.error
		assert issue.message == "test message"
		assert issue.criteria_fragment == "frag"

	def test_frozen(self) -> None:
		issue = ValidationIssue(
			severity=Severity.warning,
			message="warn",
			criteria_fragment="f",
		)
		with pytest.raises(AttributeError):
			issue.message = "changed"  # type: ignore[misc]

	def test_equality(self) -> None:
		a = ValidationIssue(Severity.error, "msg", "frag")
		b = ValidationIssue(Severity.error, "msg", "frag")
		assert a == b


class TestChainedCriteriaValidation:
	"""Each segment of a && chain is validated independently."""

	def test_each_segment_checked(self, tmp_path: Path) -> None:
		(tmp_path / "src").mkdir()
		criteria = "pytest tests/test_missing.py -v && ruff check src/"
		issues = validate_criteria(criteria, tmp_path)
		warning_fragments = {i.criteria_fragment for i in issues if i.severity == Severity.warning}
		assert "tests/test_missing.py" in warning_fragments
		assert "src/" not in warning_fragments

	def test_both_segments_can_warn(self, tmp_path: Path) -> None:
		criteria = "pytest tests/gone.py && ruff check nope/"
		issues = validate_criteria(criteria, tmp_path)
		warnings = [i for i in issues if i.severity == Severity.warning]
		assert len(warnings) == 2
