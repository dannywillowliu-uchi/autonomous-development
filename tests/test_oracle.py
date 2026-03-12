"""Tests for immutable oracle."""

from autodev.oracle import check_oracle_violation


class TestCheckOracleViolation:
	def test_no_violations(self):
		assert check_oracle_violation(["src/autodev/worker.py", "README.md"]) == []

	def test_detects_pyproject(self):
		result = check_oracle_violation(["pyproject.toml", "src/autodev/worker.py"])
		assert result == ["pyproject.toml"]

	def test_detects_nested_conftest(self):
		result = check_oracle_violation(["tests/conftest.py"])
		assert result == ["tests/conftest.py"]

	def test_detects_multiple(self):
		result = check_oracle_violation(["pyproject.toml", "src/foo.py", "ruff.toml", "tests/conftest.py"])
		assert result == ["pyproject.toml", "ruff.toml", "tests/conftest.py"]

	def test_case_sensitive(self):
		assert check_oracle_violation(["PYPROJECT.TOML"]) == []
