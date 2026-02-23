"""Tests for path traversal protection utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_control.path_security import validate_config_path


class TestValidateConfigPath:
	def test_valid_path_within_base(self, tmp_path: Path) -> None:
		subdir = tmp_path / "configs"
		subdir.mkdir()
		target = subdir / "app.toml"
		target.touch()

		result = validate_config_path(str(target), [tmp_path])
		assert result == target.resolve()

	def test_valid_path_multiple_bases(self, tmp_path: Path) -> None:
		base_a = tmp_path / "a"
		base_b = tmp_path / "b"
		base_a.mkdir()
		base_b.mkdir()
		target = base_b / "config.toml"
		target.touch()

		result = validate_config_path(str(target), [base_a, base_b])
		assert result == target.resolve()

	def test_dotdot_traversal_rejected(self, tmp_path: Path) -> None:
		allowed = tmp_path / "allowed"
		allowed.mkdir()
		traversal_path = str(allowed / ".." / ".." / "etc" / "passwd")

		with pytest.raises(ValueError, match="path outside allowed directories"):
			validate_config_path(traversal_path, [allowed])

	def test_dotdot_within_base_still_valid(self, tmp_path: Path) -> None:
		subdir = tmp_path / "a" / "b"
		subdir.mkdir(parents=True)
		# a/b/../c resolves to a/c which is still under tmp_path
		target = tmp_path / "a" / "c"
		target.touch()
		path_with_dotdot = str(subdir / ".." / "c")

		result = validate_config_path(path_with_dotdot, [tmp_path])
		assert result == target.resolve()

	def test_symlink_resolved(self, tmp_path: Path) -> None:
		allowed = tmp_path / "allowed"
		allowed.mkdir()
		outside = tmp_path / "outside"
		outside.mkdir()
		secret = outside / "secret.txt"
		secret.touch()

		link = allowed / "sneaky"
		link.symlink_to(secret)

		with pytest.raises(ValueError, match="path outside allowed directories"):
			validate_config_path(str(link), [allowed])

	def test_symlink_within_base_valid(self, tmp_path: Path) -> None:
		allowed = tmp_path / "allowed"
		allowed.mkdir()
		real_file = allowed / "real.toml"
		real_file.touch()
		link = allowed / "link.toml"
		link.symlink_to(real_file)

		result = validate_config_path(str(link), [allowed])
		assert result == real_file.resolve()

	def test_absolute_path_outside_all_bases(self, tmp_path: Path) -> None:
		allowed = tmp_path / "safe"
		allowed.mkdir()

		with pytest.raises(ValueError, match="path outside allowed directories"):
			validate_config_path("/tmp/evil/config.toml", [allowed])

	def test_empty_path_rejected(self, tmp_path: Path) -> None:
		with pytest.raises(ValueError, match="empty path"):
			validate_config_path("", [tmp_path])

	def test_null_byte_rejected(self, tmp_path: Path) -> None:
		with pytest.raises(ValueError, match="invalid path"):
			validate_config_path("config\x00.toml", [tmp_path])

	def test_no_path_details_leaked_in_error(self, tmp_path: Path) -> None:
		allowed = tmp_path / "safe"
		allowed.mkdir()
		evil_path = "/etc/shadow"

		with pytest.raises(ValueError) as exc_info:
			validate_config_path(evil_path, [allowed])

		assert evil_path not in str(exc_info.value)
		assert str(allowed) not in str(exc_info.value)

	def test_relative_path_resolved_against_cwd(self, tmp_path: Path) -> None:
		cwd = Path.cwd().resolve()
		result = validate_config_path("somefile.toml", [cwd])
		assert result == (cwd / "somefile.toml")

	def test_directory_symlink_traversal(self, tmp_path: Path) -> None:
		allowed = tmp_path / "allowed"
		allowed.mkdir()
		outside = tmp_path / "outside"
		outside.mkdir()
		(outside / "data.txt").touch()

		# Symlink a directory inside allowed that points outside
		link_dir = allowed / "escape"
		link_dir.symlink_to(outside)

		with pytest.raises(ValueError, match="path outside allowed directories"):
			validate_config_path(str(link_dir / "data.txt"), [allowed])
