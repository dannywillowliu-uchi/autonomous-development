"""Tests for the AST-based codebase snapshot builder."""

from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

from mission_control.snapshot import (
	_extract_signatures,
	_format_params,
	build_project_snapshot,
	clear_snapshot_cache,
	get_project_snapshot,
)


def _write(path: Path, content: str) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(dedent(content))
	return path


class TestExtractSignatures:
	def test_public_function(self, tmp_path: Path) -> None:
		f = _write(tmp_path / "mod.py", """\
			def hello(name: str) -> str:
				return name
		""")
		sigs = _extract_signatures(f)
		assert any("def hello(name: str) -> str" in s for s in sigs)

	def test_skips_private(self, tmp_path: Path) -> None:
		f = _write(tmp_path / "mod.py", """\
			def _internal():
				pass
			def public():
				pass
		""")
		sigs = _extract_signatures(f)
		assert not any("_internal" in s for s in sigs)
		assert any("public" in s for s in sigs)

	def test_class_with_methods(self, tmp_path: Path) -> None:
		f = _write(tmp_path / "mod.py", """\
			class Foo:
				def bar(self) -> None:
					pass
				def _secret(self):
					pass
		""")
		sigs = _extract_signatures(f)
		assert any("class Foo" in s for s in sigs)
		assert any("bar" in s for s in sigs)
		assert not any("_secret" in s for s in sigs)

	def test_private_class_skipped(self, tmp_path: Path) -> None:
		f = _write(tmp_path / "mod.py", """\
			class _Hidden:
				def method(self):
					pass
		""")
		sigs = _extract_signatures(f)
		assert not any("_Hidden" in s for s in sigs)

	def test_async_function(self, tmp_path: Path) -> None:
		f = _write(tmp_path / "mod.py", """\
			async def fetch(url: str) -> bytes:
				pass
		""")
		sigs = _extract_signatures(f)
		assert any("async def fetch" in s for s in sigs)

	def test_syntax_error_returns_empty(self, tmp_path: Path) -> None:
		f = _write(tmp_path / "bad.py", "def broken(:\n")
		sigs = _extract_signatures(f)
		assert sigs == []


class TestFormatParams:
	def _parse_args(self, func_src: str) -> ast.arguments:
		tree = ast.parse(dedent(func_src))
		return tree.body[0].args

	def test_simple(self) -> None:
		args = self._parse_args("def f(a, b): pass")
		result = _format_params(args)
		assert result == "a, b"

	def test_skips_self(self) -> None:
		args = self._parse_args("def f(self, x): pass")
		result = _format_params(args)
		assert result == "x"
		assert "self" not in result

	def test_annotations(self) -> None:
		args = self._parse_args("def f(x: int, y: str): pass")
		result = _format_params(args)
		assert "x: int" in result
		assert "y: str" in result

	def test_abbreviates_many_params(self) -> None:
		args = self._parse_args("def f(a, b, c, d, e): pass")
		result = _format_params(args)
		assert "..." in result
		assert result.count(",") == 2

	def test_defaults(self) -> None:
		args = self._parse_args("def f(x, y=5): pass")
		result = _format_params(args)
		assert "y=5" in result

	def test_kwargs(self) -> None:
		args = self._parse_args("def f(**kw): pass")
		result = _format_params(args)
		assert "**kw" in result


class TestBuildProjectSnapshot:
	def test_simple_project(self, tmp_path: Path) -> None:
		_write(tmp_path / "main.py", """\
			def run() -> None:
				pass
		""")
		_write(tmp_path / "utils.py", """\
			class Helper:
				def do_thing(self) -> str:
					return "ok"
		""")
		text = build_project_snapshot(tmp_path)
		assert "main.py" in text
		assert "utils.py" in text
		assert "def run" in text
		assert "class Helper" in text
		assert "do_thing" in text

	def test_skips_init_and_tests(self, tmp_path: Path) -> None:
		_write(tmp_path / "__init__.py", "")
		_write(tmp_path / "test_foo.py", "def test_x(): pass")
		_write(tmp_path / "conftest.py", "import pytest")
		_write(tmp_path / "real.py", "def real(): pass")
		text = build_project_snapshot(tmp_path)
		assert "__init__" not in text
		assert "test_foo" not in text
		assert "conftest" not in text
		assert "real.py" in text

	def test_skips_venv_and_pycache(self, tmp_path: Path) -> None:
		_write(tmp_path / ".venv" / "lib.py", "def venv_func(): pass")
		_write(tmp_path / "__pycache__" / "cached.py", "def cached(): pass")
		_write(tmp_path / "app.py", "def app(): pass")
		text = build_project_snapshot(tmp_path)
		assert "venv_func" not in text
		assert "cached" not in text
		assert "app.py" in text

	def test_empty_project(self, tmp_path: Path) -> None:
		text = build_project_snapshot(tmp_path)
		assert text == ""

	def test_includes_full_signatures(self, tmp_path: Path) -> None:
		_write(tmp_path / "api.py", """\
			class Server:
				def start(self, port: int) -> None:
					pass
				def stop(self) -> None:
					pass
			def create_app(config: dict) -> object:
				pass
		""")
		text = build_project_snapshot(tmp_path)
		assert "class Server:" in text
		assert "def start(port: int) -> None" in text
		assert "def stop() -> None" in text
		assert "def create_app(config: dict) -> object" in text

	def test_subdirectory_files(self, tmp_path: Path) -> None:
		_write(tmp_path / "pkg" / "core.py", "def core_fn(): pass")
		text = build_project_snapshot(tmp_path)
		assert "pkg/core.py" in text or "pkg\\core.py" in text
		assert "core_fn" in text


class TestCaching:
	def test_caches_result(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "a.py", "def a(): pass")
		r1 = get_project_snapshot(tmp_path)
		_write(tmp_path / "a.py", "def b(): pass")
		r2 = get_project_snapshot(tmp_path)
		assert r1 == r2
		assert "def a" in r2

	def test_clear_cache(self, tmp_path: Path) -> None:
		clear_snapshot_cache()
		_write(tmp_path / "a.py", "def a(): pass")
		r1 = get_project_snapshot(tmp_path)
		clear_snapshot_cache()
		_write(tmp_path / "a.py", "def b(): pass")
		r2 = get_project_snapshot(tmp_path)
		assert r1 != r2
		assert "def b" in r2
