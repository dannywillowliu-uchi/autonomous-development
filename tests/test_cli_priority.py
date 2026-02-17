"""Tests for CLI priority subcommand group."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_control.cli import (
	build_parser,
	cmd_priority_defer,
	cmd_priority_import,
	cmd_priority_list,
	cmd_priority_recalc,
	cmd_priority_set,
	parse_backlog_md,
)
from mission_control.db import Database
from mission_control.models import BacklogItem, _now_iso

SAMPLE_BACKLOG_MD = """\
# Backlog

## P0: Replace LLM Evaluator with Objective Signals

**Problem**: The evaluator is expensive and noisy.

**Files**: evaluator.py

---

## P1: N-of-M Candidate Selection for Fixup

**Problem**: Fixup agent makes one attempt.

---

## P2: Architect/Editor Model Split

**Problem**: Workers use the same model.
"""


class TestArgParsing:
	def test_priority_set_args(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["priority", "set", "abc123", "9.5"])
		assert args.command == "priority"
		assert args.priority_command == "set"
		assert args.item_id == "abc123"
		assert args.score == 9.5

	def test_priority_import_args(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["priority", "import", "--file", "/tmp/BACKLOG.md"])
		assert args.command == "priority"
		assert args.priority_command == "import"
		assert args.file == "/tmp/BACKLOG.md"


class TestParseBacklogMd:
	def test_parses_items(self) -> None:
		items = parse_backlog_md(SAMPLE_BACKLOG_MD)
		assert len(items) == 3

	def test_extracts_titles(self) -> None:
		items = parse_backlog_md(SAMPLE_BACKLOG_MD)
		assert items[0].title == "Replace LLM Evaluator with Objective Signals"
		assert items[1].title == "N-of-M Candidate Selection for Fixup"
		assert items[2].title == "Architect/Editor Model Split"

	def test_calculates_impact_from_priority(self) -> None:
		items = parse_backlog_md(SAMPLE_BACKLOG_MD)
		assert items[0].impact == 10  # P0 -> 10-0 = 10
		assert items[1].impact == 9   # P1 -> 10-1 = 9
		assert items[2].impact == 8   # P2 -> 10-2 = 8

	def test_extracts_description(self) -> None:
		items = parse_backlog_md(SAMPLE_BACKLOG_MD)
		assert "expensive and noisy" in items[0].description


def _setup_db_with_config(tmp_path: Path) -> tuple[Path, Path]:
	"""Create config and DB for priority tests."""
	config_file = tmp_path / "mission-control.toml"
	config_file.write_text('[target]\nname = "test"\npath = "/tmp/test"\n')
	db_path = tmp_path / "mission-control.db"
	Database(db_path).close()
	return config_file, db_path


def _insert_sample_items(db_path: Path) -> list[str]:
	"""Insert sample backlog items, return their IDs."""
	now = _now_iso()
	ids = []
	with Database(db_path) as db:
		for i, (title, impact, effort) in enumerate([
			("Fix auth bug", 9, 3),
			("Add caching", 7, 5),
			("Update docs", 4, 2),
		]):
			item = BacklogItem(
				id=f"item{i}",
				title=title,
				impact=impact,
				effort=effort,
				priority_score=impact * (11 - effort) / 10.0,
				track="feature",
				status="pending",
				created_at=now,
				updated_at=now,
			)
			db.insert_backlog_item(item)
			ids.append(item.id)
	return ids


class TestCmdPriorityList:
	def test_list_with_items(self, tmp_path: Path, capsys: object) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		_insert_sample_items(db_path)
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "list"])
		result = cmd_priority_list(args)
		assert result == 0


class TestCmdPrioritySet:
	def test_set_score(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		ids = _insert_sample_items(db_path)
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "set", ids[0], "99.0"])
		result = cmd_priority_set(args)
		assert result == 0
		with Database(db_path) as db:
			item = db.get_backlog_item(ids[0])
			assert item is not None
			assert item.pinned_score == 99.0

	def test_set_not_found(self, tmp_path: Path) -> None:
		config_file, _ = _setup_db_with_config(tmp_path)
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "set", "nonexistent", "5.0"])
		result = cmd_priority_set(args)
		assert result == 1


class TestCmdPriorityDefer:
	def test_defer_item(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		ids = _insert_sample_items(db_path)
		parser = build_parser()
		args = parser.parse_args(["priority", "--config", str(config_file), "defer", ids[1]])
		result = cmd_priority_defer(args)
		assert result == 0
		with Database(db_path) as db:
			item = db.get_backlog_item(ids[1])
			assert item is not None
			assert item.status == "deferred"


class TestCmdPriorityImport:
	def test_import_from_file(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		backlog_file = tmp_path / "BACKLOG.md"
		backlog_file.write_text(SAMPLE_BACKLOG_MD)
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "import", "--file", str(backlog_file),
		])
		result = cmd_priority_import(args)
		assert result == 0
		with Database(db_path) as db:
			items = db.list_backlog_items()
			assert len(items) == 3

	def test_import_skips_duplicates(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		backlog_file = tmp_path / "BACKLOG.md"
		backlog_file.write_text(SAMPLE_BACKLOG_MD)
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "import", "--file", str(backlog_file),
		])
		cmd_priority_import(args)
		# Import again -- should skip duplicates
		args2 = parser.parse_args([
			"priority", "--config", str(config_file), "import", "--file", str(backlog_file),
		])
		result = cmd_priority_import(args2)
		assert result == 0
		with Database(db_path) as db:
			items = db.list_backlog_items()
			assert len(items) == 3


class TestCmdPriorityRecalc:
	def test_recalc_arg_parsing(self) -> None:
		parser = build_parser()
		args = parser.parse_args(["priority", "recalc"])
		assert args.command == "priority"
		assert args.priority_command == "recalc"

	def test_recalc_updates_scores(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		now = _now_iso()
		with Database(db_path) as db:
			db.insert_backlog_item(BacklogItem(
				id="item1", title="Test item", impact=10, effort=1,
				priority_score=0.0, status="pending",
				created_at=now, updated_at=now,
			))

		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "recalc",
		])
		result = cmd_priority_recalc(args)
		assert result == 0

		with Database(db_path) as db:
			item = db.get_backlog_item("item1")
			assert item is not None
			assert item.priority_score == pytest.approx(10.0)

	def test_recalc_no_changes(self, tmp_path: Path) -> None:
		config_file, db_path = _setup_db_with_config(tmp_path)
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "recalc",
		])
		result = cmd_priority_recalc(args)
		assert result == 0

	def test_recalc_no_db(self, tmp_path: Path) -> None:
		config_file = tmp_path / "mission-control.toml"
		config_file.write_text('[target]\nname = "test"\npath = "/tmp/test"\n')
		parser = build_parser()
		args = parser.parse_args([
			"priority", "--config", str(config_file), "recalc",
		])
		result = cmd_priority_recalc(args)
		assert result == 1
