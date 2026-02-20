"""Tests for Pydantic Handoff model, DB round-trip, and _build_handoff."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mission_control.db import Database
from mission_control.models import Epoch, Handoff, Mission, Plan, WorkUnit
from mission_control.worker import _build_handoff

# -- Handoff model tests --


class TestHandoffModel:
	def test_defaults(self) -> None:
		h = Handoff()
		assert h.commits == []
		assert h.discoveries == []
		assert h.concerns == []
		assert h.files_changed == []
		assert h.summary == ""
		assert h.status == ""
		assert h.epoch_id is None

	def test_construction_with_lists(self) -> None:
		h = Handoff(
			commits=["abc123"],
			discoveries=["found X"],
			concerns=["watch Y"],
			files_changed=["src/a.py", "src/b.py"],
		)
		assert h.commits == ["abc123"]
		assert h.discoveries == ["found X"]
		assert h.concerns == ["watch Y"]
		assert h.files_changed == ["src/a.py", "src/b.py"]

	def test_mutation(self) -> None:
		h = Handoff()
		h.concerns.append("new concern")
		assert h.concerns == ["new concern"]

	def test_model_dump(self) -> None:
		h = Handoff(
			work_unit_id="wu1",
			status="completed",
			commits=["abc"],
			discoveries=["d1"],
		)
		d = h.model_dump()
		assert d["commits"] == ["abc"]
		assert d["discoveries"] == ["d1"]
		assert isinstance(d["concerns"], list)

	def test_rejects_string_for_list_field(self) -> None:
		with pytest.raises(ValidationError):
			Handoff(commits="not a list")  # type: ignore[arg-type]

	def test_rejects_int_for_list_field(self) -> None:
		with pytest.raises(ValidationError):
			Handoff(discoveries=42)  # type: ignore[arg-type]


# -- DB round-trip tests --


class TestHandoffDBRoundTrip:
	@pytest.fixture()
	def db(self) -> Database:
		d = Database()
		yield d
		d.close()

	def _seed(self, db: Database) -> None:
		db.insert_mission(Mission(id="m1", objective="test"))
		db.insert_plan(Plan(id="p1", objective="test"))
		epoch = Epoch(id="ep1", mission_id="m1", number=1)
		db.insert_epoch(epoch)
		wu = WorkUnit(id="wu1", plan_id="p1", title="Task", epoch_id="ep1")
		db.insert_work_unit(wu)

	def test_insert_and_get(self, db: Database) -> None:
		self._seed(db)
		h = Handoff(
			id="h1",
			work_unit_id="wu1",
			epoch_id="ep1",
			status="completed",
			commits=["abc123", "def456"],
			discoveries=["found pattern"],
			concerns=["watch out"],
			files_changed=["src/x.py"],
		)
		db.insert_handoff(h)
		fetched = db.get_handoff("h1")
		assert fetched is not None
		assert fetched.commits == ["abc123", "def456"]
		assert fetched.discoveries == ["found pattern"]
		assert fetched.concerns == ["watch out"]
		assert fetched.files_changed == ["src/x.py"]

	def test_empty_lists(self, db: Database) -> None:
		self._seed(db)
		h = Handoff(id="h2", work_unit_id="wu1", epoch_id="ep1", status="completed")
		db.insert_handoff(h)
		fetched = db.get_handoff("h2")
		assert fetched is not None
		assert fetched.commits == []
		assert fetched.discoveries == []
		assert fetched.concerns == []
		assert fetched.files_changed == []

	def test_backward_compat_raw_json_in_db(self, db: Database) -> None:
		"""Legacy rows with raw JSON strings in DB are still deserialized correctly."""
		self._seed(db)
		# Simulate a legacy row by writing raw JSON strings directly
		db.conn.execute(
			"""INSERT INTO handoffs
			(id, work_unit_id, round_id, epoch_id, status, commits,
			 summary, discoveries, concerns, files_changed)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			(
				"h_legacy", "wu1", "", "ep1", "completed",
				json.dumps(["old_commit"]),
				"legacy summary",
				json.dumps(["old_discovery"]),
				json.dumps(["old_concern"]),
				json.dumps(["old_file.py"]),
			),
		)
		db.conn.commit()
		fetched = db.get_handoff("h_legacy")
		assert fetched is not None
		assert fetched.commits == ["old_commit"]
		assert fetched.discoveries == ["old_discovery"]
		assert fetched.concerns == ["old_concern"]
		assert fetched.files_changed == ["old_file.py"]

	def test_malformed_json_in_db_returns_empty(self, db: Database) -> None:
		"""Malformed JSON in DB columns returns empty lists."""
		self._seed(db)
		db.conn.execute(
			"""INSERT INTO handoffs
			(id, work_unit_id, round_id, epoch_id, status, commits,
			 summary, discoveries, concerns, files_changed)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
			("h_bad", "wu1", "", "ep1", "completed", "not json", "", "{}", "null", ""),
		)
		db.conn.commit()
		fetched = db.get_handoff("h_bad")
		assert fetched is not None
		assert fetched.commits == []
		assert fetched.discoveries == []
		assert fetched.concerns == []
		assert fetched.files_changed == []


# -- _build_handoff tests --


class TestBuildHandoff:
	def test_valid_mc_result(self) -> None:
		mc = {
			"status": "completed",
			"summary": "did stuff",
			"commits": ["abc123"],
			"discoveries": ["found X"],
			"concerns": ["watch Y"],
			"files_changed": ["src/a.py"],
		}
		h = _build_handoff(mc, "wu1", "r1")
		assert h.status == "completed"
		assert h.commits == ["abc123"]
		assert h.discoveries == ["found X"]
		assert h.concerns == ["watch Y"]
		assert h.files_changed == ["src/a.py"]
		assert h.work_unit_id == "wu1"
		assert h.round_id == "r1"

	def test_missing_fields_default_to_empty(self) -> None:
		mc = {"status": "completed", "summary": "done"}
		h = _build_handoff(mc, "wu2", "r2")
		assert h.commits == []
		assert h.discoveries == []
		assert h.concerns == []
		assert h.files_changed == []

	def test_empty_mc_result(self) -> None:
		h = _build_handoff({}, "wu3", "r3")
		assert h.status == "failed"
		assert h.summary == ""
		assert h.commits == []
