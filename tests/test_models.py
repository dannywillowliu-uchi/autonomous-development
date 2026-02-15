"""Tests for data models -- defaults, computed properties, edge cases."""

from __future__ import annotations

from mission_control.models import (
	Handoff,
	MergeRequest,
	Mission,
	Plan,
	PlanNode,
	Round,
	Session,
	Snapshot,
	SnapshotDelta,
	TaskRecord,
	Worker,
	WorkUnit,
	_new_id,
	_now_iso,
)


class TestHelpers:
	def test_new_id_length(self) -> None:
		assert len(_new_id()) == 12

	def test_new_id_unique(self) -> None:
		ids = {_new_id() for _ in range(100)}
		assert len(ids) == 100

	def test_now_iso_format(self) -> None:
		ts = _now_iso()
		assert "T" in ts
		assert "+" in ts or "Z" in ts


class TestSnapshotDelta:
	def test_improved_tests_fixed(self) -> None:
		delta = SnapshotDelta(tests_fixed=3)
		assert delta.improved is True
		assert delta.regressed is False

	def test_improved_lint_reduced(self) -> None:
		delta = SnapshotDelta(lint_delta=-2)
		assert delta.improved is True

	def test_improved_type_errors_reduced(self) -> None:
		delta = SnapshotDelta(type_delta=-1)
		assert delta.improved is True

	def test_improved_security_reduced(self) -> None:
		delta = SnapshotDelta(security_delta=-1)
		assert delta.improved is True

	def test_not_improved_when_tests_broken(self) -> None:
		delta = SnapshotDelta(tests_fixed=5, tests_broken=1)
		assert delta.improved is False

	def test_not_improved_when_security_increased(self) -> None:
		delta = SnapshotDelta(tests_fixed=5, security_delta=1)
		assert delta.improved is False

	def test_not_improved_no_changes(self) -> None:
		delta = SnapshotDelta()
		assert delta.improved is False

	def test_regressed_tests_broken(self) -> None:
		delta = SnapshotDelta(tests_broken=2)
		assert delta.regressed is True

	def test_regressed_security_increased(self) -> None:
		delta = SnapshotDelta(security_delta=1)
		assert delta.regressed is True

	def test_not_regressed_clean(self) -> None:
		delta = SnapshotDelta(tests_fixed=3, lint_delta=-1)
		assert delta.regressed is False

	def test_both_improved_and_regressed_impossible(self) -> None:
		"""If tests_broken > 0, improved must be False."""
		delta = SnapshotDelta(tests_fixed=5, tests_broken=1)
		assert delta.improved is False
		assert delta.regressed is True


class TestWorkUnitDefaults:
	def test_defaults(self) -> None:
		wu = WorkUnit()
		assert wu.status == "pending"
		assert wu.priority == 1
		assert wu.attempt == 0
		assert wu.max_attempts == 3
		assert wu.timeout is None
		assert wu.verification_command is None
		assert wu.depends_on == ""
		assert wu.exit_code is None
		assert wu.commit_hash is None

	def test_unique_ids(self) -> None:
		wu1 = WorkUnit()
		wu2 = WorkUnit()
		assert wu1.id != wu2.id

	def test_unit_type_single_field_with_correct_default(self) -> None:
		"""unit_type must appear exactly once and default to 'implementation'."""
		import dataclasses

		fields = dataclasses.fields(WorkUnit)
		unit_type_fields = [f for f in fields if f.name == "unit_type"]
		assert len(unit_type_fields) == 1
		wu = WorkUnit()
		assert wu.unit_type == "implementation"

	def test_field_order_timeout_and_verification_before_epoch(self) -> None:
		"""timeout and verification_command must appear between unit_type and epoch_id."""
		import dataclasses

		names = [f.name for f in dataclasses.fields(WorkUnit)]
		ut_idx = names.index("unit_type")
		ep_idx = names.index("epoch_id")
		assert "timeout" in names[ut_idx:ep_idx]
		assert "verification_command" in names[ut_idx:ep_idx]

	def test_per_unit_overrides(self) -> None:
		wu = WorkUnit(timeout=600, verification_command="make test")
		assert wu.timeout == 600
		assert wu.verification_command == "make test"


class TestSessionDefaults:
	def test_defaults(self) -> None:
		s = Session()
		assert s.status == "pending"
		assert s.target_name == ""
		assert s.exit_code is None
		assert s.cost_usd is None

	def test_with_values(self) -> None:
		s = Session(target_name="proj", task_description="Fix bug", status="running")
		assert s.target_name == "proj"
		assert s.task_description == "Fix bug"
		assert s.status == "running"


class TestSnapshotDefaults:
	def test_defaults(self) -> None:
		snap = Snapshot()
		assert snap.test_total == 0
		assert snap.test_passed == 0
		assert snap.test_failed == 0
		assert snap.lint_errors == 0
		assert snap.type_errors == 0
		assert snap.security_findings == 0


class TestPlanDefaults:
	def test_defaults(self) -> None:
		p = Plan()
		assert p.status == "pending"
		assert p.total_units == 0
		assert p.completed_units == 0
		assert p.failed_units == 0


class TestMissionDefaults:
	def test_defaults(self) -> None:
		m = Mission()
		assert m.status == "pending"
		assert m.total_rounds == 0
		assert m.final_score == 0.0

	def test_with_objective(self) -> None:
		m = Mission(objective="Build API")
		assert m.objective == "Build API"


class TestRoundDefaults:
	def test_defaults(self) -> None:
		r = Round()
		assert r.status == "pending"
		assert r.objective_score == 0.0
		assert r.objective_met is False


class TestPlanNodeDefaults:
	def test_defaults(self) -> None:
		node = PlanNode()
		assert node.node_type == "branch"
		assert node.status == "pending"
		assert node.work_unit_id is None
		assert node.children_ids == ""


class TestHandoffDefaults:
	def test_defaults(self) -> None:
		h = Handoff()
		assert h.status == ""
		assert h.commits == ""
		assert h.summary == ""


class TestWorkerDefaults:
	def test_defaults(self) -> None:
		w = Worker()
		assert w.status == "idle"
		assert w.units_completed == 0
		assert w.units_failed == 0
		assert w.backend_type == "local"


class TestTaskRecordDefaults:
	def test_defaults(self) -> None:
		t = TaskRecord()
		assert t.status == "discovered"
		assert t.priority == 7
		assert t.source == ""


class TestMergeRequestDefaults:
	def test_defaults(self) -> None:
		mr = MergeRequest()
		assert mr.status == "pending"
		assert mr.position == 0
		assert mr.rebase_attempts == 0
