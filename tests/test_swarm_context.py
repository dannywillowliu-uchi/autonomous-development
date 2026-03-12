"""Tests for swarm context synthesizer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from autodev.swarm.context import (
	DEFAULT_KEEP_MESSAGES,
	DEFAULT_MAX_INBOX_BYTES,
	ContextSynthesizer,
	parse_structured_report,
	rotate_inbox,
)
from autodev.swarm.models import (
	AgentRole,
	AgentStatus,
	SwarmAgent,
	SwarmTask,
	TaskStatus,
)


def _make_config() -> MagicMock:
	config = MagicMock()
	config.target.name = "test-project"
	config.target.objective = "Build a compiler"
	config.target.resolved_path = "/tmp/test-project"
	return config


def _make_db() -> MagicMock:
	db = MagicMock()
	db.get_knowledge_for_mission.return_value = []
	return db


class TestContextSynthesizer:
	def test_build_state_empty(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		assert state.cycle_number == 1
		assert state.agents == []
		assert state.tasks == []
		assert state.mission_objective == "Build a compiler"

	def test_build_state_increments_cycle(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state1 = ctx.build_state(agents=[], tasks=[])
		state2 = ctx.build_state(agents=[], tasks=[])
		assert state1.cycle_number == 1
		assert state2.cycle_number == 2

	def test_recent_completions(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title="Done task", status=TaskStatus.COMPLETED, result_summary="Fixed it"),
			SwarmTask(title="Pending task", status=TaskStatus.PENDING),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		assert len(state.recent_completions) == 1
		assert state.recent_completions[0]["title"] == "Done task"

	def test_recent_failures(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title="Failed task", status=TaskStatus.FAILED, attempt_count=2, result_summary="Segfault"),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		assert len(state.recent_failures) == 1
		assert state.recent_failures[0]["attempt"] == 2

	def test_files_in_flight(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		task = SwarmTask(id="t1", title="Fix parser", status=TaskStatus.IN_PROGRESS, files_hint=["src/parser.c"])
		agent = SwarmAgent(name="worker-1", status=AgentStatus.WORKING, current_task_id="t1")
		state = ctx.build_state(agents=[agent], tasks=[task])
		assert "src/parser.c" in state.files_in_flight

	def test_core_test_results_passed_through(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		results = {"pass": 216, "fail": 5, "skip": 0, "total": 221}
		state = ctx.build_state(agents=[], tasks=[], core_test_results=results)
		assert state.core_test_results["pass"] == 216


class TestStagnationDetection:
	def test_no_stagnation_initially(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[], core_test_results={"pass": 200})
		assert state.stagnation_signals == []

	def test_stagnation_after_flat_metric(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		ctx._stagnation_window = 3
		# Simulate 3 cycles with same test count
		for _ in range(3):
			state = ctx.build_state(agents=[], tasks=[], core_test_results={"pass": 200})
		# By cycle 3, should detect stagnation
		assert any(s.metric == "test_pass_count" for s in state.stagnation_signals)

	def test_no_stagnation_when_improving(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		ctx._stagnation_window = 3
		for i in range(5):
			state = ctx.build_state(agents=[], tasks=[], core_test_results={"pass": 200 + i})
		assert not any(s.metric == "test_pass_count" for s in state.stagnation_signals)

	def test_high_failure_rate_signal(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title="t1", status=TaskStatus.FAILED),
			SwarmTask(title="t2", status=TaskStatus.FAILED),
			SwarmTask(title="t3", status=TaskStatus.COMPLETED),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		assert any(s.metric == "high_failure_rate" for s in state.stagnation_signals)


class TestRenderForPlanner:
	def test_render_includes_mission(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		rendered = ctx.render_for_planner(state)
		assert "Build a compiler" in rendered

	def test_render_includes_agents_with_elapsed_time(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		agent = SwarmAgent(name="impl-1", role=AgentRole.IMPLEMENTER, status=AgentStatus.WORKING)
		state = ctx.build_state(agents=[agent], tasks=[])
		rendered = ctx.render_for_planner(state)
		assert "impl-1" in rendered
		assert "implementer" in rendered
		assert "elapsed=" in rendered

	def test_render_includes_agent_task_title(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		task = SwarmTask(id="t1", title="Fix parser bug", status=TaskStatus.IN_PROGRESS)
		agent = SwarmAgent(name="impl-1", role=AgentRole.IMPLEMENTER, status=AgentStatus.WORKING, current_task_id="t1")
		state = ctx.build_state(agents=[agent], tasks=[task])
		rendered = ctx.render_for_planner(state)
		assert 'task: "Fix parser bug"' in rendered

	def test_render_includes_stagnation_warnings(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		ctx._stagnation_window = 2
		for _ in range(2):
			state = ctx.build_state(agents=[], tasks=[], core_test_results={"pass": 100})
		rendered = ctx.render_for_planner(state)
		assert "STAGNATION" in rendered

	def test_render_includes_core_tests(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		results = {"pass": 216, "fail": 5, "skip": 0, "total": 221}
		state = ctx.build_state(agents=[], tasks=[], core_test_results=results)
		rendered = ctx.render_for_planner(state)
		assert "216" in rendered
		assert "Core Test" in rendered

	def test_render_task_progress_summary(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title="t1", status=TaskStatus.COMPLETED),
			SwarmTask(title="t2", status=TaskStatus.COMPLETED),
			SwarmTask(title="t3", status=TaskStatus.IN_PROGRESS),
			SwarmTask(title="t4", status=TaskStatus.PENDING),
			SwarmTask(title="t5", status=TaskStatus.FAILED),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		rendered = ctx.render_for_planner(state)
		assert "Task Progress" in rendered
		assert "2/5 completed" in rendered

	def test_render_dependency_status(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		dep_task = SwarmTask(id="dep1", title="Build lexer", status=TaskStatus.COMPLETED)
		blocked_task = SwarmTask(id="t2", title="Build parser", status=TaskStatus.PENDING, depends_on=["dep1"])
		state = ctx.build_state(agents=[], tasks=[dep_task, blocked_task])
		rendered = ctx.render_for_planner(state)
		assert "Build lexer [completed]" in rendered

	def test_render_blocked_flag_for_unmet_deps(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		dep_task = SwarmTask(id="dep1", title="Build lexer", status=TaskStatus.PENDING)
		blocked_task = SwarmTask(id="t2", title="Build parser", status=TaskStatus.PENDING, depends_on=["dep1"])
		state = ctx.build_state(agents=[], tasks=[dep_task, blocked_task])
		rendered = ctx.render_for_planner(state)
		assert "**BLOCKED**" in rendered
		assert "Build lexer [pending]" in rendered

	def test_render_completed_tasks_section(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title="Done task", status=TaskStatus.COMPLETED, result_summary="Fixed the parser"),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		rendered = ctx.render_for_planner(state)
		assert "## Completed Tasks" in rendered
		assert "Fixed the parser" in rendered

	def test_render_failed_tasks_with_retry_info(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(
				title="Broken task", status=TaskStatus.FAILED,
				attempt_count=2, max_attempts=3, result_summary="Segfault",
			),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		rendered = ctx.render_for_planner(state)
		assert "## Failed Tasks" in rendered
		assert "1 retries left" in rendered
		assert "Segfault" in rendered

	def test_render_failed_tasks_no_retries(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title="Exhausted task", status=TaskStatus.FAILED, attempt_count=3, max_attempts=3),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		rendered = ctx.render_for_planner(state)
		assert "NO retries left" in rendered

	def test_render_dead_agent_with_accomplishment(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		task = SwarmTask(id="t1", title="Fix lexer", status=TaskStatus.COMPLETED, result_summary="Lexer fixed")
		dead_agent = SwarmAgent(
			id="dead1", name="old-worker", role=AgentRole.IMPLEMENTER,
			status=AgentStatus.DEAD, current_task_id="t1",
			tasks_completed=1, tasks_failed=0,
		)
		state = ctx.build_state(agents=[], tasks=[task], dead_agent_history=[dead_agent])
		rendered = ctx.render_for_planner(state)
		assert "Recently Cleaned Up Agents" in rendered
		assert "old-worker" in rendered
		assert 'completed "Fix lexer"' in rendered

	def test_render_claimer_resolved_to_name(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		agent = SwarmAgent(
			id="a1", name="parser-fixer", role=AgentRole.IMPLEMENTER,
			status=AgentStatus.WORKING, current_task_id="t1",
		)
		task = SwarmTask(id="t1", title="Fix parser", status=TaskStatus.IN_PROGRESS, claimed_by="a1")
		state = ctx.build_state(agents=[agent], tasks=[task])
		rendered = ctx.render_for_planner(state)
		assert "agent: parser-fixer" in rendered


class TestHelperMethods:
	def test_format_elapsed_seconds(self) -> None:
		from datetime import datetime, timezone
		now = datetime(2025, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
		assert ContextSynthesizer._format_elapsed("2025-01-01T12:00:00+00:00", now) == "30s"

	def test_format_elapsed_minutes(self) -> None:
		from datetime import datetime, timezone
		now = datetime(2025, 1, 1, 12, 5, 30, tzinfo=timezone.utc)
		assert ContextSynthesizer._format_elapsed("2025-01-01T12:00:00+00:00", now) == "5m30s"

	def test_format_elapsed_hours(self) -> None:
		from datetime import datetime, timezone
		now = datetime(2025, 1, 1, 13, 5, 0, tzinfo=timezone.utc)
		assert ContextSynthesizer._format_elapsed("2025-01-01T12:00:00+00:00", now) == "1h5m"

	def test_format_elapsed_invalid(self) -> None:
		from datetime import datetime, timezone
		now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
		assert ContextSynthesizer._format_elapsed("not-a-date", now) == "?"

	def test_has_unmet_deps_all_completed(self) -> None:
		task_map = {"d1": SwarmTask(id="d1", status=TaskStatus.COMPLETED)}
		assert not ContextSynthesizer._has_unmet_deps(["d1"], task_map)

	def test_has_unmet_deps_some_pending(self) -> None:
		task_map = {"d1": SwarmTask(id="d1", status=TaskStatus.PENDING)}
		assert ContextSynthesizer._has_unmet_deps(["d1"], task_map)

	def test_has_unmet_deps_unknown_id(self) -> None:
		assert ContextSynthesizer._has_unmet_deps(["nonexistent"], {})

	def test_count_task_statuses(self) -> None:
		tasks = [
			SwarmTask(status=TaskStatus.COMPLETED),
			SwarmTask(status=TaskStatus.PENDING),
			SwarmTask(status=TaskStatus.IN_PROGRESS),
			SwarmTask(status=TaskStatus.FAILED),
			SwarmTask(status=TaskStatus.PENDING, depends_on=["dep"]),
		]
		counts = ContextSynthesizer._count_task_statuses(tasks)
		assert counts["completed"] == 1
		assert counts["pending"] == 1
		assert counts["in_progress"] == 1
		assert counts["failed"] == 1
		assert counts["blocked"] == 1  # pending with deps counts as blocked

	def test_resolve_claimer_name_found(self) -> None:
		agents = [SwarmAgent(id="a1", name="worker-1")]
		assert ContextSynthesizer._resolve_claimer_name("a1", agents) == "worker-1"

	def test_resolve_claimer_name_not_found(self) -> None:
		assert ContextSynthesizer._resolve_claimer_name("unknown123", []) == "unknown1"

	def test_resolve_claimer_name_none(self) -> None:
		assert ContextSynthesizer._resolve_claimer_name(None, []) == "?"

	def test_build_agent_task_map_completed(self) -> None:
		task = SwarmTask(id="t1", title="Fix bug", status=TaskStatus.COMPLETED, result_summary="Done")
		agent = SwarmAgent(id="a1", name="w1", current_task_id="t1")
		result = ContextSynthesizer._build_agent_task_map([agent], [], {"t1": task})
		assert "a1" in result
		assert 'completed "Fix bug"' in result["a1"]

	def test_build_agent_task_map_failed(self) -> None:
		task = SwarmTask(id="t1", title="Broken", status=TaskStatus.FAILED, result_summary="Crash")
		agent = SwarmAgent(id="a1", current_task_id="t1")
		result = ContextSynthesizer._build_agent_task_map([], [agent], {"t1": task})
		assert 'failed "Broken"' in result["a1"]

	def test_build_agent_task_map_no_task_id(self) -> None:
		agent = SwarmAgent(id="a1", current_task_id=None)
		result = ContextSynthesizer._build_agent_task_map([agent], [], {})
		assert result == {}

	def test_build_agent_task_map_unknown_task(self) -> None:
		agent = SwarmAgent(id="a1", current_task_id="nonexistent")
		result = ContextSynthesizer._build_agent_task_map([agent], [], {})
		assert result == {}

	def test_build_agent_task_map_in_progress_not_included(self) -> None:
		task = SwarmTask(id="t1", title="WIP", status=TaskStatus.IN_PROGRESS)
		agent = SwarmAgent(id="a1", current_task_id="t1")
		result = ContextSynthesizer._build_agent_task_map([agent], [], {"t1": task})
		assert "a1" not in result


class TestRotateInbox:
	def test_nonexistent_file_returns_false(self, tmp_path: Path) -> None:
		assert rotate_inbox(tmp_path / "missing.json") is False

	def test_small_file_not_rotated(self, tmp_path: Path) -> None:
		inbox = tmp_path / "agent.json"
		messages = [{"from": "a", "type": "report", "text": "hi"}]
		inbox.write_text(json.dumps(messages))
		assert rotate_inbox(inbox) is False

	def test_rotation_by_message_count(self, tmp_path: Path) -> None:
		inbox = tmp_path / "agent.json"
		messages = [{"from": "a", "type": "report", "text": f"msg-{i}"} for i in range(600)]
		inbox.write_text(json.dumps(messages))
		assert rotate_inbox(inbox, max_messages=500, keep_messages=200) is True
		rotated = json.loads(inbox.read_text())
		assert len(rotated) == 200
		# Should keep the most recent messages
		assert rotated[-1]["text"] == "msg-599"
		assert rotated[0]["text"] == "msg-400"

	def test_rotation_by_byte_size(self, tmp_path: Path) -> None:
		inbox = tmp_path / "agent.json"
		# Create messages that exceed max_bytes
		messages = [{"from": "a", "type": "report", "text": "x" * 200} for _ in range(100)]
		data = json.dumps(messages)
		inbox.write_text(data)
		assert rotate_inbox(inbox, max_bytes=1024, keep_messages=10) is True
		rotated = json.loads(inbox.read_text())
		assert len(rotated) == 10

	def test_no_rotation_when_under_limits(self, tmp_path: Path) -> None:
		inbox = tmp_path / "agent.json"
		messages = [{"from": "a", "type": "report", "text": f"msg-{i}"} for i in range(50)]
		inbox.write_text(json.dumps(messages))
		assert rotate_inbox(inbox, max_messages=500, max_bytes=DEFAULT_MAX_INBOX_BYTES) is False

	def test_malformed_json_returns_false(self, tmp_path: Path) -> None:
		inbox = tmp_path / "agent.json"
		inbox.write_text("{not valid json" + "x" * 2000)
		assert rotate_inbox(inbox, max_bytes=1024) is False

	def test_non_list_json_returns_false(self, tmp_path: Path) -> None:
		inbox = tmp_path / "agent.json"
		inbox.write_text(json.dumps({"not": "a list"}) + " " * 2000)
		assert rotate_inbox(inbox, max_bytes=1024) is False

	def test_rotation_creates_valid_json(self, tmp_path: Path) -> None:
		inbox = tmp_path / "agent.json"
		messages = [{"id": i} for i in range(300)]
		inbox.write_text(json.dumps(messages))
		rotate_inbox(inbox, max_messages=100, keep_messages=50)
		# Must be valid JSON after rotation
		result = json.loads(inbox.read_text())
		assert isinstance(result, list)
		assert len(result) == 50

	def test_rotation_preserves_message_content(self, tmp_path: Path) -> None:
		inbox = tmp_path / "agent.json"
		messages = [{"from": f"agent-{i}", "type": "discovery", "text": f"found-{i}"} for i in range(300)]
		inbox.write_text(json.dumps(messages))
		rotate_inbox(inbox, max_messages=100, keep_messages=50)
		result = json.loads(inbox.read_text())
		# Verify last message is preserved exactly
		assert result[-1] == {"from": "agent-299", "type": "discovery", "text": "found-299"}

	def test_lock_file_created(self, tmp_path: Path) -> None:
		inbox = tmp_path / "agent.json"
		messages = [{"id": i} for i in range(600)]
		inbox.write_text(json.dumps(messages))
		rotate_inbox(inbox, max_messages=500, keep_messages=200)
		assert (tmp_path / "agent.lock").exists()

	def test_empty_file_returns_false(self, tmp_path: Path) -> None:
		inbox = tmp_path / "agent.json"
		inbox.write_text("")
		assert rotate_inbox(inbox) is False


class TestGetRecentDiscoveries:
	def _make_ctx(self, tmp_path: Path, team_name: str = "test-team") -> ContextSynthesizer:
		config = _make_config()
		config.target.resolved_path = str(tmp_path)
		return ContextSynthesizer(config, _make_db(), team_name)

	def _setup_inbox_dir(self, tmp_path: Path, team_name: str = "test-team") -> Path:
		inbox_dir = tmp_path / ".claude" / "teams" / team_name / "inboxes"
		inbox_dir.mkdir(parents=True)
		return inbox_dir

	def test_discovery_message_type_included(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"from": "agent-1", "type": "discovery", "text": "Found a race condition"}]
		(inbox_dir / "agent-1.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("race condition" in d for d in discoveries)

	def test_report_message_type_included(self, tmp_path: Path) -> None:
		"""The 'report' type must be in the filter list for planner visibility."""
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"from": "worker-1", "type": "report", "text": "Tests passing now"}]
		(inbox_dir / "worker-1.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("Tests passing now" in d for d in discoveries)
		assert any("(report)" in d for d in discoveries)

	def test_blocked_message_type_included(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"from": "w1", "type": "blocked", "text": "Need API key"}]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("Need API key" in d for d in discoveries)

	def test_question_message_type_included(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"from": "w1", "type": "question", "text": "Which DB to use?"}]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("Which DB to use?" in d for d in discoveries)

	def test_unknown_message_type_excluded(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"from": "w1", "type": "chat", "text": "Random chat message"}]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert not any("Random chat message" in d for d in discoveries)

	def test_keyword_discovery_in_text_fallback(self, tmp_path: Path) -> None:
		"""Messages with 'discovery:' or 'found:' in text are included regardless of type."""
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"from": "w1", "type": "info", "text": "Discovery: new pattern identified"}]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("new pattern identified" in d for d in discoveries)

	def test_keyword_found_in_text_fallback(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"from": "w1", "type": "info", "text": "Found: memory leak in parser"}]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("memory leak" in d for d in discoveries)

	def test_empty_inbox_dir(self, tmp_path: Path) -> None:
		self._setup_inbox_dir(tmp_path)
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert discoveries == []

	def test_missing_inbox_dir(self, tmp_path: Path) -> None:
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert discoveries == []

	def test_malformed_json_inbox_skipped(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		(inbox_dir / "broken.json").write_text("{invalid json content")
		(inbox_dir / "good.json").write_text(json.dumps([
			{"from": "w1", "type": "discovery", "text": "Valid message"},
		]))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("Valid message" in d for d in discoveries)

	def test_empty_inbox_file(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		(inbox_dir / "empty.json").write_text("[]")
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert discoveries == []

	def test_only_last_20_messages_read(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"from": "w1", "type": "discovery", "text": f"msg-{i}"} for i in range(30)]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		# Should not include messages 0-9
		assert not any("msg-5" in d for d in discoveries)
		# Should include messages 10-29
		assert any("msg-25" in d for d in discoveries)

	def test_sender_from_field(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"from": "parser-agent", "type": "discovery", "text": "Bug found"}]
		(inbox_dir / "agent.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("[parser-agent]" in d for d in discoveries)

	def test_sender_falls_back_to_filename(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"type": "discovery", "text": "No from field"}]
		(inbox_dir / "my-agent.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("[my-agent]" in d for d in discoveries)

	def test_discoveries_from_completed_task_json_result(self, tmp_path: Path) -> None:
		ctx = self._make_ctx(tmp_path)
		result = json.dumps({"discoveries": ["Root cause is a null pointer"], "status": "completed"})
		tasks = [SwarmTask(title="Debug crash", status=TaskStatus.COMPLETED, result_summary=result)]
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries(tasks=tasks)
		assert any("null pointer" in d for d in discoveries)
		assert any("[task:Debug crash]" in d for d in discoveries)

	def test_non_json_task_result_skipped(self, tmp_path: Path) -> None:
		ctx = self._make_ctx(tmp_path)
		tasks = [SwarmTask(title="Simple fix", status=TaskStatus.COMPLETED, result_summary="Just fixed it")]
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries(tasks=tasks)
		# Plain text result_summary should not cause errors
		assert not any("Just fixed it" in d for d in discoveries)

	def test_task_result_without_discoveries_key(self, tmp_path: Path) -> None:
		ctx = self._make_ctx(tmp_path)
		result = json.dumps({"status": "completed", "summary": "Done"})
		tasks = [SwarmTask(title="Task", status=TaskStatus.COMPLETED, result_summary=result)]
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries(tasks=tasks)
		assert not any("Done" in d for d in discoveries)

	def test_pending_task_results_ignored(self, tmp_path: Path) -> None:
		ctx = self._make_ctx(tmp_path)
		result = json.dumps({"discoveries": ["Should not appear"]})
		tasks = [SwarmTask(title="WIP", status=TaskStatus.PENDING, result_summary=result)]
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries(tasks=tasks)
		assert not any("Should not appear" in d for d in discoveries)

	def test_knowledge_items_included(self, tmp_path: Path) -> None:
		config = _make_config()
		config.target.resolved_path = str(tmp_path)
		db = _make_db()
		knowledge_item = MagicMock()
		knowledge_item.content = "Important architectural decision"
		db.get_knowledge_for_mission.return_value = [knowledge_item]
		ctx = ContextSynthesizer(config, db, "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("architectural decision" in d for d in discoveries)
		assert any("[knowledge]" in d for d in discoveries)

	def test_db_exception_handled(self, tmp_path: Path) -> None:
		config = _make_config()
		config.target.resolved_path = str(tmp_path)
		db = _make_db()
		db.get_knowledge_for_mission.side_effect = RuntimeError("DB down")
		ctx = ContextSynthesizer(config, db, "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert isinstance(discoveries, list)

	def test_multiple_inbox_files_aggregated(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox_dir(tmp_path)
		(inbox_dir / "agent-1.json").write_text(json.dumps([
			{"from": "a1", "type": "discovery", "text": "From agent 1"},
		]))
		(inbox_dir / "agent-2.json").write_text(json.dumps([
			{"from": "a2", "type": "discovery", "text": "From agent 2"},
		]))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("From agent 1" in d for d in discoveries)
		assert any("From agent 2" in d for d in discoveries)

	def test_rotate_called_during_discovery_read(self, tmp_path: Path) -> None:
		"""Large inboxes should be rotated during reads (lazy rotation)."""
		inbox_dir = self._setup_inbox_dir(tmp_path)
		messages = [{"from": "w1", "type": "report", "text": f"msg-{i}"} for i in range(600)]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = self._make_ctx(tmp_path)
		with patch.object(Path, "home", return_value=tmp_path):
			ctx._get_recent_discoveries()
		# After reading, the inbox should have been rotated
		remaining = json.loads((inbox_dir / "w1.json").read_text())
		assert len(remaining) == DEFAULT_KEEP_MESSAGES


class TestDiscoverSkills:
	def test_discovers_skills_in_project(self, tmp_path: Path) -> None:
		config = _make_config()
		config.target.resolved_path = str(tmp_path)
		skills_dir = tmp_path / ".claude" / "skills" / "test-skill"
		skills_dir.mkdir(parents=True)
		(skills_dir / "SKILL.md").write_text("---\nname: test-skill\n---\nDo things")
		ctx = ContextSynthesizer(config, _make_db(), "test-team")
		skills = ctx._discover_skills()
		assert "test-skill" in skills

	def test_no_skills_dir(self, tmp_path: Path) -> None:
		config = _make_config()
		config.target.resolved_path = str(tmp_path)
		ctx = ContextSynthesizer(config, _make_db(), "test-team")
		assert ctx._discover_skills() == []

	def test_skill_dir_without_skill_md_excluded(self, tmp_path: Path) -> None:
		config = _make_config()
		config.target.resolved_path = str(tmp_path)
		skills_dir = tmp_path / ".claude" / "skills" / "incomplete"
		skills_dir.mkdir(parents=True)
		# No SKILL.md
		ctx = ContextSynthesizer(config, _make_db(), "test-team")
		assert ctx._discover_skills() == []


class TestRenderEdgeCases:
	def test_render_empty_state(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		rendered = ctx.render_for_planner(state)
		assert "## Mission" in rendered
		assert "## Meta" in rendered
		assert "None currently active" in rendered

	def test_render_no_task_progress_when_no_tasks(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		rendered = ctx.render_for_planner(state)
		assert "Task Progress" not in rendered

	def test_render_meta_section_formatting(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[], total_cost_usd=1.5, wall_time_seconds=300)
		rendered = ctx.render_for_planner(state)
		assert "$1.50" in rendered
		assert "5.0min" in rendered

	def test_render_files_in_flight_section(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		task = SwarmTask(id="t1", title="Edit", status=TaskStatus.IN_PROGRESS, files_hint=["a.py", "b.py"])
		agent = SwarmAgent(name="w1", status=AgentStatus.WORKING, current_task_id="t1")
		state = ctx.build_state(agents=[agent], tasks=[task])
		rendered = ctx.render_for_planner(state)
		assert "Files Currently Being Modified" in rendered
		assert "a.py" in rendered
		assert "b.py" in rendered

	def test_render_available_skills(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		state.available_skills = ["test-tool", "debug-helper"]
		rendered = ctx.render_for_planner(state)
		assert "Available Skills" in rendered
		assert "test-tool" in rendered

	def test_render_discoveries_section(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		state.recent_discoveries = ["Found race condition in parser", "DB needs index"]
		rendered = ctx.render_for_planner(state)
		assert "Recent Discoveries" in rendered
		assert "race condition" in rendered

	def test_render_discoveries_capped_at_10(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		state.recent_discoveries = [f"discovery-{i}" for i in range(20)]
		rendered = ctx.render_for_planner(state)
		assert "discovery-9" in rendered
		assert "discovery-10" not in rendered

	def test_render_completed_tasks_capped_at_10(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title=f"task-{i}", status=TaskStatus.COMPLETED, result_summary=f"done-{i}")
			for i in range(15)
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		rendered = ctx.render_for_planner(state)
		assert "done-9" in rendered
		assert "done-10" not in rendered

	def test_render_long_result_summary_truncated(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [
			SwarmTask(title="Task", status=TaskStatus.COMPLETED, result_summary="x" * 200),
		]
		state = ctx.build_state(agents=[], tasks=tasks)
		rendered = ctx.render_for_planner(state)
		# result_summary is truncated to 120 chars in render
		assert "x" * 120 in rendered
		assert "x" * 121 not in rendered

	def test_render_agent_with_unknown_task_id(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		agent = SwarmAgent(name="w1", status=AgentStatus.WORKING, current_task_id="nonexistent")
		state = ctx.build_state(agents=[agent], tasks=[])
		rendered = ctx.render_for_planner(state)
		assert "task: nonexistent" in rendered

	def test_render_dead_agents_capped_at_10(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		dead = [SwarmAgent(name=f"dead-{i}", status=AgentStatus.DEAD) for i in range(15)]
		state = ctx.build_state(agents=[], tasks=[], dead_agent_history=dead)
		rendered = ctx.render_for_planner(state)
		assert "dead-14" in rendered
		# Only last 10 shown
		assert "dead-4" not in rendered

	def test_render_blocked_task_status(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		tasks = [SwarmTask(id="t1", title="Blocked task", status=TaskStatus.BLOCKED)]
		state = ctx.build_state(agents=[], tasks=tasks)
		rendered = ctx.render_for_planner(state)
		assert "[BLOCKED]" in rendered

	def test_render_unknown_dependency(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		task = SwarmTask(id="t1", title="Has dep", status=TaskStatus.PENDING, depends_on=["unknown-id"])
		state = ctx.build_state(agents=[], tasks=[task])
		rendered = ctx.render_for_planner(state)
		assert "unknown-id [unknown]" in rendered


class TestBuildStateIntegration:
	def test_cost_and_wall_time_passed_through(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[], total_cost_usd=5.25, wall_time_seconds=1800.0)
		assert state.total_cost_usd == 5.25
		assert state.wall_time_seconds == 1800.0

	def test_dead_agent_history_passed_through(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		dead = [SwarmAgent(name="old", status=AgentStatus.DEAD, tasks_completed=5)]
		state = ctx.build_state(agents=[], tasks=[], dead_agent_history=dead)
		assert len(state.dead_agent_history) == 1
		assert state.dead_agent_history[0].tasks_completed == 5

	def test_files_in_flight_deduped_and_sorted(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		t1 = SwarmTask(id="t1", title="Task 1", status=TaskStatus.IN_PROGRESS, files_hint=["b.py", "a.py"])
		t2 = SwarmTask(id="t2", title="Task 2", status=TaskStatus.IN_PROGRESS, files_hint=["a.py", "c.py"])
		a1 = SwarmAgent(name="w1", status=AgentStatus.WORKING, current_task_id="t1")
		a2 = SwarmAgent(name="w2", status=AgentStatus.WORKING, current_task_id="t2")
		state = ctx.build_state(agents=[a1, a2], tasks=[t1, t2])
		assert state.files_in_flight == ["a.py", "b.py", "c.py"]

	def test_files_in_flight_ignores_idle_agents(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		task = SwarmTask(id="t1", title="Task", status=TaskStatus.IN_PROGRESS, files_hint=["x.py"])
		agent = SwarmAgent(name="w1", status=AgentStatus.IDLE, current_task_id="t1")
		state = ctx.build_state(agents=[agent], tasks=[task])
		assert state.files_in_flight == []

	def test_none_core_test_results_defaults_to_empty_dict(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[], core_test_results=None)
		assert state.core_test_results == {}

	def test_none_dead_agent_history_defaults_to_empty_list(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[], dead_agent_history=None)
		assert state.dead_agent_history == []

	def test_metric_tracking_bounded(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		# Run 25 cycles to exceed the 20-metric bound
		for i in range(25):
			ctx.build_state(agents=[], tasks=[], core_test_results={"pass": i})
		assert len(ctx._metric_history["test_pass_count"]) <= 20


class TestParseStructuredReport:
	def test_basic_fields(self) -> None:
		msg = {"from": "agent-1", "type": "report", "text": "Working on parser"}
		result = parse_structured_report(msg)
		assert result["from"] == "agent-1"
		assert result["type"] == "report"
		assert result["text"] == "Working on parser"

	def test_defaults_for_missing_fields(self) -> None:
		result = parse_structured_report({})
		assert result["from"] == "unknown"
		assert result["type"] == ""
		assert result["text"] == ""

	def test_valid_status_field(self) -> None:
		for status in ("working", "blocked", "completed"):
			result = parse_structured_report({"status": status})
			assert result["status"] == status

	def test_invalid_status_excluded(self) -> None:
		result = parse_structured_report({"status": "invalid"})
		assert "status" not in result

	def test_non_string_status_excluded(self) -> None:
		result = parse_structured_report({"status": 42})
		assert "status" not in result

	def test_progress_field(self) -> None:
		result = parse_structured_report({"progress": "50% done"})
		assert result["progress"] == "50% done"

	def test_empty_progress_excluded(self) -> None:
		result = parse_structured_report({"progress": ""})
		assert "progress" not in result

	def test_files_changed_field(self) -> None:
		result = parse_structured_report({"files_changed": ["a.py", "b.py"]})
		assert result["files_changed"] == ["a.py", "b.py"]

	def test_files_changed_non_string_items_excluded(self) -> None:
		result = parse_structured_report({"files_changed": [1, 2]})
		assert "files_changed" not in result

	def test_files_changed_non_list_excluded(self) -> None:
		result = parse_structured_report({"files_changed": "a.py"})
		assert "files_changed" not in result

	def test_tests_passing_int(self) -> None:
		result = parse_structured_report({"tests_passing": 42})
		assert result["tests_passing"] == 42

	def test_tests_passing_float_coerced_to_int(self) -> None:
		result = parse_structured_report({"tests_passing": 42.7})
		assert result["tests_passing"] == 42

	def test_tests_passing_negative_excluded(self) -> None:
		result = parse_structured_report({"tests_passing": -1})
		assert "tests_passing" not in result

	def test_error_field(self) -> None:
		result = parse_structured_report({"error": "Segfault"})
		assert result["error"] == "Segfault"

	def test_empty_error_excluded(self) -> None:
		result = parse_structured_report({"error": ""})
		assert "error" not in result

	def test_timestamp_preserved(self) -> None:
		result = parse_structured_report({"timestamp": "2025-01-01T00:00:00Z"})
		assert result["timestamp"] == "2025-01-01T00:00:00Z"

	def test_all_fields_together(self) -> None:
		msg = {
			"from": "w1", "type": "report", "text": "Done",
			"status": "completed", "progress": "100%",
			"files_changed": ["x.py"], "tests_passing": 50,
			"error": "", "timestamp": "2025-01-01",
		}
		result = parse_structured_report(msg)
		assert result["status"] == "completed"
		assert result["progress"] == "100%"
		assert result["files_changed"] == ["x.py"]
		assert result["tests_passing"] == 50
		assert "error" not in result  # empty string excluded
		assert result["timestamp"] == "2025-01-01"


class TestGetAgentReports:
	def _setup_inbox(self, tmp_path: Path, team_name: str = "test-team") -> Path:
		inbox_dir = tmp_path / ".claude" / "teams" / team_name / "inboxes"
		inbox_dir.mkdir(parents=True)
		return inbox_dir

	def test_returns_latest_structured_report(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox(tmp_path)
		messages = [
			{"from": "w1", "type": "report", "text": "Starting", "status": "working"},
			{"from": "w1", "type": "report", "text": "Done", "status": "completed"},
		]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			reports = ctx.get_agent_reports()
		assert reports["w1"]["status"] == "completed"

	def test_empty_inbox_dir(self, tmp_path: Path) -> None:
		self._setup_inbox(tmp_path)
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			reports = ctx.get_agent_reports()
		assert reports == {}

	def test_missing_inbox_dir(self, tmp_path: Path) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			reports = ctx.get_agent_reports()
		assert reports == {}

	def test_skips_messages_without_status(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox(tmp_path)
		messages = [
			{"from": "w1", "type": "report", "text": "Just a report"},
		]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			reports = ctx.get_agent_reports()
		assert "w1" not in reports

	def test_multiple_senders(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox(tmp_path)
		# Messages from different senders in same inbox
		messages = [
			{"from": "w1", "type": "report", "text": "A", "status": "working"},
			{"from": "w2", "type": "report", "text": "B", "status": "completed"},
		]
		(inbox_dir / "team-lead.json").write_text(json.dumps(messages))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			reports = ctx.get_agent_reports()
		assert reports["w1"]["status"] == "working"
		assert reports["w2"]["status"] == "completed"

	def test_malformed_json_skipped(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox(tmp_path)
		(inbox_dir / "broken.json").write_text("{bad json")
		(inbox_dir / "good.json").write_text(json.dumps([
			{"from": "w1", "status": "working", "type": "report", "text": "ok"},
		]))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			reports = ctx.get_agent_reports()
		assert "w1" in reports

	def test_sender_falls_back_to_file_stem(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox(tmp_path)
		messages = [{"type": "report", "text": "hi", "status": "working"}]
		(inbox_dir / "my-agent.json").write_text(json.dumps(messages))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			reports = ctx.get_agent_reports()
		assert "my-agent" in reports


class TestGetHumanDirectives:
	def _setup_inbox(self, tmp_path: Path, team_name: str = "test-team") -> Path:
		inbox_dir = tmp_path / ".claude" / "teams" / team_name / "inboxes"
		inbox_dir.mkdir(parents=True)
		return inbox_dir

	def test_reads_directives_from_leader_inbox(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox(tmp_path)
		messages = [
			{"type": "directive", "text": "Focus on tests first", "timestamp": "2025-01-01"},
			{"type": "report", "text": "Ignore this"},
		]
		(inbox_dir / "team-lead.json").write_text(json.dumps(messages))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			directives = ctx._get_human_directives()
		assert len(directives) == 1
		assert "Focus on tests first" in directives[0]
		assert "2025-01-01" in directives[0]

	def test_no_directives(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox(tmp_path)
		messages = [{"type": "report", "text": "Just a report"}]
		(inbox_dir / "team-lead.json").write_text(json.dumps(messages))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			directives = ctx._get_human_directives()
		assert directives == []

	def test_missing_inbox_dir(self, tmp_path: Path) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			directives = ctx._get_human_directives()
		assert directives == []

	def test_malformed_json(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox(tmp_path)
		(inbox_dir / "team-lead.json").write_text("{bad")
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			directives = ctx._get_human_directives()
		assert directives == []

	def test_only_last_20_messages_checked(self, tmp_path: Path) -> None:
		inbox_dir = self._setup_inbox(tmp_path)
		messages = [{"type": "directive", "text": f"dir-{i}", "timestamp": f"t{i}"} for i in range(30)]
		(inbox_dir / "team-lead.json").write_text(json.dumps(messages))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			directives = ctx._get_human_directives()
		# Only last 20 messages read, so dir-0 through dir-9 excluded
		assert not any("dir-5" in d for d in directives)
		assert any("dir-25" in d for d in directives)


class TestDiscoverTools:
	def test_returns_tool_names(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		mock_tool = MagicMock()
		mock_tool.name = "my-tool"
		mock_registry_cls = MagicMock()
		mock_registry_cls.return_value.list_all.return_value = [mock_tool]
		mock_module = MagicMock(MCPToolRegistry=mock_registry_cls)
		with patch.dict("sys.modules", {"autodev.mcp_registry": mock_module}):
			tools = ctx._discover_tools()
		assert tools == ["my-tool"]

	def test_import_error_returns_empty(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.dict("sys.modules", {"autodev.mcp_registry": None}):
			tools = ctx._discover_tools()
		assert tools == []

	def test_caps_at_20_tools(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		mock_tools = []
		for i in range(30):
			t = MagicMock()
			t.name = f"tool-{i}"
			mock_tools.append(t)
		mock_registry_cls = MagicMock()
		mock_registry_cls.return_value.list_all.return_value = mock_tools
		mock_module = MagicMock(MCPToolRegistry=mock_registry_cls)
		with patch.dict("sys.modules", {"autodev.mcp_registry": mock_module}):
			tools = ctx._discover_tools()
		assert len(tools) == 20


class TestRenderHumanDirectives:
	def test_directives_shown_in_render(self, tmp_path: Path) -> None:
		inbox_dir = tmp_path / ".claude" / "teams" / "test-team" / "inboxes"
		inbox_dir.mkdir(parents=True)
		messages = [{"type": "directive", "text": "Stop all work", "timestamp": "now"}]
		(inbox_dir / "team-lead.json").write_text(json.dumps(messages))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			state = ctx.build_state(agents=[], tasks=[])
			rendered = ctx.render_for_planner(state)
		assert "HUMAN DIRECTIVES" in rendered
		assert "Stop all work" in rendered

	def test_no_directives_section_when_empty(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		rendered = ctx.render_for_planner(state)
		assert "HUMAN DIRECTIVES" not in rendered


class TestRenderAgentReports:
	def test_agent_reports_in_render(self, tmp_path: Path) -> None:
		inbox_dir = tmp_path / ".claude" / "teams" / "test-team" / "inboxes"
		inbox_dir.mkdir(parents=True)
		messages = [
			{
				"from": "w1", "type": "report", "text": "Working",
				"status": "working", "progress": "50% done",
				"tests_passing": 10, "files_changed": ["a.py"],
			},
		]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			state = ctx.build_state(agents=[], tasks=[])
			rendered = ctx.render_for_planner(state)
		assert "Agent Progress Reports" in rendered
		assert "**w1**" in rendered
		assert "50% done" in rendered
		assert "tests_passing=10" in rendered
		assert "a.py" in rendered

	def test_no_reports_section_when_empty(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		rendered = ctx.render_for_planner(state)
		assert "Agent Progress Reports" not in rendered

	def test_error_shown_truncated(self, tmp_path: Path) -> None:
		inbox_dir = tmp_path / ".claude" / "teams" / "test-team" / "inboxes"
		inbox_dir.mkdir(parents=True)
		messages = [
			{
				"from": "w1", "type": "report", "text": "Stuck",
				"status": "blocked", "error": "x" * 200,
			},
		]
		(inbox_dir / "w1.json").write_text(json.dumps(messages))
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			state = ctx.build_state(agents=[], tasks=[])
			rendered = ctx.render_for_planner(state)
		# Error is truncated to 100 chars in render
		assert "x" * 100 in rendered
		assert "x" * 101 not in rendered


class TestDirectiveInDiscoveries:
	"""Directive message type should be included in _get_recent_discoveries."""

	def test_directive_type_included(self, tmp_path: Path) -> None:
		inbox_dir = tmp_path / ".claude" / "teams" / "test-team" / "inboxes"
		inbox_dir.mkdir(parents=True)
		messages = [{"from": "human", "type": "directive", "text": "Prioritize tests"}]
		(inbox_dir / "team-lead.json").write_text(json.dumps(messages))
		config = _make_config()
		config.target.resolved_path = str(tmp_path)
		ctx = ContextSynthesizer(config, _make_db(), "test-team")
		with patch.object(Path, "home", return_value=tmp_path):
			discoveries = ctx._get_recent_discoveries()
		assert any("Prioritize tests" in d for d in discoveries)
		assert any("(directive)" in d for d in discoveries)


class TestContextCostBreakdown:
	def test_cost_breakdown_in_render(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(
			agents=[], tasks=[],
			total_cost_usd=5.50,
			agent_costs={"w1": 3.00, "w2": 2.50},
		)
		rendered = ctx.render_for_planner(state)
		assert "$5.50" in rendered
		assert "Avg cost/agent: $2.75" in rendered
		assert "Top spender: w1 ($3.00)" in rendered

	def test_no_cost_breakdown_when_zero(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[], total_cost_usd=0.0)
		rendered = ctx.render_for_planner(state)
		assert "Avg cost/agent" not in rendered
		assert "Top spender" not in rendered

	def test_agent_costs_passed_through(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(
			agents=[], tasks=[],
			agent_costs={"w1": 1.0, "w2": 2.0},
		)
		assert state.agent_costs == {"w1": 1.0, "w2": 2.0}


class TestContextRecentChanges:
	def test_recent_changes_in_render(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(
			agents=[], tasks=[],
			recent_file_changes={"worker-1": ["src/parser.py", "src/lexer.py"]},
		)
		rendered = ctx.render_for_planner(state)
		assert "Recent Changes" in rendered
		assert "worker-1" in rendered
		assert "src/parser.py" in rendered
		assert "src/lexer.py" in rendered

	def test_no_recent_changes_section_when_empty(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(agents=[], tasks=[])
		rendered = ctx.render_for_planner(state)
		assert "Recent Changes" not in rendered

	def test_recent_file_changes_passed_through(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		state = ctx.build_state(
			agents=[], tasks=[],
			recent_file_changes={"w1": ["a.py"]},
		)
		assert state.recent_file_changes == {"w1": ["a.py"]}

	def test_recent_changes_truncated_when_many_files(self) -> None:
		ctx = ContextSynthesizer(_make_config(), _make_db(), "test-team")
		many_files = [f"src/file_{i}.py" for i in range(15)]
		state = ctx.build_state(
			agents=[], tasks=[],
			recent_file_changes={"worker-1": many_files},
		)
		rendered = ctx.render_for_planner(state)
		assert "(+5 more)" in rendered
