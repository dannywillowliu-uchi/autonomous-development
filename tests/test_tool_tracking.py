"""Tests for tool call tracking in swarm agents.

Tests the stream-json parsing, tool call recording in DB,
MCP status tracking, and failure summary queries.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autodev.config import SwarmConfig
from autodev.db import Database
from autodev.swarm.controller import SwarmController

# -- Helpers ------------------------------------------------------------------


def _make_config(tmp_path: Path) -> MagicMock:
	config = MagicMock()
	config.target.name = "test-project"
	config.target.objective = "Track tool calls"
	config.target.resolved_path = str(tmp_path)
	config.notification = MagicMock()
	return config


def _make_swarm_config(**overrides: object) -> SwarmConfig:
	sc = SwarmConfig()
	for k, v in overrides.items():
		setattr(sc, k, v)
	return sc


def _make_db() -> MagicMock:
	db = MagicMock()
	db.get_knowledge_for_mission.return_value = []
	return db


@pytest.fixture
def ctrl(tmp_path: Path) -> SwarmController:
	"""Minimal SwarmController with mocked config and db."""
	return SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())


@pytest.fixture
def real_db() -> Database:
	"""Real Database backed by a temp file."""
	with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
		db_path = f.name
	return Database(db_path)


# -- Stream Event Parsing Tests ----------------------------------------------


class TestParseStreamEvent:
	"""Tests for _parse_stream_event on SwarmController."""

	def test_parse_stream_init_captures_mcp_status(self, ctrl: SwarmController) -> None:
		"""Init event with mcp_servers should record status for each server."""
		event = {
			"type": "system",
			"subtype": "init",
			"mcp_servers": [
				{"name": "obsidian", "status": "connected"},
				{"name": "stitch", "status": "failed"},
			],
		}
		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		ctrl._parse_stream_event(
			json.dumps(event), "agent-1", "worker-1", pending, tool_calls,
		)

		assert ctrl._db.record_mcp_status.call_count == 2
		calls = ctrl._db.record_mcp_status.call_args_list
		assert calls[0].kwargs["server_name"] == "obsidian"
		assert calls[0].kwargs["status"] == "connected"
		assert calls[1].kwargs["server_name"] == "stitch"
		assert calls[1].kwargs["status"] == "failed"

	def test_parse_stream_tool_use_and_result(self, ctrl: SwarmController) -> None:
		"""Tool use followed by tool result should produce one tool_calls entry."""
		tool_use_event = {
			"type": "assistant",
			"message": {
				"content": [
					{"id": "tu_1", "name": "Read", "type": "tool_use", "input": {}},
				],
			},
		}
		tool_result_event = {
			"type": "user",
			"message": {
				"content": [
					{
						"type": "tool_result",
						"tool_use_id": "tu_1",
						"content": "file contents here",
						"is_error": False,
					},
				],
			},
		}

		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		ctrl._parse_stream_event(
			json.dumps(tool_use_event), "agent-1", "worker-1", pending, tool_calls,
		)
		# Small delay to ensure duration_ms > 0
		time.sleep(0.01)
		ctrl._parse_stream_event(
			json.dumps(tool_result_event), "agent-1", "worker-1", pending, tool_calls,
		)

		assert len(tool_calls) == 1
		assert tool_calls[0]["tool_name"] == "Read"
		assert tool_calls[0]["success"] is True
		assert tool_calls[0]["duration_ms"] > 0
		assert tool_calls[0]["error_message"] == ""

	def test_parse_stream_tool_error(self, ctrl: SwarmController) -> None:
		"""Tool result with is_error=True should capture error_message."""
		tool_use_event = {
			"type": "assistant",
			"message": {
				"content": [
					{"id": "tu_err", "name": "Bash", "type": "tool_use", "input": {}},
				],
			},
		}
		tool_result_event = {
			"type": "user",
			"message": {
				"content": [
					{
						"type": "tool_result",
						"tool_use_id": "tu_err",
						"content": "Command failed: exit code 1",
						"is_error": True,
					},
				],
			},
		}

		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		ctrl._parse_stream_event(
			json.dumps(tool_use_event), "agent-1", "worker-1", pending, tool_calls,
		)
		ctrl._parse_stream_event(
			json.dumps(tool_result_event), "agent-1", "worker-1", pending, tool_calls,
		)

		assert len(tool_calls) == 1
		assert tool_calls[0]["success"] is False
		assert "Command failed" in tool_calls[0]["error_message"]

	def test_parse_stream_result_extracts_text(self, ctrl: SwarmController) -> None:
		"""Result event should store text in _agent_final_results."""
		event = {
			"type": "result",
			"result": 'AD_RESULT: {"status": "completed"}',
		}
		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		ctrl._parse_stream_event(
			json.dumps(event), "agent-42", "worker-1", pending, tool_calls,
		)

		assert "agent-42" in ctrl._agent_final_results
		assert "AD_RESULT" in ctrl._agent_final_results["agent-42"]

	def test_parse_stream_invalid_json_skipped(self, ctrl: SwarmController) -> None:
		"""Non-JSON lines should be silently skipped."""
		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		# Should not raise
		ctrl._parse_stream_event(
			"this is not json", "agent-1", "worker-1", pending, tool_calls,
		)

		assert len(tool_calls) == 0
		assert ctrl._db.record_mcp_status.call_count == 0
		assert ctrl._db.record_tool_call.call_count == 0

	def test_mcp_tool_name_parsing(self, ctrl: SwarmController) -> None:
		"""MCP-prefixed tool names should extract the server name."""
		# MCP tool: mcp__obsidian__read_note -> mcp_server = "obsidian"
		mcp_tool_use = {
			"type": "assistant",
			"message": {
				"content": [
					{
						"id": "tu_mcp",
						"name": "mcp__obsidian__read_note",
						"type": "tool_use",
						"input": {},
					},
				],
			},
		}
		# Regular tool: Read -> mcp_server = ""
		regular_tool_use = {
			"type": "assistant",
			"message": {
				"content": [
					{"id": "tu_reg", "name": "Read", "type": "tool_use", "input": {}},
				],
			},
		}

		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		ctrl._parse_stream_event(
			json.dumps(mcp_tool_use), "agent-1", "worker-1", pending, tool_calls,
		)
		ctrl._parse_stream_event(
			json.dumps(regular_tool_use), "agent-1", "worker-1", pending, tool_calls,
		)

		# Verify pending entries have correct mcp_server
		assert pending["tu_mcp"]["mcp_server"] == "obsidian"
		assert pending["tu_reg"]["mcp_server"] == ""

		# Now send results to get tool_calls entries
		for tool_id in ["tu_mcp", "tu_reg"]:
			result_event = {
				"type": "user",
				"message": {
					"content": [
						{
							"type": "tool_result",
							"tool_use_id": tool_id,
							"content": "ok",
							"is_error": False,
						},
					],
				},
			}
			ctrl._parse_stream_event(
				json.dumps(result_event), "agent-1", "worker-1", pending, tool_calls,
			)

		assert len(tool_calls) == 2
		mcp_call = next(c for c in tool_calls if c["tool_name"] == "mcp__obsidian__read_note")
		reg_call = next(c for c in tool_calls if c["tool_name"] == "Read")
		assert mcp_call["mcp_server"] == "obsidian"
		assert reg_call["mcp_server"] == ""


# -- DB Method Tests ----------------------------------------------------------


class TestDBToolTracking:
	"""Tests for tool call and MCP status DB methods using a real Database."""

	def test_db_record_and_query_tool_calls(self, real_db: Database) -> None:
		"""record_tool_call should store data retrievable by get_tool_usage."""
		real_db.record_tool_call(
			run_id="run-1",
			agent_id="agent-1",
			agent_name="worker-1",
			tool_name="Read",
			mcp_server="",
			success=True,
			error_message="",
			timestamp="2026-03-15T10:00:00",
			duration_ms=42.5,
		)

		results = real_db.get_tool_usage(run_id="run-1")
		assert len(results) == 1
		row = results[0]
		assert row["run_id"] == "run-1"
		assert row["agent_id"] == "agent-1"
		assert row["agent_name"] == "worker-1"
		assert row["tool_name"] == "Read"
		assert row["mcp_server"] == ""
		assert row["is_error"] is False
		assert row["error_message"] == ""
		assert row["duration_ms"] == pytest.approx(42.5)

	def test_tool_failure_summary_groups_correctly(self, real_db: Database) -> None:
		"""Failure summary should group by tool and order by most failures."""
		# Record 3 failures for Read
		for i in range(3):
			real_db.record_tool_call(
				run_id="run-1",
				agent_id="agent-1",
				agent_name="worker-1",
				tool_name="Read",
				mcp_server="",
				success=False,
				error_message=f"Read error {i}",
				timestamp=f"2026-03-15T10:0{i}:00",
				duration_ms=10.0,
			)

		# Record 1 failure for Write
		real_db.record_tool_call(
			run_id="run-1",
			agent_id="agent-1",
			agent_name="worker-1",
			tool_name="Write",
			mcp_server="",
			success=False,
			error_message="Write error",
			timestamp="2026-03-15T10:03:00",
			duration_ms=15.0,
		)

		summary = real_db.get_tool_failure_summary(run_id="run-1")
		assert len(summary) == 2

		# Most failures first
		assert summary[0]["tool_name"] == "Read"
		assert summary[0]["failure_count"] == 3
		assert summary[1]["tool_name"] == "Write"
		assert summary[1]["failure_count"] == 1

	def test_db_mcp_status(self, real_db: Database) -> None:
		"""record_mcp_status and get_mcp_status should round-trip correctly."""
		real_db.record_mcp_status(
			run_id="run-1",
			agent_id="agent-1",
			server_name="obsidian",
			status="connected",
			timestamp="2026-03-15T10:00:00",
		)
		real_db.record_mcp_status(
			run_id="run-1",
			agent_id="agent-1",
			server_name="stitch",
			status="failed",
			timestamp="2026-03-15T10:00:01",
		)

		results = real_db.get_mcp_status(run_id="run-1")
		assert len(results) == 2
		names = {r["server_name"] for r in results}
		assert names == {"obsidian", "stitch"}

		failed = [r for r in results if r["status"] == "failed"]
		assert len(failed) == 1
		assert failed[0]["server_name"] == "stitch"


# -- Integration: AD_RESULT from stream-json ---------------------------------


class TestADResultFromStream:
	"""Verify that _agent_final_results is preferred over _agent_outputs."""

	def test_ad_result_from_stream_json(self, ctrl: SwarmController) -> None:
		"""When _agent_final_results has a value, it should be used for AD_RESULT parsing."""
		agent_id = "agent-stream"
		ad_result_text = 'AD_RESULT:{"status":"completed","summary":"done","commits":[],"files_changed":[]}'

		# Simulate stream-json result event populating _agent_final_results
		ctrl._agent_final_results[agent_id] = ad_result_text

		# Also set _agent_outputs with different (stale) content
		ctrl._agent_outputs[agent_id] = "some raw text output without AD_RESULT"

		# _agent_final_results should take priority
		final = ctrl._agent_final_results.get(agent_id, "")
		fallback = ctrl._agent_outputs.get(agent_id, "")

		# The code uses: output = self._agent_final_results.pop(agent_id, "") or self._agent_outputs.pop(agent_id, "")
		output = final or fallback
		assert "AD_RESULT" in output
		assert '"completed"' in output

		# Verify _parse_ad_result works on this output
		result = ctrl._parse_ad_result(output)
		assert result is not None
		assert result["status"] == "completed"


# -- Edge Cases ---------------------------------------------------------------


class TestStreamParsingEdgeCases:
	"""Edge cases for stream event parsing."""

	def test_tool_result_without_prior_use_ignored(self, ctrl: SwarmController) -> None:
		"""A tool_result with no matching pending tool_use should be silently ignored."""
		orphan_result = {
			"type": "user",
			"message": {
				"content": [
					{
						"type": "tool_result",
						"tool_use_id": "tu_orphan",
						"content": "result without use",
						"is_error": False,
					},
				],
			},
		}
		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		ctrl._parse_stream_event(
			json.dumps(orphan_result), "agent-1", "worker-1", pending, tool_calls,
		)

		assert len(tool_calls) == 0

	def test_multiple_tool_uses_in_single_event(self, ctrl: SwarmController) -> None:
		"""Multiple tool_use blocks in one assistant event should all be tracked."""
		event = {
			"type": "assistant",
			"message": {
				"content": [
					{"id": "tu_a", "name": "Read", "type": "tool_use", "input": {}},
					{"id": "tu_b", "name": "Grep", "type": "tool_use", "input": {}},
				],
			},
		}
		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		ctrl._parse_stream_event(
			json.dumps(event), "agent-1", "worker-1", pending, tool_calls,
		)

		assert "tu_a" in pending
		assert "tu_b" in pending
		assert pending["tu_a"]["name"] == "Read"
		assert pending["tu_b"]["name"] == "Grep"

	def test_tool_result_with_list_content(self, ctrl: SwarmController) -> None:
		"""Tool result content as a list of text blocks should be concatenated."""
		tool_use_event = {
			"type": "assistant",
			"message": {
				"content": [
					{"id": "tu_list", "name": "Read", "type": "tool_use", "input": {}},
				],
			},
		}
		tool_result_event = {
			"type": "user",
			"message": {
				"content": [
					{
						"type": "tool_result",
						"tool_use_id": "tu_list",
						"content": [
							{"type": "text", "text": "line 1"},
							{"type": "text", "text": "line 2"},
						],
						"is_error": True,
					},
				],
			},
		}

		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		ctrl._parse_stream_event(
			json.dumps(tool_use_event), "agent-1", "worker-1", pending, tool_calls,
		)
		ctrl._parse_stream_event(
			json.dumps(tool_result_event), "agent-1", "worker-1", pending, tool_calls,
		)

		assert len(tool_calls) == 1
		assert tool_calls[0]["success"] is False
		# Error message from concatenated list content
		assert "line 1" in tool_calls[0]["error_message"]
		assert "line 2" in tool_calls[0]["error_message"]

	def test_empty_event_ignored(self, ctrl: SwarmController) -> None:
		"""Empty string should be silently ignored."""
		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		ctrl._parse_stream_event("", "agent-1", "worker-1", pending, tool_calls)
		assert len(tool_calls) == 0

	def test_init_event_without_mcp_servers(self, ctrl: SwarmController) -> None:
		"""Init event with no mcp_servers key should not crash."""
		event = {"type": "system", "subtype": "init"}
		pending: dict[str, dict] = {}
		tool_calls: list[dict] = []

		ctrl._parse_stream_event(
			json.dumps(event), "agent-1", "worker-1", pending, tool_calls,
		)
		assert ctrl._db.record_mcp_status.call_count == 0
