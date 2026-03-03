"""Tests for HITL approval timeout and auto-escalation."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

import pytest

from mission_control.config import MissionConfig
from mission_control.hitl import ApprovalGate, ApprovalRequest, ApprovalResult


def _make_config(tmp_path: Path) -> MissionConfig:
	cfg = MissionConfig()
	cfg.target.path = str(tmp_path)
	cfg.hitl.approvals_dir = ".mc-approvals"
	cfg.hitl.telegram_poll_interval = 0.01
	cfg.notifications.telegram.bot_token = ""
	cfg.notifications.telegram.chat_id = ""
	return cfg


class TestApprovalWithinTimeout:
	@pytest.mark.asyncio
	async def test_approved_via_file_within_timeout(self, tmp_path: Path) -> None:
		"""Approval received before timeout returns approved=True, timed_out=False."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push")

		async def approve_after_delay() -> None:
			await asyncio.sleep(0.05)
			approval_file = tmp_path / ".mc-approvals" / f"{req.request_id}.json"
			data = json.loads(approval_file.read_text())
			data["status"] = "approved"
			approval_file.write_text(json.dumps(data))

		task = asyncio.create_task(approve_after_delay())
		result = await gate.wait_for_approval(req, timeout_seconds=2.0)
		await task

		assert isinstance(result, ApprovalResult)
		assert result.approved is True
		assert result.timed_out is False
		assert result.elapsed_seconds < 2.0

	@pytest.mark.asyncio
	async def test_denied_via_file_within_timeout(self, tmp_path: Path) -> None:
		"""Denial received before timeout returns approved=False, timed_out=False."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push")

		async def deny_after_delay() -> None:
			await asyncio.sleep(0.05)
			approval_file = tmp_path / ".mc-approvals" / f"{req.request_id}.json"
			data = json.loads(approval_file.read_text())
			data["status"] = "denied"
			approval_file.write_text(json.dumps(data))

		task = asyncio.create_task(deny_after_delay())
		result = await gate.wait_for_approval(req, timeout_seconds=2.0)
		await task

		assert result.approved is False
		assert result.timed_out is False


class TestTimeoutExpiry:
	@pytest.mark.asyncio
	async def test_timeout_returns_timed_out(self, tmp_path: Path) -> None:
		"""No approval within timeout returns timed_out=True, approved=False."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push")

		result = await gate.wait_for_approval(req, timeout_seconds=0.1)

		assert result.approved is False
		assert result.timed_out is True
		assert result.elapsed_seconds >= 0.1

	@pytest.mark.asyncio
	async def test_timeout_finalizes_file_as_timeout(self, tmp_path: Path) -> None:
		"""Timeout updates the approval file status to 'timeout'."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push")

		await gate.wait_for_approval(req, timeout_seconds=0.1)

		approval_file = tmp_path / ".mc-approvals" / f"{req.request_id}.json"
		data = json.loads(approval_file.read_text())
		assert data["status"] == "timeout"


class TestReminderCallbacks:
	@pytest.mark.asyncio
	async def test_reminders_fired_at_intervals(self, tmp_path: Path) -> None:
		"""Reminder callback fires at configured intervals before timeout."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push", context={"description": "test push"})

		calls: list[tuple[str, float, Path]] = []

		def on_reminder(desc: str, elapsed: float, path: Path) -> None:
			calls.append((desc, elapsed, path))

		result = await gate.wait_for_approval(
			req,
			timeout_seconds=0.5,
			reminder_interval_seconds=0.1,
			reminder_callback=on_reminder,
		)

		assert result.timed_out is True
		assert len(calls) >= 2
		assert result.reminder_count == len(calls)
		for desc, elapsed, path in calls:
			assert desc == "test push"
			assert elapsed > 0
			assert path.name.endswith(".json")

	@pytest.mark.asyncio
	async def test_async_reminder_callback(self, tmp_path: Path) -> None:
		"""Async reminder callbacks are awaited correctly."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push")

		call_count = 0

		async def async_reminder(desc: str, elapsed: float, path: Path) -> None:
			nonlocal call_count
			call_count += 1

		result = await gate.wait_for_approval(
			req,
			timeout_seconds=0.3,
			reminder_interval_seconds=0.1,
			reminder_callback=async_reminder,
		)

		assert result.timed_out is True
		assert call_count >= 1
		assert result.reminder_count == call_count

	@pytest.mark.asyncio
	async def test_no_reminders_without_callback(self, tmp_path: Path) -> None:
		"""No reminder_callback -> reminder_count stays 0."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push")

		result = await gate.wait_for_approval(
			req,
			timeout_seconds=0.1,
			reminder_interval_seconds=0.05,
		)

		assert result.reminder_count == 0

	@pytest.mark.asyncio
	async def test_description_falls_back_to_gate_type(self, tmp_path: Path) -> None:
		"""Without 'description' in context, reminder uses gate_type."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="large_merge")

		calls: list[str] = []

		def on_reminder(desc: str, elapsed: float, path: Path) -> None:
			calls.append(desc)

		await gate.wait_for_approval(
			req,
			timeout_seconds=0.2,
			reminder_interval_seconds=0.05,
			reminder_callback=on_reminder,
		)

		assert len(calls) >= 1
		assert calls[0] == "large_merge"


class TestEscalationCallback:
	@pytest.mark.asyncio
	async def test_escalation_on_timeout(self, tmp_path: Path) -> None:
		"""Escalation callback is called when timeout is reached."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push", context={"description": "deploy prod"})

		escalation_calls: list[tuple[str, float, Path]] = []

		def on_escalate(desc: str, elapsed: float, path: Path) -> None:
			escalation_calls.append((desc, elapsed, path))

		result = await gate.wait_for_approval(
			req,
			timeout_seconds=0.1,
			escalation_callback=on_escalate,
		)

		assert result.timed_out is True
		assert len(escalation_calls) == 1
		desc, elapsed, path = escalation_calls[0]
		assert desc == "deploy prod"
		assert elapsed >= 0.1
		assert path.name.endswith(".json")

	@pytest.mark.asyncio
	async def test_async_escalation_callback(self, tmp_path: Path) -> None:
		"""Async escalation callbacks are awaited correctly."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push")

		called = False

		async def async_escalate(desc: str, elapsed: float, path: Path) -> None:
			nonlocal called
			called = True

		await gate.wait_for_approval(
			req,
			timeout_seconds=0.1,
			escalation_callback=async_escalate,
		)

		assert called is True

	@pytest.mark.asyncio
	async def test_no_escalation_when_approved(self, tmp_path: Path) -> None:
		"""Escalation callback not called if approval received before timeout."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push")

		escalation_called = False

		def on_escalate(desc: str, elapsed: float, path: Path) -> None:
			nonlocal escalation_called
			escalation_called = True

		async def approve_after_delay() -> None:
			await asyncio.sleep(0.05)
			approval_file = tmp_path / ".mc-approvals" / f"{req.request_id}.json"
			data = json.loads(approval_file.read_text())
			data["status"] = "approved"
			approval_file.write_text(json.dumps(data))

		task = asyncio.create_task(approve_after_delay())
		result = await gate.wait_for_approval(
			req,
			timeout_seconds=2.0,
			escalation_callback=on_escalate,
		)
		await task

		assert result.approved is True
		assert escalation_called is False


class TestBackwardCompatibility:
	@pytest.mark.asyncio
	async def test_zero_timeout_blocks_until_approval(self, tmp_path: Path) -> None:
		"""timeout_seconds=0 means infinite wait; approval via file resolves it."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(gate_type="push")

		async def approve_after_delay() -> None:
			await asyncio.sleep(0.1)
			approval_file = tmp_path / ".mc-approvals" / f"{req.request_id}.json"
			data = json.loads(approval_file.read_text())
			data["status"] = "approved"
			approval_file.write_text(json.dumps(data))

		task = asyncio.create_task(approve_after_delay())
		result = await gate.wait_for_approval(req, timeout_seconds=0)
		await task

		assert result.approved is True
		assert result.timed_out is False

	@pytest.mark.asyncio
	async def test_request_approval_still_returns_bool(self, tmp_path: Path) -> None:
		"""Original request_approval still returns a bool, unmodified."""
		cfg = _make_config(tmp_path)
		gate = ApprovalGate(cfg, notifier=None)
		req = ApprovalRequest(
			gate_type="push",
			timeout_seconds=0,
			timeout_action="approve",
		)
		result = await gate.request_approval(req)
		assert isinstance(result, bool)
		assert result is True

	def test_default_timeout_is_1800(self) -> None:
		"""Default timeout_seconds parameter is 1800 (30 minutes)."""
		sig = inspect.signature(ApprovalGate.wait_for_approval)
		assert sig.parameters["timeout_seconds"].default == 1800

	def test_default_reminder_interval_is_600(self) -> None:
		"""Default reminder_interval_seconds parameter is 600 (10 minutes)."""
		sig = inspect.signature(ApprovalGate.wait_for_approval)
		assert sig.parameters["reminder_interval_seconds"].default == 600
