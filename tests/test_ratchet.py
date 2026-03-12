"""Tests for GitRatchet."""

from unittest.mock import AsyncMock, patch

import pytest

from autodev.ratchet import GitRatchet


@pytest.fixture
def ratchet(tmp_path):
	return GitRatchet(tmp_path)


def _make_proc_mock(returncode=0, stdout=b"", stderr=b""):
	proc = AsyncMock()
	proc.returncode = returncode
	proc.communicate = AsyncMock(return_value=(stdout, stderr))
	return proc


class TestCheckpoint:
	async def test_checkpoint_creates_tag(self, ratchet):
		proc = _make_proc_mock()
		with patch("autodev.ratchet.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
			tag = await ratchet.checkpoint("abc123")
		assert tag == "autodev/pre-abc123"
		mock_exec.assert_called_once()
		args = mock_exec.call_args[0]
		assert args == ("git", "tag", "autodev/pre-abc123", "HEAD")


class TestVerifyAndDecide:
	async def test_verify_and_decide_keeps_on_pass(self, ratchet):
		proc = _make_proc_mock(returncode=0)
		with patch("autodev.ratchet.asyncio.create_subprocess_shell", return_value=proc):
			result = await ratchet.verify_and_decide("autodev/pre-abc", "pytest -q")
		assert result is True

	async def test_verify_and_decide_rollbacks_on_fail(self, ratchet):
		fail_proc = _make_proc_mock(returncode=1)
		reset_proc = _make_proc_mock(returncode=0)
		with (
			patch("autodev.ratchet.asyncio.create_subprocess_shell", return_value=fail_proc),
			patch("autodev.ratchet.asyncio.create_subprocess_exec", return_value=reset_proc) as mock_exec,
		):
			result = await ratchet.verify_and_decide("autodev/pre-abc", "pytest -q")
		assert result is False
		args = mock_exec.call_args[0]
		assert args == ("git", "reset", "--hard", "autodev/pre-abc")


class TestRollback:
	async def test_rollback_runs_reset(self, ratchet):
		proc = _make_proc_mock()
		with patch("autodev.ratchet.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
			await ratchet.rollback("autodev/pre-xyz")
		args = mock_exec.call_args[0]
		assert args == ("git", "reset", "--hard", "autodev/pre-xyz")


class TestAppendExperimentLog:
	def test_append_experiment_log_creates_file(self, ratchet, tmp_path):
		ratchet.append_experiment_log("abc123", 10, 12, "keep", "Add feature", 45.0, 0.12)
		tsv_path = tmp_path / ".autodev-experiments.tsv"
		assert tsv_path.exists()
		lines = tsv_path.read_text().splitlines()
		assert len(lines) == 2
		assert lines[0].startswith("commit\t")
		assert "abc123" in lines[1]
		assert "keep" in lines[1]

	def test_append_experiment_log_appends(self, ratchet, tmp_path):
		ratchet.append_experiment_log("abc", 10, 12, "keep", "First", 10.0, 0.05)
		ratchet.append_experiment_log("def", 12, 15, "discard", "Second", 20.0, 0.10)
		tsv_path = tmp_path / ".autodev-experiments.tsv"
		lines = tsv_path.read_text().splitlines()
		assert len(lines) == 3
		assert "abc" in lines[1]
		assert "def" in lines[2]
