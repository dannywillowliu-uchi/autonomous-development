"""Tests for ZFC fixup prompt generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mission_control.config import MissionConfig
from mission_control.green_branch import GreenBranchManager


def _make_gbm(zfc_fixup: bool = False) -> GreenBranchManager:
	config = MissionConfig()
	config.zfc.zfc_fixup_prompts = zfc_fixup
	config.zfc.llm_timeout = 5
	config.green_branch.fixup_candidates = 3
	db = MagicMock()
	gbm = GreenBranchManager(config, db)
	gbm.workspace = "/tmp/test-workspace"
	return gbm


class TestZFCFixupPrompts:
	@pytest.mark.asyncio
	async def test_zfc_disabled_uses_static_prompts(self) -> None:
		"""toggle off -> FIXUP_PROMPTS used."""
		gbm = _make_gbm(zfc_fixup=False)

		# Mock out all the internals that run_fixup depends on
		gbm._run_fixup_candidate = AsyncMock(return_value=MagicMock(
			verification_passed=False, branch="mc/fixup-0",
		))
		gbm._run_git = AsyncMock(return_value=(True, ""))

		with patch.object(gbm, "_zfc_generate_fixup_strategies") as mock_zfc:
			await gbm.run_fixup("test failure")
			mock_zfc.assert_not_called()

	@pytest.mark.asyncio
	async def test_zfc_enabled_calls_llm(self) -> None:
		"""toggle on -> _zfc_generate_fixup_strategies called."""
		gbm = _make_gbm(zfc_fixup=True)

		gbm._run_fixup_candidate = AsyncMock(return_value=MagicMock(
			verification_passed=False, branch="mc/fixup-0",
		))
		gbm._run_git = AsyncMock(return_value=(True, ""))

		with patch.object(
			gbm, "_zfc_generate_fixup_strategies",
			return_value=["Fix A", "Fix B", "Fix C"],
		) as mock_zfc:
			await gbm.run_fixup("test failure")
			mock_zfc.assert_called_once_with("test failure", 3)

	@pytest.mark.asyncio
	async def test_zfc_llm_failure_falls_back(self) -> None:
		"""LLM returns None -> static prompts used as fallback."""
		gbm = _make_gbm(zfc_fixup=True)

		gbm._run_fixup_candidate = AsyncMock(return_value=MagicMock(
			verification_passed=False, branch="mc/fixup-0",
		))
		gbm._run_git = AsyncMock(return_value=(True, ""))

		with patch.object(gbm, "_zfc_generate_fixup_strategies", return_value=None):
			# Should not raise -- falls back to static prompts
			result = await gbm.run_fixup("test failure")
			assert result is not None

	@pytest.mark.asyncio
	async def test_zfc_parse_strategies_output(self) -> None:
		"""Correct FIXUP_STRATEGIES marker is parsed."""
		gbm = _make_gbm(zfc_fixup=True)

		output = (
			'Some reasoning about fixes.\n'
			'FIXUP_STRATEGIES:{"strategies": ["Fix the import", "Update the mock", "Rewrite test"]}'
		)

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (output.encode(), b"")
		mock_proc.returncode = 0

		with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
			strategies = await gbm._zfc_generate_fixup_strategies("test failure", 3)

		assert strategies is not None
		assert len(strategies) == 3
		assert "Fix the import" in strategies[0]

	@pytest.mark.asyncio
	async def test_zfc_timeout_returns_none(self) -> None:
		"""Subprocess timeout -> returns None."""
		gbm = _make_gbm(zfc_fixup=True)
		gbm.config.zfc.llm_timeout = 0  # Immediate timeout

		mock_proc = AsyncMock()
		mock_proc.communicate.side_effect = TimeoutError()
		mock_proc.kill = MagicMock()
		mock_proc.wait = AsyncMock()

		with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
			with patch("asyncio.wait_for", side_effect=TimeoutError()):
				strategies = await gbm._zfc_generate_fixup_strategies("test failure", 3)

		assert strategies is None
