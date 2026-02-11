"""Tests for objective evaluator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from mission_control.config import (
	MissionConfig,
	SchedulerConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.evaluator import _parse_evaluation, evaluate_objective


def _config() -> MissionConfig:
	return MissionConfig(
		target=TargetConfig(
			name="test",
			path="/tmp/test",
			branch="main",
			objective="Build stuff",
			verification=VerificationConfig(command="pytest -q", timeout=60),
		),
		scheduler=SchedulerConfig(model="sonnet"),
	)


class TestParseEvaluation:
	def test_valid_json(self) -> None:
		raw = json.dumps({
			"score": 0.75,
			"met": False,
			"reasoning": "Good progress but not done",
			"remaining": ["task1", "task2"],
		})
		result = _parse_evaluation(raw)
		assert result.score == 0.75
		assert result.met is False
		assert result.reasoning == "Good progress but not done"
		assert result.remaining == ["task1", "task2"]

	def test_markdown_fenced_json(self) -> None:
		raw = '```json\n{"score": 0.9, "met": true, "reasoning": "Nearly done", "remaining": ["polish"]}\n```'
		result = _parse_evaluation(raw)
		assert result.score == 0.9
		assert result.met is True
		assert result.reasoning == "Nearly done"
		assert result.remaining == ["polish"]

	def test_markdown_fenced_no_lang(self) -> None:
		raw = '```\n{"score": 0.5, "met": false, "reasoning": "Half done", "remaining": []}\n```'
		result = _parse_evaluation(raw)
		assert result.score == 0.5
		assert result.met is False
		assert result.reasoning == "Half done"
		assert result.remaining == []

	def test_invalid_json_returns_defaults(self) -> None:
		result = _parse_evaluation("this is not json at all")
		assert result.score == 0.0
		assert result.met is False
		assert result.reasoning == ""
		assert result.remaining == []

	def test_score_clamped_above_one(self) -> None:
		raw = json.dumps({"score": 1.5, "met": True, "reasoning": "Over", "remaining": []})
		result = _parse_evaluation(raw)
		assert result.score == 1.0

	def test_score_clamped_below_zero(self) -> None:
		raw = json.dumps({"score": -0.5, "met": False, "reasoning": "Negative", "remaining": []})
		result = _parse_evaluation(raw)
		assert result.score == 0.0

	def test_json_embedded_in_text(self) -> None:
		raw = (
			'Here is my evaluation:\n'
			'{"score": 0.6, "met": false, "reasoning": "Partial", "remaining": ["x"]}\n'
			'That is all.'
		)
		result = _parse_evaluation(raw)
		assert result.score == 0.6
		assert result.met is False
		assert result.reasoning == "Partial"
		assert result.remaining == ["x"]

	def test_missing_fields_use_defaults(self) -> None:
		raw = json.dumps({"score": 0.3})
		result = _parse_evaluation(raw)
		assert result.score == 0.3
		assert result.met is False
		assert result.reasoning == ""
		assert result.remaining == []


class TestEvaluateObjective:
	async def test_successful_evaluation(self) -> None:
		config = _config()
		response = json.dumps({
			"score": 0.8,
			"met": False,
			"reasoning": "Most work done",
			"remaining": ["cleanup"],
		})

		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (response.encode(), b"")

		with patch("mission_control.evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await evaluate_objective(
				config,
				snapshot_hash="abc123def456",
				round_summary="Implemented feature X",
				objective="Build feature X",
			)

		assert result.score == 0.8
		assert result.met is False
		assert result.reasoning == "Most work done"
		assert result.remaining == ["cleanup"]

	async def test_nonzero_exit_returns_defaults(self) -> None:
		config = _config()

		mock_proc = AsyncMock()
		mock_proc.returncode = 1
		mock_proc.communicate.return_value = (b"", b"error occurred")

		with patch("mission_control.evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await evaluate_objective(
				config,
				snapshot_hash="abc123",
				round_summary="Failed round",
				objective="Build feature Y",
			)

		assert result.score == 0.0
		assert result.met is False
		assert result.reasoning == ""
		assert result.remaining == []

	async def test_empty_output_returns_defaults(self) -> None:
		config = _config()

		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (b"", b"")

		with patch("mission_control.evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
			result = await evaluate_objective(
				config,
				snapshot_hash="abc123",
				round_summary="Empty",
				objective="Build feature Z",
			)

		assert result.score == 0.0
		assert result.met is False

	async def test_os_error_returns_defaults(self) -> None:
		config = _config()

		with patch(
			"mission_control.evaluator.asyncio.create_subprocess_exec",
			side_effect=FileNotFoundError("claude not found"),
		):
			result = await evaluate_objective(
				config,
				snapshot_hash="abc123",
				round_summary="No claude",
				objective="Build feature",
			)

		assert result.score == 0.0
		assert result.met is False

	async def test_passes_correct_model_to_command(self) -> None:
		config = _config()
		config.scheduler.model = "opus"

		mock_proc = AsyncMock()
		mock_proc.returncode = 0
		mock_proc.communicate.return_value = (
			json.dumps({"score": 0.5, "met": False, "reasoning": "ok", "remaining": []}).encode(),
			b"",
		)

		with patch("mission_control.evaluator.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
			await evaluate_objective(
				config,
				snapshot_hash="abc123",
				round_summary="Summary",
				objective="Objective",
			)

		# Verify the model flag was passed correctly
		call_args = mock_exec.call_args[0]
		model_idx = list(call_args).index("--model")
		assert call_args[model_idx + 1] == "opus"
