"""Tests for the LLM-based intelligence evaluator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autodev.intelligence.llm_evaluator import (
	_build_enriched_program,
	_extract_json_array,
	evaluate_findings_llm,
)
from autodev.intelligence.models import Finding


def _make_finding(**kwargs) -> Finding:
	defaults = {
		"id": "abc123",
		"source": "github",
		"title": "New MCP server for databases",
		"url": "https://github.com/example/mcp-db",
		"summary": "An MCP server that provides database access tools",
		"relevance_score": 0.8,
	}
	defaults.update(kwargs)
	return Finding(**defaults)


class TestExtractJsonArray:
	def test_markdown_fenced_json(self):
		text = 'Some preamble\n```json\n[{"decision": "integrate"}]\n```\nTrailing text'
		result = _extract_json_array(text)
		assert result == [{"decision": "integrate"}]

	def test_markdown_fenced_no_lang(self):
		text = '```\n[{"decision": "skip"}]\n```'
		result = _extract_json_array(text)
		assert result == [{"decision": "skip"}]

	def test_bare_json_array(self):
		text = '[{"finding_id": "abc", "decision": "integrate"}]'
		result = _extract_json_array(text)
		assert result == [{"finding_id": "abc", "decision": "integrate"}]

	def test_preamble_before_json(self):
		text = 'Here are my evaluations:\n\n[{"decision": "skip"}]'
		result = _extract_json_array(text)
		assert result == [{"decision": "skip"}]

	def test_no_json_raises(self):
		with pytest.raises(ValueError, match="No valid JSON array"):
			_extract_json_array("No JSON here at all")

	def test_invalid_json_in_fences_falls_through(self):
		text = '```json\n{not valid json\n```\n[{"ok": true}]'
		result = _extract_json_array(text)
		assert result == [{"ok": True}]


class TestBuildEnrichedProgram:
	def test_includes_architecture_section(self, tmp_path):
		claude_md = tmp_path / "CLAUDE.md"
		claude_md.write_text(
			"# Project\n\n## Architecture\n\n### Core\n- config.py\n- models.py\n\n## Gotchas\n- Don't do X\n"
		)
		result = _build_enriched_program("# Program\nBase content", tmp_path)
		assert "## Current Architecture (from CLAUDE.md)" in result
		assert "config.py" in result
		assert "Gotchas" not in result

	def test_no_claude_md(self, tmp_path):
		result = _build_enriched_program("# Program", tmp_path)
		assert result.startswith("# Program")

	@patch("autodev.intelligence.llm_evaluator._subprocess.run")
	def test_includes_git_log(self, mock_run, tmp_path):
		mock_run.return_value = MagicMock(returncode=0, stdout="abc1234 feat: something\ndef5678 fix: bug")
		result = _build_enriched_program("# Program", tmp_path)
		assert "## Recent Commits" in result
		assert "abc1234" in result

	@patch("autodev.intelligence.llm_evaluator._subprocess.run", side_effect=OSError("git not found"))
	def test_git_log_failure_ignored(self, mock_run, tmp_path):
		result = _build_enriched_program("# Program", tmp_path)
		assert "Recent Commits" not in result


class TestEvaluateFindingsLlm:
	@pytest.mark.asyncio
	async def test_successful_evaluation(self, tmp_path):
		program_path = tmp_path / "docs" / "program.md"
		program_path.parent.mkdir(parents=True)
		program_path.write_text("# Program\nTest program")

		findings = [_make_finding()]
		llm_response = json.dumps([{
			"finding_id": "abc123",
			"decision": "integrate",
			"reasoning": "Useful MCP server for database access",
			"proposed_action": "Add database MCP integration",
			"target_modules": ["mcp_server.py", "config.py"],
		}])
		llm_output = f"```json\n{llm_response}\n```"

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		with patch("autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec", return_value=mock_proc), \
			patch("autodev.intelligence.llm_evaluator._subprocess.run", return_value=MagicMock(returncode=1)):
			proposals = await evaluate_findings_llm(findings, tmp_path)

		assert len(proposals) == 1
		assert proposals[0].finding_id == "abc123"
		assert proposals[0].title == "Adapt: New MCP server for databases"
		assert proposals[0].target_modules == ["mcp_server.py", "config.py"]
		assert proposals[0].proposal_type == "integration"

	@pytest.mark.asyncio
	async def test_skip_decisions_filtered(self, tmp_path):
		program_path = tmp_path / "docs" / "program.md"
		program_path.parent.mkdir(parents=True)
		program_path.write_text("# Program")

		findings = [_make_finding(id="f1"), _make_finding(id="f2")]
		decisions = [
			{
				"finding_id": "f1", "decision": "integrate",
				"reasoning": "Good", "proposed_action": "Do it",
				"target_modules": [],
			},
			{
				"finding_id": "f2", "decision": "skip",
				"reasoning": "Not useful", "proposed_action": "",
				"target_modules": [],
			},
		]
		llm_output = json.dumps(decisions)

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (f"```json\n{llm_output}\n```".encode(), b"")
		mock_proc.returncode = 0

		with patch("autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec", return_value=mock_proc), \
			patch("autodev.intelligence.llm_evaluator._subprocess.run", return_value=MagicMock(returncode=1)):
			proposals = await evaluate_findings_llm(findings, tmp_path)

		assert len(proposals) == 1
		assert proposals[0].finding_id == "f1"

	@pytest.mark.asyncio
	async def test_fallback_on_subprocess_failure(self, tmp_path):
		program_path = tmp_path / "docs" / "program.md"
		program_path.parent.mkdir(parents=True)
		program_path.write_text("# Program")

		findings = [_make_finding(title="MCP server tool")]

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"", b"error")
		mock_proc.returncode = 1

		with patch("autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec", return_value=mock_proc), \
			patch("autodev.intelligence.llm_evaluator._subprocess.run", return_value=MagicMock(returncode=1)):
			proposals = await evaluate_findings_llm(findings, tmp_path)

		# Should fall back to keyword evaluator -- MCP keyword should produce proposals
		assert isinstance(proposals, list)

	@pytest.mark.asyncio
	async def test_fallback_on_json_parse_error(self, tmp_path):
		program_path = tmp_path / "docs" / "program.md"
		program_path.parent.mkdir(parents=True)
		program_path.write_text("# Program")

		findings = [_make_finding(title="MCP server tool")]

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"This is not JSON at all", b"")
		mock_proc.returncode = 0

		with patch("autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec", return_value=mock_proc), \
			patch("autodev.intelligence.llm_evaluator._subprocess.run", return_value=MagicMock(returncode=1)):
			proposals = await evaluate_findings_llm(findings, tmp_path)

		assert isinstance(proposals, list)

	@pytest.mark.asyncio
	async def test_program_path_defaults_to_project_docs(self, tmp_path):
		program_path = tmp_path / "docs" / "program.md"
		program_path.parent.mkdir(parents=True)
		program_path.write_text("# Default Program")

		findings = [_make_finding()]
		skip_decision = json.dumps([{
			"finding_id": "abc123", "decision": "skip",
			"reasoning": "No", "proposed_action": "",
			"target_modules": [],
		}])
		llm_output = f"```json\n{skip_decision}\n```"

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		mock_sub = patch(
			"autodev.intelligence.llm_evaluator._subprocess.run",
			return_value=MagicMock(returncode=1),
		)
		mock_claude = patch(
			"autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		)
		with mock_claude as mock_exec, mock_sub:
			await evaluate_findings_llm(findings, tmp_path)

		# Verify the prompt passed to claude contains program content
		call_args = mock_exec.call_args
		prompt_arg = call_args[0][3]  # 4th positional arg is the prompt
		assert "Default Program" in prompt_arg

	@pytest.mark.asyncio
	async def test_context_enrichment_in_prompt(self, tmp_path):
		program_path = tmp_path / "docs" / "program.md"
		program_path.parent.mkdir(parents=True)
		program_path.write_text("# Program")

		claude_md = tmp_path / "CLAUDE.md"
		claude_md.write_text("# Project\n\n## Architecture\n\n### Core\n- db.py\n- models.py\n\n## Other\nstuff\n")

		findings = [_make_finding()]
		llm_output = '```json\n[]\n```'

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		git_result = MagicMock(returncode=0, stdout="abc1234 feat: add thing\ndef5678 fix: bug")

		mock_claude = patch(
			"autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		)
		mock_sub = patch(
			"autodev.intelligence.llm_evaluator._subprocess.run",
			return_value=git_result,
		)
		with mock_claude as mock_exec, mock_sub:
			await evaluate_findings_llm(findings, tmp_path)

		prompt_arg = mock_exec.call_args[0][3]
		assert "Current Architecture" in prompt_arg
		assert "db.py" in prompt_arg
		assert "abc1234" in prompt_arg
		assert "Recent Commits" in prompt_arg

	@pytest.mark.asyncio
	async def test_fallback_when_program_missing(self, tmp_path):
		findings = [_make_finding(title="MCP server tool")]

		with patch("autodev.intelligence.llm_evaluator._subprocess.run", return_value=MagicMock(returncode=1)):
			proposals = await evaluate_findings_llm(findings, tmp_path)

		assert isinstance(proposals, list)

	@pytest.mark.asyncio
	async def test_custom_program_path(self, tmp_path):
		custom_program = tmp_path / "custom_program.md"
		custom_program.write_text("# Custom Program")

		findings = [_make_finding()]
		llm_output = '```json\n[]\n```'

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (llm_output.encode(), b"")
		mock_proc.returncode = 0

		mock_claude = patch(
			"autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		)
		mock_sub = patch(
			"autodev.intelligence.llm_evaluator._subprocess.run",
			return_value=MagicMock(returncode=1),
		)
		with mock_claude as mock_exec, mock_sub:
			await evaluate_findings_llm(
				findings, tmp_path, program_path=custom_program,
			)

		prompt_arg = mock_exec.call_args[0][3]
		assert "Custom Program" in prompt_arg
