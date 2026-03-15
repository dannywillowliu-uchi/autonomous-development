"""Tests for the LLM-based intelligence evaluator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from autodev.intelligence.llm_evaluator import (
	_build_enriched_program,
	_decisions_to_proposals,
	_extract_json_array,
	evaluate_findings,
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
	@pytest.mark.asyncio
	async def test_includes_architecture_section(self, tmp_path):
		program = tmp_path / "program.md"
		program.write_text("Base content")
		claude_md = tmp_path / "CLAUDE.md"
		claude_md.write_text(
			"## Architecture\n\n### Core\n- config.py\n- models.py\n\n## Gotchas\n- stuff\n"
		)
		git_proc = AsyncMock()
		git_proc.communicate.return_value = (b"", b"")
		git_proc.returncode = 1
		with patch("asyncio.create_subprocess_exec", return_value=git_proc):
			result = await _build_enriched_program(tmp_path, program)
		assert "Current Architecture" in result
		assert "config.py" in result
		assert "Gotchas" not in result

	@pytest.mark.asyncio
	async def test_no_claude_md(self, tmp_path):
		program = tmp_path / "program.md"
		program.write_text("Base content")
		git_proc = AsyncMock()
		git_proc.communicate.return_value = (b"", b"")
		git_proc.returncode = 1
		with patch("asyncio.create_subprocess_exec", return_value=git_proc):
			result = await _build_enriched_program(tmp_path, program)
		assert "Base content" in result

	@pytest.mark.asyncio
	async def test_includes_git_log(self, tmp_path):
		program = tmp_path / "program.md"
		program.write_text("Base")
		git_proc = AsyncMock()
		git_proc.communicate.return_value = (b"abc1234 feat: something", b"")
		git_proc.returncode = 0
		with patch("asyncio.create_subprocess_exec", return_value=git_proc):
			result = await _build_enriched_program(tmp_path, program)
		assert "Recent Activity" in result
		assert "abc1234" in result

	@pytest.mark.asyncio
	async def test_git_log_failure_ignored(self, tmp_path):
		program = tmp_path / "program.md"
		program.write_text("Base")
		with patch("asyncio.create_subprocess_exec", side_effect=OSError("git not found")):
			result = await _build_enriched_program(tmp_path, program)
		assert "Recent Activity" not in result

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

		with patch("autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
			proposals = await evaluate_findings(findings, tmp_path)

		assert len(proposals) == 1
		assert proposals[0].finding_id == "abc123"
		assert proposals[0].title == "Add database MCP integration"
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

		with patch("autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
			proposals = await evaluate_findings(findings, tmp_path)

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

		with patch("autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
			proposals = await evaluate_findings(findings, tmp_path)

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

		with patch("autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
			proposals = await evaluate_findings(findings, tmp_path)

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

		mock_claude = patch(
			"autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		)
		with mock_claude as mock_exec:
			await evaluate_findings(findings, tmp_path)

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

		mock_claude = patch(
			"autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec",
			return_value=mock_proc,
		)
		with mock_claude as mock_exec:
			await evaluate_findings(findings, tmp_path)

		prompt_arg = mock_exec.call_args[0][3]
		assert "Current Architecture" in prompt_arg
		assert "db.py" in prompt_arg

	@pytest.mark.asyncio
	async def test_fallback_when_program_missing(self, tmp_path):
		findings = [_make_finding(title="MCP server tool")]
		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (b"not json", b"")
		mock_proc.returncode = 0

		with patch("autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec", return_value=mock_proc):
			proposals = await evaluate_findings(findings, tmp_path)

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
		with mock_claude as mock_exec:
			await evaluate_findings(
				findings, tmp_path, program_path=custom_program,
			)

		prompt_arg = mock_exec.call_args[0][3]
		assert "Custom Program" in prompt_arg

	@pytest.mark.asyncio
	async def test_empty_findings_returns_empty(self, tmp_path):
		result = await evaluate_findings([], tmp_path)
		assert result == []

	@pytest.mark.asyncio
	async def test_large_finding_set_no_batching(self, tmp_path):
		program_path = tmp_path / "docs" / "program.md"
		program_path.parent.mkdir(parents=True)
		program_path.write_text("# Program")

		findings = [_make_finding(id=f"f{i}") for i in range(50)]
		decisions = [{"finding_id": f"f{i}", "decision": "skip", "reasoning": "No", "proposed_action": "", "target_modules": []} for i in range(50)]
		llm_output = json.dumps(decisions)

		mock_proc = AsyncMock()
		mock_proc.communicate.return_value = (f"```json\n{llm_output}\n```".encode(), b"")
		mock_proc.returncode = 0

		with patch("autodev.intelligence.llm_evaluator.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
			await evaluate_findings(findings, tmp_path)

		# Should be called once for git log in _build_enriched_program, once for claude
		# Verify only ONE claude call (no batching)
		claude_calls = [c for c in mock_exec.call_args_list if "--print" in str(c)]
		assert len(claude_calls) == 1


class TestDecisionsToProposals:
	def test_integrate_decisions(self):
		findings = {"f1": _make_finding(id="f1")}
		decisions = [{
			"finding_id": "f1",
			"decision": "integrate",
			"reasoning": "Useful tool",
			"proposed_action": "Add integration",
			"target_modules": ["config.py"],
		}]
		result = _decisions_to_proposals(decisions, findings)
		assert len(result) == 1
		assert result[0].finding_id == "f1"
		assert result[0].title == "Add integration"
		assert result[0].effort_estimate == "medium"
		assert result[0].risk_level == "low"

	def test_skip_decisions_filtered(self):
		findings = {"f1": _make_finding(id="f1")}
		decisions = [{
			"finding_id": "f1",
			"decision": "skip",
			"reasoning": "Not useful",
			"proposed_action": "",
			"target_modules": [],
		}]
		result = _decisions_to_proposals(decisions, findings)
		assert len(result) == 0

	def test_fallback_title_from_finding(self):
		findings = {"f1": _make_finding(id="f1", title="Cool Tool")}
		decisions = [{
			"finding_id": "f1",
			"decision": "integrate",
			"reasoning": "Yes",
			"proposed_action": "",
			"target_modules": [],
		}]
		result = _decisions_to_proposals(decisions, findings)
		assert result[0].title == "Adapt: Cool Tool"

