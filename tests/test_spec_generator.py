"""Tests for spec generation and auto_update integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autodev.auto_update import AutoUpdatePipeline
from autodev.config import MissionConfig, NotificationConfig, TelegramConfig
from autodev.db import Database
from autodev.intelligence.models import AdaptationProposal, Finding
from autodev.intelligence.spec_generator import SpecGenerator, _strip_html

# -- Fixtures --


@pytest.fixture
def db() -> Database:
	return Database(":memory:")


@pytest.fixture
def config() -> MissionConfig:
	cfg = MissionConfig()
	cfg.notifications = NotificationConfig(
		telegram=TelegramConfig(bot_token="123:FAKE", chat_id="456"),
	)
	return cfg


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
	"""Create a minimal project structure for tests."""
	src = tmp_path / "src" / "autodev"
	src.mkdir(parents=True)
	(src / "config.py").write_text('"""Config module."""\n\nclass MissionConfig:\n\tpass\n')
	(src / "worker.py").write_text('"""Worker module."""\n\ndef run_worker():\n\tpass\n')
	(tmp_path / "CLAUDE.md").write_text(
		"# autodev\n\n## Architecture\n\n### Core\n- config.py\n- worker.py\n\n## Gotchas\n\n- None\n"
	)
	return tmp_path


@pytest.fixture
def proposal() -> AdaptationProposal:
	return AdaptationProposal(
		id="prop_test",
		finding_id="find_test",
		title="Adapt: New MCP pattern",
		description="Based on finding from hackernews: New MCP server pattern for agents",
		proposal_type="integration",
		target_modules=["config.py", "worker.py"],
		priority=3,
		effort_estimate="medium",
		risk_level="low",
	)


# -- SpecGenerator unit tests --


@pytest.mark.asyncio
async def test_generate_spec_returns_markdown(project_path: Path, proposal: AdaptationProposal) -> None:
	"""generate_spec calls claude --print and returns the LLM output."""
	generator = SpecGenerator(project_path)

	spec_content = (
		"# Spec\n\n## Problem Statement\nImprove MCP.\n\n## Changes Needed\n"
		"Edit config.py.\n\n## Testing Requirements\nAdd tests.\n\n## Risk Assessment\nLow risk."
	)
	mock_proc = AsyncMock()
	mock_proc.communicate = AsyncMock(return_value=(spec_content.encode(), b""))
	mock_proc.returncode = 0

	with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
		result = await generator.generate_spec(proposal)

	assert "Problem Statement" in result
	assert "Changes Needed" in result
	assert "Testing Requirements" in result
	assert "Risk Assessment" in result

	# Verify claude was called with --print
	call_args = mock_exec.call_args
	assert call_args[0][0] == "claude"
	assert "--print" in call_args[0]


@pytest.mark.asyncio
async def test_generate_spec_raises_on_failure(project_path: Path, proposal: AdaptationProposal) -> None:
	"""generate_spec raises RuntimeError when claude subprocess fails."""
	generator = SpecGenerator(project_path)

	mock_proc = AsyncMock()
	mock_proc.communicate = AsyncMock(return_value=(b"", b"error: model unavailable"))
	mock_proc.returncode = 1

	with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
		with pytest.raises(RuntimeError, match="LLM spec generation failed"):
			await generator.generate_spec(proposal)


@pytest.mark.asyncio
async def test_read_source_context_with_url() -> None:
	"""_read_source_context fetches and strips HTML from the URL."""
	generator = SpecGenerator(Path("/tmp"))

	html_content = "<html><body><h1>Title</h1><p>Content about agents</p></body></html>"

	mock_response = MagicMock()
	mock_response.text = html_content
	mock_response.raise_for_status = MagicMock()

	mock_client = AsyncMock()
	mock_client.get = AsyncMock(return_value=mock_response)
	mock_client.__aenter__ = AsyncMock(return_value=mock_client)
	mock_client.__aexit__ = AsyncMock(return_value=False)

	with patch("httpx.AsyncClient", return_value=mock_client):
		result = await generator._read_source_context("https://example.com/article")

	assert "Title" in result
	assert "Content about agents" in result
	assert "<html>" not in result


@pytest.mark.asyncio
async def test_read_source_context_empty_url() -> None:
	"""_read_source_context returns empty string for empty URL."""
	generator = SpecGenerator(Path("/tmp"))
	result = await generator._read_source_context("")
	assert result == ""


@pytest.mark.asyncio
async def test_read_source_context_fetch_failure() -> None:
	"""_read_source_context returns empty string on fetch failure."""
	generator = SpecGenerator(Path("/tmp"))

	mock_client = AsyncMock()
	mock_client.get = AsyncMock(side_effect=Exception("connection failed"))
	mock_client.__aenter__ = AsyncMock(return_value=mock_client)
	mock_client.__aexit__ = AsyncMock(return_value=False)

	with patch("httpx.AsyncClient", return_value=mock_client):
		result = await generator._read_source_context("https://example.com/broken")

	assert result == ""


@pytest.mark.asyncio
async def test_read_project_context(project_path: Path) -> None:
	"""_read_project_context reads target modules and CLAUDE.md architecture."""
	generator = SpecGenerator(project_path)

	result = await generator._read_project_context(["config.py", "worker.py"])

	assert "config.py" in result
	assert "MissionConfig" in result
	assert "worker.py" in result
	assert "Architecture" in result


@pytest.mark.asyncio
async def test_read_project_context_missing_module(project_path: Path) -> None:
	"""_read_project_context skips missing modules without error."""
	generator = SpecGenerator(project_path)

	result = await generator._read_project_context(["nonexistent.py", "config.py"])

	assert "config.py" in result
	assert "nonexistent" not in result


def test_build_spec_prompt(project_path: Path, proposal: AdaptationProposal) -> None:
	"""_build_spec_prompt includes all required sections."""
	generator = SpecGenerator(project_path)

	prompt = generator._build_spec_prompt(proposal, "Source text here", "Project architecture here")

	assert proposal.title in prompt
	assert proposal.description in prompt
	assert "config.py" in prompt
	assert "Source Material" in prompt
	assert "Source text here" in prompt
	assert "Project Architecture" in prompt
	assert "Project architecture here" in prompt
	assert "Problem Statement" in prompt
	assert "Changes Needed" in prompt
	assert "Testing Requirements" in prompt
	assert "Risk Assessment" in prompt


def test_build_spec_prompt_no_source_context(project_path: Path, proposal: AdaptationProposal) -> None:
	"""_build_spec_prompt omits source section when empty."""
	generator = SpecGenerator(project_path)

	prompt = generator._build_spec_prompt(proposal, "", "Project arch")

	assert "Source Material" not in prompt
	assert "Project Architecture" in prompt


def test_build_spec_prompt_no_project_context(project_path: Path, proposal: AdaptationProposal) -> None:
	"""_build_spec_prompt omits project section when empty."""
	generator = SpecGenerator(project_path)

	prompt = generator._build_spec_prompt(proposal, "Source text", "")

	assert "Source Material" in prompt
	assert "Project Architecture" not in prompt


# -- _strip_html tests --


def test_strip_html_basic() -> None:
	assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_scripts_and_styles() -> None:
	html = '<script>alert("x")</script><style>.a{color:red}</style><p>Keep this</p>'
	assert _strip_html(html) == "Keep this"


def test_strip_html_entities() -> None:
	assert "don't" in _strip_html("<p>don&#x27;t stop</p>")


# -- auto_update.py integration tests --


@pytest.mark.asyncio
async def test_auto_update_generate_spec(config: MissionConfig, db: Database, tmp_path: Path) -> None:
	"""_generate_spec creates a SpecGenerator and writes the spec file."""
	config.target.path = str(tmp_path)
	pipeline = AutoUpdatePipeline(config, db)

	proposal = AdaptationProposal(
		id="p1",
		finding_id="f1",
		title="Adapt: Test Feature",
		description="Test description",
		target_modules=["config.py"],
	)
	finding = Finding(id="f1", url="https://example.com/article", title="Test Feature")
	pipeline._findings_by_id = {"f1": finding}

	spec_content = "# Spec\n\n## Problem Statement\nTest."

	with patch("autodev.intelligence.spec_generator.SpecGenerator") as mock_gen_cls:
		mock_instance = AsyncMock()
		mock_instance.generate_spec = AsyncMock(return_value=spec_content)
		mock_gen_cls.return_value = mock_instance

		result = await pipeline._generate_spec(proposal)

	assert result == spec_content
	mock_instance.generate_spec.assert_called_once_with(proposal, source_url="https://example.com/article")

	# Verify spec file was written
	specs_dir = tmp_path / "docs" / "superpowers" / "specs"
	spec_files = list(specs_dir.glob("auto-*.md"))
	assert len(spec_files) == 1
	assert spec_files[0].read_text() == spec_content


@pytest.mark.asyncio
async def test_auto_update_generate_spec_no_finding(config: MissionConfig, db: Database, tmp_path: Path) -> None:
	"""_generate_spec passes empty URL when finding is not found."""
	config.target.path = str(tmp_path)
	pipeline = AutoUpdatePipeline(config, db)
	pipeline._findings_by_id = {}

	proposal = AdaptationProposal(
		id="p1",
		finding_id="unknown",
		title="Adapt: Orphan",
		description="No finding",
		target_modules=[],
	)

	with patch("autodev.intelligence.spec_generator.SpecGenerator") as mock_gen_cls:
		mock_instance = AsyncMock()
		mock_instance.generate_spec = AsyncMock(return_value="# Spec")
		mock_gen_cls.return_value = mock_instance

		await pipeline._generate_spec(proposal)

	mock_instance.generate_spec.assert_called_once_with(proposal, source_url="")


@pytest.mark.asyncio
async def test_generate_spec_or_objective_fallback(config: MissionConfig, db: Database) -> None:
	"""_generate_spec_or_objective falls back to _generate_objective on failure."""
	pipeline = AutoUpdatePipeline(config, db)
	pipeline._findings_by_id = {}

	proposal = AdaptationProposal(
		id="p1",
		finding_id="f1",
		title="Adapt: Failing",
		description="This will fail spec gen",
		target_modules=["config.py"],
		effort_estimate="small",
	)

	with patch.object(pipeline, "_generate_spec", side_effect=RuntimeError("LLM down")):
		result = await pipeline._generate_spec_or_objective(proposal)

	assert "[AUTO-UPDATE]" in result
	assert proposal.title in result


@pytest.mark.asyncio
async def test_generate_spec_or_objective_success(config: MissionConfig, db: Database) -> None:
	"""_generate_spec_or_objective returns spec content when spec generation succeeds."""
	pipeline = AutoUpdatePipeline(config, db)

	proposal = AdaptationProposal(id="p1", title="Test", description="Test")

	with patch.object(pipeline, "_generate_spec", return_value="# Full Spec"):
		result = await pipeline._generate_spec_or_objective(proposal)

	assert result == "# Full Spec"


@pytest.mark.asyncio
async def test_auto_launch_uses_spec(config: MissionConfig, db: Database) -> None:
	"""_auto_launch now calls _generate_spec_or_objective instead of _generate_objective."""
	pipeline = AutoUpdatePipeline(config, db)

	proposal = AdaptationProposal(
		id="p1",
		finding_id="f1",
		title="Adapt: Spec Test",
		description="Test",
		target_modules=["config.py"],
		risk_level="low",
	)

	mock_ratchet = MagicMock()
	mock_ratchet.checkpoint = AsyncMock(return_value="autodev/pre-test")

	with (
		patch("autodev.ratchet.GitRatchet", return_value=mock_ratchet),
		patch.object(pipeline, "_generate_spec_or_objective", return_value="# Spec Content") as mock_gen,
	):
		result = await pipeline._auto_launch(proposal)

	mock_gen.assert_called_once_with(proposal)
	assert result.action == "launched"
	assert result.mission_id != ""
