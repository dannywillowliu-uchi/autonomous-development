"""Tests for auto_discovery module."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from mission_control.auto_discovery import (
	AnalysisOutput,
	DiscoveryEngine,
	ResearchOutput,
	_compute_priority,
)
from mission_control.config import (
	DiscoveryConfig,
	MissionConfig,
	TargetConfig,
	VerificationConfig,
)
from mission_control.db import Database
from mission_control.models import BacklogItem, DiscoveryResult


def _config(*, research_enabled: bool = True) -> MissionConfig:
	return MissionConfig(
		target=TargetConfig(
			name="test-proj",
			path="/tmp/test",
			branch="main",
			objective="",
			verification=VerificationConfig(command="pytest -q"),
		),
		discovery=DiscoveryConfig(
			enabled=True,
			tracks=["feature", "quality", "security"],
			max_items_per_track=3,
			min_priority_score=3.0,
			model="sonnet",
			budget_per_call_usd=1.0,
			research_enabled=research_enabled,
		),
	)


def _discovery_json(items: list[dict]) -> str:
	return f"DISCOVERY_RESULT:\n```json\n{json.dumps({'items': items})}\n```"


def _analysis_json(
	gaps: list[dict] | None = None,
	architecture: str = "modular",
	patterns: list[str] | None = None,
) -> str:
	default_gaps = [{"category": "testing", "description": "Missing tests", "files": "src/foo.py", "severity": "high"}]
	data = {
		"architecture": architecture,
		"patterns": patterns if patterns is not None else ["dataclass", "async"],
		"gaps": gaps if gaps is not None else default_gaps,
	}
	return f"ANALYSIS_RESULT:\n```json\n{json.dumps(data)}\n```"


def _research_json(category: str = "testing") -> str:
	data = {
		"gap_category": category,
		"best_practices": "Use pytest fixtures",
		"examples": "FastAPI project uses pytest-asyncio",
		"sources": "https://example.com",
	}
	return f"RESEARCH_RESULT:\n```json\n{json.dumps(data)}\n```"


def _sample_item(
	track: str = "quality",
	title: str = "Fix test coverage",
	impact: int = 7,
	effort: int = 3,
) -> dict:
	return {
		"track": track,
		"title": title,
		"description": "Add missing tests for utils module.",
		"rationale": "Low coverage in critical paths.",
		"files_hint": "src/utils.py, tests/test_utils.py",
		"impact": impact,
		"effort": effort,
	}


def _mock_proc(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> AsyncMock:
	mock = AsyncMock()
	mock.communicate = AsyncMock(return_value=(stdout, stderr))
	mock.returncode = returncode
	return mock


class TestComputePriority:
	def test_basic(self) -> None:
		assert _compute_priority(7, 3) == 5.6

	def test_clamps_values(self) -> None:
		assert _compute_priority(0, 0) == 1.0  # clamped to 1,1
		assert _compute_priority(15, 15) == 1.0  # clamped to 10,10


class TestParseDiscoveryOutput:
	def setup_method(self) -> None:
		self.config = _config()
		self.db = Database(":memory:")
		self.engine = DiscoveryEngine(self.config, self.db)

	def test_parse_valid_output(self) -> None:
		output = _discovery_json([_sample_item()])
		result, items = self.engine._parse_discovery_output(output)
		assert result.target_path == "/tmp/test"
		assert result.model == "sonnet"
		assert result.item_count == 1
		assert items[0].title == "Fix test coverage"
		assert items[0].track == "quality"
		assert items[0].priority_score == 5.6

	def test_filters_low_priority(self) -> None:
		output = _discovery_json([
			_sample_item(impact=1, effort=10),  # priority 0.1
			_sample_item(impact=8, effort=2, title="High priority"),
		])
		result, items = self.engine._parse_discovery_output(output)
		assert result.item_count == 1
		assert items[0].title == "High priority"

	def test_empty_output(self) -> None:
		result, items = self.engine._parse_discovery_output("")
		assert result.item_count == 0
		assert items == []

	def test_fallback_json_without_marker(self) -> None:
		raw = json.dumps({"items": [_sample_item()]})
		result, items = self.engine._parse_discovery_output(raw)
		assert result.item_count == 1


class TestParseAnalysisOutput:
	def setup_method(self) -> None:
		self.config = _config()
		self.db = Database(":memory:")
		self.engine = DiscoveryEngine(self.config, self.db)

	def test_parse_valid_analysis(self) -> None:
		output = _analysis_json()
		result = self.engine._parse_analysis_output(output)
		assert result.architecture == "modular"
		assert "dataclass" in result.patterns
		assert len(result.gaps) == 1
		assert result.gaps[0]["category"] == "testing"

	def test_parse_empty_output(self) -> None:
		result = self.engine._parse_analysis_output("")
		assert result.architecture == ""
		assert result.gaps == []
		assert result.raw == ""


class TestParseResearchResult:
	def setup_method(self) -> None:
		self.config = _config()
		self.db = Database(":memory:")
		self.engine = DiscoveryEngine(self.config, self.db)

	def test_parse_valid_research(self) -> None:
		output = _research_json("testing")
		result = self.engine._parse_research_result(output, "testing")
		assert result is not None
		assert result["gap_category"] == "testing"
		assert "pytest" in result["best_practices"]


class TestSynthesizePrompt:
	def setup_method(self) -> None:
		self.config = _config()
		self.db = Database(":memory:")
		self.engine = DiscoveryEngine(self.config, self.db)

	def test_includes_tracks(self) -> None:
		analysis = AnalysisOutput(architecture="modular", gaps=[])
		prompt = self.engine._build_synthesize_prompt(analysis, None)
		assert "Track A - Features" in prompt
		assert "Track B - Code Quality" in prompt
		assert "Track C - Security" in prompt

	def test_includes_analysis_context(self) -> None:
		analysis = AnalysisOutput(
			architecture="layered MVC",
			patterns=["repository pattern"],
			gaps=[{"category": "testing", "description": "no tests"}],
		)
		prompt = self.engine._build_synthesize_prompt(analysis, None)
		assert "layered MVC" in prompt
		assert "repository pattern" in prompt
		assert "no tests" in prompt

	def test_includes_research_context(self) -> None:
		analysis = AnalysisOutput(gaps=[])
		research = ResearchOutput(findings=[
			{"gap_category": "testing", "best_practices": "Use pytest fixtures", "examples": "", "sources": ""},
		])
		prompt = self.engine._build_synthesize_prompt(analysis, research)
		assert "Research Findings" in prompt
		assert "pytest fixtures" in prompt


class TestComposeObjective:
	def setup_method(self) -> None:
		self.config = _config()
		self.db = Database(":memory:")
		self.engine = DiscoveryEngine(self.config, self.db)

	def test_basic(self) -> None:
		items = [
			BacklogItem(
				track="feature",
				title="Add endpoint",
				description="Add /users endpoint",
				priority_score=8.0,
			),
		]
		obj = self.engine.compose_objective(items)
		assert "Add endpoint" in obj
		assert "Add /users endpoint" in obj
		assert "Features" in obj


class TestDBIntegration:
	def test_insert_and_retrieve(self) -> None:
		db = Database(":memory:")
		dr = DiscoveryResult(target_path="/tmp/test", model="sonnet", item_count=2)
		items = [
			BacklogItem(track="feature", title="Item 1", priority_score=8.0),
			BacklogItem(track="quality", title="Item 2", priority_score=5.0),
		]
		db.insert_discovery_result(dr, items)

		latest, retrieved_items = db.get_latest_discovery()
		assert latest is not None
		assert latest.id == dr.id
		assert latest.item_count == 2
		assert len(retrieved_items) == 2
		# Should be sorted by priority_score DESC
		assert retrieved_items[0].title == "Item 1"
		assert retrieved_items[1].title == "Item 2"


class TestStageAnalyze:
	@pytest.mark.asyncio
	async def test_analyze_success(self) -> None:
		config = _config()
		db = Database(":memory:")
		engine = DiscoveryEngine(config, db)

		analysis_output = _analysis_json()
		mock = _mock_proc(stdout=analysis_output.encode())

		with patch("mission_control.auto_discovery.asyncio.create_subprocess_exec", return_value=mock):
			result = await engine._stage_analyze()

		assert result is not None
		assert result.architecture == "modular"
		assert len(result.gaps) == 1


class TestStageResearch:
	@pytest.mark.asyncio
	async def test_research_success(self) -> None:
		config = _config()
		db = Database(":memory:")
		engine = DiscoveryEngine(config, db)

		analysis = AnalysisOutput(
			gaps=[{"category": "testing", "description": "Missing tests"}],
		)
		research_output = _research_json("testing")
		mock = _mock_proc(stdout=research_output.encode())

		with patch("mission_control.auto_discovery.asyncio.create_subprocess_exec", return_value=mock):
			result = await engine._stage_research(analysis)

		assert result is not None
		assert len(result.findings) == 1
		assert result.findings[0]["gap_category"] == "testing"

	@pytest.mark.asyncio
	async def test_research_partial_failure(self) -> None:
		"""One query fails, others succeed -- partial results collected."""
		config = _config()
		config.discovery.research_parallel_queries = 3
		db = Database(":memory:")
		engine = DiscoveryEngine(config, db)

		analysis = AnalysisOutput(
			gaps=[
				{"category": "testing", "description": "Missing tests"},
				{"category": "security", "description": "No validation"},
			],
		)

		call_count = 0

		async def mock_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			if call_count == 1:
				# First call (analyze stage won't be made, this is research)
				return _mock_proc(stdout=_research_json("testing").encode())
			else:
				# Second research query fails
				return _mock_proc(returncode=1, stderr=b"timeout")

		with patch("mission_control.auto_discovery.asyncio.create_subprocess_exec", side_effect=mock_exec):
			result = await engine._stage_research(analysis)

		assert result is not None
		assert len(result.findings) == 1
		assert result.findings[0]["gap_category"] == "testing"


class TestFullPipeline:
	@pytest.mark.asyncio
	async def test_pipeline_research_disabled(self) -> None:
		"""Only 2 subprocess calls when research is off (analyze + synthesize)."""
		config = _config(research_enabled=False)
		db = Database(":memory:")
		engine = DiscoveryEngine(config, db)

		call_count = 0

		async def mock_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			if call_count == 1:
				# Stage 1: analyze
				return _mock_proc(stdout=_analysis_json().encode())
			else:
				# Stage 3: synthesize
				return _mock_proc(stdout=_discovery_json([_sample_item()]).encode())

		with patch("mission_control.auto_discovery.asyncio.create_subprocess_exec", side_effect=mock_exec):
			result, items = await engine.discover()

		assert call_count == 2
		assert result.item_count == 1
		assert items[0].title == "Fix test coverage"

	@pytest.mark.asyncio
	async def test_full_pipeline_with_research(self) -> None:
		"""All 3 stages succeed, verify 3+ subprocess calls."""
		config = _config(research_enabled=True)
		db = Database(":memory:")
		engine = DiscoveryEngine(config, db)

		call_count = 0

		async def mock_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			if call_count == 1:
				# Stage 1: analyze
				return _mock_proc(stdout=_analysis_json().encode())
			elif call_count == 2:
				# Stage 2: research (one gap category)
				return _mock_proc(stdout=_research_json("testing").encode())
			else:
				# Stage 3: synthesize
				return _mock_proc(stdout=_discovery_json([_sample_item()]).encode())

		with patch("mission_control.auto_discovery.asyncio.create_subprocess_exec", side_effect=mock_exec):
			result, items = await engine.discover()

		assert call_count == 3  # analyze + 1 research + synthesize
		assert result.item_count == 1
		assert result.error_type == ""

	@pytest.mark.asyncio
	async def test_persists_to_db(self) -> None:
		"""Results are persisted to DB."""
		config = _config(research_enabled=False)
		db = Database(":memory:")
		engine = DiscoveryEngine(config, db)

		call_count = 0

		async def mock_exec(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			if call_count == 1:
				return _mock_proc(stdout=_analysis_json().encode())
			else:
				return _mock_proc(stdout=_discovery_json([_sample_item()]).encode())

		with patch("mission_control.auto_discovery.asyncio.create_subprocess_exec", side_effect=mock_exec):
			result, items = await engine.discover()

		latest, db_items = db.get_latest_discovery()
		assert latest is not None
		assert latest.id == result.id
		assert len(db_items) == 1


class TestRunStageSubprocess:
	@pytest.mark.asyncio
	async def test_timeout_error(self) -> None:
		config = _config()
		db = Database(":memory:")
		engine = DiscoveryEngine(config, db)

		mock = AsyncMock()
		mock.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
		mock.kill = lambda: None
		mock.wait = AsyncMock()

		with patch("mission_control.auto_discovery.asyncio.create_subprocess_exec", return_value=mock):
			output, error_type, error_detail = await engine._run_stage_subprocess(
				"test prompt", model="sonnet", stage_name="test",
			)

		assert error_type == "timeout"
		assert "timed out" in error_detail

	@pytest.mark.asyncio
	async def test_success_no_error(self) -> None:
		config = _config()
		db = Database(":memory:")
		engine = DiscoveryEngine(config, db)

		mock = _mock_proc(stdout=b"hello world")

		with patch("mission_control.auto_discovery.asyncio.create_subprocess_exec", return_value=mock):
			output, error_type, error_detail = await engine._run_stage_subprocess(
				"test prompt", model="sonnet", stage_name="test",
			)

		assert output == "hello world"
		assert error_type == ""
		assert error_detail == ""


class TestDiscoveryToBacklog:
	"""Test that DiscoveryEngine._insert_items_to_backlog inserts into backlog_items."""

	def test_discovery_items_inserted_into_backlog(self, config: MissionConfig, db: Database) -> None:
		"""BacklogItems are inserted into backlog."""
		engine = DiscoveryEngine(config, db)
		items = [
			BacklogItem(
				id="d1", title="Add caching", description="Redis layer",
				priority_score=7.0, impact=8, effort=5, track="feature",
			),
			BacklogItem(
				id="d2", title="Fix XSS", description="Sanitize inputs",
				priority_score=9.0, impact=10, effort=3, track="security",
			),
		]

		engine._insert_items_to_backlog(items)

		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 2
		titles = {item.title for item in backlog}
		assert "Add caching" in titles
		assert "Fix XSS" in titles

	def test_duplicate_titles_not_reinserted(self, config: MissionConfig, db: Database) -> None:
		"""Items with titles already in backlog_items are skipped."""
		# Pre-insert a backlog item with the same title
		db.insert_backlog_item(BacklogItem(
			id="existing", title="Add caching", description="Already exists",
			priority_score=5.0, impact=5, effort=5, track="quality",
		))

		engine = DiscoveryEngine(config, db)
		items = [
			BacklogItem(
				id="d1", title="Add caching", description="Redis layer",
				priority_score=7.0, impact=8, effort=5, track="feature",
			),
			BacklogItem(
				id="d2", title="New item", description="Something new",
				priority_score=6.0, impact=7, effort=4, track="quality",
			),
		]

		engine._insert_items_to_backlog(items)

		backlog = db.list_backlog_items(limit=10)
		assert len(backlog) == 2
		# The existing one should still have the original description
		existing = [i for i in backlog if i.title == "Add caching"]
		assert len(existing) == 1
		assert existing[0].description == "Already exists"
		# The new one should be inserted
		new = [i for i in backlog if i.title == "New item"]
		assert len(new) == 1
