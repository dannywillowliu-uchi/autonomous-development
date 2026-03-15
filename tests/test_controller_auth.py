"""Tests for swarm controller auth request/response protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autodev.config import SwarmConfig
from autodev.swarm.controller import SwarmController
from autodev.swarm.models import AgentRole, SwarmAgent


def _make_config(tmp_path: Path) -> MagicMock:
	config = MagicMock()
	config.target.name = "test-project"
	config.target.objective = "Build a compiler"
	config.target.resolved_path = str(tmp_path)
	config.notifications.telegram.bot_token = ""
	config.notifications.telegram.chat_id = ""
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


@dataclass
class FakeAuthResult:
	success: bool
	service: str
	credential_type: str = ""
	error: str = ""
	required_human: bool = False


class TestHandleAuthRequest:
	@pytest.mark.asyncio
	async def test_successful_auth_writes_response(self, tmp_path: Path) -> None:
		"""handle_auth_request delegates to gateway and writes auth_response to worker inbox."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		with patch.object(Path, "home", return_value=tmp_path):
			await ctrl.initialize()

		fake_result = FakeAuthResult(
			success=True,
			service="github",
			credential_type="oauth_token",
		)
		mock_gateway = AsyncMock()
		mock_gateway.authenticate.return_value = fake_result

		with patch("autodev.swarm.controller.SwarmController.handle_auth_request", wraps=ctrl.handle_auth_request), \
			patch.dict("sys.modules", {
				"autodev.auth.vault": MagicMock(),
				"autodev.auth.gateway": MagicMock(),
				"autodev.auth.browser": MagicMock(),
			}), \
			patch("autodev.swarm.controller.Path.home", return_value=tmp_path):

			# Directly mock the gateway construction path
			with patch.object(ctrl, "_write_to_inbox") as mock_write:
				# Bypass the gateway import and just test the inbox writing
				# by calling the internal logic
				pass

		# More targeted test: mock the entire auth chain and verify inbox writing
		with patch.object(Path, "home", return_value=tmp_path):
			# Create the worker inbox directory
			inbox_dir = tmp_path / ".claude" / "teams" / "autodev-test-project" / "inboxes"
			inbox_dir.mkdir(parents=True, exist_ok=True)
			(inbox_dir / "test-worker.json").write_text("[]")

			with patch("autodev.auth.vault.KeychainVault", return_value=MagicMock()), \
				patch("autodev.auth.browser.HeadlessAuthHandler") as mock_browser_cls, \
				patch("autodev.auth.gateway.AuthGateway") as mock_gw_cls:

				mock_browser = AsyncMock()
				mock_browser_cls.return_value = mock_browser

				mock_gw = AsyncMock()
				mock_gw.authenticate.return_value = fake_result
				mock_gw_cls.return_value = mock_gw

				result = await ctrl.handle_auth_request(
					service="github",
					url="https://github.com/login",
					purpose="CLI access",
					requesting_agent="test-worker",
				)

			assert result["success"] is True
			assert result["service"] == "github"
			assert result["credential_type"] == "oauth_token"

			# Verify auth_response was written to the worker's inbox
			inbox_data = json.loads((inbox_dir / "test-worker.json").read_text())
			assert len(inbox_data) == 1
			msg = inbox_data[0]
			assert msg["type"] == "auth_response"
			assert msg["service"] == "github"
			assert msg["success"] is True
			assert "oauth_token" in msg["credential_type"]
			assert "Keychain" in msg["instructions"]

	@pytest.mark.asyncio
	async def test_failed_auth_writes_failure_response(self, tmp_path: Path) -> None:
		"""Failed auth writes error details to worker inbox."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		with patch.object(Path, "home", return_value=tmp_path):
			await ctrl.initialize()

		fake_result = FakeAuthResult(
			success=False,
			service="google-workspace",
			error="OAuth consent denied",
		)

		with patch.object(Path, "home", return_value=tmp_path):
			inbox_dir = tmp_path / ".claude" / "teams" / "autodev-test-project" / "inboxes"
			(inbox_dir / "worker-a.json").write_text("[]")

			with patch("autodev.auth.vault.KeychainVault", return_value=MagicMock()), \
				patch("autodev.auth.browser.HeadlessAuthHandler") as mock_browser_cls, \
				patch("autodev.auth.gateway.AuthGateway") as mock_gw_cls:

				mock_browser_cls.return_value = AsyncMock()
				mock_gw = AsyncMock()
				mock_gw.authenticate.return_value = fake_result
				mock_gw_cls.return_value = mock_gw

				result = await ctrl.handle_auth_request(
					service="google-workspace",
					url="https://accounts.google.com",
					purpose="Calendar API",
					requesting_agent="worker-a",
				)

			assert result["success"] is False
			assert result["error"] == "OAuth consent denied"

			inbox_data = json.loads((inbox_dir / "worker-a.json").read_text())
			msg = inbox_data[0]
			assert msg["type"] == "auth_response"
			assert msg["success"] is False
			assert "failed" in msg["instructions"].lower()

	@pytest.mark.asyncio
	async def test_import_error_fallback(self, tmp_path: Path) -> None:
		"""When auth modules aren't installed, returns graceful error."""
		ctrl = SwarmController(_make_config(tmp_path), _make_swarm_config(), _make_db())
		with patch.object(Path, "home", return_value=tmp_path):
			await ctrl.initialize()
			inbox_dir = tmp_path / ".claude" / "teams" / "autodev-test-project" / "inboxes"
			(inbox_dir / "worker-b.json").write_text("[]")

			with patch.dict("sys.modules", {
				"autodev.auth.vault": None,
				"autodev.auth.gateway": None,
				"autodev.auth.browser": None,
			}):
				result = await ctrl.handle_auth_request(
					service="slack",
					url="https://slack.com/oauth",
					purpose="Workspace access",
					requesting_agent="worker-b",
				)

			assert result["success"] is False
			assert "not installed" in result["error"].lower() or "no module" in result["error"].lower()

			inbox_data = json.loads((inbox_dir / "worker-b.json").read_text())
			assert len(inbox_data) == 1
			assert inbox_data[0]["type"] == "auth_response"
			assert inbox_data[0]["success"] is False

	@pytest.mark.asyncio
	async def test_notifies_telegram_on_auth_request(self, tmp_path: Path) -> None:
		"""Auth request sends Telegram notification when configured."""
		config = _make_config(tmp_path)
		config.notifications.telegram.bot_token = "test-token"
		config.notifications.telegram.chat_id = "12345"

		ctrl = SwarmController(config, _make_swarm_config(), _make_db())
		with patch.object(Path, "home", return_value=tmp_path):
			await ctrl.initialize()
			inbox_dir = tmp_path / ".claude" / "teams" / "autodev-test-project" / "inboxes"
			(inbox_dir / "worker-c.json").write_text("[]")

			with patch("autodev.auth.vault.KeychainVault", return_value=MagicMock()), \
				patch("autodev.auth.browser.HeadlessAuthHandler") as mock_browser_cls, \
				patch("autodev.auth.gateway.AuthGateway") as mock_gw_cls, \
				patch("autodev.swarm.controller.TelegramNotifier") as mock_notifier_cls:

				mock_browser_cls.return_value = AsyncMock()
				mock_gw = AsyncMock()
				mock_gw.authenticate.return_value = FakeAuthResult(
					success=True, service="github", credential_type="oauth_token",
				)
				mock_gw_cls.return_value = mock_gw

				mock_notifier = AsyncMock()
				mock_notifier_cls.return_value = mock_notifier

				await ctrl.handle_auth_request(
					service="github",
					url="https://github.com/login",
					purpose="CLI",
					requesting_agent="worker-c",
				)

				mock_notifier.send_auth_request.assert_called_once_with(
					"github", "CLI", "https://github.com/login",
				)


class TestAuthRequestContextIntegration:
	"""Verify auth_request messages are surfaced in planner context."""

	def test_auth_request_in_discovery_filter(self, tmp_path: Path) -> None:
		"""auth_request messages appear in discoveries via context synthesizer."""
		from autodev.swarm.context import ContextSynthesizer

		config = MagicMock()
		config.target.name = "test-project"
		config.target.resolved_path = str(tmp_path)
		db = _make_db()

		team_name = "autodev-test-project"
		inbox_dir = tmp_path / ".claude" / "teams" / team_name / "inboxes"
		inbox_dir.mkdir(parents=True, exist_ok=True)

		# Write an auth_request message to team-lead inbox
		messages = [{
			"from": "worker-1",
			"type": "auth_request",
			"service": "github",
			"url": "https://github.com/login",
			"purpose": "CLI access",
			"text": "Need auth for github",
		}]
		(inbox_dir / "team-lead.json").write_text(json.dumps(messages))

		with patch.object(Path, "home", return_value=tmp_path):
			ctx = ContextSynthesizer(config, db, team_name)
			discoveries = ctx._get_recent_discoveries()

		assert any("auth_request" in d for d in discoveries)
		assert any("github" in d for d in discoveries)


class TestWorkerPromptAuth:
	"""Verify auth request instructions appear in worker prompts."""

	def test_auth_section_in_worker_prompt(self, tmp_path: Path) -> None:
		"""build_worker_prompt includes authentication instructions."""
		from autodev.swarm.worker_prompt import build_worker_prompt

		agent = SwarmAgent(name="worker-1", role=AgentRole.IMPLEMENTER, status=AgentStatus.WORKING)
		config = MagicMock()
		config.target.resolved_path = str(tmp_path)
		config.target.verification = None
		sc = SwarmConfig()

		prompt = build_worker_prompt(
			agent=agent,
			task_prompt="Do stuff",
			team_name="autodev-test",
			agents=[agent],
			tasks=[],
			config=config,
			swarm_config=sc,
		)
		assert "## Authentication" in prompt
		assert "auth_request" in prompt
		assert "auth_response" in prompt
		assert "team-lead.json" in prompt

	def test_auth_section_includes_agent_name(self, tmp_path: Path) -> None:
		"""Auth section personalizes agent name in message template."""
		from autodev.swarm.worker_prompt import _auth_request_section

		agent = SwarmAgent(name="my-agent", role=AgentRole.IMPLEMENTER, status=AgentStatus.WORKING)
		text = _auth_request_section(agent, "autodev-test")
		assert "my-agent" in text
		assert "autodev-test" in text
