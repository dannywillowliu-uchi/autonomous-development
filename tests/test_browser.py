"""Tests for headless browser auth handler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autodev.auth.browser import (
	AuthResult,
	HeadlessAuthHandler,
)


@pytest.fixture
def mock_vault():
	vault = AsyncMock()
	vault.store = AsyncMock()
	vault.get = AsyncMock(return_value=None)
	return vault


@pytest.fixture
def mock_notifier():
	notifier = AsyncMock()
	notifier.send_auth_help = AsyncMock()
	return notifier


@pytest.fixture
def handler(mock_vault, mock_notifier):
	return HeadlessAuthHandler(vault=mock_vault, notifier=mock_notifier)


def _make_mock_page(content: str = "<html></html>", url: str = "https://example.com"):
	"""Create a mock Playwright page with configurable content and URL."""
	page = AsyncMock()
	page.content = AsyncMock(return_value=content)
	page.url = url
	page.goto = AsyncMock()
	page.close = AsyncMock()
	page.screenshot = AsyncMock()
	page.wait_for_load_state = AsyncMock()

	# Default locator that finds nothing
	locator = AsyncMock()
	locator.first = locator
	locator.is_visible = AsyncMock(return_value=False)
	locator.count = AsyncMock(return_value=0)
	locator.nth = MagicMock(return_value=locator)
	locator.text_content = AsyncMock(return_value="")
	locator.get_attribute = AsyncMock(return_value=None)
	page.locator = MagicMock(return_value=locator)

	return page


def _make_mock_browser(page):
	"""Create mock browser and playwright context manager."""
	browser = AsyncMock()
	browser.new_page = AsyncMock(return_value=page)
	browser.close = AsyncMock()

	pw_instance = AsyncMock()
	pw_instance.chromium.launch = AsyncMock(return_value=browser)
	pw_instance.stop = AsyncMock()

	pw_cm = AsyncMock()
	pw_cm.start = AsyncMock(return_value=pw_instance)

	return browser, pw_instance, pw_cm


@pytest.fixture
def mock_playwright():
	"""Patch playwright and return (page, browser, pw_instance, pw_cm)."""
	page = _make_mock_page()
	browser, pw_instance, pw_cm = _make_mock_browser(page)

	with patch("autodev.auth.browser.HeadlessAuthHandler._ensure_browser") as mock_ensure:
		async def set_browser(self=None):
			pass

		yield page, browser, pw_instance, pw_cm, mock_ensure


# --- AuthResult tests ---


class TestAuthResult:
	def test_success_result(self):
		r = AuthResult(success=True, service="github", credential_type="oauth_token")
		assert r.success is True
		assert r.service == "github"
		assert r.credential_type == "oauth_token"
		assert r.error == ""
		assert r.required_human is False

	def test_failure_result(self):
		r = AuthResult(success=False, service="gitlab", error="timeout")
		assert r.success is False
		assert r.error == "timeout"

	def test_human_required(self):
		r = AuthResult(success=False, service="x", required_human=True, error="CAPTCHA")
		assert r.required_human is True


# --- Lazy browser initialization ---


class TestLazyInit:
	async def test_browser_not_initialized_on_construction(self, handler):
		assert handler._browser is None
		assert handler._pw is None

	async def test_ensure_browser_called_on_auth_flow(self, mock_vault, mock_notifier):
		handler = HeadlessAuthHandler(vault=mock_vault, notifier=mock_notifier)
		page = _make_mock_page()
		browser, pw_instance, pw_cm = _make_mock_browser(page)

		mock_pw_module = MagicMock()
		mock_pw_module.async_playwright = MagicMock(return_value=pw_cm)
		with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.async_api": mock_pw_module}):
			await handler._ensure_browser()
			assert handler._browser is not None
			assert handler._pw is not None

	async def test_ensure_browser_idempotent(self, mock_vault, mock_notifier):
		handler = HeadlessAuthHandler(vault=mock_vault, notifier=mock_notifier)
		page = _make_mock_page()
		browser, pw_instance, pw_cm = _make_mock_browser(page)

		mock_pw_module = MagicMock()
		mock_pw_module.async_playwright = MagicMock(return_value=pw_cm)
		with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.async_api": mock_pw_module}):
			await handler._ensure_browser()
			first_browser = handler._browser
			await handler._ensure_browser()
			assert handler._browser is first_browser


# --- Flow detection ---


class TestDetectFlowType:
	async def test_detect_oauth(self, handler):
		page = _make_mock_page(content="<html><body>Sign in with Google</body></html>")
		result = await handler._detect_flow_type(page)
		assert result == "oauth"

	async def test_detect_oauth_github(self, handler):
		page = _make_mock_page(content="<html><body>Authorize application on GitHub</body></html>")
		result = await handler._detect_flow_type(page)
		assert result == "oauth"

	async def test_detect_api_key(self, handler):
		page = _make_mock_page(content="<html><body>Your API Key: abc123</body></html>")
		result = await handler._detect_flow_type(page)
		assert result == "api_key"

	async def test_detect_cli_login(self, handler):
		page = _make_mock_page(content="<html><body>Paste this code into your CLI</body></html>")
		result = await handler._detect_flow_type(page)
		assert result == "cli_login"

	async def test_detect_captcha(self, handler):
		page = _make_mock_page(content="<html><body>Please complete the CAPTCHA</body></html>")
		result = await handler._detect_flow_type(page)
		assert result == "captcha"

	async def test_detect_2fa(self, handler):
		page = _make_mock_page(content="<html><body>Enter your two-factor authentication code</body></html>")
		result = await handler._detect_flow_type(page)
		assert result == "2fa"

	async def test_detect_unknown(self, handler):
		page = _make_mock_page(content="<html><body>Nothing relevant here</body></html>")
		result = await handler._detect_flow_type(page)
		assert result == "unknown"

	async def test_captcha_takes_priority_over_oauth(self, handler):
		page = _make_mock_page(content="<html>Sign in with Google<div class='captcha'>CAPTCHA</div></html>")
		result = await handler._detect_flow_type(page)
		assert result == "captcha"


# --- Handler tests ---


class TestHandleStuck:
	async def test_stuck_sends_screenshot_to_notifier(self, handler, mock_notifier):
		page = _make_mock_page()
		result = await handler._handle_stuck(page, "test-service", "something went wrong")
		assert result.success is False
		assert result.required_human is True
		assert result.service == "test-service"
		assert "something went wrong" in result.error
		mock_notifier.send_auth_help.assert_called_once()
		page.screenshot.assert_called_once()

	async def test_stuck_without_notifier(self, mock_vault):
		handler = HeadlessAuthHandler(vault=mock_vault, notifier=None)
		page = _make_mock_page()
		result = await handler._handle_stuck(page, "test-service", "no notifier")
		assert result.success is False
		assert result.required_human is True
		assert result.error == "no notifier"

	async def test_stuck_screenshot_failure_still_returns_result(self, handler, mock_notifier):
		page = _make_mock_page()
		page.screenshot = AsyncMock(side_effect=Exception("screenshot failed"))
		result = await handler._handle_stuck(page, "svc", "reason")
		assert result.success is False
		assert result.required_human is True


class TestHandleCaptcha:
	async def test_captcha_delegates_to_stuck(self, handler, mock_notifier):
		page = _make_mock_page()
		result = await handler._handle_captcha(page, "svc")
		assert result.success is False
		assert result.required_human is True
		assert "CAPTCHA" in result.error
		mock_notifier.send_auth_help.assert_called_once()


class TestHandle2FA:
	async def test_2fa_delegates_to_stuck(self, handler, mock_notifier):
		page = _make_mock_page()
		result = await handler._handle_2fa(page, "svc")
		assert result.success is False
		assert result.required_human is True
		assert "2FA" in result.error
		mock_notifier.send_auth_help.assert_called_once()


class TestHandleOAuth:
	async def test_oauth_captures_token_from_redirect(self, handler, mock_vault):
		page = _make_mock_page(url="https://callback.example.com?code=abc123&state=xyz")

		# No visible authorize buttons
		locator = AsyncMock()
		locator.first = locator
		locator.is_visible = AsyncMock(return_value=False)
		page.locator = MagicMock(return_value=locator)

		result = await handler._handle_oauth(page, "github")
		assert result.success is True
		assert result.credential_type == "oauth_token"
		mock_vault.store.assert_called_once_with("github", "oauth_token", "abc123")

	async def test_oauth_stuck_when_no_token(self, handler, mock_notifier):
		page = _make_mock_page(url="https://example.com/login")
		locator = AsyncMock()
		locator.first = locator
		locator.is_visible = AsyncMock(return_value=False)
		page.locator = MagicMock(return_value=locator)

		result = await handler._handle_oauth(page, "github")
		assert result.success is False
		assert result.required_human is True


class TestHandleApiKey:
	async def test_api_key_found_in_readonly_input(self, handler, mock_vault):
		page = _make_mock_page()
		fake_key = "sk-abcdefghij1234567890abcdef"

		locator = AsyncMock()
		locator.count = AsyncMock(return_value=1)
		el = AsyncMock()
		el.text_content = AsyncMock(return_value=fake_key)
		el.get_attribute = AsyncMock(return_value=None)
		locator.nth = MagicMock(return_value=el)

		# First locator call (input[readonly]) finds the key
		call_count = 0

		def mock_locator(selector):
			nonlocal call_count
			call_count += 1
			if call_count == 1:
				return locator
			empty = AsyncMock()
			empty.count = AsyncMock(return_value=0)
			return empty

		page.locator = MagicMock(side_effect=mock_locator)

		result = await handler._handle_api_key(page, "openai")
		assert result.success is True
		assert result.credential_type == "api_key"
		mock_vault.store.assert_called_once_with("openai", "api_key", fake_key)

	async def test_api_key_not_found_triggers_stuck(self, handler, mock_notifier):
		page = _make_mock_page()
		locator = AsyncMock()
		locator.count = AsyncMock(return_value=0)
		page.locator = MagicMock(return_value=locator)

		result = await handler._handle_api_key(page, "openai")
		assert result.success is False
		assert result.required_human is True


class TestHandleCliLogin:
	async def test_cli_code_found(self, handler, mock_vault):
		page = _make_mock_page()

		locator = AsyncMock()
		locator.count = AsyncMock(return_value=1)
		el = AsyncMock()
		el.text_content = AsyncMock(return_value="ABCD-1234")
		locator.nth = MagicMock(return_value=el)

		call_count = 0

		def mock_locator(selector):
			nonlocal call_count
			call_count += 1
			if call_count == 1:
				return locator
			empty = AsyncMock()
			empty.count = AsyncMock(return_value=0)
			return empty

		page.locator = MagicMock(side_effect=mock_locator)

		result = await handler._handle_cli_login(page, "vercel")
		assert result.success is True
		assert result.credential_type == "cli_token"
		mock_vault.store.assert_called_once_with("vercel", "cli_token", "ABCD-1234")

	async def test_cli_code_not_found_triggers_stuck(self, handler, mock_notifier):
		page = _make_mock_page()
		locator = AsyncMock()
		locator.count = AsyncMock(return_value=0)
		page.locator = MagicMock(return_value=locator)

		result = await handler._handle_cli_login(page, "vercel")
		assert result.success is False
		assert result.required_human is True


# --- Full flow tests ---


class TestRunAuthFlow:
	async def test_timeout_returns_error_result(self, mock_vault, mock_notifier):
		handler = HeadlessAuthHandler(vault=mock_vault, notifier=mock_notifier)

		async def slow_inner(*args, **kwargs):
			await asyncio.sleep(10)

		with patch.object(handler, "_run_auth_flow_inner", side_effect=slow_inner):
			result = await handler.run_auth_flow("https://example.com", "svc", timeout_s=0)
			assert result.success is False
			assert "timed out" in result.error

	async def test_exception_returns_error_result(self, mock_vault, mock_notifier):
		handler = HeadlessAuthHandler(vault=mock_vault, notifier=mock_notifier)

		with patch.object(handler, "_run_auth_flow_inner", side_effect=RuntimeError("browser crashed")):
			result = await handler.run_auth_flow("https://example.com", "svc")
			assert result.success is False
			assert "browser crashed" in result.error

	async def test_explicit_flow_type_skips_detection(self, mock_vault, mock_notifier):
		handler = HeadlessAuthHandler(vault=mock_vault, notifier=mock_notifier)
		page = _make_mock_page(
			content="<html>Nothing relevant</html>",
			url="https://callback.example.com?code=tok123",
		)
		browser, pw_instance, pw_cm = _make_mock_browser(page)

		handler._browser = browser
		handler._pw = pw_instance
		result = await handler.run_auth_flow(
			"https://example.com",
			"github",
			flow_type="oauth",
		)
		assert result.success is True
		assert result.credential_type == "oauth_token"

	async def test_auto_detection_routes_correctly(self, mock_vault, mock_notifier):
		handler = HeadlessAuthHandler(vault=mock_vault, notifier=mock_notifier)
		page = _make_mock_page(
			content="<html>Your API Key is here</html>",
		)
		browser, pw_instance, pw_cm = _make_mock_browser(page)

		# Set up locator to return no API keys (will fall through to stuck)
		locator = AsyncMock()
		locator.count = AsyncMock(return_value=0)
		page.locator = MagicMock(return_value=locator)

		handler._browser = browser
		handler._pw = pw_instance
		result = await handler.run_auth_flow(
			"https://example.com/api-keys",
			"openai",
			flow_type="auto",
		)
		# Detected api_key but couldn't find one -> stuck
		assert result.success is False
		assert result.required_human is True

	async def test_unknown_flow_type_triggers_stuck(self, mock_vault, mock_notifier):
		handler = HeadlessAuthHandler(vault=mock_vault, notifier=mock_notifier)
		page = _make_mock_page()
		browser, pw_instance, pw_cm = _make_mock_browser(page)

		handler._browser = browser
		handler._pw = pw_instance
		result = await handler.run_auth_flow(
			"https://example.com",
			"svc",
			flow_type="magic",
		)
		assert result.success is False
		assert "Unknown flow type" in result.error


# --- Close / cleanup ---


class TestClose:
	async def test_close_cleans_up_resources(self, mock_vault):
		handler = HeadlessAuthHandler(vault=mock_vault)
		handler._browser = AsyncMock()
		handler._pw = AsyncMock()

		await handler.close()
		handler._browser is None
		handler._pw is None

	async def test_close_noop_when_not_initialized(self, mock_vault):
		handler = HeadlessAuthHandler(vault=mock_vault)
		await handler.close()  # Should not raise
		assert handler._browser is None
		assert handler._pw is None
