"""Headless browser auth handler using Playwright."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from autodev.auth.vault import KeychainVault
	from autodev.notifier import TelegramNotifier

log = logging.getLogger(__name__)

# Patterns used to detect auth flow types from page content
OAUTH_PATTERNS = [
	"sign in with google",
	"sign in with github",
	"authorize application",
	"authorize access",
	"oauth",
	"consent screen",
	"allow access",
	"grant permission",
]

API_KEY_PATTERNS = [
	"api key",
	"api token",
	"access token",
	"secret key",
	"copy to clipboard",
	"generate key",
	"create token",
	"your token",
]

CLI_LOGIN_PATTERNS = [
	"paste this code",
	"enter this code",
	"device code",
	"verification code",
	"cli login",
	"authenticate your cli",
	"one-time code",
]

CAPTCHA_PATTERNS = [
	"captcha",
	"recaptcha",
	"hcaptcha",
	"i'm not a robot",
	"verify you are human",
]

TWO_FA_PATTERNS = [
	"two-factor",
	"2fa",
	"verification code",
	"authenticator app",
	"enter the code",
	"security code",
]


@dataclass
class AuthResult:
	success: bool
	service: str
	credential_type: str = ""  # oauth_token, api_key, session_cookie, cli_token
	error: str = ""
	required_human: bool = False


class HeadlessAuthHandler:
	"""Handle authentication flows via headless Playwright browser."""

	def __init__(self, vault: KeychainVault, notifier: TelegramNotifier | None = None):
		self._vault = vault
		self._notifier = notifier
		self._browser = None
		self._pw = None

	async def _ensure_browser(self) -> None:
		"""Launch headless Chromium on first use."""
		if self._browser is not None:
			return
		try:
			from playwright.async_api import async_playwright
		except ImportError:
			raise RuntimeError(
				"playwright is not installed. Install with: "
				"pip install playwright && playwright install chromium"
			)
		self._pw = await async_playwright().start()
		self._browser = await self._pw.chromium.launch(headless=True)

	async def close(self) -> None:
		"""Clean up browser resources."""
		if self._browser:
			await self._browser.close()
		if self._pw:
			await self._pw.stop()
		self._browser = None
		self._pw = None

	async def run_auth_flow(
		self,
		url: str,
		service: str,
		flow_type: str = "auto",
		timeout_s: int = 300,
	) -> AuthResult:
		"""Navigate to auth URL and attempt to complete the flow."""
		try:
			return await asyncio.wait_for(
				self._run_auth_flow_inner(url, service, flow_type),
				timeout=timeout_s,
			)
		except asyncio.TimeoutError:
			return AuthResult(success=False, service=service, error=f"Auth flow timed out after {timeout_s}s")
		except Exception as exc:
			return AuthResult(success=False, service=service, error=str(exc))

	async def _run_auth_flow_inner(self, url: str, service: str, flow_type: str) -> AuthResult:
		"""Inner auth flow without timeout wrapper."""
		await self._ensure_browser()
		page = await self._browser.new_page()
		try:
			await page.goto(url, timeout=30000)

			if flow_type == "auto":
				flow_type = await self._detect_flow_type(page)

			if flow_type == "oauth":
				return await self._handle_oauth(page, service)
			elif flow_type == "api_key":
				return await self._handle_api_key(page, service)
			elif flow_type == "cli_login":
				return await self._handle_cli_login(page, service)
			elif flow_type == "captcha":
				return await self._handle_captcha(page, service)
			elif flow_type == "2fa":
				return await self._handle_2fa(page, service)
			else:
				return await self._handle_stuck(page, service, f"Unknown flow type: {flow_type}")
		finally:
			await page.close()

	async def _detect_flow_type(self, page: object) -> str:
		"""Analyze page content to determine auth flow type."""
		content = (await page.content()).lower()

		for pattern in CAPTCHA_PATTERNS:
			if pattern in content:
				return "captcha"

		for pattern in TWO_FA_PATTERNS:
			if pattern in content:
				return "2fa"

		for pattern in OAUTH_PATTERNS:
			if pattern in content:
				return "oauth"

		for pattern in API_KEY_PATTERNS:
			if pattern in content:
				return "api_key"

		for pattern in CLI_LOGIN_PATTERNS:
			if pattern in content:
				return "cli_login"

		return "unknown"

	async def _handle_oauth(self, page: object, service: str) -> AuthResult:
		"""Handle OAuth redirect flow."""
		try:
			# Look for authorize/allow buttons and click them
			for selector in [
				"button:has-text('Authorize')",
				"button:has-text('Allow')",
				"button:has-text('Accept')",
				"button:has-text('Grant')",
				"input[type='submit'][value*='Authorize']",
				"input[type='submit'][value*='Allow']",
			]:
				btn = page.locator(selector).first
				if await btn.is_visible():
					await btn.click()
					await page.wait_for_load_state("networkidle", timeout=10000)
					break

			# Check for token in URL after redirect
			current_url = page.url
			if "token=" in current_url or "code=" in current_url:
				from urllib.parse import parse_qs, urlparse

				parsed = urlparse(current_url)
				params = parse_qs(parsed.query)
				fragment_params = parse_qs(parsed.fragment)

				token = (
					params.get("access_token", [None])[0]
					or fragment_params.get("access_token", [None])[0]
					or params.get("code", [None])[0]
				)
				if token:
					await self._vault.store(service, "oauth_token", token)
					return AuthResult(success=True, service=service, credential_type="oauth_token")

			return await self._handle_stuck(page, service, "OAuth flow did not complete")
		except Exception as exc:
			return await self._handle_stuck(page, service, f"OAuth error: {exc}")

	async def _handle_api_key(self, page: object, service: str) -> AuthResult:
		"""Handle API key page."""
		try:
			for selector in [
				"input[readonly]",
				"code",
				"pre",
				".api-key",
				"[data-testid='api-key']",
				"input[type='text'][value]",
			]:
				elements = page.locator(selector)
				count = await elements.count()
				for i in range(count):
					el = elements.nth(i)
					text = await el.text_content() or await el.get_attribute("value") or ""
					text = text.strip()
					if len(text) >= 20 and text.replace("-", "").replace("_", "").isalnum():
						await self._vault.store(service, "api_key", text)
						return AuthResult(success=True, service=service, credential_type="api_key")

			return await self._handle_stuck(page, service, "Could not find API key on page")
		except Exception as exc:
			return await self._handle_stuck(page, service, f"API key error: {exc}")

	async def _handle_cli_login(self, page: object, service: str) -> AuthResult:
		"""Handle CLI login flow."""
		try:
			for selector in ["code", "pre", ".code", "[data-testid='code']", "strong"]:
				elements = page.locator(selector)
				count = await elements.count()
				for i in range(count):
					el = elements.nth(i)
					text = (await el.text_content() or "").strip()
					if 4 <= len(text) <= 20:
						await self._vault.store(service, "cli_token", text)
						return AuthResult(success=True, service=service, credential_type="cli_token")

			return await self._handle_stuck(page, service, "Could not find CLI code on page")
		except Exception as exc:
			return await self._handle_stuck(page, service, f"CLI login error: {exc}")

	async def _handle_stuck(self, page: object, service: str, reason: str) -> AuthResult:
		"""When stuck: screenshot page, send to Telegram, wait for help."""
		log.warning("Auth stuck for %s: %s", service, reason)

		if self._notifier and hasattr(self._notifier, "send_auth_help"):
			try:
				screenshot_bytes = await page.screenshot()
				with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
					f.write(screenshot_bytes)
					screenshot_path = f.name
				await self._notifier.send_auth_help(service, reason, screenshot_path)
			except Exception:
				log.exception("Failed to send auth help notification")

		return AuthResult(
			success=False,
			service=service,
			error=f"Stuck: {reason}",
			required_human=True,
		)

	async def _handle_captcha(self, page: object, service: str) -> AuthResult:
		"""CAPTCHA detected, send screenshot to Telegram."""
		return await self._handle_stuck(page, service, "CAPTCHA detected")

	async def _handle_2fa(self, page: object, service: str) -> AuthResult:
		"""2FA prompt detected, send screenshot to Telegram."""
		return await self._handle_stuck(page, service, "2FA prompt detected")
