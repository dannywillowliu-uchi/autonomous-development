"""Tiered auth handler -- avoids headless browser OAuth where possible.

Strategy (in order of preference):
1. Environment variable / pre-seeded token (instant, no interaction)
2. Service account key file (non-interactive, covers GCP)
3. Device code flow RFC 8628 (user approves on phone, agent polls)
4. Console flow (agent prints URL, Telegrams Danny, he pastes code back)
5. Headless Playwright (last resort for simple consent/API key pages)
6. Telegram manual fallback (screenshot + ask Danny to do it)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from autodev.auth.vault import KeychainVault
	from autodev.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


@dataclass
class AuthResult:
	success: bool
	service: str
	credential_type: str = ""  # env_var, service_account, device_code, oauth_token, api_key, cli_token
	error: str = ""
	required_human: bool = False
	instructions: str = ""  # how to use the credential


# Well-known env var mappings: service -> env var name
ENV_VAR_MAP: dict[str, list[str]] = {
	"github": ["GH_TOKEN", "GITHUB_TOKEN"],
	"google-cloud": ["GOOGLE_APPLICATION_CREDENTIALS"],
	"openai": ["OPENAI_API_KEY"],
	"anthropic": ["ANTHROPIC_API_KEY"],
	"huggingface": ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
	"vercel": ["VERCEL_TOKEN"],
	"netlify": ["NETLIFY_AUTH_TOKEN"],
	"npm": ["NPM_TOKEN"],
	"docker": ["DOCKER_TOKEN"],
	"aws": ["AWS_ACCESS_KEY_ID"],
}

# Services with CLI tools that support non-interactive auth
CLI_AUTH_MAP: dict[str, dict[str, str]] = {
	"github": {
		"check": "gh",
		"auth_cmd": "gh auth login --with-token",
		"env_var": "GH_TOKEN",
	},
	"google-cloud": {
		"check": "gcloud",
		"auth_cmd": "gcloud auth activate-service-account --key-file={key_file}",
		"env_var": "GOOGLE_APPLICATION_CREDENTIALS",
	},
	"vercel": {
		"check": "vercel",
		"auth_cmd": "vercel login --token {token}",
		"env_var": "VERCEL_TOKEN",
	},
}

# Playwright page patterns (for tier 5 fallback)
OAUTH_PATTERNS = [
	"sign in with google", "sign in with github", "authorize application",
	"authorize access", "oauth", "consent screen", "allow access", "grant permission",
]
API_KEY_PATTERNS = [
	"api key", "api token", "access token", "secret key",
	"copy to clipboard", "generate key", "create token", "your token",
]
CLI_LOGIN_PATTERNS = [
	"paste this code", "enter this code", "device code",
	"verification code", "cli login", "authenticate your cli", "one-time code",
]
CAPTCHA_PATTERNS = ["captcha", "recaptcha", "hcaptcha", "i'm not a robot", "verify you are human"]
TWO_FA_PATTERNS = ["two-factor", "2fa", "verification code", "authenticator app", "security code"]


class AuthHandler:
	"""Tiered auth handler. Tries non-browser methods first, falls back to Playwright."""

	def __init__(self, vault: KeychainVault, notifier: TelegramNotifier | None = None):
		self._vault = vault
		self._notifier = notifier
		self._browser = None
		self._pw = None

	async def close(self) -> None:
		"""Clean up browser resources if launched."""
		if self._browser:
			await self._browser.close()
		if self._pw:
			await self._pw.stop()
		self._browser = None
		self._pw = None

	async def authenticate(
		self,
		service: str,
		purpose: str = "",
		url: str = "",
		flow_type: str = "auto",
		timeout_s: int = 300,
	) -> AuthResult:
		"""Try all tiers in order until one succeeds."""
		# Tier 1: Check env vars
		result = await self._try_env_var(service)
		if result.success:
			return result

		# Tier 2: Check for service account key file
		result = await self._try_service_account(service)
		if result.success:
			return result

		# Tier 3: Device code flow
		result = await self._try_device_code(service, purpose)
		if result.success:
			return result

		# Tier 4: Console flow (print URL, Telegram Danny, he pastes code)
		result = await self._try_console_flow(service, purpose, url)
		if result.success:
			return result

		# Tier 5: Headless Playwright (simple pages only)
		if url:
			result = await self._try_playwright(url, service, flow_type, timeout_s)
			if result.success:
				return result

		# Tier 6: Full manual fallback via Telegram
		return await self._telegram_manual(service, purpose, url)

	# -- Tier 1: Environment variables --

	async def _try_env_var(self, service: str) -> AuthResult:
		"""Check if credentials exist as environment variables."""
		env_vars = ENV_VAR_MAP.get(service, [])
		for var in env_vars:
			value = os.environ.get(var)
			if value:
				logger.info("Found %s env var for %s", var, service)
				await self._vault.store(service, "env_token", value)
				return AuthResult(
					success=True, service=service, credential_type="env_var",
					instructions=f"Token from ${var}",
				)
		return AuthResult(success=False, service=service)

	# -- Tier 2: Service account key files --

	async def _try_service_account(self, service: str) -> AuthResult:
		"""Check for service account key files (GCP pattern)."""
		if service not in ("google-cloud", "gcp", "google-workspace"):
			return AuthResult(success=False, service=service)

		# Check GOOGLE_APPLICATION_CREDENTIALS
		creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
		if creds_path and Path(creds_path).exists():
			return AuthResult(
				success=True, service=service, credential_type="service_account",
				instructions=f"Service account at {creds_path}",
			)

		# Check common locations
		for candidate in [
			Path.home() / ".config" / "gcloud" / "application_default_credentials.json",
			Path.home() / ".config" / "gcloud" / "credentials.db",
		]:
			if candidate.exists():
				return AuthResult(
					success=True, service=service, credential_type="service_account",
					instructions=f"Credentials at {candidate}",
				)

		return AuthResult(success=False, service=service)

	# -- Tier 3: Device code flow (RFC 8628) --

	async def _try_device_code(self, service: str, purpose: str) -> AuthResult:
		"""Try device code flow for services that support it."""
		if service == "github" and shutil.which("gh"):
			return await self._github_device_flow(purpose)
		return AuthResult(success=False, service=service)

	async def _github_device_flow(self, purpose: str) -> AuthResult:
		"""Check gh CLI auth status. Telegram Danny if not authenticated."""
		proc = await asyncio.create_subprocess_exec(
			"gh", "auth", "status",
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		await proc.communicate()
		if proc.returncode == 0:
			return AuthResult(
				success=True, service="github", credential_type="cli_auth",
				instructions="Already authenticated via gh CLI",
			)

		if self._notifier:
			try:
				await self._notifier.send(
					f"GitHub auth needed: {purpose}\n"
					f"Run `gh auth login` on your machine, or set GH_TOKEN env var.",
				)
			except Exception:
				pass
		return AuthResult(
			success=False, service="github",
			error="GitHub CLI not authenticated. Set GH_TOKEN or run gh auth login.",
			required_human=True,
		)

	# -- Tier 4: Console flow (URL + paste code) --

	async def _try_console_flow(self, service: str, purpose: str, url: str) -> AuthResult:
		"""Try console-based auth: Telegram URL to Danny, he pastes code back."""
		cli_info = CLI_AUTH_MAP.get(service)
		if not cli_info:
			return AuthResult(success=False, service=service)

		cli_bin = cli_info.get("check", "")
		if cli_bin and not shutil.which(cli_bin):
			return AuthResult(success=False, service=service, error=f"{cli_bin} not installed")

		# For gcloud: check if already authenticated
		if service == "google-cloud" and shutil.which("gcloud"):
			proc = await asyncio.create_subprocess_exec(
				"gcloud", "auth", "list", "--format=json",
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
			)
			stdout, _ = await proc.communicate()
			if proc.returncode == 0:
				try:
					accounts = json.loads(stdout.decode())
					active = [a for a in accounts if a.get("status") == "ACTIVE"]
					if active:
						return AuthResult(
							success=True, service=service, credential_type="cli_auth",
							instructions=f"Already authenticated as {active[0].get('account', 'unknown')}",
						)
				except (json.JSONDecodeError, KeyError):
					pass

		# Not authenticated -- Telegram Danny with instructions
		if self._notifier:
			msg = (
				f"Auth needed for {service}: {purpose}\n\n"
				f"Options:\n"
				f"1. Run `{cli_info.get('auth_cmd', '')}` on your machine\n"
				f"2. Set ${cli_info.get('env_var', '')} environment variable\n"
			)
			if url:
				msg += f"3. Visit: {url}\n"
			try:
				await self._notifier.send(msg)
			except Exception:
				logger.warning("Failed to send auth notification for %s", service)

		return AuthResult(
			success=False, service=service,
			error=f"Manual auth needed for {service}",
			required_human=True,
		)

	# -- Tier 5: Headless Playwright (simple pages only) --

	async def _try_playwright(
		self, url: str, service: str, flow_type: str, timeout_s: int,
	) -> AuthResult:
		"""Last resort: headless browser for simple consent/API key pages."""
		try:
			return await asyncio.wait_for(
				self._playwright_inner(url, service, flow_type),
				timeout=timeout_s,
			)
		except asyncio.TimeoutError:
			return AuthResult(success=False, service=service, error=f"Browser timed out after {timeout_s}s")
		except Exception as exc:
			logger.warning("Playwright auth failed for %s: %s", service, exc)
			return AuthResult(success=False, service=service, error=str(exc))

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

	async def _playwright_inner(self, url: str, service: str, flow_type: str) -> AuthResult:
		"""Run Playwright-based auth flow."""
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
			elif flow_type in ("captcha", "2fa"):
				return await self._handle_stuck(page, service, f"{flow_type} detected")
			else:
				return AuthResult(success=False, service=service, error=f"Unhandled flow: {flow_type}")
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
		"""Handle OAuth consent pages (not login pages)."""
		try:
			for selector in [
				"button:has-text('Authorize')", "button:has-text('Allow')",
				"button:has-text('Accept')", "button:has-text('Grant')",
				"input[type='submit'][value*='Authorize']",
				"input[type='submit'][value*='Allow']",
			]:
				btn = page.locator(selector).first
				if await btn.is_visible():
					await btn.click()
					await page.wait_for_load_state("networkidle", timeout=10000)
					break

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
		"""Scrape API key from page."""
		try:
			for selector in [
				"input[readonly]", "code", "pre", ".api-key",
				"[data-testid='api-key']", "input[type='text'][value]",
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
		"""Scrape CLI login code from page."""
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
		"""Screenshot page, send to Telegram, return failure."""
		logger.warning("Auth stuck for %s: %s", service, reason)
		if self._notifier and hasattr(self._notifier, "send_auth_help"):
			try:
				screenshot_bytes = await page.screenshot()
				with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
					f.write(screenshot_bytes)
					screenshot_path = f.name
				await self._notifier.send_auth_help(service, reason, screenshot_path)
			except Exception:
				logger.exception("Failed to send auth help notification")
		return AuthResult(
			success=False, service=service, error=f"Stuck: {reason}", required_human=True,
		)

	# -- Tier 6: Full manual fallback --

	async def _telegram_manual(self, service: str, purpose: str, url: str) -> AuthResult:
		"""Ask Danny to do it manually via Telegram."""
		if self._notifier:
			msg = (
				f"Manual auth needed for {service}\n"
				f"Purpose: {purpose}\n"
			)
			if url:
				msg += f"URL: {url}\n"
			msg += (
				"\nPlease authenticate and store the credential:\n"
				f"security add-generic-password -U "
				f"-s \"autodev/{service}\" -a \"default\" -w"
			)
			try:
				await self._notifier.send(msg)
			except Exception:
				pass
		return AuthResult(
			success=False, service=service,
			error=f"No automated auth path for {service}. Manual auth requested via Telegram.",
			required_human=True,
		)


# Backward compat alias
HeadlessAuthHandler = AuthHandler
