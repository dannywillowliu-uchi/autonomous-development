"""Human-in-the-Loop approval gates for irreversible actions.

Provides file-based and Telegram-based approval flows for push and
large merge operations. When a gate is triggered, an approval request
file is written and (optionally) a Telegram notification is sent.
Approval can come from Telegram commands or by editing the file directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from mission_control.config import HITLGateConfig, MissionConfig
from mission_control.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


@dataclass
class ApprovalRequest:
	"""A pending approval request for an HITL gate."""

	request_id: str = ""
	gate_type: str = ""  # "push" | "large_merge"
	context: dict[str, Any] = field(default_factory=dict)
	created_at: float = 0.0
	timeout_seconds: int = 300
	timeout_action: str = "approve"

	def __post_init__(self) -> None:
		if not self.request_id:
			self.request_id = uuid.uuid4().hex[:12]
		if not self.created_at:
			self.created_at = time.monotonic()


class ApprovalGate:
	"""Manages HITL approval requests via file-based and Telegram polling."""

	def __init__(self, config: MissionConfig, notifier: TelegramNotifier | None) -> None:
		self._config = config
		self._notifier = notifier
		self._approvals_dir = self._resolve_approvals_dir()
		self._last_update_id: int = 0
		self._telegram_lock = asyncio.Lock()
		self._bot_token = config.notifications.telegram.bot_token
		self._chat_id = config.notifications.telegram.chat_id

	def _resolve_approvals_dir(self) -> Path:
		base = self._config.target.resolved_path
		return base / self._config.hitl.approvals_dir

	def _gate_config(self, gate_type: str) -> HITLGateConfig:
		if gate_type == "push":
			return self._config.hitl.push_gate
		return self._config.hitl.large_merge_gate

	async def request_approval(self, req: ApprovalRequest) -> bool:
		"""Request human approval for an action.

		Returns True if approved, False if denied.
		"""
		# Write pending approval file
		self._approvals_dir.mkdir(parents=True, exist_ok=True)
		approval_file = self._approvals_dir / f"{req.request_id}.json"
		file_data = {
			"request_id": req.request_id,
			"gate_type": req.gate_type,
			"context": req.context,
			"status": "pending",
			"created_at": req.created_at,
			"decided_at": None,
		}
		approval_file.write_text(json.dumps(file_data, indent=2))
		logger.info(
			"HITL approval requested: gate=%s id=%s timeout=%ds action=%s",
			req.gate_type, req.request_id, req.timeout_seconds, req.timeout_action,
		)

		# Send Telegram notification if available
		if self._notifier:
			context_lines = [f"  {k}: {v}" for k, v in req.context.items()]
			context_str = "\n".join(context_lines)
			msg = (
				f"*HITL Approval Required* ({req.gate_type})\n"
				f"ID: `{req.request_id}`\n"
				f"Context:\n{context_str}\n\n"
				f"Reply with:\n"
				f"`/approve_{req.request_id}`\n"
				f"`/deny_{req.request_id}`\n\n"
				f"Timeout: {req.timeout_seconds}s -> {req.timeout_action}"
			)
			await self._notifier.send(msg)

		# Poll for approval
		poll_interval = self._config.hitl.telegram_poll_interval
		deadline = time.monotonic() + req.timeout_seconds

		while True:
			# Check file-based approval
			status = self._check_file_status(approval_file)
			if status == "approved":
				self._finalize_file(approval_file, "approved")
				return True
			if status == "denied":
				self._finalize_file(approval_file, "denied")
				return False

			# Check Telegram updates if notifier is available
			if self._notifier and self._bot_token:
				async with self._telegram_lock:
					telegram_status = await self._check_telegram_updates(req.request_id)
				if telegram_status == "approved":
					self._finalize_file(approval_file, "approved")
					return True
				if telegram_status == "denied":
					self._finalize_file(approval_file, "denied")
					return False

			# Check timeout
			if time.monotonic() >= deadline:
				if req.timeout_action == "approve":
					self._finalize_file(approval_file, "timeout_approved")
					logger.info("HITL timeout -> approve (id=%s)", req.request_id)
					return True
				else:
					self._finalize_file(approval_file, "timeout_denied")
					logger.info("HITL timeout -> deny (id=%s)", req.request_id)
					return False

			await asyncio.sleep(poll_interval)

	def _check_file_status(self, path: Path) -> str:
		"""Read the approval file and return its status."""
		try:
			data = json.loads(path.read_text())
			return str(data.get("status", "pending"))
		except (json.JSONDecodeError, OSError):
			return "pending"

	def _finalize_file(self, path: Path, status: str) -> None:
		"""Update the approval file with final status and timestamp."""
		try:
			data = json.loads(path.read_text())
			data["status"] = status
			data["decided_at"] = time.monotonic()
			path.write_text(json.dumps(data, indent=2))
		except (json.JSONDecodeError, OSError) as exc:
			logger.warning("Failed to finalize approval file: %s", exc)

	async def _check_telegram_updates(self, request_id: str) -> str:
		"""Poll Telegram getUpdates for /approve_<id> or /deny_<id> commands.

		Returns "approved", "denied", or "" if no matching command found.
		"""
		url = f"https://api.telegram.org/bot{self._bot_token}/getUpdates"
		params: dict[str, Any] = {"timeout": 0}
		if self._last_update_id:
			params["offset"] = self._last_update_id + 1

		try:
			async with httpx.AsyncClient(timeout=10.0) as client:
				resp = await client.get(url, params=params)
				if resp.status_code != 200:
					return ""
				data = resp.json()
		except (httpx.HTTPError, OSError, ValueError):
			return ""

		results = data.get("result", [])
		for update in results:
			update_id = update.get("update_id", 0)
			if update_id > self._last_update_id:
				self._last_update_id = update_id

			message = update.get("message", {})
			text = message.get("text", "").strip()
			chat_id = str(message.get("chat", {}).get("id", ""))

			# Only accept commands from the configured chat
			if chat_id != self._chat_id:
				continue

			if text == f"/approve_{request_id}":
				return "approved"
			if text == f"/deny_{request_id}":
				return "denied"

		return ""

	def cleanup_old_approvals(self, max_age_seconds: float = 86400.0) -> int:
		"""Remove approval files older than max_age_seconds. Returns count removed."""
		if not self._approvals_dir.exists():
			return 0
		removed = 0
		now = time.time()
		for f in self._approvals_dir.glob("*.json"):
			try:
				if now - f.stat().st_mtime > max_age_seconds:
					f.unlink()
					removed += 1
			except OSError:
				pass
		return removed
