"""Session spawning -- invoke Claude Code as a subprocess."""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from mission_control.json_utils import extract_json_from_text
from mission_control.models import MCResultSchema

logger = logging.getLogger(__name__)

def parse_mc_result(output: str) -> dict[str, object] | None:
	"""Extract MC_RESULT JSON from session output.

	Handles both single-line and multiline JSON after the MC_RESULT: marker.
	"""
	# Find the last MC_RESULT: marker in the output
	marker = "MC_RESULT:"
	idx = output.rfind(marker)
	if idx == -1:
		return None

	# Extract everything after the marker
	remainder = output[idx + len(marker):]

	# Try balanced brace extraction (handles multiline JSON)
	result = extract_json_from_text(remainder)
	if isinstance(result, dict):
		return validate_mc_result(result)

	# Fallback: single-line regex for simple cases
	match = re.search(r"\{.*\}", remainder.split("\n")[0])
	if match:
		try:
			raw = json.loads(match.group(0))
			if isinstance(raw, dict):
				return validate_mc_result(raw)
		except json.JSONDecodeError:
			pass

	return None


_MC_RESULT_DEFAULTS: dict[str, object] = {
	"status": "failed",
	"commits": [],
	"summary": "",
	"files_changed": [],
	"discoveries": [],
	"concerns": [],
}


_STATUS_ALIASES: dict[str, str] = {
	"success": "completed",
	"failure": "failed",
	"error": "failed",
}


def validate_mc_result(raw: dict[str, object]) -> dict[str, object]:
	"""Validate an MC_RESULT dict against MCResultSchema.

	On success, returns the validated dict. On ValidationError, extracts
	whatever valid fields exist and returns a degraded dict with defaults
	for missing/invalid fields. Logs a warning on degraded parse.
	"""
	# Normalize common status aliases before validation
	if raw.get("status") in _STATUS_ALIASES:
		raw = {**raw, "status": _STATUS_ALIASES[raw["status"]]}  # type: ignore[index]
	# Normalize common field name aliases
	if "files_modified" in raw and "files_changed" not in raw:
		raw = {**raw, "files_changed": raw["files_modified"]}
	try:
		validated = MCResultSchema.model_validate(raw)
		return validated.model_dump()
	except ValidationError as exc:
		logger.warning("MC_RESULT schema validation failed, extracting valid fields: %s", exc)
		degraded: dict[str, object] = {}
		for field_name in MCResultSchema.model_fields:
			value = raw.get(field_name)
			if value is not None:
				try:
					partial = MCResultSchema.model_validate({
						**_MC_RESULT_DEFAULTS,
						field_name: value,
					})
					degraded[field_name] = getattr(partial, field_name)
				except ValidationError:
					degraded[field_name] = _MC_RESULT_DEFAULTS[field_name]
			else:
				degraded[field_name] = _MC_RESULT_DEFAULTS[field_name]
		return degraded


def build_branch_name(session_id: str) -> str:
	"""Generate a git branch name for a session."""
	return f"mc/session-{session_id}"

