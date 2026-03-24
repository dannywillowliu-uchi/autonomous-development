"""Session spawning -- invoke Claude Code as a subprocess."""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from autodev.json_utils import extract_json_from_text
from autodev.models import MCResultSchema

logger = logging.getLogger(__name__)


def extract_text_from_stream_json(output: str) -> str:
	"""Extract plain text from stream-json (NDJSON) formatted output.

	When Claude Code runs with --output-format stream-json, stdout contains
	one JSON event per line. AD_RESULT markers are embedded inside JSON string
	fields where quotes are escaped as \\", making brace-counting parsers fail.

	This function extracts the text content from:
	1. "result" events (final result text -- preferred)
	2. "assistant" events (message text blocks -- fallback)

	Returns concatenated plain text suitable for AD_RESULT parsing.
	"""
	result_text = ""
	assistant_texts: list[str] = []

	for line in output.splitlines():
		line = line.strip()
		# Strip trace file prefixes like "[OUT] "
		if line.startswith("[OUT] "):
			line = line[6:]
		if not line or not line.startswith("{"):
			continue
		try:
			event = json.loads(line)
		except (json.JSONDecodeError, ValueError):
			continue
		if not isinstance(event, dict):
			continue

		event_type = event.get("type")

		# "result" event has the final output text -- this is the primary source
		if event_type == "result":
			text = event.get("result", "")
			if text:
				result_text = text

		# "assistant" events contain message text blocks
		elif event_type == "assistant":
			content = event.get("message", {}).get("content", [])
			if isinstance(content, list):
				for block in content:
					if isinstance(block, dict) and block.get("type") == "text":
						text = block.get("text", "")
						if text:
							assistant_texts.append(text)

	# Prefer "result" event text (clean final output)
	if result_text:
		return result_text
	# Fall back to concatenated assistant text blocks
	if assistant_texts:
		return "\n".join(assistant_texts)
	return ""


def parse_mc_result(output: str) -> dict[str, object] | None:
	"""Extract AD_RESULT JSON from session output.

	Handles plain text, multiline JSON, and stream-json (NDJSON) output.
	When output is stream-json, AD_RESULT markers are inside JSON-encoded
	text fields where quotes are escaped -- direct brace-counting fails.
	We first try the raw output, then fall back to extracting text from
	stream-json events.
	"""
	result = _parse_ad_result_from_text(output)
	if result:
		return result

	# Fallback: output may be stream-json NDJSON where AD_RESULT is
	# embedded inside JSON string fields with escaped quotes.
	# Extract plain text from the JSON events and retry.
	extracted = extract_text_from_stream_json(output)
	if extracted:
		result = _parse_ad_result_from_text(extracted)
		if result:
			return result

	return None


def _parse_ad_result_from_text(output: str) -> dict[str, object] | None:
	"""Extract AD_RESULT JSON from plain text output."""
	marker = "AD_RESULT:"
	idx = output.rfind(marker)
	if idx == -1:
		return None

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


_AD_RESULT_DEFAULTS: dict[str, object] = {
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
	"""Validate an AD_RESULT dict against MCResultSchema.

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
		logger.warning("AD_RESULT schema validation failed, extracting valid fields: %s", exc)
		degraded: dict[str, object] = {}
		for field_name in MCResultSchema.model_fields:
			value = raw.get(field_name)
			if value is not None:
				try:
					partial = MCResultSchema.model_validate({
						**_AD_RESULT_DEFAULTS,
						field_name: value,
					})
					degraded[field_name] = getattr(partial, field_name)
				except ValidationError:
					degraded[field_name] = _AD_RESULT_DEFAULTS[field_name]
			else:
				degraded[field_name] = _AD_RESULT_DEFAULTS[field_name]
		return degraded


def extract_fallback_handoff(
	output: str,
	exit_code: int | None = None,
) -> dict[str, object]:
	"""Recover structured handoff data when AD_RESULT is missing.

	Parses raw worker output for git commit hashes, changed files from
	diff --stat patterns, and classifies status from exit code and error
	signatures. Returns a partial Handoff-compatible dict.
	"""
	commits = _extract_commits(output)
	files_changed = _extract_changed_files(output)
	status = _classify_status(output, exit_code)
	summary = _build_fallback_summary(output, status)

	return {
		"status": status,
		"commits": commits,
		"summary": summary,
		"files_changed": files_changed,
		"discoveries": [],
		"concerns": ["AD_RESULT missing; handoff recovered from raw output"],
	}


def _extract_commits(output: str) -> list[str]:
	"""Extract commit hashes from raw output.

	Looks for:
	- git log lines: 'commit <hash>'
	- git commit output: '[branch <short_hash>] message'
	- Short hashes after 'Created commit' or similar markers
	"""
	seen: set[str] = set()
	commits: list[str] = []

	# git log output: "commit abc123def456..."
	for m in re.finditer(r"^commit\s+([0-9a-f]{7,40})\b", output, re.MULTILINE):
		h = m.group(1)
		if h not in seen:
			seen.add(h)
			commits.append(h)

	# git commit output: "[branch_name abc1234] message"
	for m in re.finditer(r"\[[\w/.\-]+\s+([0-9a-f]{7,12})\]", output):
		h = m.group(1)
		if h not in seen:
			seen.add(h)
			commits.append(h)

	# "Created commit abc1234" or "committed abc1234"
	for m in re.finditer(r"(?:created commit|committed)\s+([0-9a-f]{7,40})\b", output, re.IGNORECASE):
		h = m.group(1)
		if h not in seen:
			seen.add(h)
			commits.append(h)

	return commits


def _extract_changed_files(output: str) -> list[str]:
	"""Extract changed file paths from git diff --stat output.

	Matches lines like:
	  src/foo.py | 10 +++---
	  src/bar.py | 2 ++
	"""
	files: list[str] = []
	seen: set[str] = set()

	for m in re.finditer(r"^\s*([\w][\w/.\-]+(?:\.\w+)+)\s+\|\s+\d+", output, re.MULTILINE):
		path = m.group(1)
		if path not in seen:
			seen.add(path)
			files.append(path)

	return files


_FAILURE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
	("merge_conflict", re.compile(r"<{7}|>{7}|={7}|CONFLICT", re.MULTILINE)),
	("syntax_error", re.compile(r"SyntaxError|invalid syntax")),
	("import_error", re.compile(r"ModuleNotFoundError|ImportError")),
	("test_failure", re.compile(r"\bFAILED\b|AssertionError")),
	("lint_error", re.compile(r"\b[EFW]\d{3,4}\b.*(?:ruff|Found \d+ error)", re.IGNORECASE)),
	("timeout", re.compile(r"timed?\s*out|timeout|deadline exceeded|killed.*signal", re.IGNORECASE)),
]


def _classify_status(output: str, exit_code: int | None) -> str:
	"""Classify session status from exit code and error signatures."""
	# Exit code 0 with no obvious errors → completed
	if exit_code == 0:
		return "completed"

	# No output at all → failed
	if not output:
		return "failed"

	text = output[:5000]

	# Check for known failure patterns
	for _name, pattern in _FAILURE_PATTERNS:
		if pattern.search(text):
			return "failed"

	# Non-zero exit code → failed
	if exit_code is not None and exit_code != 0:
		return "failed"

	# No exit code, no obvious errors → assume completed
	return "completed"


def _build_fallback_summary(output: str, status: str) -> str:
	"""Build a brief summary from raw output."""
	if not output:
		return "No output captured"

	# Try to find the last meaningful non-empty line
	lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
	if not lines:
		return "No output captured"

	last_line = lines[-1][:200]
	prefix = "Fallback" if status == "completed" else "Failed"
	return f"[{prefix}] {last_line}"


def build_branch_name(session_id: str) -> str:
	"""Generate a git branch name for a session."""
	return f"mc/session-{session_id}"

