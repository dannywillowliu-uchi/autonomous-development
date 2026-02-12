"""Shared JSON extraction utilities for parsing LLM output."""

from __future__ import annotations

import json
import re
from typing import Any


def _find_balanced(text: str, open_char: str, close_char: str) -> str | None:
	"""Find the first balanced substring between open_char and close_char."""
	start = text.find(open_char)
	if start == -1:
		return None
	depth = 0
	in_string = False
	escape = False
	for i in range(start, len(text)):
		ch = text[i]
		if escape:
			escape = False
			continue
		if ch == "\\":
			if in_string:
				escape = True
			continue
		if ch == '"':
			in_string = not in_string
			continue
		if in_string:
			continue
		if ch == open_char:
			depth += 1
		elif ch == close_char:
			depth -= 1
			if depth == 0:
				return text[start:i + 1]
	return None


def extract_json_from_text(text: str, expect_array: bool = False) -> Any | None:
	"""Extract a JSON object or array from text that may contain markdown fences or prose.

	Tries in order:
	1. Strip markdown fences (```json ... ``` or ``` ... ```)
	2. Find a bare JSON object/array via balanced brace matching
	3. Return None if nothing parseable found

	Args:
		text: Raw text that may contain JSON.
		expect_array: If True, look for a JSON array ([...]) instead of object ({...}).

	Returns:
		Parsed JSON (dict or list) or None if extraction failed.
	"""
	if not text or not text.strip():
		return None

	# Step 1: Try to extract from markdown fences
	fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
	if fence_match:
		fenced_content = fence_match.group(1).strip()
		try:
			return json.loads(fenced_content)
		except (json.JSONDecodeError, ValueError):
			pass

	# Step 2: Strip any remaining markdown fences from the whole text
	cleaned = re.sub(r"```(?:json)?\s*", "", text)
	cleaned = re.sub(r"```", "", cleaned)
	cleaned = cleaned.strip()

	# Step 3: Try parsing the whole cleaned text
	try:
		return json.loads(cleaned)
	except (json.JSONDecodeError, ValueError):
		pass

	# Step 4: Find a bare JSON object or array via balanced brace matching
	if expect_array:
		candidate = _find_balanced(cleaned, "[", "]")
	else:
		candidate = _find_balanced(cleaned, "{", "}")

	if candidate:
		try:
			return json.loads(candidate)
		except (json.JSONDecodeError, ValueError):
			pass

	return None
