"""Feedback utilities -- worker context from past experiences."""

from __future__ import annotations

import json
import logging
import re

from autodev.db import Database
from autodev.models import WorkUnit

logger = logging.getLogger(__name__)


def _extract_keywords(text: str) -> list[str]:
	"""Extract meaningful keywords from text for experience search.

	Preserves full file paths (e.g. 'src/foo/bar.py') as whole keywords
	so that LIKE %keyword% queries can match on paths stored in DB columns.
	"""
	stop_words = {
		"the", "and", "for", "that", "this", "with", "from", "are", "was",
		"will", "have", "has", "been", "not", "but", "can", "all", "its",
		"add", "fix", "update", "implement", "create", "make", "use",
	}
	# Extract file paths first (word chars, slashes, dots, hyphens with a file extension)
	paths = re.findall(r"[\w./\-]+/[\w./\-]+\.\w+", text)

	seen: set[str] = set()
	result: list[str] = []
	for p in paths:
		key = p.lower()
		if key not in seen:
			seen.add(key)
			result.append(p)

	# Token-split for remaining keywords
	words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
	for w in words:
		if len(w) > 2 and w not in stop_words and w not in seen:
			seen.add(w)
			result.append(w)

	return result


def diagnose_failure(output: str) -> str:
	"""Pattern-match common failure types and return specific retry guidance.

	Analyzes worker output to identify the failure category and returns
	targeted advice instead of generic 'avoid the same mistake' text.
	"""
	if not output:
		return "No output captured. Check that the task produced observable results."

	text = output[:5000]  # Cap scan length

	# 1. Merge conflicts (check first -- blocks everything else)
	if re.search(r"<{7}|>{7}|={7}", text) or "CONFLICT" in text:
		return (
			"[Merge conflict] The branch has unresolved merge conflicts. "
			"Do NOT manually edit conflict markers. Instead: "
			"(1) run `git rebase --abort` to reset, "
			"(2) re-read the target files fresh, "
			"(3) apply your changes cleanly on top of the current state."
		)

	# 2. Syntax errors
	if "SyntaxError" in text or "invalid syntax" in text:
		lines = [ln.strip() for ln in text.splitlines() if "SyntaxError" in ln or "invalid syntax" in ln]
		sample = lines[0][:200] if lines else ""
		return (
			f"[Syntax error] The code has a syntax error. {sample} "
			"Read the failing file completely before editing. "
			"Check for mismatched parentheses, brackets, quotes, or indentation issues."
		)

	# 3. Import errors
	if "ModuleNotFoundError" in text or "ImportError" in text:
		# Try to extract the module name
		mod_match = re.search(r"(?:ModuleNotFoundError|ImportError).*?['\"]([^'\"]+)['\"]", text)
		module = mod_match.group(1) if mod_match else "unknown"
		return (
			f"[Import error] Cannot import '{module}'. "
			"Check that: (1) the module path matches the project structure, "
			"(2) you are using the correct package name (not a file path), "
			"(3) the module exists and is spelled correctly. "
			"Do NOT run pip install -- the environment is pre-configured."
		)

	# 4. Pytest assertion failures
	has_test_failure = (
		re.search(r"\bFAILED\b", text)
		or "AssertionError" in text
		or re.search(r"\bassert\b.*(?:==|!=|not|in|is)", text)
	)
	if has_test_failure:
		# Extract failed test names from summary lines (FAILED at start of line)
		failed = re.findall(r"^FAILED\s+([\w/.:]+)", text, re.MULTILINE)
		failed_str = ", ".join(failed[:3]) if failed else "unknown tests"
		return (
			f"[Test failure] Tests failed: {failed_str}. "
			"Read the failing test to understand the expected behavior. "
			"Check your implementation against the assertions -- "
			"fix the code to match test expectations, do not modify tests "
			"unless the task explicitly requires it."
		)

	# 5. Ruff / lint errors
	if re.search(r"\b[EFW]\d{3,4}\b", text) and ("ruff" in text.lower() or re.search(r"Found \d+ error", text)):
		codes = re.findall(r"\b([EFW]\d{3,4})\b", text)
		unique_codes = list(dict.fromkeys(codes))[:5]
		return (
			f"[Lint error] Ruff reported violations: {', '.join(unique_codes)}. "
			"Run `ruff check src/ tests/` to see all issues, then fix them. "
			"Common fixes: remove unused imports (F401), fix line length (E501), "
			"add missing whitespace (E225)."
		)

	# 6. Timeout / scope too large
	if re.search(r"timed?\s*out|timeout|deadline exceeded|killed.*signal", text, re.IGNORECASE):
		return (
			"[Timeout] The task exceeded its time limit. "
			"Reduce scope: (1) focus on fewer files, "
			"(2) break the task into smaller sub-changes, "
			"(3) avoid running expensive operations in sequence. "
			"If the timeout is during tests, check for infinite loops or missing mocks."
		)

	# 7. Generic fallback
	# Try to extract the last meaningful error line
	error_lines = [
		ln.strip() for ln in text.splitlines()
		if re.search(r"error|exception|traceback|fatal", ln, re.IGNORECASE)
	]
	hint = f" Last error: {error_lines[-1][:200]}" if error_lines else ""
	return (
		f"[Unknown failure]{hint} "
		"Review the full output above carefully. "
		"Identify the root cause before making changes. "
		"Avoid repeating the same approach that failed."
	)


def get_worker_context(
	db: Database,
	unit: WorkUnit,
) -> str:
	"""Find relevant past experiences for a work unit."""
	keywords = _extract_keywords(f"{unit.title} {unit.description} {unit.files_hint}")
	if not keywords:
		return ""

	search_keywords = keywords[:8]
	experiences = db.search_experiences(search_keywords, limit=3)
	if not experiences:
		return ""

	lines: list[str] = []
	for exp in experiences:
		if exp.status == "completed":
			lines.append(f"- [{exp.title}] succeeded (reward={exp.reward:.2f}): {exp.summary}")
			if exp.discoveries:
				try:
					discoveries = json.loads(exp.discoveries)
					if discoveries:
						lines.append(f"  Insights: {', '.join(discoveries[:2])}")
				except (json.JSONDecodeError, TypeError):
					pass
			if exp.concerns:
				try:
					concerns = json.loads(exp.concerns)
					if concerns:
						lines.append(f"  Pitfalls: {', '.join(concerns[:2])}")
				except (json.JSONDecodeError, TypeError):
					pass
		else:
			lines.append(f"- [{exp.title}] FAILED: {exp.summary}")

	return "\n".join(lines)
