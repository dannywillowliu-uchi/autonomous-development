"""LLM-powered spec generation from intel proposals.

Transforms AdaptationProposal objects into detailed implementation specs
by fetching source context, reading project files, and prompting an LLM
for a structured specification.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from pathlib import Path

from autodev.intelligence.models import AdaptationProposal
from autodev.intelligence.utils import find_claude_binary

logger = logging.getLogger(__name__)

_SOURCE_CTX_LIMIT = 2000
_PROJECT_CTX_LIMIT = 3000
_MODULE_LINE_LIMIT = 100


class SpecGenerator:
	"""Generate implementation specs from intel proposals."""

	def __init__(self, project_path: Path):
		self._project_path = project_path

	async def generate_spec(self, proposal: AdaptationProposal, source_url: str = "") -> str:
		"""Generate a detailed implementation spec from a proposal.

		Args:
			proposal: The adaptation proposal to generate a spec for.
			source_url: Optional URL to fetch source context from.

		Returns:
			Markdown spec content.
		"""
		source_ctx = await self._read_source_context(source_url)
		project_ctx = await self._read_project_context(proposal.target_modules)
		prompt = self._build_spec_prompt(proposal, source_ctx, project_ctx)

		claude_bin = find_claude_binary()
		proc = await asyncio.create_subprocess_exec(
			claude_bin, "--print", "-p", prompt,
			cwd=str(self._project_path),
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout, stderr = await proc.communicate()

		if proc.returncode != 0:
			logger.warning("claude --print failed (rc=%d): %s", proc.returncode, stderr.decode()[:500])
			raise RuntimeError(f"LLM spec generation failed: {stderr.decode()[:200]}")

		return stdout.decode().strip()

	async def _read_source_context(self, url: str) -> str:
		"""Fetch and summarize source material from a URL."""
		if not url:
			return ""

		try:
			import httpx

			async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
				resp = await client.get(url)
				resp.raise_for_status()
				text = _strip_html(resp.text)
				return text[:_SOURCE_CTX_LIMIT]
		except Exception:
			logger.warning("Failed to fetch source URL: %s", url, exc_info=True)
			return ""

	async def _read_project_context(self, target_modules: list[str]) -> str:
		"""Read relevant project files for context."""
		parts: list[str] = []
		total_len = 0

		# Read target modules
		for module in target_modules:
			if total_len >= _PROJECT_CTX_LIMIT:
				break
			path = self._project_path / "src" / "autodev" / module
			if not path.exists():
				continue
			try:
				lines = path.read_text().splitlines()[:_MODULE_LINE_LIMIT]
				content = f"## {module}\n```python\n" + "\n".join(lines) + "\n```\n"
				parts.append(content)
				total_len += len(content)
			except Exception:
				logger.debug("Could not read module %s", module)

		# Include CLAUDE.md architecture section
		claude_md = self._project_path / "CLAUDE.md"
		if claude_md.exists():
			try:
				text = claude_md.read_text()
				arch_match = re.search(r"(## Architecture.*?)(?=\n## |\Z)", text, re.DOTALL)
				if arch_match:
					arch = arch_match.group(1)
					remaining = _PROJECT_CTX_LIMIT - total_len
					if remaining > 0:
						parts.append(arch[:remaining])
			except Exception:
				logger.debug("Could not read CLAUDE.md")

		return "\n".join(parts)[:_PROJECT_CTX_LIMIT]

	def _build_spec_prompt(
		self,
		proposal: AdaptationProposal,
		source_ctx: str,
		project_ctx: str,
	) -> str:
		"""Build the LLM prompt for spec generation."""
		modules = ", ".join(proposal.target_modules) if proposal.target_modules else "TBD"

		sections = [
			"Generate a detailed implementation spec in markdown for the following proposal.",
			"",
			"# Proposal",
			f"**Title:** {proposal.title}",
			f"**Type:** {proposal.proposal_type}",
			f"**Priority:** {proposal.priority}",
			f"**Effort:** {proposal.effort_estimate}",
			f"**Risk:** {proposal.risk_level}",
			f"**Target modules:** {modules}",
			f"**Description:** {proposal.description}",
		]

		if source_ctx:
			sections.extend([
				"",
				"# Source Material",
				source_ctx,
			])

		if project_ctx:
			sections.extend([
				"",
				"# Project Architecture",
				project_ctx,
			])

		sections.extend([
			"",
			"# Required Spec Sections",
			"Your spec MUST include these sections:",
			"1. **Problem Statement** -- What this improves and why",
			"2. **Changes Needed** -- Specific files, functions, and patterns to modify or create",
			"3. **Testing Requirements** -- What tests to write, what to verify",
			"4. **Risk Assessment** -- What could go wrong, mitigation strategies",
			"",
			"Be specific about file paths (relative to project root) and function names.",
			"Reference existing patterns from the project architecture above.",
			"Output only the markdown spec, no preamble.",
		])

		return "\n".join(sections)


def _strip_html(text: str) -> str:
	"""Strip HTML tags and decode entities, keeping text content."""
	text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
	text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
	text = re.sub(r"<[^>]+>", " ", text)
	text = html.unescape(text)
	text = re.sub(r"\s+", " ", text)
	return text.strip()
