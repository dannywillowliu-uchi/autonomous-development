"""Data models for the intelligence subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from autodev.models import _new_id, _now_iso


@dataclass
class IntelSource:
	"""A source of external intelligence (HN, GitHub, arXiv, etc.)."""

	name: str
	url: str
	source_type: str  # "hackernews", "github", "arxiv"


@dataclass
class Finding:
	"""A single finding from an intelligence source."""

	id: str = field(default_factory=_new_id)
	source: str = ""
	title: str = ""
	url: str = ""
	summary: str = ""
	published_at: str = field(default_factory=_now_iso)
	raw_data: dict[str, Any] = field(default_factory=dict)
	relevance_score: float = 0.0
	trust_level: str = "medium"


@dataclass
class AdaptationProposal:
	"""A proposed adaptation based on an intelligence finding."""

	id: str = field(default_factory=_new_id)
	finding_id: str = ""
	title: str = ""
	description: str = ""
	proposal_type: Literal["integration", "pattern", "architecture"] = "integration"
	target_modules: list[str] = field(default_factory=list)
	priority: int = 3  # 1 (highest) to 5 (lowest)
	effort_estimate: str = "medium"  # "small", "medium", "large"
	risk_level: str = "low"  # "low", "high"
