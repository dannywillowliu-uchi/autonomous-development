"""Causal outcome attribution -- record (decision, outcome) pairs and compute risk factors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from mission_control.models import WorkUnit, _new_id, _now_iso

logger = logging.getLogger(__name__)


@dataclass
class CausalSignal:
	"""A single (decision, outcome) observation for a completed work unit."""

	id: str = field(default_factory=_new_id)
	work_unit_id: str = ""
	mission_id: str = ""
	epoch_id: str = ""
	timestamp: str = field(default_factory=_now_iso)
	specialist: str = ""
	model: str = ""
	file_count: int = 0
	has_dependencies: bool = False
	attempt: int = 0
	unit_type: str = "implementation"
	epoch_size: int = 0
	concurrent_units: int = 0
	has_overlap: bool = False
	outcome: str = ""  # "merged" or "failed"
	failure_stage: str = ""  # "execution", "merge", "", etc.


class CausalAttributor:
	"""Computes conditional failure probabilities from causal signals."""

	def __init__(self, db: object) -> None:
		from mission_control.db import Database
		self._db: Database = db  # type: ignore[assignment]

	def record(self, signal: CausalSignal) -> None:
		"""Persist a causal signal to the database."""
		try:
			self._db.insert_causal_signal(signal)
		except Exception as exc:
			logger.warning("Failed to record causal signal: %s", exc)

	def p_failure(
		self,
		decision_type: str,
		decision_value: str,
		min_samples: int = 5,
	) -> float | None:
		"""Compute P(failure | decision_type=decision_value).

		Returns None if fewer than min_samples observations exist.
		"""
		try:
			if decision_type == "file_count":
				counts = self._db.count_causal_outcomes_bucketed(decision_value)
			else:
				counts = self._db.count_causal_outcomes(decision_type, decision_value)
		except Exception as exc:
			logger.warning("Failed to count causal outcomes: %s", exc)
			return None

		total = counts.get("merged", 0) + counts.get("failed", 0)
		if total < min_samples:
			return None
		return counts.get("failed", 0) / total

	def top_risk_factors(
		self,
		unit: WorkUnit,
		model: str = "",
		epoch_size: int = 0,
		concurrent_units: int = 0,
		limit: int = 3,
	) -> list[tuple[str, float]]:
		"""Evaluate risk dimensions for a unit and return top factors sorted by p_failure desc."""
		dimensions: list[tuple[str, str]] = []

		if unit.specialist:
			dimensions.append(("specialist", unit.specialist))
		if model:
			dimensions.append(("model", model))

		file_bucket = self._bucket_file_count(unit)
		dimensions.append(("file_count", file_bucket))

		has_deps = "true" if unit.depends_on else "false"
		dimensions.append(("has_dependencies", has_deps))

		if unit.unit_type:
			dimensions.append(("unit_type", unit.unit_type))

		risks: list[tuple[str, float]] = []
		for dim_type, dim_value in dimensions:
			p = self.p_failure(dim_type, dim_value)
			if p is not None:
				risks.append((f"{dim_type}={dim_value}", p))

		risks.sort(key=lambda x: x[1], reverse=True)
		return risks[:limit]

	@staticmethod
	def _bucket_file_count(unit: WorkUnit) -> str:
		"""Bucket a unit's file count into a category string."""
		count = len(unit.files_hint.split(",")) if unit.files_hint else 0
		if count <= 1:
			return "1"
		if count <= 3:
			return "2-3"
		if count <= 5:
			return "4-5"
		return "6+"

	@staticmethod
	def format_risk_section(risks: list[tuple[str, float]]) -> str:
		"""Format risk factors as a markdown section for prompt injection."""
		if not risks:
			return ""
		lines = ["## Causal Risk Factors"]
		for label, p in risks:
			lines.append(f"- {label}: {p:.0%} historical failure rate")
		return "\n".join(lines)
