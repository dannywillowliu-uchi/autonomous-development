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


_MITIGATION_ADVICE: dict[str, str] = {
	"file_count=6+": "Units with 6+ files have high failure rates - keep changes focused",
	"file_count=4-5": "Units touching 4-5 files are moderately risky - consider splitting",
	"has_dependencies=true": "Dependent units fail more often - verify prerequisites completed",
	"unit_type=research": "Research units are exploratory - set clear scope boundaries",
}


def _mitigation_for(label: str, p_failure: float) -> str:
	"""Return mitigation advice for a risk factor, with fallback to generic advice."""
	if label in _MITIGATION_ADVICE:
		return _MITIGATION_ADVICE[label]
	dim, _, value = label.partition("=")
	return f"{dim}={value} has {p_failure:.0%} failure rate - review historical failures before dispatch"


def format_dispatch_context(
	attributor: CausalAttributor,
	unit: WorkUnit,
	model: str = "",
	epoch_size: int = 0,
) -> str:
	"""Build a concise risk warning for worker dispatch.

	Calls attributor.top_risk_factors() and returns a warning section
	for any risk factor with p_failure > 0.3. Returns empty string if
	no significant risks are found.
	"""
	risks = attributor.top_risk_factors(unit, model=model, epoch_size=epoch_size)
	significant = [(label, p) for label, p in risks if p > 0.3]
	if not significant:
		return ""
	lines = ["## Dispatch Risk Warning"]
	for label, p in significant:
		lines.append(f"- {_mitigation_for(label, p)} ({p:.0%} failure rate)")
	return "\n".join(lines)


def get_mission_success_summary(db: object, mission_id: str) -> dict:
	"""Query causal signals for a mission and return outcome summary.

	Returns a dict with keys: merged, failed, total, success_rate, top_failure_reasons.
	"""
	from mission_control.db import Database
	_db: Database = db  # type: ignore[assignment]

	signals = _db.get_causal_signals_for_mission(mission_id, limit=10000)
	merged = 0
	failed = 0
	failure_reasons: dict[str, int] = {}
	for s in signals:
		if s.outcome == "merged":
			merged += 1
		elif s.outcome == "failed":
			failed += 1
			stage = s.failure_stage or "unknown"
			failure_reasons[stage] = failure_reasons.get(stage, 0) + 1

	total = merged + failed
	success_rate = (merged / total) if total > 0 else 0.0
	top_reasons = sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)[:3]

	return {
		"merged": merged,
		"failed": failed,
		"total": total,
		"success_rate": success_rate,
		"top_failure_reasons": top_reasons,
	}
