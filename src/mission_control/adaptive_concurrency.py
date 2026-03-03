"""Adaptive concurrency controller for worker pool sizing.

Tracks merge outcomes over a sliding window and recommends worker
capacity adjustments based on rolling success and conflict rates.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MergeOutcome(Enum):
	SUCCESS = "success"
	CONFLICT = "conflict"
	FAIL = "fail"


@dataclass
class AdaptiveConcurrencyController:
	"""Recommends worker pool size based on recent merge outcomes.

	Maintains a fixed-size sliding window of merge outcomes and uses
	rolling success/conflict rates to recommend scaling decisions.
	Uses exponential backoff on conflict rate spikes.
	"""

	max_workers: int = 8
	min_workers: int = 1
	window_size: int = 20
	current_workers: int = 4
	_outcomes: deque[MergeOutcome] = field(default_factory=deque)
	_consecutive_scale_downs: int = 0

	def __post_init__(self) -> None:
		self.current_workers = max(self.min_workers, min(self.current_workers, self.max_workers))

	def record_outcome(self, outcome: MergeOutcome) -> None:
		"""Record a merge outcome into the sliding window."""
		self._outcomes.append(outcome)
		while len(self._outcomes) > self.window_size:
			self._outcomes.popleft()

	def rolling_success_rate(self) -> float:
		"""Compute the success rate over the current window."""
		if not self._outcomes:
			return 0.0
		return sum(1 for o in self._outcomes if o == MergeOutcome.SUCCESS) / len(self._outcomes)

	def rolling_conflict_rate(self) -> float:
		"""Compute the conflict rate over the current window."""
		if not self._outcomes:
			return 0.0
		return sum(1 for o in self._outcomes if o == MergeOutcome.CONFLICT) / len(self._outcomes)

	def recommend_capacity(self, success_rate: float, conflict_rate: float) -> int:
		"""Recommend optimal worker count based on current rates.

		Scaling rules:
		- Scale up: success_rate > 0.8 AND conflict_rate < 0.1
		- Scale down: conflict_rate > 0.3 OR success_rate < 0.5
		- Hold steady: otherwise

		Uses exponential backoff on consecutive scale-downs to avoid
		oscillation during conflict spikes.
		"""
		if conflict_rate > 0.3 or success_rate < 0.5:
			self._consecutive_scale_downs += 1
			backoff = min(self._consecutive_scale_downs, self.current_workers - self.min_workers)
			new = max(self.min_workers, self.current_workers - backoff)
			logger.info(
				"Scaling down: %d -> %d (success=%.2f, conflict=%.2f, backoff=%d)",
				self.current_workers, new, success_rate, conflict_rate, backoff,
			)
			self.current_workers = new
		elif success_rate > 0.8 and conflict_rate < 0.1:
			self._consecutive_scale_downs = 0
			new = min(self.current_workers + 1, self.max_workers)
			logger.info(
				"Scaling up: %d -> %d (success=%.2f, conflict=%.2f)",
				self.current_workers, new, success_rate, conflict_rate,
			)
			self.current_workers = new
		else:
			self._consecutive_scale_downs = 0
			logger.debug(
				"Holding steady at %d (success=%.2f, conflict=%.2f)",
				self.current_workers, success_rate, conflict_rate,
			)

		return self.current_workers

	def step(self) -> int:
		"""Convenience: compute rates from window and recommend capacity."""
		return self.recommend_capacity(self.rolling_success_rate(), self.rolling_conflict_rate())

	def reset(self) -> None:
		"""Clear all state and reset to defaults."""
		self._outcomes.clear()
		self._consecutive_scale_downs = 0
