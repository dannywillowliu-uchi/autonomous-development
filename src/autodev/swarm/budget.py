"""Budget enforcement for swarm cost-cap control.

Provides BudgetEnforcer which tracks cumulative spend, cost rate,
and warning thresholds. Designed to be called each cycle from the
swarm controller main loop.
"""

from __future__ import annotations

import logging
import time

from autodev.config import BudgetConfig

logger = logging.getLogger(__name__)


class BudgetEnforcer:
	"""Enforces cost-cap policies for a swarm run.

	Tracks cumulative cost and cost rate across cycles, providing
	three enforcement mechanisms:
	  1. Hard ceiling: stop spawning when cumulative cost >= max_per_run_usd
	  2. Cost rate: reduce max_agents when $/min exceeds max_cost_rate_usd_per_min
	  3. Warning threshold: log once when spend crosses warn_threshold_pct of ceiling
	"""

	def __init__(self, budget: BudgetConfig | None = None) -> None:
		self._budget = budget
		self._warn_fired = False
		self._last_cost: float = 0.0
		self._last_time: float = time.monotonic()
		self._budget_exhausted = False

	@property
	def budget(self) -> BudgetConfig | None:
		return self._budget

	@property
	def is_exhausted(self) -> bool:
		return self._budget_exhausted

	@property
	def warn_fired(self) -> bool:
		return self._warn_fired

	def check_cycle(
		self,
		cumulative_cost_usd: float,
		current_max_agents: int,
	) -> BudgetCycleResult:
		"""Run all budget checks for one planner cycle.

		Args:
			cumulative_cost_usd: Total spend so far this run.
			current_max_agents: Current max_agents setting.

		Returns:
			BudgetCycleResult with enforcement actions to take.
		"""
		if self._budget is None:
			return BudgetCycleResult()

		result = BudgetCycleResult()
		now = time.monotonic()

		# 1. Hard ceiling check
		if self._budget.max_per_run_usd > 0 and cumulative_cost_usd >= self._budget.max_per_run_usd:
			if not self._budget_exhausted:
				logger.warning(
					"Budget exhausted: $%.2f spent (limit $%.2f). Stopping new spawns.",
					cumulative_cost_usd,
					self._budget.max_per_run_usd,
				)
				self._budget_exhausted = True
			result.stop_spawning = True

		# 2. Warning threshold (fires once)
		if (
			not self._warn_fired
			and self._budget.max_per_run_usd > 0
			and self._budget.warn_threshold_pct > 0
		):
			threshold = self._budget.warn_threshold_pct * self._budget.max_per_run_usd
			if cumulative_cost_usd >= threshold:
				logger.warning(
					"Budget warning: $%.2f spent (%.0f%% of $%.2f limit)",
					cumulative_cost_usd,
					(cumulative_cost_usd / self._budget.max_per_run_usd) * 100,
					self._budget.max_per_run_usd,
				)
				self._warn_fired = True
				result.warning_fired = True

		# 3. Cost rate check
		if self._budget.max_cost_rate_usd_per_min > 0:
			elapsed_minutes = (now - self._last_time) / 60.0
			if elapsed_minutes > 0:
				cost_delta = cumulative_cost_usd - self._last_cost
				rate = cost_delta / elapsed_minutes
				if rate > self._budget.max_cost_rate_usd_per_min and current_max_agents > 1:
					new_max = max(current_max_agents - 1, 1)
					logger.warning(
						"Cost rate $%.2f/min exceeds limit $%.2f/min. Reducing max_agents %d -> %d",
						rate,
						self._budget.max_cost_rate_usd_per_min,
						current_max_agents,
						new_max,
					)
					result.new_max_agents = new_max

		# Update tracking state for next cycle
		self._last_cost = cumulative_cost_usd
		self._last_time = now

		return result


class BudgetCycleResult:
	"""Result of a budget enforcement cycle check."""

	def __init__(self) -> None:
		self.stop_spawning: bool = False
		self.warning_fired: bool = False
		self.new_max_agents: int | None = None  # None = no change
