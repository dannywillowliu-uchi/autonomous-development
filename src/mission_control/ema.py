"""Exponential Moving Average budget tracking for mission-control."""

from __future__ import annotations

import math


class ExponentialMovingAverage:
	"""EMA tracker with outlier dampening and conservatism factor.

	Used to predict next work unit cost for budget gate decisions.
	"""

	def __init__(
		self,
		alpha: float = 0.3,
		outlier_multiplier: float = 3.0,
		conservatism_base: float = 0.5,
	) -> None:
		self._alpha = alpha
		self._outlier_multiplier = outlier_multiplier
		self._conservatism_base = conservatism_base
		self._ema: float | None = None
		self._count: int = 0

	@property
	def value(self) -> float | None:
		"""Current EMA value, or None if no data points have been added."""
		return self._ema

	@property
	def count(self) -> int:
		"""Number of data points ingested."""
		return self._count

	def _conservatism_factor(self) -> float:
		"""k = 1.0 + conservatism_base / sqrt(n). Decays toward 1.0 as n grows."""
		if self._count <= 0:
			return 1.0 + self._conservatism_base
		return 1.0 + self._conservatism_base / math.sqrt(self._count)

	def projected_cost(self) -> float | None:
		"""EMA * conservatism factor. Returns None if no data."""
		if self._ema is None:
			return None
		return self._ema * self._conservatism_factor()

	def update(self, value: float) -> float:
		"""Add a new data point and return the updated EMA.

		Outlier dampening: after 3+ data points, spikes > outlier_multiplier * EMA
		are clamped to 2x EMA before the EMA update.
		"""
		self._count += 1

		if self._ema is None:
			self._ema = value
			return self._ema

		# Outlier dampening: only after enough data to establish a baseline
		effective = value
		if self._count > 3 and value > self._outlier_multiplier * self._ema:
			effective = 2.0 * self._ema

		self._ema = self._alpha * effective + (1.0 - self._alpha) * self._ema
		return self._ema

	def would_exceed_budget(self, spent: float, budget: float) -> bool:
		"""Check if projected next unit cost would exceed remaining budget."""
		projected = self.projected_cost()
		if projected is None:
			return False
		return spent + projected > budget
