"""Tests for DAG planner: topological layers, PlanNode multi-parent, layer dispatch."""

from __future__ import annotations

from mission_control.models import PlanNode, WorkUnit
from mission_control.overlap import resolve_file_overlaps, topological_layers


def _wu(title: str = "", files_hint: str = "", priority: int = 1, depends_on: str = "") -> WorkUnit:
	return WorkUnit(
		plan_id="plan-1",
		title=title,
		files_hint=files_hint,
		priority=priority,
		depends_on=depends_on,
	)


class TestTopologicalLayers:
	def test_empty_input(self) -> None:
		assert topological_layers([]) == []

	def test_no_dependencies_single_layer(self) -> None:
		"""All units with no deps form a single parallel layer."""
		a = _wu("A", "a.py")
		b = _wu("B", "b.py")
		c = _wu("C", "c.py")
		layers = topological_layers([a, b, c])
		assert len(layers) == 1
		assert set(u.id for u in layers[0]) == {a.id, b.id, c.id}

	def test_linear_chain(self) -> None:
		"""A -> B -> C forms 3 layers of 1 unit each."""
		a = _wu("A", "a.py")
		b = _wu("B", "b.py", depends_on=a.id)
		c = _wu("C", "c.py", depends_on=b.id)
		layers = topological_layers([a, b, c])
		assert len(layers) == 3
		assert layers[0][0].id == a.id
		assert layers[1][0].id == b.id
		assert layers[2][0].id == c.id

	def test_diamond_dag(self) -> None:
		"""Diamond: A -> (B, C) -> D forms 3 layers."""
		a = _wu("A", "a.py")
		b = _wu("B", "b.py", depends_on=a.id)
		c = _wu("C", "c.py", depends_on=a.id)
		d = _wu("D", "d.py", depends_on=f"{b.id},{c.id}")
		layers = topological_layers([a, b, c, d])
		assert len(layers) == 3
		# Layer 0: A
		assert layers[0][0].id == a.id
		# Layer 1: B and C (parallel)
		layer1_ids = {u.id for u in layers[1]}
		assert layer1_ids == {b.id, c.id}
		# Layer 2: D
		assert layers[2][0].id == d.id

	def test_orphan_deps_ignored(self) -> None:
		"""Dependencies on non-existent IDs are silently ignored."""
		a = _wu("A", "a.py", depends_on="nonexistent-id")
		b = _wu("B", "b.py")
		layers = topological_layers([a, b])
		# Both should be in layer 0 (orphan dep ignored)
		assert len(layers) == 1
		assert len(layers[0]) == 2

	def test_single_unit(self) -> None:
		a = _wu("A", "a.py")
		layers = topological_layers([a])
		assert len(layers) == 1
		assert layers[0][0].id == a.id

	def test_mixed_deps_and_independent(self) -> None:
		"""Mix of dependent and independent units."""
		a = _wu("A", "a.py")
		b = _wu("B", "b.py", depends_on=a.id)
		c = _wu("C", "c.py")  # independent
		layers = topological_layers([a, b, c])
		assert len(layers) == 2
		# Layer 0: A and C
		layer0_ids = {u.id for u in layers[0]}
		assert a.id in layer0_ids
		assert c.id in layer0_ids
		# Layer 1: B
		assert layers[1][0].id == b.id


class TestPlanNodeMultiParent:
	def test_parent_ids_field_exists(self) -> None:
		"""PlanNode has parent_ids field defaulting to empty string."""
		node = PlanNode()
		assert node.parent_ids == ""

	def test_parent_ids_populated(self) -> None:
		"""parent_ids can hold comma-separated parent IDs."""
		node = PlanNode(parent_ids="parent-1,parent-2")
		assert "parent-1" in node.parent_ids
		assert "parent-2" in node.parent_ids

	def test_backward_compat_parent_id(self) -> None:
		"""parent_id still works for single parent."""
		node = PlanNode(parent_id="single-parent", parent_ids="single-parent")
		assert node.parent_id == "single-parent"
		assert node.parent_ids == "single-parent"

	def test_parent_id_none_by_default(self) -> None:
		node = PlanNode()
		assert node.parent_id is None
		assert node.parent_ids == ""


class TestOverlapWithLayers:
	def test_resolve_then_layer(self) -> None:
		"""resolve_file_overlaps + topological_layers integration."""
		# Two units share a file -- overlap resolution adds dependency
		a = _wu("A", "shared.py,a.py", priority=1)
		b = _wu("B", "shared.py,b.py", priority=2)
		c = _wu("C", "c.py", priority=3)
		resolve_file_overlaps([a, b, c])
		# B should now depend on A
		assert a.id in b.depends_on
		layers = topological_layers([a, b, c])
		assert len(layers) == 2
		# Layer 0: A and C
		layer0_ids = {u.id for u in layers[0]}
		assert a.id in layer0_ids
		assert c.id in layer0_ids
		# Layer 1: B
		assert layers[1][0].id == b.id

	def test_diamond_from_overlaps(self) -> None:
		"""File overlaps creating a diamond pattern layer correctly."""
		a = _wu("A", "base.py", priority=1)
		b = _wu("B", "base.py,feature.py", priority=2)
		c = _wu("C", "base.py,util.py", priority=2)
		d = _wu("D", "feature.py,util.py", priority=3)
		resolve_file_overlaps([a, b, c, d])
		layers = topological_layers([a, b, c, d])
		# A is first (highest priority), B and C second, D third
		assert layers[0][0].id == a.id
		# D should be in a later layer than B and C
		d_layer = None
		for idx, layer in enumerate(layers):
			if any(u.id == d.id for u in layer):
				d_layer = idx
		assert d_layer is not None and d_layer >= 2
