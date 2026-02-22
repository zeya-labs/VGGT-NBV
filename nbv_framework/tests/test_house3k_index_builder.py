from __future__ import annotations

from nbv_framework.infrastructure.datasets.house3k_index_builder import apply_max_mesh_limit


def test_apply_max_mesh_limit_returns_copy_when_disabled() -> None:
    objects = [{"id": i} for i in range(5)]
    result = apply_max_mesh_limit(objects, max_meshes=None, seed=42)
    assert result == objects
    assert result is not objects


def test_apply_max_mesh_limit_is_deterministic_for_same_seed() -> None:
    objects = [{"id": i} for i in range(100)]
    limited_a = apply_max_mesh_limit(objects, max_meshes=10, seed=123)
    limited_b = apply_max_mesh_limit(objects, max_meshes=10, seed=123)
    assert limited_a == limited_b


def test_apply_max_mesh_limit_changes_with_different_seed() -> None:
    objects = [{"id": i} for i in range(100)]
    limited_a = apply_max_mesh_limit(objects, max_meshes=10, seed=123)
    limited_b = apply_max_mesh_limit(objects, max_meshes=10, seed=456)
    assert limited_a != limited_b
