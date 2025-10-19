"""Utilities for logging point clouds as TensorBoard meshes."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.structures import Pointclouds

ColorSpec = Sequence[int]
PointCloudSpec = Tuple[Pointclouds, ColorSpec]


def _downsample_points(
    points: torch.Tensor, max_points: Optional[int]
) -> torch.Tensor:
    """Uniformly downsample points to the requested count."""
    if max_points is None or points.shape[0] <= max_points:
        return points
    indices = torch.linspace(
        0, points.shape[0] - 1, steps=max_points, device=points.device
    )
    indices = indices.round().long()
    return points.index_select(0, indices)


def build_mesh_from_point_clouds(
    point_cloud_specs: Sequence[PointCloudSpec],
    *,
    batch_index: int = 0,
    max_points_per_cloud: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Convert a batch of point clouds into stacked vertex/color arrays."""
    vertices_list: List[np.ndarray] = []
    colors_list: List[np.ndarray] = []

    for point_cloud, color in point_cloud_specs:
        points_list = point_cloud.points_list()
        if batch_index >= len(points_list):
            continue

        points = points_list[batch_index]
        if points.numel() == 0:
            continue

        sampled = _downsample_points(points, max_points_per_cloud)
        points_np = sampled.detach().cpu().numpy()
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            continue

        color_np = np.asarray(color, dtype=np.uint8).reshape(-1)
        if color_np.size != 3:
            continue
        color_np = color_np.reshape(1, 3)
        tiled_colors = np.repeat(color_np, points_np.shape[0], axis=0)

        vertices_list.append(points_np)
        colors_list.append(tiled_colors)

    if not vertices_list:
        return None

    combined_vertices = np.vstack(vertices_list)
    combined_colors = np.vstack(colors_list)
    return combined_vertices[np.newaxis, ...], combined_colors[np.newaxis, ...]


def log_point_clouds_to_tensorboard(
    writer,
    *,
    tag: str,
    point_cloud_specs: Sequence[PointCloudSpec],
    step: int,
    batch_index: int = 0,
    max_points_per_cloud: Optional[int] = None,
) -> None:
    """Write point clouds to TensorBoard if a mesh payload can be prepared."""
    payload = build_mesh_from_point_clouds(
        point_cloud_specs,
        batch_index=batch_index,
        max_points_per_cloud=max_points_per_cloud,
    )
    if payload is None:
        return
    vertices, colors = payload
    writer.add_mesh(tag, vertices=vertices, colors=colors, global_step=step)


__all__ = [
    "build_mesh_from_point_clouds",
    "log_point_clouds_to_tensorboard",
]
