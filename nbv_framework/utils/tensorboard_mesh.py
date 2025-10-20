"""Utilities for logging point clouds as TensorBoard meshes and GLB exports."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.structures import Pointclouds

ColorSpec = Sequence[int]
PointCloudSpec = Tuple[str, Pointclouds, ColorSpec]
SampledPointCloud = Tuple[str, np.ndarray, np.ndarray]


def _downsample_points(points: torch.Tensor, max_points: Optional[int]) -> torch.Tensor:
    """Uniformly downsample points to the requested count."""
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points

    if points.shape[0] == 0:
        return points

    indices = torch.linspace(
        0, points.shape[0] - 1, steps=max_points, dtype=torch.float32, device=points.device
    )
    indices = indices.round().long()
    indices = torch.clamp(indices, min=0, max=points.shape[0] - 1)
    return points.index_select(0, torch.unique(indices, sorted=True))


def build_mesh_from_point_clouds(
    point_cloud_specs: Sequence[PointCloudSpec],
    *,
    batch_index: int = 0,
    max_points_per_cloud: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[SampledPointCloud]]]:
    """Convert a batch of point clouds into stacked vertex/color arrays."""
    vertices_list: List[np.ndarray] = []
    colors_list: List[np.ndarray] = []
    sampled_clouds: List[SampledPointCloud] = []

    for name, point_cloud, color in point_cloud_specs:
        points_list = point_cloud.points_list()
        if batch_index >= len(points_list):
            continue

        points = points_list[batch_index]
        if points.numel() == 0:
            continue

        points_cpu = points.detach().cpu()
        sampled = _downsample_points(points_cpu, max_points_per_cloud)
        sampled_np = sampled.numpy()
        if sampled_np.ndim != 2 or sampled_np.shape[1] != 3 or sampled_np.shape[0] == 0:
            continue

        color_np = np.asarray(color, dtype=np.uint8).reshape(-1)
        if color_np.size != 3:
            continue
        tiled_colors = np.repeat(color_np.reshape(1, 3), sampled_np.shape[0], axis=0)

        vertices_list.append(sampled_np)
        colors_list.append(tiled_colors)
        sampled_clouds.append((name, sampled_np, tiled_colors))

    if not vertices_list:
        return None

    combined_vertices = np.vstack(vertices_list)[np.newaxis, ...]
    combined_colors = np.vstack(colors_list)[np.newaxis, ...]
    return combined_vertices, combined_colors, sampled_clouds


def _write_point_clouds_glb(
    output_path: str,
    sampled_clouds: Sequence[SampledPointCloud],
) -> None:
    """Write all sampled point clouds into a colored GLB."""
    if not sampled_clouds:
        return

    buffer_data = bytearray()
    buffer_views: List[dict] = []
    accessors: List[dict] = []
    meshes: List[dict] = []
    nodes: List[dict] = []

    def _append_buffer(data: bytes) -> Tuple[int, int]:
        offset = len(buffer_data)
        buffer_data.extend(data)
        while len(buffer_data) % 4 != 0:
            buffer_data.append(0)
        return offset, len(data)

    for idx, (name, positions, colors) in enumerate(sampled_clouds):
        if positions.size == 0 or colors.size == 0:
            continue

        positions_f32 = positions.astype(np.float32, copy=False)
        colors_u8 = colors.astype(np.uint8, copy=False)

        pos_offset, pos_length = _append_buffer(positions_f32.tobytes())
        pos_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": pos_length,
            }
        )
        pos_accessor_index = len(accessors)
        accessors.append(
            {
                "bufferView": pos_view_index,
                "componentType": 5126,
                "count": int(positions_f32.shape[0]),
                "type": "VEC3",
                "min": positions_f32.min(axis=0).tolist(),
                "max": positions_f32.max(axis=0).tolist(),
            }
        )

        color_offset, color_length = _append_buffer(colors_u8.tobytes())
        color_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": color_offset,
                "byteLength": color_length,
            }
        )
        color_accessor_index = len(accessors)
        accessors.append(
            {
                "bufferView": color_view_index,
                "componentType": 5121,
                "count": int(colors_u8.shape[0]),
                "type": "VEC3",
                "normalized": True,
            }
        )

        mesh_index = len(meshes)
        meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": pos_accessor_index,
                            "COLOR_0": color_accessor_index,
                        },
                        "mode": 0,
                    }
                ],
            }
        )
        nodes.append({"mesh": mesh_index, "name": name})

    if not meshes:
        return

    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(buffer_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": meshes,
        "nodes": nodes,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "scene": 0,
    }

    import json

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    header_len = 12
    chunk_header_len = 8
    total_length = header_len + chunk_header_len + len(json_bytes)
    if buffer_data:
        total_length += chunk_header_len + len(buffer_data)

    with open(output_path, "wb") as f:
        f.write(b"glTF")
        f.write((2).to_bytes(4, "little"))
        f.write(total_length.to_bytes(4, "little"))

        f.write(len(json_bytes).to_bytes(4, "little"))
        f.write(b"JSON")
        f.write(json_bytes)

        if buffer_data:
            f.write(len(buffer_data).to_bytes(4, "little"))
            f.write(b"BIN\0")
            f.write(buffer_data)


def log_point_clouds_to_tensorboard(
    writer,
    *,
    tag: str,
    point_cloud_specs: Sequence[PointCloudSpec],
    step: int,
    batch_index: int = 0,
    max_points_per_cloud: Optional[int] = None,
    glb_output_path: Optional[str] = None,
) -> None:
    """Write point clouds to TensorBoard and optionally export a colored GLB."""
    payload = build_mesh_from_point_clouds(
        point_cloud_specs,
        batch_index=batch_index,
        max_points_per_cloud=max_points_per_cloud,
    )
    if payload is None:
        return
    vertices, colors, sampled_clouds = payload
    if glb_output_path is not None:
        os.makedirs(os.path.dirname(glb_output_path), exist_ok=True)
        _write_point_clouds_glb(glb_output_path, sampled_clouds)
    if writer is not None:
        writer.add_mesh(tag, vertices=vertices, colors=colors, global_step=step)


__all__ = [
    "build_mesh_from_point_clouds",
    "log_point_clouds_to_tensorboard",
]
