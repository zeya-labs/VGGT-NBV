"""Domain-level depth reconstruction utilities."""

from __future__ import annotations

from typing import Optional

import torch
from pytorch3d.transforms import quaternion_to_matrix


def world_points_to_camera_depth(
    point_maps: torch.Tensor,
    camera_poses: torch.Tensor,
    *,
    valid_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert world-space point maps to camera-space Z depth."""
    if point_maps.shape[-1] != 3:
        raise ValueError(f"point_maps last dim must be 3, got {tuple(point_maps.shape)}")
    if camera_poses.shape[-1] != 7:
        raise ValueError(f"camera_poses last dim must be 7, got {tuple(camera_poses.shape)}")

    is_sequence = (point_maps.ndim == 5) and (camera_poses.ndim == 3)
    if point_maps.ndim == 4 and camera_poses.ndim == 2:
        batch_size = point_maps.shape[0]
        num_views = 1
        height, width = point_maps.shape[1:3]
        points_flat = point_maps.reshape(-1, height, width, 3)
        poses_flat = camera_poses.reshape(-1, 7)
        masks_flat = valid_masks.reshape(-1, height, width) if valid_masks is not None else None
    elif is_sequence:
        batch_size = point_maps.shape[0]
        num_views = point_maps.shape[1]
        height, width = point_maps.shape[2:4]
        points_flat = point_maps.reshape(-1, height, width, 3)
        poses_flat = camera_poses.reshape(-1, 7)
        masks_flat = valid_masks.reshape(-1, height, width) if valid_masks is not None else None
    else:
        raise ValueError(
            f"Unsupported shapes: point_maps={tuple(point_maps.shape)}, camera_poses={tuple(camera_poses.shape)}"
        )

    device = points_flat.device
    dtype = points_flat.dtype

    positions = poses_flat[:, :3].to(device=device, dtype=dtype)
    quaternions = poses_flat[:, 3:].to(device=device, dtype=dtype)
    quaternion_wxyz = torch.stack(
        (quaternions[:, 3], quaternions[:, 0], quaternions[:, 1], quaternions[:, 2]),
        dim=-1,
    )

    rotation_w2c = quaternion_to_matrix(quaternion_wxyz)
    points_vec = points_flat.view(points_flat.shape[0], -1, 3)
    relative = points_vec - positions.unsqueeze(1)
    camera_points = torch.bmm(relative, rotation_w2c)
    depth = camera_points[..., 2:3].view(points_flat.shape[0], height, width, 1)

    if masks_flat is not None:
        depth = depth.masked_fill(~masks_flat.to(device=device).unsqueeze(-1), 0.0)

    if is_sequence:
        return depth.view(batch_size, num_views, height, width, 1)
    return depth


__all__ = ["world_points_to_camera_depth"]
