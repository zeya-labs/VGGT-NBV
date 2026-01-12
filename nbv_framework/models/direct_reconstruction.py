"""
Simplified reconstruction helper that bypasses MapAnything.

直接利用已有的世界坐标点云(如 new_point_maps)和相机位姿构建与
``MapAnythingWrapper.reconstruct_and_evaluate`` 对齐的 recon 字典。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..utils.camera_utils import world_points_to_camera_depth

TensorDict = Dict[str, torch.Tensor]


def build_recon_from_point_maps(
    point_maps: torch.Tensor,
    camera_poses: torch.Tensor,
    *,
    valid_masks: Optional[torch.Tensor] = None,
    depth_z: Optional[torch.Tensor] = None,
) -> TensorDict:
    """
    使用预先获得的点云直接拼装重建结果, 不经过 MapAnything 前向。

    Args:
        point_maps: [B, S, H, W, 3] 世界坐标点云(例如 new_point_maps 或拼接后的 updated_point_maps)。
        camera_poses: [B, S, 7] 或 [B, 7] 的相机位姿 (position xyz + quaternion qx,qy,qz,qw)。
        valid_masks: 可选的有效像素掩码, [B, S, H, W], 用作 non_ambiguous_mask。
        depth_z: 可选的深度张量, 若为空则由 point_maps 和 camera_poses 计算。

    Returns:
        recon 字典, 仅包含训练下游所需的关键字段:
        - world_points: 世界坐标点云 [B, S, H, W, 3]
        - cam_trans: 相机平移 [B, S, 3]
        - cam_quats: 相机旋转四元数 [B, S, 4]
        - non_ambiguous_mask: 有效区域掩码 [B, S, H, W]
        - depth: 相机坐标系下的 Z 深度 [B, S, H, W, 1]
    """
    if point_maps.ndim != 5 or point_maps.shape[-1] != 3:
        raise ValueError(
            f"point_maps must have shape [B, S, H, W, 3], but got {tuple(point_maps.shape)}"
        )

    if camera_poses.ndim != 3 or camera_poses.shape[-1] != 7:
        raise ValueError(
            f"camera_poses must have shape [B, S, 7] (or [B, 7] before unsqueeze), got {tuple(camera_poses.shape)}"
        )

    batch_size, num_views, height, width, _ = point_maps.shape
    if camera_poses.shape[0] != batch_size or camera_poses.shape[1] != num_views:
        raise ValueError(
            "Batch/view dimensions of camera_poses must match point_maps "
            f"({camera_poses.shape[:2]} vs {(batch_size, num_views)})"
        )

    device = point_maps.device
    dtype = point_maps.dtype
    camera_poses = camera_poses.to(device=device, dtype=dtype)

    if valid_masks.shape[:4] != (batch_size, num_views, height, width):
        raise ValueError(
            "valid_masks must have shape [B, S, H, W] matching point_maps "
            f"({valid_masks.shape} vs {(batch_size, num_views, height, width)})"
        )
    masks = valid_masks.to(device=device, dtype=torch.bool)

    if depth_z is None:
        depth = world_points_to_camera_depth(
            point_maps,
            camera_poses,
            valid_masks=masks,
        )
    else:
        depth = depth_z
        if depth.dim() == 4:
            depth = depth.unsqueeze(-1)
        if depth.shape[:4] != (batch_size, num_views, height, width):
            raise ValueError(
                "depth_z must align with point_maps spatial dimensions "
                f"({depth.shape} vs {(batch_size, num_views, height, width)})"
            )
        depth = depth.to(device=device, dtype=dtype)

    conf = masks.to(device=device, dtype=point_maps.dtype)

    recon: TensorDict = {
        "world_points": point_maps.to(dtype=dtype),
        "cam_trans": camera_poses[..., :3].contiguous(),
        "cam_quats": camera_poses[..., 3:].contiguous(),
        "non_ambiguous_mask": masks,
        "depth": depth,
        "conf": conf,
        "world_points_conf": conf,
        "depth_conf": conf,
    }
    return recon


__all__ = ["build_recon_from_point_maps"]
