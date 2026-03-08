"""
Simplified reconstruction helper that bypasses MapAnything.

直接利用已有的世界坐标点云(如 new_point_maps)构建训练所需的最小重建结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from pytorch3d.transforms import quaternion_to_matrix


@dataclass(frozen=True)
class ReconstructionData:
    """Minimal reconstruction tensors consumed by NBV loss stack."""

    recon_world_points: torch.Tensor
    recon_conf: torch.Tensor
    recon_mask: torch.Tensor


def build_recon_from_point_maps(
    point_maps: Optional[torch.Tensor] = None,
    *,
    valid_masks: Optional[torch.Tensor] = None,
    camera_poses: Optional[torch.Tensor] = None,
    depth_z: Optional[torch.Tensor] = None,
    fov_degrees: float = 60.0,
    xy_signs: Tuple[int, int] = (-1, -1),
) -> ReconstructionData:
    """构建最小重建数据: 世界点云 + 置信度 + 有效掩码。"""
    if point_maps is not None:
        if camera_poses is not None or depth_z is not None:
            raise ValueError(
                "build_recon_from_point_maps accepts either `point_maps` or "
                "`camera_poses` + `depth_z`, but not both."
            )
        return _build_recon_from_world_point_maps(
            point_maps=point_maps,
            valid_masks=valid_masks,
        )

    if camera_poses is None or depth_z is None:
        raise ValueError(
            "When `point_maps` is not provided, both `camera_poses` and `depth_z` are required."
        )
    return build_recon_from_depth_z(
        camera_poses=camera_poses,
        depth_z=depth_z,
        valid_masks=valid_masks,
        fov_degrees=fov_degrees,
        xy_signs=xy_signs,
    )


def build_recon_from_depth_z(
    *,
    camera_poses: torch.Tensor,
    depth_z: torch.Tensor,
    valid_masks: Optional[torch.Tensor] = None,
    fov_degrees: float = 60.0,
    xy_signs: Tuple[int, int] = (-1, -1),
) -> ReconstructionData:
    """用 7D 相机位姿和相机坐标系下的 Z 深度图构建最小重建结果。"""
    world_points, masks = _backproject_depth_z_to_world(
        depth_z=depth_z,
        camera_poses=camera_poses,
        valid_masks=valid_masks,
        fov_degrees=fov_degrees,
        xy_signs=xy_signs,
    )
    conf = masks.to(device=world_points.device, dtype=world_points.dtype)
    return ReconstructionData(
        recon_world_points=world_points,
        recon_conf=conf,
        recon_mask=masks,
    )


def _build_recon_from_world_point_maps(
    *,
    point_maps: torch.Tensor,
    valid_masks: Optional[torch.Tensor],
) -> ReconstructionData:
    """构建最小重建数据: 世界点云 + 置信度 + 有效掩码。"""
    if point_maps.ndim != 5 or point_maps.shape[-1] != 3:
        raise ValueError(
            f"point_maps must have shape [B, S, H, W, 3], but got {tuple(point_maps.shape)}"
        )

    batch_size, num_views, height, width, _ = point_maps.shape
    device = point_maps.device
    dtype = point_maps.dtype

    if valid_masks is None:
        raise ValueError("valid_masks is required and must match point_maps shape [B, S, H, W].")
    if valid_masks.shape[:4] != (batch_size, num_views, height, width):
        raise ValueError(
            "valid_masks must have shape [B, S, H, W] matching point_maps "
            f"({valid_masks.shape} vs {(batch_size, num_views, height, width)})"
        )
    masks = valid_masks.to(device=device, dtype=torch.bool)

    conf = masks.to(device=device, dtype=point_maps.dtype)
    return ReconstructionData(
        recon_world_points=point_maps.to(dtype=dtype),
        recon_conf=conf,
        recon_mask=masks,
    )


def _backproject_depth_z_to_world(
    *,
    depth_z: torch.Tensor,
    camera_poses: torch.Tensor,
    valid_masks: Optional[torch.Tensor],
    fov_degrees: float,
    xy_signs: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if camera_poses.ndim not in {2, 3} or camera_poses.shape[-1] != 7:
        raise ValueError(
            "camera_poses must have shape [B, 7] or [B, S, 7], "
            f"but got {tuple(camera_poses.shape)}"
        )

    if depth_z.ndim >= 3 and depth_z.shape[-1] == 1:
        depth_z = depth_z.squeeze(-1)
    if depth_z.ndim not in {3, 4}:
        raise ValueError(
            "depth_z must have shape [B, H, W], [B, H, W, 1], [B, S, H, W], "
            f"or [B, S, H, W, 1], but got {tuple(depth_z.shape)}"
        )

    leading_shape = tuple(depth_z.shape[:-2])
    if camera_poses.shape[:-1] != leading_shape:
        raise ValueError(
            "camera_poses leading dims must match depth_z leading dims. "
            f"Got camera_poses={tuple(camera_poses.shape)} vs depth_z={tuple(depth_z.shape)}."
        )

    height, width = depth_z.shape[-2:]
    device = depth_z.device
    dtype = depth_z.dtype

    mask = torch.isfinite(depth_z) & (depth_z > 0)
    if valid_masks is not None:
        if valid_masks.ndim >= 3 and valid_masks.shape[-1] == 1:
            valid_masks = valid_masks.squeeze(-1)
        if tuple(valid_masks.shape) != tuple(depth_z.shape):
            raise ValueError(
                "valid_masks must match depth_z after squeezing the optional channel dim. "
                f"Got valid_masks={tuple(valid_masks.shape)} vs depth_z={tuple(depth_z.shape)}."
            )
        mask = mask & valid_masks.to(device=device, dtype=torch.bool)

    depth_flat = depth_z.reshape(-1, height, width)
    poses_flat = camera_poses.reshape(-1, 7).to(device=device, dtype=dtype)

    fx, fy, cx, cy = _compute_pinhole_intrinsics(
        height=height,
        width=width,
        fov_degrees=fov_degrees,
    )
    fx = torch.as_tensor(fx, device=device, dtype=dtype)
    fy = torch.as_tensor(fy, device=device, dtype=dtype)
    cx = torch.as_tensor(cx, device=device, dtype=dtype)
    cy = torch.as_tensor(cy, device=device, dtype=dtype)

    u = torch.arange(width, device=device, dtype=dtype)
    v = torch.arange(height, device=device, dtype=dtype)
    try:
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    except TypeError:  # pragma: no cover
        v_grid, u_grid = torch.meshgrid(v, u)

    sx, sy = int(xy_signs[0]), int(xy_signs[1])
    x_cam = (u_grid - cx) / fx * depth_flat * float(sx)
    y_cam = (v_grid - cy) / fy * depth_flat * float(sy)
    z_cam = depth_flat
    cam_points = torch.stack((x_cam, y_cam, z_cam), dim=-1)

    positions = poses_flat[:, :3]
    quaternions_xyzw = poses_flat[:, 3:]
    quaternions_xyzw = quaternions_xyzw / quaternions_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]]
    rotation_w2c = quaternion_to_matrix(quaternions_wxyz)
    rotation_c2w = rotation_w2c.transpose(1, 2)

    cam_points_vec = cam_points.view(cam_points.shape[0], -1, 3)
    world_points_vec = torch.bmm(cam_points_vec, rotation_c2w) + positions.unsqueeze(1)
    world_points = world_points_vec.view(*leading_shape, height, width, 3)
    world_points = world_points.masked_fill(~mask.unsqueeze(-1), 0.0)
    return world_points, mask


def _compute_pinhole_intrinsics(
    *,
    height: int,
    width: int,
    fov_degrees: float,
) -> Tuple[float, float, float, float]:
    fov_radians = math.radians(float(fov_degrees))
    fy = 0.5 * float(height) / math.tan(fov_radians / 2.0)
    fx = 0.5 * float(width) / math.tan(fov_radians / 2.0)
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return fx, fy, cx, cy


__all__ = ["ReconstructionData", "build_recon_from_depth_z", "build_recon_from_point_maps"]
