from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch3d.transforms import quaternion_to_matrix

from mapanything.utils.geometry import (
    normalize_pose_translations,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    transform_pose_using_quats_and_trans_2_to_1,
)
from ..utils.camera_utils import position_to_pose_tensor
from ..utils.mapanything_views import _compute_pose_quats_and_trans_for_across_views_in_ref_view


logger = logging.getLogger(__name__)


def compute_pose_for_across_views_in_ref_view(
    views: List[Dict[str, Any]],
) -> torch.Tensor:
    """
    计算跨视角策略评估时的相机位姿，均转换到参考视角坐标系下。
    返回形状 (B, S, 7) 的张量，最后一维为 [tx, ty, tz, qx, qy, qz, qw]。
    """
    num_views = len(views)
    batch_size_per_view = views[0]["img"].shape[0]
    device = views[0]["img"].device
    dtype = views[0]["img"].dtype

    per_sample_cam_input_mask = torch.ones(
        batch_size_per_view * num_views,
        dtype=torch.bool,
        device=device,
    )

    pose_quats_flat, pose_trans_flat, _ = _compute_pose_quats_and_trans_for_across_views_in_ref_view(
        views=views,
        num_views=num_views,
        device=device,
        dtype=dtype,
        batch_size_per_view=batch_size_per_view,
        per_sample_cam_input_mask=per_sample_cam_input_mask,
    )

    pose_flat_7d = torch.cat([pose_trans_flat, pose_quats_flat], dim=-1)
    pose_sb7 = pose_flat_7d.view(num_views, batch_size_per_view, 7)
    pose_bs7 = pose_sb7.transpose(0, 1).contiguous()
    return pose_bs7


def compute_policy_pose(
    policy_output: torch.Tensor,
    camera_poses_batch: torch.Tensor,
    *,
    min_radius: float = 1.3,
    max_radius: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据策略输出计算绝对位姿及相关中间量。"""
    reference_position = camera_poses_batch[:, 0, :3]
    predicted_relative_position = policy_output[:, :3]

    if camera_poses_batch.ndim != 3 or camera_poses_batch.shape[-1] != 7:
        raise ValueError(
            f"camera_poses_batch expected shape [B, S, 7], got {tuple(camera_poses_batch.shape)}"
        )
    ref_quat_xyzw = camera_poses_batch[:, 0, 3:]
    quat_wxyz = ref_quat_xyzw[:, [3, 0, 1, 2]]
    rotation_w2c_row = quaternion_to_matrix(quat_wxyz)
    rotation_c2w_row = rotation_w2c_row.transpose(1, 2)
    predicted_relative_position_world = torch.bmm(
        predicted_relative_position.unsqueeze(1),
        rotation_c2w_row,
    ).squeeze(1)
    absolute_position = reference_position + predicted_relative_position_world
    dist = torch.norm(absolute_position, dim=-1, keepdim=True)

    clamped_dist = torch.clamp(dist, min=min_radius, max=max_radius)
    absolute_position = absolute_position * (clamped_dist / (dist + 1e-8))

    next_camera_pose = position_to_pose_tensor(absolute_position)
    return next_camera_pose, predicted_relative_position, absolute_position


def compute_pose_scale_factor(
    camera_poses_batch: torch.Tensor,
    *,
    log_stats: bool = True,
) -> torch.Tensor:
    """将所有视角变换到 view0 坐标系后，按跨视角平均范数归一化平移，返回归一化因子。"""
    if camera_poses_batch.ndim != 3 or camera_poses_batch.shape[-1] != 7:
        raise ValueError(
            f"camera_poses_batch expected shape [B, S, 7], got {tuple(camera_poses_batch.shape)}"
        )
    positions_world = camera_poses_batch[..., :3]
    quats_world_to_cam = camera_poses_batch[..., 3:]

    B, S, _ = positions_world.shape
    R_wc = quaternion_to_rotation_matrix(quats_world_to_cam.reshape(-1, 4)).view(B, S, 3, 3)
    R_cw = R_wc.transpose(-1, -2)
    quats_cam2world = rotation_matrix_to_quaternion(R_cw.reshape(-1, 3, 3)).view(B, S, 4)
    trans_cam2world = positions_world

    ref_quat = quats_cam2world[:, 0]
    ref_trans = trans_cam2world[:, 0]

    ref_quat_exp = ref_quat.unsqueeze(1).expand(-1, S, -1).reshape(-1, 4)
    ref_trans_exp = ref_trans.unsqueeze(1).expand(-1, S, -1).reshape(-1, 3)
    quats_flat = quats_cam2world.reshape(-1, 4)
    trans_flat = trans_cam2world.reshape(-1, 3)

    _, rel_trans_flat = transform_pose_using_quats_and_trans_2_to_1(
        ref_quat_exp,
        ref_trans_exp,
        quats_flat,
        trans_flat,
    )
    rel_trans = rel_trans_flat.view(B, S, 3)

    _, norm_factor = normalize_pose_translations(rel_trans, return_norm_factor=True)
    if log_stats:
        logger.info(
            "Pose scale factor stats — mean: %.4f, min: %.4f, max: %.4f",
            norm_factor.mean().item(),
            norm_factor.min().item(),
            norm_factor.max().item(),
        )
    return norm_factor
