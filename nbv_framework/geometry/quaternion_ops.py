"""Quaternion and pose helpers shared by the NBV core pipeline."""

from __future__ import annotations

from typing import Tuple

import torch
from pytorch3d.transforms import matrix_to_quaternion


def _normalize_quaternion_xyzw(quaternions: torch.Tensor) -> torch.Tensor:
    return quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def standardize_quaternion_xyzw(quaternions: torch.Tensor) -> torch.Tensor:
    sign = torch.where(quaternions[..., 3:4] < 0, -1.0, 1.0).to(quaternions.dtype)
    return quaternions * sign


def quaternion_inverse_xyzw(quaternions: torch.Tensor) -> torch.Tensor:
    quaternions = _normalize_quaternion_xyzw(quaternions)
    xyz = -quaternions[..., :3]
    w = quaternions[..., 3:]
    return torch.cat([xyz, w], dim=-1)


def quaternion_multiply_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


def quaternion_to_rotation_matrix_xyzw(quaternions: torch.Tensor) -> torch.Tensor:
    squeeze_batch_dim = False
    if quaternions.dim() == 1:
        quaternions = quaternions.unsqueeze(0)
        squeeze_batch_dim = True

    quaternions = _normalize_quaternion_xyzw(quaternions)
    x, y, z, w = quaternions.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotation = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    if squeeze_batch_dim:
        return rotation.squeeze(0)
    return rotation


def rotation_matrix_to_quaternion_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    quaternion_wxyz = matrix_to_quaternion(matrix)
    quaternion_xyzw = quaternion_wxyz[..., [1, 2, 3, 0]]
    quaternion_xyzw = _normalize_quaternion_xyzw(quaternion_xyzw)
    return standardize_quaternion_xyzw(quaternion_xyzw)


def transform_pose_using_quats_and_trans_2_to_1(
    quats1: torch.Tensor,
    trans1: torch.Tensor,
    quats2: torch.Tensor,
    trans2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    squeeze_batch_dim = False
    if quats1.dim() == 1:
        quats1 = quats1.unsqueeze(0)
        trans1 = trans1.unsqueeze(0)
        quats2 = quats2.unsqueeze(0)
        trans2 = trans2.unsqueeze(0)
        squeeze_batch_dim = True

    target_dtype = quats1.dtype
    trans1 = trans1.to(dtype=target_dtype)
    quats2 = quats2.to(dtype=target_dtype)
    trans2 = trans2.to(dtype=target_dtype)

    inv_quats1 = quaternion_inverse_xyzw(quats1)
    rel_quats = quaternion_multiply_xyzw(inv_quats1, quats2)

    trans_diff_world = trans2 - trans1
    rotation_1_inv = quaternion_to_rotation_matrix_xyzw(inv_quats1)
    rel_trans = torch.bmm(rotation_1_inv, trans_diff_world.unsqueeze(-1)).squeeze(-1)

    if squeeze_batch_dim:
        return rel_quats.squeeze(0), rel_trans.squeeze(0)
    return rel_quats, rel_trans


def normalize_pose_translations(
    pose_translations: torch.Tensor,
    return_norm_factor: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    if pose_translations.ndim != 3 or pose_translations.shape[-1] != 3:
        raise ValueError(
            "pose_translations must have shape [B, V, 3], "
            f"got {tuple(pose_translations.shape)}"
        )

    distances = pose_translations.norm(dim=-1)
    non_zero = distances > 0
    norm_factor = distances.sum(dim=1) / (non_zero.sum(dim=1) + 1e-8)
    norm_factor = norm_factor.clamp_min(1e-8)
    normalized = pose_translations / norm_factor.unsqueeze(-1).unsqueeze(-1)

    if return_norm_factor:
        return normalized, norm_factor
    return normalized


def apply_log_to_norm(input_data: torch.Tensor) -> torch.Tensor:
    distances = input_data.norm(dim=-1, keepdim=True)
    unit_vectors = input_data / distances.clamp_min(1e-8)
    return unit_vectors * torch.log1p(distances)


__all__ = [
    "apply_log_to_norm",
    "normalize_pose_translations",
    "quaternion_inverse_xyzw",
    "quaternion_multiply_xyzw",
    "quaternion_to_rotation_matrix_xyzw",
    "rotation_matrix_to_quaternion_xyzw",
    "standardize_quaternion_xyzw",
    "transform_pose_using_quats_and_trans_2_to_1",
]
