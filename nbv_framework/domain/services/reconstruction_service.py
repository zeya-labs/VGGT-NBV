"""
Simplified reconstruction helper that bypasses MapAnything.

直接利用已有的世界坐标点云(如 new_point_maps)构建训练所需的最小重建结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ReconstructionData:
    """Minimal reconstruction tensors consumed by NBV loss stack."""

    recon_world_points: torch.Tensor
    recon_conf: torch.Tensor
    recon_mask: torch.Tensor


def build_recon_from_point_maps(
    point_maps: torch.Tensor,
    *,
    valid_masks: Optional[torch.Tensor] = None,
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


__all__ = ["ReconstructionData", "build_recon_from_point_maps"]
