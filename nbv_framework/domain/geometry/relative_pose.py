"""Domain helpers for relative pose transformations across views."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from mapanything.utils.geometry import transform_pose_using_quats_and_trans_2_to_1


def compute_relative_pose_quats_and_trans(
    views: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-view poses in the reference-view coordinate frame."""
    num_views = len(views)
    batch_size_per_view = views[0]["img"].shape[0]
    dtype = views[0]["img"].dtype

    all_quats = torch.stack([view["camera_pose_quats"] for view in views]).to(dtype)
    all_trans = torch.stack([view["camera_pose_trans"] for view in views]).to(dtype)

    ref_quats = all_quats[0:1].expand(num_views, -1, -1)
    ref_trans = all_trans[0:1].expand(num_views, -1, -1)

    rel_quats, rel_trans = transform_pose_using_quats_and_trans_2_to_1(
        ref_quats.reshape(-1, 4),
        ref_trans.reshape(-1, 3),
        all_quats.reshape(-1, 4),
        all_trans.reshape(-1, 3),
    )
    return rel_quats, rel_trans


__all__ = ["compute_relative_pose_quats_and_trans"]
