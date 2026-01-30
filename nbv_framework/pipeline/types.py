from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, NamedTuple

import torch


@dataclass
class PreparedBatch:
    initial_images: torch.Tensor
    camera_poses: torch.Tensor
    depth_z: Optional[torch.Tensor]
    gt_mesh_data: Dict[str, torch.Tensor]
    trimmed_gt_mesh_data: Dict[str, torch.Tensor]
    mesh_batch: Any
    mesh_paths: Optional[List[Optional[str]]]
    selection: torch.Tensor
    active_view_count: int


@dataclass
class PolicyInferenceOutput:
    next_camera_pose: torch.Tensor
    predicted_relative_position: torch.Tensor


@dataclass
class RandomBaselineOutput:
    chamfer_loss: float
    images: torch.Tensor
    position_norm_mean: float


class PoseEvaluationResult(NamedTuple):
    total_loss: torch.Tensor
    loss_components: Dict[str, float]
    new_images: torch.Tensor
    gt_mesh_data: Dict[str, torch.Tensor]
    depth_z: Optional[torch.Tensor]
