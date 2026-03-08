"""Typed batch contracts used across services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class NBVBatch:
    """Raw collated batch before preparation."""

    initial_images: Optional[torch.Tensor]
    camera_poses: torch.Tensor
    gt_mesh_data: Dict[str, Any]
    mesh_batch: Any
    mesh_paths: Optional[List[Optional[str]]]
    normalize_methods: Optional[List[Optional[str]]]
    meta: Any


@dataclass
class PreparedBatch:
    """Prepared batch with guaranteed tensors for the train step."""

    initial_images: torch.Tensor
    camera_poses: torch.Tensor
    depth_z: Optional[torch.Tensor]
    gt_mesh_data: Dict[str, torch.Tensor]
    trimmed_gt_mesh_data: Dict[str, torch.Tensor]
    mesh_batch: Any
    mesh_paths: Optional[List[Optional[str]]]
    selection: torch.Tensor
    active_view_count: int
