"""Evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class PoseEvaluationResult:
    total_loss: torch.Tensor
    loss_components: Dict[str, float]
    new_images: torch.Tensor
    gt_mesh_data: Dict[str, torch.Tensor]
    depth_z: Optional[torch.Tensor]


@dataclass
class MetricSummary:
    name: str
    model: Tuple[float, float, int]
    random: Tuple[float, float, int]
