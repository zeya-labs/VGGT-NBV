"""Rendering and metric-input contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class MultiViewRenderBatch:
    """Rendered tensors for known multi-view camera batches.

    Shapes:
    - rgb: [B, S, 3, H, W]
    - points: [B, S, H, W, 3]
    - mask: [B, S, H, W]
    - depth: [B, S, H, W, 1]
    """

    rgb: Optional[torch.Tensor] = None
    points: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None


@dataclass
class CandidateRenderBatch:
    """Rendered tensors for one candidate view per batch item.

    Shapes:
    - rgb: [B, 3, H, W]
    - points: [B, H, W, 3]
    - mask: [B, H, W]
    - depth: [B, H, W, 1]
    """

    rgb: Optional[torch.Tensor] = None
    points: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None


@dataclass
class MetricPointCloudBatch:
    """Point-cloud tensors extracted for evaluation metrics."""

    pred_points_list: List[torch.Tensor]
    gt_points: Optional[torch.Tensor]
