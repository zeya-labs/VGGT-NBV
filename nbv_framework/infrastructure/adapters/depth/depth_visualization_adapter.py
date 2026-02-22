"""Infrastructure adapter for depth visualization helpers."""

from __future__ import annotations

import torch

from nbv_framework.infrastructure.utils.camera_utils import normalize_depth_for_visualization


class DepthVisualizationAdapter:
    def normalize_depth_for_visualization(
        self,
        depth: torch.Tensor,
        valid_masks: torch.Tensor,
    ) -> torch.Tensor:
        return normalize_depth_for_visualization(depth, valid_masks)
