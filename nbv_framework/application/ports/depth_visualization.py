"""Depth post-processing port interfaces."""

from __future__ import annotations

from typing import Protocol

import torch


class DepthVisualizationPort(Protocol):
    def normalize_depth_for_visualization(
        self,
        depth: torch.Tensor,
        valid_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize depth maps for debug/visual outputs."""
