"""Metrics port interface."""

from __future__ import annotations

from typing import Dict, List, Protocol

import torch


class MetricsPort(Protocol):
    def compute(self, pred_points_list: List[torch.Tensor], gt_points: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics from predicted and ground-truth points."""
