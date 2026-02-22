"""Loss port interface."""

from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple

import torch
from nbv_framework.domain.services import ReconstructionData


class LossPort(Protocol):
    def compute_loss(
        self,
        recon_data: ReconstructionData,
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        combined_camera_poses: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss and per-term components."""

    def export_point_clouds(
        self,
        recon_data: ReconstructionData,
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        *,
        point_cloud_dir: Optional[str],
    ) -> None:
        """Export debug point clouds when enabled."""

    def extract_pred_points(
        self,
        recon_data: ReconstructionData,
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
    ):
        """Return predicted point clouds for metric computation."""
