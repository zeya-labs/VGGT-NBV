"""Reconstruction loss implementation of LossPort."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from nbv_framework.reconstruction import ReconstructionData
from nbv_framework.training.losses import ReconstructionLoss


class ReconstructionLossAdapter:
    def __init__(self, loss_module: ReconstructionLoss) -> None:
        self.loss_module = loss_module

    def compute_loss(
        self,
        recon_data: ReconstructionData,
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        combined_camera_poses: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.loss_module(
            recon_data,
            gt_data,
            combined_images_batch,
            combined_camera_poses,
            return_components=True,
        )

    def export_point_clouds(
        self,
        recon_data: ReconstructionData,
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        *,
        point_cloud_dir: Optional[str],
    ) -> None:
        if point_cloud_dir is None:
            return
        self.loss_module.export_point_clouds(
            recon_data,
            gt_data,
            combined_images_batch,
            point_cloud_dir=point_cloud_dir,
        )

    def extract_pred_points(
        self,
        recon_data: ReconstructionData,
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
    ):
        chamfer_reg = self.loss_module.chamfer_regularizer
        pred_points_list, _ = chamfer_reg.extractor(
            recon_data=recon_data,
            combined_images_batch=combined_images_batch,
            confidence_threshold=chamfer_reg.confidence_threshold,
        )
        gt_points = gt_data.get("gt_points")
        return pred_points_list, gt_points
