"""Test metric evaluation service."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from nbv_framework.domain.models.direct_reconstruction import build_recon_from_point_maps
from nbv_framework.application.ports import LossPort, MetricsPort


class TestEvaluationService:
    def __init__(self, *, loss: LossPort, metrics: MetricsPort) -> None:
        self.loss = loss
        self.metrics = metrics

    def compute_metrics(
        self,
        *,
        gt_mesh_data: Dict[str, torch.Tensor],
        combined_images_batch: torch.Tensor,
        combined_camera_poses: torch.Tensor,
        depth_z: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        point_maps = gt_mesh_data.get("gt_point_maps")
        valid_masks = gt_mesh_data.get("gt_valid_masks")
        if point_maps is None or valid_masks is None:
            raise RuntimeError("gt_point_maps or gt_valid_masks missing for test metrics")

        recon_data = build_recon_from_point_maps(
            point_maps=point_maps,
            camera_poses=combined_camera_poses,
            valid_masks=valid_masks,
            depth_z=depth_z,
        )

        pred_points_list, gt_points = self.loss.extract_pred_points(
            recon_data,
            gt_mesh_data,
            combined_images_batch,
        )
        if gt_points is None:
            raise RuntimeError("gt_points missing for test metrics")

        return self.metrics.compute(pred_points_list, gt_points)
