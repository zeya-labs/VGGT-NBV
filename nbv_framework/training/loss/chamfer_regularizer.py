"""Chamfer regularizer component."""

from typing import Dict, List, Optional, Tuple

import logging
import torch
from pytorch3d.structures import Pointclouds

from .chamfer import ChamferDistance
from .pointcloud_builder import PointCloudExtractor


class ChamferRegularizer:
    """Builds predicted clouds and measures Chamfer distance to GT."""

    def __init__(
        self,
        weight: float,
        extractor: PointCloudExtractor,
        point_source: str = "vggt",
        confidence_threshold: float = 0.0,
        max_points_per_cloud: int = 4096,
        save_point_clouds: bool = True,
        point_cloud_dir_name: str = "point_clouds",
        log_tensorboard: bool = False,
        use_log_warp_for_chamfer: bool = False,
    ) -> None:
        self.weight = weight
        self.extractor = extractor
        self.point_source = point_source
        self.confidence_threshold = confidence_threshold

        self.chamfer = ChamferDistance(
            max_points_per_cloud=max_points_per_cloud,
            use_log_warp=use_log_warp_for_chamfer,
        )
        self.save_point_clouds = bool(save_point_clouds)
        self.point_cloud_dir_name = point_cloud_dir_name
        self.chamfer.configure_point_cloud_logging(
            enable_save=self.save_point_clouds,
            subdir_name=self.point_cloud_dir_name,
            max_points_per_cloud=max_points_per_cloud,
            log_to_tensorboard=log_tensorboard,
        )

    def __call__(
        self,
        recon_data: Dict[str, torch.Tensor],
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        combined_camera_poses: Optional[torch.Tensor],
        writer,
        step,
        device: torch.device,
        dtype: torch.dtype,
        point_cloud_dir: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        zero = torch.zeros((), device=device, dtype=dtype)
        if self.weight <= 0 or "gt_points" not in gt_data:
            return zero, zero, None

        if combined_camera_poses is None:
            raise ValueError(
                "combined_camera_poses must be provided when Chamfer loss is enabled."
            )

        gt_point_maps = gt_data.get("gt_point_maps")
        gt_valid_masks = gt_data.get("gt_valid_masks")
        if gt_point_maps is None or gt_valid_masks is None:
            raise KeyError(
                "gt_mesh_data must contain 'gt_point_maps' and 'gt_valid_masks' for Chamfer loss."
            )

        pred_pointclouds, correspondence_mask = self.extractor(
            recon_data=recon_data,
            combined_images_batch=combined_images_batch,
            confidence_threshold=self.confidence_threshold,
            source=self.point_source,  # type: ignore[arg-type]
            gt_valid_masks=gt_valid_masks,
        )

        gt_points_batch = [
            torch.as_tensor(points, device=device, dtype=dtype)
            for points in gt_data["gt_points"]
        ]
        gt_pointclouds = Pointclouds(points=gt_points_batch)

        correspondence_points: List[torch.Tensor] = []
        for i in range(correspondence_mask.shape[0]):
            mask_i = correspondence_mask[i]
            if mask_i.any():
                gt_points_i = gt_point_maps[i][mask_i]
            else:
                gt_points_i = torch.empty(
                    (0, 3),
                    device=gt_point_maps.device,
                    dtype=gt_point_maps.dtype,
                )
            correspondence_points.append(gt_points_i)

        if len(pred_pointclouds) != len(gt_pointclouds):
            logging.warning("预测点云列表的批次大小与GT点云不匹配。跳过Chamfer损失计算。")
            return zero, zero, correspondence_mask

        chamfer_loss_value = self.chamfer(
            pred_pointclouds,
            gt_pointclouds,
            correspondence_points=correspondence_points,
            writer=writer,
            step=step,
            point_cloud_dir=point_cloud_dir,
        )

        weighted_loss = self.weight * chamfer_loss_value
        return weighted_loss, chamfer_loss_value, correspondence_mask

__all__ = ["ChamferRegularizer"]
