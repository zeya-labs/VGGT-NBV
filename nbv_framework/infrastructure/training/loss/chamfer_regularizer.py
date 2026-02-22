"""Chamfer regularizer component."""

from typing import Dict, Optional, Tuple
import torch

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
        max_points_per_cloud: int = 32768,
        save_point_clouds: bool = True,
        point_cloud_dir_name: str = "point_clouds",
        use_log_warp_for_chamfer: bool = False,
    ) -> None:
        self.weight = weight
        self.extractor = extractor
        self.point_source = point_source
        self.confidence_threshold = confidence_threshold

        self.chamfer = ChamferDistance(
            max_points_per_cloud=max_points_per_cloud,
            save_point_clouds=save_point_clouds,
            use_log_warp=use_log_warp_for_chamfer,
            point_cloud_dir_name=point_cloud_dir_name
        )

    def __call__(
        self,
        recon_data: Dict[str, torch.Tensor],
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pred_points_list, confidence_mask = self.extractor(
            recon_data=recon_data,
            combined_images_batch=combined_images_batch,
            confidence_threshold=self.confidence_threshold,
            source=self.point_source,
            gt_valid_masks=gt_data["gt_valid_masks"],
        )

        gt_points_tensor = gt_data["gt_points"]

        chamfer_loss_value = self.chamfer(
            pred_points_list,
            gt_points_tensor,
        )

        weighted_loss = self.weight * chamfer_loss_value
        return weighted_loss, chamfer_loss_value, confidence_mask

    @torch.no_grad()
    def export_point_clouds(
        self,
        recon_data: Dict[str, torch.Tensor],
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        point_cloud_dir: Optional[str],
    ) -> None:
        if not point_cloud_dir:
            return
        if not self.chamfer.save_point_clouds:
            return
        if "gt_valid_masks" not in gt_data:
            return
        gt_points_tensor = gt_data.get("gt_points")
        if gt_points_tensor is None:
            return

        pred_points_list, _ = self.extractor(
            recon_data=recon_data,
            combined_images_batch=combined_images_batch,
            confidence_threshold=self.confidence_threshold,
            source=self.point_source,
            gt_valid_masks=gt_data["gt_valid_masks"],
        )
        self.chamfer.export_point_clouds(
            pred_points_list,
            gt_points_tensor,
            point_cloud_dir=point_cloud_dir,
        )
