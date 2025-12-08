"""Chamfer regularizer component."""

from typing import Dict, List, Optional, Tuple, Any
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

    def _prepare_gt_points(self, raw_gt_points: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        简化版 GT 点云准备：假设输入为 Tensor [B, N, 3] 或 [N, 3]，仅做 dtype/device 对齐。
        """
        if not torch.is_tensor(raw_gt_points):
            raw_gt_points = torch.as_tensor(raw_gt_points)
        gt_tensor = raw_gt_points.to(device=device, dtype=dtype)
        if gt_tensor.ndim == 2:
            gt_tensor = gt_tensor.unsqueeze(0)
        if gt_tensor.ndim != 3 or gt_tensor.shape[-1] != 3:
            raise ValueError(f"gt_points must have shape [B, N, 3] (or [N, 3]), got {tuple(gt_tensor.shape)}")
        return gt_tensor

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

        pred_points_list, confidence_mask = self.extractor(
            recon_data=recon_data,
            combined_images_batch=combined_images_batch,
            confidence_threshold=self.confidence_threshold,
            source=self.point_source,
            gt_valid_masks=gt_valid_masks,
        )

        raw_gt_points = gt_data["gt_points"]
        gt_points_list = self._prepare_gt_points(raw_gt_points, device, dtype)

        chamfer_loss_value = self.chamfer(
            pred_points_list,
            gt_points_list,
            writer=writer,
            step=step,
            point_cloud_dir=point_cloud_dir,
        )

        weighted_loss = self.weight * chamfer_loss_value
        return weighted_loss, chamfer_loss_value, confidence_mask
