from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn as nn

from .chamfer_regularizer import ChamferRegularizer
from .confidence_regularizer import ConfidenceRegularizer
from .pointcloud_builder import PointCloudExtractor
from .pose_penalty import PosePenalty
from .viewpoint_regularizer import ViewpointRegularizer

if TYPE_CHECKING:
    from ...rendering import DifferentiableRenderer


class ReconstructionLoss(nn.Module):
    """Combine Chamfer, confidence, viewpoint, and pose penalties."""

    def __init__(
        self,
        chamfer_weight: float = 10.0,
        confidence_weight: float = 0.0,
        viewpoint_weight: float = 0.0,
        pose_penalty_weight: float = 1.0,
        renderer: Optional["DifferentiableRenderer"] = None,
        pose_up_axis: str = "Y",
        save_point_clouds: bool = True,
        point_cloud_dir_name: str = "point_clouds",
        max_points_per_cloud: int = 32768,
        point_source: str = "vggt",
        confidence_threshold: float = 0.0, # 百分之
        black_pixel_threshold: float = 0.1,
        pose_outer_radius: float = 3.5,
        pose_inner_radius: float = 2.5,
        pose_floor_margin: float = 1.0,
        default_device: Optional[torch.device] = None,
        tensor_dtype: torch.dtype = torch.float32,
        use_log_warp_for_chamfer: bool = False,
    ) -> None:
        super().__init__()

        self.renderer = renderer

        normalized_source = point_source.lower()
        if normalized_source not in {"vggt", "depth"}:
            raise ValueError("point_source must be either 'vggt' or 'depth'.")

        extractor = PointCloudExtractor(black_threshold=black_pixel_threshold)
        self.chamfer_regularizer = ChamferRegularizer(
            weight=chamfer_weight,
            extractor=extractor,
            point_source=normalized_source,
            confidence_threshold=confidence_threshold,
            max_points_per_cloud=max_points_per_cloud,
            save_point_clouds=save_point_clouds,
            point_cloud_dir_name=point_cloud_dir_name,
            use_log_warp_for_chamfer=use_log_warp_for_chamfer,
        )
        self.confidence_regularizer = ConfidenceRegularizer(weight=confidence_weight)
        self.viewpoint_regularizer = ViewpointRegularizer(weight=viewpoint_weight)
        self.pose_penalty = PosePenalty(
            weight=pose_penalty_weight,
            up_axis=pose_up_axis,
            outer_radius=pose_outer_radius,
            inner_radius=pose_inner_radius,
            floor_margin=pose_floor_margin,
        )
        self.default_device = default_device or torch.device("cpu")
        self.tensor_dtype = tensor_dtype

    # ---------- helpers ----------

    @staticmethod
    def _to_float(x: torch.Tensor) -> float:
        """Detach and convert to Python float for logging."""
        return float(x.detach())

    @staticmethod
    def _add_loss(
        total: torch.Tensor,
        components: Dict[str, float],
        name: str,
        weighted: torch.Tensor,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate loss and record raw / weighted values."""
        total = total + weighted
        components[f"{name}_loss"] = ReconstructionLoss._to_float(raw)
        components[f"weighted_{name}_loss"] = ReconstructionLoss._to_float(weighted)
        return total

    # ---------- main forward ----------

    def forward(
        self,
        recon_data: Dict[str, torch.Tensor],
        gt_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        combined_camera_poses: Optional[torch.Tensor],
        return_components: bool = False,
        point_cloud_dir: Optional[str] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:

        device = self.default_device
        dtype = self.tensor_dtype

        total_loss = torch.zeros((), device=device, dtype=dtype)
        loss_components: Dict[str, float] = {}

        # --- Chamfer ---
        weighted_chamfer, chamfer_raw, confidence_mask = self.chamfer_regularizer(
            recon_data,
            gt_data,
            combined_images_batch,
            combined_camera_poses,
            device,
            dtype,
            point_cloud_dir=point_cloud_dir,
        )
        total_loss = self._add_loss(
            total_loss,
            loss_components,
            "chamfer",
            weighted_chamfer,
            chamfer_raw,
        )
        # if confidence_mask is not None:
        #     with torch.no_grad():
        #         flat_counts = confidence_mask.reshape(confidence_mask.shape[0], -1).sum(dim=1)
        #         loss_components["chamfer_pred_points_mean"] = float(flat_counts.float().mean())
        #         loss_components["chamfer_pred_points_min"] = float(flat_counts.min())
        #         loss_components["chamfer_pred_points_zero_frac"] = float(
        #             (flat_counts == 0).float().mean()
        #         )

        #         if confidence_mask.dim() >= 4:
        #             per_view_counts = confidence_mask.sum(dim=tuple(range(2, confidence_mask.dim())))
        #             last_view_counts = per_view_counts[:, -1]
        #             loss_components["chamfer_pred_points_last_view_mean"] = float(
        #                 last_view_counts.float().mean()
        #             )
        #             loss_components["chamfer_pred_points_last_view_min"] = float(
        #                 last_view_counts.min()
        #             )
        #             loss_components["chamfer_pred_points_last_view_zero_frac"] = float(
        #                 (last_view_counts == 0).float().mean()
        #             )

        # --- Confidence ---
        weighted_confidence, confidence_raw = self.confidence_regularizer(
            recon_data,
            device,
            dtype,
            confidence_mask=confidence_mask,
        )
        total_loss = self._add_loss(
            total_loss,
            loss_components,
            "confidence",
            weighted_confidence,
            confidence_raw,
        )

        # --- Viewpoint ---
        weighted_viewpoint, viewpoint_raw = self.viewpoint_regularizer(
            combined_images_batch,
            device,
            dtype,
        )
        total_loss = self._add_loss(
            total_loss,
            loss_components,
            "viewpoint",
            weighted_viewpoint,
            viewpoint_raw,
        )

        # --- Pose penalty ---
        weighted_pose_penalty, pose_penalty_raw, pose_penalty_terms = self.pose_penalty(
            combined_camera_poses,
            device,
            dtype,
        )
        total_loss = self._add_loss(
            total_loss,
            loss_components,
            "pose_penalty",
            weighted_pose_penalty,
            pose_penalty_raw,
        )

        # 单独记录 pose 的各个子项
        for term_name, term_value in pose_penalty_terms.items():
            loss_components[f"pose_{term_name}"] = self._to_float(term_value)

        loss_components["total_loss"] = self._to_float(total_loss)

        if return_components:
            return total_loss, loss_components
        return total_loss


__all__ = ["ReconstructionLoss"]
