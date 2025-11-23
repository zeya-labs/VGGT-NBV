from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn as nn

from .chamfer_regularizer import ChamferRegularizer
from .confidence_regularizer import ConfidenceRegularizer
from .pointcloud_builder import PointCloudExtractor
from .pose_penalty import PosePenalty
from .viewpoint_regularizer import ViewpointRegularizer

if TYPE_CHECKING:
    from ..rendering import DifferentiableRenderer


class ReconstructionLoss(nn.Module):
    """Combine Chamfer, confidence, viewpoint, and pose penalties."""

    def __init__(
        self,
        chamfer_weight: float = 1.0,
        confidence_weight: float = 0.0,
        viewpoint_weight: float = 0.0,
        pose_penalty_weight: float = 0.02,
        renderer: Optional["DifferentiableRenderer"] = None,
        pose_up_axis: str = "Y",
        save_point_clouds: bool = True,
        point_cloud_dir_name: str = "point_clouds",
        max_points_per_cloud: int = 4096,
        log_tensorboard: bool = False,
        point_source: str = "vggt",
        confidence_threshold: float = 0.0,
        black_pixel_threshold: float = 0.1,
        pose_outer_radius: float = 4.0,
        pose_inner_radius: float = 2.0,
        pose_floor_margin: float = 1.0,
        default_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # kept for API compatibility, not used internally yet
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
            log_tensorboard=log_tensorboard,
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
        self.default_device = default_device

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
        writer=None,
        step=None,
        train_flag: bool = False,  # kept for backwards compatibility
        point_cloud_dir: Optional[str] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:

        _ = train_flag  # silence "unused" warnings

        device = self.default_device

        total_loss = torch.zeros((), device=device)
        loss_components: Dict[str, float] = {}

        # --- Chamfer ---
        weighted_chamfer, chamfer_raw, confidence_mask = self.chamfer_regularizer(
            recon_data,
            gt_data,
            combined_images_batch,
            combined_camera_poses,
            writer,
            step,
            device,
            point_cloud_dir=point_cloud_dir,
        )
        total_loss = self._add_loss(
            total_loss,
            loss_components,
            "chamfer",
            weighted_chamfer,
            chamfer_raw,
        )

        # --- Confidence ---
        weighted_confidence, confidence_raw = self.confidence_regularizer(
            recon_data,
            device,
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
