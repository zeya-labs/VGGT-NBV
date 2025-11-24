"""Chamfer distance loss helpers."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_farthest_points

from nbv_framework.utils.logging_utils import get_logger

from ...utils.tensorboard_mesh import log_point_clouds_to_tensorboard
from mapanything.utils.geometry import apply_log_to_norm

LOGGER = get_logger(__name__)


class ChamferDistance(nn.Module):
    """Compute an aligned Chamfer distance with optional TensorBoard logging."""

    def __init__(
        self,
        *,
        max_points_per_cloud: int = 4096,
        use_log_warp: bool = False,
    ) -> None:
        super().__init__()
        self.max_points_per_cloud = max_points_per_cloud
        self.save_point_clouds: bool = False
        self.point_cloud_subdir: str = "point_clouds"
        self.log_tensorboard: bool = True
        self.use_log_warp: bool = use_log_warp

    def configure_point_cloud_logging(
        self,
        *,
        max_points_per_cloud: Optional[int] = None,
        enable_save: Optional[bool] = None,
        subdir_name: Optional[str] = None,
        log_to_tensorboard: Optional[bool] = None,
    ) -> None:
        """Adjust TensorBoard logging behaviour."""
        if max_points_per_cloud is not None:
            self.max_points_per_cloud = max_points_per_cloud
        if enable_save is not None:
            self.save_point_clouds = bool(enable_save)
        if subdir_name is not None:
            self.point_cloud_subdir = subdir_name
        if log_to_tensorboard is not None:
            self.log_tensorboard = bool(log_to_tensorboard)

    def _umeyama_alignment(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source and target must be rank-2 tensors shaped [N, 3].")
        if source.shape[0] != target.shape[0]:
            raise ValueError(
                "source and target must contain the same number of points; got "
                f"{source.shape[0]} and {target.shape[0]}."
            )

        device = source.device
        dtype = source.dtype
        n_points = source.shape[0]

        if n_points < 3 or target.shape[0] < 3:
            rotation = torch.eye(3, device=device, dtype=dtype)
            translation = torch.zeros(3, device=device, dtype=dtype)
            scale = torch.tensor(1.0, device=device, dtype=dtype)
            return scale, rotation, translation

        source64 = source.to(dtype=torch.float64)
        target64 = target.to(dtype=torch.float64)

        mu_x = source64.mean(dim=0)
        mu_y = target64.mean(dim=0)
        X = source64 - mu_x
        Y = target64 - mu_y

        cov = (Y.T @ X) / n_points

        U, S, Vh = torch.linalg.svd(cov)

        d = torch.ones(3, device=device, dtype=torch.float64)
        if torch.det(U @ Vh) < 0:
            d[-1] = -1

        D = torch.diag(d)
        rotation = U @ D @ Vh

        var_x = torch.clamp((X ** 2).sum() / n_points, min=1e-8)
        scale = torch.sum(S * d) / var_x

        translation = mu_y - scale * (rotation @ mu_x)

        return (
            scale.to(dtype=dtype),
            rotation.to(dtype=dtype),
            translation.to(dtype=dtype),
        )

    @staticmethod
    def _apply_similarity_transform(
        points: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        if points.numel() == 0:
            return points
        return scale * (points @ rotation.transpose(0, 1)) + translation

    @staticmethod
    def _maybe_log_warp(points: torch.Tensor) -> torch.Tensor:
        if points.numel() == 0:
            return points
        return apply_log_to_norm(points)

    def forward(
        self,
        p_pred: Pointclouds,
        p_gt: Pointclouds,
        correspondence_points: Optional[List[torch.Tensor]] = None,
        writer=None,
        step=None,
        point_cloud_dir: Optional[str] = None,
    ) -> torch.Tensor:
        if correspondence_points is None:
            raise ValueError("correspondence_points must be provided for Umeyama alignment.")

        pred_list = [p.to(dtype=torch.float32) for p in p_pred.points_list()]
        gt_list = [p.to(dtype=torch.float32) for p in p_gt.points_list()]
        corr_list = [cp.to(dtype=torch.float32) for cp in correspondence_points]

        aligned_points_list: List[torch.Tensor] = []

        for pred_points, corr_points in zip(pred_list, corr_list):
            pred_points_f32 = pred_points.float()
            corr_points_f32 = corr_points.float()

            if corr_points_f32.numel() >= 3 and pred_points_f32.numel() >= 3:
                scale, rotation, translation = self._umeyama_alignment(
                    pred_points_f32, corr_points_f32
                )
                aligned = self._apply_similarity_transform(
                    pred_points_f32, scale, rotation, translation
                )
            else:
                aligned = pred_points_f32
            aligned_points_list.append(aligned)

        target_fps_points = 32768
        aligned_pointclouds = Pointclouds(points=aligned_points_list)
        aligned_lengths = aligned_pointclouds.num_points_per_cloud()

        if (aligned_lengths < target_fps_points).any():
            target_fps_points = int(aligned_lengths.min().item())
            LOGGER.warning(
                "Farthest point sampling requires >= %d points per cloud. Minimum length=%d",
                target_fps_points,
                int(aligned_lengths.min().item()),
            )

        padded_aligned = aligned_pointclouds.points_padded()
        sampled_points, _ = sample_farthest_points(
            padded_aligned, lengths=aligned_lengths, K=target_fps_points
        )
        fps_aligned_points_list = [sampled_points[i] for i in range(sampled_points.shape[0])]

        if self.use_log_warp:
            fps_aligned_points_list = [self._maybe_log_warp(pts) for pts in fps_aligned_points_list]
            gt_list = [self._maybe_log_warp(pts) for pts in gt_list]

        p_pred_aligned = Pointclouds(points=fps_aligned_points_list)

        
        # point_counts = [int(points.shape[0]) for points in p_pred_aligned.points_list()]
        # LOGGER.info("Predicted point counts per batch: %s", point_counts)

        p_gt_float = Pointclouds(points=gt_list)

        should_log_tb = self.log_tensorboard and writer is not None and step is not None
        save_directory = self._resolve_point_cloud_directory(point_cloud_dir)
        should_export_glb = save_directory is not None
        if should_log_tb or should_export_glb:
            if len(aligned_points_list) > 0 and len(gt_list) > 0:
                point_cloud_specs: List[Tuple[str, Pointclouds, np.ndarray]] = [
                    (
                        "predicted",
                        Pointclouds(points=pred_list),
                        np.array([0, 0, 255], dtype=np.uint8),
                    ),
                    (
                        "ground_truth",
                        Pointclouds(points=gt_list),
                        np.array([0, 255, 0], dtype=np.uint8),
                    ),
                    (
                        "aligned_predicted",
                        Pointclouds(points=fps_aligned_points_list),
                        np.array([255, 0, 0], dtype=np.uint8),
                    ),
                ]
                if len(corr_list) > 0 and corr_list[0].numel() > 0:
                    point_cloud_specs.append(
                        (
                            "correspondence",
                            Pointclouds(points=[corr_list[0]]),
                            np.array([255, 0, 255], dtype=np.uint8),
                        )
                    )
                target_batch = 0
                glb_path = None
                if should_export_glb:
                    os.makedirs(save_directory, exist_ok=True)
                    glb_path = os.path.join(
                        save_directory, f"batch_{target_batch:03d}_comparison.glb"
                    )
                if should_log_tb or glb_path is not None:
                    log_point_clouds_to_tensorboard(
                        writer if should_log_tb else None,
                        tag="Chamfer/Comparison",
                        point_cloud_specs=point_cloud_specs,
                        step=step if step is not None else 0,
                        batch_index=target_batch,
                        max_points_per_cloud=self.max_points_per_cloud,
                        glb_output_path=glb_path,
                    )

        loss, _ = chamfer_distance(p_pred_aligned, p_gt_float)
        return loss

    def _resolve_point_cloud_directory(self, base_dir: Optional[str]) -> Optional[str]:
        if not self.save_point_clouds or base_dir is None:
            return None
        return os.path.join(base_dir, self.point_cloud_subdir)


__all__ = ["ChamferDistance"]
