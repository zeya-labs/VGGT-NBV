"""Chamfer distance loss helpers."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds

from ...utils.tensorboard_mesh import log_point_clouds_to_tensorboard


class ChamferDistance(nn.Module):
    """Compute an aligned Chamfer distance with optional TensorBoard logging."""

    def __init__(self) -> None:
        super().__init__()

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

    def forward(
        self,
        p_pred: Pointclouds,
        p_gt: Pointclouds,
        correspondence_points: Optional[List[torch.Tensor]] = None,
        writer=None,
        step=None,
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

        p_pred_aligned = Pointclouds(points=aligned_points_list)
        p_gt_float = Pointclouds(points=gt_list)

        if writer is not None and step is not None:
            if len(aligned_points_list) > 0 and len(gt_list) > 0:
                point_cloud_specs: List[Tuple[Pointclouds, np.ndarray]] = [
                    (
                        Pointclouds(points=pred_list),
                        np.array([0, 0, 255], dtype=np.uint8),
                    ),
                    (
                        Pointclouds(points=gt_list),
                        np.array([0, 255, 0], dtype=np.uint8),
                    ),
                    (
                        Pointclouds(points=aligned_points_list),
                        np.array([255, 0, 0], dtype=np.uint8),
                    ),
                ]
                if len(corr_list) > 0 and corr_list[0].numel() > 0:
                    point_cloud_specs.append(
                        (
                            Pointclouds(points=[corr_list[0]]),
                            np.array([255, 255, 0], dtype=np.uint8),
                        )
                    )
                log_point_clouds_to_tensorboard(
                    writer,
                    tag="Chamfer/Comparison",
                    point_cloud_specs=point_cloud_specs,
                    step=step,
                    batch_index=0,
                    max_points_per_cloud=4096,
                )

        loss, _ = chamfer_distance(p_pred_aligned, p_gt_float)
        return loss

__all__ = ["ChamferDistance"]
