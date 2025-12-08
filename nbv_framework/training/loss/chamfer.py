"""Chamfer distance loss helpers."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pytorch3d.loss import chamfer_distance
from mapanything.utils.geometry import apply_log_to_norm

class ChamferDistance(nn.Module):
    """Compute Chamfer distance with optional downsampling."""

    def __init__(
        self,
        *,
        max_points_per_cloud: int = 32768,
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
        if max_points_per_cloud is not None:
            self.max_points_per_cloud = max_points_per_cloud
        if enable_save is not None:
            self.save_point_clouds = bool(enable_save)
        if subdir_name is not None:
            self.point_cloud_subdir = subdir_name
        if log_to_tensorboard is not None:
            self.log_tensorboard = bool(log_to_tensorboard)

    def _to_batched(
        self, data: Union[List[torch.Tensor], torch.Tensor], *, downsample: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将输入统一为 (B, N, 3) 和长度张量。
        若需要，按 max_points_per_cloud 下采样。
        """
        if torch.is_tensor(data):
            if data.ndim == 2:
                data = data.unsqueeze(0)
            if data.ndim != 3:
                raise ValueError(f"Expected tensor with shape [B, N, 3], got {tuple(data.shape)}")
            lengths = torch.full(
                (data.shape[0],),
                data.shape[1],
                device=data.device,
                dtype=torch.long,
            )
            batched = data
        else:
            if len(data) == 0:
                return torch.empty((0, 0, 3)), torch.zeros((), dtype=torch.long)
            lengths = torch.tensor([p.shape[0] for p in data], device=data[0].device, dtype=torch.long)
            batched = rnn_utils.pad_sequence(data, batch_first=True)

        if downsample and self.max_points_per_cloud > 0 and lengths.numel() > 0:
            new_pts: List[torch.Tensor] = []
            new_lengths: List[int] = []
            for i in range(batched.shape[0]):
                pts = batched[i, : lengths[i]]
                if pts.numel() == 0:
                    new_pts.append(pts)
                    new_lengths.append(0)
                    continue
                target = min(self.max_points_per_cloud, pts.shape[0])
                if target < pts.shape[0]:
                    idx = torch.randperm(pts.shape[0], device=pts.device)[:target]
                    pts = pts.index_select(0, idx)
                new_pts.append(pts)
                new_lengths.append(pts.shape[0])
            lengths = torch.tensor(new_lengths, device=batched.device, dtype=torch.long)
            batched = rnn_utils.pad_sequence(new_pts, batch_first=True)

        return batched, lengths

    def forward(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        gt: Union[List[torch.Tensor], torch.Tensor],
        writer=None,
        step=None,
        point_cloud_dir: Optional[str] = None,
    ) -> torch.Tensor:
        pred_batched, pred_lengths = self._to_batched(pred, downsample=True)
        gt_batched, gt_lengths = self._to_batched(gt, downsample=False)

        if pred_lengths.numel() == 0 or gt_lengths.numel() == 0:
            device = pred_batched.device if torch.is_tensor(pred_batched) else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=True)

        if pred_batched.shape[0] != gt_batched.shape[0]:
            raise ValueError(
                f"Batch size mismatch in Chamfer loss: Pred {pred_batched.shape[0]} vs GT {gt_batched.shape[0]}."
            )

        if self.use_log_warp:
            pred_batched = apply_log_to_norm(pred_batched)
            gt_batched = apply_log_to_norm(gt_batched)
        print("pred_batched shape:", pred_batched.shape)
        print("gt_batched shape:", gt_batched.shape)
        loss, _ = chamfer_distance(
            pred_batched,
            gt_batched,
            x_lengths=pred_lengths,
            y_lengths=gt_lengths,
        )

        return loss

    def _resolve_point_cloud_directory(self, base_dir: Optional[str]) -> Optional[str]:
        if not self.save_point_clouds or base_dir is None:
            return None
        return os.path.join(base_dir, self.point_cloud_subdir)
