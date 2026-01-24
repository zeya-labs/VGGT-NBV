from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from mapanything.utils.geometry import apply_log_to_norm
import trimesh

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
dcd_dir = os.path.join(current_dir, "Density_aware_Chamfer_Distance")
if dcd_dir not in sys.path:
    sys.path.append(dcd_dir)
from utils_v2.model_utils import calc_dcd, calc_cd, calc_emd

MetricType = Literal["cd", "emd", "dcd"]
CDVariant = Literal["cd_p", "cd_t"]

class ChamferDistance(nn.Module):
    """Compute point cloud distance (cd/emd/dcd) with optional downsampling."""
    def __init__(
        self,
        max_points_per_cloud: int = 32768,
        save_point_clouds: bool = False,
        point_cloud_dir_name: str = "point_clouds",
        use_log_warp: bool = False, # 是否开启对对数压缩
        distance_type: MetricType = "emd",
        cd_variant: CDVariant = "cd_t",
        dcd_alpha: float = 40.0,
        dcd_n_lambda: float = 0.5,
        dcd_non_reg: bool = False,
        emd_eps: float = 0.005,
        emd_iterations: int = 50,
    ) -> None:
        super().__init__()
        self.max_points_per_cloud = max_points_per_cloud
        self.save_point_clouds: bool = save_point_clouds
        self.point_cloud_dir_name: str = point_cloud_dir_name
        self.use_log_warp: bool = use_log_warp
        self.distance_type: MetricType = distance_type
        self.cd_variant: CDVariant = cd_variant
        self.dcd_alpha = dcd_alpha
        self.dcd_n_lambda = dcd_n_lambda
        self.dcd_non_reg = dcd_non_reg
        self.emd_eps = emd_eps
        self.emd_iterations = emd_iterations
        
    def configure_distance(
        self,
        distance_type: Optional[MetricType] = None,
    ) -> None:
        if distance_type is not None:
            self.distance_type = distance_type
            
    @staticmethod
    def _sample_points(points: torch.Tensor, target: int) -> torch.Tensor:
        if points.shape[0] <= target:
            return points
        idx = torch.randperm(points.shape[0], device=points.device)[:target]
        return points.index_select(0, idx)

    def _can_compute_batched(
        self,
        pred_batched: torch.Tensor,
        gt_batched: torch.Tensor,
        pred_lengths: torch.Tensor,
        gt_lengths: torch.Tensor,
    ) -> bool:
        # 0. 空数据防御
        if pred_lengths.numel() == 0 or gt_lengths.numel() == 0:
            return False

        # 1. 检查长度一致性 (利用 unique)
        p_unique = pred_lengths.unique()
        g_unique = gt_lengths.unique()

        # 如果 unique 后的元素数量不为 1，说明 batch 内长度参差不齐，含 padding
        if p_unique.numel() != 1 or g_unique.numel() != 1:
            return False

        # 提取统一后的长度数值
        p_len, g_len = p_unique.item(), g_unique.item()

        # 2. 检查是否存在 Padding (有效长度必须等于 Tensor 的物理维度)
        # 如果由 rnn_utils.pad_sequence 产生，物理维度是最大长度。
        # 这里要求物理维度必须等于当前有效长度，才算完全没 padding。
        if p_len != pred_batched.shape[1] or g_len != gt_batched.shape[1]:
            return False

        # 3. EMD 特殊限制: 预测点数必须等于真值点数
        if self.distance_type == "emd" and p_len != g_len:
            return False

        return True

    def _compute_distance_batched(
        self, pred_batched: torch.Tensor, gt_batched: torch.Tensor
    ) -> torch.Tensor:
        pred_batched = pred_batched.contiguous().float()
        gt_batched = gt_batched.contiguous().float()
        if self.distance_type == "cd":
            cd_p, cd_t = calc_cd(pred_batched, gt_batched)
            return cd_p if self.cd_variant == "cd_p" else cd_t
        if self.distance_type == "emd":
            return calc_emd(pred_batched, gt_batched, eps=self.emd_eps, iterations=self.emd_iterations)
        if self.distance_type == "dcd":
            return calc_dcd(
                pred_batched,
                gt_batched,
                alpha=self.dcd_alpha,
                n_lambda=self.dcd_n_lambda,
                non_reg=self.dcd_non_reg,
            )[0]
        raise ValueError(f"Unknown distance_type: {self.distance_type}")

    def _compute_distance(
        self,
        pred_batched: torch.Tensor,
        gt_batched: torch.Tensor,
        pred_lengths: torch.Tensor,
        gt_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self._can_compute_batched(pred_batched, gt_batched, pred_lengths, gt_lengths):
            return self._compute_distance_batched(pred_batched, gt_batched)

        losses: List[torch.Tensor] = []
        for i in range(pred_batched.shape[0]):
            p_len = int(pred_lengths[i].item())
            g_len = int(gt_lengths[i].item())
            pred_i = pred_batched[i, :p_len]
            gt_i = gt_batched[i, :g_len]
            if self.distance_type == "emd" and p_len != g_len:
                target = min(p_len, g_len)
                pred_i = self._sample_points(pred_i, target)
                gt_i = self._sample_points(gt_i, target)
            loss_i = self._compute_distance_batched(pred_i.unsqueeze(0), gt_i.unsqueeze(0))
            losses.append(loss_i.squeeze(0))
        return torch.stack(losses, dim=0)

    def _to_batched(
        self, data: Union[List[torch.Tensor], torch.Tensor], *, downsample: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将输入统一为 (B, N, 3) 和长度张量。
        若需要，按 max_points_per_cloud 随机下采样。
        
        返回：
            batched: (B, N, 3) 张量，包含所有点云。
            lengths: (B,) 张量，记录每个点云的有效点数。
        """
        if torch.is_tensor(data):
            assert data.ndim == 3, f"Expected tensor with shape [B, N, 3], got {tuple(data.shape)}"
            lengths = torch.full(
                (data.shape[0],),
                data.shape[1],
                device=data.device,
                dtype=torch.long,
            )
            batched = data
        else:
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
        point_cloud_dir: Optional[str] = None,
    ) -> torch.Tensor:
        pred_batched, pred_lengths = self._to_batched(pred, downsample=True)
        gt_batched, gt_lengths = self._to_batched(gt, downsample=False)

        assert pred_batched.shape[0] == gt_batched.shape[0], (
            f"Batch size mismatch in Chamfer loss: Pred {pred_batched.shape[0]} vs GT {gt_batched.shape[0]}."
        )

        if self.use_log_warp:
            pred_batched = apply_log_to_norm(pred_batched)
            gt_batched = apply_log_to_norm(gt_batched)

        loss_per_batch = self._compute_distance(pred_batched, gt_batched, pred_lengths, gt_lengths)
        loss = loss_per_batch.mean()

        if point_cloud_dir is not None:
            self._save_point_clouds(
                pred_batched.detach(),
                pred_lengths,
                gt_batched.detach(),
                gt_lengths,
                os.path.join(point_cloud_dir, self.point_cloud_dir_name),
            )

        return loss

    def _save_point_clouds(
        self,
        pred: torch.Tensor,
        pred_lengths: torch.Tensor,
        gt: torch.Tensor,
        gt_lengths: torch.Tensor,
        base_dir: str,
    ) -> None:
        os.makedirs(base_dir, exist_ok=True)
        batch_size = pred.shape[0]
        red = torch.tensor([255, 0, 0, 255], dtype=torch.uint8, device=pred.device)
        green = torch.tensor([0, 255, 0, 255], dtype=torch.uint8, device=pred.device)

        for idx in range(batch_size):
            p_len = int(pred_lengths[idx].item()) if pred_lengths.numel() > idx else 0
            g_len = int(gt_lengths[idx].item()) if gt_lengths.numel() > idx else 0
            if p_len == 0 and g_len == 0:
                continue

            pcs: List[torch.Tensor] = []
            colors: List[torch.Tensor] = []

            if p_len > 0:
                pcs.append(pred[idx, :p_len])
                colors.append(red.unsqueeze(0).repeat(p_len, 1))
            if g_len > 0:
                pcs.append(gt[idx, :g_len])
                colors.append(green.unsqueeze(0).repeat(g_len, 1))

            all_pts = torch.cat(pcs, dim=0).cpu()
            all_cols = torch.cat(colors, dim=0).cpu()

            file_path = os.path.join(base_dir, f"cloud_{idx:03d}.glb")
            self._write_colored_glb(file_path, all_pts, all_cols)

    def _write_colored_glb(self, path: str, points: torch.Tensor, colors: torch.Tensor) -> None:
        if points.shape[0] == 0:
            return
        pts_np = points.detach().cpu().numpy().astype("float32")
        cols_np = colors.detach().cpu().numpy().astype("uint8")
        cloud = trimesh.points.PointCloud(vertices=pts_np, colors=cols_np)
        cloud.export(path, file_type="glb")
