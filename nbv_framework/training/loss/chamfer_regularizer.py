"""Chamfer regularizer component."""

from typing import Dict, List, Optional, Tuple, Any

import logging
import torch
import torch.nn.utils.rnn as rnn_utils
import numpy as np

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

    def _prepare_gt_points(self, raw_gt_points: List[Any], device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        """
        🚀 性能优化核心：
        在 CPU 上完成 Tensor 转换和 Padding，一次性传输到 GPU，然后再切分。
        避免在循环中调用 .to(device) 或 torch.tensor(..., device=device)。
        """
        # 1. 快速检查：如果已经是 GPU Tensor 列表，直接返回
        if len(raw_gt_points) > 0 and isinstance(raw_gt_points[0], torch.Tensor):
            if raw_gt_points[0].device == device:
                return raw_gt_points # type: ignore

        # 2. 在 CPU 上转为 Tensor List
        cpu_tensors = []
        for p in raw_gt_points:
            if isinstance(p, torch.Tensor):
                cpu_tensors.append(p.detach().cpu())
            elif isinstance(p, np.ndarray):
                # copy=True 确保内存连续，避免后续警告
                cpu_tensors.append(torch.from_numpy(p.copy()))
            else:
                cpu_tensors.append(torch.tensor(p))

        # 3. 记录原始长度
        lengths = [t.shape[0] for t in cpu_tensors]

        # 4. CPU 上 Pad 成一个大 Tensor (B, MaxPoints, 3)
        # 这一步纯 CPU 操作，很快
        if len(cpu_tensors) == 0:
            return []

        padded_cpu = rnn_utils.pad_sequence(cpu_tensors, batch_first=True, padding_value=0.0)

        # 5. 🚀 关键步骤：一次性传输到 GPU
        # non_blocking=True 允许 CPU 继续往下跑，只要后续操作不立即读回 CPU
        padded_gpu = padded_cpu.to(device=device, dtype=dtype, non_blocking=True)

        # 6. GPU 上切分回 List (View 操作，几乎无开销)
        gt_points_list = []
        for i, length in enumerate(lengths):
            # narrow 返回的是 view，不分配新内存
            gt_points_list.append(padded_gpu[i].narrow(0, 0, length))

        return gt_points_list

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

        # Extractor 返回 List[Tensor]
        pred_points_list, correspondence_mask = self.extractor(
            recon_data=recon_data,
            combined_images_batch=combined_images_batch,
            confidence_threshold=self.confidence_threshold,
            source=self.point_source,
            gt_valid_masks=gt_valid_masks,
        )

        # 🚀 使用优化后的批量传输
        raw_gt_points = gt_data["gt_points"]
        gt_points_list = self._prepare_gt_points(raw_gt_points, device, dtype)

        # 构建 correspondence points
        # 优化: 尽量使用 Tensor 操作而非 Python 列表循环，但在 List[Tensor] 结构下
        # 循环是不可避免的，但我们可以减少循环内的操作
        correspondence_points: List[torch.Tensor] = []

        # gt_point_maps 假设已经是 (B, H, W, 3) 且在 GPU 上
        # correspondence_mask 是 (B, H, W)
        # 这里使用 boolean masking，GPU 上非常快
        for i in range(len(pred_points_list)):
            mask_i = correspondence_mask[i]
            if mask_i.any():
                gt_points_i = gt_point_maps[i][mask_i]
            else:
                gt_points_i = torch.empty((0, 3), device=device, dtype=dtype)
            correspondence_points.append(gt_points_i)

        if len(pred_points_list) != len(gt_points_list):
            logging.warning(
                f"Batch size mismatch in Chamfer loss: Pred {len(pred_points_list)} vs GT {len(gt_points_list)}. Skipping."
            )
            return zero, zero, correspondence_mask

        chamfer_loss_value = self.chamfer(
            pred_points_list,
            gt_points_list,
            correspondence_points=correspondence_points,
            writer=writer,
            step=step,
            point_cloud_dir=point_cloud_dir,
        )

        weighted_loss = self.weight * chamfer_loss_value
        return weighted_loss, chamfer_loss_value, correspondence_mask