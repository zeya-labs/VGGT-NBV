"""Utilities to build point clouds and masks from reconstruction outputs."""

from typing import List, Optional, Tuple
import torch
from nbv_framework.reconstruction import ReconstructionData

class PointCloudExtractor:
    """Extract predicted point clouds with confidence + validity masking."""

    def __init__(self, black_threshold: float = 0.1) -> None:
        self.black_threshold = black_threshold

    def __call__(
        self,
        recon_data: ReconstructionData,
        combined_images_batch: Optional[torch.Tensor],
        confidence_threshold: float,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        points_data = recon_data.recon_world_points
        conf_data = recon_data.recon_conf
        valid_mask = recon_data.recon_mask

        # 2. 形状标准化
        # points_data 形状为 (B, S, H, W, 3)
        # 我们统一展平为 (B, N, 3) 以便进行批处理 Mask 计算
        B = points_data.shape[0]
        flat_points = points_data.view(B, -1, 3)
        flat_conf = conf_data.view(B, -1)  # (B, N)
        flat_valid = valid_mask.view(B, -1)

        # 3. 计算 Mask
        if confidence_threshold > 0.1:
            thresh_val = torch.quantile(flat_conf, confidence_threshold / 100.0)
            mask = flat_conf >= thresh_val
        else:
            mask = flat_conf > 1e-5

        if combined_images_batch is not None:
            intensity_map = combined_images_batch.mean(dim=2)
            pixel_intensity = intensity_map.view(B, -1)
            mask = mask & (pixel_intensity > self.black_threshold)

        mask = mask & flat_valid

        # 4. 提取点云 (List Construction)
        # 这一步无法完全避免 Python 循环，因为输出是不定长的 List
        point_clouds_list = []
        for i in range(B):
            mask_i = mask[i]
            if mask_i.any():
                # Boolean masking 触发一次 GPU 拷贝
                point_clouds_list.append(flat_points[i][mask_i])
            else:
                # 创建空 Tensor，确保设备和类型正确
                point_clouds_list.append(
                    torch.empty((0, 3), device=flat_points.device, dtype=flat_points.dtype)
                )

        # 恢复 mask 形状以便返回 (B, ...)
        reshaped_mask = mask.view(conf_data.shape)

        return point_clouds_list, reshaped_mask
