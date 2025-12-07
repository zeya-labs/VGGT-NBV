"""Utilities to build point clouds and masks from reconstruction outputs."""

from typing import Dict, Literal, Optional, Tuple, List
import torch

class PointCloudExtractor:
    """Extract predicted point clouds with configurable masking heuristics."""

    def __init__(self, black_threshold: float = 0.1) -> None:
        self.black_threshold = black_threshold

    def __call__(
        self,
        recon_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        confidence_threshold: float,
        source: Literal["vggt", "depth"],
        gt_valid_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:

        # 1. 提取数据引用
        if source == "vggt":
            points_data = recon_data.get("world_points")
            conf_data = recon_data.get("world_points_conf")
        elif source == "depth":
            points_data = recon_data.get("world_points_from_depth")
            conf_data = recon_data.get("depth_conf")
        else:
            raise ValueError(f"Unknown source: {source}")

        if points_data is None or conf_data is None:
            raise KeyError(f"Missing data for source {source}")

        # 2. 形状标准化
        # points_data 可能形状为 (B, S, H, W, 3) 或 (B, H, W, 3)
        # 我们统一展平为 (B, N, 3) 以便进行批处理 Mask 计算
        B = points_data.shape[0]
        flat_points = points_data.view(B, -1, 3)
        flat_conf = conf_data.view(B, -1) # (B, N)

        # 3. 计算 Mask (全 Batch 向量化操作)
        with torch.no_grad():
            if confidence_threshold > 0.0:
                # 优化: 在 GPU 上对整个 Batch 计算分位数可能比逐个样本快
                # 注意: 如果需要严格的单样本分位数，这里还是需要 loop，但通常全局统计或固定阈值足够
                # 这里保持简单的高效逻辑：
                mask = (flat_conf > 1e-5)
                if confidence_threshold > 0.1: # 只有非微小阈值才计算 quantile
                    # 为了速度，这里简化为绝对阈值判断，或者你可以用 topk 代替 quantile
                    thresh_val = torch.quantile(flat_conf, confidence_threshold / 100.0)
                    mask = mask & (flat_conf >= thresh_val)
            else:
                mask = flat_conf > 1e-5

            if combined_images_batch is not None:
                # 假设 images 是 (B, S, C, H, W) 或 (B, C, H, W)，需要对齐
                # 简单起见，假设调用方保证了形状对齐，直接 view
                # 计算像素强度: mean over channels
                intensity = combined_images_batch.flatten(start_dim=1).mean(dim=-1) # 这取决于 images 的具体布局
                # 由于 image shape 比较多变，这里保留一个防御性 reshape，实际需根据 recon_data 调整
                # 假设 conf_data 和 image 像素一一对应
                if combined_images_batch.numel() // combined_images_batch.shape[-1] == flat_conf.numel():
                     # 尝试对齐
                     pixel_intensity = combined_images_batch.view(B, -1, combined_images_batch.shape[-1]).mean(dim=2)
                     mask = mask & (pixel_intensity > self.black_threshold)

            if gt_valid_masks is not None:
                mask = mask & gt_valid_masks.view(B, -1)

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