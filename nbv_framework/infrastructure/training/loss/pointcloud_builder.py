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
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:        # 1. 提取数据引用
        if source == "vggt":
            points_data = recon_data.get("world_points")
            conf_data = recon_data.get("world_points_conf")
        elif source == "depth":
            points_data = recon_data.get("world_points_from_depth")
            conf_data = recon_data.get("depth_conf")
        else:
            raise ValueError(f"Unknown source: {source}")
        # save_image(combined_images_batch.view(-1, 3, 518, 518),"combined_images_batch.png",nrow=3)
        # save_image(conf_data.view(-1, 518, 518).unsqueeze(1).repeat(1,3,1,1),"conf_data.png",nrow=3)
        # save_image(gt_valid_masks.view(-1, 1, 518, 518).repeat(1,3,1,1).float(),"gt_valid_masks.png",nrow=3)
        # save_image(points_data.view(-1, 518, 518, 3).permute(0, 3, 1, 2),"points_data_z.png",nrow=3, normalize=True)
        if points_data is None or conf_data is None:
            raise KeyError(f"Missing data for source {source}")

        # 2. 形状标准化
        # points_data 形状为 (B, S, H, W, 3) 
        # 我们统一展平为 (B, N, 3) 以便进行批处理 Mask 计算
        B = points_data.shape[0]
        flat_points = points_data.view(B, -1, 3)
        flat_conf = conf_data.view(B, -1) # (B, N)

        # 3. 计算 Mask
        if confidence_threshold > 0.1:
            thresh_val = torch.quantile(flat_conf, confidence_threshold / 100.0)
            mask = (flat_conf >= thresh_val)
        else:
            mask = flat_conf > 1e-5

        # if combined_images_batch is not None:
        #     # 如果明确知道输入是 (B, S, C, H, W)
        #     # 先在 Channel 维度 (dim=2) 求平均，得到 (B, S, H, W)
        #     intensity_map = combined_images_batch.mean(dim=2)
        #     # 再展平以匹配 flat_conf
        #     pixel_intensity = intensity_map.view(B, -1)
        #     # 然后进行掩码计算
        #     mask = mask & (pixel_intensity > self.black_threshold)

        # if gt_valid_masks is not None:
        #     mask = mask & gt_valid_masks.view(B, -1)

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
        # save_image(reshaped_mask.view(-1, 1, 518, 518).repeat(1,3,1,1).float(),"reshaped_mask.png",nrow=3)

        return point_clouds_list, reshaped_mask