"""Utilities to build point clouds and masks from reconstruction outputs."""

from typing import Dict, Literal, Optional, Tuple

import torch
from pytorch3d.structures import Pointclouds


class PointCloudExtractor:
    """Extract predicted point clouds with configurable masking heuristics.

    This keeps masking logic reusable across different loss terms and makes the
    main reconstruction loss simpler.
    """

    def __init__(self, black_threshold: float = 0.1) -> None:
        self.black_threshold = black_threshold

    def __call__(
        self,
        recon_data: Dict[str, torch.Tensor],
        combined_images_batch: Optional[torch.Tensor],
        confidence_threshold: float,
        source: Literal["vggt", "depth"],
        gt_valid_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[Pointclouds, torch.Tensor]:
        if source == "vggt":
            points_data = recon_data.get("world_points")
            conf_data = recon_data.get("world_points_conf")
            if points_data is None or conf_data is None:
                raise KeyError(
                    "Source 'vggt' selected, but 'world_points' or 'world_points_conf' not found in recon_data."
                )
        elif source == "depth":
            points_data = recon_data.get("world_points_from_depth")
            conf_data = recon_data.get("depth_conf")
            if points_data is None or conf_data is None:
                raise KeyError(
                    "Source 'depth' selected, but 'world_points_from_depth' or 'depth_conf' not found in recon_data."
                )
        else:
            raise ValueError(f"未知的 source: {source}。应为 'vggt' 或 'depth'。")

        if points_data is None or conf_data is None:
            raise ValueError("Point or confidence tensors are missing for point cloud extraction.")

        B, S, H, W, _ = points_data.shape

        with torch.no_grad():
            if confidence_threshold == 0.0:
                conf_threshold_value = 0.0
            else:
                conf_flat = conf_data.reshape(-1)
                conf_threshold_value = torch.quantile(
                    conf_flat, confidence_threshold / 100.0
                )

            high_conf_mask = (conf_data >= conf_threshold_value) & (conf_data > 1e-5)

            if combined_images_batch is not None:
                pixel_intensity = combined_images_batch.mean(dim=2)
                non_black_mask = pixel_intensity > self.black_threshold
                combined_mask = high_conf_mask & non_black_mask
            else:
                combined_mask = high_conf_mask

            if gt_valid_masks is not None:
                if gt_valid_masks.shape != combined_mask.shape:
                    raise ValueError(
                        "gt_valid_masks shape {gt_valid_masks.shape} does not match combined mask "
                        f"shape {combined_mask.shape}"
                    )
                combined_mask = combined_mask & gt_valid_masks

        point_clouds_list = []
        for i in range(B):
            mask_i = combined_mask[i]
            if mask_i.any():
                points_i = points_data[i][mask_i]
                point_clouds_list.append(points_i)
            else:
                point_clouds_list.append(
                    torch.empty((0, 3), device=points_data.device, dtype=points_data.dtype)
                )

        return Pointclouds(points=point_clouds_list), combined_mask


__all__ = ["PointCloudExtractor"]
