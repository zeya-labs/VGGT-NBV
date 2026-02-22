"""Artifact writing helpers for train/eval diagnostics."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torchvision
from loguru import logger


def save_pre_images_grid(
    *,
    initial_images: torch.Tensor,
    new_images: torch.Tensor,
    step_output_dir: Optional[str],
    is_global_zero: bool,
) -> None:
    if not is_global_zero or step_output_dir is None:
        return
    if initial_images.ndim != 5 or new_images.ndim != 4:
        logger.warning(
            "Skip pre_images grid due to invalid tensor shapes: initial={}, new={}",
            tuple(initial_images.shape),
            tuple(new_images.shape),
        )
        return

    batch_size, num_views, channels, height, width = initial_images.shape
    if new_images.shape[0] != batch_size:
        logger.warning("Skip pre_images grid due to batch mismatch")
        return
    if tuple(new_images.shape[1:]) != (channels, height, width):
        logger.warning("Skip pre_images grid due to image-shape mismatch")
        return

    initial_cpu = initial_images.detach().float().cpu().clamp(0.0, 1.0)
    new_cpu = new_images.detach().float().cpu().clamp(0.0, 1.0)

    images_for_grid = []
    for sample_idx in range(batch_size):
        for view_idx in range(num_views):
            images_for_grid.append(initial_cpu[sample_idx, view_idx])
        images_for_grid.append(new_cpu[sample_idx])

    grid_tensor = torch.stack(images_for_grid, dim=0)
    grid = torchvision.utils.make_grid(grid_tensor, nrow=num_views + 1, padding=2)

    pre_images_dir = os.path.join(step_output_dir, "pre_images")
    os.makedirs(pre_images_dir, exist_ok=True)
    save_path = os.path.join(pre_images_dir, "pre_images.png")
    torchvision.utils.save_image(grid, save_path)
