"""View selection helpers for batch preparation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def select_initial_views(
    initial_images: torch.Tensor,
    camera_poses: torch.Tensor,
    *,
    depth_z: Optional[torch.Tensor] = None,
    randomize: bool,
    min_initial_views: int,
    max_initial_views: int,
    randomize_initial_views: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]:
    """Select a subset of initial views used by policy inference."""
    min_views = max(int(min_initial_views), 1)
    max_views = min(int(max_initial_views), initial_images.shape[1])
    total_views = initial_images.shape[1]

    should_randomize = bool(randomize and randomize_initial_views)
    if should_randomize:
        sampled = torch.randint(
            low=min_views,
            high=max_views + 1,
            size=(1,),
            device=initial_images.device,
        )
        num_views = int(sampled.item())
        perm = torch.randperm(total_views, device=initial_images.device, dtype=torch.long)
    else:
        num_views = max_views
        perm = torch.arange(total_views, device=initial_images.device, dtype=torch.long)

    selection = perm[:num_views]
    selection, _ = torch.sort(selection)

    initial_images = initial_images.index_select(1, selection)
    camera_poses = camera_poses.index_select(1, selection)
    if depth_z is not None:
        depth_z = depth_z.index_select(1, selection)

    return initial_images, camera_poses, depth_z, selection, num_views
