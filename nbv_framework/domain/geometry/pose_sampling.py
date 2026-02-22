"""Pose sampling helpers."""

from __future__ import annotations

import torch


def sample_random_positions(
    *,
    batch_size: int,
    device: torch.device,
    loss_fn,
) -> torch.Tensor:
    """Sample random camera positions constrained by loss settings."""
    inner_radius = float(getattr(loss_fn, "pose_inner_radius", 1.5))
    outer_radius = float(getattr(loss_fn, "pose_outer_radius", inner_radius + 1.0))

    floor_margin = float(getattr(loss_fn, "pose_floor_margin", 1.0))
    up_axis = getattr(loss_fn, "pose_up_axis", "Y").upper()
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(up_axis, 1)
    min_height = -floor_margin

    dtype = torch.float32
    positions = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    filled = 0
    attempts = 0
    while filled < batch_size and attempts < 20:
        remaining = batch_size - filled
        sample_count = max(remaining * 2, 4)
        directions = torch.randn(sample_count, 3, device=device, dtype=dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        radii = torch.rand(sample_count, 1, device=device, dtype=dtype)
        radii = radii * (outer_radius - inner_radius) + inner_radius
        samples = directions * radii
        valid_mask = samples[:, axis_index] >= min_height
        valid_samples = samples[valid_mask]
        if valid_samples.numel() == 0:
            attempts += 1
            continue
        take = min(valid_samples.size(0), remaining)
        positions[filled:filled + take] = valid_samples[:take]
        filled += take
        attempts += 1

    if filled < batch_size:
        fallback = torch.randn(batch_size - filled, 3, device=device, dtype=dtype)
        fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        radius = (inner_radius + outer_radius) * 0.5
        fallback = fallback * radius
        fallback[:, axis_index] = torch.clamp(fallback[:, axis_index], min=min_height + 1e-4)
        positions[filled:] = fallback

    return positions
