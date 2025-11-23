"""Confidence regularization helper."""

from typing import Dict, Optional, Tuple

import torch


class ConfidenceRegularizer:
    """Encourages predicted point confidences to stay high."""

    def __init__(self, weight: float = 0.0) -> None:
        self.weight = weight

    def __call__(
        self,
        recon_data: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        confidence_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), device=device, dtype=dtype)
        if self.weight <= 0:
            return zero, zero

        world_points_conf = recon_data.get("world_points_conf")
        if world_points_conf is None:
            return zero, zero

        if confidence_mask is not None:
            if confidence_mask.shape != world_points_conf.shape:
                raise ValueError(
                    "confidence_mask shape {confidence_mask.shape} does not match "
                    f"world_points_conf shape {world_points_conf.shape}"
                )
            mask = confidence_mask.to(
                device=world_points_conf.device, dtype=world_points_conf.dtype
            )
            valid_count = mask.sum()
            if valid_count.item() > 0:
                masked_mean = (world_points_conf * mask).sum() / valid_count
            else:
                masked_mean = world_points_conf.new_tensor(1.0)
            loss = -torch.log(masked_mean + 1e-8)
        else:
            loss = -torch.log(world_points_conf.mean() + 1e-8)

        weighted_loss = self.weight * loss
        return weighted_loss, loss


__all__ = ["ConfidenceRegularizer"]
