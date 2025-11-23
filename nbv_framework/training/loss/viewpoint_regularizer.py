"""Viewpoint regularization helpers."""

from typing import Optional, Tuple

import torch

from .viewpoint import ViewpointLoss


class ViewpointRegularizer:
    """Thin wrapper to keep the main reconstruction loss focused."""

    def __init__(self, weight: float = 0.0) -> None:
        self.weight = weight
        self.viewpoint_loss = ViewpointLoss()

    def __call__(
        self,
        combined_images_batch: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), device=device, dtype=dtype)
        if self.weight <= 0 or combined_images_batch is None:
            return zero, zero

        new_images = combined_images_batch[:, -1, :, :, :]
        viewpoint_loss_value = self.viewpoint_loss(new_images)
        weighted_loss = self.weight * viewpoint_loss_value
        return weighted_loss, viewpoint_loss_value


__all__ = ["ViewpointRegularizer"]
