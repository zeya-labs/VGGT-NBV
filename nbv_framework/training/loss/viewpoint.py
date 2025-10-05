"""Viewpoint quality regularisation losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewpointLoss(nn.Module):
    """Penalise degenerate viewpoints (black, low variance, or low detail)."""

    def __init__(
        self,
        black_screen_threshold: float = 0.5,
        low_variance_threshold: float = 0.05,
        edge_density_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.black_screen_threshold = black_screen_threshold
        self.low_variance_threshold = low_variance_threshold
        self.edge_density_threshold = edge_density_threshold

        sobel_x_kernel = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y_kernel = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x_kernel)
        self.register_buffer("sobel_y", sobel_y_kernel)

    def compute_black_screen_penalty(self, images: torch.Tensor) -> torch.Tensor:
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        black_pixels = (gray_images < 0.1).float()
        black_ratio = black_pixels.mean(dim=[1, 2])
        penalty = F.relu(black_ratio - self.black_screen_threshold)
        return penalty.mean()

    def compute_low_variance_penalty(self, images: torch.Tensor) -> torch.Tensor:
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        variance = torch.var(gray_images.view(gray_images.shape[0], -1), dim=1)
        penalty = F.relu(self.low_variance_threshold - variance) * 10.0
        return penalty.mean()

    def compute_edge_density_penalty(self, images: torch.Tensor) -> torch.Tensor:
        gray_images = (
            0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        ).unsqueeze(1)

        sobel_x = self.sobel_x.to(device=gray_images.device, dtype=gray_images.dtype)
        sobel_y = self.sobel_y.to(device=gray_images.device, dtype=gray_images.dtype)
        grad_x = F.conv2d(gray_images, sobel_x, padding="same")
        grad_y = F.conv2d(gray_images, sobel_y, padding="same")

        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        strong_edges = (edge_magnitude > 0.1).float()
        edge_density = strong_edges.mean(dim=[1, 2, 3])

        penalty = torch.where(
            edge_density < self.edge_density_threshold,
            (self.edge_density_threshold - edge_density) * 3.0,
            torch.zeros_like(edge_density),
        )
        return penalty.mean()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        black_penalty = self.compute_black_screen_penalty(images)
        variance_penalty = self.compute_low_variance_penalty(images)
        edge_penalty = self.compute_edge_density_penalty(images)
        return black_penalty + variance_penalty + edge_penalty


__all__ = ["ViewpointLoss"]
