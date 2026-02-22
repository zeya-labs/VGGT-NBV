"""Policy network port interface."""

from __future__ import annotations

from typing import Protocol

import torch


class PolicyNetworkPort(Protocol):
    def __call__(
        self,
        scene_features: torch.Tensor,
        camera_poses_batch_across_views: torch.Tensor,
    ) -> torch.Tensor:
        """Run policy inference from scene features and relative camera poses."""
