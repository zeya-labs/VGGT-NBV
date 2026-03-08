"""Scene encoder port interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch

from nbv_framework.reconstruction import ReconstructionData


class SceneEncoderPort(Protocol):
    def extract_scene_features(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Extract scene-level features and auxiliary per-view metadata."""

    def reconstruct_and_evaluate(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
    ) -> ReconstructionData:
        """Run 3D reconstruction and return training-time reconstruction tensors."""
