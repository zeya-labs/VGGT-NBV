"""Depth Anything 3 implementation of SceneEncoderPort."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from nbv_framework.models.scene_encoder.depthanything3_encoder import DepthAnything3Wrapper
from nbv_framework.reconstruction import ReconstructionData


class DepthAnything3SceneEncoderAdapter:
    def __init__(self, wrapper: DepthAnything3Wrapper) -> None:
        self.wrapper = wrapper

    def extract_scene_features(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        return self.wrapper.extract_scene_features(
            images,
            camera_poses,
            depth_z=depth_z,
            is_metric_scale=True,
        )

    def reconstruct_and_evaluate(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
    ) -> ReconstructionData:
        return self.wrapper.reconstruct_and_evaluate(
            images,
            camera_poses,
            depth_z=depth_z,
            is_metric_scale=True,
            align_pts3d_to_input_world=True,
        )
