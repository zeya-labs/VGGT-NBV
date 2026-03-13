"""MapAnything implementation of SceneEncoderPort."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from nbv_framework.dto import SceneFeatureBatch
from nbv_framework.reconstruction import ReconstructionData

if TYPE_CHECKING:
    from nbv_framework.models.scene_encoder.mapanything_encoder import MapAnythingWrapper


class MapAnythingSceneEncoderAdapter:
    def __init__(self, wrapper: "MapAnythingWrapper") -> None:
        self.wrapper = wrapper

    def extract_scene_features(
        self,
        images: torch.Tensor,
        camera_poses: torch.Tensor,
        *,
        depth_z: Optional[torch.Tensor] = None,
    ) -> SceneFeatureBatch:
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
